// Runtime-friendly Krylov polynomial / stencil-basis SYCL miniapp.
//
// This is intentionally simpler than full GMRES.  It keeps the runtime-facing
// Krylov structure: repeated 2-D operator applications, read-mostly basis
// planes, independent branch kernels, a growing basis-mixing cost, and a final
// multi-vector fan-in.  It avoids reductions, scalar dependency buffers, and
// read_write kernels so the current offline scheduler/split implementation can
// handle it like the local 3mm-style benchmarks.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#ifdef USE_DOUBLE
using data_t = double;
#else
using data_t = float;
#endif

class KrylovPolyBasisInitKernel;
class KrylovPolyFieldInitKernel;
class KrylovPolyBranchInitKernel;
class KrylovPolyStencilBasisKernel;
class KrylovPolyStencilFieldKernel;
class KrylovPolySmoothSeedKernel;
class KrylovPolyBranchKernel;
class KrylovPolyMixKernel;
class KrylovPolyPublishKernel;
class KrylovPolyProjectionKernel;
class KrylovPolyResidualKernel;
class KrylovPolyResidualSeedKernel;
class KrylovPolySampleKernel;

namespace {

constexpr std::size_t kDefaultNx = 2048;
constexpr std::size_t kDefaultNy = 2048;
constexpr std::size_t kDefaultM = 8;
constexpr std::size_t kDefaultCycles = 2;
constexpr std::size_t kDefaultBranches = 3;

using field_buffer_t = sycl::buffer<data_t, 2>;
using basis_buffer_t = sycl::buffer<data_t, 2>;
using branch_buffer_t = sycl::buffer<data_t, 2>;
using sample_buffer_t = sycl::buffer<data_t, 1>;

enum class Mode {
  Full,
  NoFanin,
  FaninOnly,
  OperatorOnly,
  BasisOnly,
};

struct Config {
  std::size_t nx = kDefaultNx;
  std::size_t ny = kDefaultNy;
  std::size_t m = kDefaultM;
  std::size_t cycles = kDefaultCycles;
  std::size_t fanout = 0;
  std::size_t branches = kDefaultBranches;
  Mode mode = Mode::Full;
  bool verify = false;
  bool print_samples = true;
  bool wait_each_kernel = false;
  bool host_read_full = false;
};

struct Dimensions {
  std::size_t n = 0;
  std::size_t basis_slots = 0;
  std::size_t basis_elements = 0;
  std::size_t branch_elements = 0;
};

struct MemoryEstimate {
  std::size_t field_bytes = 0;
  std::size_t basis_bytes = 0;
  std::size_t branch_bytes = 0;
  std::size_t total_bytes = 0;
};

struct Buffers {
  field_buffer_t &x;
  field_buffer_t &b;
  field_buffer_t &work;
  field_buffer_t &candidate;
  field_buffer_t &residual;
  basis_buffer_t &basis;
  branch_buffer_t &branches;
  sample_buffer_t &samples;
};

const char *mode_name(Mode mode) {
  switch (mode) {
    case Mode::Full:
      return "full";
    case Mode::NoFanin:
      return "no-fanin";
    case Mode::FaninOnly:
      return "fanin-only";
    case Mode::OperatorOnly:
      return "operator-only";
    case Mode::BasisOnly:
      return "basis-only";
  }
  return "unknown";
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --nx <int>                    Grid rows (default 2048)\n"
      << "  --ny <int>                    Grid columns (default 2048)\n"
      << "  --m <int>                     Krylov polynomial depth (default 8)\n"
      << "  --cycles <int>                Restart cycles (default 2)\n"
      << "  --fanout <int>                Basis planes in projection (default m)\n"
      << "  --branches <int>              Independent branches per stage (default 3)\n"
      << "  --mode <full|no-fanin|fanin-only|operator-only|basis-only>\n"
      << "  --verify <0|1>                Check samples/checksum are finite\n"
      << "  --print-samples <0|1>         Print sparse samples (default 1)\n"
      << "  --wait-each-kernel <0|1>      Debug-only per-kernel wait\n"
      << "  --host-read-full <0|1>        Materialize and checksum full x output\n";
}

bool parse_positive_size(const std::string &text, std::size_t &value) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_binary_flag(const std::string &text, bool &value) {
  if (text == "0") {
    value = false;
    return true;
  }
  if (text == "1") {
    value = true;
    return true;
  }
  return false;
}

bool parse_mode(const std::string &text, Mode &mode) {
  if (text == "full") {
    mode = Mode::Full;
  } else if (text == "no-fanin") {
    mode = Mode::NoFanin;
  } else if (text == "fanin-only") {
    mode = Mode::FaninOnly;
  } else if (text == "operator-only") {
    mode = Mode::OperatorOnly;
  } else if (text == "basis-only" || text == "orthogonalization-only") {
    mode = Mode::BasisOnly;
  } else {
    return false;
  }
  return true;
}

bool parse_arguments(int argc, char **argv, Config &config,
                     std::string &error, bool &show_help) {
  bool fanout_was_set = false;
  for (int arg = 1; arg < argc; ++arg) {
    const std::string option = argv[arg];
    if (option == "--help" || option == "-h") {
      show_help = true;
      return true;
    }
    if (arg + 1 >= argc) {
      error = "missing value for " + option;
      return false;
    }

    const std::string value = argv[++arg];
    bool parsed = false;
    if (option == "--nx") {
      parsed = parse_positive_size(value, config.nx);
    } else if (option == "--ny") {
      parsed = parse_positive_size(value, config.ny);
    } else if (option == "--m") {
      parsed = parse_positive_size(value, config.m);
    } else if (option == "--cycles") {
      parsed = parse_positive_size(value, config.cycles);
    } else if (option == "--fanout") {
      parsed = parse_positive_size(value, config.fanout);
      fanout_was_set = parsed;
    } else if (option == "--branches") {
      parsed = parse_positive_size(value, config.branches);
    } else if (option == "--mode") {
      parsed = parse_mode(value, config.mode);
    } else if (option == "--verify") {
      parsed = parse_binary_flag(value, config.verify);
    } else if (option == "--print-samples") {
      parsed = parse_binary_flag(value, config.print_samples);
    } else if (option == "--wait-each-kernel") {
      parsed = parse_binary_flag(value, config.wait_each_kernel);
    } else if (option == "--host-read-full") {
      parsed = parse_binary_flag(value, config.host_read_full);
    } else {
      error = "unknown option " + option;
      return false;
    }

    if (!parsed) {
      error = "invalid value '" + value + "' for " + option;
      return false;
    }
  }

  if (!fanout_was_set) {
    config.fanout = config.m;
  }
  return true;
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t &result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

bool checked_add(std::size_t left, std::size_t right, std::size_t &result) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

bool derive_dimensions_and_memory(Config &config, Dimensions &dimensions,
                                  MemoryEstimate &memory,
                                  std::string &error) {
  if (!checked_multiply(config.nx, config.ny, dimensions.n)) {
    error = "nx * ny overflows size_t";
    return false;
  }
  if (config.m == std::numeric_limits<std::size_t>::max()) {
    error = "m is too large";
    return false;
  }
  dimensions.basis_slots = config.m + 1;
  if (config.fanout == 0 || config.fanout > dimensions.basis_slots) {
    error = "fanout must be in [1, m + 1]";
    return false;
  }

  if (!checked_multiply(dimensions.basis_slots, dimensions.n,
                        dimensions.basis_elements) ||
      !checked_multiply(config.branches, dimensions.n,
                        dimensions.branch_elements)) {
    error = "basis or branch allocation overflows size_t";
    return false;
  }

  if (!checked_multiply(dimensions.n, sizeof(data_t), memory.field_bytes) ||
      !checked_multiply(dimensions.basis_elements, sizeof(data_t),
                        memory.basis_bytes) ||
      !checked_multiply(dimensions.branch_elements, sizeof(data_t),
                        memory.branch_bytes)) {
    error = "byte estimate overflows size_t";
    return false;
  }

  std::size_t total_elements = 0;
  std::size_t field_elements = 0;
  if (!checked_multiply(dimensions.n, std::size_t{5}, field_elements) ||
      !checked_add(field_elements, dimensions.basis_elements,
                   total_elements) ||
      !checked_add(total_elements, dimensions.branch_elements,
                   total_elements) ||
      !checked_add(total_elements, std::size_t{4}, total_elements) ||
      !checked_multiply(total_elements, sizeof(data_t), memory.total_bytes)) {
    error = "total buffer byte estimate overflows size_t";
    return false;
  }
  return true;
}

inline data_t rhs_value(std::size_t row, std::size_t col) {
  return static_cast<data_t>(0.65) +
         static_cast<data_t>(row % 101) * static_cast<data_t>(0.0017) +
         static_cast<data_t>(col % 97) * static_cast<data_t>(0.0013);
}

inline data_t seed_value(std::size_t slot, std::size_t row, std::size_t col) {
  const std::size_t mixed =
      (row * std::size_t{17} + col * std::size_t{29} +
       slot * std::size_t{37}) %
      std::size_t{257};
  return static_cast<data_t>(0.35) +
         static_cast<data_t>(mixed) * static_cast<data_t>(0.0015);
}

inline data_t projection_coeff(std::size_t slot) {
  const data_t sign =
      (slot % 2 == 0) ? static_cast<data_t>(1) : static_cast<data_t>(-1);
  return sign * static_cast<data_t>(0.08) / static_cast<data_t>(slot + 1);
}

template <typename CommandGroup>
void submit_command(sycl::queue &queue, bool wait_each_kernel,
                    CommandGroup &&command_group) {
  queue.submit(std::forward<CommandGroup>(command_group));
  if (wait_each_kernel) {
    queue.wait();
  }
}

void wait_cycle_boundary(sycl::queue &queue) { queue.wait(); }

void initialize_on_host(Buffers &buffers, const Config &config,
                        const Dimensions &dimensions) {
  sycl::host_accessor x_acc{buffers.x, sycl::write_only};
  sycl::host_accessor b_acc{buffers.b, sycl::write_only};
  sycl::host_accessor work_acc{buffers.work, sycl::write_only};
  sycl::host_accessor cand_acc{buffers.candidate, sycl::write_only};
  sycl::host_accessor residual_acc{buffers.residual, sycl::write_only};
  sycl::host_accessor basis_acc{buffers.basis, sycl::write_only};
  sycl::host_accessor branch_acc{buffers.branches, sycl::write_only};
  sycl::host_accessor sample_acc{buffers.samples, sycl::write_only};

  for (std::size_t row = 0; row < config.nx; ++row) {
    for (std::size_t col = 0; col < config.ny; ++col) {
      const sycl::id<2> cell(row, col);
      const data_t rhs = rhs_value(row, col);
      x_acc[cell] = static_cast<data_t>(0);
      b_acc[cell] = rhs;
      work_acc[cell] = static_cast<data_t>(0);
      cand_acc[cell] = static_cast<data_t>(0);
      residual_acc[cell] = rhs;
    }
  }

  for (std::size_t slot = 0; slot < dimensions.basis_slots; ++slot) {
    const data_t scale =
        slot == 0 ? static_cast<data_t>(1) : static_cast<data_t>(0.15);
    for (std::size_t row = 0; row < config.nx; ++row) {
      for (std::size_t col = 0; col < config.ny; ++col) {
        basis_acc[sycl::id<2>(slot * config.nx + row, col)] =
            scale * seed_value(slot, row, col);
      }
    }
  }

  for (std::size_t branch = 0; branch < config.branches; ++branch) {
    for (std::size_t row = 0; row < config.nx; ++row) {
      for (std::size_t col = 0; col < config.ny; ++col) {
        branch_acc[sycl::id<2>(branch * config.nx + row, col)] =
            static_cast<data_t>(0);
      }
    }
  }

  for (std::size_t lane = 0; lane < 4; ++lane) {
    sample_acc[lane] = static_cast<data_t>(0);
  }
}

void submit_stencil_basis_to_field(sycl::queue &queue, bool wait_each_kernel,
                                   basis_buffer_t &basis,
                                   std::size_t input_slot,
                                   field_buffer_t &output,
                                   std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(input_slot * nx, 0));
    auto out = output.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolyStencilBasisKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const data_t center = input[sycl::id<2>(row, col)];
          const data_t north =
              row > 0 ? input[sycl::id<2>(row - 1, col)]
                      : static_cast<data_t>(0);
          const data_t south =
              row + 1 < nx ? input[sycl::id<2>(row + 1, col)]
                           : static_cast<data_t>(0);
          const data_t west =
              col > 0 ? input[sycl::id<2>(row, col - 1)]
                      : static_cast<data_t>(0);
          const data_t east =
              col + 1 < ny ? input[sycl::id<2>(row, col + 1)]
                           : static_cast<data_t>(0);
          const data_t lap =
              static_cast<data_t>(4) * center - north - south - west - east;
          out[sycl::id<2>(row, col)] =
              static_cast<data_t>(0.72) * center -
              static_cast<data_t>(0.045) * lap;
        });
  });
}

void submit_stencil_field_to_field(sycl::queue &queue, bool wait_each_kernel,
                                   field_buffer_t &input,
                                   field_buffer_t &output,
                                   std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto in = input.get_access<sycl::access::mode::read>(cgh);
    auto out = output.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolyStencilFieldKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const data_t center = in[sycl::id<2>(row, col)];
          const data_t north =
              row > 0 ? in[sycl::id<2>(row - 1, col)]
                      : static_cast<data_t>(0);
          const data_t south =
              row + 1 < nx ? in[sycl::id<2>(row + 1, col)]
                           : static_cast<data_t>(0);
          const data_t west =
              col > 0 ? in[sycl::id<2>(row, col - 1)]
                      : static_cast<data_t>(0);
          const data_t east =
              col + 1 < ny ? in[sycl::id<2>(row, col + 1)]
                           : static_cast<data_t>(0);
          out[sycl::id<2>(row, col)] =
              static_cast<data_t>(0.70) * center +
              static_cast<data_t>(0.075) * (north + south + west + east);
        });
  });
}

void submit_smooth_seed(sycl::queue &queue, bool wait_each_kernel,
                        basis_buffer_t &basis, std::size_t input_slot,
                        field_buffer_t &b, field_buffer_t &output,
                        std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(input_slot * nx, 0));
    auto rhs = b.get_access<sycl::access::mode::read>(cgh);
    auto out = output.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolySmoothSeedKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> cell(row, col);
          out[cell] = static_cast<data_t>(0.80) * input[cell] +
                      static_cast<data_t>(0.04) * rhs[cell] +
                      static_cast<data_t>((row + col) % 23) *
                          static_cast<data_t>(0.00002);
        });
  });
}

void submit_independent_branch(sycl::queue &queue, bool wait_each_kernel,
                               basis_buffer_t &basis, std::size_t basis_slot,
                               field_buffer_t &b, branch_buffer_t &branches,
                               std::size_t branch_slot, std::size_t nx,
                               std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(basis_slot * nx, 0));
    auto rhs = b.get_access<sycl::access::mode::read>(cgh);
    auto branch = branches.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(branch_slot * nx, 0));
    cgh.parallel_for<KrylovPolyBranchKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> cell(row, col);
          const data_t weight =
              static_cast<data_t>(0.012 + 0.004 * branch_slot);
          const data_t phase =
              static_cast<data_t>((row + branch_slot * 11 + col * 3) % 41) *
              static_cast<data_t>(0.00001);
          branch[cell] =
              weight * (rhs[cell] - input[cell]) +
              static_cast<data_t>(0.10) * input[cell] + phase;
        });
  });
}

void submit_basis_mix(sycl::queue &queue, bool wait_each_kernel,
                      basis_buffer_t &basis, std::size_t read_slots,
                      field_buffer_t &work, field_buffer_t &b,
                      branch_buffer_t &branches, std::size_t branch_slots,
                      field_buffer_t &candidate, std::size_t nx,
                      std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(read_slots * nx, ny), sycl::id<2>(0, 0));
    auto work_acc = work.get_access<sycl::access::mode::read>(cgh);
    auto rhs_acc = b.get_access<sycl::access::mode::read>(cgh);
    auto branch_acc = branches.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(branch_slots * nx, ny), sycl::id<2>(0, 0));
    auto out = candidate.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolyMixKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> cell(row, col);
          data_t value = static_cast<data_t>(0.64) * work_acc[cell] +
                         static_cast<data_t>(0.02) * rhs_acc[cell];

          for (std::size_t slot = 0; slot < read_slots; ++slot) {
            const data_t coeff =
                static_cast<data_t>(0.018) / static_cast<data_t>(slot + 1);
            value += coeff * basis_acc[sycl::id<2>(slot * nx + row, col)];
          }
          for (std::size_t branch = 0; branch < branch_slots; ++branch) {
            const data_t coeff =
                static_cast<data_t>(0.025) / static_cast<data_t>(branch + 1);
            value += coeff * branch_acc[sycl::id<2>(branch * nx + row, col)];
          }
          out[cell] = value;
        });
  });
}

void submit_publish_candidate(sycl::queue &queue, bool wait_each_kernel,
                              field_buffer_t &candidate,
                              basis_buffer_t &basis,
                              std::size_t output_slot, std::size_t nx,
                              std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input = candidate.get_access<sycl::access::mode::read>(cgh);
    auto output = basis.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(output_slot * nx, 0));
    cgh.parallel_for<KrylovPolyPublishKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          output[sycl::id<2>(row, col)] = input[sycl::id<2>(row, col)];
        });
  });
}

void submit_projection(sycl::queue &queue, bool wait_each_kernel,
                       basis_buffer_t &basis, field_buffer_t &x,
                       std::size_t fanout, std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(fanout * nx, ny), sycl::id<2>(0, 0));
    auto x_acc = x.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolyProjectionKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          data_t sum = static_cast<data_t>(0);
          for (std::size_t slot = 0; slot < fanout; ++slot) {
            sum += projection_coeff(slot) *
                   basis_acc[sycl::id<2>(slot * nx + row, col)];
          }
          x_acc[sycl::id<2>(row, col)] = sum;
        });
  });
}

void submit_residual(sycl::queue &queue, bool wait_each_kernel,
                     field_buffer_t &b, field_buffer_t &ax,
                     field_buffer_t &residual, std::size_t nx,
                     std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto rhs = b.get_access<sycl::access::mode::read>(cgh);
    auto op = ax.get_access<sycl::access::mode::read>(cgh);
    auto out = residual.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolyResidualKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> cell(row, col);
          out[cell] = rhs[cell] - op[cell];
        });
  });
}

void submit_residual_seed(sycl::queue &queue, bool wait_each_kernel,
                          field_buffer_t &residual, field_buffer_t &b,
                          basis_buffer_t &basis, std::size_t output_slot,
                          std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto residual_acc = residual.get_access<sycl::access::mode::read>(cgh);
    auto rhs = b.get_access<sycl::access::mode::read>(cgh);
    auto output = basis.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(output_slot * nx, 0));
    cgh.parallel_for<KrylovPolyResidualSeedKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> cell(row, col);
          output[sycl::id<2>(row, col)] =
              static_cast<data_t>(0.18) * residual_acc[cell] +
              static_cast<data_t>(0.02) * rhs[cell];
        });
  });
}

void submit_basis_cycle(sycl::queue &queue, const Config &config,
                        const Dimensions &dimensions, Buffers &buffers,
                        bool use_stencil) {
  for (std::size_t k = 0; k < config.m; ++k) {
    if (use_stencil) {
      submit_stencil_basis_to_field(queue, config.wait_each_kernel,
                                    buffers.basis, k, buffers.work,
                                    config.nx, config.ny);
    } else {
      submit_smooth_seed(queue, config.wait_each_kernel, buffers.basis, k,
                         buffers.b, buffers.work, config.nx, config.ny);
    }

    for (std::size_t branch = 0; branch < config.branches; ++branch) {
      submit_independent_branch(queue, config.wait_each_kernel, buffers.basis,
                                k, buffers.b, buffers.branches, branch,
                                config.nx, config.ny);
    }

    submit_basis_mix(queue, config.wait_each_kernel, buffers.basis, k + 1,
                     buffers.work, buffers.b, buffers.branches,
                     config.branches, buffers.candidate, config.nx,
                     config.ny);
    submit_publish_candidate(queue, config.wait_each_kernel, buffers.candidate,
                             buffers.basis, k + 1, config.nx, config.ny);
  }
}

void submit_full_mode(sycl::queue &queue, const Config &config,
                      const Dimensions &dimensions, Buffers &buffers) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_basis_cycle(queue, config, dimensions, buffers, true);
    submit_projection(queue, config.wait_each_kernel, buffers.basis, buffers.x,
                      config.fanout, config.nx, config.ny);

    if (cycle + 1 < config.cycles) {
      submit_stencil_field_to_field(queue, config.wait_each_kernel, buffers.x,
                                    buffers.work, config.nx, config.ny);
      submit_residual(queue, config.wait_each_kernel, buffers.b, buffers.work,
                      buffers.residual, config.nx, config.ny);
      submit_residual_seed(queue, config.wait_each_kernel, buffers.residual,
                           buffers.b, buffers.basis, 0, config.nx, config.ny);
    }
    wait_cycle_boundary(queue);
  }
}

void submit_no_fanin_mode(sycl::queue &queue, const Config &config,
                          const Dimensions &dimensions, Buffers &buffers,
                          bool use_stencil) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_basis_cycle(queue, config, dimensions, buffers, use_stencil);
    if (cycle + 1 < config.cycles) {
      submit_publish_candidate(queue, config.wait_each_kernel,
                               buffers.candidate, buffers.basis, 0,
                               config.nx, config.ny);
    }
    wait_cycle_boundary(queue);
  }
}

void submit_operator_only_mode(sycl::queue &queue, const Config &config,
                               Buffers &buffers) {
  bool result_in_basis = true;
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    for (std::size_t step = 0; step < config.m; ++step) {
      if (result_in_basis) {
        submit_stencil_basis_to_field(queue, config.wait_each_kernel,
                                      buffers.basis, 0, buffers.work,
                                      config.nx, config.ny);
      } else {
        submit_residual_seed(queue, config.wait_each_kernel, buffers.work,
                             buffers.b, buffers.basis, 0, config.nx,
                             config.ny);
      }
      result_in_basis = !result_in_basis;
    }
    wait_cycle_boundary(queue);
  }
}

void submit_fanin_only_mode(sycl::queue &queue, const Config &config,
                            Buffers &buffers) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_projection(queue, config.wait_each_kernel, buffers.basis, buffers.x,
                      config.fanout, config.nx, config.ny);
    wait_cycle_boundary(queue);
  }
}

void submit_samples(sycl::queue &queue, bool wait_each_kernel, Buffers &buffers,
                    const Config &config, const Dimensions &dimensions) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto x_acc = buffers.x.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = buffers.basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(dimensions.basis_slots * config.nx, config.ny),
        sycl::id<2>(0, 0));
    auto branch_acc = buffers.branches.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(config.branches * config.nx, config.ny),
        sycl::id<2>(0, 0));
    auto sample_acc =
        buffers.samples.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovPolySampleKernel>(
        sycl::range<1>(4), [=](sycl::item<1> item) {
          const std::size_t lane = item[0];
          const std::size_t index = lane == 0 ? 0
                                  : lane == 1 ? dimensions.n / 2
                                  : lane == 2 ? dimensions.n - 1
                                              : dimensions.n / 3;
          const std::size_t row = index / config.ny;
          const std::size_t col = index - row * config.ny;
          const sycl::id<2> cell(row, col);
          if (lane == 0) {
            sample_acc[lane] = x_acc[cell];
          } else if (lane == 1) {
            sample_acc[lane] =
                basis_acc[sycl::id<2>(config.m * config.nx + row, col)];
          } else if (lane == 2) {
            sample_acc[lane] = branch_acc[sycl::id<2>(row, col)];
          } else {
            data_t branch_mix = static_cast<data_t>(0);
            for (std::size_t branch = 0; branch < config.branches; ++branch) {
              branch_mix +=
                  static_cast<data_t>(0.1) *
                  branch_acc[sycl::id<2>(branch * config.nx + row, col)];
            }
            sample_acc[lane] =
                x_acc[cell] +
                static_cast<data_t>(0.25) *
                    basis_acc[sycl::id<2>(config.m * config.nx + row, col)] +
                branch_mix;
          }
        });
  });
}

double checksum_x(Buffers &buffers, const Config &config) {
  double checksum = 0.0;
  sycl::host_accessor x_acc{buffers.x, sycl::read_only};
  for (std::size_t row = 0; row < config.nx; ++row) {
    for (std::size_t col = 0; col < config.ny; ++col) {
      checksum += static_cast<double>(x_acc[sycl::id<2>(row, col)]);
    }
  }
  return checksum;
}

}  // namespace

int main(int argc, char **argv) {
  Config config;
  Dimensions dimensions;
  MemoryEstimate memory;
  std::string error;
  bool show_help = false;
  if (!parse_arguments(argc, argv, config, error, show_help)) {
    std::cerr << "ERROR " << error << '\n';
    print_usage(argv[0]);
    return 1;
  }
  if (show_help) {
    print_usage(argv[0]);
    return 0;
  }
  if (!derive_dimensions_and_memory(config, dimensions, memory, error)) {
    std::cerr << "ERROR " << error << '\n';
    return 1;
  }

  std::cout << "CONFIG nx=" << config.nx << " ny=" << config.ny
            << " N=" << dimensions.n << " m=" << config.m
            << " cycles=" << config.cycles << " fanout=" << config.fanout
            << " branches=" << config.branches
            << " mode=" << mode_name(config.mode)
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " host_read_full=" << (config.host_read_full ? 1 : 0)
            << '\n';
  std::cout << "MEMORY field_bytes=" << memory.field_bytes
            << " basis_bytes=" << memory.basis_bytes
            << " branch_bytes=" << memory.branch_bytes
            << " total_estimated_bytes=" << memory.total_bytes << '\n';

  try {
    const auto init_begin = std::chrono::steady_clock::now();

    field_buffer_t x_buf{sycl::range<2>(config.nx, config.ny)};
    field_buffer_t b_buf{sycl::range<2>(config.nx, config.ny)};
    field_buffer_t work_buf{sycl::range<2>(config.nx, config.ny)};
    field_buffer_t candidate_buf{sycl::range<2>(config.nx, config.ny)};
    field_buffer_t residual_buf{sycl::range<2>(config.nx, config.ny)};
    basis_buffer_t basis_buf{
        sycl::range<2>(dimensions.basis_slots * config.nx, config.ny)};
    branch_buffer_t branch_buf{
        sycl::range<2>(config.branches * config.nx, config.ny)};
    sample_buffer_t sample_buf{sycl::range<1>(4)};

    Buffers buffers{x_buf,       b_buf,      work_buf, candidate_buf,
                    residual_buf, basis_buf, branch_buf, sample_buf};

    initialize_on_host(buffers, config, dimensions);
    const auto init_end = std::chrono::steady_clock::now();

    sycl::queue queue;
    const auto run_begin = std::chrono::steady_clock::now();

    switch (config.mode) {
      case Mode::Full:
        submit_full_mode(queue, config, dimensions, buffers);
        break;
      case Mode::NoFanin:
        submit_no_fanin_mode(queue, config, dimensions, buffers, true);
        break;
      case Mode::FaninOnly:
        submit_fanin_only_mode(queue, config, buffers);
        break;
      case Mode::OperatorOnly:
        submit_operator_only_mode(queue, config, buffers);
        break;
      case Mode::BasisOnly:
        submit_no_fanin_mode(queue, config, dimensions, buffers, false);
        break;
    }

    submit_samples(queue, config.wait_each_kernel, buffers, config, dimensions);
    queue.wait();
    const auto run_end = std::chrono::steady_clock::now();

    data_t samples[4] = {static_cast<data_t>(0), static_cast<data_t>(0),
                         static_cast<data_t>(0), static_cast<data_t>(0)};
    const auto host_begin = std::chrono::steady_clock::now();
    {
      sycl::host_accessor sample_acc{sample_buf, sycl::read_only};
      for (std::size_t lane = 0; lane < 4; ++lane) {
        samples[lane] = sample_acc[lane];
      }
    }

    double checksum = static_cast<double>(samples[0]) +
                      static_cast<double>(samples[1]) +
                      static_cast<double>(samples[2]) +
                      static_cast<double>(samples[3]);
    if (config.host_read_full) {
      checksum = checksum_x(buffers, config);
    }
    const auto host_end = std::chrono::steady_clock::now();

    const double init_seconds =
        std::chrono::duration<double>(init_end - init_begin).count();
    const double run_seconds =
        std::chrono::duration<double>(run_end - run_begin).count();
    const double host_seconds =
        std::chrono::duration<double>(host_end - host_begin).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_seconds << '\n';
    std::cout << "TIMING run_sec=" << run_seconds << '\n';
    std::cout << "TIMING host_sec=" << host_seconds << '\n';
    if (config.print_samples) {
      std::cout << "RESULT checksum=" << checksum
                << " sample0=" << static_cast<double>(samples[0])
                << " sample1=" << static_cast<double>(samples[1])
                << " sample2=" << static_cast<double>(samples[2]) << '\n';
    } else {
      std::cout << "RESULT checksum=" << checksum << '\n';
    }

    if (config.verify) {
      bool finite = std::isfinite(checksum);
      for (const data_t sample : samples) {
        finite = finite && std::isfinite(static_cast<double>(sample));
      }
      std::cout << "VERIFY passed=" << (finite ? 1 : 0) << '\n';
      return finite ? 0 : 2;
    }
  } catch (const sycl::exception &exception) {
    std::cerr << "ERROR SYCL " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }

  return 0;
}
