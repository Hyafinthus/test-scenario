// Standalone restarted GMRES / Arnoldi-inspired SYCL buffer benchmark.
//
// H uses a column-major Arnoldi layout:
//   H[k * (m + 1) + i] == h(i, k), 0 <= i <= k + 1.
//
// The default is deliberately safe for a single GPU.  For large-memory runs,
// useful starting points are --nx 8192 --ny 8192 --m 8, or --nx 16384
// --ny 16384 --m 4 when the device has sufficient memory.

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

// Keep named kernels at global scope so older SYCL toolchains can give each
// device kernel a stable name.
class KrylovDeviceInitKernel;
class KrylovStencilApplyKernel;
class KrylovOrthogonalizationSeedKernel;
class KrylovIndependentBranchKernel;
class KrylovDotStage1Kernel;
class KrylovDotStage2Kernel;
class KrylovAxpyUpdateKernel;
class KrylovNormStage1Kernel;
class KrylovNormStage2Kernel;
class KrylovNormalizeKernel;
class KrylovRestartMarkerKernel;
class KrylovFinalProjectionKernel;
class KrylovResidualKernel;
class KrylovCopyBasisKernel;
class KrylovSampleKernel;

namespace {

constexpr std::size_t kDefaultNx = 2048;
constexpr std::size_t kDefaultNy = 2048;
constexpr std::size_t kDefaultM = 8;
constexpr std::size_t kDefaultCycles = 2;
constexpr std::size_t kDefaultPartials = 1024;
constexpr data_t kNormFloor = static_cast<data_t>(1.0e-20);

using buffer_t = sycl::buffer<data_t, 1>;

enum class Mode {
  Full,
  NoFanin,
  FaninOnly,
  OperatorOnly,
  OrthogonalizationOnly,
};

struct Config {
  std::size_t nx = kDefaultNx;
  std::size_t ny = kDefaultNy;
  std::size_t m = kDefaultM;
  std::size_t cycles = kDefaultCycles;
  std::size_t partials = kDefaultPartials;
  std::size_t fanout = 0;  // Filled from m unless --fanout is supplied.
  Mode mode = Mode::Full;
  bool verify = false;
  bool print_samples = true;
  bool wait_each_kernel = false;
  bool host_read_full = false;
  bool init_on_device = false;
};

struct Dimensions {
  std::size_t n = 0;
  std::size_t basis_slots = 0;
  std::size_t basis_elements = 0;
  std::size_t h_elements = 0;
};

struct MemoryEstimate {
  std::size_t vector_bytes = 0;
  std::size_t basis_bytes = 0;
  std::size_t total_bytes = 0;
};

struct Buffers {
  buffer_t &x;
  buffer_t &b;
  buffer_t &basis;
  buffer_t &w;
  buffer_t &residual;
  buffer_t &auxiliary;
  buffer_t &partials;
  buffer_t &h;
  buffer_t &coefficients;
  buffer_t &restart_norm;
  buffer_t &cycle_token;
  buffer_t &samples;
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
    case Mode::OrthogonalizationOnly:
      return "orthogonalization-only";
  }
  return "unknown";
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --nx <int>                    Grid rows (default 2048)\n"
      << "  --ny <int>                    Grid columns (default 2048)\n"
      << "  --m <int>                     Restart/Krylov length (default 8)\n"
      << "  --cycles <int>                Restart cycles (default 2)\n"
      << "  --partials <int>              Explicit reduction chunks (default 1024)\n"
      << "  --fanout <int>                Basis vectors in projection (default m)\n"
      << "  --mode <full|no-fanin|fanin-only|operator-only|orthogonalization-only>\n"
      << "  --verify <0|1>                Check final samples are finite (default 0)\n"
      << "  --print-samples <0|1>         Print individual sparse samples (default 1)\n"
      << "  --wait-each-kernel <0|1>      Debug-only per-kernel queue wait (default 0)\n"
      << "  --host-read-full <0|1>        Materialize and checksum full primary output\n"
      << "  --init-on-device <0|1>        Device initialization instead of host initialization\n";
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
  } else if (text == "orthogonalization-only") {
    mode = Mode::OrthogonalizationOnly;
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
    } else if (option == "--partials") {
      parsed = parse_positive_size(value, config.partials);
    } else if (option == "--fanout") {
      parsed = parse_positive_size(value, config.fanout);
      fanout_was_set = parsed;
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
    } else if (option == "--init-on-device") {
      parsed = parse_binary_flag(value, config.init_on_device);
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
  if (config.fanout == 0 || config.fanout > config.m) {
    error = "fanout must be in [1, m]";
    return false;
  }
  if (config.partials > dimensions.n) {
    config.partials = dimensions.n;
  }
  if (!checked_multiply(dimensions.basis_slots, dimensions.n,
                        dimensions.basis_elements) ||
      !checked_multiply(dimensions.basis_slots, config.m,
                        dimensions.h_elements)) {
    error = "basis or Hessenberg allocation overflows size_t";
    return false;
  }

  if (!checked_multiply(dimensions.n, sizeof(data_t), memory.vector_bytes) ||
      !checked_multiply(dimensions.basis_elements, sizeof(data_t),
                        memory.basis_bytes)) {
    error = "byte estimate overflows size_t";
    return false;
  }

  // x, b, w, residual, and auxiliary are the five full-length work vectors.
  std::size_t total_elements = 0;
  std::size_t full_vector_elements = 0;
  if (!checked_multiply(dimensions.n, std::size_t{5}, full_vector_elements) ||
      !checked_add(full_vector_elements, dimensions.basis_elements,
                   total_elements) ||
      !checked_add(total_elements, dimensions.h_elements, total_elements) ||
      !checked_add(total_elements, config.m, total_elements) ||
      !checked_add(total_elements, config.partials, total_elements) ||
      !checked_add(total_elements, std::size_t{1}, total_elements) ||
      !checked_add(total_elements, std::size_t{1}, total_elements) ||
      !checked_add(total_elements, std::size_t{4}, total_elements) ||
      !checked_multiply(total_elements, sizeof(data_t), memory.total_bytes)) {
    error = "total buffer byte estimate overflows size_t";
    return false;
  }
  return true;
}

inline std::size_t h_index(std::size_t column, std::size_t row,
                           std::size_t m) {
  return column * (m + 1) + row;
}

// These arithmetic-only functions are callable from both host initialization
// and the optional device initializer.  They deliberately avoid random input.
inline data_t rhs_value(std::size_t index, std::size_t ny) {
  const std::size_t row = index / ny;
  const std::size_t col = index - row * ny;
  return static_cast<data_t>(0.75) +
         static_cast<data_t>(row % 97) * static_cast<data_t>(0.0025) +
         static_cast<data_t>(col % 89) * static_cast<data_t>(0.0015);
}

inline data_t basis_seed(std::size_t basis_index, std::size_t index,
                         std::size_t ny) {
  const std::size_t row = index / ny;
  const std::size_t col = index - row * ny;
  const std::size_t mixed =
      (row * std::size_t{17} + col * std::size_t{29} +
       basis_index * std::size_t{31}) %
      std::size_t{251};
  return static_cast<data_t>(0.60) +
         static_cast<data_t>(mixed) * static_cast<data_t>(0.002);
}

inline data_t coefficient_value(std::size_t index) {
  const data_t sign = (index % 2 == 0) ? static_cast<data_t>(1)
                                        : static_cast<data_t>(-1);
  return sign * static_cast<data_t>(0.05) /
         static_cast<data_t>(index + 1);
}

void initialize_on_host(Buffers &buffers, const Config &config,
                        const Dimensions &dimensions) {
  // Host accessors keep the default initialization deterministic and avoid any
  // USM allocation.  write_only is appropriate because every element is set.
  sycl::host_accessor x_acc{buffers.x, sycl::write_only};
  sycl::host_accessor b_acc{buffers.b, sycl::write_only};
  sycl::host_accessor basis_acc{buffers.basis, sycl::write_only};
  sycl::host_accessor w_acc{buffers.w, sycl::write_only};
  sycl::host_accessor residual_acc{buffers.residual, sycl::write_only};
  sycl::host_accessor auxiliary_acc{buffers.auxiliary, sycl::write_only};
  sycl::host_accessor h_acc{buffers.h, sycl::write_only};
  sycl::host_accessor coefficient_acc{buffers.coefficients, sycl::write_only};
  sycl::host_accessor restart_norm_acc{buffers.restart_norm, sycl::write_only};
  sycl::host_accessor cycle_token_acc{buffers.cycle_token, sycl::write_only};

  double q0_squared_norm = 0.0;
  for (std::size_t index = 0; index < dimensions.n; ++index) {
    const double value = static_cast<double>(basis_seed(0, index, config.ny));
    q0_squared_norm += value * value;
  }
  const data_t q0_scale =
      static_cast<data_t>(1.0 / std::sqrt(q0_squared_norm));
  const data_t other_basis_scale = static_cast<data_t>(
      1.0 / std::sqrt(static_cast<double>(dimensions.n)));

  for (std::size_t index = 0; index < dimensions.n; ++index) {
    const data_t rhs = rhs_value(index, config.ny);
    x_acc[index] = static_cast<data_t>(0);
    b_acc[index] = rhs;
    w_acc[index] = static_cast<data_t>(0);
    residual_acc[index] = rhs;
    auxiliary_acc[index] = static_cast<data_t>(0.001) * rhs;
  }
  for (std::size_t basis_index = 0; basis_index < dimensions.basis_slots;
       ++basis_index) {
    const data_t scale = basis_index == 0 ? q0_scale : other_basis_scale;
    const std::size_t offset = basis_index * dimensions.n;
    for (std::size_t index = 0; index < dimensions.n; ++index) {
      basis_acc[offset + index] =
          scale * basis_seed(basis_index, index, config.ny);
    }
  }
  for (std::size_t index = 0; index < dimensions.h_elements; ++index) {
    h_acc[index] = static_cast<data_t>(0);
  }
  for (std::size_t index = 0; index < config.m; ++index) {
    coefficient_acc[index] = coefficient_value(index);
  }
  restart_norm_acc[0] = static_cast<data_t>(1);
  cycle_token_acc[0] = static_cast<data_t>(1);
}

void initialize_on_device(sycl::queue &queue, Buffers &buffers,
                          const Config &config,
                          const Dimensions &dimensions) {
  const std::size_t init_items =
      std::max(std::max(dimensions.n, dimensions.basis_elements),
               std::max(dimensions.h_elements, config.m));
  const data_t basis_scale = static_cast<data_t>(
      1.0 / std::sqrt(static_cast<double>(dimensions.n)));

  queue.submit([&](sycl::handler &cgh) {
    // discard_write is the portable native-SYCL no-initialization spelling:
    // this kernel overwrites every element in each accessor's declared range.
    auto x_acc =
        buffers.x.get_access<sycl::access::mode::discard_write>(cgh);
    auto b_acc =
        buffers.b.get_access<sycl::access::mode::discard_write>(cgh);
    auto basis_acc =
        buffers.basis.get_access<sycl::access::mode::discard_write>(cgh);
    auto w_acc =
        buffers.w.get_access<sycl::access::mode::discard_write>(cgh);
    auto residual_acc =
        buffers.residual.get_access<sycl::access::mode::discard_write>(cgh);
    auto auxiliary_acc =
        buffers.auxiliary.get_access<sycl::access::mode::discard_write>(cgh);
    auto h_acc =
        buffers.h.get_access<sycl::access::mode::discard_write>(cgh);
    auto coefficient_acc = buffers.coefficients.get_access<
        sycl::access::mode::discard_write>(cgh);
    auto restart_norm_acc = buffers.restart_norm.get_access<
        sycl::access::mode::discard_write>(cgh);
    auto cycle_token_acc = buffers.cycle_token.get_access<
        sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for<KrylovDeviceInitKernel>(sycl::range<1>(init_items),
                                             [=](sycl::item<1> item) {
      const std::size_t index = item[0];
      if (index < dimensions.n) {
        const data_t rhs = rhs_value(index, config.ny);
        x_acc[index] = static_cast<data_t>(0);
        b_acc[index] = rhs;
        w_acc[index] = static_cast<data_t>(0);
        residual_acc[index] = rhs;
        auxiliary_acc[index] = static_cast<data_t>(0.001) * rhs;
      }
      if (index < dimensions.basis_elements) {
        const std::size_t basis_index = index / dimensions.n;
        const std::size_t local_index = index - basis_index * dimensions.n;
        basis_acc[index] =
            basis_scale * basis_seed(basis_index, local_index, config.ny);
      }
      if (index < dimensions.h_elements) {
        h_acc[index] = static_cast<data_t>(0);
      }
      if (index < config.m) {
        coefficient_acc[index] = coefficient_value(index);
      }
      if (index == 0) {
        restart_norm_acc[0] = static_cast<data_t>(1);
        cycle_token_acc[0] = static_cast<data_t>(1);
      }
    });
  });

  // Initialization is intentionally outside the timed DAG.  This is the one
  // required synchronization before the benchmark submissions begin.
  queue.wait();
}

template <typename CommandGroup>
void submit_command(sycl::queue &queue, bool wait_each_kernel,
                    CommandGroup &&command_group) {
  queue.submit(std::forward<CommandGroup>(command_group));
  if (wait_each_kernel) {
    queue.wait();
  }
}

void submit_stencil(sycl::queue &queue, bool wait_each_kernel,
                    buffer_t &input, std::size_t input_offset,
                    buffer_t &output, std::size_t output_offset,
                    std::size_t nx, std::size_t ny, std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(input_offset));
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<1>(n), sycl::id<1>(output_offset));
    cgh.parallel_for<KrylovStencilApplyKernel>(sycl::range<1>(n),
                                                [=](sycl::item<1> item) {
      const std::size_t index = item[0];
      const std::size_t row = index / ny;
      const std::size_t col = index - row * ny;
      const data_t center = input_acc[index];
      const data_t north = row > 0 ? input_acc[index - ny]
                                   : static_cast<data_t>(0);
      const data_t south = row + 1 < nx ? input_acc[index + ny]
                                         : static_cast<data_t>(0);
      const data_t west = col > 0 ? input_acc[index - 1]
                                   : static_cast<data_t>(0);
      const data_t east = col + 1 < ny ? input_acc[index + 1]
                                        : static_cast<data_t>(0);
      output_acc[index] = static_cast<data_t>(4) * center - north - south -
                          west - east;
    });
  });
}

void submit_orthogonalization_seed(sycl::queue &queue, bool wait_each_kernel,
                                   buffer_t &rhs, buffer_t &basis,
                                   std::size_t basis_offset, buffer_t &w,
                                   std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto rhs_acc = rhs.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(basis_offset));
    auto w_acc = w.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovOrthogonalizationSeedKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t index = item[0];
          const data_t perturbation = static_cast<data_t>(index % 31) *
                                      static_cast<data_t>(0.0001);
          w_acc[index] = rhs_acc[index] - static_cast<data_t>(0.125) *
                                             basis_acc[index] +
                         perturbation;
        });
  });
}

void submit_independent_branch(sycl::queue &queue, bool wait_each_kernel,
                               buffer_t &rhs, buffer_t &basis,
                               std::size_t basis_offset, buffer_t &auxiliary,
                               std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto rhs_acc = rhs.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(basis_offset));
    auto auxiliary_acc =
        auxiliary.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<KrylovIndependentBranchKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t index = item[0];
          // A lightweight residual-monitor branch.  It is independent of the
          // stencil/w path for this iteration, but is sampled at the end.
          auxiliary_acc[index] =
              static_cast<data_t>(0.997) * auxiliary_acc[index] +
              static_cast<data_t>(0.003) * (rhs_acc[index] - basis_acc[index]);
        });
  });
}

void submit_dot_stage1(sycl::queue &queue, bool wait_each_kernel,
                       buffer_t &basis, std::size_t basis_offset, buffer_t &w,
                       buffer_t &partials, std::size_t n,
                       std::size_t partial_count) {
  const std::size_t chunk =
      n / partial_count + (n % partial_count == 0 ? 0 : 1);
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(basis_offset));
    auto w_acc = w.get_access<sycl::access::mode::read>(cgh);
    auto partial_acc =
        partials.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovDotStage1Kernel>(
        sycl::range<1>(partial_count), [=](sycl::item<1> item) {
          const std::size_t partial_index = item[0];
          const std::size_t begin = partial_index * chunk;
          const std::size_t unclamped_end = begin + chunk;
          const std::size_t end = unclamped_end < n ? unclamped_end : n;
          data_t sum = static_cast<data_t>(0);
          for (std::size_t index = begin; index < end; ++index) {
            sum += basis_acc[index] * w_acc[index];
          }
          partial_acc[partial_index] = sum;
        });
  });
}

void submit_dot_stage2(sycl::queue &queue, bool wait_each_kernel,
                       buffer_t &partials, std::size_t partial_count,
                       buffer_t &scalar_output,
                       std::size_t scalar_output_offset) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto partial_acc = partials.get_access<sycl::access::mode::read>(cgh);
    auto scalar_acc = scalar_output.get_access<
        sycl::access::mode::discard_write>(
        cgh, sycl::range<1>(1), sycl::id<1>(scalar_output_offset));
    cgh.parallel_for<KrylovDotStage2Kernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          data_t sum = static_cast<data_t>(0);
          for (std::size_t partial_index = 0; partial_index < partial_count;
               ++partial_index) {
            sum += partial_acc[partial_index];
          }
          scalar_acc[0] = sum;
        });
  });
}

void submit_axpy_update(sycl::queue &queue, bool wait_each_kernel,
                        buffer_t &w, buffer_t &basis,
                        std::size_t basis_offset, buffer_t &h,
                        std::size_t h_offset, std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto w_acc = w.get_access<sycl::access::mode::read_write>(cgh);
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(basis_offset));
    auto h_acc = h.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(1), sycl::id<1>(h_offset));
    cgh.parallel_for<KrylovAxpyUpdateKernel>(sycl::range<1>(n),
                                              [=](sycl::item<1> item) {
      const std::size_t index = item[0];
      w_acc[index] -= h_acc[0] * basis_acc[index];
    });
  });
}

void submit_norm_stage1(sycl::queue &queue, bool wait_each_kernel,
                        buffer_t &input, std::size_t input_offset,
                        buffer_t &partials, std::size_t n,
                        std::size_t partial_count) {
  const std::size_t chunk =
      n / partial_count + (n % partial_count == 0 ? 0 : 1);
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(input_offset));
    auto partial_acc =
        partials.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovNormStage1Kernel>(
        sycl::range<1>(partial_count), [=](sycl::item<1> item) {
          const std::size_t partial_index = item[0];
          const std::size_t begin = partial_index * chunk;
          const std::size_t unclamped_end = begin + chunk;
          const std::size_t end = unclamped_end < n ? unclamped_end : n;
          data_t sum = static_cast<data_t>(0);
          for (std::size_t index = begin; index < end; ++index) {
            const data_t value = input_acc[index];
            sum += value * value;
          }
          partial_acc[partial_index] = sum;
        });
  });
}

void submit_norm_stage2(sycl::queue &queue, bool wait_each_kernel,
                        buffer_t &partials, std::size_t partial_count,
                        buffer_t &scalar_output,
                        std::size_t scalar_output_offset) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto partial_acc = partials.get_access<sycl::access::mode::read>(cgh);
    auto scalar_acc = scalar_output.get_access<
        sycl::access::mode::discard_write>(
        cgh, sycl::range<1>(1), sycl::id<1>(scalar_output_offset));
    cgh.parallel_for<KrylovNormStage2Kernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          data_t sum = static_cast<data_t>(0);
          for (std::size_t partial_index = 0; partial_index < partial_count;
               ++partial_index) {
            sum += partial_acc[partial_index];
          }
          scalar_acc[0] = sycl::sqrt(sum > static_cast<data_t>(0)
                                         ? sum
                                         : static_cast<data_t>(0));
        });
  });
}

void submit_normalize(sycl::queue &queue, bool wait_each_kernel,
                      buffer_t &input, std::size_t input_offset,
                      buffer_t &norm, std::size_t norm_offset,
                      buffer_t &output, std::size_t output_offset,
                      std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(input_offset));
    auto norm_acc = norm.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(1), sycl::id<1>(norm_offset));
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<1>(n), sycl::id<1>(output_offset));
    cgh.parallel_for<KrylovNormalizeKernel>(sycl::range<1>(n),
                                             [=](sycl::item<1> item) {
      const std::size_t index = item[0];
      const data_t denominator = norm_acc[0];
      output_acc[index] = denominator > kNormFloor
                              ? input_acc[index] / denominator
                              : static_cast<data_t>(0);
    });
  });
}

void submit_restart_marker(sycl::queue &queue, bool wait_each_kernel,
                           buffer_t &h, std::size_t last_norm_offset,
                           buffer_t &basis, std::size_t last_basis_offset,
                           buffer_t &cycle_token, std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto h_acc = h.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(1), sycl::id<1>(last_norm_offset));
    // Reading q_m gives the marker a dependency on the final normalize too,
    // even when the correction itself only consumes q_0 through q_(m-1).
    auto last_basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(last_basis_offset));
    auto token_acc = cycle_token.get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovRestartMarkerKernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          const data_t h_value = h_acc[0];
          const data_t q_value = last_basis_acc[0];
          const data_t h_abs = h_value < static_cast<data_t>(0) ? -h_value
                                                                  : h_value;
          const data_t q_abs = q_value < static_cast<data_t>(0) ? -q_value
                                                                  : q_value;
          // A near-one stabilization factor also makes this data dependency
          // semantically visible to the following projection kernel.
          token_acc[0] = static_cast<data_t>(1) /
                         (static_cast<data_t>(1) +
                          static_cast<data_t>(1.0e-6) * (h_abs + q_abs));
        });
  });
}

void submit_final_projection(sycl::queue &queue, bool wait_each_kernel,
                             buffer_t &x, buffer_t &basis,
                             buffer_t &coefficients, buffer_t &cycle_token,
                             std::size_t fanout, std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto x_acc = x.get_access<sycl::access::mode::read_write>(cgh);
    // This single contiguous range represents the read-mostly block of
    // Krylov vectors consumed by x += Q*y.
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(fanout * n), sycl::id<1>(0));
    auto coefficient_acc = coefficients.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(fanout), sycl::id<1>(0));
    auto token_acc =
        cycle_token.get_access<sycl::access::mode::read>(cgh);
    cgh.parallel_for<KrylovFinalProjectionKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t index = item[0];
          data_t correction = static_cast<data_t>(0);
          for (std::size_t basis_index = 0; basis_index < fanout;
               ++basis_index) {
            correction += coefficient_acc[basis_index] *
                          basis_acc[basis_index * n + index];
          }
          x_acc[index] += token_acc[0] * correction;
        });
  });
}

void submit_residual(sycl::queue &queue, bool wait_each_kernel, buffer_t &b,
                     buffer_t &ax, buffer_t &residual, std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
    auto ax_acc = ax.get_access<sycl::access::mode::read>(cgh);
    auto residual_acc =
        residual.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovResidualKernel>(sycl::range<1>(n),
                                            [=](sycl::item<1> item) {
      const std::size_t index = item[0];
      residual_acc[index] = b_acc[index] - ax_acc[index];
    });
  });
}

void submit_copy_basis(sycl::queue &queue, bool wait_each_kernel,
                       buffer_t &basis, std::size_t source_offset,
                       std::size_t destination_offset, std::size_t n) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto source_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(source_offset));
    auto destination_acc = basis.get_access<
        sycl::access::mode::discard_write>(
        cgh, sycl::range<1>(n), sycl::id<1>(destination_offset));
    cgh.parallel_for<KrylovCopyBasisKernel>(sycl::range<1>(n),
                                             [=](sycl::item<1> item) {
      destination_acc[item[0]] = source_acc[item[0]];
    });
  });
}

void submit_arnoldi_cycle(sycl::queue &queue, const Config &config,
                           const Dimensions &dimensions, Buffers &buffers,
                           bool use_stencil) {
  for (std::size_t k = 0; k < config.m; ++k) {
    const std::size_t qk_offset = k * dimensions.n;
    if (use_stencil) {
      // Large regular phase: Celerity-friendly in a later neighborhood-mapped
      // implementation and worthwhile for a multi-device scheduler to split.
      submit_stencil(queue, config.wait_each_kernel, buffers.basis, qk_offset,
                     buffers.w, 0, config.nx, config.ny, dimensions.n);
    } else {
      submit_orthogonalization_seed(
          queue, config.wait_each_kernel, buffers.b, buffers.basis, qk_offset,
          buffers.w, dimensions.n);
    }

    // It shares q_k and b with the main path but does not touch w, so it is a
    // real independent branch for a DAG-aware runtime.
    submit_independent_branch(queue, config.wait_each_kernel, buffers.b,
                              buffers.basis, qk_offset, buffers.auxiliary,
                              dimensions.n);

    // Modified Gram-Schmidt: the amount of dot/update work grows from one to
    // m passes across a restart cycle, exposing changing compute cost.
    for (std::size_t i = 0; i <= k; ++i) {
      const std::size_t qi_offset = i * dimensions.n;
      const std::size_t h_offset = h_index(k, i, config.m);
      submit_dot_stage1(queue, config.wait_each_kernel, buffers.basis,
                        qi_offset, buffers.w, buffers.partials, dimensions.n,
                        config.partials);
      submit_dot_stage2(queue, config.wait_each_kernel, buffers.partials,
                        config.partials, buffers.h, h_offset);
      submit_axpy_update(queue, config.wait_each_kernel, buffers.w,
                         buffers.basis, qi_offset, buffers.h, h_offset,
                         dimensions.n);
    }

    const std::size_t norm_offset = h_index(k, k + 1, config.m);
    submit_norm_stage1(queue, config.wait_each_kernel, buffers.w, 0,
                       buffers.partials, dimensions.n, config.partials);
    submit_norm_stage2(queue, config.wait_each_kernel, buffers.partials,
                       config.partials, buffers.h, norm_offset);
    submit_normalize(queue, config.wait_each_kernel, buffers.w, 0, buffers.h,
                     norm_offset, buffers.basis, (k + 1) * dimensions.n,
                     dimensions.n);
  }
}

void submit_full_mode(sycl::queue &queue, const Config &config,
                      const Dimensions &dimensions, Buffers &buffers) {
  const std::size_t last_norm_offset = h_index(config.m - 1, config.m, config.m);
  const std::size_t last_basis_offset = config.m * dimensions.n;
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_arnoldi_cycle(queue, config, dimensions, buffers, true);
    // Device-side marker closes the restart before the Q*y fan-in without a
    // host scalar read or a queue wait.
    submit_restart_marker(queue, config.wait_each_kernel, buffers.h,
                          last_norm_offset, buffers.basis, last_basis_offset,
                          buffers.cycle_token, dimensions.n);
    submit_final_projection(queue, config.wait_each_kernel, buffers.x,
                            buffers.basis, buffers.coefficients,
                            buffers.cycle_token, config.fanout, dimensions.n);

    if (cycle + 1 < config.cycles) {
      // Form an approximate restarted residual r = b - A*x and normalize it
      // into q_0.  All scalar flow remains in small device buffers.
      submit_stencil(queue, config.wait_each_kernel, buffers.x, 0, buffers.w,
                     0, config.nx, config.ny, dimensions.n);
      submit_residual(queue, config.wait_each_kernel, buffers.b, buffers.w,
                      buffers.residual, dimensions.n);
      submit_norm_stage1(queue, config.wait_each_kernel, buffers.residual, 0,
                         buffers.partials, dimensions.n, config.partials);
      submit_norm_stage2(queue, config.wait_each_kernel, buffers.partials,
                         config.partials, buffers.restart_norm, 0);
      submit_normalize(queue, config.wait_each_kernel, buffers.residual, 0,
                       buffers.restart_norm, 0, buffers.basis, 0,
                       dimensions.n);
    }
  }
}

void submit_no_fanin_mode(sycl::queue &queue, const Config &config,
                          const Dimensions &dimensions, Buffers &buffers,
                          bool use_stencil) {
  const std::size_t last_basis_offset = config.m * dimensions.n;
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_arnoldi_cycle(queue, config, dimensions, buffers, use_stencil);
    if (cycle + 1 < config.cycles) {
      // Carry a deterministic device-produced seed into the next stress cycle
      // without accidentally adding the projection fan-in to this mode.
      submit_copy_basis(queue, config.wait_each_kernel, buffers.basis,
                        last_basis_offset, 0, dimensions.n);
    }
  }
}

void submit_operator_only_mode(sycl::queue &queue, const Config &config,
                               const Dimensions &dimensions, Buffers &buffers,
                               bool &last_result_in_basis) {
  last_result_in_basis = true;
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    for (std::size_t step = 0; step < config.m; ++step) {
      if (last_result_in_basis) {
        submit_stencil(queue, config.wait_each_kernel, buffers.basis, 0,
                       buffers.w, 0, config.nx, config.ny, dimensions.n);
      } else {
        submit_stencil(queue, config.wait_each_kernel, buffers.w, 0,
                       buffers.basis, 0, config.nx, config.ny, dimensions.n);
      }
      last_result_in_basis = !last_result_in_basis;
    }
  }
}

void submit_fanin_only_mode(sycl::queue &queue, const Config &config,
                            const Dimensions &dimensions, Buffers &buffers) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_final_projection(queue, config.wait_each_kernel, buffers.x,
                            buffers.basis, buffers.coefficients,
                            buffers.cycle_token, config.fanout, dimensions.n);
  }
}

void submit_samples(sycl::queue &queue, bool wait_each_kernel, Buffers &buffers,
                    std::size_t n, std::size_t basis_offset,
                    int primary_kind) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto x_acc = buffers.x.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = buffers.basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(n), sycl::id<1>(basis_offset));
    auto w_acc = buffers.w.get_access<sycl::access::mode::read>(cgh);
    auto auxiliary_acc =
        buffers.auxiliary.get_access<sycl::access::mode::read>(cgh);
    auto sample_acc =
        buffers.samples.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovSampleKernel>(
        sycl::range<1>(4), [=](sycl::item<1> item) {
          const std::size_t lane = item[0];
          const std::size_t index = lane == 0 ? 0 : lane == 1 ? n / 2
                                              : lane == 2 ? n - 1 : n / 3;
          data_t primary = x_acc[index];
          if (primary_kind == 1) {
            primary = basis_acc[index];
          } else if (primary_kind == 2) {
            primary = w_acc[index];
          }
          if (lane < 3) {
            sample_acc[lane] = primary;
          } else {
            // This mixed lane makes the independent auxiliary branch part of
            // the observable result without forcing a full host readback.
            sample_acc[lane] = x_acc[index] +
                               static_cast<data_t>(0.125) * basis_acc[index] +
                               static_cast<data_t>(0.0625) * w_acc[index] +
                               auxiliary_acc[index];
          }
        });
  });
}

double checksum_full_primary(Buffers &buffers, std::size_t n,
                             std::size_t basis_offset, int primary_kind) {
  double checksum = 0.0;
  if (primary_kind == 0) {
    sycl::host_accessor x_acc{buffers.x, sycl::read_only};
    for (std::size_t index = 0; index < n; ++index) {
      checksum += static_cast<double>(x_acc[index]);
    }
  } else if (primary_kind == 1) {
    sycl::host_accessor basis_acc{buffers.basis, sycl::read_only};
    for (std::size_t index = 0; index < n; ++index) {
      checksum += static_cast<double>(basis_acc[basis_offset + index]);
    }
  } else {
    sycl::host_accessor w_acc{buffers.w, sycl::read_only};
    for (std::size_t index = 0; index < n; ++index) {
      checksum += static_cast<double>(w_acc[index]);
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
            << " cycles=" << config.cycles << " partials=" << config.partials
            << " fanout=" << config.fanout << " mode=" << mode_name(config.mode)
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " host_read_full=" << (config.host_read_full ? 1 : 0)
            << " init_on_device=" << (config.init_on_device ? 1 : 0) << '\n';
  std::cout << "MEMORY vector_bytes=" << memory.vector_bytes
            << " Q_bytes=" << memory.basis_bytes
            << " total_estimated_bytes=" << memory.total_bytes << '\n';

  try {
    const auto init_begin = std::chrono::steady_clock::now();

    buffer_t x_buf{sycl::range<1>(dimensions.n)};
    buffer_t b_buf{sycl::range<1>(dimensions.n)};
    buffer_t basis_buf{sycl::range<1>(dimensions.basis_elements)};
    buffer_t w_buf{sycl::range<1>(dimensions.n)};
    buffer_t residual_buf{sycl::range<1>(dimensions.n)};
    buffer_t auxiliary_buf{sycl::range<1>(dimensions.n)};
    buffer_t partial_buf{sycl::range<1>(config.partials)};
    buffer_t h_buf{sycl::range<1>(dimensions.h_elements)};
    buffer_t coefficient_buf{sycl::range<1>(config.m)};
    buffer_t restart_norm_buf{sycl::range<1>(1)};
    buffer_t cycle_token_buf{sycl::range<1>(1)};
    buffer_t sample_buf{sycl::range<1>(4)};
    Buffers buffers{x_buf,          b_buf,       basis_buf, w_buf,
                    residual_buf,   auxiliary_buf, partial_buf, h_buf,
                    coefficient_buf, restart_norm_buf, cycle_token_buf,
                    sample_buf};

    sycl::queue queue;
    if (config.init_on_device) {
      initialize_on_device(queue, buffers, config, dimensions);
    } else {
      initialize_on_host(buffers, config, dimensions);
    }
    const auto init_end = std::chrono::steady_clock::now();

    const auto run_begin = std::chrono::steady_clock::now();
    int primary_kind = 0;  // 0=x, 1=one basis vector, 2=w.
    std::size_t sample_basis_offset = config.m * dimensions.n;
    switch (config.mode) {
      case Mode::Full:
        submit_full_mode(queue, config, dimensions, buffers);
        primary_kind = 0;
        sample_basis_offset = config.m * dimensions.n;
        break;
      case Mode::NoFanin:
        submit_no_fanin_mode(queue, config, dimensions, buffers, true);
        primary_kind = 1;
        sample_basis_offset = config.m * dimensions.n;
        break;
      case Mode::FaninOnly:
        submit_fanin_only_mode(queue, config, dimensions, buffers);
        primary_kind = 0;
        sample_basis_offset = (config.fanout - 1) * dimensions.n;
        break;
      case Mode::OperatorOnly: {
        bool last_result_in_basis = true;
        submit_operator_only_mode(queue, config, dimensions, buffers,
                                  last_result_in_basis);
        primary_kind = last_result_in_basis ? 1 : 2;
        sample_basis_offset = 0;
        break;
      }
      case Mode::OrthogonalizationOnly:
        submit_no_fanin_mode(queue, config, dimensions, buffers, false);
        primary_kind = 1;
        sample_basis_offset = config.m * dimensions.n;
        break;
    }

    // The tiny result buffer provides sparse host materialization by default.
    // It also makes all principal paths, including the independent branch,
    // observable before the final timing wait.
    submit_samples(queue, config.wait_each_kernel, buffers, dimensions.n,
                   sample_basis_offset, primary_kind);
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
      checksum = checksum_full_primary(buffers, dimensions.n,
                                       sample_basis_offset, primary_kind);
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
