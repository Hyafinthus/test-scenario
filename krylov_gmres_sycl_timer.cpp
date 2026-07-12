// Standalone restarted GMRES / Arnoldi-inspired SYCL buffer benchmark.
//
// H uses a 2-D Arnoldi layout:
//   H[column, row] == h(row, column), 0 <= row <= column + 1.
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
class KrylovFieldInitKernel;
class KrylovBasisInitKernel;
class KrylovBranchInitKernel;
class KrylovHInitKernel;
class KrylovCoefficientInitKernel;
class KrylovScalarInitKernel;
class KrylovStencilBasisToFieldKernel;
class KrylovStencilFieldToFieldKernel;
class KrylovStencilFieldToBasisKernel;
class KrylovOrthogonalizationSeedKernel;
class KrylovIndependentBranchKernel;
class KrylovDotStage1Kernel;
class KrylovDotStage2Kernel;
class KrylovAxpyUpdateKernel;
class KrylovNormStage1Kernel;
class KrylovNormStage2Kernel;
class KrylovNormStage2ScalarKernel;
class KrylovNormalizeHKernel;
class KrylovNormalizeRestartKernel;
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
constexpr std::size_t kBranchSlots = 3;
constexpr data_t kNormFloor = static_cast<data_t>(1.0e-20);

using field_buffer_t = sycl::buffer<data_t, 2>;
using basis_buffer_t = sycl::buffer<data_t, 2>;
using branch_buffer_t = sycl::buffer<data_t, 2>;
using vector_buffer_t = sycl::buffer<data_t, 1>;
using h_buffer_t = sycl::buffer<data_t, 2>;

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
  std::size_t branch_slots = kBranchSlots;
  std::size_t branch_elements = 0;
  std::size_t h_elements = 0;
};

struct MemoryEstimate {
  std::size_t vector_bytes = 0;
  std::size_t basis_bytes = 0;
  std::size_t branch_bytes = 0;
  std::size_t total_bytes = 0;
};

struct Buffers {
  field_buffer_t &x;
  field_buffer_t &b;
  basis_buffer_t &basis;
  field_buffer_t &w;
  field_buffer_t &residual;
  branch_buffer_t &branches;
  vector_buffer_t &partials;
  h_buffer_t &h;
  vector_buffer_t &coefficients;
  vector_buffer_t &restart_norm;
  vector_buffer_t &cycle_token;
  vector_buffer_t &samples;
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
      !checked_multiply(dimensions.branch_slots, dimensions.n,
                        dimensions.branch_elements) ||
      !checked_multiply(dimensions.basis_slots, config.m,
                        dimensions.h_elements)) {
    error = "basis, branch, or Hessenberg allocation overflows size_t";
    return false;
  }

  if (!checked_multiply(dimensions.n, sizeof(data_t), memory.vector_bytes) ||
      !checked_multiply(dimensions.basis_elements, sizeof(data_t),
                        memory.basis_bytes) ||
      !checked_multiply(dimensions.branch_elements, sizeof(data_t),
                        memory.branch_bytes)) {
    error = "byte estimate overflows size_t";
    return false;
  }

  // x, b, w, residual, and each branch plane are full 2-D fields.
  std::size_t total_elements = 0;
  std::size_t full_vector_elements = 0;
  if (!checked_multiply(dimensions.n, std::size_t{4}, full_vector_elements) ||
      !checked_add(full_vector_elements, dimensions.basis_elements,
                   total_elements) ||
      !checked_add(total_elements, dimensions.branch_elements, total_elements) ||
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

// These arithmetic-only functions are callable from both host initialization
// and the optional device initializer.  They deliberately avoid random input.
inline data_t rhs_value(std::size_t row, std::size_t col) {
  return static_cast<data_t>(0.75) +
         static_cast<data_t>(row % 97) * static_cast<data_t>(0.0025) +
         static_cast<data_t>(col % 89) * static_cast<data_t>(0.0015);
}

inline data_t basis_seed(std::size_t basis_index, std::size_t row,
                         std::size_t col) {
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
  sycl::host_accessor branch_acc{buffers.branches, sycl::write_only};
  sycl::host_accessor h_acc{buffers.h, sycl::write_only};
  sycl::host_accessor coefficient_acc{buffers.coefficients, sycl::write_only};
  sycl::host_accessor restart_norm_acc{buffers.restart_norm, sycl::write_only};
  sycl::host_accessor cycle_token_acc{buffers.cycle_token, sycl::write_only};

  double q0_squared_norm = 0.0;
  for (std::size_t row = 0; row < config.nx; ++row) {
    for (std::size_t col = 0; col < config.ny; ++col) {
      const double value = static_cast<double>(basis_seed(0, row, col));
      q0_squared_norm += value * value;
    }
  }
  const data_t q0_scale =
      static_cast<data_t>(1.0 / std::sqrt(q0_squared_norm));
  const data_t other_basis_scale = static_cast<data_t>(
      1.0 / std::sqrt(static_cast<double>(dimensions.n)));

  for (std::size_t row = 0; row < config.nx; ++row) {
    for (std::size_t col = 0; col < config.ny; ++col) {
      const sycl::id<2> cell(row, col);
      const data_t rhs = rhs_value(row, col);
      x_acc[cell] = static_cast<data_t>(0);
      b_acc[cell] = rhs;
      w_acc[cell] = static_cast<data_t>(0);
      residual_acc[cell] = rhs;
      for (std::size_t branch = 0; branch < dimensions.branch_slots; ++branch) {
        branch_acc[sycl::id<2>(branch * config.nx + row, col)] =
            static_cast<data_t>(0.001 * static_cast<double>(branch + 1)) * rhs;
      }
    }
  }
  for (std::size_t basis_index = 0; basis_index < dimensions.basis_slots;
       ++basis_index) {
    const data_t scale = basis_index == 0 ? q0_scale : other_basis_scale;
    for (std::size_t row = 0; row < config.nx; ++row) {
      for (std::size_t col = 0; col < config.ny; ++col) {
        basis_acc[sycl::id<2>(basis_index * config.nx + row, col)] =
            scale * basis_seed(basis_index, row, col);
      }
    }
  }
  for (std::size_t column = 0; column < config.m; ++column) {
    for (std::size_t row = 0; row < dimensions.basis_slots; ++row) {
      h_acc[sycl::id<2>(column, row)] = static_cast<data_t>(0);
    }
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
  double q0_squared_norm = 0.0;
  for (std::size_t row = 0; row < config.nx; ++row) {
    for (std::size_t col = 0; col < config.ny; ++col) {
      const double value = static_cast<double>(basis_seed(0, row, col));
      q0_squared_norm += value * value;
    }
  }
  const data_t q0_scale =
      static_cast<data_t>(1.0 / std::sqrt(q0_squared_norm));
  const data_t basis_scale = static_cast<data_t>(
      1.0 / std::sqrt(static_cast<double>(dimensions.n)));

  queue.submit([&](sycl::handler &cgh) {
    auto x_acc =
        buffers.x.get_access<sycl::access::mode::discard_write>(cgh);
    auto b_acc =
        buffers.b.get_access<sycl::access::mode::discard_write>(cgh);
    auto w_acc =
        buffers.w.get_access<sycl::access::mode::discard_write>(cgh);
    auto residual_acc =
        buffers.residual.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovFieldInitKernel>(
        sycl::range<2>(config.nx, config.ny), [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const sycl::id<2> cell(row, col);
      const data_t rhs = rhs_value(row, col);
      x_acc[cell] = static_cast<data_t>(0);
      b_acc[cell] = rhs;
      w_acc[cell] = static_cast<data_t>(0);
      residual_acc[cell] = rhs;
    });
  });

  queue.submit([&](sycl::handler &cgh) {
    auto branch_acc =
        buffers.branches.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovBranchInitKernel>(
        sycl::range<2>(dimensions.branch_slots * config.nx, config.ny),
        [=](sycl::item<2> item) {
      const std::size_t stacked_row = item[0];
      const std::size_t branch = stacked_row / config.nx;
      const std::size_t row = stacked_row - branch * config.nx;
      const std::size_t col = item[1];
      const data_t rhs = rhs_value(row, col);
      branch_acc[sycl::id<2>(stacked_row, col)] =
          static_cast<data_t>(0.001 * static_cast<double>(branch + 1)) * rhs;
    });
  });

  queue.submit([&](sycl::handler &cgh) {
    auto basis_acc =
        buffers.basis.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovBasisInitKernel>(
        sycl::range<2>(dimensions.basis_slots * config.nx, config.ny),
        [=](sycl::item<2> item) {
      const std::size_t stacked_row = item[0];
      const std::size_t basis_index = stacked_row / config.nx;
      const std::size_t row = stacked_row - basis_index * config.nx;
      const std::size_t col = item[1];
      const data_t scale = basis_index == 0 ? q0_scale : basis_scale;
      basis_acc[sycl::id<2>(stacked_row, col)] =
          scale * basis_seed(basis_index, row, col);
    });
  });

  queue.submit([&](sycl::handler &cgh) {
    auto h_acc =
        buffers.h.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovHInitKernel>(
        sycl::range<2>(config.m, dimensions.basis_slots),
        [=](sycl::item<2> item) {
      h_acc[sycl::id<2>(item[0], item[1])] = static_cast<data_t>(0);
    });
  });

  queue.submit([&](sycl::handler &cgh) {
    auto coefficient_acc = buffers.coefficients.get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovCoefficientInitKernel>(
        sycl::range<1>(config.m), [=](sycl::item<1> item) {
      coefficient_acc[item[0]] = coefficient_value(item[0]);
    });
  });

  queue.submit([&](sycl::handler &cgh) {
    auto restart_norm_acc = buffers.restart_norm.get_access<
        sycl::access::mode::discard_write>(cgh);
    auto cycle_token_acc = buffers.cycle_token.get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovScalarInitKernel>(sycl::range<1>(1),
                                             [=](sycl::item<1>) {
      restart_norm_acc[0] = static_cast<data_t>(1);
      cycle_token_acc[0] = static_cast<data_t>(1);
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

void wait_cycle_boundary(sycl::queue &queue) {
  queue.wait();
}

void submit_stencil_basis_to_field(sycl::queue &queue, bool wait_each_kernel,
                                   basis_buffer_t &input,
                                   std::size_t input_slot,
                                   field_buffer_t &output,
                                   std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(input_slot * nx, 0));
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh);
    cgh.parallel_for<KrylovStencilBasisToFieldKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const sycl::id<2> center_id(row, col);
      const data_t center = input_acc[center_id];
      const data_t north =
          row > 0 ? input_acc[sycl::id<2>(row - 1, col)]
                  : static_cast<data_t>(0);
      const data_t south =
          row + 1 < nx ? input_acc[sycl::id<2>(row + 1, col)]
                       : static_cast<data_t>(0);
      const data_t west =
          col > 0 ? input_acc[sycl::id<2>(row, col - 1)]
                  : static_cast<data_t>(0);
      const data_t east =
          col + 1 < ny ? input_acc[sycl::id<2>(row, col + 1)]
                       : static_cast<data_t>(0);
      output_acc[sycl::id<2>(row, col)] =
          static_cast<data_t>(4) * center - north - south - west - east;
    });
  });
}

void submit_stencil_field_to_field(sycl::queue &queue, bool wait_each_kernel,
                                   field_buffer_t &input,
                                   field_buffer_t &output,
                                   std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh);
    cgh.parallel_for<KrylovStencilFieldToFieldKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const data_t center = input_acc[sycl::id<2>(row, col)];
      const data_t north =
          row > 0 ? input_acc[sycl::id<2>(row - 1, col)]
                  : static_cast<data_t>(0);
      const data_t south =
          row + 1 < nx ? input_acc[sycl::id<2>(row + 1, col)]
                       : static_cast<data_t>(0);
      const data_t west =
          col > 0 ? input_acc[sycl::id<2>(row, col - 1)]
                  : static_cast<data_t>(0);
      const data_t east =
          col + 1 < ny ? input_acc[sycl::id<2>(row, col + 1)]
                       : static_cast<data_t>(0);
      output_acc[sycl::id<2>(row, col)] =
          static_cast<data_t>(4) * center - north - south - west - east;
    });
  });
}

void submit_stencil_field_to_basis(sycl::queue &queue, bool wait_each_kernel,
                                   field_buffer_t &input,
                                   basis_buffer_t &output,
                                   std::size_t output_slot,
                                   std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(output_slot * nx, 0));
    cgh.parallel_for<KrylovStencilFieldToBasisKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const data_t center = input_acc[sycl::id<2>(row, col)];
      const data_t north =
          row > 0 ? input_acc[sycl::id<2>(row - 1, col)]
                  : static_cast<data_t>(0);
      const data_t south =
          row + 1 < nx ? input_acc[sycl::id<2>(row + 1, col)]
                       : static_cast<data_t>(0);
      const data_t west =
          col > 0 ? input_acc[sycl::id<2>(row, col - 1)]
                  : static_cast<data_t>(0);
      const data_t east =
          col + 1 < ny ? input_acc[sycl::id<2>(row, col + 1)]
                       : static_cast<data_t>(0);
      output_acc[sycl::id<2>(row, col)] =
          static_cast<data_t>(4) * center - north - south - west - east;
    });
  });
}

void submit_orthogonalization_seed(sycl::queue &queue, bool wait_each_kernel,
                                   field_buffer_t &rhs, basis_buffer_t &basis,
                                   std::size_t basis_slot, field_buffer_t &w,
                                   std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto rhs_acc = rhs.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(basis_slot * nx, 0));
    auto w_acc = w.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovOrthogonalizationSeedKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const std::size_t index = row * ny + col;
          const sycl::id<2> cell(row, col);
          const data_t perturbation = static_cast<data_t>(index % 31) *
                                      static_cast<data_t>(0.0001);
          w_acc[cell] = rhs_acc[cell] - static_cast<data_t>(0.125) *
                                           basis_acc[sycl::id<2>(row, col)] +
                        perturbation;
        });
  });
}

void submit_independent_branch(sycl::queue &queue, bool wait_each_kernel,
                               field_buffer_t &rhs, basis_buffer_t &basis,
                               std::size_t basis_slot,
                               branch_buffer_t &branches,
                               std::size_t branch_slot,
                               std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto rhs_acc = rhs.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(basis_slot * nx, 0));
    auto branch_acc = branches.get_access<sycl::access::mode::read_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(branch_slot * nx, 0));
    cgh.parallel_for<KrylovIndependentBranchKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> cell(row, col);
          const sycl::id<2> branch_id(row, col);
          const data_t branch_weight =
              static_cast<data_t>(0.002 + 0.001 * branch_slot);
          const data_t spatial =
              static_cast<data_t>((row + branch_slot * 13 + col * 3) % 37) *
              static_cast<data_t>(0.00001);
          branch_acc[branch_id] =
              static_cast<data_t>(0.995) * branch_acc[branch_id] +
              branch_weight *
                  (rhs_acc[cell] - basis_acc[sycl::id<2>(row, col)]) +
              spatial;
        });
  });
}

void submit_dot_stage1(sycl::queue &queue, bool wait_each_kernel,
                       basis_buffer_t &basis, std::size_t basis_slot,
                       field_buffer_t &w, vector_buffer_t &partials,
                       std::size_t nx, std::size_t ny, std::size_t n,
                       std::size_t partial_count) {
  const std::size_t chunk =
      n / partial_count + (n % partial_count == 0 ? 0 : 1);
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(basis_slot * nx, 0));
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
            const std::size_t row = index / ny;
            const std::size_t col = index - row * ny;
            sum += basis_acc[sycl::id<2>(row, col)] *
                   w_acc[sycl::id<2>(row, col)];
          }
          partial_acc[partial_index] = sum;
        });
  });
}

void submit_dot_stage2(sycl::queue &queue, bool wait_each_kernel,
                       vector_buffer_t &partials, std::size_t partial_count,
                       h_buffer_t &scalar_output, std::size_t h_column,
                       std::size_t h_row) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto partial_acc = partials.get_access<sycl::access::mode::read>(cgh);
    auto scalar_acc = scalar_output.get_access<
        sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(1, 1), sycl::id<2>(h_column, h_row));
    cgh.parallel_for<KrylovDotStage2Kernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          data_t sum = static_cast<data_t>(0);
          for (std::size_t partial_index = 0; partial_index < partial_count;
               ++partial_index) {
            sum += partial_acc[partial_index];
          }
          scalar_acc[sycl::id<2>(0, 0)] = sum;
        });
  });
}

void submit_axpy_update(sycl::queue &queue, bool wait_each_kernel,
                        field_buffer_t &w, basis_buffer_t &basis,
                        std::size_t basis_slot, h_buffer_t &h,
                        std::size_t h_column, std::size_t h_row,
                        std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto w_acc = w.get_access<sycl::access::mode::read_write>(cgh);
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(basis_slot * nx, 0));
    auto h_acc = h.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(1, 1), sycl::id<2>(h_column, h_row));
    cgh.parallel_for<KrylovAxpyUpdateKernel>(sycl::range<2>(nx, ny),
                                              [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      w_acc[sycl::id<2>(row, col)] -=
          h_acc[sycl::id<2>(0, 0)] * basis_acc[sycl::id<2>(row, col)];
    });
  });
}

void submit_norm_stage1(sycl::queue &queue, bool wait_each_kernel,
                        field_buffer_t &input, vector_buffer_t &partials,
                        std::size_t nx, std::size_t ny, std::size_t n,
                        std::size_t partial_count) {
  const std::size_t chunk =
      n / partial_count + (n % partial_count == 0 ? 0 : 1);
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(cgh);
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
            const std::size_t row = index / ny;
            const std::size_t col = index - row * ny;
            const data_t value = input_acc[sycl::id<2>(row, col)];
            sum += value * value;
          }
          partial_acc[partial_index] = sum;
        });
  });
}

void submit_norm_stage2(sycl::queue &queue, bool wait_each_kernel,
                        vector_buffer_t &partials, std::size_t partial_count,
                        h_buffer_t &scalar_output, std::size_t h_column,
                        std::size_t h_row) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto partial_acc = partials.get_access<sycl::access::mode::read>(cgh);
    auto scalar_acc = scalar_output.get_access<
        sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(1, 1), sycl::id<2>(h_column, h_row));
    cgh.parallel_for<KrylovNormStage2Kernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          data_t sum = static_cast<data_t>(0);
          for (std::size_t partial_index = 0; partial_index < partial_count;
               ++partial_index) {
            sum += partial_acc[partial_index];
          }
          scalar_acc[sycl::id<2>(0, 0)] =
              sycl::sqrt(sum > static_cast<data_t>(0) ? sum
                                                       : static_cast<data_t>(0));
        });
  });
}

void submit_norm_stage2_scalar(sycl::queue &queue, bool wait_each_kernel,
                               vector_buffer_t &partials,
                               std::size_t partial_count,
                               vector_buffer_t &scalar_output) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto partial_acc = partials.get_access<sycl::access::mode::read>(cgh);
    auto scalar_acc = scalar_output.get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovNormStage2ScalarKernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          data_t sum = static_cast<data_t>(0);
          for (std::size_t partial_index = 0; partial_index < partial_count;
               ++partial_index) {
            sum += partial_acc[partial_index];
          }
          scalar_acc[0] =
              sycl::sqrt(sum > static_cast<data_t>(0) ? sum
                                                       : static_cast<data_t>(0));
        });
  });
}

void submit_normalize_h_to_basis(sycl::queue &queue, bool wait_each_kernel,
                                 field_buffer_t &input, h_buffer_t &norm,
                                 std::size_t h_column, std::size_t h_row,
                                 basis_buffer_t &output,
                                 std::size_t output_slot,
                                 std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(cgh);
    auto norm_acc = norm.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(1, 1), sycl::id<2>(h_column, h_row));
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(output_slot * nx, 0));
    cgh.parallel_for<KrylovNormalizeHKernel>(sycl::range<2>(nx, ny),
                                             [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const data_t denominator = norm_acc[sycl::id<2>(0, 0)];
      output_acc[sycl::id<2>(row, col)] =
          denominator > kNormFloor
              ? input_acc[sycl::id<2>(row, col)] / denominator
              : static_cast<data_t>(0);
    });
  });
}

void submit_normalize_scalar_to_basis(sycl::queue &queue, bool wait_each_kernel,
                                      field_buffer_t &input,
                                      vector_buffer_t &norm,
                                      basis_buffer_t &output,
                                      std::size_t output_slot,
                                      std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto input_acc = input.get_access<sycl::access::mode::read>(cgh);
    auto norm_acc = norm.get_access<sycl::access::mode::read>(cgh);
    auto output_acc = output.get_access<sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(output_slot * nx, 0));
    cgh.parallel_for<KrylovNormalizeRestartKernel>(sycl::range<2>(nx, ny),
                                                  [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const data_t denominator = norm_acc[0];
      output_acc[sycl::id<2>(row, col)] =
          denominator > kNormFloor
              ? input_acc[sycl::id<2>(row, col)] / denominator
              : static_cast<data_t>(0);
    });
  });
}

void submit_restart_marker(sycl::queue &queue, bool wait_each_kernel,
                           h_buffer_t &h, std::size_t h_column,
                           std::size_t h_row, basis_buffer_t &basis,
                           std::size_t last_basis_slot,
                           vector_buffer_t &cycle_token,
                           std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto h_acc = h.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(1, 1), sycl::id<2>(h_column, h_row));
    // Reading q_m gives the marker a dependency on the final normalize too,
    // even when the correction itself only consumes q_0 through q_(m-1).
    auto last_basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(last_basis_slot * nx, 0));
    auto token_acc = cycle_token.get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovRestartMarkerKernel>(
        sycl::range<1>(1), [=](sycl::item<1>) {
          const data_t h_value = h_acc[sycl::id<2>(0, 0)];
          const data_t q_value = last_basis_acc[sycl::id<2>(0, 0)];
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
                             field_buffer_t &x, basis_buffer_t &basis,
                             vector_buffer_t &coefficients,
                             vector_buffer_t &cycle_token,
                             std::size_t fanout,
                             std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto x_acc = x.get_access<sycl::access::mode::read_write>(cgh);
    // This single contiguous range represents the read-mostly block of
    // Krylov vectors consumed by x += Q*y.
    auto basis_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(fanout * nx, ny), sycl::id<2>(0, 0));
    auto coefficient_acc = coefficients.get_access<sycl::access::mode::read>(
        cgh, sycl::range<1>(fanout), sycl::id<1>(0));
    auto token_acc =
        cycle_token.get_access<sycl::access::mode::read>(cgh);
    cgh.parallel_for<KrylovFinalProjectionKernel>(
        sycl::range<2>(nx, ny), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          data_t correction = static_cast<data_t>(0);
          for (std::size_t basis_index = 0; basis_index < fanout;
               ++basis_index) {
            correction += coefficient_acc[basis_index] *
                          basis_acc[sycl::id<2>(basis_index * nx + row, col)];
          }
          x_acc[sycl::id<2>(row, col)] += token_acc[0] * correction;
        });
  });
}

void submit_residual(sycl::queue &queue, bool wait_each_kernel,
                     field_buffer_t &b, field_buffer_t &ax,
                     field_buffer_t &residual,
                     std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto b_acc = b.get_access<sycl::access::mode::read>(cgh);
    auto ax_acc = ax.get_access<sycl::access::mode::read>(cgh);
    auto residual_acc =
        residual.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovResidualKernel>(sycl::range<2>(nx, ny),
                                            [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      const sycl::id<2> cell(row, col);
      residual_acc[cell] = b_acc[cell] - ax_acc[cell];
    });
  });
}

void submit_copy_basis(sycl::queue &queue, bool wait_each_kernel,
                       basis_buffer_t &basis, std::size_t source_slot,
                       std::size_t destination_slot,
                       std::size_t nx, std::size_t ny) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto source_acc = basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(source_slot * nx, 0));
    auto destination_acc = basis.get_access<
        sycl::access::mode::discard_write>(
        cgh, sycl::range<2>(nx, ny), sycl::id<2>(destination_slot * nx, 0));
    cgh.parallel_for<KrylovCopyBasisKernel>(sycl::range<2>(nx, ny),
                                             [=](sycl::item<2> item) {
      const std::size_t row = item[0];
      const std::size_t col = item[1];
      destination_acc[sycl::id<2>(row, col)] =
          source_acc[sycl::id<2>(row, col)];
    });
  });
}

void submit_arnoldi_cycle(sycl::queue &queue, const Config &config,
                           const Dimensions &dimensions, Buffers &buffers,
                           bool use_stencil) {
  for (std::size_t k = 0; k < config.m; ++k) {
    if (use_stencil) {
      // Large regular phase: Celerity-friendly in a later neighborhood-mapped
      // implementation and worthwhile for a multi-device scheduler to split.
      submit_stencil_basis_to_field(queue, config.wait_each_kernel,
                                    buffers.basis, k, buffers.w, config.nx,
                                    config.ny);
    } else {
      submit_orthogonalization_seed(
          queue, config.wait_each_kernel, buffers.b, buffers.basis, k,
          buffers.w, config.nx, config.ny);
    }

    // These branches share q_k and b with the main path but do not touch w or
    // each other.  Distinct branch planes make the logical parallelism visible
    // through accessor subranges instead of through artificial buffers.
    for (std::size_t branch = 0; branch < dimensions.branch_slots; ++branch) {
      submit_independent_branch(queue, config.wait_each_kernel, buffers.b,
                                buffers.basis, k, buffers.branches, branch,
                                config.nx, config.ny);
    }

    // Modified Gram-Schmidt: the amount of dot/update work grows from one to
    // m passes across a restart cycle, exposing changing compute cost.
    for (std::size_t i = 0; i <= k; ++i) {
      submit_dot_stage1(queue, config.wait_each_kernel, buffers.basis,
                        i, buffers.w, buffers.partials, config.nx, config.ny,
                        dimensions.n, config.partials);
      submit_dot_stage2(queue, config.wait_each_kernel, buffers.partials,
                        config.partials, buffers.h, k, i);
      submit_axpy_update(queue, config.wait_each_kernel, buffers.w,
                         buffers.basis, i, buffers.h, k, i, config.nx,
                         config.ny);
    }

    submit_norm_stage1(queue, config.wait_each_kernel, buffers.w,
                       buffers.partials, config.nx, config.ny, dimensions.n,
                       config.partials);
    submit_norm_stage2(queue, config.wait_each_kernel, buffers.partials,
                       config.partials, buffers.h, k, k + 1);
    submit_normalize_h_to_basis(queue, config.wait_each_kernel, buffers.w,
                                buffers.h, k, k + 1, buffers.basis, k + 1,
                                config.nx, config.ny);
  }
}

void submit_full_mode(sycl::queue &queue, const Config &config,
                      const Dimensions &dimensions, Buffers &buffers) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_arnoldi_cycle(queue, config, dimensions, buffers, true);
    // Device-side marker closes the restart before the Q*y fan-in without a
    // host scalar read or a queue wait.
    submit_restart_marker(queue, config.wait_each_kernel, buffers.h,
                          config.m - 1, config.m, buffers.basis, config.m,
                          buffers.cycle_token, config.nx, config.ny);
    submit_final_projection(queue, config.wait_each_kernel, buffers.x,
                            buffers.basis, buffers.coefficients,
                            buffers.cycle_token, config.fanout, config.nx,
                            config.ny);

    if (cycle + 1 < config.cycles) {
      // Form an approximate restarted residual r = b - A*x and normalize it
      // into q_0.  All scalar flow remains in small device buffers.
      submit_stencil_field_to_field(queue, config.wait_each_kernel, buffers.x,
                                    buffers.w, config.nx, config.ny);
      submit_residual(queue, config.wait_each_kernel, buffers.b, buffers.w,
                      buffers.residual, config.nx, config.ny);
      submit_norm_stage1(queue, config.wait_each_kernel, buffers.residual,
                         buffers.partials, config.nx, config.ny, dimensions.n,
                         config.partials);
      submit_norm_stage2_scalar(queue, config.wait_each_kernel,
                                buffers.partials, config.partials,
                                buffers.restart_norm);
      submit_normalize_scalar_to_basis(queue, config.wait_each_kernel,
                                       buffers.residual, buffers.restart_norm,
                                       buffers.basis, 0, config.nx, config.ny);
    }
    wait_cycle_boundary(queue);
  }
}

void submit_no_fanin_mode(sycl::queue &queue, const Config &config,
                          const Dimensions &dimensions, Buffers &buffers,
                          bool use_stencil) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_arnoldi_cycle(queue, config, dimensions, buffers, use_stencil);
    if (cycle + 1 < config.cycles) {
      // Carry a deterministic device-produced seed into the next stress cycle
      // without accidentally adding the projection fan-in to this mode.
      submit_copy_basis(queue, config.wait_each_kernel, buffers.basis,
                        config.m, 0, config.nx, config.ny);
    }
    wait_cycle_boundary(queue);
  }
}

void submit_operator_only_mode(sycl::queue &queue, const Config &config,
                               const Dimensions &dimensions, Buffers &buffers,
                               bool &last_result_in_basis) {
  last_result_in_basis = true;
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    for (std::size_t step = 0; step < config.m; ++step) {
      if (last_result_in_basis) {
        submit_stencil_basis_to_field(queue, config.wait_each_kernel,
                                      buffers.basis, 0, buffers.w, config.nx,
                                      config.ny);
      } else {
        submit_stencil_field_to_basis(queue, config.wait_each_kernel,
                                      buffers.w, buffers.basis, 0, config.nx,
                                      config.ny);
      }
      last_result_in_basis = !last_result_in_basis;
    }
    wait_cycle_boundary(queue);
  }
}

void submit_fanin_only_mode(sycl::queue &queue, const Config &config,
                            const Dimensions &dimensions, Buffers &buffers) {
  for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
    submit_final_projection(queue, config.wait_each_kernel, buffers.x,
                            buffers.basis, buffers.coefficients,
                            buffers.cycle_token, config.fanout, config.nx,
                            config.ny);
    wait_cycle_boundary(queue);
  }
}

void submit_samples(sycl::queue &queue, bool wait_each_kernel, Buffers &buffers,
                    const Dimensions &dimensions, const Config &config,
                    std::size_t basis_slot, int primary_kind) {
  submit_command(queue, wait_each_kernel, [&](sycl::handler &cgh) {
    auto x_acc = buffers.x.get_access<sycl::access::mode::read>(cgh);
    auto basis_acc = buffers.basis.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(config.nx, config.ny),
        sycl::id<2>(basis_slot * config.nx, 0));
    auto w_acc = buffers.w.get_access<sycl::access::mode::read>(cgh);
    auto branch_acc = buffers.branches.get_access<sycl::access::mode::read>(
        cgh, sycl::range<2>(dimensions.branch_slots * config.nx, config.ny),
        sycl::id<2>(0, 0));
    auto sample_acc =
        buffers.samples.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<KrylovSampleKernel>(
        sycl::range<1>(4), [=](sycl::item<1> item) {
          const std::size_t lane = item[0];
          const std::size_t index = lane == 0 ? 0
                                  : lane == 1 ? dimensions.n / 2
                                  : lane == 2 ? dimensions.n - 1
                                              : dimensions.n / 3;
          const std::size_t row = index / config.ny;
          const std::size_t col = index - row * config.ny;
          const sycl::id<2> cell(row, col);
          data_t primary = x_acc[cell];
          if (primary_kind == 1) {
            primary = basis_acc[cell];
          } else if (primary_kind == 2) {
            primary = w_acc[cell];
          }
          if (lane < 3) {
            sample_acc[lane] = primary;
          } else {
            data_t branch_mix = static_cast<data_t>(0);
            for (std::size_t branch = 0; branch < dimensions.branch_slots;
                 ++branch) {
              branch_mix +=
                  static_cast<data_t>(0.25) *
                  branch_acc[sycl::id<2>(branch * config.nx + row, col)];
            }
            sample_acc[lane] =
                x_acc[cell] +
                static_cast<data_t>(0.125) * basis_acc[cell] +
                static_cast<data_t>(0.0625) * w_acc[cell] + branch_mix;
          }
        });
  });
}

double checksum_full_primary(Buffers &buffers, const Config &config,
                             std::size_t basis_slot, int primary_kind) {
  double checksum = 0.0;
  if (primary_kind == 0) {
    sycl::host_accessor x_acc{buffers.x, sycl::read_only};
    for (std::size_t row = 0; row < config.nx; ++row) {
      for (std::size_t col = 0; col < config.ny; ++col) {
        checksum += static_cast<double>(x_acc[sycl::id<2>(row, col)]);
      }
    }
  } else if (primary_kind == 1) {
    sycl::host_accessor basis_acc{buffers.basis, sycl::read_only};
    for (std::size_t row = 0; row < config.nx; ++row) {
      for (std::size_t col = 0; col < config.ny; ++col) {
        checksum += static_cast<double>(
            basis_acc[sycl::id<2>(basis_slot * config.nx + row, col)]);
      }
    }
  } else {
    sycl::host_accessor w_acc{buffers.w, sycl::read_only};
    for (std::size_t row = 0; row < config.nx; ++row) {
      for (std::size_t col = 0; col < config.ny; ++col) {
        checksum += static_cast<double>(w_acc[sycl::id<2>(row, col)]);
      }
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
            << " branch_bytes=" << memory.branch_bytes
            << " total_estimated_bytes=" << memory.total_bytes << '\n';

  try {
    const auto init_begin = std::chrono::steady_clock::now();

    field_buffer_t x_buf{sycl::range<2>(config.nx, config.ny)};
    field_buffer_t b_buf{sycl::range<2>(config.nx, config.ny)};
    basis_buffer_t basis_buf{
        sycl::range<2>(dimensions.basis_slots * config.nx, config.ny)};
    field_buffer_t w_buf{sycl::range<2>(config.nx, config.ny)};
    field_buffer_t residual_buf{sycl::range<2>(config.nx, config.ny)};
    branch_buffer_t branch_buf{
        sycl::range<2>(dimensions.branch_slots * config.nx, config.ny)};
    vector_buffer_t partial_buf{sycl::range<1>(config.partials)};
    h_buffer_t h_buf{sycl::range<2>(config.m, dimensions.basis_slots)};
    vector_buffer_t coefficient_buf{sycl::range<1>(config.m)};
    vector_buffer_t restart_norm_buf{sycl::range<1>(1)};
    vector_buffer_t cycle_token_buf{sycl::range<1>(1)};
    vector_buffer_t sample_buf{sycl::range<1>(4)};
    Buffers buffers{x_buf,          b_buf,       basis_buf, w_buf,
                    residual_buf,   branch_buf,  partial_buf, h_buf,
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
    std::size_t sample_basis_slot = config.m;
    switch (config.mode) {
      case Mode::Full:
        submit_full_mode(queue, config, dimensions, buffers);
        primary_kind = 0;
        sample_basis_slot = config.m;
        break;
      case Mode::NoFanin:
        submit_no_fanin_mode(queue, config, dimensions, buffers, true);
        primary_kind = 1;
        sample_basis_slot = config.m;
        break;
      case Mode::FaninOnly:
        submit_fanin_only_mode(queue, config, dimensions, buffers);
        primary_kind = 0;
        sample_basis_slot = config.fanout - 1;
        break;
      case Mode::OperatorOnly: {
        bool last_result_in_basis = true;
        submit_operator_only_mode(queue, config, dimensions, buffers,
                                  last_result_in_basis);
        primary_kind = last_result_in_basis ? 1 : 2;
        sample_basis_slot = 0;
        break;
      }
      case Mode::OrthogonalizationOnly:
        submit_no_fanin_mode(queue, config, dimensions, buffers, false);
        primary_kind = 1;
        sample_basis_slot = config.m;
        break;
    }

    // The tiny result buffer provides sparse host materialization by default.
    // It also makes all principal paths, including the independent branch,
    // observable before the final timing wait.
    submit_samples(queue, config.wait_each_kernel, buffers, dimensions, config,
                   sample_basis_slot, primary_kind);
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
      checksum =
          checksum_full_primary(buffers, config, sample_basis_slot,
                                primary_kind);
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
