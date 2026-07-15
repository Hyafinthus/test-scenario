// Multi-system real-space Kohn-Sham DFT / Chebyshev-filtered subspace
// iteration (CheFSI) miniapp. Each material system owns a private orbital
// block and advances one SCF window through:
//
//   Chebyshev filter -> {overlap, H*Psi -> projected H} -> transform
//                    -> subspace rotation -> density -> potential mixing
//
// The kernels deliberately mirror realistic block electronic-structure
// phases while keeping the program standalone and buffer/accessor-only.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/time.h>
#include <utility>
#include <vector>

class DftChebFirstKernel;
class DftChebRecurrenceKernel;
class DftOverlapPartialKernel;
class DftOverlapReduceKernel;
class DftHamiltonianApplyKernel;
class DftProjectedPartialKernel;
class DftProjectedReduceKernel;
class DftSubspaceTransformKernel;
class DftSubspaceRotateKernel;
class DftDensityKernel;
class DftPotentialMixKernel;

namespace {

enum class Mode { Mixed, Uniform, SingleLarge };

struct Config {
  std::size_t nx = 96;
  std::size_t ny = 96;
  std::size_t nz = 96;
  std::size_t bands = 32;
  std::size_t systems = 8;
  std::size_t classes = 4;
  std::size_t coarsen_percent = 8;
  std::size_t scf_cycles = 8;
  std::size_t cheb_degree = 12;
  std::size_t partials = 128;
  std::size_t window_scf = 1;
  double memory_limit_gib = 0.0;
  Mode mode = Mode::Mixed;
  bool wait_each_kernel = false;
  bool host_read_full = false;
  bool verify = true;
};

struct SystemSpec {
  std::size_t class_id = 0;
  std::size_t systems = 0;
  std::size_t nx = 0;
  std::size_t ny = 0;
  std::size_t nz = 0;
  std::size_t grid = 0;
  std::size_t bands = 0;
  std::size_t orbital_values = 0;
  std::size_t bytes_per_system = 0;
  float potential_scale = 0.0f;
};

double get_time() {
  timeval tv{};
  gettimeofday(&tv, nullptr);
  return static_cast<double>(tv.tv_sec) +
         static_cast<double>(tv.tv_usec) / 1.0e6;
}

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Mixed:
    return "mixed";
  case Mode::Uniform:
    return "uniform";
  case Mode::SingleLarge:
    return "single-large";
  }
  return "unknown";
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --nx/--ny/--nz <int>       Finest real-space grid (default 96^3)\n"
      << "  --bands <int>              Orbital block width (default 32)\n"
      << "  --systems <int>            Independent material systems (default 8)\n"
      << "  --classes <int>            Mixed fidelity classes (default 4)\n"
      << "  --coarsen-percent <int>    Grid reduction per class (default 8)\n"
      << "  --scf-cycles <int>         Timed SCF windows (default 8)\n"
      << "  --cheb-degree <int>        Filter degree, >=2 (default 12)\n"
      << "  --partials <int>           Gram reduction chunks (default 128)\n"
      << "  --window-scf <int>         SCF cycles per queue.wait (default 1)\n"
      << "  --memory-limit-gib <real>  Fail above estimate; 0 is unlimited\n"
      << "  --mode <mixed|uniform|single-large>\n"
      << "  --wait-each-kernel <0|1>   Debug only (default 0)\n"
      << "  --host-read-full <0|1>     Check every final orbital (default 0)\n"
      << "  --verify <0|1>             Enable finite/physical checks (default 1)\n";
}

bool parse_size(const std::string &text, std::size_t &value,
                bool allow_zero = false) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      (!allow_zero && parsed == 0) ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_double(const std::string &text, double &value) {
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_binary(const std::string &text, bool &value) {
  if (text == "0" || text == "1") {
    value = text == "1";
    return true;
  }
  return false;
}

bool parse_mode(const std::string &text, Mode &mode) {
  if (text == "mixed") {
    mode = Mode::Mixed;
  } else if (text == "uniform") {
    mode = Mode::Uniform;
  } else if (text == "single-large") {
    mode = Mode::SingleLarge;
  } else {
    return false;
  }
  return true;
}

bool parse_arguments(int argc, char **argv, Config &config,
                     std::string &error, bool &help) {
  for (int arg = 1; arg < argc; ++arg) {
    const std::string option = argv[arg];
    if (option == "--help" || option == "-h") {
      help = true;
      return true;
    }
    if (arg + 1 >= argc) {
      error = "missing value for " + option;
      return false;
    }
    const std::string value = argv[++arg];
    bool parsed = false;
    if (option == "--nx") {
      parsed = parse_size(value, config.nx);
    } else if (option == "--ny") {
      parsed = parse_size(value, config.ny);
    } else if (option == "--nz") {
      parsed = parse_size(value, config.nz);
    } else if (option == "--bands") {
      parsed = parse_size(value, config.bands);
    } else if (option == "--systems") {
      parsed = parse_size(value, config.systems);
    } else if (option == "--classes") {
      parsed = parse_size(value, config.classes);
    } else if (option == "--coarsen-percent") {
      parsed = parse_size(value, config.coarsen_percent, true);
    } else if (option == "--scf-cycles") {
      parsed = parse_size(value, config.scf_cycles);
    } else if (option == "--cheb-degree") {
      parsed = parse_size(value, config.cheb_degree);
    } else if (option == "--partials") {
      parsed = parse_size(value, config.partials);
    } else if (option == "--window-scf") {
      parsed = parse_size(value, config.window_scf);
    } else if (option == "--memory-limit-gib") {
      parsed = parse_double(value, config.memory_limit_gib);
    } else if (option == "--mode") {
      parsed = parse_mode(value, config.mode);
    } else if (option == "--wait-each-kernel") {
      parsed = parse_binary(value, config.wait_each_kernel);
    } else if (option == "--host-read-full") {
      parsed = parse_binary(value, config.host_read_full);
    } else if (option == "--verify") {
      parsed = parse_binary(value, config.verify);
    } else {
      error = "unknown option " + option;
      return false;
    }
    if (!parsed) {
      error = "invalid value '" + value + "' for " + option;
      return false;
    }
  }
  return true;
}

bool checked_mul(std::size_t a, std::size_t b, std::size_t &result) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    return false;
  }
  result = a * b;
  return true;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t &result) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    return false;
  }
  result = a + b;
  return true;
}

std::size_t align_down(std::size_t value, std::size_t alignment) {
  return value - value % alignment;
}

bool derive_specs(const Config &config, std::vector<SystemSpec> &specs,
                  std::size_t &effective_systems, std::size_t &total_bytes,
                  std::string &error) {
  if (config.nx < 8 || config.ny < 8 || config.nz < 8 || config.bands < 4 ||
      config.bands % 4 != 0 || config.cheb_degree < 2 ||
      config.memory_limit_gib < 0.0) {
    error = "grid extents must be >=8, bands divisible by 4, degree >=2";
    return false;
  }
  effective_systems =
      config.mode == Mode::SingleLarge ? std::size_t{1} : config.systems;
  const std::size_t class_count =
      config.mode == Mode::Mixed ? config.classes : std::size_t{1};
  if (class_count == 0 || class_count > effective_systems) {
    error = "classes must be in [1, systems]";
    return false;
  }
  if (class_count > 1 && config.coarsen_percent == 0) {
    error = "mixed multi-class mode requires non-zero coarsen-percent";
    return false;
  }
  if (class_count > 1 &&
      config.coarsen_percent > 74 / (class_count - 1)) {
    error = "coarsest class would be below 25 percent";
    return false;
  }

  total_bytes = 0;
  for (std::size_t c = 0; c < class_count; ++c) {
    const std::size_t percent = 100 - c * config.coarsen_percent;
    const auto scaled = [percent](std::size_t value) {
      return align_down(std::max<std::size_t>(8, value * percent / 100), 4);
    };
    SystemSpec spec;
    spec.class_id = c;
    spec.systems = effective_systems / class_count +
                   (c < effective_systems % class_count ? 1 : 0);
    spec.nx = c == 0 ? config.nx : scaled(config.nx);
    spec.ny = c == 0 ? config.ny : scaled(config.ny);
    spec.nz = c == 0 ? config.nz : scaled(config.nz);
    spec.bands = config.bands;
    std::size_t plane = 0;
    std::size_t matrix = 0;
    std::size_t partial_values = 0;
    if (!checked_mul(spec.nx, spec.ny, plane) ||
        !checked_mul(plane, spec.nz, spec.grid) ||
        !checked_mul(spec.grid, spec.bands, spec.orbital_values) ||
        !checked_mul(spec.bands, spec.bands, matrix) ||
        !checked_mul(config.partials, matrix, partial_values)) {
      error = "element count overflow";
      return false;
    }
    // Five orbital blocks, two potentials, density, three band matrices and
    // one reusable partial Gram matrix.
    std::size_t values = 0;
    std::size_t term = 0;
    if (!checked_mul(std::size_t{5}, spec.orbital_values, values) ||
        !checked_mul(std::size_t{3}, spec.grid, term) ||
        !checked_add(values, term, values) ||
        !checked_mul(std::size_t{3}, matrix, term) ||
        !checked_add(values, term, values) ||
        !checked_add(values, partial_values, values) ||
        !checked_mul(values, sizeof(float), spec.bytes_per_system)) {
      error = "memory estimate overflow";
      return false;
    }
    std::size_t class_bytes = 0;
    std::size_t next_total = 0;
    if (!checked_mul(spec.bytes_per_system, spec.systems, class_bytes) ||
        !checked_add(total_bytes, class_bytes, next_total)) {
      error = "total memory estimate overflow";
      return false;
    }
    spec.potential_scale = 0.04f + 0.01f * static_cast<float>(c);
    specs.push_back(spec);
    total_bytes = next_total;
  }
  if (config.memory_limit_gib > 0.0 &&
      static_cast<long double>(total_bytes) >
          static_cast<long double>(config.memory_limit_gib) * 1024.0L *
              1024.0L * 1024.0L) {
    error = "estimated buffers exceed --memory-limit-gib";
    return false;
  }
  return true;
}

template <typename T> struct Buffer1D {
  sycl::buffer<T, 1> buffer;
  explicit Buffer1D(std::size_t n) : buffer(sycl::range<1>(n)) {
    buffer.set_write_back(false);
  }
  void initialize(const std::vector<T> &values) {
    auto out = buffer.get_host_access(sycl::write_only);
    for (std::size_t i = 0; i < values.size(); ++i) {
      out[i] = values[i];
    }
  }
};

std::vector<float> make_orbitals(const SystemSpec &spec,
                                 std::size_t system_id) {
  std::vector<float> values(spec.orbital_values);
  const float scale = std::sqrt(3.0f / static_cast<float>(spec.grid));
  for (std::size_t g = 0; g < spec.grid; ++g) {
    for (std::size_t b = 0; b < spec.bands; ++b) {
      const std::size_t hash =
          (g * 1103515245ULL + b * 2654435761ULL + system_id * 97ULL) & 1023;
      const float centered = static_cast<float>(hash) / 511.5f - 1.0f;
      values[g * spec.bands + b] = scale * centered;
    }
  }
  return values;
}

std::vector<float> make_potential(const SystemSpec &spec,
                                  std::size_t system_id) {
  std::vector<float> values(spec.grid);
  for (std::size_t g = 0; g < spec.grid; ++g) {
    const std::size_t x = g % spec.nx;
    const std::size_t y = (g / spec.nx) % spec.ny;
    const std::size_t z = g / (spec.nx * spec.ny);
    const float fx = static_cast<float>((x + 3 * system_id) % 17) / 16.0f;
    const float fy = static_cast<float>((y + 5 * system_id) % 19) / 18.0f;
    const float fz = static_cast<float>((z + 7 * system_id) % 23) / 22.0f;
    values[g] = spec.potential_scale * (fx + fy + fz - 1.5f);
  }
  return values;
}

struct SystemBuffers {
  std::size_t system_id;
  const SystemSpec *spec;
  Buffer1D<float> orbital0;
  Buffer1D<float> orbital1;
  Buffer1D<float> scratch0;
  Buffer1D<float> scratch1;
  Buffer1D<float> h_orbital;
  Buffer1D<float> potential0;
  Buffer1D<float> potential1;
  Buffer1D<float> density;
  Buffer1D<float> overlap;
  Buffer1D<float> projected_h;
  Buffer1D<float> transform;
  Buffer1D<float> partial_matrix;

  SystemBuffers(std::size_t id, const SystemSpec &s, std::size_t partials)
      : system_id(id), spec(&s), orbital0(s.orbital_values),
        orbital1(s.orbital_values), scratch0(s.orbital_values),
        scratch1(s.orbital_values), h_orbital(s.orbital_values),
        potential0(s.grid), potential1(s.grid), density(s.grid),
        overlap(s.bands * s.bands), projected_h(s.bands * s.bands),
        transform(s.bands * s.bands),
        partial_matrix(partials * s.bands * s.bands) {
    orbital0.initialize(make_orbitals(s, id));
    potential0.initialize(make_potential(s, id));
  }
};

Buffer1D<float> &orbital_state(SystemBuffers &system, std::size_t scf) {
  return scf % 2 == 0 ? system.orbital0 : system.orbital1;
}
Buffer1D<float> &orbital_next(SystemBuffers &system, std::size_t scf) {
  return scf % 2 == 0 ? system.orbital1 : system.orbital0;
}
Buffer1D<float> &potential_state(SystemBuffers &system, std::size_t scf) {
  return scf % 2 == 0 ? system.potential0 : system.potential1;
}
Buffer1D<float> &potential_next(SystemBuffers &system, std::size_t scf) {
  return scf % 2 == 0 ? system.potential1 : system.potential0;
}

void debug_wait(sycl::queue &queue, bool enabled) {
  if (enabled) {
    queue.wait();
  }
}

template <typename OrbitalAccessor, typename PotentialAccessor>
inline float apply_scaled_hamiltonian(const OrbitalAccessor &orbital,
                                      const PotentialAccessor &potential,
                                      std::size_t g, std::size_t band,
                                      std::size_t nx, std::size_t ny,
                                      std::size_t nz, std::size_t bands) {
  const std::size_t x = g % nx;
  const std::size_t y = (g / nx) % ny;
  const std::size_t z = g / (nx * ny);
  const std::size_t xm = x == 0 ? nx - 1 : x - 1;
  const std::size_t xp = x + 1 == nx ? 0 : x + 1;
  const std::size_t ym = y == 0 ? ny - 1 : y - 1;
  const std::size_t yp = y + 1 == ny ? 0 : y + 1;
  const std::size_t zm = z == 0 ? nz - 1 : z - 1;
  const std::size_t zp = z + 1 == nz ? 0 : z + 1;
  const auto at = [=](std::size_t zz, std::size_t yy, std::size_t xx) {
    return orbital[((zz * ny + yy) * nx + xx) * bands + band];
  };
  const float center = orbital[g * bands + band];
  const float neighbors = at(z, y, xm) + at(z, y, xp) + at(z, ym, x) +
                          at(z, yp, x) + at(zm, y, x) + at(zp, y, x);
  return 0.52f * center - (1.0f / 12.0f) * neighbors +
         0.08f * potential[g] * center;
}

void submit_cheb_first(sycl::queue &queue, SystemBuffers &system,
                       std::size_t scf, bool wait_each) {
  const SystemSpec &s = *system.spec;
  queue.submit([&](sycl::handler &cgh) {
    auto in = orbital_state(system, scf)
                  .buffer.template get_access<sycl::access::mode::read>(cgh);
    auto potential = potential_state(system, scf)
                         .buffer.template get_access<sycl::access::mode::read>(
                             cgh);
    auto out = system.scratch0.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftChebFirstKernel>(
        sycl::range<1>(s.orbital_values), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t g = linear / s.bands;
          const std::size_t b = linear - g * s.bands;
          out[linear] = apply_scaled_hamiltonian(
              in, potential, g, b, s.nx, s.ny, s.nz, s.bands);
        });
  });
  debug_wait(queue, wait_each);
}

void submit_cheb_recurrence(sycl::queue &queue, Buffer1D<float> &previous,
                            Buffer1D<float> &current, Buffer1D<float> &next,
                            Buffer1D<float> &potential, const SystemSpec &s,
                            bool wait_each) {
  queue.submit([&](sycl::handler &cgh) {
    auto prev = previous.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto cur = current.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto pot = potential.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto out = next.buffer.template get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftChebRecurrenceKernel>(
        sycl::range<1>(s.orbital_values), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t g = linear / s.bands;
          const std::size_t b = linear - g * s.bands;
          out[linear] =
              2.0f * apply_scaled_hamiltonian(cur, pot, g, b, s.nx, s.ny,
                                               s.nz, s.bands) -
              prev[linear];
        });
  });
  debug_wait(queue, wait_each);
}

void submit_overlap_partial(sycl::queue &queue, Buffer1D<float> &filtered,
                            SystemBuffers &system, std::size_t partials,
                            bool wait_each) {
  const SystemSpec &s = *system.spec;
  const std::size_t matrix = s.bands * s.bands;
  queue.submit([&](sycl::handler &cgh) {
    auto psi = filtered.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto partial = system.partial_matrix.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftOverlapPartialKernel>(
        sycl::range<1>(partials * matrix), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t chunk = linear / matrix;
          const std::size_t pair = linear - chunk * matrix;
          const std::size_t row = pair / s.bands;
          const std::size_t col = pair - row * s.bands;
          const std::size_t begin = s.grid * chunk / partials;
          const std::size_t end = s.grid * (chunk + 1) / partials;
          float sum = 0.0f;
          for (std::size_t g = begin; g < end; ++g) {
            sum += psi[g * s.bands + row] * psi[g * s.bands + col];
          }
          partial[linear] = sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_overlap_reduce(sycl::queue &queue, SystemBuffers &system,
                           std::size_t partials, bool wait_each) {
  const SystemSpec &s = *system.spec;
  const std::size_t matrix = s.bands * s.bands;
  queue.submit([&](sycl::handler &cgh) {
    auto partial = system.partial_matrix.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto overlap = system.overlap.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftOverlapReduceKernel>(
        sycl::range<1>(matrix), [=](sycl::item<1> item) {
          const std::size_t pair = item[0];
          float sum = 0.0f;
          for (std::size_t chunk = 0; chunk < partials; ++chunk) {
            sum += partial[chunk * matrix + pair];
          }
          overlap[pair] = sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_hamiltonian(sycl::queue &queue, Buffer1D<float> &filtered,
                        Buffer1D<float> &potential, SystemBuffers &system,
                        bool wait_each) {
  const SystemSpec &s = *system.spec;
  queue.submit([&](sycl::handler &cgh) {
    auto psi = filtered.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto pot = potential.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto hpsi = system.h_orbital.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftHamiltonianApplyKernel>(
        sycl::range<1>(s.orbital_values), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t g = linear / s.bands;
          const std::size_t b = linear - g * s.bands;
          hpsi[linear] = apply_scaled_hamiltonian(
              psi, pot, g, b, s.nx, s.ny, s.nz, s.bands);
        });
  });
  debug_wait(queue, wait_each);
}

void submit_projected_partial(sycl::queue &queue, Buffer1D<float> &filtered,
                              SystemBuffers &system, std::size_t partials,
                              bool wait_each) {
  const SystemSpec &s = *system.spec;
  const std::size_t matrix = s.bands * s.bands;
  queue.submit([&](sycl::handler &cgh) {
    auto psi = filtered.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto hpsi = system.h_orbital.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto partial = system.partial_matrix.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftProjectedPartialKernel>(
        sycl::range<1>(partials * matrix), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t chunk = linear / matrix;
          const std::size_t pair = linear - chunk * matrix;
          const std::size_t row = pair / s.bands;
          const std::size_t col = pair - row * s.bands;
          const std::size_t begin = s.grid * chunk / partials;
          const std::size_t end = s.grid * (chunk + 1) / partials;
          float sum = 0.0f;
          for (std::size_t g = begin; g < end; ++g) {
            sum += psi[g * s.bands + row] * hpsi[g * s.bands + col];
          }
          partial[linear] = sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_projected_reduce(sycl::queue &queue, SystemBuffers &system,
                             std::size_t partials, bool wait_each) {
  const SystemSpec &s = *system.spec;
  const std::size_t matrix = s.bands * s.bands;
  queue.submit([&](sycl::handler &cgh) {
    auto partial = system.partial_matrix.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto projected = system.projected_h.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftProjectedReduceKernel>(
        sycl::range<1>(matrix), [=](sycl::item<1> item) {
          const std::size_t pair = item[0];
          float sum = 0.0f;
          for (std::size_t chunk = 0; chunk < partials; ++chunk) {
            sum += partial[chunk * matrix + pair];
          }
          projected[pair] = sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_transform(sycl::queue &queue, SystemBuffers &system,
                      bool wait_each) {
  const SystemSpec &s = *system.spec;
  const std::size_t matrix = s.bands * s.bands;
  queue.submit([&](sycl::handler &cgh) {
    auto overlap = system.overlap.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto projected = system.projected_h.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto transform = system.transform.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftSubspaceTransformKernel>(
        sycl::range<1>(matrix), [=](sycl::item<1> item) {
          const std::size_t pair = item[0];
          const std::size_t row = pair / s.bands;
          const std::size_t col = pair - row * s.bands;
          const float sii = sycl::fmax(overlap[row * s.bands + row], 1.0e-8f);
          const float sjj = sycl::fmax(overlap[col * s.bands + col], 1.0e-8f);
          const float inv_row_norm = 1.0f / sycl::sqrt(sii);
          const float normalized_overlap =
              overlap[pair] / sycl::sqrt(sii * sjj);
          const float normalized_projected =
              projected[pair] / sycl::sqrt(sii * sjj);
          // A damped first-order inverse-square-root/Rayleigh-Ritz step. The
          // 1/sqrt(bands) factor keeps the accumulated off-diagonal rotation
          // bounded as the subspace width grows.
          const float coupling_scale =
              1.0f / sycl::sqrt(static_cast<float>(s.bands));
          transform[pair] =
              row == col
                  ? inv_row_norm
                  : inv_row_norm * coupling_scale *
                        (-0.10f * normalized_overlap -
                         0.02f * normalized_projected);
        });
  });
  debug_wait(queue, wait_each);
}

void submit_rotate(sycl::queue &queue, Buffer1D<float> &filtered,
                   SystemBuffers &system, std::size_t scf, bool wait_each) {
  const SystemSpec &s = *system.spec;
  queue.submit([&](sycl::handler &cgh) {
    auto psi = filtered.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto transform = system.transform.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto out = orbital_next(system, scf)
                   .buffer.template get_access<
                       sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftSubspaceRotateKernel>(
        sycl::range<1>(s.orbital_values), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t g = linear / s.bands;
          const std::size_t col = linear - g * s.bands;
          float sum = 0.0f;
          for (std::size_t row = 0; row < s.bands; ++row) {
            sum += psi[g * s.bands + row] *
                   transform[row * s.bands + col];
          }
          out[linear] = sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_density(sycl::queue &queue, SystemBuffers &system,
                    std::size_t scf, bool wait_each) {
  const SystemSpec &s = *system.spec;
  queue.submit([&](sycl::handler &cgh) {
    auto psi = orbital_next(system, scf)
                   .buffer.template get_access<sycl::access::mode::read>(cgh);
    auto density = system.density.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftDensityKernel>(
        sycl::range<1>(s.grid), [=](sycl::item<1> item) {
          const std::size_t g = item[0];
          float rho = 0.0f;
          for (std::size_t b = 0; b < s.bands; ++b) {
            const float value = psi[g * s.bands + b];
            rho += value * value;
          }
          density[g] = rho;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_potential_mix(sycl::queue &queue, SystemBuffers &system,
                          std::size_t scf, bool wait_each) {
  const SystemSpec &s = *system.spec;
  const float density_scale =
      static_cast<float>(s.grid) / static_cast<float>(s.bands);
  queue.submit([&](sycl::handler &cgh) {
    auto old_potential = potential_state(system, scf)
                             .buffer.template get_access<
                                 sycl::access::mode::read>(cgh);
    auto density = system.density.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto new_potential = potential_next(system, scf)
                             .buffer.template get_access<
                                 sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<DftPotentialMixKernel>(
        sycl::range<1>(s.grid), [=](sycl::item<1> item) {
          const std::size_t g = item[0];
          const float target =
              0.08f * (density[g] * density_scale - 1.0f);
          new_potential[g] = 0.85f * old_potential[g] + 0.15f * target;
        });
  });
  debug_wait(queue, wait_each);
}

Buffer1D<float> &submit_filter(sycl::queue &queue, SystemBuffers &system,
                               std::size_t scf, std::size_t degree,
                               bool wait_each) {
  submit_cheb_first(queue, system, scf, wait_each);
  Buffer1D<float> *slots[3] = {&orbital_state(system, scf), &system.scratch0,
                               &system.scratch1};
  Buffer1D<float> &potential = potential_state(system, scf);
  for (std::size_t order = 2; order <= degree; ++order) {
    submit_cheb_recurrence(queue, *slots[(order - 2) % 3],
                           *slots[(order - 1) % 3], *slots[order % 3],
                           potential, *system.spec, wait_each);
  }
  return *slots[degree % 3];
}

void submit_scf_window(
    sycl::queue &queue,
    std::vector<std::unique_ptr<SystemBuffers>> &systems, std::size_t scf,
    const Config &config) {
  std::vector<Buffer1D<float> *> filtered;
  filtered.reserve(systems.size());
  for (auto &system : systems) {
    filtered.push_back(&submit_filter(queue, *system, scf,
                                     config.cheb_degree,
                                     config.wait_each_kernel));
  }
  for (std::size_t i = 0; i < systems.size(); ++i) {
    submit_overlap_partial(queue, *filtered[i], *systems[i], config.partials,
                           config.wait_each_kernel);
    submit_hamiltonian(queue, *filtered[i], potential_state(*systems[i], scf),
                       *systems[i], config.wait_each_kernel);
  }
  for (auto &system : systems) {
    submit_overlap_reduce(queue, *system, config.partials,
                          config.wait_each_kernel);
  }
  for (std::size_t i = 0; i < systems.size(); ++i) {
    submit_projected_partial(queue, *filtered[i], *systems[i], config.partials,
                             config.wait_each_kernel);
  }
  for (auto &system : systems) {
    submit_projected_reduce(queue, *system, config.partials,
                            config.wait_each_kernel);
    submit_transform(queue, *system, config.wait_each_kernel);
  }
  for (std::size_t i = 0; i < systems.size(); ++i) {
    submit_rotate(queue, *filtered[i], *systems[i], scf,
                  config.wait_each_kernel);
  }
  for (auto &system : systems) {
    submit_density(queue, *system, scf, config.wait_each_kernel);
    submit_potential_mix(queue, *system, scf, config.wait_each_kernel);
  }
}

struct Result {
  double checksum = 0.0;
  double sample_density = 0.0;
  double sample_orbital = 0.0;
  bool valid = true;
};

Result read_result(std::vector<std::unique_ptr<SystemBuffers>> &systems,
                   const Config &config) {
  Result result;
  for (auto &pointer : systems) {
    SystemBuffers &system = *pointer;
    const SystemSpec &s = *system.spec;
    auto psi = orbital_state(system, config.scf_cycles)
                   .buffer.get_host_access(sycl::read_only);
    auto density = system.density.buffer.get_host_access(sycl::read_only);
    const std::size_t orbital_samples =
        config.host_read_full ? s.orbital_values : std::size_t{8};
    for (std::size_t sample = 0; sample < orbital_samples; ++sample) {
      const std::size_t index = config.host_read_full
                                    ? sample
                                    : sample * (s.orbital_values - 1) / 7;
      const float value = psi[index];
      result.valid = result.valid && std::isfinite(value);
      result.checksum += (1.0 + 0.01 * system.system_id) * value;
    }
    for (std::size_t g = 0; g < s.grid; ++g) {
      const float rho = density[g];
      result.valid = result.valid && std::isfinite(rho) && rho >= 0.0f;
      result.checksum += 1.0e-3 * rho;
    }
    if (system.system_id == 0) {
      result.sample_density = density[s.grid / 2];
      result.sample_orbital = psi[(s.grid / 2) * s.bands];
    }
  }
  result.valid = result.valid && std::isfinite(result.checksum);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  Config config;
  std::string error;
  bool help = false;
  if (!parse_arguments(argc, argv, config, error, help)) {
    std::cerr << "ERROR " << error << '\n';
    print_usage(argv[0]);
    return 1;
  }
  if (help) {
    print_usage(argv[0]);
    return 0;
  }

  std::vector<SystemSpec> specs;
  std::size_t effective_systems = 0;
  std::size_t total_bytes = 0;
  if (!derive_specs(config, specs, effective_systems, total_bytes, error)) {
    std::cerr << "ERROR " << error << '\n';
    return 1;
  }
  std::size_t kernels_per_system = 0;
  std::size_t kernels_per_scf = 0;
  std::size_t total_kernels = 0;
  std::size_t max_width = 0;
  std::size_t critical_path_levels = 0;
  if (!checked_add(config.cheb_degree, std::size_t{9}, kernels_per_system) ||
      !checked_add(config.cheb_degree, std::size_t{7},
                   critical_path_levels) ||
      !checked_mul(kernels_per_system, effective_systems, kernels_per_scf) ||
      !checked_mul(kernels_per_scf, config.scf_cycles, total_kernels) ||
      !checked_mul(std::size_t{2}, effective_systems, max_width)) {
    std::cerr << "ERROR kernel-count overflow\n";
    return 1;
  }

  std::cout << std::setprecision(9);
  std::cout << "CONFIG nx=" << config.nx << " ny=" << config.ny
            << " nz=" << config.nz << " bands=" << config.bands
            << " systems=" << effective_systems
            << " classes=" << specs.size()
            << " scf_cycles=" << config.scf_cycles
            << " cheb_degree=" << config.cheb_degree
            << " partials=" << config.partials
            << " mode=" << mode_name(config.mode)
            << " window_scf=" << config.window_scf
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << '\n';
  std::cout << "DAG kernels_per_system=" << kernels_per_system
            << " kernels_per_scf=" << kernels_per_scf
            << " max_width=" << max_width
            << " critical_path_levels=" << critical_path_levels
            << " total_kernels=" << total_kernels << '\n';
  for (const SystemSpec &s : specs) {
    std::cout << "CLASS id=" << s.class_id << " systems=" << s.systems
              << " nx=" << s.nx << " ny=" << s.ny << " nz=" << s.nz
              << " grid=" << s.grid << " bands=" << s.bands
              << " bytes_per_system=" << s.bytes_per_system << '\n';
  }
  std::cout << "MEMORY total_estimated_bytes=" << total_bytes << '\n';

  try {
    const double total_begin = get_time();
    std::vector<std::unique_ptr<SystemBuffers>> systems;
    systems.reserve(effective_systems);
    std::size_t system_id = 0;
    for (const SystemSpec &s : specs) {
      for (std::size_t local = 0; local < s.systems; ++local) {
        systems.push_back(
            std::make_unique<SystemBuffers>(system_id++, s, config.partials));
      }
    }
    const double init_end = get_time();
    sycl::queue queue;
    const double queue_end = get_time();

    const double run_begin = get_time();
    for (std::size_t scf = 0; scf < config.scf_cycles; ++scf) {
      submit_scf_window(queue, systems, scf, config);
      if ((scf + 1) % config.window_scf == 0 ||
          scf + 1 == config.scf_cycles) {
        queue.wait();
      }
    }
    const double run_end = get_time();
    const double host_begin = get_time();
    const Result result = read_result(systems, config);
    const double host_end = get_time();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_end - total_begin << '\n';
    std::cout << "TIMING queue_sec=" << queue_end - init_end << '\n';
    std::cout << "TIMING run_sec=" << run_end - run_begin << '\n';
    std::cout << "TIMING host_sec=" << host_end - host_begin << '\n';
    std::cout << "TIMING total_sec=" << host_end - total_begin << '\n';
    std::cout << "RESULT checksum=" << result.checksum
              << " sample_density=" << result.sample_density
              << " sample_orbital=" << result.sample_orbital << '\n';
    std::cout << "VERIFY passed=" << (result.valid ? 1 : 0) << '\n';
    return config.verify && !result.valid ? 2 : 0;
  } catch (const sycl::exception &exception) {
    std::cerr << "ERROR SYCL " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }
}
