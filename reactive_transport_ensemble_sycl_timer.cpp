// Multi-fidelity ensemble reactive-transport SYCL miniapp.
//
// Each ensemble member advances three reacting species with Strang splitting:
//
//   state -> Robertson reaction(dt/2) -> {WENO5 flux-x, WENO5 flux-y}
//         -> transport + reaction(dt/2) -> next state
//
// Members are independent inside one time step.  A wait at the time-step
// boundary gives the offline runtime one repeated, semantically meaningful DAG
// window.  All kernels use full 2-D buffers, read-only inputs, discard-write
// outputs, and row-block writes that are safe for dim-0 splitting.  Versioned
// outputs are a design choice, not a runtime limitation: read_write splitting
// is also valid when writes are partition-disjoint and merge is restricted to
// each split part's owned row block.

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
#include <unistd.h>
#include <utility>
#include <vector>

#ifdef USE_DOUBLE
using data_t = double;
#else
using data_t = float;
#endif

class ReactiveTransportReactionKernel;
class ReactiveTransportWenoXKernel;
class ReactiveTransportWenoYKernel;
class ReactiveTransportFinishKernel;

namespace {

constexpr std::size_t kDefaultNx = 2048;
constexpr std::size_t kDefaultNy = 2048;
constexpr std::size_t kDefaultMembers = 6;
constexpr std::size_t kDefaultSteps = 10;
constexpr std::size_t kDefaultChemSubsteps = 16;
constexpr std::size_t kDefaultCoarsenPercent = 10;
constexpr std::size_t kMaxMembers = 8;
constexpr std::size_t kSplitAlignment = 4;
constexpr std::size_t kWenoRadius = 3;

using field_buffer_t = sycl::buffer<data_t, 2>;

enum class Mode { Full, ChemistryOnly, TransportOnly };

struct Config {
  std::size_t nx = kDefaultNx;
  std::size_t ny = kDefaultNy;
  std::size_t members = kDefaultMembers;
  std::size_t steps = kDefaultSteps;
  std::size_t chem_substeps = kDefaultChemSubsteps;
  std::size_t coarsen_percent = kDefaultCoarsenPercent;
  data_t cfl = static_cast<data_t>(0.32);
  Mode mode = Mode::Full;
  bool wait_each_kernel = false;
  bool verify = true;
  bool host_read_full = false;
};

struct MemberSpec {
  std::size_t index = 0;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t elements = 0;
  std::size_t chem_substeps = 0;
  data_t rate_scale = static_cast<data_t>(1);
  data_t dt = static_cast<data_t>(0);
};

double get_time() {
  timeval tv{};
  gettimeofday(&tv, nullptr);
  return static_cast<double>(tv.tv_sec) +
         static_cast<double>(tv.tv_usec) / 1.0e6;
}

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Full:
    return "full";
  case Mode::ChemistryOnly:
    return "chemistry-only";
  case Mode::TransportOnly:
    return "transport-only";
  }
  return "unknown";
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --nx <int>                 Finest-grid rows (default 2048)\n"
      << "  --ny <int>                 Finest-grid columns (default 2048)\n"
      << "  --members <int>            Ensemble members, 1..8 (default 6)\n"
      << "  --steps <int>              Macro time steps (default 10)\n"
      << "  --chem-substeps <int>      Base Rosenbrock substeps per half-step (default 16)\n"
      << "  --coarsen-percent <int>    Mesh reduction per member (default 10)\n"
      << "  --cfl <real>               2-D advection CFL in (0, 0.45] (default 0.32)\n"
      << "  --mode <full|chemistry-only|transport-only>\n"
      << "  --wait-each-kernel <0|1>   Debug-only; destroys the DAG window\n"
      << "  --verify <0|1>             Check finite, non-negative output (default 1)\n"
      << "  --host-read-full <0|1>     Full checksum instead of sparse samples\n";
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

bool parse_real(const std::string &text, data_t &value) {
  if (text.empty()) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  value = static_cast<data_t>(parsed);
  return true;
}

bool parse_binary(const std::string &text, bool &value) {
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
  } else if (text == "chemistry-only") {
    mode = Mode::ChemistryOnly;
  } else if (text == "transport-only") {
    mode = Mode::TransportOnly;
  } else {
    return false;
  }
  return true;
}

bool parse_arguments(int argc, char **argv, Config &config,
                     std::string &error, bool &show_help) {
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
    } else if (option == "--members") {
      parsed = parse_positive_size(value, config.members);
    } else if (option == "--steps") {
      parsed = parse_positive_size(value, config.steps);
    } else if (option == "--chem-substeps") {
      parsed = parse_positive_size(value, config.chem_substeps);
    } else if (option == "--coarsen-percent") {
      parsed = parse_positive_size(value, config.coarsen_percent);
    } else if (option == "--cfl") {
      parsed = parse_real(value, config.cfl);
    } else if (option == "--mode") {
      parsed = parse_mode(value, config.mode);
    } else if (option == "--wait-each-kernel") {
      parsed = parse_binary(value, config.wait_each_kernel);
    } else if (option == "--verify") {
      parsed = parse_binary(value, config.verify);
    } else if (option == "--host-read-full") {
      parsed = parse_binary(value, config.host_read_full);
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

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t &result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

std::size_t align_down(std::size_t value, std::size_t alignment) {
  return value - value % alignment;
}

std::size_t scale_percent(std::size_t value, std::size_t percent) {
  return (value / 100) * percent + (value % 100) * percent / 100;
}

bool derive_specs(const Config &config, std::vector<MemberSpec> &specs,
                  std::size_t &total_bytes, std::string &error) {
  const std::size_t minimum_extent = 2 * kWenoRadius + 2;
  if (config.nx < minimum_extent || config.ny < minimum_extent) {
    error = "nx and ny must be at least 8 for the WENO5 stencil";
    return false;
  }
  if (config.members == 0 || config.members > kMaxMembers) {
    error = "members must be in [1, 8]";
    return false;
  }
  if (config.members > 1 &&
      config.coarsen_percent > 79 / (config.members - 1)) {
    error = "coarsen-percent leaves the coarsest member below 20%";
    return false;
  }
  if (config.chem_substeps >
      (std::numeric_limits<std::size_t>::max() -
       2 * (config.members - 1)) /
          2) {
    error = "chem-substeps is too large";
    return false;
  }
  if (!(config.cfl > static_cast<data_t>(0) &&
        config.cfl <= static_cast<data_t>(0.45))) {
    error = "cfl must be in (0, 0.45]";
    return false;
  }

  total_bytes = 0;
  for (std::size_t member = 0; member < config.members; ++member) {
    const std::size_t percent = 100 - member * config.coarsen_percent;
    std::size_t rows =
        align_down(scale_percent(config.nx, percent), kSplitAlignment);
    std::size_t cols =
        align_down(scale_percent(config.ny, percent), kSplitAlignment);
    rows = std::max(rows, align_down(minimum_extent + kSplitAlignment - 1,
                                    kSplitAlignment));
    cols = std::max(cols, align_down(minimum_extent + kSplitAlignment - 1,
                                    kSplitAlignment));
    if (!specs.empty() && rows == specs.back().rows &&
        cols == specs.back().cols) {
      error = "member grids are not distinct; increase nx/ny or coarsen-percent";
      return false;
    }

    std::size_t elements = 0;
    std::size_t member_values = 0;
    std::size_t member_bytes = 0;
    if (!checked_multiply(rows, cols, elements) ||
        !checked_multiply(elements, std::size_t{17}, member_values) ||
        !checked_multiply(member_values, sizeof(data_t), member_bytes) ||
        member_bytes > std::numeric_limits<std::size_t>::max() - total_bytes) {
      error = "buffer size estimate overflows size_t";
      return false;
    }

    // Positive velocities are bounded by u<=0.72 and v<=0.52.  For a unit
    // square, dt = CFL / (|u|/dx + |v|/dy) is a conservative 2-D step.
    const data_t denom = static_cast<data_t>(0.72) *
                             static_cast<data_t>(cols) +
                         static_cast<data_t>(0.52) *
                             static_cast<data_t>(rows);
    MemberSpec spec;
    spec.index = member;
    spec.rows = rows;
    spec.cols = cols;
    spec.elements = elements;
    spec.chem_substeps = config.chem_substeps + 2 * member;
    spec.rate_scale =
        static_cast<data_t>(1) + static_cast<data_t>(0.12) * member;
    spec.dt = config.cfl / denom;
    specs.push_back(spec);
    total_bytes += member_bytes;
  }
  return true;
}

std::vector<data_t> make_constant_field(const MemberSpec &spec,
                                        data_t value) {
  return std::vector<data_t>(spec.elements, value);
}

std::vector<data_t> make_initial_species(const MemberSpec &spec,
                                         int species) {
  std::vector<data_t> values(spec.elements);
  constexpr double two_pi = 6.283185307179586476925286766559;
  const double phase = 0.37 * static_cast<double>(spec.index);
  for (std::size_t row = 0; row < spec.rows; ++row) {
    const double y = (static_cast<double>(row) + 0.5) /
                     static_cast<double>(spec.rows);
    for (std::size_t col = 0; col < spec.cols; ++col) {
      const double x = (static_cast<double>(col) + 0.5) /
                       static_cast<double>(spec.cols);
      const double dx = x - (0.30 + 0.025 * spec.index);
      const double dy = y - (0.46 - 0.018 * spec.index);
      const double pulse = std::exp(-70.0 * (dx * dx + dy * dy));
      const double wave =
          0.5 + 0.5 * std::sin(two_pi * x + phase) *
                          std::sin(two_pi * y - phase);
      const double b = 2.0e-5 + 5.0e-5 * pulse + 1.0e-5 * wave;
      const double c = 0.025 + 0.19 * pulse + 0.012 * wave;
      const double a = 1.0 - b - c;
      const std::size_t index = row * spec.cols + col;
      values[index] = static_cast<data_t>(species == 0 ? a
                                              : species == 1 ? b
                                                             : c);
    }
  }
  return values;
}

std::vector<data_t> make_velocity(const MemberSpec &spec, bool x_direction) {
  std::vector<data_t> values(spec.elements);
  constexpr double two_pi = 6.283185307179586476925286766559;
  const double phase = 0.21 * static_cast<double>(spec.index);
  for (std::size_t row = 0; row < spec.rows; ++row) {
    const double y = (static_cast<double>(row) + 0.5) /
                     static_cast<double>(spec.rows);
    for (std::size_t col = 0; col < spec.cols; ++col) {
      const double x = (static_cast<double>(col) + 0.5) /
                       static_cast<double>(spec.cols);
      const double velocity =
          x_direction
              ? 0.60 + 0.10 * std::sin(two_pi * y + phase) *
                           std::cos(two_pi * x)
              : 0.42 + 0.08 * std::cos(two_pi * x - phase) *
                           std::sin(two_pi * y);
      values[row * spec.cols + col] = static_cast<data_t>(velocity);
    }
  }
  return values;
}

struct Field {
  std::vector<data_t> host;
  field_buffer_t buffer;

  Field(std::vector<data_t> values, std::size_t rows, std::size_t cols)
      : host(std::move(values)),
        buffer(host.data(), sycl::range<2>(rows, cols)) {
    // Final values are obtained through host accessors.  Avoid copying every
    // scratch buffer to its initialization vector during destruction.
    buffer.set_write_back(false);
  }
};

enum class SpeciesInit { Initial, Zero };

struct SpeciesBuffers {
  Field a;
  Field b;
  Field c;

  SpeciesBuffers(const MemberSpec &spec, SpeciesInit init)
      : a(init == SpeciesInit::Initial ? make_initial_species(spec, 0)
                                       : make_constant_field(spec, 0),
          spec.rows, spec.cols),
        b(init == SpeciesInit::Initial ? make_initial_species(spec, 1)
                                       : make_constant_field(spec, 0),
          spec.rows, spec.cols),
        c(init == SpeciesInit::Initial ? make_initial_species(spec, 2)
                                       : make_constant_field(spec, 0),
          spec.rows, spec.cols) {}
};

struct MemberBuffers {
  MemberSpec spec;
  SpeciesBuffers state0;
  SpeciesBuffers state1;
  SpeciesBuffers chemistry;
  SpeciesBuffers flux_x;
  SpeciesBuffers flux_y;
  Field velocity_x;
  Field velocity_y;

  explicit MemberBuffers(const MemberSpec &member_spec)
      : spec(member_spec), state0(spec, SpeciesInit::Initial),
        state1(spec, SpeciesInit::Zero), chemistry(spec, SpeciesInit::Zero),
        flux_x(spec, SpeciesInit::Zero), flux_y(spec, SpeciesInit::Zero),
        velocity_x(make_velocity(spec, true), spec.rows, spec.cols),
        velocity_y(make_velocity(spec, false), spec.rows, spec.cols) {}
};

inline data_t positive_or_zero(data_t value) {
  return value > static_cast<data_t>(0) ? value : static_cast<data_t>(0);
}

inline data_t safe_denominator(data_t value) {
  const data_t epsilon = static_cast<data_t>(1.0e-20);
  if (value >= static_cast<data_t>(0) && value < epsilon) {
    return epsilon;
  }
  if (value < static_cast<data_t>(0) && value > -epsilon) {
    return -epsilon;
  }
  return value;
}

// One linearly implicit Rosenbrock-Euler step for the Robertson system.  The
// 3x3 solve is expanded into scalar Gaussian elimination to keep the kernel
// standalone and free of external dense-linear-algebra dependencies.
inline void rosenbrock_robertson_step(data_t &a, data_t &b, data_t &c,
                                      data_t h, data_t rate_scale) {
  const data_t k1 = static_cast<data_t>(0.04) * rate_scale;
  const data_t k2 = static_cast<data_t>(3.0e7) * rate_scale;
  const data_t k3 = static_cast<data_t>(1.0e4) * rate_scale;

  const data_t f0 = -k1 * a + k3 * b * c;
  const data_t f1 = k1 * a - k3 * b * c - k2 * b * b;
  const data_t f2 = k2 * b * b;

  data_t m00 = static_cast<data_t>(1) + h * k1;
  const data_t m01 = -h * k3 * c;
  const data_t m02 = -h * k3 * b;
  const data_t m10 = -h * k1;
  data_t m11 = static_cast<data_t>(1) +
               h * (k3 * c + static_cast<data_t>(2) * k2 * b);
  data_t m12 = h * k3 * b;
  data_t m21 = -h * static_cast<data_t>(2) * k2 * b;
  data_t m22 = static_cast<data_t>(1);
  data_t r0 = h * f0;
  data_t r1 = h * f1;
  data_t r2 = h * f2;

  m00 = safe_denominator(m00);
  const data_t factor10 = m10 / m00;
  m11 -= factor10 * m01;
  m12 -= factor10 * m02;
  r1 -= factor10 * r0;

  m11 = safe_denominator(m11);
  const data_t factor21 = m21 / m11;
  m22 -= factor21 * m12;
  r2 -= factor21 * r1;

  m22 = safe_denominator(m22);
  const data_t dc = r2 / m22;
  const data_t db = (r1 - m12 * dc) / m11;
  const data_t da = (r0 - m01 * db - m02 * dc) / m00;

  const data_t target_mass = a + b + c;
  a = positive_or_zero(a + da);
  b = positive_or_zero(b + db);
  c = positive_or_zero(c + dc);
  const data_t updated_mass = safe_denominator(a + b + c);
  const data_t scale = target_mass / updated_mass;
  a *= scale;
  b *= scale;
  c *= scale;
}

inline void integrate_robertson(data_t &a, data_t &b, data_t &c,
                                data_t interval, data_t rate_scale,
                                std::size_t substeps) {
  if (substeps == 0 || interval == static_cast<data_t>(0)) {
    return;
  }
  const data_t h = interval / static_cast<data_t>(substeps);
  for (std::size_t step = 0; step < substeps; ++step) {
    rosenbrock_robertson_step(a, b, c, h, rate_scale);
  }
}

inline data_t weno5_left(data_t fm2, data_t fm1, data_t f0, data_t fp1,
                         data_t fp2) {
  const data_t q0 = (static_cast<data_t>(2) * fm2 -
                     static_cast<data_t>(7) * fm1 +
                     static_cast<data_t>(11) * f0) /
                    static_cast<data_t>(6);
  const data_t q1 = (-fm1 + static_cast<data_t>(5) * f0 +
                     static_cast<data_t>(2) * fp1) /
                    static_cast<data_t>(6);
  const data_t q2 = (static_cast<data_t>(2) * f0 +
                     static_cast<data_t>(5) * fp1 - fp2) /
                    static_cast<data_t>(6);

  const data_t beta0 =
      static_cast<data_t>(13.0 / 12.0) *
          (fm2 - static_cast<data_t>(2) * fm1 + f0) *
          (fm2 - static_cast<data_t>(2) * fm1 + f0) +
      static_cast<data_t>(0.25) *
          (fm2 - static_cast<data_t>(4) * fm1 +
           static_cast<data_t>(3) * f0) *
          (fm2 - static_cast<data_t>(4) * fm1 +
           static_cast<data_t>(3) * f0);
  const data_t beta1 =
      static_cast<data_t>(13.0 / 12.0) *
          (fm1 - static_cast<data_t>(2) * f0 + fp1) *
          (fm1 - static_cast<data_t>(2) * f0 + fp1) +
      static_cast<data_t>(0.25) * (fm1 - fp1) * (fm1 - fp1);
  const data_t beta2 =
      static_cast<data_t>(13.0 / 12.0) *
          (f0 - static_cast<data_t>(2) * fp1 + fp2) *
          (f0 - static_cast<data_t>(2) * fp1 + fp2) +
      static_cast<data_t>(0.25) *
          (static_cast<data_t>(3) * f0 -
           static_cast<data_t>(4) * fp1 + fp2) *
          (static_cast<data_t>(3) * f0 -
           static_cast<data_t>(4) * fp1 + fp2);

  const data_t epsilon = static_cast<data_t>(1.0e-6);
  const data_t d0 = epsilon + beta0;
  const data_t d1 = epsilon + beta1;
  const data_t d2 = epsilon + beta2;
  const data_t alpha0 = static_cast<data_t>(0.1) / (d0 * d0);
  const data_t alpha1 = static_cast<data_t>(0.6) / (d1 * d1);
  const data_t alpha2 = static_cast<data_t>(0.3) / (d2 * d2);
  const data_t alpha_sum = alpha0 + alpha1 + alpha2;
  return (alpha0 * q0 + alpha1 * q1 + alpha2 * q2) / alpha_sum;
}

inline std::size_t clamp_minus(std::size_t index, std::size_t amount) {
  return index >= amount ? index - amount : 0;
}

inline std::size_t clamp_plus(std::size_t index, std::size_t amount,
                              std::size_t extent) {
  const std::size_t value = index + amount;
  return value < extent ? value : extent - 1;
}

template <typename FieldAccessor, typename VelocityAccessor>
inline data_t weno_divergence_x(const FieldAccessor &field,
                                const VelocityAccessor &velocity,
                                std::size_t row, std::size_t col,
                                std::size_t cols) {
  const std::size_t cm3 = clamp_minus(col, 3);
  const std::size_t cm2 = clamp_minus(col, 2);
  const std::size_t cm1 = clamp_minus(col, 1);
  const std::size_t cp1 = clamp_plus(col, 1, cols);
  const std::size_t cp2 = clamp_plus(col, 2, cols);
  const data_t fm3 = velocity[sycl::id<2>(row, cm3)] *
                     field[sycl::id<2>(row, cm3)];
  const data_t fm2 = velocity[sycl::id<2>(row, cm2)] *
                     field[sycl::id<2>(row, cm2)];
  const data_t fm1 = velocity[sycl::id<2>(row, cm1)] *
                     field[sycl::id<2>(row, cm1)];
  const data_t f0 = velocity[sycl::id<2>(row, col)] *
                    field[sycl::id<2>(row, col)];
  const data_t fp1 = velocity[sycl::id<2>(row, cp1)] *
                     field[sycl::id<2>(row, cp1)];
  const data_t fp2 = velocity[sycl::id<2>(row, cp2)] *
                     field[sycl::id<2>(row, cp2)];
  const data_t right = weno5_left(fm2, fm1, f0, fp1, fp2);
  const data_t left = weno5_left(fm3, fm2, fm1, f0, fp1);
  return (right - left) * static_cast<data_t>(cols);
}

template <typename FieldAccessor, typename VelocityAccessor>
inline data_t weno_divergence_y(const FieldAccessor &field,
                                const VelocityAccessor &velocity,
                                std::size_t row, std::size_t col,
                                std::size_t rows) {
  const std::size_t rm3 = clamp_minus(row, 3);
  const std::size_t rm2 = clamp_minus(row, 2);
  const std::size_t rm1 = clamp_minus(row, 1);
  const std::size_t rp1 = clamp_plus(row, 1, rows);
  const std::size_t rp2 = clamp_plus(row, 2, rows);
  const data_t fm3 = velocity[sycl::id<2>(rm3, col)] *
                     field[sycl::id<2>(rm3, col)];
  const data_t fm2 = velocity[sycl::id<2>(rm2, col)] *
                     field[sycl::id<2>(rm2, col)];
  const data_t fm1 = velocity[sycl::id<2>(rm1, col)] *
                     field[sycl::id<2>(rm1, col)];
  const data_t f0 = velocity[sycl::id<2>(row, col)] *
                    field[sycl::id<2>(row, col)];
  const data_t fp1 = velocity[sycl::id<2>(rp1, col)] *
                     field[sycl::id<2>(rp1, col)];
  const data_t fp2 = velocity[sycl::id<2>(rp2, col)] *
                     field[sycl::id<2>(rp2, col)];
  const data_t upper = weno5_left(fm2, fm1, f0, fp1, fp2);
  const data_t lower = weno5_left(fm3, fm2, fm1, f0, fp1);
  return (upper - lower) * static_cast<data_t>(rows);
}

void optional_debug_wait(sycl::queue &queue, bool wait_each_kernel) {
  if (wait_each_kernel) {
    queue.wait();
  }
}

void submit_reaction(sycl::queue &queue, SpeciesBuffers &input,
                     SpeciesBuffers &output, const MemberSpec &spec,
                     data_t interval, std::size_t substeps,
                     bool wait_each_kernel) {
  queue.submit([&](sycl::handler &cgh) {
    auto a_in = input.a.buffer.get_access<sycl::access::mode::read>(cgh);
    auto b_in = input.b.buffer.get_access<sycl::access::mode::read>(cgh);
    auto c_in = input.c.buffer.get_access<sycl::access::mode::read>(cgh);
    auto a_out = output.a.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto b_out = output.b.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto c_out = output.c.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<ReactiveTransportReactionKernel>(
        sycl::range<2>(spec.rows, spec.cols), [=](sycl::item<2> item) {
          const sycl::id<2> id(item[0], item[1]);
          data_t a = a_in[id];
          data_t b = b_in[id];
          data_t c = c_in[id];
          integrate_robertson(a, b, c, interval, spec.rate_scale, substeps);
          a_out[id] = a;
          b_out[id] = b;
          c_out[id] = c;
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_weno_x(sycl::queue &queue, SpeciesBuffers &input,
                   Field &velocity, SpeciesBuffers &output,
                   const MemberSpec &spec, bool wait_each_kernel) {
  queue.submit([&](sycl::handler &cgh) {
    auto a_in = input.a.buffer.get_access<sycl::access::mode::read>(cgh);
    auto b_in = input.b.buffer.get_access<sycl::access::mode::read>(cgh);
    auto c_in = input.c.buffer.get_access<sycl::access::mode::read>(cgh);
    auto u = velocity.buffer.get_access<sycl::access::mode::read>(cgh);
    auto a_out = output.a.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto b_out = output.b.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto c_out = output.c.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<ReactiveTransportWenoXKernel>(
        sycl::range<2>(spec.rows, spec.cols), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> id(row, col);
          a_out[id] = weno_divergence_x(a_in, u, row, col, spec.cols);
          b_out[id] = weno_divergence_x(b_in, u, row, col, spec.cols);
          c_out[id] = weno_divergence_x(c_in, u, row, col, spec.cols);
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_weno_y(sycl::queue &queue, SpeciesBuffers &input,
                   Field &velocity, SpeciesBuffers &output,
                   const MemberSpec &spec, bool wait_each_kernel) {
  queue.submit([&](sycl::handler &cgh) {
    auto a_in = input.a.buffer.get_access<sycl::access::mode::read>(cgh);
    auto b_in = input.b.buffer.get_access<sycl::access::mode::read>(cgh);
    auto c_in = input.c.buffer.get_access<sycl::access::mode::read>(cgh);
    auto v = velocity.buffer.get_access<sycl::access::mode::read>(cgh);
    auto a_out = output.a.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto b_out = output.b.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto c_out = output.c.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<ReactiveTransportWenoYKernel>(
        sycl::range<2>(spec.rows, spec.cols), [=](sycl::item<2> item) {
          const std::size_t row = item[0];
          const std::size_t col = item[1];
          const sycl::id<2> id(row, col);
          a_out[id] = weno_divergence_y(a_in, v, row, col, spec.rows);
          b_out[id] = weno_divergence_y(b_in, v, row, col, spec.rows);
          c_out[id] = weno_divergence_y(c_in, v, row, col, spec.rows);
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_finish(sycl::queue &queue, SpeciesBuffers &input,
                   SpeciesBuffers &flux_x, SpeciesBuffers &flux_y,
                   SpeciesBuffers &output, const MemberSpec &spec,
                   data_t reaction_interval, std::size_t reaction_substeps,
                   bool wait_each_kernel) {
  queue.submit([&](sycl::handler &cgh) {
    auto a_in = input.a.buffer.get_access<sycl::access::mode::read>(cgh);
    auto b_in = input.b.buffer.get_access<sycl::access::mode::read>(cgh);
    auto c_in = input.c.buffer.get_access<sycl::access::mode::read>(cgh);
    auto ax = flux_x.a.buffer.get_access<sycl::access::mode::read>(cgh);
    auto bx = flux_x.b.buffer.get_access<sycl::access::mode::read>(cgh);
    auto cx = flux_x.c.buffer.get_access<sycl::access::mode::read>(cgh);
    auto ay = flux_y.a.buffer.get_access<sycl::access::mode::read>(cgh);
    auto by = flux_y.b.buffer.get_access<sycl::access::mode::read>(cgh);
    auto cy = flux_y.c.buffer.get_access<sycl::access::mode::read>(cgh);
    auto a_out = output.a.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto b_out = output.b.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    auto c_out = output.c.buffer.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<ReactiveTransportFinishKernel>(
        sycl::range<2>(spec.rows, spec.cols), [=](sycl::item<2> item) {
          const sycl::id<2> id(item[0], item[1]);
          data_t a = positive_or_zero(a_in[id] - spec.dt * (ax[id] + ay[id]));
          data_t b = positive_or_zero(b_in[id] - spec.dt * (bx[id] + by[id]));
          data_t c = positive_or_zero(c_in[id] - spec.dt * (cx[id] + cy[id]));
          integrate_robertson(a, b, c, reaction_interval, spec.rate_scale,
                              reaction_substeps);
          a_out[id] = a;
          b_out[id] = b;
          c_out[id] = c;
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

SpeciesBuffers &input_state(MemberBuffers &member, std::size_t step) {
  return step % 2 == 0 ? member.state0 : member.state1;
}

SpeciesBuffers &output_state(MemberBuffers &member, std::size_t step) {
  return step % 2 == 0 ? member.state1 : member.state0;
}

SpeciesBuffers &final_state(MemberBuffers &member, std::size_t steps) {
  return steps % 2 == 0 ? member.state0 : member.state1;
}

void submit_full_step(sycl::queue &queue,
                      std::vector<std::unique_ptr<MemberBuffers>> &members,
                      std::size_t step, bool wait_each_kernel) {
  // Submit all roots first, then the two independent WENO branches, then all
  // fan-in consumers.  Dependencies are carried only by buffer accessors.
  for (auto &member_ptr : members) {
    MemberBuffers &member = *member_ptr;
    submit_reaction(queue, input_state(member, step), member.chemistry,
                    member.spec, member.spec.dt * static_cast<data_t>(0.5),
                    member.spec.chem_substeps, wait_each_kernel);
  }
  for (auto &member_ptr : members) {
    MemberBuffers &member = *member_ptr;
    submit_weno_x(queue, member.chemistry, member.velocity_x, member.flux_x,
                  member.spec, wait_each_kernel);
    submit_weno_y(queue, member.chemistry, member.velocity_y, member.flux_y,
                  member.spec, wait_each_kernel);
  }
  for (auto &member_ptr : members) {
    MemberBuffers &member = *member_ptr;
    submit_finish(queue, member.chemistry, member.flux_x, member.flux_y,
                  output_state(member, step), member.spec,
                  member.spec.dt * static_cast<data_t>(0.5),
                  member.spec.chem_substeps, wait_each_kernel);
  }
}

void submit_chemistry_step(
    sycl::queue &queue,
    std::vector<std::unique_ptr<MemberBuffers>> &members, std::size_t step,
    bool wait_each_kernel) {
  for (auto &member_ptr : members) {
    MemberBuffers &member = *member_ptr;
    submit_reaction(queue, input_state(member, step),
                    output_state(member, step), member.spec, member.spec.dt,
                    2 * member.spec.chem_substeps, wait_each_kernel);
  }
}

void submit_transport_step(
    sycl::queue &queue,
    std::vector<std::unique_ptr<MemberBuffers>> &members, std::size_t step,
    bool wait_each_kernel) {
  for (auto &member_ptr : members) {
    MemberBuffers &member = *member_ptr;
    SpeciesBuffers &input = input_state(member, step);
    submit_weno_x(queue, input, member.velocity_x, member.flux_x, member.spec,
                  wait_each_kernel);
    submit_weno_y(queue, input, member.velocity_y, member.flux_y, member.spec,
                  wait_each_kernel);
  }
  for (auto &member_ptr : members) {
    MemberBuffers &member = *member_ptr;
    SpeciesBuffers &input = input_state(member, step);
    submit_finish(queue, input, member.flux_x, member.flux_y,
                  output_state(member, step), member.spec,
                  static_cast<data_t>(0), 0, wait_each_kernel);
  }
}

struct ResultSummary {
  double checksum = 0.0;
  data_t sample_a = static_cast<data_t>(0);
  data_t sample_b = static_cast<data_t>(0);
  data_t sample_c = static_cast<data_t>(0);
  bool valid = true;
};

void accumulate_value(ResultSummary &result, data_t a, data_t b, data_t c,
                      double weight) {
  result.checksum += weight *
                     (static_cast<double>(a) +
                      3.0 * static_cast<double>(b) +
                      7.0 * static_cast<double>(c));
  result.valid = result.valid && std::isfinite(static_cast<double>(a)) &&
                 std::isfinite(static_cast<double>(b)) &&
                 std::isfinite(static_cast<double>(c)) &&
                 a >= static_cast<data_t>(0) && b >= static_cast<data_t>(0) &&
                 c >= static_cast<data_t>(0);
}

ResultSummary read_result(
    std::vector<std::unique_ptr<MemberBuffers>> &members,
    const Config &config) {
  ResultSummary result;
  for (std::size_t member_index = 0; member_index < members.size();
       ++member_index) {
    MemberBuffers &member = *members[member_index];
    SpeciesBuffers &state = final_state(member, config.steps);
    sycl::host_accessor a{state.a.buffer, sycl::read_only};
    sycl::host_accessor b{state.b.buffer, sycl::read_only};
    sycl::host_accessor c{state.c.buffer, sycl::read_only};

    if (config.host_read_full) {
      for (std::size_t row = 0; row < member.spec.rows; ++row) {
        for (std::size_t col = 0; col < member.spec.cols; ++col) {
          const sycl::id<2> id(row, col);
          accumulate_value(result, a[id], b[id], c[id], 1.0);
        }
      }
    } else {
      const sycl::id<2> ids[4] = {
          sycl::id<2>(0, 0),
          sycl::id<2>(member.spec.rows / 3, member.spec.cols / 3),
          sycl::id<2>(member.spec.rows / 2, member.spec.cols / 2),
          sycl::id<2>(member.spec.rows - 1, member.spec.cols - 1)};
      for (std::size_t sample = 0; sample < 4; ++sample) {
        const double weight = 1.0 + 0.125 * member_index + 0.03125 * sample;
        accumulate_value(result, a[ids[sample]], b[ids[sample]],
                         c[ids[sample]], weight);
      }
    }

    if (member_index == 0) {
      const sycl::id<2> middle(member.spec.rows / 2, member.spec.cols / 2);
      result.sample_a = a[middle];
      result.sample_b = b[middle];
      result.sample_c = c[middle];
    }
  }
  result.valid = result.valid && std::isfinite(result.checksum);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  Config config;
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

  std::vector<MemberSpec> specs;
  std::size_t total_bytes = 0;
  if (!derive_specs(config, specs, total_bytes, error)) {
    std::cerr << "ERROR " << error << '\n';
    return 1;
  }

  const std::size_t kernels_per_member = config.mode == Mode::Full
                                             ? 4
                                         : config.mode == Mode::ChemistryOnly
                                             ? 1
                                             : 3;
  std::cout << "CONFIG nx=" << config.nx << " ny=" << config.ny
            << " members=" << config.members << " steps=" << config.steps
            << " chem_substeps=" << config.chem_substeps
            << " coarsen_percent=" << config.coarsen_percent
            << " cfl=" << static_cast<double>(config.cfl)
            << " mode=" << mode_name(config.mode)
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " host_read_full=" << (config.host_read_full ? 1 : 0) << '\n';
  std::cout << "DAG kernels_per_step="
            << kernels_per_member * config.members
            << " max_width="
            << (config.mode == Mode::ChemistryOnly ? config.members
                                                    : 2 * config.members)
            << " total_kernels="
            << kernels_per_member * config.members * config.steps << '\n';
  for (const MemberSpec &spec : specs) {
    std::cout << "MEMBER id=" << spec.index << " rows=" << spec.rows
              << " cols=" << spec.cols << " elements=" << spec.elements
              << " chem_substeps=" << spec.chem_substeps
              << " rate_scale=" << static_cast<double>(spec.rate_scale)
              << " dt=" << static_cast<double>(spec.dt) << '\n';
  }
  std::cout << "MEMORY fields_per_member=17 total_estimated_bytes="
            << total_bytes << '\n';

  try {
    const double total_begin = get_time();
    std::vector<std::unique_ptr<MemberBuffers>> members;
    members.reserve(specs.size());
    for (const MemberSpec &spec : specs) {
      members.push_back(std::make_unique<MemberBuffers>(spec));
    }
    const double init_end = get_time();
    std::cout << getpid() << "======Init array " << init_end - total_begin
              << " seconds\n";

    sycl::queue queue;
    const double queue_end = get_time();
    std::cout << getpid() << "======Queue init " << queue_end - init_end
              << " seconds\n";

    const double run_begin = get_time();
    for (std::size_t step = 0; step < config.steps; ++step) {
      if (config.mode == Mode::Full) {
        submit_full_step(queue, members, step, config.wait_each_kernel);
      } else if (config.mode == Mode::ChemistryOnly) {
        submit_chemistry_step(queue, members, step, config.wait_each_kernel);
      } else {
        submit_transport_step(queue, members, step, config.wait_each_kernel);
      }

      // One complete physical time step is the repeated scheduling/profile
      // window.  No wait is inserted between kernels in the normal mode.
      queue.wait();
    }
    const double run_end = get_time();

    const double host_begin = get_time();
    const ResultSummary result = read_result(members, config);
    const double host_end = get_time();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_end - total_begin << '\n';
    std::cout << "TIMING queue_sec=" << queue_end - init_end << '\n';
    std::cout << "TIMING run_sec=" << run_end - run_begin << '\n';
    std::cout << "TIMING host_sec=" << host_end - host_begin << '\n';
    std::cout << "TIMING total_sec=" << host_end - total_begin << '\n';
    std::cout << "RESULT checksum=" << result.checksum
              << " sample_a=" << static_cast<double>(result.sample_a)
              << " sample_b=" << static_cast<double>(result.sample_b)
              << " sample_c=" << static_cast<double>(result.sample_c) << '\n';

    if (config.verify) {
      std::cout << "VERIFY passed=" << (result.valid ? 1 : 0) << '\n';
      return result.valid ? 0 : 2;
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
