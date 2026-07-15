// Multi-region PDE forecast + local ensemble transform Kalman filter (LETKF)
// miniapp. Each independent region advances a shallow-water-like ensemble
// through:
//
//   forecast stencil(s) -> ensemble mean/anomalies
//       -> {observation Gram, innovation RHS}
//       -> {ensemble transform, mean weights} -> analysis update
//
// The implementation is standalone SYCL and deliberately uses buffers and
// accessors so a runtime can infer the region-local DAG from data accesses.

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

class LetkfForecastKernel;
class LetkfMeanAnomalyKernel;
class LetkfObservationGramKernel;
class LetkfInnovationRhsKernel;
class LetkfTransformKernel;
class LetkfMeanWeightsKernel;
class LetkfAnalysisKernel;

namespace {

enum class Mode { Mixed, Uniform, SingleLarge };

struct State4 {
  float h;
  float u;
  float v;
  float tracer;
};

struct Config {
  std::size_t nx = 256;
  std::size_t ny = 256;
  std::size_t ensemble = 32;
  std::size_t regions = 8;
  std::size_t classes = 4;
  std::size_t coarsen_percent = 8;
  std::size_t cycles = 12;
  std::size_t forecast_steps = 4;
  std::size_t obs_stride = 4;
  std::size_t window_cycles = 1;
  double memory_limit_gib = 0.0;
  Mode mode = Mode::Mixed;
  bool wait_each_kernel = false;
  bool host_read_full = false;
  bool verify = true;
};

struct RegionSpec {
  std::size_t class_id = 0;
  std::size_t regions = 0;
  std::size_t nx = 0;
  std::size_t ny = 0;
  std::size_t grid = 0;
  std::size_t ensemble = 0;
  std::size_t state_values = 0;
  std::size_t observations = 0;
  std::size_t bytes_per_region = 0;
  float dt = 0.0f;
  float coriolis = 0.0f;
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
      << "  --nx/--ny <int>            Finest regional grid (default 256^2)\n"
      << "  --ensemble <int>           Ensemble members, multiple of 4 (default 32)\n"
      << "  --regions <int>            Independent forecast regions (default 8)\n"
      << "  --classes <int>            Mixed resolution classes (default 4)\n"
      << "  --coarsen-percent <int>    Grid reduction per class (default 8)\n"
      << "  --cycles <int>             Timed forecast/analysis cycles (default 12)\n"
      << "  --forecast-steps <int>     PDE steps before each analysis (default 4)\n"
      << "  --obs-stride <int>         Assimilate every Nth grid point (default 4)\n"
      << "  --window-cycles <int>      Cycles per queue.wait (default 1)\n"
      << "  --memory-limit-gib <real>  Fail above estimate; 0 is unlimited\n"
      << "  --mode <mixed|uniform|single-large>\n"
      << "  --wait-each-kernel <0|1>   Debug only (default 0)\n"
      << "  --host-read-full <0|1>     Check every final state (default 0)\n"
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
    } else if (option == "--ensemble") {
      parsed = parse_size(value, config.ensemble);
    } else if (option == "--regions") {
      parsed = parse_size(value, config.regions);
    } else if (option == "--classes") {
      parsed = parse_size(value, config.classes);
    } else if (option == "--coarsen-percent") {
      parsed = parse_size(value, config.coarsen_percent, true);
    } else if (option == "--cycles") {
      parsed = parse_size(value, config.cycles);
    } else if (option == "--forecast-steps") {
      parsed = parse_size(value, config.forecast_steps);
    } else if (option == "--obs-stride") {
      parsed = parse_size(value, config.obs_stride);
    } else if (option == "--window-cycles") {
      parsed = parse_size(value, config.window_cycles);
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

bool derive_specs(const Config &config, std::vector<RegionSpec> &specs,
                  std::size_t &effective_regions, std::size_t &total_bytes,
                  std::string &error) {
  if (config.nx < 16 || config.ny < 16 || config.ensemble < 4 ||
      config.ensemble % 4 != 0 || config.memory_limit_gib < 0.0) {
    error = "grid extents must be >=16 and ensemble must be divisible by 4";
    return false;
  }
  effective_regions =
      config.mode == Mode::SingleLarge ? std::size_t{1} : config.regions;
  const std::size_t class_count =
      config.mode == Mode::Mixed ? config.classes : std::size_t{1};
  if (class_count == 0 || class_count > effective_regions) {
    error = "classes must be in [1, regions]";
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
      return align_down(std::max<std::size_t>(16, value * percent / 100), 8);
    };
    RegionSpec spec;
    spec.class_id = c;
    spec.regions = effective_regions / class_count +
                   (c < effective_regions % class_count ? 1 : 0);
    spec.nx = c == 0 ? config.nx : scaled(config.nx);
    spec.ny = c == 0 ? config.ny : scaled(config.ny);
    spec.ensemble = config.ensemble;
    std::size_t matrix = 0;
    if (!checked_mul(spec.nx, spec.ny, spec.grid) ||
        !checked_mul(spec.grid, spec.ensemble, spec.state_values) ||
        !checked_mul(spec.ensemble, spec.ensemble, matrix)) {
      error = "element count overflow";
      return false;
    }
    spec.observations = spec.grid / config.obs_stride +
                        (spec.grid % config.obs_stride == 0 ? 0 : 1);

    // Two ensemble states, one anomaly ensemble, one mean field, scalar
    // observations, two E-vectors, and two E-by-E matrices.
    std::size_t bytes = 0;
    std::size_t term = 0;
    std::size_t matrix_values = 0;
    std::size_t vector_values = 0;
    std::size_t small_values = 0;
    if (!checked_mul(std::size_t{3}, spec.state_values, term) ||
        !checked_mul(term, sizeof(State4), bytes) ||
        !checked_mul(spec.grid, sizeof(State4), term) ||
        !checked_add(bytes, term, bytes) ||
        !checked_mul(spec.grid, sizeof(float), term) ||
        !checked_add(bytes, term, bytes) ||
        !checked_mul(std::size_t{2}, matrix, matrix_values) ||
        !checked_mul(std::size_t{2}, spec.ensemble, vector_values) ||
        !checked_add(matrix_values, vector_values, small_values) ||
        !checked_mul(small_values, sizeof(float), term) ||
        !checked_add(bytes, term, spec.bytes_per_region)) {
      error = "memory estimate overflow";
      return false;
    }
    std::size_t class_bytes = 0;
    std::size_t next_total = 0;
    if (!checked_mul(spec.bytes_per_region, spec.regions, class_bytes) ||
        !checked_add(total_bytes, class_bytes, next_total)) {
      error = "total memory estimate overflow";
      return false;
    }
    spec.dt = 0.006f * (1.0f - 0.04f * static_cast<float>(c));
    spec.coriolis = 0.04f + 0.006f * static_cast<float>(c);
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

std::vector<State4> make_initial_state(const RegionSpec &spec,
                                       std::size_t region_id) {
  std::vector<State4> values(spec.state_values);
  for (std::size_t g = 0; g < spec.grid; ++g) {
    const std::size_t x = g % spec.nx;
    const std::size_t y = g / spec.nx;
    for (std::size_t e = 0; e < spec.ensemble; ++e) {
      const std::size_t hash =
          (x * 73856093ULL + y * 19349663ULL + e * 83492791ULL +
           region_id * 2654435761ULL) &
          1023ULL;
      const float perturb = static_cast<float>(hash) / 1023.0f - 0.5f;
      const float member =
          (static_cast<float>(e) - 0.5f * static_cast<float>(spec.ensemble)) /
          static_cast<float>(spec.ensemble);
      values[g * spec.ensemble + e] =
          State4{1.0f + 0.018f * perturb + 0.004f * member,
                 0.012f * perturb, -0.010f * perturb,
                 0.45f + 0.025f * perturb + 0.005f * member};
    }
  }
  return values;
}

std::vector<float> make_observations(const RegionSpec &spec,
                                     std::size_t region_id) {
  std::vector<float> values(spec.grid);
  for (std::size_t g = 0; g < spec.grid; ++g) {
    const std::size_t x = g % spec.nx;
    const std::size_t y = g / spec.nx;
    const float wave =
        static_cast<float>((3 * x + 5 * y + 7 * region_id) % 29) / 28.0f -
        0.5f;
    values[g] = 1.0f + 0.012f * wave;
  }
  return values;
}

struct RegionBuffers {
  std::size_t region_id;
  const RegionSpec *spec;
  Buffer1D<State4> state0;
  Buffer1D<State4> state1;
  Buffer1D<State4> anomaly;
  Buffer1D<State4> mean;
  Buffer1D<float> observations;
  Buffer1D<float> gram;
  Buffer1D<float> rhs;
  Buffer1D<float> transform;
  Buffer1D<float> weights;

  RegionBuffers(std::size_t id, const RegionSpec &s)
      : region_id(id), spec(&s), state0(s.state_values),
        state1(s.state_values), anomaly(s.state_values), mean(s.grid),
        observations(s.grid), gram(s.ensemble * s.ensemble),
        rhs(s.ensemble), transform(s.ensemble * s.ensemble),
        weights(s.ensemble) {
    state0.initialize(make_initial_state(s, id));
    observations.initialize(make_observations(s, id));
  }
};

Buffer1D<State4> &state_slot(RegionBuffers &region, int slot) {
  return slot == 0 ? region.state0 : region.state1;
}

void debug_wait(sycl::queue &queue, bool enabled) {
  if (enabled) {
    queue.wait();
  }
}

template <typename Accessor>
inline State4 load_periodic(const Accessor &state, std::size_t x,
                            std::size_t y, std::size_t e, std::size_t nx,
                            std::size_t ensemble) {
  return state[(y * nx + x) * ensemble + e];
}

void submit_forecast(sycl::queue &queue, RegionBuffers &region, int input_slot,
                     bool wait_each) {
  const RegionSpec &s = *region.spec;
  const int output_slot = 1 - input_slot;
  const float dt = s.dt;
  const float coriolis = s.coriolis;
  queue.submit([&](sycl::handler &cgh) {
    auto in = state_slot(region, input_slot)
                  .buffer.template get_access<sycl::access::mode::read>(cgh);
    auto out = state_slot(region, output_slot)
                   .buffer.template get_access<
                       sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfForecastKernel>(
        sycl::range<1>(s.state_values), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t e = linear % s.ensemble;
          const std::size_t g = linear / s.ensemble;
          const std::size_t x = g % s.nx;
          const std::size_t y = g / s.nx;
          const std::size_t xm = x == 0 ? s.nx - 1 : x - 1;
          const std::size_t xp = x + 1 == s.nx ? 0 : x + 1;
          const std::size_t ym = y == 0 ? s.ny - 1 : y - 1;
          const std::size_t yp = y + 1 == s.ny ? 0 : y + 1;
          const State4 c = load_periodic(in, x, y, e, s.nx, s.ensemble);
          const State4 l = load_periodic(in, xm, y, e, s.nx, s.ensemble);
          const State4 r = load_periodic(in, xp, y, e, s.nx, s.ensemble);
          const State4 d = load_periodic(in, x, ym, e, s.nx, s.ensemble);
          const State4 u = load_periodic(in, x, yp, e, s.nx, s.ensemble);

          const float dhdx = 0.5f * (r.h - l.h);
          const float dhdy = 0.5f * (u.h - d.h);
          const float divergence = 0.5f * ((r.u - l.u) + (u.v - d.v));
          const float lap_h = l.h + r.h + d.h + u.h - 4.0f * c.h;
          const float lap_u = l.u + r.u + d.u + u.u - 4.0f * c.u;
          const float lap_v = l.v + r.v + d.v + u.v - 4.0f * c.v;
          const float lap_t = l.tracer + r.tracer + d.tracer + u.tracer -
                              4.0f * c.tracer;
          State4 next;
          next.h = c.h + dt * (-0.55f * divergence + 0.035f * lap_h);
          next.u = c.u + dt * (-0.75f * dhdx + coriolis * c.v -
                               0.012f * c.u + 0.025f * lap_u);
          next.v = c.v + dt * (-0.75f * dhdy - coriolis * c.u -
                               0.012f * c.v + 0.025f * lap_v);
          next.tracer = c.tracer + dt *
                          (-0.10f * (c.u * 0.5f * (r.tracer - l.tracer) +
                                    c.v * 0.5f * (u.tracer - d.tracer)) +
                           0.020f * lap_t);
          out[linear] = next;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_mean_anomaly(sycl::queue &queue, RegionBuffers &region,
                         int input_slot, bool wait_each) {
  const RegionSpec &s = *region.spec;
  queue.submit([&](sycl::handler &cgh) {
    auto state = state_slot(region, input_slot)
                     .buffer.template get_access<sycl::access::mode::read>(cgh);
    auto mean = region.mean.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    auto anomaly = region.anomaly.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfMeanAnomalyKernel>(
        sycl::range<1>(s.grid), [=](sycl::item<1> item) {
          const std::size_t g = item[0];
          State4 sum{0.0f, 0.0f, 0.0f, 0.0f};
          for (std::size_t e = 0; e < s.ensemble; ++e) {
            const State4 value = state[g * s.ensemble + e];
            sum.h += value.h;
            sum.u += value.u;
            sum.v += value.v;
            sum.tracer += value.tracer;
          }
          const float inv = 1.0f / static_cast<float>(s.ensemble);
          sum.h *= inv;
          sum.u *= inv;
          sum.v *= inv;
          sum.tracer *= inv;
          mean[g] = sum;
          for (std::size_t e = 0; e < s.ensemble; ++e) {
            const State4 value = state[g * s.ensemble + e];
            anomaly[g * s.ensemble + e] =
                State4{value.h - sum.h, value.u - sum.u, value.v - sum.v,
                       value.tracer - sum.tracer};
          }
        });
  });
  debug_wait(queue, wait_each);
}

void submit_observation_gram(sycl::queue &queue, RegionBuffers &region,
                             std::size_t obs_stride, bool wait_each) {
  const RegionSpec &s = *region.spec;
  const std::size_t matrix = s.ensemble * s.ensemble;
  const float scale = 18.0f /
                      (static_cast<float>(s.observations) *
                       static_cast<float>(s.ensemble - 1));
  queue.submit([&](sycl::handler &cgh) {
    auto anomaly = region.anomaly.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto gram = region.gram.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfObservationGramKernel>(
        sycl::range<1>(matrix), [=](sycl::item<1> item) {
          const std::size_t pair = item[0];
          const std::size_t row = pair / s.ensemble;
          const std::size_t col = pair - row * s.ensemble;
          float sum = 0.0f;
          for (std::size_t g = 0; g < s.grid; g += obs_stride) {
            sum += anomaly[g * s.ensemble + row].h *
                   anomaly[g * s.ensemble + col].h;
          }
          gram[pair] = (row == col ? 1.0f : 0.0f) + scale * sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_innovation_rhs(sycl::queue &queue, RegionBuffers &region,
                           std::size_t obs_stride, bool wait_each) {
  const RegionSpec &s = *region.spec;
  const float scale = 18.0f / static_cast<float>(s.observations);
  queue.submit([&](sycl::handler &cgh) {
    auto anomaly = region.anomaly.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto mean = region.mean.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto observation = region.observations.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto rhs = region.rhs.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfInnovationRhsKernel>(
        sycl::range<1>(s.ensemble), [=](sycl::item<1> item) {
          const std::size_t e = item[0];
          float sum = 0.0f;
          for (std::size_t g = 0; g < s.grid; g += obs_stride) {
            sum += anomaly[g * s.ensemble + e].h *
                   (observation[g] - mean[g].h);
          }
          rhs[e] = scale * sum;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_transform(sycl::queue &queue, RegionBuffers &region,
                      bool wait_each) {
  const RegionSpec &s = *region.spec;
  const std::size_t matrix = s.ensemble * s.ensemble;
  queue.submit([&](sycl::handler &cgh) {
    auto gram = region.gram.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto transform = region.transform.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfTransformKernel>(
        sycl::range<1>(matrix), [=](sycl::item<1> item) {
          const std::size_t pair = item[0];
          const std::size_t row = pair / s.ensemble;
          const std::size_t col = pair - row * s.ensemble;
          const float gii = sycl::fmax(gram[row * s.ensemble + row], 1.0e-6f);
          const float gjj = sycl::fmax(gram[col * s.ensemble + col], 1.0e-6f);
          transform[pair] =
              row == col
                  ? 1.0f / sycl::sqrt(gii)
                  : -0.22f * gram[pair] / sycl::sqrt(gii * gjj);
        });
  });
  debug_wait(queue, wait_each);
}

void submit_mean_weights(sycl::queue &queue, RegionBuffers &region,
                         bool wait_each) {
  const RegionSpec &s = *region.spec;
  queue.submit([&](sycl::handler &cgh) {
    auto gram = region.gram.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto rhs = region.rhs.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto weights = region.weights.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfMeanWeightsKernel>(
        sycl::range<1>(s.ensemble), [=](sycl::item<1> item) {
          const std::size_t e = item[0];
          const float diagonal =
              sycl::fmax(gram[e * s.ensemble + e], 1.0e-6f);
          weights[e] = 0.35f * rhs[e] / diagonal;
        });
  });
  debug_wait(queue, wait_each);
}

void submit_analysis(sycl::queue &queue, RegionBuffers &region,
                     int forecast_slot, bool wait_each) {
  const RegionSpec &s = *region.spec;
  const int output_slot = 1 - forecast_slot;
  queue.submit([&](sycl::handler &cgh) {
    auto mean = region.mean.buffer.template get_access<sycl::access::mode::read>(cgh);
    auto anomaly = region.anomaly.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto transform = region.transform.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto weights = region.weights.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto out = state_slot(region, output_slot)
                   .buffer.template get_access<
                       sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<LetkfAnalysisKernel>(
        sycl::range<1>(s.state_values), [=](sycl::item<1> item) {
          const std::size_t linear = item[0];
          const std::size_t e = linear % s.ensemble;
          const std::size_t g = linear / s.ensemble;
          State4 shift{0.0f, 0.0f, 0.0f, 0.0f};
          State4 spread{0.0f, 0.0f, 0.0f, 0.0f};
          for (std::size_t k = 0; k < s.ensemble; ++k) {
            const State4 a = anomaly[g * s.ensemble + k];
            const float w = weights[k];
            const float t = transform[k * s.ensemble + e];
            shift.h += a.h * w;
            shift.u += a.u * w;
            shift.v += a.v * w;
            shift.tracer += a.tracer * w;
            spread.h += a.h * t;
            spread.u += a.u * t;
            spread.v += a.v * t;
            spread.tracer += a.tracer * t;
          }
          const State4 m = mean[g];
          out[linear] = State4{m.h + shift.h + 1.015f * spread.h,
                               m.u + shift.u + 1.015f * spread.u,
                               m.v + shift.v + 1.015f * spread.v,
                               m.tracer + shift.tracer +
                                   1.015f * spread.tracer};
        });
  });
  debug_wait(queue, wait_each);
}

void submit_cycle(sycl::queue &queue,
                  std::vector<std::unique_ptr<RegionBuffers>> &regions,
                  std::vector<int> &slots, const Config &config) {
  // Submit one phase for every region before moving to the next phase. This
  // keeps the inferred DAG wide without removing any region-local dependency.
  for (std::size_t step = 0; step < config.forecast_steps; ++step) {
    for (std::size_t r = 0; r < regions.size(); ++r) {
      submit_forecast(queue, *regions[r], slots[r],
                      config.wait_each_kernel);
      slots[r] = 1 - slots[r];
    }
  }
  for (std::size_t r = 0; r < regions.size(); ++r) {
    submit_mean_anomaly(queue, *regions[r], slots[r],
                        config.wait_each_kernel);
  }
  for (auto &region : regions) {
    submit_observation_gram(queue, *region, config.obs_stride,
                            config.wait_each_kernel);
    submit_innovation_rhs(queue, *region, config.obs_stride,
                          config.wait_each_kernel);
  }
  for (auto &region : regions) {
    submit_transform(queue, *region, config.wait_each_kernel);
    submit_mean_weights(queue, *region, config.wait_each_kernel);
  }
  for (std::size_t r = 0; r < regions.size(); ++r) {
    submit_analysis(queue, *regions[r], slots[r], config.wait_each_kernel);
    slots[r] = 1 - slots[r];
  }
}

struct Result {
  double checksum = 0.0;
  double sample_h = 0.0;
  double sample_tracer = 0.0;
  double mean_h = 0.0;
  bool valid = true;
};

Result read_result(std::vector<std::unique_ptr<RegionBuffers>> &regions,
                   const std::vector<int> &slots, const Config &config) {
  Result result;
  double height_sum = 0.0;
  std::size_t height_count = 0;
  for (std::size_t r = 0; r < regions.size(); ++r) {
    RegionBuffers &region = *regions[r];
    const RegionSpec &s = *region.spec;
    auto state = state_slot(region, slots[r])
                     .buffer.get_host_access(sycl::read_only);
    const std::size_t samples =
        config.host_read_full ? s.state_values : std::size_t{16};
    for (std::size_t sample = 0; sample < samples; ++sample) {
      const std::size_t index = config.host_read_full
                                    ? sample
                                    : sample * (s.state_values - 1) / 15;
      const State4 value = state[index];
      result.valid = result.valid && std::isfinite(value.h) &&
                     std::isfinite(value.u) && std::isfinite(value.v) &&
                     std::isfinite(value.tracer) && value.h > 0.2f &&
                     value.h < 2.0f;
      result.checksum += (1.0 + 0.01 * region.region_id) *
                         (value.h + 0.2 * value.u + 0.3 * value.v +
                          0.5 * value.tracer);
      height_sum += value.h;
      ++height_count;
    }
    if (region.region_id == 0) {
      const State4 value = state[(s.grid / 2) * s.ensemble];
      result.sample_h = value.h;
      result.sample_tracer = value.tracer;
    }
  }
  result.mean_h = height_sum / static_cast<double>(height_count);
  result.valid = result.valid && std::isfinite(result.checksum) &&
                 result.mean_h > 0.8 && result.mean_h < 1.2;
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

  std::vector<RegionSpec> specs;
  std::size_t effective_regions = 0;
  std::size_t total_bytes = 0;
  if (!derive_specs(config, specs, effective_regions, total_bytes, error)) {
    std::cerr << "ERROR " << error << '\n';
    return 1;
  }
  std::size_t kernels_per_region = 0;
  std::size_t kernels_per_cycle = 0;
  std::size_t total_kernels = 0;
  std::size_t max_width = 0;
  std::size_t critical_path_levels = 0;
  if (!checked_add(config.forecast_steps, std::size_t{6},
                   kernels_per_region) ||
      !checked_add(config.forecast_steps, std::size_t{4},
                   critical_path_levels) ||
      !checked_mul(kernels_per_region, effective_regions, kernels_per_cycle) ||
      !checked_mul(kernels_per_cycle, config.cycles, total_kernels) ||
      !checked_mul(std::size_t{2}, effective_regions, max_width)) {
    std::cerr << "ERROR kernel-count overflow\n";
    return 1;
  }

  std::cout << std::setprecision(9);
  std::cout << "CONFIG nx=" << config.nx << " ny=" << config.ny
            << " ensemble=" << config.ensemble
            << " regions=" << effective_regions
            << " classes=" << specs.size() << " cycles=" << config.cycles
            << " forecast_steps=" << config.forecast_steps
            << " obs_stride=" << config.obs_stride
            << " mode=" << mode_name(config.mode)
            << " window_cycles=" << config.window_cycles
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << '\n';
  std::cout << "DAG kernels_per_region=" << kernels_per_region
            << " kernels_per_cycle=" << kernels_per_cycle
            << " max_width=" << max_width
            << " critical_path_levels=" << critical_path_levels
            << " total_kernels=" << total_kernels << '\n';
  for (const RegionSpec &s : specs) {
    std::cout << "CLASS id=" << s.class_id << " regions=" << s.regions
              << " nx=" << s.nx << " ny=" << s.ny
              << " grid=" << s.grid << " ensemble=" << s.ensemble
              << " observations=" << s.observations
              << " dt=" << s.dt
              << " bytes_per_region=" << s.bytes_per_region << '\n';
  }
  std::cout << "MEMORY total_estimated_bytes=" << total_bytes << '\n';

  try {
    const double total_begin = get_time();
    std::vector<std::unique_ptr<RegionBuffers>> regions;
    regions.reserve(effective_regions);
    std::size_t region_id = 0;
    for (const RegionSpec &s : specs) {
      for (std::size_t local = 0; local < s.regions; ++local) {
        regions.push_back(std::make_unique<RegionBuffers>(region_id++, s));
      }
    }
    std::vector<int> slots(effective_regions, 0);
    const double init_end = get_time();
    sycl::queue queue;
    const double queue_end = get_time();

    const double run_begin = get_time();
    for (std::size_t cycle = 0; cycle < config.cycles; ++cycle) {
      submit_cycle(queue, regions, slots, config);
      if ((cycle + 1) % config.window_cycles == 0 ||
          cycle + 1 == config.cycles) {
        queue.wait();
      }
    }
    const double run_end = get_time();
    const double host_begin = get_time();
    const Result result = read_result(regions, slots, config);
    const double host_end = get_time();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_end - total_begin << '\n';
    std::cout << "TIMING queue_sec=" << queue_end - init_end << '\n';
    std::cout << "TIMING run_sec=" << run_end - run_begin << '\n';
    std::cout << "TIMING host_sec=" << host_end - host_begin << '\n';
    std::cout << "TIMING total_sec=" << host_end - total_begin << '\n';
    std::cout << "RESULT checksum=" << result.checksum
              << " sample_h=" << result.sample_h
              << " sample_tracer=" << result.sample_tracer
              << " mean_h=" << result.mean_h << '\n';
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
