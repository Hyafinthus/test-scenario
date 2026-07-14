// Multi-replica weakly-compressible SPH (WCSPH) SYCL miniapp.
//
// Each replica owns a complete particle system and advances this physical DAG:
//
//   Density -> EOS -> PressureForce --\
//       \--------> ViscosityForce ----> Integrate -> next position/velocity
//
// Replicas never share buffers. All replicas are submitted one DAG level at a
// time, and normal execution waits only at physical/window boundaries. This
// exposes whole-replica placement, producer affinity, and indirect-gather data
// residency without encoding scheduling decisions in the application.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/time.h>
#include <type_traits>
#include <utility>
#include <vector>

class SphDensityKernel;
class SphEosKernel;
class SphPressureForceKernel;
class SphViscosityForceKernel;
class SphIntegrateKernel;

namespace {

constexpr std::size_t kDefaultParticles = 131072;
constexpr std::size_t kDefaultReplicas = 64;
constexpr std::size_t kDefaultSteps = 30;
constexpr std::size_t kDefaultMaxNeighbors = 192;
constexpr std::size_t kDefaultClasses = 8;
constexpr std::size_t kDefaultCoarsenPercent = 4;
constexpr std::size_t kParticleAlignment = 256;
constexpr float kBoxLength = 1.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kWendlandAlpha = 21.0f / (16.0f * kPi);
constexpr float kGamma = 7.0f;
constexpr float kDamping = 0.9995f;
constexpr float kGravityX = 0.0f;
constexpr float kGravityY = -9.81f;
constexpr float kGravityZ = 0.0f;

struct Vec4 {
  float x;
  float y;
  float z;
  float w;
};

static_assert(sizeof(Vec4) == 16, "Vec4 must remain a 16-byte buffer element");
static_assert(std::is_trivially_copyable<Vec4>::value,
              "Vec4 must be trivially copyable");

enum class Mode { Mixed, Uniform, SingleLarge };

struct Config {
  std::size_t particles = kDefaultParticles;
  std::size_t replicas = kDefaultReplicas;
  std::size_t steps = kDefaultSteps;
  std::size_t max_neighbors = kDefaultMaxNeighbors;
  std::size_t classes = kDefaultClasses;
  std::size_t coarsen_percent = kDefaultCoarsenPercent;
  std::size_t window_steps = 1;
  std::size_t split_parts = 2;
  float dt = 2.0e-5f;
  float smoothing_length = 0.0f;
  float skin = 0.0f;
  double memory_limit_gib = 0.0;
  Mode mode = Mode::Mixed;
  bool wait_each_kernel = false;
  bool host_read_full = false;
  bool verify = true;
};

struct ClassSpec {
  std::size_t id = 0;
  std::size_t particles = 0;
  std::size_t replicas = 0;
  std::size_t bytes_per_replica = 0;
  float spacing = 0.0f;
  float smoothing_length = 0.0f;
  float skin = 0.0f;
  float rho0 = 0.0f;
  float viscosity = 0.0f;
  float sound_speed = 0.0f;
  float mass = 0.0f;
  float dt = 0.0f;
  double average_neighbors = 0.0;
};

struct ClassGeometry {
  std::vector<Vec4> initial_position;
  std::vector<std::uint32_t> neighbor_count;
  std::vector<std::uint32_t> neighbor_index;
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
      << "  --particles <int>          Finest/base particle count (default 131072)\n"
      << "  --replicas <int>           Replica count (default 64; single-large uses 1)\n"
      << "  --steps <int>              Physical time steps (default 30)\n"
      << "  --max-neighbors <int>      Fixed Verlet-row stride (default 192)\n"
      << "  --classes <int>            Mixed-mode fidelity classes (default 8)\n"
      << "  --coarsen-percent <int>    Particle reduction per class (default 4)\n"
      << "  --dt <real>                Integration time step (default 2e-5)\n"
      << "  --smoothing-length <real>  Finest-class h; 0 derives it from spacing\n"
      << "  --skin <real>              Finest-class Verlet skin; 0 derives it\n"
      << "  --window-steps <int>       Steps per queue.wait window (default 1)\n"
      << "  --split-parts <int>        Validate N divisibility for this part count\n"
      << "  --memory-limit-gib <real>  Fail above this estimate; 0 means unlimited\n"
      << "  --mode <mixed|uniform|single-large>\n"
      << "  --wait-each-kernel <0|1>   Debug-only synchronization (default 0)\n"
      << "  --host-read-full <0|1>     Materialize/check every final field (default 0)\n"
      << "  --verify <0|1>             Enable reference/basic verification (default 1)\n";
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

bool parse_float(const std::string &text, float &value) {
  if (text.empty()) {
    return false;
  }
  char *end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str(), &end);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      !std::isfinite(parsed) ||
      parsed > static_cast<double>(std::numeric_limits<float>::max()) ||
      parsed < -static_cast<double>(std::numeric_limits<float>::max())) {
    return false;
  }
  value = static_cast<float>(parsed);
  return true;
}

bool parse_double(const std::string &text, double &value) {
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
  value = parsed;
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
    if (option == "--particles") {
      parsed = parse_size(value, config.particles);
    } else if (option == "--replicas") {
      parsed = parse_size(value, config.replicas);
    } else if (option == "--steps") {
      parsed = parse_size(value, config.steps);
    } else if (option == "--max-neighbors") {
      parsed = parse_size(value, config.max_neighbors);
    } else if (option == "--classes") {
      parsed = parse_size(value, config.classes);
    } else if (option == "--coarsen-percent") {
      parsed = parse_size(value, config.coarsen_percent, true);
    } else if (option == "--dt") {
      parsed = parse_float(value, config.dt);
    } else if (option == "--smoothing-length") {
      parsed = parse_float(value, config.smoothing_length);
    } else if (option == "--skin") {
      parsed = parse_float(value, config.skin);
    } else if (option == "--window-steps") {
      parsed = parse_size(value, config.window_steps);
    } else if (option == "--split-parts") {
      parsed = parse_size(value, config.split_parts);
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

std::size_t align_down(std::size_t value, std::size_t alignment) {
  return value - value % alignment;
}

std::size_t scale_percent(std::size_t value, std::size_t percent) {
  return (value / 100) * percent + (value % 100) * percent / 100;
}

std::size_t lattice_side(std::size_t particles) {
  std::size_t side = static_cast<std::size_t>(
      std::ceil(std::cbrt(static_cast<long double>(particles))));
  side = std::max<std::size_t>(side, 1);
  // particles is constrained to uint32_t, so these cubes fit in size_t on all
  // supported 64-bit SYCL hosts.
  while (side > 1 &&
         (side - 1) * (side - 1) * (side - 1) >= particles) {
    --side;
  }
  while (side * side * side < particles) {
    ++side;
  }
  return side;
}

bool derive_specs(const Config &config, std::vector<ClassSpec> &specs,
                  std::size_t &effective_replicas,
                  std::size_t &total_estimated_bytes, std::string &error) {
  if (config.particles < kParticleAlignment ||
      config.particles % kParticleAlignment != 0) {
    error = "particles must be at least 256 and divisible by 256";
    return false;
  }
  if (config.particles > std::numeric_limits<std::uint32_t>::max()) {
    error = "particles must fit in uint32_t neighbor indices";
    return false;
  }
  if (config.max_neighbors == 0 ||
      config.max_neighbors > std::numeric_limits<std::uint32_t>::max()) {
    error = "max-neighbors must fit in uint32_t and be non-zero";
    return false;
  }
  if (config.split_parts == 0 || config.split_parts > config.particles) {
    error = "split-parts must be in [1, particles]";
    return false;
  }
  if (!(config.dt > 0.0f) || !(config.smoothing_length >= 0.0f) ||
      !(config.skin >= 0.0f) || config.memory_limit_gib < 0.0) {
    error = "dt must be positive; smoothing-length, skin, and memory limit must be non-negative";
    return false;
  }

  effective_replicas =
      config.mode == Mode::SingleLarge ? std::size_t{1} : config.replicas;
  const std::size_t effective_classes =
      config.mode == Mode::Mixed ? config.classes : std::size_t{1};
  if (effective_classes == 0 || effective_classes > effective_replicas) {
    error = "mixed mode requires classes in [1, replicas]";
    return false;
  }
  if (config.mode == Mode::Mixed && effective_classes > 1 &&
      config.coarsen_percent == 0) {
    error = "mixed mode with multiple classes requires non-zero coarsen-percent";
    return false;
  }
  if (effective_classes > 1 &&
      config.coarsen_percent > 79 / (effective_classes - 1)) {
    error = "coarsen-percent leaves the coarsest class below 20%";
    return false;
  }

  const std::size_t base_side = lattice_side(config.particles);
  const float base_spacing = kBoxLength / static_cast<float>(base_side);
  total_estimated_bytes = 0;
  specs.reserve(effective_classes);

  for (std::size_t class_id = 0; class_id < effective_classes; ++class_id) {
    const std::size_t percent = 100 - class_id * config.coarsen_percent;
    const std::size_t particles =
        class_id == 0
            ? config.particles
            : align_down(scale_percent(config.particles, percent),
                         kParticleAlignment);
    if (particles < kParticleAlignment || particles % config.split_parts != 0) {
      error = "every class particle count must be non-zero and divisible by split-parts";
      return false;
    }
    if (!specs.empty() && particles >= specs.back().particles) {
      error = "class particle counts are not distinct; increase particles or coarsen-percent";
      return false;
    }

    const std::size_t side = lattice_side(particles);
    const float spacing = kBoxLength / static_cast<float>(side);
    const float scale = spacing / base_spacing;
    const float class_factor = 1.0f + 0.01f * static_cast<float>(class_id);
    const float smoothing_length =
        config.smoothing_length > 0.0f
            ? config.smoothing_length * scale * class_factor
            : spacing * (1.15f + 0.02f * static_cast<float>(class_id));
    const float skin = config.skin > 0.0f ? config.skin * scale
                                          : spacing * 0.35f;
    if (!(smoothing_length > 0.0f) || !(skin > 0.0f) ||
        2.0f * smoothing_length + skin >= 0.5f * kBoxLength) {
      error = "each class requires h>0, skin>0, and 2*h+skin < box_length/2";
      return false;
    }

    std::size_t bytes_per_particle = 0;
    std::size_t neighbor_bytes = 0;
    std::size_t bytes_per_replica = 0;
    if (!checked_multiply(config.max_neighbors, sizeof(std::uint32_t),
                          neighbor_bytes) ||
        !checked_add(std::size_t{108}, neighbor_bytes, bytes_per_particle) ||
        !checked_multiply(particles, bytes_per_particle, bytes_per_replica)) {
      error = "per-replica memory estimate overflows size_t";
      return false;
    }

    const std::size_t class_replicas =
        effective_replicas / effective_classes +
        (class_id < effective_replicas % effective_classes ? 1 : 0);
    std::size_t class_bytes = 0;
    std::size_t next_total = 0;
    if (!checked_multiply(bytes_per_replica, class_replicas, class_bytes) ||
        !checked_add(total_estimated_bytes, class_bytes, next_total)) {
      error = "total memory estimate overflows size_t";
      return false;
    }

    ClassSpec spec;
    spec.id = class_id;
    spec.particles = particles;
    spec.replicas = class_replicas;
    spec.bytes_per_replica = bytes_per_replica;
    spec.spacing = spacing;
    spec.smoothing_length = smoothing_length;
    spec.skin = skin;
    spec.rho0 = 1000.0f *
                (1.0f + 0.01f * static_cast<float>(class_id));
    spec.viscosity =
        0.010f * (1.0f + 0.15f * static_cast<float>(class_id));
    spec.sound_speed =
        20.0f * (1.0f + 0.03f * static_cast<float>(class_id));
    spec.mass = spec.rho0 / static_cast<float>(particles);
    spec.dt = config.dt;
    specs.push_back(spec);
    total_estimated_bytes = next_total;
  }

  if (config.memory_limit_gib > 0.0) {
    const long double limit_bytes =
        static_cast<long double>(config.memory_limit_gib) * 1024.0L *
        1024.0L * 1024.0L;
    if (limit_bytes >
        static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
      error = "memory-limit-gib is too large to represent";
      return false;
    }
    if (static_cast<long double>(total_estimated_bytes) > limit_bytes) {
      error = "estimated buffers exceed --memory-limit-gib";
      return false;
    }
  }
  return true;
}

inline float minimum_image(float delta) {
  if (delta > 0.5f * kBoxLength) {
    delta -= kBoxLength;
  } else if (delta < -0.5f * kBoxLength) {
    delta += kBoxLength;
  }
  return delta;
}

float wrap_host(float value) {
  value -= std::floor(value / kBoxLength) * kBoxLength;
  if (value >= kBoxLength) {
    value -= kBoxLength;
  }
  if (value < 0.0f) {
    value += kBoxLength;
  }
  return value;
}

bool build_geometry(ClassSpec &spec, std::size_t max_neighbors,
                    ClassGeometry &geometry, std::string &error) {
  const std::size_t n = spec.particles;
  std::size_t neighbor_elements = 0;
  if (!checked_multiply(n, max_neighbors, neighbor_elements)) {
    error = "neighbor table element count overflows size_t";
    return false;
  }
  geometry.initial_position.resize(n);
  geometry.neighbor_count.assign(n, 0);
  geometry.neighbor_index.assign(neighbor_elements, 0);

  const std::size_t side = lattice_side(n);
  const float spacing = kBoxLength / static_cast<float>(side);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t ix = i % side;
    const std::size_t iy = (i / side) % side;
    const std::size_t iz = i / (side * side);
    const float phase = static_cast<float>(i % 4096);
    const float jitter = 0.025f * spacing;
    geometry.initial_position[i] = {
        wrap_host((static_cast<float>(ix) + 0.5f) * spacing +
                  jitter * std::sin(0.73f * phase + 0.11f * spec.id)),
        wrap_host((static_cast<float>(iy) + 0.5f) * spacing +
                  jitter * std::sin(0.51f * phase + 0.37f * spec.id)),
        wrap_host((static_cast<float>(iz) + 0.5f) * spacing +
                  jitter * std::sin(0.29f * phase + 0.67f * spec.id)),
        0.0f};
  }

  const float verlet_radius = 2.0f * spec.smoothing_length + spec.skin;
  const float verlet_radius2 = verlet_radius * verlet_radius;
  const std::size_t cells_per_dim = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::floor(kBoxLength / verlet_radius)));
  std::size_t cell_count = 0;
  std::size_t cells_squared = 0;
  if (!checked_multiply(cells_per_dim, cells_per_dim, cells_squared) ||
      !checked_multiply(cells_squared, cells_per_dim, cell_count)) {
    error = "cell-list dimensions overflow size_t";
    return false;
  }
  std::vector<std::vector<std::uint32_t>> cells(cell_count);
  auto cell_coordinate = [cells_per_dim](float coordinate) {
    return std::min(
        cells_per_dim - 1,
        static_cast<std::size_t>(coordinate / kBoxLength *
                                 static_cast<float>(cells_per_dim)));
  };
  auto cell_id = [cells_per_dim](std::size_t x, std::size_t y,
                                 std::size_t z) {
    return (z * cells_per_dim + y) * cells_per_dim + x;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const Vec4 p = geometry.initial_position[i];
    cells[cell_id(cell_coordinate(p.x), cell_coordinate(p.y),
                  cell_coordinate(p.z))]
        .push_back(static_cast<std::uint32_t>(i));
  }

  std::uint64_t total_neighbors = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Vec4 pi = geometry.initial_position[i];
    const std::size_t cx = cell_coordinate(pi.x);
    const std::size_t cy = cell_coordinate(pi.y);
    const std::size_t cz = cell_coordinate(pi.z);
    std::size_t nearby_cells[27]{};
    std::size_t nearby_count = 0;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const auto periodic = [cells_per_dim](std::size_t center, int off) {
            const long long extent = static_cast<long long>(cells_per_dim);
            long long value = static_cast<long long>(center) + off;
            value %= extent;
            if (value < 0) {
              value += extent;
            }
            return static_cast<std::size_t>(value);
          };
          const std::size_t candidate_cell =
              cell_id(periodic(cx, dx), periodic(cy, dy), periodic(cz, dz));
          bool duplicate = false;
          for (std::size_t c = 0; c < nearby_count; ++c) {
            duplicate = duplicate || nearby_cells[c] == candidate_cell;
          }
          if (!duplicate) {
            nearby_cells[nearby_count++] = candidate_cell;
          }
        }
      }
    }

    std::size_t count = 0;
    const std::size_t row_begin = i * max_neighbors;
    for (std::size_t c = 0; c < nearby_count; ++c) {
      for (const std::uint32_t j32 : cells[nearby_cells[c]]) {
        const std::size_t j = static_cast<std::size_t>(j32);
        if (j == i) {
          continue;
        }
        const Vec4 pj = geometry.initial_position[j];
        const float dx = minimum_image(pi.x - pj.x);
        const float dy = minimum_image(pi.y - pj.y);
        const float dz = minimum_image(pi.z - pj.z);
        const float distance2 = dx * dx + dy * dy + dz * dz;
        if (distance2 < verlet_radius2) {
          if (count == max_neighbors) {
            error = "class " + std::to_string(spec.id) + " particle " +
                    std::to_string(i) +
                    " exceeds max-neighbors; increase --max-neighbors";
            return false;
          }
          geometry.neighbor_index[row_begin + count] = j32;
          ++count;
        }
      }
    }
    std::sort(geometry.neighbor_index.begin() + row_begin,
              geometry.neighbor_index.begin() + row_begin + count);
    geometry.neighbor_count[i] = static_cast<std::uint32_t>(count);
    total_neighbors += count;
  }
  spec.average_neighbors = static_cast<double>(total_neighbors) /
                           static_cast<double>(n);
  return true;
}

std::vector<Vec4> make_initial_velocity(const ClassGeometry &geometry,
                                        std::size_t replica_index) {
  std::vector<Vec4> velocity(geometry.initial_position.size());
  const float phase = 0.173f * static_cast<float>(replica_index);
  for (std::size_t i = 0; i < velocity.size(); ++i) {
    const Vec4 p = geometry.initial_position[i];
    velocity[i] = {
        0.006f * std::sin(2.0f * kPi * p.y + phase),
        0.004f * std::cos(2.0f * kPi * p.z - phase),
        0.005f * std::sin(2.0f * kPi * p.x + 0.5f * phase), 0.0f};
  }
  return velocity;
}

template <typename T> struct Buffer1D {
  sycl::buffer<T, 1> buffer;

  explicit Buffer1D(std::size_t elements)
      : buffer(sycl::range<1>(elements)) {
    buffer.set_write_back(false);
  }

  void initialize(const std::vector<T> &values) {
    auto output = buffer.get_host_access(sycl::write_only);
    for (std::size_t i = 0; i < values.size(); ++i) {
      output[i] = values[i];
    }
  }
};

struct ReplicaBuffers {
  std::size_t replica_index;
  std::size_t class_id;
  const ClassSpec *spec;
  Buffer1D<Vec4> position0;
  Buffer1D<Vec4> position1;
  Buffer1D<Vec4> velocity0;
  Buffer1D<Vec4> velocity1;
  Buffer1D<float> density;
  Buffer1D<float> pressure;
  Buffer1D<Vec4> pressure_accel;
  Buffer1D<Vec4> viscosity_accel;
  Buffer1D<std::uint32_t> neighbor_count;
  Buffer1D<std::uint32_t> neighbor_index;

  ReplicaBuffers(std::size_t index, const ClassSpec &class_spec,
                 const ClassGeometry &geometry, std::size_t max_neighbors)
      : replica_index(index), class_id(class_spec.id), spec(&class_spec),
        position0(class_spec.particles), position1(class_spec.particles),
        velocity0(class_spec.particles), velocity1(class_spec.particles),
        density(class_spec.particles), pressure(class_spec.particles),
        pressure_accel(class_spec.particles),
        viscosity_accel(class_spec.particles),
        neighbor_count(class_spec.particles),
        neighbor_index(class_spec.particles * max_neighbors) {
    position0.initialize(geometry.initial_position);
    velocity0.initialize(make_initial_velocity(geometry, index));
    neighbor_count.initialize(geometry.neighbor_count);
    neighbor_index.initialize(geometry.neighbor_index);
  }
};

Buffer1D<Vec4> &current_position(ReplicaBuffers &replica, std::size_t step) {
  return step % 2 == 0 ? replica.position0 : replica.position1;
}

Buffer1D<Vec4> &next_position(ReplicaBuffers &replica, std::size_t step) {
  return step % 2 == 0 ? replica.position1 : replica.position0;
}

Buffer1D<Vec4> &current_velocity(ReplicaBuffers &replica, std::size_t step) {
  return step % 2 == 0 ? replica.velocity0 : replica.velocity1;
}

Buffer1D<Vec4> &next_velocity(ReplicaBuffers &replica, std::size_t step) {
  return step % 2 == 0 ? replica.velocity1 : replica.velocity0;
}

Buffer1D<Vec4> &final_position(ReplicaBuffers &replica, std::size_t steps) {
  return steps % 2 == 0 ? replica.position0 : replica.position1;
}

Buffer1D<Vec4> &final_velocity(ReplicaBuffers &replica, std::size_t steps) {
  return steps % 2 == 0 ? replica.velocity0 : replica.velocity1;
}

inline float wendland_weight(float distance, float h, float inv_h3) {
  const float q = distance / h;
  if (!(q < 2.0f)) {
    return 0.0f;
  }
  const float a = 1.0f - 0.5f * q;
  const float a2 = a * a;
  return kWendlandAlpha * inv_h3 * a2 * a2 * (2.0f * q + 1.0f);
}

void optional_debug_wait(sycl::queue &queue, bool wait_each_kernel) {
  if (wait_each_kernel) {
    queue.wait();
  }
}

void submit_density(sycl::queue &queue, ReplicaBuffers &replica,
                    std::size_t step, std::size_t max_neighbors,
                    bool wait_each_kernel) {
  const std::size_t n = replica.spec->particles;
  const float h = replica.spec->smoothing_length;
  const float h2 = h * h;
  const float inv_h = 1.0f / h;
  const float inv_h3 = inv_h * inv_h * inv_h;
  const float mass = replica.spec->mass;
  queue.submit([&](sycl::handler &cgh) {
    auto position = current_position(replica, step)
                        .buffer.template get_access<sycl::access::mode::read>(
                            cgh);
    auto count = replica.neighbor_count.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto index = replica.neighbor_index.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto density = replica.density.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<SphDensityKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t i = item[0];
          const Vec4 pi = position[i];
          float rho = mass * kWendlandAlpha * inv_h3;
          const std::size_t row = i * max_neighbors;
          const std::size_t neighbors = count[i];
          for (std::size_t entry = 0; entry < neighbors; ++entry) {
            const std::size_t j = index[row + entry];
            const Vec4 pj = position[j];
            const float dx = minimum_image(pi.x - pj.x);
            const float dy = minimum_image(pi.y - pj.y);
            const float dz = minimum_image(pi.z - pj.z);
            const float distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 < 4.0f * h2) {
              rho += mass *
                     wendland_weight(sycl::sqrt(distance2), h, inv_h3);
            }
          }
          density[i] = rho;
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_eos(sycl::queue &queue, ReplicaBuffers &replica,
                bool wait_each_kernel) {
  const std::size_t n = replica.spec->particles;
  const float rho0 = replica.spec->rho0;
  const float pressure_scale =
      rho0 * replica.spec->sound_speed * replica.spec->sound_speed / kGamma;
  queue.submit([&](sycl::handler &cgh) {
    auto density = replica.density.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto pressure = replica.pressure.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<SphEosKernel>(sycl::range<1>(n),
                                   [=](sycl::item<1> item) {
      const std::size_t i = item[0];
      const float rho = sycl::fmax(density[i], rho0 * 1.0e-6f);
      const float ratio = rho / rho0;
      const float ratio2 = ratio * ratio;
      const float ratio4 = ratio2 * ratio2;
      float value = pressure_scale * (ratio4 * ratio2 * ratio - 1.0f);
      if (!sycl::isfinite(value)) {
        value = 0.0f;
      }
      pressure[i] = value;
    });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_pressure_force(sycl::queue &queue, ReplicaBuffers &replica,
                           std::size_t step, std::size_t max_neighbors,
                           bool wait_each_kernel) {
  const std::size_t n = replica.spec->particles;
  const float h = replica.spec->smoothing_length;
  const float h2 = h * h;
  const float inv_h = 1.0f / h;
  const float inv_h5 = inv_h * inv_h * inv_h * inv_h * inv_h;
  const float mass = replica.spec->mass;
  const float rho_floor = replica.spec->rho0 * 1.0e-6f;
  queue.submit([&](sycl::handler &cgh) {
    auto position = current_position(replica, step)
                        .buffer.template get_access<sycl::access::mode::read>(
                            cgh);
    auto density = replica.density.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto pressure = replica.pressure.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto count = replica.neighbor_count.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto index = replica.neighbor_index.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto acceleration = replica.pressure_accel.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<SphPressureForceKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t i = item[0];
          const Vec4 pi = position[i];
          const float rhoi = sycl::fmax(density[i], rho_floor);
          const float pressure_i = pressure[i];
          float ax = 0.0f;
          float ay = 0.0f;
          float az = 0.0f;
          const std::size_t row = i * max_neighbors;
          const std::size_t neighbors = count[i];
          for (std::size_t entry = 0; entry < neighbors; ++entry) {
            const std::size_t j = index[row + entry];
            const Vec4 pj = position[j];
            const float dx = minimum_image(pi.x - pj.x);
            const float dy = minimum_image(pi.y - pj.y);
            const float dz = minimum_image(pi.z - pj.z);
            const float distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 > 0.0f && distance2 < 4.0f * h2) {
              const float distance = sycl::sqrt(distance2);
              const float q = distance / h;
              const float a = 1.0f - 0.5f * q;
              const float gradient_coefficient =
                  -5.0f * kWendlandAlpha * inv_h5 * a * a * a;
              const float rhoj = sycl::fmax(density[j], rho_floor);
              const float symmetric_pressure =
                  pressure_i / (rhoi * rhoi) +
                  pressure[j] / (rhoj * rhoj);
              const float scale =
                  -mass * symmetric_pressure * gradient_coefficient;
              ax += scale * dx;
              ay += scale * dy;
              az += scale * dz;
            }
          }
          acceleration[i] = {ax, ay, az, 0.0f};
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_viscosity_force(sycl::queue &queue, ReplicaBuffers &replica,
                            std::size_t step, std::size_t max_neighbors,
                            bool wait_each_kernel) {
  const std::size_t n = replica.spec->particles;
  const float h = replica.spec->smoothing_length;
  const float h2 = h * h;
  const float inv_h = 1.0f / h;
  const float inv_h5 = inv_h * inv_h * inv_h * inv_h * inv_h;
  const float mass_nu = replica.spec->mass * replica.spec->viscosity;
  const float rho_floor = replica.spec->rho0 * 1.0e-6f;
  queue.submit([&](sycl::handler &cgh) {
    auto position = current_position(replica, step)
                        .buffer.template get_access<sycl::access::mode::read>(
                            cgh);
    auto velocity = current_velocity(replica, step)
                        .buffer.template get_access<sycl::access::mode::read>(
                            cgh);
    auto density = replica.density.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto count = replica.neighbor_count.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto index = replica.neighbor_index.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto acceleration = replica.viscosity_accel.buffer.template get_access<
        sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<SphViscosityForceKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t i = item[0];
          const Vec4 pi = position[i];
          const Vec4 vi = velocity[i];
          float ax = 0.0f;
          float ay = 0.0f;
          float az = 0.0f;
          const std::size_t row = i * max_neighbors;
          const std::size_t neighbors = count[i];
          for (std::size_t entry = 0; entry < neighbors; ++entry) {
            const std::size_t j = index[row + entry];
            const Vec4 pj = position[j];
            const float dx = minimum_image(pi.x - pj.x);
            const float dy = minimum_image(pi.y - pj.y);
            const float dz = minimum_image(pi.z - pj.z);
            const float distance2 = dx * dx + dy * dy + dz * dz;
            if (distance2 > 0.0f && distance2 < 4.0f * h2) {
              const float distance = sycl::sqrt(distance2);
              const float q = distance / h;
              const float a = 1.0f - 0.5f * q;
              const float gradient_coefficient =
                  -5.0f * kWendlandAlpha * inv_h5 * a * a * a;
              const float laplacian =
                  -10.0f * gradient_coefficient * distance2 /
                  (distance2 + 0.01f * h2);
              const float rhoj = sycl::fmax(density[j], rho_floor);
              const float scale = mass_nu * laplacian / rhoj;
              const Vec4 vj = velocity[j];
              ax += scale * (vj.x - vi.x);
              ay += scale * (vj.y - vi.y);
              az += scale * (vj.z - vi.z);
            }
          }
          acceleration[i] = {ax, ay, az, 0.0f};
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_integrate(sycl::queue &queue, ReplicaBuffers &replica,
                      std::size_t step, bool wait_each_kernel) {
  const std::size_t n = replica.spec->particles;
  const float dt = replica.spec->dt;
  queue.submit([&](sycl::handler &cgh) {
    auto position_in = current_position(replica, step)
                           .buffer.template get_access<
                               sycl::access::mode::read>(cgh);
    auto velocity_in = current_velocity(replica, step)
                           .buffer.template get_access<
                               sycl::access::mode::read>(cgh);
    auto pressure_accel = replica.pressure_accel.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto viscosity_accel = replica.viscosity_accel.buffer.template get_access<
        sycl::access::mode::read>(cgh);
    auto position_out = next_position(replica, step)
                            .buffer.template get_access<
                                sycl::access::mode::discard_write>(cgh);
    auto velocity_out = next_velocity(replica, step)
                            .buffer.template get_access<
                                sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<SphIntegrateKernel>(
        sycl::range<1>(n), [=](sycl::item<1> item) {
          const std::size_t i = item[0];
          const Vec4 p = position_in[i];
          const Vec4 v = velocity_in[i];
          const Vec4 ap = pressure_accel[i];
          const Vec4 av = viscosity_accel[i];
          float vx = kDamping *
                     (v.x + dt * (ap.x + av.x + kGravityX));
          float vy = kDamping *
                     (v.y + dt * (ap.y + av.y + kGravityY));
          float vz = kDamping *
                     (v.z + dt * (ap.z + av.z + kGravityZ));
          float x = p.x + dt * vx;
          float y = p.y + dt * vy;
          float z = p.z + dt * vz;
          x -= sycl::floor(x / kBoxLength) * kBoxLength;
          y -= sycl::floor(y / kBoxLength) * kBoxLength;
          z -= sycl::floor(z / kBoxLength) * kBoxLength;
          position_out[i] = {x, y, z, 0.0f};
          velocity_out[i] = {vx, vy, vz, 0.0f};
        });
  });
  optional_debug_wait(queue, wait_each_kernel);
}

void submit_step(sycl::queue &queue,
                 std::vector<std::unique_ptr<ReplicaBuffers>> &replicas,
                 std::size_t step, std::size_t max_neighbors,
                 bool wait_each_kernel) {
  // Submit by physical DAG level. Accessor dependencies, rather than submit
  // order or fake guard buffers, carry all intra-replica edges.
  for (auto &replica : replicas) {
    submit_density(queue, *replica, step, max_neighbors, wait_each_kernel);
  }
  for (auto &replica : replicas) {
    submit_eos(queue, *replica, wait_each_kernel);
  }
  for (auto &replica : replicas) {
    submit_pressure_force(queue, *replica, step, max_neighbors,
                          wait_each_kernel);
  }
  for (auto &replica : replicas) {
    submit_viscosity_force(queue, *replica, step, max_neighbors,
                           wait_each_kernel);
  }
  for (auto &replica : replicas) {
    submit_integrate(queue, *replica, step, wait_each_kernel);
  }
}

struct ReferenceState {
  std::vector<Vec4> position;
  std::vector<Vec4> velocity;
  std::vector<float> density;
  std::vector<float> pressure;
  std::vector<Vec4> pressure_accel;
  std::vector<Vec4> viscosity_accel;
};

ReferenceState run_host_reference(const ClassSpec &spec,
                                  const ClassGeometry &geometry,
                                  std::size_t replica_index,
                                  std::size_t max_neighbors,
                                  std::size_t steps) {
  const std::size_t n = spec.particles;
  ReferenceState state;
  state.position = geometry.initial_position;
  state.velocity = make_initial_velocity(geometry, replica_index);
  state.density.resize(n);
  state.pressure.resize(n);
  state.pressure_accel.resize(n);
  state.viscosity_accel.resize(n);
  std::vector<Vec4> next_position_values(n);
  std::vector<Vec4> next_velocity_values(n);

  const float h = spec.smoothing_length;
  const float h2 = h * h;
  const float inv_h = 1.0f / h;
  const float inv_h3 = inv_h * inv_h * inv_h;
  const float inv_h5 = inv_h3 * inv_h * inv_h;
  const float rho_floor = spec.rho0 * 1.0e-6f;
  const float pressure_scale =
      spec.rho0 * spec.sound_speed * spec.sound_speed / kGamma;

  for (std::size_t step = 0; step < steps; ++step) {
    for (std::size_t i = 0; i < n; ++i) {
      const Vec4 pi = state.position[i];
      float rho = spec.mass * kWendlandAlpha * inv_h3;
      const std::size_t row = i * max_neighbors;
      for (std::size_t entry = 0; entry < geometry.neighbor_count[i]; ++entry) {
        const std::size_t j = geometry.neighbor_index[row + entry];
        const Vec4 pj = state.position[j];
        const float dx = minimum_image(pi.x - pj.x);
        const float dy = minimum_image(pi.y - pj.y);
        const float dz = minimum_image(pi.z - pj.z);
        const float distance2 = dx * dx + dy * dy + dz * dz;
        if (distance2 < 4.0f * h2) {
          rho += spec.mass *
                 wendland_weight(std::sqrt(distance2), h, inv_h3);
        }
      }
      state.density[i] = rho;
    }

    for (std::size_t i = 0; i < n; ++i) {
      const float rho = std::max(state.density[i], rho_floor);
      const float ratio = rho / spec.rho0;
      const float ratio2 = ratio * ratio;
      const float ratio4 = ratio2 * ratio2;
      float value = pressure_scale * (ratio4 * ratio2 * ratio - 1.0f);
      state.pressure[i] = std::isfinite(value) ? value : 0.0f;
    }

    for (std::size_t i = 0; i < n; ++i) {
      const Vec4 pi = state.position[i];
      const Vec4 vi = state.velocity[i];
      const float rhoi = std::max(state.density[i], rho_floor);
      float pax = 0.0f;
      float pay = 0.0f;
      float paz = 0.0f;
      float vax = 0.0f;
      float vay = 0.0f;
      float vaz = 0.0f;
      const std::size_t row = i * max_neighbors;
      for (std::size_t entry = 0; entry < geometry.neighbor_count[i]; ++entry) {
        const std::size_t j = geometry.neighbor_index[row + entry];
        const Vec4 pj = state.position[j];
        const float dx = minimum_image(pi.x - pj.x);
        const float dy = minimum_image(pi.y - pj.y);
        const float dz = minimum_image(pi.z - pj.z);
        const float distance2 = dx * dx + dy * dy + dz * dz;
        if (distance2 > 0.0f && distance2 < 4.0f * h2) {
          const float distance = std::sqrt(distance2);
          const float q = distance / h;
          const float a = 1.0f - 0.5f * q;
          const float gradient_coefficient =
              -5.0f * kWendlandAlpha * inv_h5 * a * a * a;
          const float rhoj = std::max(state.density[j], rho_floor);
          const float symmetric_pressure =
              state.pressure[i] / (rhoi * rhoi) +
              state.pressure[j] / (rhoj * rhoj);
          const float pressure_term =
              -spec.mass * symmetric_pressure * gradient_coefficient;
          pax += pressure_term * dx;
          pay += pressure_term * dy;
          paz += pressure_term * dz;

          const float laplacian =
              -10.0f * gradient_coefficient * distance2 /
              (distance2 + 0.01f * h2);
          const float viscosity_term =
              spec.mass * spec.viscosity * laplacian / rhoj;
          const Vec4 vj = state.velocity[j];
          vax += viscosity_term * (vj.x - vi.x);
          vay += viscosity_term * (vj.y - vi.y);
          vaz += viscosity_term * (vj.z - vi.z);
        }
      }
      state.pressure_accel[i] = {pax, pay, paz, 0.0f};
      state.viscosity_accel[i] = {vax, vay, vaz, 0.0f};
    }

    for (std::size_t i = 0; i < n; ++i) {
      const Vec4 p = state.position[i];
      const Vec4 v = state.velocity[i];
      const Vec4 ap = state.pressure_accel[i];
      const Vec4 av = state.viscosity_accel[i];
      float vx = kDamping *
                 (v.x + spec.dt * (ap.x + av.x + kGravityX));
      float vy = kDamping *
                 (v.y + spec.dt * (ap.y + av.y + kGravityY));
      float vz = kDamping *
                 (v.z + spec.dt * (ap.z + av.z + kGravityZ));
      next_position_values[i] = {wrap_host(p.x + spec.dt * vx),
                                 wrap_host(p.y + spec.dt * vy),
                                 wrap_host(p.z + spec.dt * vz), 0.0f};
      next_velocity_values[i] = {vx, vy, vz, 0.0f};
    }
    state.position.swap(next_position_values);
    state.velocity.swap(next_velocity_values);
  }
  return state;
}

struct ResultSummary {
  double checksum = 0.0;
  double sample_rho = 0.0;
  Vec4 sample_position{0.0f, 0.0f, 0.0f, 0.0f};
  double max_displacement = 0.0;
  double reference_relative_l2 = 0.0;
  double reference_max_scaled = 0.0;
  std::size_t reference_replicas = 0;
  bool finite_and_physical = true;
  bool displacement_valid = true;
  bool reference_valid = true;
};

void compare_reference_value(double actual, double expected,
                             long double &difference2,
                             long double &reference2,
                             double &max_scaled) {
  const double difference = actual - expected;
  difference2 += static_cast<long double>(difference) * difference;
  reference2 += static_cast<long double>(expected) * expected;
  max_scaled =
      std::max(max_scaled, std::abs(difference) / (1.0 + std::abs(expected)));
}

ResultSummary read_and_verify(
    std::vector<std::unique_ptr<ReplicaBuffers>> &replicas,
    const std::vector<ClassSpec> &specs,
    const std::vector<ClassGeometry> &geometries, const Config &config) {
  ResultSummary result;
  std::vector<bool> class_reference_done(specs.size(), false);
  long double reference_difference2 = 0.0L;
  long double reference_norm2 = 0.0L;

  for (auto &replica_pointer : replicas) {
    ReplicaBuffers &replica = *replica_pointer;
    const ClassSpec &spec = *replica.spec;
    const ClassGeometry &geometry = geometries[replica.class_id];
    const bool run_reference = config.verify && spec.particles <= 2048 &&
                               !class_reference_done[replica.class_id];
    ReferenceState reference;
    if (run_reference) {
      reference = run_host_reference(spec, geometry, replica.replica_index,
                                     config.max_neighbors, config.steps);
      class_reference_done[replica.class_id] = true;
      ++result.reference_replicas;
    }

    auto position = final_position(replica, config.steps)
                        .buffer.get_host_access(sycl::read_only);
    auto velocity = final_velocity(replica, config.steps)
                        .buffer.get_host_access(sycl::read_only);
    auto density = replica.density.buffer.get_host_access(sycl::read_only);
    auto pressure = replica.pressure.buffer.get_host_access(sycl::read_only);
    auto pressure_accel =
        replica.pressure_accel.buffer.get_host_access(sycl::read_only);
    auto viscosity_accel =
        replica.viscosity_accel.buffer.get_host_access(sycl::read_only);

    for (std::size_t i = 0; i < spec.particles; ++i) {
      const Vec4 p = position[i];
      const Vec4 initial = geometry.initial_position[i];
      const float dx = minimum_image(p.x - initial.x);
      const float dy = minimum_image(p.y - initial.y);
      const float dz = minimum_image(p.z - initial.z);
      const double displacement =
          std::sqrt(static_cast<double>(dx) * dx +
                    static_cast<double>(dy) * dy +
                    static_cast<double>(dz) * dz);
      result.max_displacement = std::max(result.max_displacement, displacement);
      result.finite_and_physical =
          result.finite_and_physical && std::isfinite(p.x) &&
          std::isfinite(p.y) && std::isfinite(p.z) && p.x >= 0.0f &&
          p.x < kBoxLength && p.y >= 0.0f && p.y < kBoxLength &&
          p.z >= 0.0f && p.z < kBoxLength;
      result.displacement_valid =
          result.displacement_valid && displacement < 0.5 * spec.skin;
      const double weight =
          1.0 + 0.0001 * static_cast<double>(replica.replica_index + 1);
      result.checksum +=
          weight * (static_cast<double>(p.x) + 3.0 * p.y + 7.0 * p.z);

      if (run_reference) {
        const Vec4 rp = reference.position[i];
        compare_reference_value(p.x, rp.x, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(p.y, rp.y, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(p.z, rp.z, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
      }
    }

    const std::size_t sample_ids[4] = {0, spec.particles / 3,
                                       spec.particles / 2,
                                       spec.particles - 1};
    const bool inspect_full_fields = config.host_read_full || run_reference;
    const std::size_t sample_count =
        inspect_full_fields ? spec.particles : std::size_t{4};
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
      const std::size_t i =
          inspect_full_fields ? sample : sample_ids[sample];
      const Vec4 v = velocity[i];
      const float rho = density[i];
      const float p = pressure[i];
      const Vec4 ap = pressure_accel[i];
      const Vec4 av = viscosity_accel[i];
      result.finite_and_physical =
          result.finite_and_physical && std::isfinite(v.x) &&
          std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(rho) &&
          rho > 0.0f && std::isfinite(p) && std::isfinite(ap.x) &&
          std::isfinite(ap.y) && std::isfinite(ap.z) &&
          std::isfinite(av.x) && std::isfinite(av.y) && std::isfinite(av.z);
      if (config.host_read_full) {
        result.checksum +=
            1.0e-4 * (static_cast<double>(v.x) + 3.0 * v.y + 7.0 * v.z) +
            1.0e-8 * static_cast<double>(rho) +
            1.0e-12 * static_cast<double>(p + ap.x + av.x);
      }
      if (run_reference) {
        const Vec4 rv = reference.velocity[i];
        const Vec4 rap = reference.pressure_accel[i];
        const Vec4 rav = reference.viscosity_accel[i];
        compare_reference_value(v.x, rv.x, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(v.y, rv.y, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(v.z, rv.z, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(rho, reference.density[i],
                                reference_difference2, reference_norm2,
                                result.reference_max_scaled);
        compare_reference_value(p, reference.pressure[i],
                                reference_difference2, reference_norm2,
                                result.reference_max_scaled);
        compare_reference_value(ap.x, rap.x, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(ap.y, rap.y, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(ap.z, rap.z, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(av.x, rav.x, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(av.y, rav.y, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
        compare_reference_value(av.z, rav.z, reference_difference2,
                                reference_norm2, result.reference_max_scaled);
      }
    }

    if (replica.replica_index == 0) {
      const std::size_t middle = spec.particles / 2;
      result.sample_rho = density[middle];
      result.sample_position = position[middle];
    }
  }

  if (result.reference_replicas > 0) {
    result.reference_relative_l2 =
        std::sqrt(static_cast<double>(reference_difference2 /
                                     std::max(reference_norm2, 1.0e-30L)));
    result.reference_valid = result.reference_relative_l2 <= 5.0e-3 &&
                             result.reference_max_scaled <= 5.0e-3;
  }
  result.finite_and_physical =
      result.finite_and_physical && std::isfinite(result.checksum);
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

  std::vector<ClassSpec> specs;
  std::size_t effective_replicas = 0;
  std::size_t total_estimated_bytes = 0;
  if (!derive_specs(config, specs, effective_replicas,
                    total_estimated_bytes, error)) {
    std::cerr << "ERROR " << error << '\n';
    return 1;
  }

  std::size_t kernels_per_step = 0;
  std::size_t total_kernels = 0;
  std::size_t max_width = 0;
  if (!checked_multiply(std::size_t{5}, effective_replicas,
                        kernels_per_step) ||
      !checked_multiply(kernels_per_step, config.steps, total_kernels) ||
      !checked_multiply(std::size_t{2}, effective_replicas, max_width)) {
    std::cerr << "ERROR DAG kernel-count arithmetic overflows size_t\n";
    return 1;
  }

  std::cout << std::setprecision(9);
  std::cout << "CONFIG particles=" << config.particles
            << " replicas=" << effective_replicas
            << " steps=" << config.steps
            << " max_neighbors=" << config.max_neighbors
            << " classes=" << specs.size()
            << " coarsen_percent=" << config.coarsen_percent
            << " mode=" << mode_name(config.mode)
            << " dt=" << config.dt
            << " window_steps=" << config.window_steps
            << " split_parts=" << config.split_parts
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " host_read_full=" << (config.host_read_full ? 1 : 0)
            << '\n';
  std::cout << "DAG kernels_per_step=" << kernels_per_step
            << " max_width=" << max_width
            << " critical_path_levels=4 total_kernels=" << total_kernels
            << '\n';
  for (const ClassSpec &spec : specs) {
    std::cout << "MEMORY class=" << spec.id
              << " bytes_per_replica=" << spec.bytes_per_replica
              << " replicas=" << spec.replicas
              << " class_bytes=" << spec.bytes_per_replica * spec.replicas
              << '\n';
  }
  std::cout << "MEMORY bytes_per_particle_formula="
            << 108 + 4 * config.max_neighbors
            << " total_estimated_bytes=" << total_estimated_bytes << '\n';

  try {
    const double total_begin = get_time();
    std::vector<ClassGeometry> geometries(specs.size());
    for (std::size_t class_id = 0; class_id < specs.size(); ++class_id) {
      if (!build_geometry(specs[class_id], config.max_neighbors,
                          geometries[class_id], error)) {
        std::cerr << "ERROR " << error << '\n';
        return 1;
      }
    }

    std::vector<std::unique_ptr<ReplicaBuffers>> replicas;
    replicas.reserve(effective_replicas);
    std::size_t replica_index = 0;
    for (const ClassSpec &spec : specs) {
      for (std::size_t local = 0; local < spec.replicas; ++local) {
        replicas.push_back(std::make_unique<ReplicaBuffers>(
            replica_index, spec, geometries[spec.id], config.max_neighbors));
        ++replica_index;
      }
    }
    // Large-run verification only needs the initial positions. Keep the full
    // host topology solely for the small CPU reference path; every replica
    // already owns an independent device-visible copy at this point.
    for (std::size_t class_id = 0; class_id < specs.size(); ++class_id) {
      if (!(config.verify && specs[class_id].particles <= 2048)) {
        std::vector<std::uint32_t>().swap(
            geometries[class_id].neighbor_count);
        std::vector<std::uint32_t>().swap(
            geometries[class_id].neighbor_index);
      }
    }
    const double init_end = get_time();

    for (const ClassSpec &spec : specs) {
      std::cout << "CLASS id=" << spec.id << " replicas=" << spec.replicas
                << " particles=" << spec.particles
                << " avg_neighbors=" << spec.average_neighbors
                << " h=" << spec.smoothing_length
                << " skin=" << spec.skin << " rho0=" << spec.rho0
                << " viscosity=" << spec.viscosity
                << " sound_speed=" << spec.sound_speed
                << " dt=" << spec.dt << '\n';
    }

    const double queue_begin = get_time();
    sycl::queue queue;
    const double queue_end = get_time();

    const double run_begin = get_time();
    for (std::size_t step = 0; step < config.steps; ++step) {
      submit_step(queue, replicas, step, config.max_neighbors,
                  config.wait_each_kernel);
      const bool end_of_window =
          (step + 1) % config.window_steps == 0 || step + 1 == config.steps;
      if (end_of_window) {
        queue.wait();
      }
    }
    const double run_end = get_time();

    const double host_begin = get_time();
    const ResultSummary result =
        read_and_verify(replicas, specs, geometries, config);
    const double host_end = get_time();
    const bool passed = result.finite_and_physical &&
                        result.displacement_valid && result.reference_valid;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_end - total_begin << '\n';
    std::cout << "TIMING queue_sec=" << queue_end - queue_begin << '\n';
    std::cout << "TIMING run_sec=" << run_end - run_begin << '\n';
    std::cout << "TIMING host_sec=" << host_end - host_begin << '\n';
    std::cout << "TIMING total_sec=" << host_end - total_begin << '\n';
    std::cout << "RESULT checksum=" << result.checksum
              << " sample_rho=" << result.sample_rho
              << " sample_pos=" << result.sample_position.x << ','
              << result.sample_position.y << ',' << result.sample_position.z
              << " max_displacement=" << result.max_displacement << '\n';
    std::cout << "VERIFY passed=" << (passed ? 1 : 0)
              << " enabled=" << (config.verify ? 1 : 0)
              << " finite=" << (result.finite_and_physical ? 1 : 0)
              << " displacement=" << (result.displacement_valid ? 1 : 0)
              << " reference_replicas=" << result.reference_replicas
              << " reference_rel_l2=" << result.reference_relative_l2
              << " reference_max_scaled=" << result.reference_max_scaled
              << '\n';
    return config.verify && !passed ? 2 : 0;
  } catch (const sycl::exception &exception) {
    std::cerr << "ERROR SYCL " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }
}
