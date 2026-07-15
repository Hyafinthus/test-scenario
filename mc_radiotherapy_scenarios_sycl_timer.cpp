// Scenario-parallel Monte Carlo photon-transport SYCL miniapp.
//
// Each scenario owns all of its input, material, and tally buffers. One long
// kernel transports a complete batch of photon histories for one scenario.
// Scenarios are submitted together and are independent at the kernel level;
// epochs are semantic Monte Carlo batches and are the only normal wait points.
//
// This is a scheduling benchmark inspired by radiotherapy robustness and 4-D
// dose calculations. It is not a clinically validated dose engine.

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
#include <vector>

class McPhotonTransportKernel;

namespace {

constexpr std::size_t kDefaultWorkItems = 262144;
constexpr std::size_t kDefaultScenarios = 4;
constexpr std::size_t kDefaultEpochs = 2;
// A deliberately heavy candidate default. The calibration mode remains the
// authority for the actual per-kernel duration on a particular GPU.
constexpr std::size_t kDefaultHistoriesPerItem = 256;
constexpr std::size_t kDefaultMaxCollisions = 128;
constexpr std::size_t kDefaultLayers = 128;
constexpr std::size_t kDefaultSplitParts = 2;
constexpr double kDefaultTargetKernelSec = 20.0;
constexpr float kSlabThickness = 20.0f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kMinPhotonWeight = 1.0e-4f;
constexpr float kInvUint24 = 1.0f / 16777216.0f;

struct Config {
  std::size_t work_items = kDefaultWorkItems;
  std::size_t scenarios = kDefaultScenarios;
  std::size_t epochs = kDefaultEpochs;
  std::size_t histories_per_item = kDefaultHistoriesPerItem;
  std::size_t max_collisions = kDefaultMaxCollisions;
  std::size_t layers = kDefaultLayers;
  std::size_t split_parts = kDefaultSplitParts;
  double target_kernel_sec = kDefaultTargetKernelSec;
  double memory_limit_gib = 0.0;
  bool calibrate = false;
  bool wait_each_kernel = false;
  bool host_read_full = false;
  bool verify = true;
};

double get_time() {
  timeval tv{};
  gettimeofday(&tv, nullptr);
  return static_cast<double>(tv.tv_sec) +
         static_cast<double>(tv.tv_usec) / 1.0e6;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t &result) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
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

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --work-items <int>          Work-items per scenario (default 262144)\n"
      << "  --scenarios <int>           Independent scenario kernels (default 4)\n"
      << "  --epochs <int>              Monte Carlo batches/wait windows (default 2)\n"
      << "  --histories-per-item <int>  Photon histories per work-item (default 256)\n"
      << "  --max-collisions <int>      Transport cap per history (default 128)\n"
      << "  --layers <int>              Material layers per scenario (default 128)\n"
      << "  --split-parts <int>         Validate dim0 divisibility (default 2)\n"
      << "  --memory-limit-gib <real>   Fail above estimate; 0 means unlimited\n"
      << "  --calibrate <0|1>           Time one kernel, print recommendation, exit\n"
      << "  --target-kernel-sec <real>  Calibration target (default 20)\n"
      << "  --wait-each-kernel <0|1>    Debug-only serialization (default 0)\n"
      << "  --host-read-full <0|1>      Verify every tally entry (default 0)\n"
      << "  --verify <0|1>              Enable conservation checks (default 1)\n";
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
    if (option == "--work-items") {
      parsed = parse_size(value, config.work_items);
    } else if (option == "--scenarios") {
      parsed = parse_size(value, config.scenarios);
    } else if (option == "--epochs") {
      parsed = parse_size(value, config.epochs);
    } else if (option == "--histories-per-item") {
      parsed = parse_size(value, config.histories_per_item);
    } else if (option == "--max-collisions") {
      parsed = parse_size(value, config.max_collisions);
    } else if (option == "--layers") {
      parsed = parse_size(value, config.layers);
    } else if (option == "--split-parts") {
      parsed = parse_size(value, config.split_parts);
    } else if (option == "--target-kernel-sec") {
      parsed = parse_double(value, config.target_kernel_sec);
    } else if (option == "--memory-limit-gib") {
      parsed = parse_double(value, config.memory_limit_gib);
    } else if (option == "--calibrate") {
      parsed = parse_binary(value, config.calibrate);
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
      error = "invalid value for " + option + ": " + value;
      return false;
    }
  }
  return true;
}

std::uint32_t host_mix32(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value == 0 ? 0x9e3779b9u : value;
}

inline std::uint32_t device_mix32(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value == 0 ? 0x9e3779b9u : value;
}

inline std::uint32_t rng_next(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

inline float rng_open01(std::uint32_t &state) {
  const std::uint32_t mantissa = rng_next(state) >> 8;
  return (static_cast<float>(mantissa) + 0.5f) * kInvUint24;
}

struct ScenarioBuffers {
  std::size_t scenario_id;
  std::size_t work_items;
  std::size_t layers;
  sycl::buffer<std::uint32_t, 1> seeds;
  sycl::buffer<float, 1> source_energy;
  sycl::buffer<float, 1> source_cosine;
  sycl::buffer<float, 1> absorption;
  sycl::buffer<float, 1> scattering;
  sycl::buffer<float, 1> anisotropy;
  sycl::buffer<float, 1> deposited;
  sycl::buffer<float, 1> reflected;
  sycl::buffer<float, 1> transmitted;
  sycl::buffer<float, 1> residual;
  sycl::buffer<float, 1> path_length;
  sycl::buffer<std::uint32_t, 1> collision_count;

  ScenarioBuffers(std::size_t id, std::size_t item_count,
                  std::size_t layer_count)
      : scenario_id(id), work_items(item_count), layers(layer_count),
        seeds(sycl::range<1>(item_count)),
        source_energy(sycl::range<1>(item_count)),
        source_cosine(sycl::range<1>(item_count)),
        absorption(sycl::range<1>(layer_count)),
        scattering(sycl::range<1>(layer_count)),
        anisotropy(sycl::range<1>(layer_count)),
        deposited(sycl::range<1>(item_count)),
        reflected(sycl::range<1>(item_count)),
        transmitted(sycl::range<1>(item_count)),
        residual(sycl::range<1>(item_count)),
        path_length(sycl::range<1>(item_count)),
        collision_count(sycl::range<1>(item_count)) {
    initialize();
  }

private:
  void initialize() {
    auto seed = seeds.get_host_access(sycl::write_only);
    auto energy = source_energy.get_host_access(sycl::write_only);
    auto cosine = source_cosine.get_host_access(sycl::write_only);
    auto mu_a = absorption.get_host_access(sycl::write_only);
    auto mu_s = scattering.get_host_access(sycl::write_only);
    auto g = anisotropy.get_host_access(sycl::write_only);
    auto dep = deposited.get_host_access(sycl::write_only);
    auto refl = reflected.get_host_access(sycl::write_only);
    auto trans = transmitted.get_host_access(sycl::write_only);
    auto remain = residual.get_host_access(sycl::write_only);
    auto path = path_length.get_host_access(sycl::write_only);
    auto collisions = collision_count.get_host_access(sycl::write_only);

    for (std::size_t i = 0; i < work_items; ++i) {
      const std::uint32_t mixed = host_mix32(
          static_cast<std::uint32_t>(i) ^
          static_cast<std::uint32_t>((scenario_id + 1) * 0x9e3779b9u));
      seed[i] = mixed;
      energy[i] = 0.95f + 0.10f * static_cast<float>(mixed & 0xffffu) /
                                65535.0f;
      cosine[i] = 0.97f +
                  0.029f * static_cast<float>((mixed >> 16) & 0xffffu) /
                      65535.0f;
      dep[i] = 0.0f;
      refl[i] = 0.0f;
      trans[i] = 0.0f;
      remain[i] = 0.0f;
      path[i] = 0.0f;
      collisions[i] = 0;
    }

    const float scenario_shift = 0.002f * static_cast<float>(scenario_id % 7);
    for (std::size_t layer = 0; layer < layers; ++layer) {
      const float depth = (static_cast<float>(layer) + 0.5f) /
                          static_cast<float>(layers);
      const float tissue_wave =
          0.5f + 0.5f * std::sin(kTwoPi * depth * 3.0f +
                                 0.37f * static_cast<float>(scenario_id));
      mu_a[layer] = 0.012f + scenario_shift + 0.010f * tissue_wave;
      mu_s[layer] = 4.2f + 0.15f * static_cast<float>(scenario_id % 5) +
                    1.3f * (1.0f - tissue_wave);
      g[layer] = 0.76f + 0.12f * tissue_wave;
    }
  }
};

bool estimate_memory(const Config &config, std::size_t &bytes_per_scenario,
                     std::size_t &total_bytes) {
  std::size_t item_bytes = 0;
  std::size_t layer_bytes = 0;
  std::size_t tmp = 0;
  // Per work-item: one uint seed, two float source values, five float tallies,
  // and one uint collision tally: 36 bytes in total.
  if (!checked_mul(config.work_items, std::size_t{36}, item_bytes) ||
      !checked_mul(config.layers, std::size_t{3 * sizeof(float)},
                   layer_bytes) ||
      !checked_add(item_bytes, layer_bytes, bytes_per_scenario) ||
      !checked_mul(bytes_per_scenario, config.scenarios, tmp)) {
    return false;
  }
  total_bytes = tmp;
  return true;
}

bool validate_config(const Config &config, std::size_t total_bytes,
                     std::string &error) {
  if (config.layers < 2) {
    error = "layers must be at least 2";
    return false;
  }
  if (config.work_items % config.split_parts != 0) {
    error = "work-items must be divisible by split-parts";
    return false;
  }
  if (config.target_kernel_sec <= 0.0 || config.memory_limit_gib < 0.0) {
    error = "target-kernel-sec must be positive and memory-limit-gib nonnegative";
    return false;
  }
  std::size_t per_item_collision_cap = 0;
  if (!checked_mul(config.epochs, config.histories_per_item,
                   per_item_collision_cap) ||
      !checked_mul(per_item_collision_cap, config.max_collisions,
                   per_item_collision_cap) ||
      per_item_collision_cap > std::numeric_limits<std::uint32_t>::max()) {
    error = "per-item collision counter can overflow uint32";
    return false;
  }
  if (config.memory_limit_gib > 0.0) {
    const long double limit = static_cast<long double>(config.memory_limit_gib) *
                              1024.0L * 1024.0L * 1024.0L;
    if (static_cast<long double>(total_bytes) > limit) {
      error = "estimated buffers exceed memory-limit-gib";
      return false;
    }
  }
  return true;
}

void submit_scenario(sycl::queue &queue, ScenarioBuffers &scenario,
                     const Config &config, std::size_t epoch) {
  const std::size_t work_items = scenario.work_items;
  const std::size_t layers = scenario.layers;
  const std::size_t histories_per_item = config.histories_per_item;
  const std::size_t max_collisions = config.max_collisions;
  const std::uint32_t epoch_key = host_mix32(
      static_cast<std::uint32_t>(epoch + 1) ^
      static_cast<std::uint32_t>((scenario.scenario_id + 1) * 0x85ebca6bu));

  queue.submit([&](sycl::handler &cgh) {
    auto seeds = scenario.seeds.get_access<sycl::access::mode::read>(cgh);
    auto source_energy =
        scenario.source_energy.get_access<sycl::access::mode::read>(cgh);
    auto source_cosine =
        scenario.source_cosine.get_access<sycl::access::mode::read>(cgh);
    auto absorption =
        scenario.absorption.get_access<sycl::access::mode::read>(cgh);
    auto scattering =
        scenario.scattering.get_access<sycl::access::mode::read>(cgh);
    auto anisotropy =
        scenario.anisotropy.get_access<sycl::access::mode::read>(cgh);
    auto deposited =
        scenario.deposited.get_access<sycl::access::mode::read_write>(cgh);
    auto reflected =
        scenario.reflected.get_access<sycl::access::mode::read_write>(cgh);
    auto transmitted =
        scenario.transmitted.get_access<sycl::access::mode::read_write>(cgh);
    auto residual =
        scenario.residual.get_access<sycl::access::mode::read_write>(cgh);
    auto path_length =
        scenario.path_length.get_access<sycl::access::mode::read_write>(cgh);
    auto collision_count =
        scenario.collision_count.get_access<sycl::access::mode::read_write>(
            cgh);

    cgh.parallel_for<McPhotonTransportKernel>(
        sycl::range<1>(work_items), [=](sycl::id<1> index) {
          const std::size_t item = index[0];
          const float initial_energy = source_energy[item];
          const float initial_cosine = source_cosine[item];
          float local_deposited = 0.0f;
          float local_reflected = 0.0f;
          float local_transmitted = 0.0f;
          float local_residual = 0.0f;
          float local_path = 0.0f;
          std::uint32_t local_collisions = 0;

          for (std::size_t history = 0; history < histories_per_item;
               ++history) {
            std::uint32_t rng = device_mix32(
                seeds[item] ^ epoch_key ^
                static_cast<std::uint32_t>((history + 1) * 0x27d4eb2du));
            float weight = initial_energy;
            float x = 0.0f;
            float y = 0.0f;
            float z = 1.0e-4f;
            const float phi0 = kTwoPi * rng_open01(rng);
            const float radial =
                sycl::sqrt(sycl::fmax(0.0f, 1.0f -
                                               initial_cosine * initial_cosine));
            float ux = radial * sycl::cos(phi0);
            float uy = radial * sycl::sin(phi0);
            float uz = initial_cosine;
            bool terminated = false;

            for (std::size_t collision = 0; collision < max_collisions;
                 ++collision) {
              const float normalized_z =
                  sycl::fmin(0.999999f,
                             sycl::fmax(0.0f, z / kSlabThickness));
              std::size_t layer = static_cast<std::size_t>(
                  normalized_z * static_cast<float>(layers));
              if (layer >= layers) {
                layer = layers - 1;
              }
              const float mu_a = absorption[layer];
              const float mu_s = scattering[layer];
              const float mu_t = mu_a + mu_s;
              const float step = -sycl::log(rng_open01(rng)) / mu_t;

              x += step * ux;
              y += step * uy;
              z += step * uz;
              local_path += step;
              ++local_collisions;

              if (z < 0.0f) {
                local_reflected += weight;
                terminated = true;
                break;
              }
              if (z >= kSlabThickness) {
                local_transmitted += weight;
                terminated = true;
                break;
              }

              const float absorbed_weight = weight * (mu_a / mu_t);
              local_deposited += absorbed_weight;
              weight -= absorbed_weight;
              if (weight <= kMinPhotonWeight) {
                local_deposited += weight;
                terminated = true;
                break;
              }

              const float scatter_u = rng_open01(rng);
              const float g = anisotropy[layer];
              float cos_theta = 2.0f * scatter_u - 1.0f;
              if (sycl::fabs(g) > 1.0e-3f) {
                const float ratio =
                    (1.0f - g * g) / (1.0f - g + 2.0f * g * scatter_u);
                cos_theta =
                    (1.0f + g * g - ratio * ratio) / (2.0f * g);
              }
              cos_theta =
                  sycl::fmin(1.0f, sycl::fmax(-1.0f, cos_theta));
              const float sin_theta = sycl::sqrt(
                  sycl::fmax(0.0f, 1.0f - cos_theta * cos_theta));
              const float phi = kTwoPi * rng_open01(rng);
              const float cos_phi = sycl::cos(phi);
              const float sin_phi = sycl::sin(phi);

              if (sycl::fabs(uz) > 0.99999f) {
                ux = sin_theta * cos_phi;
                uy = sin_theta * sin_phi;
                uz = (uz >= 0.0f ? 1.0f : -1.0f) * cos_theta;
              } else {
                const float transverse =
                    sycl::sqrt(sycl::fmax(0.0f, 1.0f - uz * uz));
                const float next_ux =
                    sin_theta * (ux * uz * cos_phi - uy * sin_phi) /
                        transverse +
                    ux * cos_theta;
                const float next_uy =
                    sin_theta * (uy * uz * cos_phi + ux * sin_phi) /
                        transverse +
                    uy * cos_theta;
                const float next_uz =
                    -sin_theta * cos_phi * transverse + uz * cos_theta;
                ux = next_ux;
                uy = next_uy;
                uz = next_uz;
              }
            }

            if (!terminated) {
              local_residual += weight;
            }
            // Keep the transverse trajectory live and observable without a
            // global dose-grid atomic. Real codes reduce private tallies later.
            local_path += 1.0e-7f * (sycl::fabs(x) + sycl::fabs(y));
          }

          deposited[item] += local_deposited;
          reflected[item] += local_reflected;
          transmitted[item] += local_transmitted;
          residual[item] += local_residual;
          path_length[item] += local_path;
          collision_count[item] += local_collisions;
        });
  });
}

struct Result {
  double checksum = 0.0;
  double max_relative_energy_error = 0.0;
  std::uint64_t sampled_collisions = 0;
  std::size_t checked_items = 0;
  std::size_t nonfinite = 0;
  bool valid = true;
};

Result read_result(std::vector<std::unique_ptr<ScenarioBuffers>> &scenarios,
                   const Config &config) {
  Result result;
  constexpr std::size_t sparse_samples = 32;
  for (auto &scenario_ptr : scenarios) {
    ScenarioBuffers &scenario = *scenario_ptr;
    auto energy = scenario.source_energy.get_host_access(sycl::read_only);
    auto dep = scenario.deposited.get_host_access(sycl::read_only);
    auto refl = scenario.reflected.get_host_access(sycl::read_only);
    auto trans = scenario.transmitted.get_host_access(sycl::read_only);
    auto remain = scenario.residual.get_host_access(sycl::read_only);
    auto path = scenario.path_length.get_host_access(sycl::read_only);
    auto collisions =
        scenario.collision_count.get_host_access(sycl::read_only);

    const std::size_t checks =
        config.host_read_full
            ? scenario.work_items
            : std::min(scenario.work_items, sparse_samples);
    for (std::size_t sample = 0; sample < checks; ++sample) {
      const std::size_t item =
          config.host_read_full
              ? sample
              : (sample * scenario.work_items) / checks;
      const double expected =
          static_cast<double>(energy[item]) *
          static_cast<double>(config.histories_per_item) *
          static_cast<double>(config.epochs);
      const double accounted = static_cast<double>(dep[item]) +
                               static_cast<double>(refl[item]) +
                               static_cast<double>(trans[item]) +
                               static_cast<double>(remain[item]);
      const double relative_error =
          std::abs(accounted - expected) / std::max(1.0, expected);
      result.max_relative_energy_error =
          std::max(result.max_relative_energy_error, relative_error);
      const bool finite = std::isfinite(accounted) &&
                          std::isfinite(static_cast<double>(path[item]));
      if (!finite) {
        ++result.nonfinite;
      }
      result.valid = result.valid && finite && dep[item] >= 0.0f &&
                     refl[item] >= 0.0f && trans[item] >= 0.0f &&
                     remain[item] >= 0.0f && path[item] >= 0.0f &&
                     relative_error < 2.0e-4;
      result.sampled_collisions += collisions[item];
      result.checksum +=
          (1.0 + 0.01 * static_cast<double>(scenario.scenario_id)) *
          (accounted + 1.0e-4 * static_cast<double>(path[item]) +
           1.0e-7 * static_cast<double>(collisions[item]));
      ++result.checked_items;
    }
  }
  result.valid = result.valid && result.nonfinite == 0 &&
                 std::isfinite(result.checksum);
  return result;
}

int run_calibration(const Config &config) {
  try {
    const double init_begin = get_time();
    // The daemon profile key does not include scalar kernel captures such as
    // histories_per_item. Use one extra material-table element so a short
    // calibration sample cannot poison the timed kernel's persistent profile.
    std::size_t calibration_layers = 0;
    if (!checked_add(config.layers, std::size_t{1}, calibration_layers)) {
      std::cerr << "ERROR calibration layer-count overflow\n";
      return 2;
    }
    ScenarioBuffers scenario(0, config.work_items, calibration_layers);
    sycl::queue queue;
    const double init_end = get_time();
    const double kernel_begin = get_time();
    submit_scenario(queue, scenario, config, 0);
    queue.wait();
    const double kernel_end = get_time();
    auto dep = scenario.deposited.get_host_access(sycl::read_only);
    const float sample = dep[config.work_items / 2];
    const double measured = kernel_end - kernel_begin;
    if (!(measured > 0.0) || !std::isfinite(measured)) {
      std::cerr << "ERROR invalid calibration time\n";
      return 2;
    }
    const long double scaled =
        std::ceil(static_cast<long double>(config.histories_per_item) *
                  static_cast<long double>(config.target_kernel_sec) /
                  static_cast<long double>(measured));
    const std::size_t recommended = static_cast<std::size_t>(std::max(
        1.0L, std::min(scaled, static_cast<long double>(
                                   std::numeric_limits<std::size_t>::max()))));
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CALIBRATION measured_sec=" << measured
              << " initial_histories_per_item="
              << config.histories_per_item
              << " calibration_layers=" << calibration_layers
              << " target_kernel_sec=" << config.target_kernel_sec
              << " recommended_histories_per_item=" << recommended
              << " sample_deposited=" << sample << '\n';
    std::cout << "CALIBRATION_COMMAND --work-items " << config.work_items
              << " --histories-per-item " << recommended
              << " --max-collisions " << config.max_collisions
              << " --layers " << config.layers << '\n';
    std::cout << "TIMING calibration_init_sec=" << init_end - init_begin
              << " calibration_kernel_sec=" << measured << '\n';
    return 0;
  } catch (const sycl::exception &exception) {
    std::cerr << "ERROR SYCL " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }
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

  std::size_t bytes_per_scenario = 0;
  std::size_t total_bytes = 0;
  if (!estimate_memory(config, bytes_per_scenario, total_bytes)) {
    std::cerr << "ERROR memory estimate overflow\n";
    return 1;
  }
  if (!validate_config(config, total_bytes, error)) {
    std::cerr << "ERROR " << error << '\n';
    return 1;
  }

  std::size_t histories_per_kernel = 0;
  std::size_t histories_per_epoch = 0;
  std::size_t total_histories = 0;
  std::size_t collision_budget_per_kernel = 0;
  std::size_t total_collision_budget = 0;
  std::size_t total_kernels = 0;
  if (!checked_mul(config.work_items, config.histories_per_item,
                   histories_per_kernel) ||
      !checked_mul(histories_per_kernel, config.scenarios,
                   histories_per_epoch) ||
      !checked_mul(histories_per_epoch, config.epochs, total_histories) ||
      !checked_mul(histories_per_kernel, config.max_collisions,
                   collision_budget_per_kernel) ||
      !checked_mul(total_histories, config.max_collisions,
                   total_collision_budget) ||
      !checked_mul(config.scenarios, config.epochs, total_kernels)) {
    std::cerr << "ERROR work-count overflow\n";
    return 1;
  }

  std::cout << std::setprecision(9);
  std::cout << "CONFIG work_items=" << config.work_items
            << " scenarios=" << config.scenarios
            << " epochs=" << config.epochs
            << " histories_per_item=" << config.histories_per_item
            << " max_collisions=" << config.max_collisions
            << " layers=" << config.layers
            << " split_parts=" << config.split_parts
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " calibrate=" << (config.calibrate ? 1 : 0) << '\n';
  std::cout << "DAG kernels_per_window=" << config.scenarios
            << " max_width=" << config.scenarios
            << " critical_path_levels=1"
            << " total_kernels=" << total_kernels << '\n';
  std::cout << "WORK histories_per_kernel=" << histories_per_kernel
            << " histories_per_epoch=" << histories_per_epoch
            << " total_histories=" << total_histories
            << " max_collision_steps_per_kernel="
            << collision_budget_per_kernel
            << " max_total_collision_steps=" << total_collision_budget
            << '\n';
  std::cout << "MEMORY bytes_per_scenario=" << bytes_per_scenario
            << " total_estimated_bytes=" << total_bytes << '\n';

  if (config.calibrate) {
    return run_calibration(config);
  }

  try {
    const double total_begin = get_time();
    std::vector<std::unique_ptr<ScenarioBuffers>> scenarios;
    scenarios.reserve(config.scenarios);
    for (std::size_t scenario = 0; scenario < config.scenarios; ++scenario) {
      scenarios.push_back(std::make_unique<ScenarioBuffers>(
          scenario, config.work_items, config.layers));
    }
    const double init_end = get_time();
    sycl::queue queue;
    const double queue_end = get_time();

    const double run_begin = get_time();
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
      for (auto &scenario : scenarios) {
        submit_scenario(queue, *scenario, config, epoch);
        if (config.wait_each_kernel) {
          queue.wait();
        }
      }
      if (!config.wait_each_kernel) {
        queue.wait();
      }
    }
    const double run_end = get_time();
    const double host_begin = get_time();
    const Result result = read_result(scenarios, config);
    const double host_end = get_time();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_end - total_begin << '\n';
    std::cout << "TIMING queue_sec=" << queue_end - init_end << '\n';
    std::cout << "TIMING run_sec=" << run_end - run_begin << '\n';
    std::cout << "TIMING host_sec=" << host_end - host_begin << '\n';
    std::cout << "TIMING total_sec=" << host_end - total_begin << '\n';
    std::cout << "RESULT checksum=" << result.checksum
              << " checked_items=" << result.checked_items
              << " sampled_collisions=" << result.sampled_collisions
              << " max_relative_energy_error="
              << result.max_relative_energy_error << '\n';
    std::cout << "VERIFY passed=" << (result.valid ? 1 : 0)
              << " nonfinite=" << result.nonfinite << '\n';
    return config.verify && !result.valid ? 2 : 0;
  } catch (const sycl::exception &exception) {
    std::cerr << "ERROR SYCL " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }
}
