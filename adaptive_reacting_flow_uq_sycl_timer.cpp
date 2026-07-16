// Adaptive reacting-flow patch ensemble with uncertainty/radiation analysis.
//
// One physical step exposes two kinds of parallelism in the same wait window:
//
//   per patch: directional transport(state, next layout) --+
//              detailed chemistry(state, current layout) --+--> combine
//
//   all patches --> ensemble moments --> fused radiation risk
//
// Chemistry integrates the same physical interval in every cell, but uses a
// spatially varying number of adaptive micro-steps. A clustered flame zone is
// deliberately kept contiguous because reacting-flow stiffness is spatially
// localized in real meshes. Alternating directional sweeps also transpose the
// storage layout while preserving the same physical discretization.
//
// The ensemble/radiation tail is strict dim-0 partition-local dataflow. The
// SFINAE helper below calls the experimental SNMD contract when the modified
// runtime provides it and becomes a no-op with a stock SYCL implementation.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
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

class ArfDirectionalTransportKernelV2;
class ArfDetailedChemistryKernelV2;
class ArfLayoutCombineKernelV2;
class ArfEnsembleMoments4Kernel;
class ArfEnsembleMoments8Kernel;
class ArfFusedRadiationRiskKernel;

namespace {

using field_buffer_t = sycl::buffer<float, 2>;

constexpr std::size_t kDefaultRows = 512;
constexpr std::size_t kDefaultCols = 512;
constexpr std::size_t kDefaultPatches = 8;
constexpr std::size_t kDefaultSteps = 32;
constexpr std::size_t kDefaultColdSubsteps = 64;
constexpr std::size_t kDefaultHotSubsteps = 8192;
constexpr std::size_t kDefaultReactionChannels = 512;
constexpr std::size_t kDefaultSpectralSamples = 1024;
constexpr std::size_t kDefaultSplitParts = 4;

enum class Mode { Full, PatchOnly, StatisticsOnly };
enum class StiffnessLayout { Clustered, Distributed, Uniform };
enum class DirectionalLayout { Alternating, Aligned };

struct Config {
  std::size_t rows = kDefaultRows;
  std::size_t cols = kDefaultCols;
  std::size_t patches = kDefaultPatches;
  std::size_t steps = kDefaultSteps;
  std::size_t cold_substeps = kDefaultColdSubsteps;
  std::size_t hot_substeps = kDefaultHotSubsteps;
  std::size_t reaction_channels = kDefaultReactionChannels;
  std::size_t spectral_samples = kDefaultSpectralSamples;
  std::size_t split_parts = kDefaultSplitParts;
  double memory_limit_gib = 0.0;
  Mode mode = Mode::Full;
  StiffnessLayout stiffness_layout = StiffnessLayout::Clustered;
  DirectionalLayout directional_layout = DirectionalLayout::Alternating;
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

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Full:
    return "full";
  case Mode::PatchOnly:
    return "patch-only";
  case Mode::StatisticsOnly:
    return "statistics-only";
  }
  return "unknown";
}

const char *layout_name(StiffnessLayout layout) {
  switch (layout) {
  case StiffnessLayout::Clustered:
    return "clustered";
  case StiffnessLayout::Distributed:
    return "distributed";
  case StiffnessLayout::Uniform:
    return "uniform";
  }
  return "unknown";
}

const char *directional_layout_name(DirectionalLayout layout) {
  return layout == DirectionalLayout::Alternating ? "alternating" : "aligned";
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

bool parse_size(const std::string &text, std::size_t &value) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' || parsed == 0 ||
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
  errno = 0;
  char *end = nullptr;
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
  if (text == "full") {
    mode = Mode::Full;
  } else if (text == "patch-only") {
    mode = Mode::PatchOnly;
  } else if (text == "statistics-only") {
    mode = Mode::StatisticsOnly;
  } else {
    return false;
  }
  return true;
}

bool parse_layout(const std::string &text, StiffnessLayout &layout) {
  if (text == "clustered") {
    layout = StiffnessLayout::Clustered;
  } else if (text == "distributed") {
    layout = StiffnessLayout::Distributed;
  } else if (text == "uniform") {
    layout = StiffnessLayout::Uniform;
  } else {
    return false;
  }
  return true;
}

bool parse_directional_layout(const std::string &text,
                              DirectionalLayout &layout) {
  if (text == "alternating") {
    layout = DirectionalLayout::Alternating;
  } else if (text == "aligned") {
    layout = DirectionalLayout::Aligned;
  } else {
    return false;
  }
  return true;
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --rows <int>                 Patch rows (default 512)\n"
      << "  --cols <int>                 Patch columns (default 512)\n"
      << "  --patches <4|8>              Independent patches (default 8)\n"
      << "  --steps <int>                Physical wait windows (default 32)\n"
      << "  --cold-substeps <int>        Chemistry steps outside flame (default 64)\n"
      << "  --hot-substeps <int>         Chemistry steps in flame (default 8192)\n"
      << "  --reaction-channels <int>    Detailed kinetics channels (default 512)\n"
      << "  --spectral-samples <int>     Radiation quadrature samples (default 1024)\n"
      << "  --split-parts <int>          Validate dim-0 divisibility (default 4)\n"
      << "  --layout <clustered|distributed|uniform>\n"
      << "  --directional-layout <alternating|aligned>\n"
      << "  --mode <full|patch-only|statistics-only>\n"
      << "  --memory-limit-gib <real>    Fail above estimate; 0 is unlimited\n"
      << "  --wait-each-kernel <0|1>     Debug serialization only\n"
      << "  --host-read-full <0|1>       Validate every cell (default 0)\n"
      << "  --verify <0|1>               Enable output checks (default 1)\n";
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
    if (option == "--rows") {
      parsed = parse_size(value, config.rows);
    } else if (option == "--cols") {
      parsed = parse_size(value, config.cols);
    } else if (option == "--patches") {
      parsed = parse_size(value, config.patches);
    } else if (option == "--steps") {
      parsed = parse_size(value, config.steps);
    } else if (option == "--cold-substeps") {
      parsed = parse_size(value, config.cold_substeps);
    } else if (option == "--hot-substeps") {
      parsed = parse_size(value, config.hot_substeps);
    } else if (option == "--reaction-channels") {
      parsed = parse_size(value, config.reaction_channels);
    } else if (option == "--spectral-samples") {
      parsed = parse_size(value, config.spectral_samples);
    } else if (option == "--split-parts") {
      parsed = parse_size(value, config.split_parts);
    } else if (option == "--memory-limit-gib") {
      parsed = parse_double(value, config.memory_limit_gib);
    } else if (option == "--layout") {
      parsed = parse_layout(value, config.stiffness_layout);
    } else if (option == "--directional-layout") {
      parsed = parse_directional_layout(value, config.directional_layout);
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
      error = "invalid value for " + option + ": " + value;
      return false;
    }
  }
  return true;
}

template <typename HandlerT, typename AccessorT>
auto mark_partition_local_impl(HandlerT &handler, AccessorT accessor, int)
    -> decltype(handler.ext_snmd_partition_local(accessor), void()) {
  handler.ext_snmd_partition_local(accessor);
}

template <typename HandlerT, typename AccessorT>
void mark_partition_local_impl(HandlerT &, AccessorT, long) {}

template <typename HandlerT, typename AccessorT>
void mark_partition_local(HandlerT &handler, AccessorT accessor) {
  mark_partition_local_impl(handler, accessor, 0);
}

inline float transport_rhs(float center, float up, float down, float left,
                           float right, float diffusion, float velocity_x,
                           float velocity_y) {
  const float lap = up + down + left + right - 4.0f * center;
  const float grad_x = 0.5f * (right - left);
  const float grad_y = 0.5f * (down - up);
  return diffusion * lap - velocity_x * grad_x - velocity_y * grad_y;
}

bool is_hot_cell(std::size_t row, std::size_t col, const Config &config) {
  if (config.stiffness_layout == StiffnessLayout::Uniform) {
    return false;
  }
  if (config.stiffness_layout == StiffnessLayout::Clustered) {
    return row < config.rows / 4 && col < config.cols / 4;
  }
  // A tiled distribution keeps exactly one sixteenth of cells hot while
  // spreading compact flamelets across every large geometric chunk.
  const std::size_t tile = std::max<std::size_t>(
      1, std::min<std::size_t>(32, std::min(config.rows, config.cols) / 8));
  return (5 * (row / tile) + 3 * (col / tile)) % 16 == 0;
}

std::size_t chemistry_substeps_for_cell(std::size_t row, std::size_t col,
                                        const Config &config) {
  if (config.stiffness_layout != StiffnessLayout::Uniform) {
    return is_hot_cell(row, col, config) ? config.hot_substeps
                                         : config.cold_substeps;
  }
  // Preserve the exact total work of the 1/16-hot layouts while making every
  // cell's cost equal to within one micro-step.
  const std::size_t total_for_sixteen =
      config.hot_substeps + 15 * config.cold_substeps;
  const std::size_t base = total_for_sixteen / 16;
  const std::size_t remainder = total_for_sixteen % 16;
  const std::size_t linear_mod = (row * config.cols + col) % 16;
  return base + (linear_mod < remainder ? 1 : 0);
}

struct PatchBuffers {
  std::size_t patch_id;
  std::size_t rows;
  std::size_t cols;

  field_buffer_t temperature0;
  field_buffer_t temperature1;
  field_buffer_t fuel0;
  field_buffer_t fuel1;
  field_buffer_t oxidizer0;
  field_buffer_t oxidizer1;
  field_buffer_t product0;
  field_buffer_t product1;

  field_buffer_t transport_temperature;
  field_buffer_t transport_fuel;
  field_buffer_t transport_oxidizer;
  field_buffer_t transport_product;
  field_buffer_t chemistry_temperature;
  field_buffer_t chemistry_fuel;
  field_buffer_t chemistry_oxidizer;
  field_buffer_t chemistry_product;
  sycl::buffer<std::uint16_t, 2> chemistry_steps0;
  sycl::buffer<std::uint16_t, 2> chemistry_steps1;

  PatchBuffers(std::size_t id, const Config &config)
      : patch_id(id), rows(config.rows), cols(config.cols),
        temperature0(sycl::range<2>(rows, cols)),
        temperature1(sycl::range<2>(rows, cols)),
        fuel0(sycl::range<2>(rows, cols)),
        fuel1(sycl::range<2>(rows, cols)),
        oxidizer0(sycl::range<2>(rows, cols)),
        oxidizer1(sycl::range<2>(rows, cols)),
        product0(sycl::range<2>(rows, cols)),
        product1(sycl::range<2>(rows, cols)),
        transport_temperature(sycl::range<2>(rows, cols)),
        transport_fuel(sycl::range<2>(rows, cols)),
        transport_oxidizer(sycl::range<2>(rows, cols)),
        transport_product(sycl::range<2>(rows, cols)),
        chemistry_temperature(sycl::range<2>(rows, cols)),
        chemistry_fuel(sycl::range<2>(rows, cols)),
        chemistry_oxidizer(sycl::range<2>(rows, cols)),
        chemistry_product(sycl::range<2>(rows, cols)),
        chemistry_steps0(sycl::range<2>(rows, cols)),
        chemistry_steps1(sycl::range<2>(rows, cols)) {
    initialize(config);
  }

  field_buffer_t &temperature(std::size_t index) {
    return index == 0 ? temperature0 : temperature1;
  }
  field_buffer_t &fuel(std::size_t index) {
    return index == 0 ? fuel0 : fuel1;
  }
  field_buffer_t &oxidizer(std::size_t index) {
    return index == 0 ? oxidizer0 : oxidizer1;
  }
  field_buffer_t &product(std::size_t index) {
    return index == 0 ? product0 : product1;
  }
  sycl::buffer<std::uint16_t, 2> &chemistry_steps(std::size_t orientation) {
    return orientation == 0 ? chemistry_steps0 : chemistry_steps1;
  }

private:
  void initialize(const Config &config) {
    auto t0 = temperature0.get_host_access(sycl::write_only);
    auto t1 = temperature1.get_host_access(sycl::write_only);
    auto f0 = fuel0.get_host_access(sycl::write_only);
    auto f1 = fuel1.get_host_access(sycl::write_only);
    auto o0 = oxidizer0.get_host_access(sycl::write_only);
    auto o1 = oxidizer1.get_host_access(sycl::write_only);
    auto p0 = product0.get_host_access(sycl::write_only);
    auto p1 = product1.get_host_access(sycl::write_only);
    auto tr_t = transport_temperature.get_host_access(sycl::write_only);
    auto tr_f = transport_fuel.get_host_access(sycl::write_only);
    auto tr_o = transport_oxidizer.get_host_access(sycl::write_only);
    auto tr_p = transport_product.get_host_access(sycl::write_only);
    auto ch_t = chemistry_temperature.get_host_access(sycl::write_only);
    auto ch_f = chemistry_fuel.get_host_access(sycl::write_only);
    auto ch_o = chemistry_oxidizer.get_host_access(sycl::write_only);
    auto ch_p = chemistry_product.get_host_access(sycl::write_only);
    auto substeps0 = chemistry_steps0.get_host_access(sycl::write_only);
    auto substeps1 = chemistry_steps1.get_host_access(sycl::write_only);

    const float patch_phase = 0.17f * static_cast<float>(patch_id);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        const sycl::id<2> id(row, col);
        const float x = static_cast<float>(col) / static_cast<float>(cols);
        const float y = static_cast<float>(row) / static_cast<float>(rows);
        const bool hot = is_hot_cell(row, col, config);
        const float wave = std::sin(6.28318530718f * (x + patch_phase)) *
                           std::cos(6.28318530718f * (y - patch_phase));
        const float flame = hot ? 1.0f : 0.0f;
        t0[id] = 0.82f + 0.03f * wave + 0.95f * flame;
        f0[id] = 0.82f - 0.18f * flame + 0.02f * wave;
        o0[id] = 1.00f - 0.08f * flame - 0.01f * wave;
        p0[id] = 0.02f + 0.12f * flame;
        t1[id] = f1[id] = o1[id] = p1[id] = 0.0f;
        tr_t[id] = tr_f[id] = tr_o[id] = tr_p[id] = 0.0f;
        ch_t[id] = ch_f[id] = ch_o[id] = ch_p[id] = 0.0f;
        const std::size_t steps =
            chemistry_substeps_for_cell(row, col, config);
        substeps0[id] = static_cast<std::uint16_t>(steps);
        substeps1[sycl::id<2>(col, row)] =
            static_cast<std::uint16_t>(steps);
      }
    }
  }
};

struct MechanismBuffers {
  sycl::buffer<float, 1> forward_activation;
  sycl::buffer<float, 1> backward_activation;
  sycl::buffer<float, 1> forward_weight;
  sycl::buffer<float, 1> backward_weight;

  explicit MechanismBuffers(const Config &config)
      : forward_activation(sycl::range<1>(config.reaction_channels)),
        backward_activation(sycl::range<1>(config.reaction_channels)),
        forward_weight(sycl::range<1>(config.reaction_channels)),
        backward_weight(sycl::range<1>(config.reaction_channels)) {
    auto forward_e = forward_activation.get_host_access(sycl::write_only);
    auto backward_e = backward_activation.get_host_access(sycl::write_only);
    auto forward_w = forward_weight.get_host_access(sycl::write_only);
    auto backward_w = backward_weight.get_host_access(sycl::write_only);
    double forward_sum = 0.0;
    double backward_sum = 0.0;
    for (std::size_t channel = 0; channel < config.reaction_channels;
         ++channel) {
      const float x = (static_cast<float>(channel) + 0.5f) /
                      static_cast<float>(config.reaction_channels);
      forward_e[channel] = 4.2f + 1.2f * x;
      backward_e[channel] = 0.8f + 0.6f * x;
      forward_w[channel] = 0.9f + 0.1f * std::sin(6.28318530718f * x);
      backward_w[channel] = 0.9f + 0.1f * std::cos(6.28318530718f * x);
      forward_sum += forward_w[channel];
      backward_sum += backward_w[channel];
    }
    for (std::size_t channel = 0; channel < config.reaction_channels;
         ++channel) {
      forward_w[channel] /= static_cast<float>(forward_sum);
      backward_w[channel] /= static_cast<float>(backward_sum);
    }
  }
};

struct StatisticsBuffers {
  field_buffer_t mean;
  field_buffer_t variance;
  field_buffer_t risk;
  sycl::buffer<float, 1> spectral_weight;
  sycl::buffer<float, 1> spectral_kappa;

  explicit StatisticsBuffers(const Config &config)
      : mean(sycl::range<2>(config.rows, config.cols)),
        variance(sycl::range<2>(config.rows, config.cols)),
        risk(sycl::range<2>(config.rows, config.cols)),
        spectral_weight(sycl::range<1>(config.spectral_samples)),
        spectral_kappa(sycl::range<1>(config.spectral_samples)) {
    auto weights = spectral_weight.get_host_access(sycl::write_only);
    auto kappas = spectral_kappa.get_host_access(sycl::write_only);
    for (std::size_t group = 0; group < config.spectral_samples; ++group) {
      const float x = (static_cast<float>(group) + 0.5f) /
                      static_cast<float>(config.spectral_samples);
      weights[group] =
          (0.65f + 0.35f * std::sin(3.14159265359f * x)) /
          static_cast<float>(config.spectral_samples);
      kappas[group] = 0.015f + 2.75f * x * x;
    }
  }
};

void submit_transport(sycl::queue &queue, PatchBuffers &patch,
                      std::size_t current, DirectionalLayout layout) {
  const std::size_t rows = patch.rows;
  const std::size_t cols = patch.cols;
  const float velocity_x = 0.18f + 0.01f * static_cast<float>(patch.patch_id);
  const float velocity_y = -0.11f + 0.008f * static_cast<float>(patch.patch_id);
  const float diffusion = 0.025f;
  const bool transpose = layout == DirectionalLayout::Alternating;
  const bool axes_swapped = transpose && current == 1;

  queue.submit([&](sycl::handler &cgh) {
    auto temperature =
        patch.temperature(current).get_access<sycl::access::mode::read>(cgh);
    auto fuel = patch.fuel(current).get_access<sycl::access::mode::read>(cgh);
    auto oxidizer =
        patch.oxidizer(current).get_access<sycl::access::mode::read>(cgh);
    auto product =
        patch.product(current).get_access<sycl::access::mode::read>(cgh);
    auto out_t = patch.transport_temperature
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_f = patch.transport_fuel
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_o = patch.transport_oxidizer
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_p = patch.transport_product
                     .get_access<sycl::access::mode::discard_write>(cgh);
    mark_partition_local(cgh, out_t);
    mark_partition_local(cgh, out_f);
    mark_partition_local(cgh, out_o);
    mark_partition_local(cgh, out_p);

    cgh.parallel_for<ArfDirectionalTransportKernelV2>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
          // Alternating directional sweeps write the next state in transposed
          // storage. This makes the swept dimension contiguous without a
          // separate materialized-transpose kernel.
          const std::size_t row = transpose ? id[1] : id[0];
          const std::size_t col = transpose ? id[0] : id[1];
          // Zero-normal-gradient patch boundaries keep the exact read region
          // representable by Celerity's clamped neighborhood mapper.
          const std::size_t up = row == 0 ? 0 : row - 1;
          const std::size_t down = row + 1 == rows ? row : row + 1;
          const std::size_t left = col == 0 ? 0 : col - 1;
          const std::size_t right = col + 1 == cols ? col : col + 1;

          const sycl::id<2> source(row, col);
          const sycl::id<2> id_up =
              axes_swapped ? sycl::id<2>(row, left) : sycl::id<2>(up, col);
          const sycl::id<2> id_down =
              axes_swapped ? sycl::id<2>(row, right)
                           : sycl::id<2>(down, col);
          const sycl::id<2> id_left =
              axes_swapped ? sycl::id<2>(up, col)
                           : sycl::id<2>(row, left);
          const sycl::id<2> id_right =
              axes_swapped ? sycl::id<2>(down, col)
                           : sycl::id<2>(row, right);

          out_t[id] = transport_rhs(
              temperature[source], temperature[id_up], temperature[id_down],
              temperature[id_left], temperature[id_right], diffusion,
              velocity_x, velocity_y);
          out_f[id] = transport_rhs(fuel[source], fuel[id_up], fuel[id_down],
                                    fuel[id_left], fuel[id_right], diffusion,
                                    velocity_x, velocity_y);
          out_o[id] = transport_rhs(
              oxidizer[source], oxidizer[id_up], oxidizer[id_down],
              oxidizer[id_left], oxidizer[id_right], diffusion, velocity_x,
              velocity_y);
          out_p[id] = transport_rhs(
              product[source], product[id_up], product[id_down],
              product[id_left], product[id_right], diffusion, velocity_x,
              velocity_y);
        });
  });
}

void submit_chemistry(sycl::queue &queue, PatchBuffers &patch,
                      MechanismBuffers &mechanism, std::size_t current,
                      std::size_t orientation, const Config &config) {
  const std::size_t rows = patch.rows;
  const std::size_t cols = patch.cols;
  const float chemistry_dt = 0.0025f;
  const float rate_scale = 1.0f + 0.035f * static_cast<float>(patch.patch_id);
  const std::size_t channels = config.reaction_channels;

  queue.submit([&](sycl::handler &cgh) {
    auto temperature =
        patch.temperature(current).get_access<sycl::access::mode::read>(cgh);
    auto fuel = patch.fuel(current).get_access<sycl::access::mode::read>(cgh);
    auto oxidizer =
        patch.oxidizer(current).get_access<sycl::access::mode::read>(cgh);
    auto product =
        patch.product(current).get_access<sycl::access::mode::read>(cgh);
    auto substeps = patch.chemistry_steps(orientation)
                        .get_access<sycl::access::mode::read>(cgh);
    auto forward_e = mechanism.forward_activation
                         .get_access<sycl::access::mode::read>(cgh);
    auto backward_e = mechanism.backward_activation
                          .get_access<sycl::access::mode::read>(cgh);
    auto forward_w =
        mechanism.forward_weight.get_access<sycl::access::mode::read>(cgh);
    auto backward_w =
        mechanism.backward_weight.get_access<sycl::access::mode::read>(cgh);
    auto out_t = patch.chemistry_temperature
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_f = patch.chemistry_fuel
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_o = patch.chemistry_oxidizer
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_p = patch.chemistry_product
                     .get_access<sycl::access::mode::discard_write>(cgh);

    mark_partition_local(cgh, temperature);
    mark_partition_local(cgh, fuel);
    mark_partition_local(cgh, oxidizer);
    mark_partition_local(cgh, product);
    mark_partition_local(cgh, substeps);
    mark_partition_local(cgh, out_t);
    mark_partition_local(cgh, out_f);
    mark_partition_local(cgh, out_o);
    mark_partition_local(cgh, out_p);

    cgh.parallel_for<ArfDetailedChemistryKernelV2>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
          float temp = temperature[id];
          float f = fuel[id];
          float o = oxidizer[id];
          float p = product[id];
          const std::uint32_t steps =
              static_cast<std::uint32_t>(substeps[id]);
          const float dt = chemistry_dt / static_cast<float>(steps);

          // Reduced reversible Arrhenius kinetics. More micro-steps improve
          // stability in the hot zone while advancing the same macro interval.
          for (std::uint32_t step = 0; step < steps; ++step) {
            const float inv_temp = 1.0f / (temp + 0.20f);
            float forward_kernel = 0.0f;
            float backward_kernel = 0.0f;
            for (std::size_t channel = 0; channel < channels; ++channel) {
              forward_kernel += forward_w[channel] *
                                sycl::exp(-forward_e[channel] * inv_temp);
              backward_kernel += backward_w[channel] *
                                 sycl::exp(-backward_e[channel] * inv_temp);
            }
            const float forward_rate =
                rate_scale * 24.0f * forward_kernel * f * o;
            const float backward_rate = 0.035f * backward_kernel * p;
            float delta = dt * (forward_rate - backward_rate);
            const float positive_cap = 0.12f * sycl::fmin(f, 2.0f * o);
            const float negative_cap = 0.10f * p;
            delta = sycl::fmin(positive_cap,
                               sycl::fmax(-negative_cap, delta));
            f = sycl::fmax(0.0f, f - delta);
            o = sycl::fmax(0.0f, o - 0.5f * delta);
            p = sycl::fmax(0.0f, p + 1.5f * delta);
            temp += 3.8f * delta - dt * 0.07f * (temp - 0.80f);
          }

          out_t[id] = temp;
          out_f[id] = f;
          out_o[id] = o;
          out_p[id] = p;
        });
  });
}

void submit_combine(sycl::queue &queue, PatchBuffers &patch,
                    std::size_t next, DirectionalLayout layout) {
  const std::size_t rows = patch.rows;
  const std::size_t cols = patch.cols;
  const float transport_dt = 0.0015f;
  const bool transpose = layout == DirectionalLayout::Alternating;

  queue.submit([&](sycl::handler &cgh) {
    auto tr_t = patch.transport_temperature
                    .get_access<sycl::access::mode::read>(cgh);
    auto tr_f =
        patch.transport_fuel.get_access<sycl::access::mode::read>(cgh);
    auto tr_o = patch.transport_oxidizer
                    .get_access<sycl::access::mode::read>(cgh);
    auto tr_p =
        patch.transport_product.get_access<sycl::access::mode::read>(cgh);
    auto ch_t = patch.chemistry_temperature
                    .get_access<sycl::access::mode::read>(cgh);
    auto ch_f =
        patch.chemistry_fuel.get_access<sycl::access::mode::read>(cgh);
    auto ch_o = patch.chemistry_oxidizer
                    .get_access<sycl::access::mode::read>(cgh);
    auto ch_p =
        patch.chemistry_product.get_access<sycl::access::mode::read>(cgh);
    auto out_t = patch.temperature(next)
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_f = patch.fuel(next)
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_o = patch.oxidizer(next)
                     .get_access<sycl::access::mode::discard_write>(cgh);
    auto out_p = patch.product(next)
                     .get_access<sycl::access::mode::discard_write>(cgh);

    mark_partition_local(cgh, tr_t);
    mark_partition_local(cgh, tr_f);
    mark_partition_local(cgh, tr_o);
    mark_partition_local(cgh, tr_p);
    if (!transpose) {
      mark_partition_local(cgh, ch_t);
      mark_partition_local(cgh, ch_f);
      mark_partition_local(cgh, ch_o);
      mark_partition_local(cgh, ch_p);
    }
    mark_partition_local(cgh, out_t);
    mark_partition_local(cgh, out_f);
    mark_partition_local(cgh, out_o);
    mark_partition_local(cgh, out_p);

    cgh.parallel_for<ArfLayoutCombineKernelV2>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
          const sycl::id<2> chemistry_id =
              transpose ? sycl::id<2>(id[1], id[0]) : id;
          out_t[id] = sycl::fmin(4.0f,
                                 sycl::fmax(0.2f, ch_t[chemistry_id] +
                                                     transport_dt * tr_t[id]));
          out_f[id] = sycl::fmax(
              0.0f, ch_f[chemistry_id] + transport_dt * tr_f[id]);
          out_o[id] = sycl::fmax(
              0.0f, ch_o[chemistry_id] + transport_dt * tr_o[id]);
          out_p[id] = sycl::fmax(
              0.0f, ch_p[chemistry_id] + transport_dt * tr_p[id]);
        });
  });
}

void submit_moments4(sycl::queue &queue,
                     std::vector<std::unique_ptr<PatchBuffers>> &patches,
                     StatisticsBuffers &statistics, std::size_t state) {
  const std::size_t rows = patches[0]->rows;
  const std::size_t cols = patches[0]->cols;
  queue.submit([&](sycl::handler &cgh) {
    auto t0 = patches[0]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t1 = patches[1]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t2 = patches[2]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t3 = patches[3]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto mean = statistics.mean
                    .get_access<sycl::access::mode::discard_write>(cgh);
    auto variance = statistics.variance
                        .get_access<sycl::access::mode::discard_write>(cgh);
    mark_partition_local(cgh, t0);
    mark_partition_local(cgh, t1);
    mark_partition_local(cgh, t2);
    mark_partition_local(cgh, t3);
    mark_partition_local(cgh, mean);
    mark_partition_local(cgh, variance);
    cgh.parallel_for<ArfEnsembleMoments4Kernel>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
          const float sum = t0[id] + t1[id] + t2[id] + t3[id];
          const float avg = 0.25f * sum;
          const float d0 = t0[id] - avg;
          const float d1 = t1[id] - avg;
          const float d2 = t2[id] - avg;
          const float d3 = t3[id] - avg;
          mean[id] = avg;
          variance[id] = 0.25f *
                         (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
        });
  });
}

void submit_moments8(sycl::queue &queue,
                     std::vector<std::unique_ptr<PatchBuffers>> &patches,
                     StatisticsBuffers &statistics, std::size_t state) {
  const std::size_t rows = patches[0]->rows;
  const std::size_t cols = patches[0]->cols;
  queue.submit([&](sycl::handler &cgh) {
    auto t0 = patches[0]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t1 = patches[1]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t2 = patches[2]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t3 = patches[3]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t4 = patches[4]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t5 = patches[5]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t6 = patches[6]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto t7 = patches[7]->temperature(state)
                  .get_access<sycl::access::mode::read>(cgh);
    auto mean = statistics.mean
                    .get_access<sycl::access::mode::discard_write>(cgh);
    auto variance = statistics.variance
                        .get_access<sycl::access::mode::discard_write>(cgh);
    mark_partition_local(cgh, t0);
    mark_partition_local(cgh, t1);
    mark_partition_local(cgh, t2);
    mark_partition_local(cgh, t3);
    mark_partition_local(cgh, t4);
    mark_partition_local(cgh, t5);
    mark_partition_local(cgh, t6);
    mark_partition_local(cgh, t7);
    mark_partition_local(cgh, mean);
    mark_partition_local(cgh, variance);
    cgh.parallel_for<ArfEnsembleMoments8Kernel>(
        sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
          const float sum = t0[id] + t1[id] + t2[id] + t3[id] + t4[id] +
                            t5[id] + t6[id] + t7[id];
          const float avg = 0.125f * sum;
          const float d0 = t0[id] - avg;
          const float d1 = t1[id] - avg;
          const float d2 = t2[id] - avg;
          const float d3 = t3[id] - avg;
          const float d4 = t4[id] - avg;
          const float d5 = t5[id] - avg;
          const float d6 = t6[id] - avg;
          const float d7 = t7[id] - avg;
          mean[id] = avg;
          variance[id] =
              0.125f * (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 +
                        d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7);
        });
  });
}

void submit_radiation_risk(sycl::queue &queue,
                           StatisticsBuffers &statistics,
                           const Config &config) {
  const std::size_t samples = config.spectral_samples;
  queue.submit([&](sycl::handler &cgh) {
    auto mean = statistics.mean.get_access<sycl::access::mode::read>(cgh);
    auto variance =
        statistics.variance.get_access<sycl::access::mode::read>(cgh);
    auto weights =
        statistics.spectral_weight.get_access<sycl::access::mode::read>(cgh);
    auto kappas =
        statistics.spectral_kappa.get_access<sycl::access::mode::read>(cgh);
    auto risk = statistics.risk
                    .get_access<sycl::access::mode::discard_write>(cgh);
    mark_partition_local(cgh, mean);
    mark_partition_local(cgh, variance);
    mark_partition_local(cgh, risk);
    cgh.parallel_for<ArfFusedRadiationRiskKernel>(
        sycl::range<2>(config.rows, config.cols), [=](sycl::id<2> id) {
          const float temp = mean[id];
          const float var = variance[id];
          float opacity = 0.0f;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const float kappa = kappas[sample];
            const float planck = sycl::exp(-kappa / (temp + 0.25f));
            opacity += weights[sample] * planck * (1.0f + kappa * var);
          }
          float radiance = 0.0f;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const float kappa = kappas[sample];
            const float transmission = sycl::exp(-kappa * (0.12f + var));
            radiance += weights[sample] * (opacity + temp) *
                        (1.0f - transmission) / (0.05f + kappa);
          }
          float integral = 0.0f;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const float kappa = kappas[sample];
            const float weighted =
                weights[sample] * radiance / (1.0f + kappa * var);
            integral += sycl::sqrt(sycl::fabs(weighted) + 1.0e-12f) *
                        (0.5f + opacity * kappa);
          }
          risk[id] = integral;
        });
  });
}

struct Result {
  double checksum = 0.0;
  float min_temperature = std::numeric_limits<float>::infinity();
  float max_temperature = -std::numeric_limits<float>::infinity();
  float min_species = std::numeric_limits<float>::infinity();
  float min_risk = std::numeric_limits<float>::infinity();
  std::size_t checked_cells = 0;
  std::size_t nonfinite = 0;
  bool valid = true;
};

Result read_result(std::vector<std::unique_ptr<PatchBuffers>> &patches,
                   StatisticsBuffers &statistics, const Config &config,
                   std::size_t final_state) {
  Result result;
  const std::size_t total_cells = config.rows * config.cols;
  const std::size_t checks =
      config.host_read_full ? total_cells : std::min<std::size_t>(64, total_cells);

  if (config.mode != Mode::StatisticsOnly) {
    for (auto &patch_ptr : patches) {
      auto temperature = patch_ptr->temperature(final_state)
                             .get_host_access(sycl::read_only);
      auto fuel = patch_ptr->fuel(final_state).get_host_access(sycl::read_only);
      auto oxidizer =
          patch_ptr->oxidizer(final_state).get_host_access(sycl::read_only);
      auto product =
          patch_ptr->product(final_state).get_host_access(sycl::read_only);
      for (std::size_t sample = 0; sample < checks; ++sample) {
        const std::size_t linear = config.host_read_full
                                       ? sample
                                       : sample * total_cells / checks;
        const sycl::id<2> id(linear / config.cols, linear % config.cols);
        const float t = temperature[id];
        const float f = fuel[id];
        const float o = oxidizer[id];
        const float p = product[id];
        const bool finite = std::isfinite(t) && std::isfinite(f) &&
                            std::isfinite(o) && std::isfinite(p);
        if (!finite) {
          ++result.nonfinite;
        }
        result.valid = result.valid && finite && t >= 0.2f && t <= 4.0f &&
                       f >= 0.0f && o >= 0.0f && p >= 0.0f;
        result.min_temperature = std::min(result.min_temperature, t);
        result.max_temperature = std::max(result.max_temperature, t);
        result.min_species = std::min(result.min_species, std::min(f, std::min(o, p)));
        result.checksum += (1.0 + 0.01 * patch_ptr->patch_id) *
                           (t + 0.7 * f + 0.5 * o + 0.3 * p);
        ++result.checked_cells;
      }
    }
  }

  if (config.mode != Mode::PatchOnly) {
    auto risk = statistics.risk.get_host_access(sycl::read_only);
    for (std::size_t sample = 0; sample < checks; ++sample) {
      const std::size_t linear =
          config.host_read_full ? sample : sample * total_cells / checks;
      const sycl::id<2> id(linear / config.cols, linear % config.cols);
      const float value = risk[id];
      const bool finite = std::isfinite(value);
      if (!finite) {
        ++result.nonfinite;
      }
      result.valid = result.valid && finite && value >= 0.0f;
      result.min_risk = std::min(result.min_risk, value);
      result.checksum += 0.11 * value;
      ++result.checked_cells;
    }
  }

  result.valid = result.valid && result.nonfinite == 0 &&
                 std::isfinite(result.checksum);
  if (config.mode == Mode::StatisticsOnly) {
    result.min_temperature = 0.0f;
    result.max_temperature = 0.0f;
    result.min_species = 0.0f;
  }
  if (config.mode == Mode::PatchOnly) {
    result.min_risk = 0.0f;
  }
  return result;
}

bool estimate_memory(const Config &config, std::size_t &bytes_per_patch,
                     std::size_t &total_bytes) {
  std::size_t cells = 0;
  std::size_t patch_bytes = 0;
  std::size_t all_patch_bytes = 0;
  std::size_t stats_bytes = 0;
  std::size_t coeff_bytes = 0;
  std::size_t mechanism_bytes = 0;
  // 16 float fields plus two orientation-specific uint16 step fields.
  if (!checked_mul(config.rows, config.cols, cells) ||
      !checked_mul(cells, std::size_t{68}, patch_bytes) ||
      !checked_mul(patch_bytes, config.patches, all_patch_bytes) ||
      !checked_mul(cells, std::size_t{3 * sizeof(float)}, stats_bytes) ||
      !checked_mul(config.spectral_samples,
                   std::size_t{2 * sizeof(float)}, coeff_bytes) ||
      !checked_mul(config.reaction_channels,
                   std::size_t{4 * sizeof(float)}, mechanism_bytes) ||
      !checked_add(all_patch_bytes, stats_bytes, total_bytes) ||
      !checked_add(total_bytes, coeff_bytes, total_bytes) ||
      !checked_add(total_bytes, mechanism_bytes, total_bytes)) {
    return false;
  }
  bytes_per_patch = patch_bytes;
  return true;
}

bool validate_config(const Config &config, std::size_t total_bytes,
                     std::string &error) {
  if (config.rows < 4 || config.cols < 4) {
    error = "rows and cols must both be at least 4";
    return false;
  }
  if (config.directional_layout == DirectionalLayout::Alternating &&
      config.rows != config.cols) {
    error = "alternating directional layout requires square patches";
    return false;
  }
  if (config.patches != 4 && config.patches != 8) {
    error = "patches must be 4 or 8 so the exact ensemble fan-in stays static";
    return false;
  }
  if (config.rows % config.split_parts != 0) {
    error = "rows must be divisible by split-parts";
    return false;
  }
  if (config.cold_substeps > std::numeric_limits<std::uint16_t>::max() ||
      config.hot_substeps > std::numeric_limits<std::uint16_t>::max()) {
    error = "chemistry substeps must fit uint16";
    return false;
  }
  if (config.hot_substeps < config.cold_substeps) {
    error = "hot-substeps must be at least cold-substeps";
    return false;
  }
  if (config.memory_limit_gib < 0.0) {
    error = "memory-limit-gib must be nonnegative";
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

  std::size_t bytes_per_patch = 0;
  std::size_t total_bytes = 0;
  if (!estimate_memory(config, bytes_per_patch, total_bytes) ||
      !validate_config(config, total_bytes, error)) {
    std::cerr << "ERROR " << (error.empty() ? "memory estimate overflow" : error)
              << '\n';
    return 1;
  }

  const std::size_t cells = config.rows * config.cols;
  std::size_t hot_cells = 0;
  std::size_t chemistry_steps_per_patch = 0;
  for (std::size_t row = 0; row < config.rows; ++row) {
    for (std::size_t col = 0; col < config.cols; ++col) {
      hot_cells += is_hot_cell(row, col, config) ? 1 : 0;
      chemistry_steps_per_patch +=
          chemistry_substeps_for_cell(row, col, config);
    }
  }
  const std::size_t patch_kernels = config.mode == Mode::StatisticsOnly
                                        ? 0
                                        : 3 * config.patches;
  const std::size_t statistics_kernels = config.mode == Mode::PatchOnly ? 0 : 2;
  const std::size_t kernels_per_window = patch_kernels + statistics_kernels;
  const std::size_t max_width = config.mode == Mode::StatisticsOnly
                                    ? 1
                                    : 2 * config.patches;
  const std::size_t critical_path = config.mode == Mode::Full ? 4 : 2;

  std::cout << std::setprecision(9);
  std::cout << "CONFIG rows=" << config.rows << " cols=" << config.cols
            << " patches=" << config.patches << " steps=" << config.steps
            << " mode=" << mode_name(config.mode)
            << " layout=" << layout_name(config.stiffness_layout)
            << " directional_layout="
            << directional_layout_name(config.directional_layout)
            << " cold_substeps=" << config.cold_substeps
            << " hot_substeps=" << config.hot_substeps
            << " reaction_channels=" << config.reaction_channels
            << " spectral_samples=" << config.spectral_samples
            << " split_parts=" << config.split_parts
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " partition_contract=auto_detect\n";
  std::cout << "DAG kernels_per_window=" << kernels_per_window
            << " max_width=" << max_width
            << " critical_path_levels=" << critical_path
            << " total_kernels=" << kernels_per_window * config.steps << '\n';
  std::cout << "WORK cells_per_patch=" << cells
            << " hot_cells_per_patch=" << hot_cells
            << " chemistry_substeps_per_patch=" << chemistry_steps_per_patch
            << " chemistry_substeps_per_window="
            << chemistry_steps_per_patch * config.patches
            << " reaction_channel_evals_per_window="
            << chemistry_steps_per_patch * config.patches *
                   config.reaction_channels
            << " spectral_quadrature_evals_per_window="
            << cells * config.spectral_samples * 3 << '\n';
  std::cout << "MEMORY bytes_per_patch=" << bytes_per_patch
            << " total_estimated_bytes=" << total_bytes << '\n';

  try {
    const double total_begin = get_time();
    std::vector<std::unique_ptr<PatchBuffers>> patches;
    patches.reserve(config.patches);
    for (std::size_t patch = 0; patch < config.patches; ++patch) {
      patches.push_back(std::make_unique<PatchBuffers>(patch, config));
    }
    MechanismBuffers mechanism(config);
    StatisticsBuffers statistics(config);
    const double init_end = get_time();
    sycl::queue queue;
    const double queue_end = get_time();

    std::size_t current = 0;
    const double run_begin = get_time();
    for (std::size_t step = 0; step < config.steps; ++step) {
      const std::size_t next = 1 - current;

      if (config.mode != Mode::StatisticsOnly) {
        const std::size_t orientation =
            config.directional_layout == DirectionalLayout::Alternating
                ? current
                : 0;
        for (auto &patch : patches) {
          submit_transport(queue, *patch, current,
                           config.directional_layout);
          if (config.wait_each_kernel) {
            queue.wait_and_throw();
          }
        }
        for (auto &patch : patches) {
          submit_chemistry(queue, *patch, mechanism, current, orientation,
                           config);
          if (config.wait_each_kernel) {
            queue.wait_and_throw();
          }
        }
        for (auto &patch : patches) {
          submit_combine(queue, *patch, next, config.directional_layout);
          if (config.wait_each_kernel) {
            queue.wait_and_throw();
          }
        }
      }

      const std::size_t statistics_state =
          config.mode == Mode::StatisticsOnly ? current : next;
      if (config.mode != Mode::PatchOnly) {
        if (config.patches == 4) {
          submit_moments4(queue, patches, statistics, statistics_state);
        } else {
          submit_moments8(queue, patches, statistics, statistics_state);
        }
        if (config.wait_each_kernel) {
          queue.wait_and_throw();
        }
        submit_radiation_risk(queue, statistics, config);
        if (config.wait_each_kernel) {
          queue.wait_and_throw();
        }
      }

      if (!config.wait_each_kernel) {
        queue.wait_and_throw();
      }
      if (config.mode != Mode::StatisticsOnly) {
        current = next;
      }
    }
    const double run_end = get_time();

    const double host_begin = get_time();
    const Result result =
        read_result(patches, statistics, config, current);
    const double host_end = get_time();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "TIMING init_sec=" << init_end - total_begin << '\n';
    std::cout << "TIMING queue_sec=" << queue_end - init_end << '\n';
    std::cout << "TIMING run_sec=" << run_end - run_begin << '\n';
    std::cout << "TIMING host_sec=" << host_end - host_begin << '\n';
    std::cout << "TIMING total_sec=" << host_end - total_begin << '\n';
    std::cout << "RESULT checksum=" << result.checksum
              << " checked_cells=" << result.checked_cells
              << " min_temperature=" << result.min_temperature
              << " max_temperature=" << result.max_temperature
              << " min_species=" << result.min_species
              << " min_risk=" << result.min_risk << '\n';
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
