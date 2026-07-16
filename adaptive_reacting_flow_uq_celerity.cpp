// Celerity 0.6.0 port of adaptive_reacting_flow_uq_sycl_timer.cpp.
//
// Keep numerical kernels and command-line work parameters in lockstep with the
// SYCL version. Every mapper is an exact access contract. The extra controls
// select Celerity's 1-D/2-D split hint and local oversubscription factor.

#include <celerity.h>

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

class ArfCeleTransportKernel;
class ArfCeleChemistryKernel;
class ArfCeleCombineKernel;
class ArfCeleEnsembleMoments4Kernel;
class ArfCeleEnsembleMoments8Kernel;
class ArfCeleSpectralOpacityKernel;
class ArfCeleSpectralSourceKernel;
class ArfCeleSpectralRiskKernel;

namespace {

using field_buffer_t = celerity::buffer<float, 2>;

enum class Mode { Full, PatchOnly, StatisticsOnly };
enum class StiffnessLayout { Clustered, Distributed, Uniform };
enum class SplitShape { OneD, TwoD };

struct Config {
  std::size_t rows = 1024;
  std::size_t cols = 1024;
  std::size_t patches = 8;
  std::size_t steps = 4;
  std::size_t cold_substeps = 8;
  std::size_t hot_substeps = 1024;
  std::size_t spectral_samples = 1024;
  std::size_t split_parts = 4;
  std::size_t oversubscribe = 1;
  double memory_limit_gib = 0.0;
  Mode mode = Mode::Full;
  StiffnessLayout stiffness_layout = StiffnessLayout::Clustered;
  SplitShape split_shape = SplitShape::OneD;
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
  errno = 0;
  char *end = nullptr;
  const double parsed = std::strtod(text.c_str(), &end);
  if (text.empty() || errno != 0 || end == text.c_str() || *end != '\0' ||
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

bool parse_split_shape(const std::string &text, SplitShape &shape) {
  if (text == "1d") {
    shape = SplitShape::OneD;
  } else if (text == "2d") {
    shape = SplitShape::TwoD;
  } else {
    return false;
  }
  return true;
}

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --rows <int> --cols <int> --patches <4|8> --steps <int>\n"
      << "  --cold-substeps <int> --hot-substeps <int>\n"
      << "  --spectral-samples <int> --split-parts <int>\n"
      << "  --layout <clustered|distributed|uniform>\n"
      << "  --mode <full|patch-only|statistics-only>\n"
      << "  --celerity-split <1d|2d>       Celerity split hint (default 1d)\n"
      << "  --oversubscribe <int>           Fine chunks/device (default 1)\n"
      << "  --memory-limit-gib <real> --wait-each-kernel <0|1>\n"
      << "  --host-read-full <0|1> --verify <0|1>\n";
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
    } else if (option == "--spectral-samples") {
      parsed = parse_size(value, config.spectral_samples);
    } else if (option == "--split-parts") {
      parsed = parse_size(value, config.split_parts);
    } else if (option == "--oversubscribe") {
      parsed = parse_size(value, config.oversubscribe);
    } else if (option == "--memory-limit-gib") {
      parsed = parse_double(value, config.memory_limit_gib);
    } else if (option == "--layout") {
      parsed = parse_layout(value, config.stiffness_layout);
    } else if (option == "--mode") {
      parsed = parse_mode(value, config.mode);
    } else if (option == "--celerity-split") {
      parsed = parse_split_shape(value, config.split_shape);
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

bool is_hot_cell(std::size_t row, std::size_t col, const Config &config) {
  if (config.stiffness_layout == StiffnessLayout::Uniform) {
    return false;
  }
  if (config.stiffness_layout == StiffnessLayout::Clustered) {
    return row < config.rows / 4 && col < config.cols / 4;
  }
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
  const std::size_t total_for_sixteen =
      config.hot_substeps + 15 * config.cold_substeps;
  const std::size_t base = total_for_sixteen / 16;
  const std::size_t remainder = total_for_sixteen % 16;
  return base + ((row * config.cols + col) % 16 < remainder ? 1 : 0);
}

inline float transport_rhs(float center, float up, float down, float left,
                           float right, float diffusion, float velocity_x,
                           float velocity_y) {
  const float lap = up + down + left + right - 4.0f * center;
  const float grad_x = 0.5f * (right - left);
  const float grad_y = 0.5f * (down - up);
  return diffusion * lap - velocity_x * grad_x - velocity_y * grad_y;
}

void apply_hints(celerity::handler &cgh, const Config &config) {
  if (config.split_shape == SplitShape::TwoD) {
    celerity::experimental::hint(
        cgh, celerity::experimental::hints::split_2d{});
  } else {
    celerity::experimental::hint(
        cgh, celerity::experimental::hints::split_1d{});
  }
  if (config.oversubscribe > 1) {
    celerity::experimental::hint(
        cgh,
        celerity::experimental::hints::oversubscribe(config.oversubscribe));
  }
}

struct PatchHostData {
  std::vector<float> temperature;
  std::vector<float> fuel;
  std::vector<float> oxidizer;
  std::vector<float> product;
  std::vector<std::uint16_t> chemistry_steps;

  PatchHostData(std::size_t patch_id, const Config &config)
      : temperature(config.rows * config.cols),
        fuel(config.rows * config.cols), oxidizer(config.rows * config.cols),
        product(config.rows * config.cols),
        chemistry_steps(config.rows * config.cols) {
    const float patch_phase = 0.17f * static_cast<float>(patch_id);
    for (std::size_t row = 0; row < config.rows; ++row) {
      for (std::size_t col = 0; col < config.cols; ++col) {
        const std::size_t i = row * config.cols + col;
        const float x = static_cast<float>(col) / static_cast<float>(config.cols);
        const float y = static_cast<float>(row) / static_cast<float>(config.rows);
        const bool hot = is_hot_cell(row, col, config);
        const float wave = std::sin(6.28318530718f * (x + patch_phase)) *
                           std::cos(6.28318530718f * (y - patch_phase));
        const float flame = hot ? 1.0f : 0.0f;
        temperature[i] = 0.82f + 0.03f * wave + 0.95f * flame;
        fuel[i] = 0.82f - 0.18f * flame + 0.02f * wave;
        oxidizer[i] = 1.00f - 0.08f * flame - 0.01f * wave;
        product[i] = 0.02f + 0.12f * flame;
        chemistry_steps[i] = static_cast<std::uint16_t>(
            chemistry_substeps_for_cell(row, col, config));
      }
    }
  }
};

struct PatchBuffers {
  std::size_t patch_id;
  std::size_t rows;
  std::size_t cols;
  PatchHostData host;
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
  celerity::buffer<std::uint16_t, 2> chemistry_steps;

  PatchBuffers(std::size_t id, const Config &config)
      : patch_id(id), rows(config.rows), cols(config.cols), host(id, config),
        temperature0(host.temperature.data(), celerity::range<2>(rows, cols)),
        temperature1(celerity::range<2>(rows, cols)),
        fuel0(host.fuel.data(), celerity::range<2>(rows, cols)),
        fuel1(celerity::range<2>(rows, cols)),
        oxidizer0(host.oxidizer.data(), celerity::range<2>(rows, cols)),
        oxidizer1(celerity::range<2>(rows, cols)),
        product0(host.product.data(), celerity::range<2>(rows, cols)),
        product1(celerity::range<2>(rows, cols)),
        transport_temperature(celerity::range<2>(rows, cols)),
        transport_fuel(celerity::range<2>(rows, cols)),
        transport_oxidizer(celerity::range<2>(rows, cols)),
        transport_product(celerity::range<2>(rows, cols)),
        chemistry_temperature(celerity::range<2>(rows, cols)),
        chemistry_fuel(celerity::range<2>(rows, cols)),
        chemistry_oxidizer(celerity::range<2>(rows, cols)),
        chemistry_product(celerity::range<2>(rows, cols)),
        chemistry_steps(host.chemistry_steps.data(),
                        celerity::range<2>(rows, cols)) {}

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
};

std::vector<float> make_spectral_weights(std::size_t samples) {
  std::vector<float> values(samples);
  for (std::size_t group = 0; group < samples; ++group) {
    const float x = (static_cast<float>(group) + 0.5f) /
                    static_cast<float>(samples);
    values[group] = (0.65f + 0.35f * std::sin(3.14159265359f * x)) /
                    static_cast<float>(samples);
  }
  return values;
}

std::vector<float> make_spectral_kappas(std::size_t samples) {
  std::vector<float> values(samples);
  for (std::size_t group = 0; group < samples; ++group) {
    const float x = (static_cast<float>(group) + 0.5f) /
                    static_cast<float>(samples);
    values[group] = 0.015f + 2.75f * x * x;
  }
  return values;
}

struct StatisticsBuffers {
  std::vector<float> weights_host;
  std::vector<float> kappas_host;
  field_buffer_t mean;
  field_buffer_t variance;
  field_buffer_t opacity;
  field_buffer_t source;
  field_buffer_t risk;
  celerity::buffer<float, 1> spectral_weight;
  celerity::buffer<float, 1> spectral_kappa;

  explicit StatisticsBuffers(const Config &config)
      : weights_host(make_spectral_weights(config.spectral_samples)),
        kappas_host(make_spectral_kappas(config.spectral_samples)),
        mean(celerity::range<2>(config.rows, config.cols)),
        variance(celerity::range<2>(config.rows, config.cols)),
        opacity(celerity::range<2>(config.rows, config.cols)),
        source(celerity::range<2>(config.rows, config.cols)),
        risk(celerity::range<2>(config.rows, config.cols)),
        spectral_weight(weights_host.data(),
                        celerity::range<1>(config.spectral_samples)),
        spectral_kappa(kappas_host.data(),
                       celerity::range<1>(config.spectral_samples)) {}
};

void submit_transport(celerity::distr_queue &queue, PatchBuffers &patch,
                      std::size_t current, const Config &config) {
  const std::size_t rows = patch.rows;
  const std::size_t cols = patch.cols;
  const float velocity_x = 0.18f + 0.01f * static_cast<float>(patch.patch_id);
  const float velocity_y = -0.11f + 0.008f * static_cast<float>(patch.patch_id);
  const float diffusion = 0.025f;
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor temperature{patch.temperature(current), cgh,
                                   celerity::access::neighborhood(1, 1),
                                   celerity::read_only};
    celerity::accessor fuel{patch.fuel(current), cgh,
                            celerity::access::neighborhood(1, 1),
                            celerity::read_only};
    celerity::accessor oxidizer{patch.oxidizer(current), cgh,
                                celerity::access::neighborhood(1, 1),
                                celerity::read_only};
    celerity::accessor product{patch.product(current), cgh,
                               celerity::access::neighborhood(1, 1),
                               celerity::read_only};
    celerity::accessor out_t{patch.transport_temperature, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_f{patch.transport_fuel, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_o{patch.transport_oxidizer, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_p{patch.transport_product, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleTransportKernel>(
        celerity::range<2>(rows, cols), [=](celerity::item<2> item) {
          const auto id = item.get_id();
          const std::size_t row = id[0];
          const std::size_t col = id[1];
          const std::size_t up = row == 0 ? 0 : row - 1;
          const std::size_t down = row + 1 == rows ? row : row + 1;
          const std::size_t left = col == 0 ? 0 : col - 1;
          const std::size_t right = col + 1 == cols ? col : col + 1;
          const celerity::id<2> iu(up, col), idn(down, col), il(row, left),
              ir(row, right);
          out_t[id] = transport_rhs(temperature[id], temperature[iu],
                                    temperature[idn], temperature[il],
                                    temperature[ir], diffusion, velocity_x,
                                    velocity_y);
          out_f[id] = transport_rhs(fuel[id], fuel[iu], fuel[idn], fuel[il],
                                    fuel[ir], diffusion, velocity_x,
                                    velocity_y);
          out_o[id] = transport_rhs(oxidizer[id], oxidizer[iu], oxidizer[idn],
                                    oxidizer[il], oxidizer[ir], diffusion,
                                    velocity_x, velocity_y);
          out_p[id] = transport_rhs(product[id], product[iu], product[idn],
                                    product[il], product[ir], diffusion,
                                    velocity_x, velocity_y);
        });
  });
}

void submit_chemistry(celerity::distr_queue &queue, PatchBuffers &patch,
                      std::size_t current, const Config &config) {
  const float chemistry_dt = 0.0025f;
  const float rate_scale = 1.0f + 0.035f * static_cast<float>(patch.patch_id);
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor temperature{patch.temperature(current), cgh,
                                   celerity::access::one_to_one{},
                                   celerity::read_only};
    celerity::accessor fuel{patch.fuel(current), cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor oxidizer{patch.oxidizer(current), cgh,
                                celerity::access::one_to_one{},
                                celerity::read_only};
    celerity::accessor product{patch.product(current), cgh,
                               celerity::access::one_to_one{},
                               celerity::read_only};
    celerity::accessor substeps{patch.chemistry_steps, cgh,
                                celerity::access::one_to_one{},
                                celerity::read_only};
    celerity::accessor out_t{patch.chemistry_temperature, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_f{patch.chemistry_fuel, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_o{patch.chemistry_oxidizer, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_p{patch.chemistry_product, cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleChemistryKernel>(
        celerity::range<2>(patch.rows, patch.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          float temp = temperature[id];
          float f = fuel[id];
          float o = oxidizer[id];
          float p = product[id];
          const std::uint32_t steps =
              static_cast<std::uint32_t>(substeps[id]);
          const float dt = chemistry_dt / static_cast<float>(steps);
          for (std::uint32_t step = 0; step < steps; ++step) {
            const float inv_temp = 1.0f / (temp + 0.20f);
            const float forward_rate =
                rate_scale * 24.0f * sycl::exp(-4.8f * inv_temp) * f * o;
            const float backward_rate =
                0.035f * sycl::exp(-1.1f * inv_temp) * p;
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

void submit_combine(celerity::distr_queue &queue, PatchBuffers &patch,
                    std::size_t next, const Config &config) {
  const float transport_dt = 0.0015f;
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor tr_t{patch.transport_temperature, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor tr_f{patch.transport_fuel, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor tr_o{patch.transport_oxidizer, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor tr_p{patch.transport_product, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor ch_t{patch.chemistry_temperature, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor ch_f{patch.chemistry_fuel, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor ch_o{patch.chemistry_oxidizer, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor ch_p{patch.chemistry_product, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor out_t{patch.temperature(next), cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_f{patch.fuel(next), cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_o{patch.oxidizer(next), cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    celerity::accessor out_p{patch.product(next), cgh,
                             celerity::access::one_to_one{},
                             celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleCombineKernel>(
        celerity::range<2>(patch.rows, patch.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          out_t[id] = sycl::fmin(4.0f,
                                 sycl::fmax(0.2f, ch_t[id] +
                                                     transport_dt * tr_t[id]));
          out_f[id] = sycl::fmax(0.0f, ch_f[id] + transport_dt * tr_f[id]);
          out_o[id] = sycl::fmax(0.0f, ch_o[id] + transport_dt * tr_o[id]);
          out_p[id] = sycl::fmax(0.0f, ch_p[id] + transport_dt * tr_p[id]);
        });
  });
}

void submit_moments4(celerity::distr_queue &queue,
                     std::vector<std::unique_ptr<PatchBuffers>> &patches,
                     StatisticsBuffers &statistics, std::size_t state,
                     const Config &config) {
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor t0{patches[0]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t1{patches[1]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t2{patches[2]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t3{patches[3]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor mean{statistics.mean, cgh,
                            celerity::access::one_to_one{},
                            celerity::write_only, celerity::no_init};
    celerity::accessor variance{statistics.variance, cgh,
                                celerity::access::one_to_one{},
                                celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleEnsembleMoments4Kernel>(
        celerity::range<2>(config.rows, config.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          const float avg = 0.25f * (t0[id] + t1[id] + t2[id] + t3[id]);
          const float d0 = t0[id] - avg, d1 = t1[id] - avg;
          const float d2 = t2[id] - avg, d3 = t3[id] - avg;
          mean[id] = avg;
          variance[id] = 0.25f *
                         (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3);
        });
  });
}

void submit_moments8(celerity::distr_queue &queue,
                     std::vector<std::unique_ptr<PatchBuffers>> &patches,
                     StatisticsBuffers &statistics, std::size_t state,
                     const Config &config) {
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor t0{patches[0]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t1{patches[1]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t2{patches[2]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t3{patches[3]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t4{patches[4]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t5{patches[5]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t6{patches[6]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor t7{patches[7]->temperature(state), cgh,
                          celerity::access::one_to_one{}, celerity::read_only};
    celerity::accessor mean{statistics.mean, cgh,
                            celerity::access::one_to_one{},
                            celerity::write_only, celerity::no_init};
    celerity::accessor variance{statistics.variance, cgh,
                                celerity::access::one_to_one{},
                                celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleEnsembleMoments8Kernel>(
        celerity::range<2>(config.rows, config.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          const float avg = 0.125f *
                            (t0[id] + t1[id] + t2[id] + t3[id] + t4[id] +
                             t5[id] + t6[id] + t7[id]);
          const float d0 = t0[id] - avg, d1 = t1[id] - avg;
          const float d2 = t2[id] - avg, d3 = t3[id] - avg;
          const float d4 = t4[id] - avg, d5 = t5[id] - avg;
          const float d6 = t6[id] - avg, d7 = t7[id] - avg;
          mean[id] = avg;
          variance[id] =
              0.125f * (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 +
                        d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7);
        });
  });
}

void submit_opacity(celerity::distr_queue &queue,
                    StatisticsBuffers &statistics, const Config &config) {
  const std::size_t samples = config.spectral_samples;
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor mean{statistics.mean, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor variance{statistics.variance, cgh,
                                celerity::access::one_to_one{},
                                celerity::read_only};
    celerity::accessor weights{statistics.spectral_weight, cgh,
                               celerity::access::all{}, celerity::read_only};
    celerity::accessor kappas{statistics.spectral_kappa, cgh,
                              celerity::access::all{}, celerity::read_only};
    celerity::accessor opacity{statistics.opacity, cgh,
                               celerity::access::one_to_one{},
                               celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleSpectralOpacityKernel>(
        celerity::range<2>(config.rows, config.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          const float temp = mean[id], var = variance[id];
          float integral = 0.0f;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const float kappa = kappas[sample];
            integral += weights[sample] *
                        sycl::exp(-kappa / (temp + 0.25f)) *
                        (1.0f + kappa * var);
          }
          opacity[id] = integral;
        });
  });
}

void submit_source(celerity::distr_queue &queue,
                   StatisticsBuffers &statistics, const Config &config) {
  const std::size_t samples = config.spectral_samples;
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor mean{statistics.mean, cgh,
                            celerity::access::one_to_one{},
                            celerity::read_only};
    celerity::accessor variance{statistics.variance, cgh,
                                celerity::access::one_to_one{},
                                celerity::read_only};
    celerity::accessor opacity{statistics.opacity, cgh,
                               celerity::access::one_to_one{},
                               celerity::read_only};
    celerity::accessor weights{statistics.spectral_weight, cgh,
                               celerity::access::all{}, celerity::read_only};
    celerity::accessor kappas{statistics.spectral_kappa, cgh,
                              celerity::access::all{}, celerity::read_only};
    celerity::accessor source{statistics.source, cgh,
                              celerity::access::one_to_one{},
                              celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleSpectralSourceKernel>(
        celerity::range<2>(config.rows, config.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          const float temp = mean[id], var = variance[id], base = opacity[id];
          float integral = 0.0f;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const float kappa = kappas[sample];
            const float transmission =
                sycl::exp(-kappa * (0.12f + var));
            integral += weights[sample] * (base + temp) *
                        (1.0f - transmission) / (0.05f + kappa);
          }
          source[id] = integral;
        });
  });
}

void submit_risk(celerity::distr_queue &queue, StatisticsBuffers &statistics,
                 const Config &config) {
  const std::size_t samples = config.spectral_samples;
  queue.submit([&](celerity::handler &cgh) {
    apply_hints(cgh, config);
    celerity::accessor variance{statistics.variance, cgh,
                                celerity::access::one_to_one{},
                                celerity::read_only};
    celerity::accessor opacity{statistics.opacity, cgh,
                               celerity::access::one_to_one{},
                               celerity::read_only};
    celerity::accessor source{statistics.source, cgh,
                              celerity::access::one_to_one{},
                              celerity::read_only};
    celerity::accessor weights{statistics.spectral_weight, cgh,
                               celerity::access::all{}, celerity::read_only};
    celerity::accessor kappas{statistics.spectral_kappa, cgh,
                              celerity::access::all{}, celerity::read_only};
    celerity::accessor risk{statistics.risk, cgh,
                            celerity::access::one_to_one{},
                            celerity::write_only, celerity::no_init};
    cgh.parallel_for<ArfCeleSpectralRiskKernel>(
        celerity::range<2>(config.rows, config.cols),
        [=](celerity::item<2> item) {
          const auto id = item.get_id();
          const float var = variance[id], radiance = source[id];
          const float base = opacity[id];
          float integral = 0.0f;
          for (std::size_t sample = 0; sample < samples; ++sample) {
            const float kappa = kappas[sample];
            const float weighted =
                weights[sample] * radiance / (1.0f + kappa * var);
            integral += sycl::sqrt(sycl::fabs(weighted) + 1.0e-12f) *
                        (0.5f + base * kappa);
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

Result read_result(celerity::distr_queue &queue,
                   std::vector<std::unique_ptr<PatchBuffers>> &patches,
                   StatisticsBuffers &statistics, const Config &config,
                   std::size_t final_state) {
  Result result;
  const std::size_t total_cells = config.rows * config.cols;
  const std::size_t checks =
      config.host_read_full ? total_cells : std::min<std::size_t>(64, total_cells);
  if (config.mode != Mode::StatisticsOnly) {
    for (auto &patch : patches) {
      auto temperature = queue.fence(patch->temperature(final_state)).get();
      auto fuel = queue.fence(patch->fuel(final_state)).get();
      auto oxidizer = queue.fence(patch->oxidizer(final_state)).get();
      auto product = queue.fence(patch->product(final_state)).get();
      for (std::size_t sample = 0; sample < checks; ++sample) {
        const std::size_t linear = config.host_read_full
                                       ? sample
                                       : sample * total_cells / checks;
        const celerity::id<2> id(linear / config.cols, linear % config.cols);
        const float t = temperature[id], f = fuel[id], o = oxidizer[id];
        const float p = product[id];
        const bool finite = std::isfinite(t) && std::isfinite(f) &&
                            std::isfinite(o) && std::isfinite(p);
        result.nonfinite += finite ? 0 : 1;
        result.valid = result.valid && finite && t >= 0.2f && t <= 4.0f &&
                       f >= 0.0f && o >= 0.0f && p >= 0.0f;
        result.min_temperature = std::min(result.min_temperature, t);
        result.max_temperature = std::max(result.max_temperature, t);
        result.min_species = std::min(result.min_species, std::min(f, std::min(o, p)));
        result.checksum += (1.0 + 0.01 * patch->patch_id) *
                           (t + 0.7 * f + 0.5 * o + 0.3 * p);
        ++result.checked_cells;
      }
    }
  }
  if (config.mode != Mode::PatchOnly) {
    auto risk = queue.fence(statistics.risk).get();
    for (std::size_t sample = 0; sample < checks; ++sample) {
      const std::size_t linear =
          config.host_read_full ? sample : sample * total_cells / checks;
      const celerity::id<2> id(linear / config.cols, linear % config.cols);
      const float value = risk[id];
      const bool finite = std::isfinite(value);
      result.nonfinite += finite ? 0 : 1;
      result.valid = result.valid && finite && value >= 0.0f;
      result.min_risk = std::min(result.min_risk, value);
      result.checksum += 0.11 * value;
      ++result.checked_cells;
    }
  }
  if (config.mode == Mode::StatisticsOnly) {
    result.min_temperature = result.max_temperature = result.min_species = 0.0f;
  }
  if (config.mode == Mode::PatchOnly) {
    result.min_risk = 0.0f;
  }
  result.valid = result.valid && result.nonfinite == 0 &&
                 std::isfinite(result.checksum);
  return result;
}

bool estimate_memory(const Config &config, std::size_t &bytes_per_patch,
                     std::size_t &total_bytes) {
  std::size_t cells = 0, all_patch_bytes = 0, stats_bytes = 0, coeff_bytes = 0;
  if (!checked_mul(config.rows, config.cols, cells) ||
      !checked_mul(cells, std::size_t{66}, bytes_per_patch) ||
      !checked_mul(bytes_per_patch, config.patches, all_patch_bytes) ||
      !checked_mul(cells, std::size_t{5 * sizeof(float)}, stats_bytes) ||
      !checked_mul(config.spectral_samples,
                   std::size_t{2 * sizeof(float)}, coeff_bytes) ||
      !checked_add(all_patch_bytes, stats_bytes, total_bytes) ||
      !checked_add(total_bytes, coeff_bytes, total_bytes)) {
    return false;
  }
  return true;
}

bool validate_config(const Config &config, std::size_t total_bytes,
                     std::string &error) {
  if (config.rows < 4 || config.cols < 4) {
    error = "rows and cols must be at least 4";
  } else if (config.patches != 4 && config.patches != 8) {
    error = "patches must be 4 or 8";
  } else if (config.rows % config.split_parts != 0) {
    error = "rows must be divisible by split-parts";
  } else if (config.cold_substeps > std::numeric_limits<std::uint16_t>::max() ||
             config.hot_substeps > std::numeric_limits<std::uint16_t>::max()) {
    error = "chemistry substeps must fit uint16";
  } else if (config.hot_substeps < config.cold_substeps) {
    error = "hot-substeps must be at least cold-substeps";
  } else if (config.memory_limit_gib < 0.0) {
    error = "memory-limit-gib must be nonnegative";
  } else if (config.memory_limit_gib > 0.0 &&
             static_cast<long double>(total_bytes) >
                 static_cast<long double>(config.memory_limit_gib) * 1024.0L *
                     1024.0L * 1024.0L) {
    error = "estimated buffers exceed memory-limit-gib";
  }
  return error.empty();
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
  std::size_t bytes_per_patch = 0, total_bytes = 0;
  if (!estimate_memory(config, bytes_per_patch, total_bytes) ||
      !validate_config(config, total_bytes, error)) {
    std::cerr << "ERROR " << (error.empty() ? "memory estimate overflow" : error)
              << '\n';
    return 1;
  }

  const std::size_t cells = config.rows * config.cols;
  std::size_t hot_cells = 0, chemistry_steps_per_patch = 0;
  for (std::size_t row = 0; row < config.rows; ++row) {
    for (std::size_t col = 0; col < config.cols; ++col) {
      hot_cells += is_hot_cell(row, col, config) ? 1 : 0;
      chemistry_steps_per_patch +=
          chemistry_substeps_for_cell(row, col, config);
    }
  }
  const std::size_t patch_kernels =
      config.mode == Mode::StatisticsOnly ? 0 : 3 * config.patches;
  const std::size_t stats_kernels = config.mode == Mode::PatchOnly ? 0 : 4;
  const std::size_t kernels_per_window = patch_kernels + stats_kernels;
  const std::size_t max_width =
      config.mode == Mode::StatisticsOnly ? 1 : 2 * config.patches;
  const std::size_t critical_path = config.mode == Mode::Full
                                        ? 6
                                        : (config.mode == Mode::PatchOnly ? 2 : 4);

  std::cout << std::setprecision(9);
  std::cout << "CONFIG rows=" << config.rows << " cols=" << config.cols
            << " patches=" << config.patches << " steps=" << config.steps
            << " mode=" << mode_name(config.mode)
            << " layout=" << layout_name(config.stiffness_layout)
            << " cold_substeps=" << config.cold_substeps
            << " hot_substeps=" << config.hot_substeps
            << " spectral_samples=" << config.spectral_samples
            << " split_parts=" << config.split_parts
            << " celerity_split="
            << (config.split_shape == SplitShape::OneD ? "1d" : "2d")
            << " oversubscribe=" << config.oversubscribe
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << '\n';
  std::cout << "DAG kernels_per_window=" << kernels_per_window
            << " max_width=" << max_width
            << " critical_path_levels=" << critical_path
            << " total_kernels=" << kernels_per_window * config.steps << '\n';
  std::cout << "WORK cells_per_patch=" << cells
            << " hot_cells_per_patch=" << hot_cells
            << " chemistry_substeps_per_patch=" << chemistry_steps_per_patch
            << " chemistry_substeps_per_window="
            << chemistry_steps_per_patch * config.patches
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
    StatisticsBuffers statistics(config);
    const double init_end = get_time();
    celerity::distr_queue queue;
    const double queue_end = get_time();

    std::size_t current = 0;
    const double run_begin = get_time();
    for (std::size_t step = 0; step < config.steps; ++step) {
      const std::size_t next = 1 - current;
      if (config.mode != Mode::StatisticsOnly) {
        for (auto &patch : patches) {
          submit_transport(queue, *patch, current, config);
          if (config.wait_each_kernel) queue.slow_full_sync();
        }
        for (auto &patch : patches) {
          submit_chemistry(queue, *patch, current, config);
          if (config.wait_each_kernel) queue.slow_full_sync();
        }
        for (auto &patch : patches) {
          submit_combine(queue, *patch, next, config);
          if (config.wait_each_kernel) queue.slow_full_sync();
        }
      }
      const std::size_t stats_state =
          config.mode == Mode::StatisticsOnly ? current : next;
      if (config.mode != Mode::PatchOnly) {
        if (config.patches == 4) {
          submit_moments4(queue, patches, statistics, stats_state, config);
        } else {
          submit_moments8(queue, patches, statistics, stats_state, config);
        }
        if (config.wait_each_kernel) queue.slow_full_sync();
        submit_opacity(queue, statistics, config);
        if (config.wait_each_kernel) queue.slow_full_sync();
        submit_source(queue, statistics, config);
        if (config.wait_each_kernel) queue.slow_full_sync();
        submit_risk(queue, statistics, config);
        if (config.wait_each_kernel) queue.slow_full_sync();
      }
      if (!config.wait_each_kernel) queue.slow_full_sync();
      if (config.mode != Mode::StatisticsOnly) current = next;
    }
    const double run_end = get_time();
    const double host_begin = get_time();
    const Result result =
        read_result(queue, patches, statistics, config, current);
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
