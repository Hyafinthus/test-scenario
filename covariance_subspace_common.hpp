#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace covariance_subspace {

enum class Mode { Full, CorrelationOnly, OrthogonalizeOnly };
enum class CovarianceOrder { Forward, Folded };
enum class CorrelationLayout { RowSerial, PairParallel };

struct Config {
  std::size_t antennas = 768;
  std::size_t snapshots = 2048;
  std::size_t bands = 8;
  std::size_t time_blocks = 1;
  std::size_t epochs = 3;
  std::size_t subspace_rank = 512;
  std::size_t panel_size = 64;
  std::size_t oversubscribe = 1;
  std::size_t theory_devices = 4;
  Mode mode = Mode::Full;
  CovarianceOrder covariance_order = CovarianceOrder::Forward;
  CorrelationLayout correlation_layout = CorrelationLayout::RowSerial;
  double memory_limit_gib = 0.0;
  bool compact_gs = false;
  bool wait_each_kernel = false;
  bool verify = true;
};

inline const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Full:
    return "full";
  case Mode::CorrelationOnly:
    return "correlation";
  case Mode::OrthogonalizeOnly:
    return "orthogonalize";
  }
  return "unknown";
}

inline const char *covariance_order_name(CovarianceOrder order) {
  return order == CovarianceOrder::Forward ? "forward" : "folded";
}

inline const char *correlation_layout_name(CorrelationLayout layout) {
  return layout == CorrelationLayout::RowSerial ? "row" : "pair";
}

inline bool checked_add(std::size_t lhs, std::size_t rhs,
                        std::size_t &result) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

inline bool checked_mul(std::size_t lhs, std::size_t rhs,
                        std::size_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

inline bool parse_size(const std::string &text, std::size_t &value) {
  if (text.empty() || text.front() == '-') {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0' ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  value = static_cast<std::size_t>(parsed);
  return true;
}

inline bool parse_bool(const std::string &text, bool &value) {
  if (text == "1" || text == "true" || text == "TRUE") {
    value = true;
    return true;
  }
  if (text == "0" || text == "false" || text == "FALSE") {
    value = false;
    return true;
  }
  return false;
}

inline bool parse_double(const std::string &text, double &value) {
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

inline void print_help(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --antennas <int>             Array channels/features (default 768)\n"
      << "  --snapshots <int>            Samples per frequency band (default 2048)\n"
      << "  --bands <int>                Independent frequency bands (default 8)\n"
      << "  --time-blocks <int>          Concurrent sub-integrations (default 1)\n"
      << "  --epochs <int>               Repeated wait windows (default 3)\n"
      << "  --rank <int>                 Rows orthogonalized (default 512)\n"
      << "  --panel <int>                Reverse block-GS panel (default 64)\n"
      << "  --mode <full|correlation|orthogonalize>\n"
      << "  --covariance-order <forward|folded>\n"
      << "  --correlation-layout <row|pair>\n"
      << "  --memory-limit-gib <real>    Fail above logical state; 0 is unlimited\n"
      << "  --compact-gs <0|1>           Submit only the active GS prefix\n"
      << "  --wait-each-kernel <0|1>     Destroy the wait-window DAG (control)\n"
      << "  --oversubscribe <int>        Celerity chunks/device hint (default 1)\n"
      << "  --theory-devices <int>       Chunk-share report width (default 4)\n"
      << "  --verify <0|1>               Check sampled orthogonality (default 1)\n"
      << "  --help                        Show this message\n";
}

inline Config parse_config(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--help") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value = argv[++index];
    bool parsed = false;
    if (option == "--antennas") {
      parsed = parse_size(value, config.antennas);
    } else if (option == "--snapshots") {
      parsed = parse_size(value, config.snapshots);
    } else if (option == "--bands") {
      parsed = parse_size(value, config.bands);
    } else if (option == "--time-blocks") {
      parsed = parse_size(value, config.time_blocks);
    } else if (option == "--epochs") {
      parsed = parse_size(value, config.epochs);
    } else if (option == "--rank") {
      parsed = parse_size(value, config.subspace_rank);
    } else if (option == "--panel") {
      parsed = parse_size(value, config.panel_size);
    } else if (option == "--oversubscribe") {
      parsed = parse_size(value, config.oversubscribe);
    } else if (option == "--theory-devices") {
      parsed = parse_size(value, config.theory_devices);
    } else if (option == "--memory-limit-gib") {
      parsed = parse_double(value, config.memory_limit_gib) &&
               config.memory_limit_gib >= 0.0;
    } else if (option == "--compact-gs") {
      parsed = parse_bool(value, config.compact_gs);
    } else if (option == "--wait-each-kernel") {
      parsed = parse_bool(value, config.wait_each_kernel);
    } else if (option == "--verify") {
      parsed = parse_bool(value, config.verify);
    } else if (option == "--mode") {
      if (value == "full") {
        config.mode = Mode::Full;
        parsed = true;
      } else if (value == "correlation") {
        config.mode = Mode::CorrelationOnly;
        parsed = true;
      } else if (value == "orthogonalize") {
        config.mode = Mode::OrthogonalizeOnly;
        parsed = true;
      }
    } else if (option == "--covariance-order") {
      if (value == "forward") {
        config.covariance_order = CovarianceOrder::Forward;
        parsed = true;
      } else if (value == "folded") {
        config.covariance_order = CovarianceOrder::Folded;
        parsed = true;
      }
    } else if (option == "--correlation-layout") {
      if (value == "row") {
        config.correlation_layout = CorrelationLayout::RowSerial;
        parsed = true;
      } else if (value == "pair") {
        config.correlation_layout = CorrelationLayout::PairParallel;
        parsed = true;
      }
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
    if (!parsed) {
      throw std::invalid_argument("invalid value for " + option + ": " +
                                  value);
    }
  }

  if (config.antennas == 0 || config.snapshots == 0 || config.bands == 0 ||
      config.time_blocks == 0 || config.epochs == 0 ||
      config.subspace_rank == 0 ||
      config.panel_size == 0 || config.oversubscribe == 0 ||
      config.theory_devices == 0) {
    throw std::invalid_argument("all integer parameters must be positive");
  }
  if (config.subspace_rank > config.antennas) {
    throw std::invalid_argument("--rank must not exceed --antennas");
  }
  std::size_t tiles = 0;
  std::size_t data_values = 0;
  std::size_t basis_values = 0;
  std::size_t vector_values = 0;
  std::size_t values_per_tile = 0;
  std::size_t bytes_per_tile = 0;
  std::size_t total_bytes = 0;
  if (!checked_mul(config.bands, config.time_blocks, tiles) ||
      !checked_mul(config.antennas, config.snapshots, data_values) ||
      !checked_mul(config.antennas, config.antennas, basis_values) ||
      !checked_mul(config.antennas, std::size_t{2}, vector_values) ||
      !checked_add(data_values, basis_values, values_per_tile) ||
      !checked_add(values_per_tile, vector_values, values_per_tile) ||
      !checked_mul(values_per_tile, sizeof(float), bytes_per_tile) ||
      !checked_mul(bytes_per_tile, tiles, total_bytes)) {
    throw std::invalid_argument("tile count or logical state size overflows");
  }
  constexpr double gib = 1024.0 * 1024.0 * 1024.0;
  if (config.memory_limit_gib > 0.0 &&
      static_cast<double>(total_bytes) > config.memory_limit_gib * gib) {
    throw std::invalid_argument("logical state exceeds --memory-limit-gib");
  }
  if (config.mode == Mode::OrthogonalizeOnly && config.epochs != 1) {
    throw std::invalid_argument(
        "orthogonalize-only mode requires --epochs 1 because no covariance "
        "stage resets the basis");
  }
  return config;
}

inline std::size_t folded_index(std::size_t work_index, std::size_t size) {
  return (work_index & 1U) == 0U ? work_index / 2
                                : size - 1 - work_index / 2;
}

inline std::size_t covariance_row(std::size_t work_index, std::size_t size,
                                  CovarianceOrder order) {
  return order == CovarianceOrder::Forward
             ? work_index
             : folded_index(work_index, size);
}

inline float initial_sample(std::size_t band, std::size_t antenna,
                            std::size_t snapshot) {
  std::uint32_t value = static_cast<std::uint32_t>(snapshot + 1);
  value ^= static_cast<std::uint32_t>((antenna + 1) * 0x9e3779b9U);
  value ^= static_cast<std::uint32_t>((band + 7) * 0x85ebca6bU);
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  const float noise = static_cast<float>(value & 0xffffU) / 32767.5f - 1.0f;
  const float common = static_cast<float>((snapshot + 3 * band) % 97) / 97.0f;
  const float gain = 0.85f + 0.003f * static_cast<float>(antenna % 29);
  return gain * noise + 0.12f * common +
         0.0005f * static_cast<float>(antenna);
}

inline std::vector<float> make_samples(std::size_t band,
                                       const Config &config) {
  std::vector<float> samples(config.antennas * config.snapshots);
  for (std::size_t antenna = 0; antenna < config.antennas; ++antenna) {
    for (std::size_t snapshot = 0; snapshot < config.snapshots; ++snapshot) {
      samples[antenna * config.snapshots + snapshot] =
          initial_sample(band, antenna, snapshot);
    }
  }
  return samples;
}

inline std::vector<float> make_initial_basis(const Config &config) {
  std::vector<float> basis(config.antennas * config.antennas, 0.0f);
  for (std::size_t row = 0; row < config.antennas; ++row) {
    for (std::size_t col = row; col < config.antennas; ++col) {
      const std::size_t distance = col - row;
      basis[row * config.antennas + col] =
          row == col ? 1.0f : 0.015f / static_cast<float>(distance + 1);
    }
  }
  return basis;
}

inline std::size_t tile_count(const Config &config) {
  std::size_t count = 0;
  if (!checked_mul(config.bands, config.time_blocks, count)) {
    throw std::overflow_error("tile count overflow");
  }
  return count;
}

inline std::size_t logical_bytes_per_tile(const Config &config) {
  std::size_t data_values = 0;
  std::size_t basis_values = 0;
  std::size_t vector_values = 0;
  std::size_t values = 0;
  std::size_t bytes = 0;
  if (!checked_mul(config.antennas, config.snapshots, data_values) ||
      !checked_mul(config.antennas, config.antennas, basis_values) ||
      !checked_mul(config.antennas, std::size_t{2}, vector_values) ||
      !checked_add(data_values, basis_values, values) ||
      !checked_add(values, vector_values, values) ||
      !checked_mul(values, sizeof(float), bytes)) {
    throw std::overflow_error("logical bytes per tile overflow");
  }
  return bytes;
}

inline long double triangular_work(std::size_t begin, std::size_t end,
                                   std::size_t antennas,
                                   CovarianceOrder order) {
  long double work = 0.0L;
  for (std::size_t work_index = begin; work_index < end; ++work_index) {
    const std::size_t row =
        covariance_row(work_index, antennas, order);
    work += static_cast<long double>(antennas - row);
  }
  return work;
}

inline std::vector<double> theoretical_chunk_shares(const Config &config) {
  std::vector<double> shares(config.theory_devices, 0.0);
  const long double total =
      static_cast<long double>(config.antennas) *
      static_cast<long double>(config.antennas + 1) / 2.0L;
  for (std::size_t device = 0; device < config.theory_devices; ++device) {
    const std::size_t begin = config.antennas * device / config.theory_devices;
    const std::size_t end =
        config.antennas * (device + 1) / config.theory_devices;
    shares[device] = static_cast<double>(
        triangular_work(begin, end, config.antennas,
                        config.covariance_order) /
        total);
  }
  return shares;
}

inline std::size_t panel_count(const Config &config) {
  return (config.subspace_rank + config.panel_size - 1) / config.panel_size;
}

inline std::size_t kernels_per_band(const Config &config) {
  std::size_t kernels = 0;
  if (config.mode != Mode::OrthogonalizeOnly) {
    kernels += 4; // mean, standard deviation, normalize, triangular correlation
  }
  if (config.mode != Mode::CorrelationOnly) {
    const std::size_t panels = panel_count(config);
    kernels += panels + (panels == 0 ? 0 : panels - 1);
  }
  return kernels;
}

inline void print_configuration(const Config &config, const char *runtime) {
  const long double covariance_fmas =
      static_cast<long double>(config.snapshots) * config.antennas *
      (config.antennas + 1) / 2.0L;
  const long double gs_vector_fmas =
      static_cast<long double>(config.subspace_rank) *
      (config.subspace_rank - 1) * config.antennas;
  const std::size_t per_band = kernels_per_band(config);
  const std::size_t tiles = tile_count(config);
  const std::size_t bytes_per_tile = logical_bytes_per_tile(config);
  std::size_t total_bytes = 0;
  if (!checked_mul(bytes_per_tile, tiles, total_bytes)) {
    throw std::overflow_error("total logical state bytes overflow");
  }
  const std::vector<double> shares = theoretical_chunk_shares(config);
  double max_share = 0.0;
  for (double share : shares) {
    max_share = std::max(max_share, share);
  }

  std::cout << "CONFIG runtime=" << runtime
            << " antennas=" << config.antennas
            << " snapshots=" << config.snapshots
            << " bands=" << config.bands
            << " time_blocks=" << config.time_blocks
            << " tiles=" << tiles << " epochs=" << config.epochs
            << " rank=" << config.subspace_rank
            << " panel=" << config.panel_size
            << " mode=" << mode_name(config.mode)
            << " covariance_order="
            << covariance_order_name(config.covariance_order)
            << " correlation_layout="
            << correlation_layout_name(config.correlation_layout)
            << " compact_gs=" << (config.compact_gs ? 1 : 0)
            << " wait_each_kernel=" << (config.wait_each_kernel ? 1 : 0)
            << " oversubscribe=" << config.oversubscribe << '\n';
  std::cout << "DAG kernels_per_band=" << per_band
            << " kernels_per_tile=" << per_band
            << " kernels_per_window=" << per_band * tiles
            << " max_width=" << tiles
            << " critical_path_levels=" << per_band
            << " total_kernels="
            << per_band * tiles * config.epochs << '\n';
  std::cout << std::fixed << std::setprecision(0)
            << "WORK covariance_fmas_per_band="
            << static_cast<double>(covariance_fmas)
            << " covariance_fmas_per_window="
            << static_cast<double>(covariance_fmas * tiles)
            << " gs_dot_update_fmas_per_band="
            << static_cast<double>(gs_vector_fmas)
            << " gs_dot_update_fmas_per_window="
            << static_cast<double>(gs_vector_fmas * tiles) << '\n';
  std::cout << "MEMORY logical_bytes_per_tile=" << bytes_per_tile
            << " total_logical_state_bytes=" << total_bytes << '\n';
  std::cout << std::setprecision(6) << "THEORY devices="
            << config.theory_devices << " covariance_chunk_shares=";
  for (std::size_t device = 0; device < shares.size(); ++device) {
    if (device != 0) {
      std::cout << ',';
    }
    std::cout << shares[device];
  }
  std::cout << " deterministic_split_speedup_ceiling="
            << (max_share > 0.0 ? 1.0 / max_share : 0.0)
            << " whole_tile_speedup_ceiling="
            << std::min(config.theory_devices, tiles) << '\n';
}

struct VerificationResult {
  double checksum = 0.0;
  double max_norm_error = 0.0;
  double max_dot = 0.0;
  std::size_t nonfinite = 0;
  std::size_t checked_rows = 0;
  bool passed = true;
};

inline std::vector<std::size_t> verification_rows(std::size_t rank) {
  std::vector<std::size_t> rows;
  const std::size_t count = std::min<std::size_t>(rank, 12);
  rows.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t row = count == 1 ? 0 : index * (rank - 1) / (count - 1);
    if (rows.empty() || rows.back() != row) {
      rows.push_back(row);
    }
  }
  return rows;
}

template <typename Reader>
inline VerificationResult verify_basis(const Config &config, Reader read) {
  VerificationResult result;
  const std::vector<std::size_t> rows =
      verification_rows(config.subspace_rank);
  result.checked_rows = rows.size();

  for (std::size_t row = 0; row < config.subspace_rank; ++row) {
    for (std::size_t col = 0; col < config.antennas; ++col) {
      const double value = static_cast<double>(read(row, col));
      if (!std::isfinite(value)) {
        ++result.nonfinite;
      } else {
        result.checksum += value *
                           static_cast<double>(1 + (row * 17 + col * 13) % 31);
      }
    }
  }

  if (config.mode != Mode::CorrelationOnly) {
    for (std::size_t row : rows) {
      double norm = 0.0;
      for (std::size_t col = 0; col < config.antennas; ++col) {
        const double value = static_cast<double>(read(row, col));
        norm += value * value;
      }
      result.max_norm_error =
          std::max(result.max_norm_error, std::abs(std::sqrt(norm) - 1.0));
    }
    for (std::size_t left = 0; left < rows.size(); ++left) {
      for (std::size_t right = left + 1; right < rows.size(); ++right) {
        double dot = 0.0;
        for (std::size_t col = 0; col < config.antennas; ++col) {
          dot += static_cast<double>(read(rows[left], col)) *
                 static_cast<double>(read(rows[right], col));
        }
        result.max_dot = std::max(result.max_dot, std::abs(dot));
      }
    }
  }

  result.passed = result.nonfinite == 0 &&
                  (config.mode == Mode::CorrelationOnly ||
                   (result.max_norm_error <= 5.0e-2 &&
                    result.max_dot <= 8.0e-2));
  return result;
}

inline void merge_verification(VerificationResult &total,
                               const VerificationResult &band) {
  total.checksum += band.checksum;
  total.max_norm_error = std::max(total.max_norm_error, band.max_norm_error);
  total.max_dot = std::max(total.max_dot, band.max_dot);
  total.nonfinite += band.nonfinite;
  total.checked_rows += band.checked_rows;
  total.passed = total.passed && band.passed;
}

} // namespace covariance_subspace
