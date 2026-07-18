#include "covariance_subspace_common.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace cs = covariance_subspace;

namespace {

using clock_type = std::chrono::steady_clock;

double seconds_between(clock_type::time_point begin,
                       clock_type::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

class CovarianceMeanKernel;
class CovarianceStddevKernel;
class CovarianceNormalizeKernel;
class CovarianceTriangleKernel;
class ReverseBlockGSPanelKernel;
class ReverseBlockGSUpdateKernel;

struct BandBuffers {
  std::vector<float> data_host;
  std::vector<float> mean_host;
  std::vector<float> stddev_host;
  std::vector<float> basis_host;
  sycl::buffer<float, 2> data;
  sycl::buffer<float, 1> mean;
  sycl::buffer<float, 1> stddev;
  sycl::buffer<float, 2> basis;

  BandBuffers(std::size_t band, const cs::Config &config)
      : data_host(cs::make_samples(band, config)),
        mean_host(config.antennas, 0.0f),
        stddev_host(config.antennas, 0.0f),
        basis_host(config.mode == cs::Mode::OrthogonalizeOnly
                       ? cs::make_initial_basis(config)
                       : std::vector<float>(config.antennas * config.antennas,
                                            0.0f)),
        data(data_host.data(),
             sycl::range<2>(config.antennas, config.snapshots)),
        mean(mean_host.data(), sycl::range<1>(config.antennas)),
        stddev(stddev_host.data(), sycl::range<1>(config.antennas)),
        basis(basis_host.data(),
              sycl::range<2>(config.antennas, config.antennas)) {}
};

void maybe_wait(sycl::queue &queue, const cs::Config &config) {
  if (config.wait_each_kernel) {
    queue.wait_and_throw();
  }
}

void submit_mean(sycl::queue &queue, BandBuffers &band,
                 const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  queue.submit([&](sycl::handler &cgh) {
    auto data = band.data.get_access<sycl::access::mode::read>(cgh);
    auto mean = band.mean.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<CovarianceMeanKernel>(
        sycl::range<1>(antennas), [=](sycl::id<1> item) {
          const std::size_t antenna = item[0];
          float sum = 0.0f;
          for (std::size_t sample = 0; sample < snapshots; ++sample) {
            sum += data[{antenna, sample}];
          }
          mean[antenna] = sum / static_cast<float>(snapshots);
        });
  });
  maybe_wait(queue, config);
}

void submit_stddev(sycl::queue &queue, BandBuffers &band,
                   const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  queue.submit([&](sycl::handler &cgh) {
    auto data = band.data.get_access<sycl::access::mode::read>(cgh);
    auto mean = band.mean.get_access<sycl::access::mode::read>(cgh);
    auto stddev =
        band.stddev.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<CovarianceStddevKernel>(
        sycl::range<1>(antennas), [=](sycl::id<1> item) {
          const std::size_t antenna = item[0];
          float sum = 0.0f;
          for (std::size_t sample = 0; sample < snapshots; ++sample) {
            const float centered = data[{antenna, sample}] - mean[antenna];
            sum += centered * centered;
          }
          const float variance = sum / static_cast<float>(snapshots);
          stddev[antenna] = sycl::sqrt(sycl::fmax(variance, 1.0e-12f));
        });
  });
  maybe_wait(queue, config);
}

void submit_normalize(sycl::queue &queue, BandBuffers &band,
                      const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  queue.submit([&](sycl::handler &cgh) {
    auto data = band.data.get_access<sycl::access::mode::read_write>(cgh);
    auto mean = band.mean.get_access<sycl::access::mode::read>(cgh);
    auto stddev = band.stddev.get_access<sycl::access::mode::read>(cgh);
    cgh.parallel_for<CovarianceNormalizeKernel>(
        sycl::range<2>(antennas, snapshots), [=](sycl::id<2> item) {
          const std::size_t antenna = item[0];
          const std::size_t sample = item[1];
          data[item] =
              (data[item] - mean[antenna]) / stddev[antenna];
        });
  });
  maybe_wait(queue, config);
}

void submit_triangular_correlation(sycl::queue &queue, BandBuffers &band,
                                   const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const bool folded =
      config.covariance_order == cs::CovarianceOrder::Folded;
  queue.submit([&](sycl::handler &cgh) {
    auto data = band.data.get_access<sycl::access::mode::read>(cgh);
    auto basis =
        band.basis.get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<CovarianceTriangleKernel>(
        sycl::range<1>(antennas), [=](sycl::id<1> item) {
          const std::size_t work_row = item[0];
          const std::size_t antenna0 =
              folded ? ((work_row & 1U) == 0U
                            ? work_row / 2
                            : antennas - 1 - work_row / 2)
                     : work_row;
          for (std::size_t antenna1 = 0; antenna1 < antennas; ++antenna1) {
            if (antenna1 < antenna0) {
              basis[{work_row, antenna1}] = 0.0f;
              continue;
            }
            float dot = 0.0f;
            for (std::size_t sample = 0; sample < snapshots; ++sample) {
              dot += data[{antenna0, sample}] *
                     data[{antenna1, sample}];
            }
            float correlation = dot / static_cast<float>(snapshots);
            if (antenna1 == antenna0) {
              correlation += 0.25f;
            }
            basis[{work_row, antenna1}] = correlation;
          }
        });
  });
  maybe_wait(queue, config);
}

void submit_reverse_panel(sycl::queue &queue, BandBuffers &band,
                          std::size_t panel_begin, std::size_t panel_end,
                          const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  constexpr std::size_t local_size = 128;
  queue.submit([&](sycl::handler &cgh) {
    auto basis = band.basis.get_access<sycl::access::mode::read_write>(cgh);
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(local_size), cgh);
    cgh.parallel_for<ReverseBlockGSPanelKernel>(
        sycl::nd_range<1>(sycl::range<1>(local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
          const std::size_t lane = item.get_local_linear_id();
          for (std::size_t pivot_cursor = panel_end;
               pivot_cursor > panel_begin; --pivot_cursor) {
            const std::size_t pivot = pivot_cursor - 1;
            float norm_squared = 0.0f;
            for (std::size_t col = lane; col < antennas; col += local_size) {
              const float value = basis[{pivot, col}];
              norm_squared += value * value;
            }
            scratch[lane] = norm_squared;
            item.barrier(sycl::access::fence_space::local_space);
            for (std::size_t stride = local_size / 2; stride > 0;
                 stride /= 2) {
              if (lane < stride) {
                scratch[lane] += scratch[lane + stride];
              }
              item.barrier(sycl::access::fence_space::local_space);
            }
            const float inverse_norm =
                sycl::rsqrt(sycl::fmax(scratch[0], 1.0e-20f));
            for (std::size_t col = lane; col < antennas;
                 col += local_size) {
              basis[{pivot, col}] *= inverse_norm;
            }
            item.barrier(sycl::access::fence_space::global_and_local);
            for (std::size_t row = panel_begin; row < pivot; ++row) {
              float projection = 0.0f;
              for (std::size_t col = lane; col < antennas;
                   col += local_size) {
                projection +=
                    basis[{row, col}] * basis[{pivot, col}];
              }
              scratch[lane] = projection;
              item.barrier(sycl::access::fence_space::local_space);
              for (std::size_t stride = local_size / 2; stride > 0;
                   stride /= 2) {
                if (lane < stride) {
                  scratch[lane] += scratch[lane + stride];
                }
                item.barrier(sycl::access::fence_space::local_space);
              }
              projection = scratch[0];
              for (std::size_t col = lane; col < antennas;
                   col += local_size) {
                basis[{row, col}] -=
                    projection * basis[{pivot, col}];
              }
              item.barrier(sycl::access::fence_space::global_and_local);
            }
          }
        });
  });
  maybe_wait(queue, config);
}

void submit_reverse_update(sycl::queue &queue, BandBuffers &band,
                           std::size_t panel_begin, std::size_t panel_end,
                           const cs::Config &config) {
  if (panel_begin == 0) {
    return;
  }
  const std::size_t antennas = config.antennas;
  const std::size_t execution_rows =
      config.compact_gs ? panel_begin : config.subspace_rank;
  queue.submit([&](sycl::handler &cgh) {
    auto basis = band.basis.get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<ReverseBlockGSUpdateKernel>(
        sycl::range<1>(execution_rows), [=](sycl::id<1> item) {
          const std::size_t row = item[0];
          if (row >= panel_begin) {
            return;
          }
          for (std::size_t pivot_cursor = panel_end;
               pivot_cursor > panel_begin; --pivot_cursor) {
            const std::size_t pivot = pivot_cursor - 1;
            float projection = 0.0f;
            for (std::size_t col = 0; col < antennas; ++col) {
              projection += basis[{row, col}] * basis[{pivot, col}];
            }
            for (std::size_t col = 0; col < antennas; ++col) {
              basis[{row, col}] -= projection * basis[{pivot, col}];
            }
          }
        });
  });
  maybe_wait(queue, config);
}

void submit_band(sycl::queue &queue, BandBuffers &band,
                 const cs::Config &config) {
  if (config.mode != cs::Mode::OrthogonalizeOnly) {
    submit_mean(queue, band, config);
    submit_stddev(queue, band, config);
    submit_normalize(queue, band, config);
    submit_triangular_correlation(queue, band, config);
  }
  if (config.mode != cs::Mode::CorrelationOnly) {
    std::size_t panel_end = config.subspace_rank;
    while (panel_end > 0) {
      const std::size_t panel_begin =
          panel_end > config.panel_size ? panel_end - config.panel_size : 0;
      submit_reverse_panel(queue, band, panel_begin, panel_end, config);
      submit_reverse_update(queue, band, panel_begin, panel_end, config);
      panel_end = panel_begin;
    }
  }
}

cs::VerificationResult read_result(
    std::vector<std::unique_ptr<BandBuffers>> &bands,
    const cs::Config &config) {
  cs::VerificationResult total;
  for (auto &band : bands) {
    sycl::host_accessor basis(band->basis, sycl::read_only);
    const cs::VerificationResult current = cs::verify_basis(
        config, [&](std::size_t row, std::size_t col) {
          return basis[{row, col}];
        });
    cs::merge_verification(total, current);
  }
  return total;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const cs::Config config = cs::parse_config(argc, argv);
    cs::print_configuration(config, "sycl");

    const auto total_begin = clock_type::now();
    std::vector<std::unique_ptr<BandBuffers>> bands;
    bands.reserve(config.bands);
    for (std::size_t band = 0; band < config.bands; ++band) {
      bands.push_back(std::make_unique<BandBuffers>(band, config));
    }
    const auto init_end = clock_type::now();

    sycl::queue queue;
    const auto queue_end = clock_type::now();
    std::cout << "DEVICE name="
              << queue.get_device().get_info<sycl::info::device::name>()
              << '\n';

    std::vector<double> epoch_seconds;
    epoch_seconds.reserve(config.epochs);
    const auto run_begin = clock_type::now();
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
      const auto epoch_begin = clock_type::now();
      for (auto &band : bands) {
        submit_band(queue, *band, config);
      }
      queue.wait_and_throw();
      const auto epoch_end = clock_type::now();
      epoch_seconds.push_back(seconds_between(epoch_begin, epoch_end));
      std::cout << std::fixed << std::setprecision(6)
                << "EPOCH index=" << epoch
                << " run_sec=" << epoch_seconds.back() << '\n';
    }
    const auto run_end = clock_type::now();

    const auto host_begin = clock_type::now();
    const cs::VerificationResult result = read_result(bands, config);
    const auto host_end = clock_type::now();

    std::cout << std::fixed << std::setprecision(6)
              << "TIMING init_sec=" << seconds_between(total_begin, init_end)
              << '\n'
              << "TIMING queue_sec=" << seconds_between(init_end, queue_end)
              << '\n'
              << "TIMING run_sec=" << seconds_between(run_begin, run_end)
              << '\n'
              << "TIMING host_sec=" << seconds_between(host_begin, host_end)
              << '\n'
              << "TIMING total_sec=" << seconds_between(total_begin, host_end)
              << '\n'
              << "RESULT checksum=" << result.checksum
              << " checked_rows=" << result.checked_rows
              << " max_norm_error=" << result.max_norm_error
              << " max_dot=" << result.max_dot
              << " nonfinite=" << result.nonfinite << '\n'
              << "VERIFY passed=" << (result.passed ? 1 : 0) << '\n';
    return config.verify && !result.passed ? 2 : 0;
  } catch (const sycl::exception &exception) {
    std::cerr << "ERROR SYCL " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    std::cerr << "ERROR " << exception.what() << '\n';
    return 2;
  }
}
