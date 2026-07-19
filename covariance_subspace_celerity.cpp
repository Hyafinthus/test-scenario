#include "covariance_subspace_common.hpp"

#include <celerity.h>

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
class CovarianceTrianglePairKernel;
class ReverseBlockGSPanelKernel;
class ReverseBlockGSUpdateKernel;

struct BandBuffers {
  // One independent observation tile. The flat id is
  // time_block * config.bands + frequency_band.
  std::size_t tile_id;
  std::vector<float> data_host;
  std::vector<float> mean_host;
  std::vector<float> stddev_host;
  std::vector<float> basis_host;
  celerity::buffer<float, 2> data;
  celerity::buffer<float, 1> mean;
  celerity::buffer<float, 1> stddev;
  celerity::buffer<float, 2> basis;

  BandBuffers(std::size_t tile, const cs::Config &config)
      : tile_id(tile), data_host(cs::make_samples(tile, config)),
        mean_host(config.antennas, 0.0f),
        stddev_host(config.antennas, 0.0f),
        basis_host(config.mode == cs::Mode::OrthogonalizeOnly
                       ? cs::make_initial_basis(config)
                       : std::vector<float>(config.antennas * config.antennas,
                                            0.0f)),
        data(data_host.data(),
             celerity::range<2>(config.antennas, config.snapshots)),
        mean(mean_host.data(), celerity::range<1>(config.antennas)),
        stddev(stddev_host.data(), celerity::range<1>(config.antennas)),
        basis(basis_host.data(),
              celerity::range<2>(config.antennas, config.antennas)) {}
};

void apply_hints(celerity::handler &cgh, const cs::Config &config) {
  celerity::experimental::hint(
      cgh, celerity::experimental::hints::split_1d{});
  if (config.oversubscribe > 1) {
    celerity::experimental::hint(
        cgh,
        celerity::experimental::hints::oversubscribe(config.oversubscribe));
  }
}

void maybe_wait(celerity::distr_queue &queue, const cs::Config &config) {
  if (config.wait_each_kernel) {
    queue.slow_full_sync();
  }
}

void submit_mean(celerity::distr_queue &queue, BandBuffers &band,
                 const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    accessor data{band.data, cgh,
                  [snapshots](chunk<1> chunk) {
                    return subrange<2>{{chunk.offset[0], 0},
                                       {chunk.range[0], snapshots}};
                  },
                  read_only};
    accessor mean{band.mean, cgh, access::one_to_one{}, write_only, no_init};
    cgh.parallel_for<CovarianceMeanKernel>(range<1>(antennas),
                                           [=](item<1> item) {
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

void submit_stddev(celerity::distr_queue &queue, BandBuffers &band,
                   const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    accessor data{band.data, cgh,
                  [snapshots](chunk<1> chunk) {
                    return subrange<2>{{chunk.offset[0], 0},
                                       {chunk.range[0], snapshots}};
                  },
                  read_only};
    accessor mean{band.mean, cgh, access::one_to_one{}, read_only};
    accessor stddev{band.stddev, cgh, access::one_to_one{}, write_only,
                    no_init};
    cgh.parallel_for<CovarianceStddevKernel>(range<1>(antennas),
                                             [=](item<1> item) {
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

void submit_normalize(celerity::distr_queue &queue, BandBuffers &band,
                      const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    accessor data{band.data, cgh, access::one_to_one{}, read_write};
    const auto antenna_slice = [](chunk<2> chunk) {
      return subrange<1>{{chunk.offset[0]}, {chunk.range[0]}};
    };
    accessor mean{band.mean, cgh, antenna_slice, read_only};
    accessor stddev{band.stddev, cgh, antenna_slice, read_only};
    cgh.parallel_for<CovarianceNormalizeKernel>(
        range<2>(antennas, snapshots), [=](item<2> item) {
          const std::size_t antenna = item[0];
          data[item] =
              (data[item] - mean[antenna]) / stddev[antenna];
        });
  });
  maybe_wait(queue, config);
}

void submit_row_serial_correlation(celerity::distr_queue &queue,
                                   BandBuffers &band,
                                   const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const bool folded =
      config.covariance_order == cs::CovarianceOrder::Folded;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    // This is the exact rectangular envelope of all sample rows touched by a
    // work chunk. Each logical row j reads [j, antennas), so the union is the
    // suffix beginning at the smallest j assigned to the chunk.
    accessor data{band.data, cgh,
                  [antennas, snapshots, folded](chunk<1> chunk) {
                    std::size_t first = antennas;
                    const std::size_t end = chunk.offset[0] + chunk.range[0];
                    for (std::size_t work_row = chunk.offset[0];
                         work_row < end; ++work_row) {
                      const std::size_t antenna0 =
                          folded ? ((work_row & 1U) == 0U
                                        ? work_row / 2
                                        : antennas - 1 - work_row / 2)
                                 : work_row;
                      first = std::min(first, antenna0);
                    }
                    if (first == antennas) {
                      return subrange<2>{{0, 0}, {0, snapshots}};
                    }
                    return subrange<2>{{first, 0},
                                       {antennas - first, snapshots}};
                  },
                  read_only};
    accessor basis{band.basis, cgh,
                   [antennas](chunk<1> chunk) {
                     return subrange<2>{{chunk.offset[0], 0},
                                        {chunk.range[0], antennas}};
                   },
                   write_only, no_init};
    cgh.parallel_for<CovarianceTriangleKernel>(
        range<1>(antennas), [=](item<1> item) {
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

void submit_pair_parallel_correlation(celerity::distr_queue &queue,
                                      BandBuffers &band,
                                      const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  const std::size_t snapshots = config.snapshots;
  const bool folded =
      config.covariance_order == cs::CovarianceOrder::Folded;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    // A pair-parallel chunk reads the sample rows selected by its work-row
    // interval and by its column interval. The smallest qualifying work row
    // through the last column is their tight contiguous envelope.
    accessor data{band.data, cgh,
                  [antennas, snapshots, folded](chunk<2> chunk) {
                    if (chunk.range[0] == 0 || chunk.range[1] == 0) {
                      return subrange<2>{{0, 0}, {0, snapshots}};
                    }
                    const std::size_t last_column =
                        chunk.offset[1] + chunk.range[1] - 1;
                    std::size_t first = antennas;
                    const std::size_t row_end =
                        chunk.offset[0] + chunk.range[0];
                    for (std::size_t work_row = chunk.offset[0];
                         work_row < row_end; ++work_row) {
                      const std::size_t antenna0 =
                          folded ? ((work_row & 1U) == 0U
                                        ? work_row / 2
                                        : antennas - 1 - work_row / 2)
                                 : work_row;
                      if (antenna0 <= last_column) {
                        first = std::min(first, antenna0);
                      }
                    }
                    if (first == antennas) {
                      return subrange<2>{{0, 0}, {0, snapshots}};
                    }
                    return subrange<2>{{first, 0},
                                       {last_column - first + 1, snapshots}};
                  },
                  read_only};
    accessor basis{band.basis, cgh, access::one_to_one{}, write_only,
                   no_init};
    cgh.parallel_for<CovarianceTrianglePairKernel>(
        range<2>(antennas, antennas), [=](item<2> item) {
          const std::size_t work_row = item[0];
          const std::size_t antenna1 = item[1];
          const std::size_t antenna0 =
              folded ? ((work_row & 1U) == 0U
                            ? work_row / 2
                            : antennas - 1 - work_row / 2)
                     : work_row;
          if (antenna1 < antenna0) {
            basis[item] = 0.0f;
            return;
          }
          float dot = 0.0f;
          for (std::size_t sample = 0; sample < snapshots; ++sample) {
            dot += data[{antenna0, sample}] * data[{antenna1, sample}];
          }
          float correlation = dot / static_cast<float>(snapshots);
          if (antenna1 == antenna0) {
            correlation += 0.25f;
          }
          basis[item] = correlation;
        });
  });
  maybe_wait(queue, config);
}

void submit_triangular_correlation(celerity::distr_queue &queue,
                                   BandBuffers &band,
                                   const cs::Config &config) {
  if (config.correlation_layout == cs::CorrelationLayout::PairParallel) {
    submit_pair_parallel_correlation(queue, band, config);
  } else {
    submit_row_serial_correlation(queue, band, config);
  }
}

void submit_reverse_panel(celerity::distr_queue &queue, BandBuffers &band,
                          std::size_t panel_begin, std::size_t panel_end,
                          const cs::Config &config) {
  const std::size_t antennas = config.antennas;
  constexpr std::size_t local_size = 128;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    accessor basis{
        band.basis, cgh,
        access::fixed<2>(subrange<2>{{panel_begin, 0},
                                     {panel_end - panel_begin, antennas}}),
        read_write};
    local_accessor<float, 1> scratch{range<1>(local_size), cgh};
    cgh.parallel_for<ReverseBlockGSPanelKernel>(
        nd_range<1>{range<1>(local_size), range<1>(local_size)},
        [=](nd_item<1> item) {
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
        celerity::group_barrier(item.get_group());
        for (std::size_t stride = local_size / 2; stride > 0; stride /= 2) {
          if (lane < stride) {
            scratch[lane] += scratch[lane + stride];
          }
          celerity::group_barrier(item.get_group());
        }
        const float inverse_norm =
            sycl::rsqrt(sycl::fmax(scratch[0], 1.0e-20f));
        for (std::size_t col = lane; col < antennas; col += local_size) {
          basis[{pivot, col}] *= inverse_norm;
        }
        celerity::group_barrier(item.get_group());
        for (std::size_t row = panel_begin; row < pivot; ++row) {
          float projection = 0.0f;
          for (std::size_t col = lane; col < antennas; col += local_size) {
            projection += basis[{row, col}] * basis[{pivot, col}];
          }
          scratch[lane] = projection;
          celerity::group_barrier(item.get_group());
          for (std::size_t stride = local_size / 2; stride > 0;
               stride /= 2) {
            if (lane < stride) {
              scratch[lane] += scratch[lane + stride];
            }
            celerity::group_barrier(item.get_group());
          }
          projection = scratch[0];
          for (std::size_t col = lane; col < antennas; col += local_size) {
            basis[{row, col}] -= projection * basis[{pivot, col}];
          }
          celerity::group_barrier(item.get_group());
        }
      }
    });
  });
  maybe_wait(queue, config);
}

void submit_reverse_update(celerity::distr_queue &queue, BandBuffers &band,
                           std::size_t panel_begin, std::size_t panel_end,
                           const cs::Config &config) {
  if (panel_begin == 0) {
    return;
  }
  const std::size_t antennas = config.antennas;
  const std::size_t execution_rows =
      config.compact_gs ? panel_begin : config.subspace_rank;
  queue.submit([&](celerity::handler &cgh) {
    using namespace celerity;
    apply_hints(cgh, config);
    accessor prefix{
        band.basis, cgh,
        [panel_begin, antennas](chunk<1> chunk) {
          const std::size_t begin = std::min(chunk.offset[0], panel_begin);
          const std::size_t chunk_end = chunk.offset[0] + chunk.range[0];
          const std::size_t end = std::min(chunk_end, panel_begin);
          return subrange<2>{{begin, 0},
                             {end > begin ? end - begin : 0, antennas}};
        },
        read_write};
    accessor panel{
        band.basis, cgh,
        access::fixed<2>(subrange<2>{{panel_begin, 0},
                                     {panel_end - panel_begin, antennas}}),
        read_only};
    cgh.parallel_for<ReverseBlockGSUpdateKernel>(
        range<1>(execution_rows), [=](item<1> item) {
          const std::size_t row = item[0];
          if (row >= panel_begin) {
            return;
          }
          for (std::size_t pivot_cursor = panel_end;
               pivot_cursor > panel_begin; --pivot_cursor) {
            const std::size_t pivot = pivot_cursor - 1;
            float projection = 0.0f;
            for (std::size_t col = 0; col < antennas; ++col) {
              projection += prefix[{row, col}] * panel[{pivot, col}];
            }
            for (std::size_t col = 0; col < antennas; ++col) {
              prefix[{row, col}] -= projection * panel[{pivot, col}];
            }
          }
        });
  });
  maybe_wait(queue, config);
}

void submit_band(celerity::distr_queue &queue, BandBuffers &band,
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
    celerity::distr_queue &queue,
    std::vector<std::unique_ptr<BandBuffers>> &bands,
    const cs::Config &config) {
  std::vector<cs::VerificationResult> band_results(bands.size());
  for (std::size_t index = 0; index < bands.size(); ++index) {
    BandBuffers &band = *bands[index];
    cs::VerificationResult *result = &band_results[index];
    queue.submit([&](celerity::handler &cgh) {
      using namespace celerity;
      accessor basis{band.basis, cgh, access::all{}, read_only_host_task};
      cgh.host_task(on_master_node, [=] {
        *result = cs::verify_basis(config,
                                   [&](std::size_t row, std::size_t col) {
                                     return basis[{row, col}];
                                   });
      });
    });
  }
  queue.slow_full_sync();

  cs::VerificationResult total;
  for (const cs::VerificationResult &result : band_results) {
    cs::merge_verification(total, result);
  }
  return total;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const cs::Config config = cs::parse_config(argc, argv);
    cs::print_configuration(config, "celerity");

    const auto total_begin = clock_type::now();
    std::vector<std::unique_ptr<BandBuffers>> bands;
    const std::size_t tiles = cs::tile_count(config);
    bands.reserve(tiles);
    for (std::size_t time_block = 0; time_block < config.time_blocks;
         ++time_block) {
      for (std::size_t band = 0; band < config.bands; ++band) {
        const std::size_t tile = time_block * config.bands + band;
        bands.push_back(std::make_unique<BandBuffers>(tile, config));
      }
    }
    const auto init_end = clock_type::now();

    celerity::distr_queue queue;
    const auto queue_end = clock_type::now();

    const auto run_begin = clock_type::now();
    for (std::size_t epoch = 0; epoch < config.epochs; ++epoch) {
      const auto epoch_begin = clock_type::now();
      for (auto &band : bands) {
        submit_band(queue, *band, config);
      }
      queue.slow_full_sync();
      const auto epoch_end = clock_type::now();
      std::cout << std::fixed << std::setprecision(6)
                << "EPOCH index=" << epoch
                << " run_sec=" << seconds_between(epoch_begin, epoch_end)
                << '\n';
    }
    const auto run_end = clock_type::now();

    const auto host_begin = clock_type::now();
    const cs::VerificationResult result = read_result(queue, bands, config);
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
