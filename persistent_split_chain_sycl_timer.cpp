#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Config {
  std::size_t Items = 4 * 1024 * 1024;
  int Stages = 4;
  int Epochs = 2;
  int InnerIters = 4096;
  bool WaitEachKernel = false;
};

std::uint64_t mix(std::uint64_t x, std::uint64_t salt, int inner_iters) {
  x ^= salt;
  for (int i = 0; i < inner_iters; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x += 0x9e3779b97f4a7c15ULL + static_cast<std::uint64_t>(i);
  }
  return x;
}

std::size_t parseSize(const char *text, const char *name) {
  const unsigned long long value = std::strtoull(text, nullptr, 10);
  if (value == 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

int parseInt(const char *text, const char *name) {
  const long value = std::strtol(text, nullptr, 10);
  if (value <= 0 || value > 1'000'000) {
    throw std::invalid_argument(std::string(name) + " is out of range");
  }
  return static_cast<int>(value);
}

Config parseArgs(int argc, char **argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char *name) -> const char * {
      if (++i >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[i];
    };
    if (arg == "--items") {
      config.Items = parseSize(next("--items"), "items");
    } else if (arg == "--stages") {
      config.Stages = parseInt(next("--stages"), "stages");
    } else if (arg == "--epochs") {
      config.Epochs = parseInt(next("--epochs"), "epochs");
    } else if (arg == "--inner-iters") {
      config.InnerIters = parseInt(next("--inner-iters"), "inner-iters");
    } else if (arg == "--wait-each-kernel") {
      config.WaitEachKernel = parseInt(next("--wait-each-kernel"),
                                       "wait-each-kernel") != 0;
    } else if (arg == "--help") {
      std::cout << "--items N --stages N --epochs N --inner-iters N "
                   "--wait-each-kernel 0|1\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return config;
}

class PersistentSplitChainAtoBKernel;
class PersistentSplitChainBtoAKernel;

} // namespace

int main(int argc, char **argv) {
  try {
    const Config config = parseArgs(argc, argv);
    std::vector<std::uint64_t> a(config.Items);
    std::vector<std::uint64_t> b(config.Items, 0);
    for (std::size_t i = 0; i < config.Items; ++i) {
      a[i] = 0xd1b54a32d192ed03ULL ^ static_cast<std::uint64_t>(i);
    }

    sycl::queue queue{sycl::gpu_selector_v,
                      sycl::property::queue::in_order{}};
    sycl::buffer<std::uint64_t, 1> a_buffer(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<std::uint64_t, 1> b_buffer(b.data(), sycl::range<1>(b.size()));

    std::cout << "CONFIG items=" << config.Items
              << " stages=" << config.Stages << " epochs=" << config.Epochs
              << " inner_iters=" << config.InnerIters
              << " wait_each_kernel=" << config.WaitEachKernel << '\n';
    std::cout << "DAG kernels_per_window=" << config.Stages
              << " max_width=1 critical_path_levels=" << config.Stages
              << " total_kernels=" << config.Stages * config.Epochs << '\n';
    std::cout << "MEMORY total_bytes="
              << 2ULL * config.Items * sizeof(std::uint64_t) << '\n';

    const auto run_start = std::chrono::steady_clock::now();
    bool source_is_a = true;
    for (int epoch = 0; epoch < config.Epochs; ++epoch) {
      for (int stage = 0; stage < config.Stages; ++stage) {
        const std::uint64_t salt =
            (static_cast<std::uint64_t>(epoch) << 32) |
            static_cast<std::uint32_t>(stage + 1);
        if (source_is_a) {
          queue.submit([&](sycl::handler &handler) {
            auto src = a_buffer.get_access<sycl::access::mode::read>(handler);
            auto dst = b_buffer.get_access<sycl::access::mode::discard_write>(
                handler);
            handler.ext_snmd_partition_local(src);
            handler.ext_snmd_partition_local(dst);
            handler.parallel_for<PersistentSplitChainAtoBKernel>(
                sycl::range<1>(config.Items), [=](sycl::id<1> id) {
                  dst[id] = mix(src[id], salt, config.InnerIters);
                });
          });
        } else {
          queue.submit([&](sycl::handler &handler) {
            auto src = b_buffer.get_access<sycl::access::mode::read>(handler);
            auto dst = a_buffer.get_access<sycl::access::mode::discard_write>(
                handler);
            handler.ext_snmd_partition_local(src);
            handler.ext_snmd_partition_local(dst);
            handler.parallel_for<PersistentSplitChainBtoAKernel>(
                sycl::range<1>(config.Items), [=](sycl::id<1> id) {
                  dst[id] = mix(src[id], salt, config.InnerIters);
                });
          });
        }
        source_is_a = !source_is_a;
        if (config.WaitEachKernel) {
          queue.wait_and_throw();
        }
      }
      queue.wait_and_throw();
    }
    const auto run_end = std::chrono::steady_clock::now();

    const std::size_t sample_count = std::min<std::size_t>(64, config.Items);
    bool passed = true;
    std::uint64_t checksum = 0;
    if (source_is_a) {
      auto result = a_buffer.get_host_access<sycl::access::mode::read>();
      for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::size_t index = sample * config.Items / sample_count;
        std::uint64_t expected =
            0xd1b54a32d192ed03ULL ^ static_cast<std::uint64_t>(index);
        for (int epoch = 0; epoch < config.Epochs; ++epoch) {
          for (int stage = 0; stage < config.Stages; ++stage) {
            const std::uint64_t salt =
                (static_cast<std::uint64_t>(epoch) << 32) |
                static_cast<std::uint32_t>(stage + 1);
            expected = mix(expected, salt, config.InnerIters);
          }
        }
        checksum ^= result[index];
        passed = passed && result[index] == expected;
      }
    } else {
      auto result = b_buffer.get_host_access<sycl::access::mode::read>();
      for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::size_t index = sample * config.Items / sample_count;
        std::uint64_t expected =
            0xd1b54a32d192ed03ULL ^ static_cast<std::uint64_t>(index);
        for (int epoch = 0; epoch < config.Epochs; ++epoch) {
          for (int stage = 0; stage < config.Stages; ++stage) {
            const std::uint64_t salt =
                (static_cast<std::uint64_t>(epoch) << 32) |
                static_cast<std::uint32_t>(stage + 1);
            expected = mix(expected, salt, config.InnerIters);
          }
        }
        checksum ^= result[index];
        passed = passed && result[index] == expected;
      }
    }

    const double run_sec =
        std::chrono::duration<double>(run_end - run_start).count();
    std::cout << std::fixed << std::setprecision(6)
              << "TIMING run_sec=" << run_sec << '\n';
    std::cout << "RESULT checksum=" << checksum
              << " checked_items=" << sample_count << '\n';
    std::cout << "VERIFY passed=" << (passed ? 1 : 0) << '\n';
    return passed ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return 1;
  }
}
