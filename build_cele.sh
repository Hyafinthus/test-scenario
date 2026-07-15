#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$HOME/SYCL/test-scenario"
cd "$SCRIPT_DIR"

CELERITY_INSTALL="${CELERITY_INSTALL:-$HOME/SYCL/base-celerity/build/install}"
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/.cele_dpcpp_cuda_build}"
BUILD_SRC_DIR="$BUILD_ROOT/src"
BUILD_DIR="$BUILD_ROOT/build"

SOURCE="mc_radiotherapy_cele_timer.cpp"
TARGET="mc_radiotherapy_cele_timer"

mkdir -p "$BUILD_SRC_DIR" "$BUILD_DIR"

if [[ ! -f "$SCRIPT_DIR/$SOURCE" ]]; then
  echo "ERROR: source not found: $SOURCE" >&2
  exit 1
fi

cp "$SCRIPT_DIR/$SOURCE" "$BUILD_SRC_DIR/$SOURCE"

cat > "$BUILD_SRC_DIR/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.13)
project(mc_radiotherapy_cele_timer_dpcpp_cuda LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)

add_executable(mc_radiotherapy_cele_timer mc_radiotherapy_cele_timer.cpp)
set_property(TARGET mc_radiotherapy_cele_timer PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(
  TARGET mc_radiotherapy_cele_timer
  SOURCES mc_radiotherapy_cele_timer.cpp
)
EOF

cmake -S "$BUILD_SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_CXX_COMPILER="$(which clang++)" \
  -DCELERITY_SYCL_IMPL=DPC++ \
  -DCELERITY_DPCPP_TARGETS="nvptx64-nvidia-cuda" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CELERITY_INSTALL"

ninja -C "$BUILD_DIR" "$TARGET"

cp "$BUILD_DIR/$TARGET" "$SCRIPT_DIR/$TARGET"
echo "built: $SCRIPT_DIR/$TARGET"
