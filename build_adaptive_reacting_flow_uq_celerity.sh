#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="adaptive_reacting_flow_uq_celerity.cpp"
TARGET="adaptive_reacting_flow_uq_celerity"
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/.arf_celerity_build}"
BUILD_SOURCE="$BUILD_ROOT/src"
BUILD_DIR="$BUILD_ROOT/build"

: "${CELERITY_INSTALL:?Set CELERITY_INSTALL to the Celerity 0.6 install prefix}"

mkdir -p "$BUILD_SOURCE" "$BUILD_DIR"
cp "$SCRIPT_DIR/$SOURCE" "$BUILD_SOURCE/$SOURCE"

cat >"$BUILD_SOURCE/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.13)
project(adaptive_reacting_flow_uq_celerity LANGUAGES CXX)

find_package(Celerity 0.6.0 REQUIRED)
add_executable(adaptive_reacting_flow_uq_celerity
               adaptive_reacting_flow_uq_celerity.cpp)
set_property(TARGET adaptive_reacting_flow_uq_celerity
             PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
add_celerity_to_target(
  TARGET adaptive_reacting_flow_uq_celerity
  SOURCES adaptive_reacting_flow_uq_celerity.cpp)
EOF

CMAKE_ARGS=(
  -S "$BUILD_SOURCE"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}"
  -DCELERITY_SYCL_IMPL="${CELERITY_SYCL_IMPL:-DPC++}"
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DCMAKE_PREFIX_PATH="$CELERITY_INSTALL"
)

if [[ -n "${CELERITY_DPCPP_TARGETS:-}" ]]; then
  CMAKE_ARGS+=("-DCELERITY_DPCPP_TARGETS=$CELERITY_DPCPP_TARGETS")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target "$TARGET"
cmake -E copy "$BUILD_DIR/$TARGET" "$SCRIPT_DIR/$TARGET"
echo "built: $SCRIPT_DIR/$TARGET"
