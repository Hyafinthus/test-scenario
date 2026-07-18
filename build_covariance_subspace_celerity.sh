#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="covariance_subspace_celerity.cpp"
COMMON_HEADER="covariance_subspace_common.hpp"
PROJECT_FILE="covariance_subspace_celerity.CMakeLists.txt"
TARGET="covariance_subspace_celerity"
BUILD_ROOT="${BUILD_ROOT:-$SCRIPT_DIR/.covariance_subspace_celerity_build}"
BUILD_SOURCE="$BUILD_ROOT/src"
BUILD_DIR="$BUILD_ROOT/build"
CELERITY_INSTALL="${CELERITY_INSTALL:-$SCRIPT_DIR/../base-celerity/build/install}"

mkdir -p "$BUILD_SOURCE" "$BUILD_DIR"
cp "$SCRIPT_DIR/$SOURCE" "$BUILD_SOURCE/$SOURCE"
cp "$SCRIPT_DIR/$COMMON_HEADER" "$BUILD_SOURCE/$COMMON_HEADER"
cp "$SCRIPT_DIR/$PROJECT_FILE" "$BUILD_SOURCE/CMakeLists.txt"

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
