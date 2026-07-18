#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/covariance_subspace_sycl_timer.cpp"
TARGET="$SCRIPT_DIR/covariance_subspace_sycl_timer"
SYCL_CXX="${CXX:-clang++}"
COMPILE_FLAGS=(-O3 -DNDEBUG -std=c++17 -fsycl)
if [[ -n "${SYCL_EXTRA_FLAGS:-}" ]]; then
  read -r -a EXTRA_FLAGS <<<"$SYCL_EXTRA_FLAGS"
  COMPILE_FLAGS+=("${EXTRA_FLAGS[@]}")
fi

"$SYCL_CXX" "${COMPILE_FLAGS[@]}" "$SOURCE" -o "$TARGET"

echo "built: $TARGET"
