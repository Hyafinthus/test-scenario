#!/usr/bin/env python3
"""Validate the adaptive reacting-flow read-replica A/B experiment.

The input files may contain one run or several appended runs.  The script
checks numerical equivalence, summarizes wall time, identifies chemistry
kernels from the documented per-window submission order, and verifies that
the fixed runtime prepared the four shared mechanism tables on multiple GPUs
once and retained their version-valid replicas across wait windows.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path


CONFIG_RE = re.compile(r"^CONFIG .*\bpatches=(\d+)\b.*\bsteps=(\d+)\b")
DAG_RE = re.compile(r"^DAG kernels_per_window=(\d+)\b")
RUN_RE = re.compile(r"^TIMING run_sec=([0-9.eE+-]+)\s*$")
RESULT_RE = re.compile(r"^RESULT checksum=([0-9.eE+-]+)\b")
VERIFY_RE = re.compile(r"^VERIFY passed=(\d+)\b")
DISPATCH_RE = re.compile(
    r"CompletionQueue dispatched kernel_count: (\d+) .*"
    r"host_duration_ns: (\d+)"
)
DEVICE_RE = re.compile(
    r"Offline profiling kernel_count: (\d+) .* duration_ns: (\d+)"
)
REPLICA_RE = re.compile(
    r"prepared dispatch read replica .*memobj: (0x[0-9a-fA-F]+) "
    r"bytes: (\d+) device_index: (\d+)"
)
RETAINED_RE = re.compile(
    r"retained cross-wait read replica objects: (\d+)"
)


@dataclass
class LogData:
    path: Path
    patches: int | None = None
    steps: int | None = None
    kernels_per_window: int | None = None
    run_seconds: list[float] = field(default_factory=list)
    checksums: list[float] = field(default_factory=list)
    verify_values: list[int] = field(default_factory=list)
    host_ns_by_kernel: dict[int, list[int]] = field(default_factory=dict)
    device_ns_by_kernel: dict[int, list[int]] = field(default_factory=dict)
    replica_devices: dict[str, set[int]] = field(default_factory=dict)
    replica_bytes: dict[str, int] = field(default_factory=dict)
    replica_prepare_counts: dict[str, int] = field(default_factory=dict)
    retained_replica_objects: list[int] = field(default_factory=list)


def append_sample(samples: dict[int, list[int]], key: int, value: int) -> None:
    samples.setdefault(key, []).append(value)


def parse_log(path: Path) -> LogData:
    data = LogData(path=path)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            match = CONFIG_RE.match(line)
            if match:
                patches, steps = (int(value) for value in match.groups())
                if data.patches is not None and data.patches != patches:
                    raise ValueError(f"{path}: mixed patch counts")
                if data.steps is not None and data.steps != steps:
                    raise ValueError(f"{path}: mixed step counts")
                data.patches = patches
                data.steps = steps
                continue
            match = DAG_RE.match(line)
            if match:
                kernels_per_window = int(match.group(1))
                if (
                    data.kernels_per_window is not None
                    and data.kernels_per_window != kernels_per_window
                ):
                    raise ValueError(f"{path}: mixed kernels_per_window")
                data.kernels_per_window = kernels_per_window
                continue
            match = RUN_RE.match(line)
            if match:
                data.run_seconds.append(float(match.group(1)))
                continue
            match = RESULT_RE.match(line)
            if match:
                data.checksums.append(float(match.group(1)))
                continue
            match = VERIFY_RE.match(line)
            if match:
                data.verify_values.append(int(match.group(1)))
                continue
            match = DISPATCH_RE.search(line)
            if match:
                kernel, duration = (int(value) for value in match.groups())
                append_sample(data.host_ns_by_kernel, kernel, duration)
                continue
            match = DEVICE_RE.search(line)
            if match:
                kernel, duration = (int(value) for value in match.groups())
                append_sample(data.device_ns_by_kernel, kernel, duration)
                continue
            match = REPLICA_RE.search(line)
            if match:
                memobj, byte_count, device = match.groups()
                data.replica_devices.setdefault(memobj, set()).add(int(device))
                data.replica_bytes[memobj] = int(byte_count)
                data.replica_prepare_counts[memobj] = (
                    data.replica_prepare_counts.get(memobj, 0) + 1
                )
                continue
            match = RETAINED_RE.search(line)
            if match:
                data.retained_replica_objects.append(int(match.group(1)))
    return data


def require_metadata(data: LogData) -> None:
    missing = []
    if data.patches is None:
        missing.append("CONFIG patches")
    if data.steps is None:
        missing.append("CONFIG steps")
    if data.kernels_per_window is None:
        missing.append("DAG kernels_per_window")
    if not data.run_seconds:
        missing.append("TIMING run_sec")
    if not data.checksums:
        missing.append("RESULT checksum")
    if not data.verify_values:
        missing.append("VERIFY passed")
    if missing:
        raise ValueError(f"{data.path}: missing {', '.join(missing)}")


def is_chemistry_kernel(kernel: int, patches: int, kernels_per_window: int) -> bool:
    position = (kernel - 1) % kernels_per_window
    return patches <= position < 2 * patches


def chemistry_samples(data: LogData, source: dict[int, list[int]]) -> list[int]:
    assert data.patches is not None
    assert data.kernels_per_window is not None
    return [
        duration
        for kernel, durations in source.items()
        if is_chemistry_kernel(kernel, data.patches, data.kernels_per_window)
        for duration in durations
    ]


def median_or_nan(values: list[float] | list[int]) -> float:
    return float(statistics.median(values)) if values else math.nan


def checksum_equivalent(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= 1.0e-7 * scale


def summarize(label: str, data: LogData) -> dict[str, float]:
    chemistry_host = chemistry_samples(data, data.host_ns_by_kernel)
    chemistry_device = chemistry_samples(data, data.device_ns_by_kernel)
    run_median = median_or_nan(data.run_seconds)
    host_median_ms = median_or_nan(chemistry_host) / 1.0e6
    device_median_ms = median_or_nan(chemistry_device) / 1.0e6
    ratio = host_median_ms / device_median_ms if device_median_ms > 0 else math.nan
    print(
        f"{label}: runs={len(data.run_seconds)} run_median_sec={run_median:.6f} "
        f"chemistry_host_median_ms={host_median_ms:.3f} "
        f"chemistry_device_median_ms={device_median_ms:.3f} "
        f"host_to_device_ratio={ratio:.3f}"
    )
    return {
        "run_median": run_median,
        "host_median_ms": host_median_ms,
        "device_median_ms": device_median_ms,
        "host_to_device_ratio": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail unless structural and performance gates pass",
    )
    parser.add_argument("--min-speedup", type=float, default=1.20)
    parser.add_argument("--max-host-device-ratio", type=float, default=0.50)
    args = parser.parse_args()

    try:
        baseline = parse_log(args.baseline)
        fixed = parse_log(args.fixed)
        require_metadata(baseline)
        require_metadata(fixed)
    except (OSError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2

    baseline_summary = summarize("baseline", baseline)
    fixed_summary = summarize("fixed", fixed)

    failures: list[str] = []
    if (
        baseline.patches,
        baseline.steps,
        baseline.kernels_per_window,
    ) != (fixed.patches, fixed.steps, fixed.kernels_per_window):
        failures.append("baseline/fixed CONFIG or DAG shape differs")
    if any(value != 1 for value in baseline.verify_values):
        failures.append("baseline contains VERIFY passed=0")
    if any(value != 1 for value in fixed.verify_values):
        failures.append("fixed contains VERIFY passed=0")

    baseline_checksum = median_or_nan(baseline.checksums)
    fixed_checksum = median_or_nan(fixed.checksums)
    equivalent = checksum_equivalent(baseline_checksum, fixed_checksum)
    print(
        f"correctness: baseline_checksum={baseline_checksum:.9f} "
        f"fixed_checksum={fixed_checksum:.9f} equivalent={int(equivalent)}"
    )
    if not equivalent:
        failures.append("baseline/fixed checksums differ")
    if any(
        not checksum_equivalent(baseline_checksum, value)
        for value in baseline.checksums + fixed.checksums
    ):
        failures.append("at least one run checksum differs from the median")

    mechanism_replicas = {
        memobj: devices
        for memobj, devices in fixed.replica_devices.items()
        if fixed.replica_bytes.get(memobj) == 2048 and len(devices) >= 2
    }
    print(
        "replicas: "
        f"mechanism_tables_multi_device={len(mechanism_replicas)} "
        f"device_sets={sorted(sorted(devices) for devices in mechanism_replicas.values())}"
    )
    if len(mechanism_replicas) < 4:
        failures.append("fewer than four 2048-byte mechanism tables replicated")

    mechanism_prepare_count = sum(
        count
        for memobj, count in fixed.replica_prepare_counts.items()
        if fixed.replica_bytes.get(memobj) == 2048
    )
    retained_windows = sum(
        count >= 4 for count in fixed.retained_replica_objects
    )
    expected_retained_windows = max(
        0, (fixed.steps - 1) * len(fixed.run_seconds)
    )
    max_one_time_prepares = 16 * len(fixed.run_seconds)
    print(
        "cross_wait_cache: "
        f"mechanism_prepare_events={mechanism_prepare_count} "
        f"retained_windows_ge_4={retained_windows} "
        f"expected_min={expected_retained_windows}"
    )
    if mechanism_prepare_count > max_one_time_prepares:
        failures.append(
            "mechanism tables were prepared more than once per device/run"
        )
    if retained_windows < expected_retained_windows:
        failures.append(
            "read replicas were not retained across enough wait windows"
        )

    if fixed_summary["run_median"] <= 0:
        failures.append("fixed median run time is not positive")
        speedup = math.nan
    else:
        speedup = baseline_summary["run_median"] / fixed_summary["run_median"]
    print(f"performance: baseline_over_fixed={speedup:.3f}x")
    if not math.isfinite(speedup) or speedup < args.min_speedup:
        failures.append(
            f"speedup {speedup:.3f}x is below {args.min_speedup:.3f}x"
        )

    fixed_ratio = fixed_summary["host_to_device_ratio"]
    if not math.isfinite(fixed_ratio):
        failures.append("fixed log lacks chemistry host/device profiling")
    elif fixed_ratio > args.max_host_device_ratio:
        failures.append(
            f"fixed host/device ratio {fixed_ratio:.3f} exceeds "
            f"{args.max_host_device_ratio:.3f}"
        )
    baseline_ratio = baseline_summary["host_to_device_ratio"]
    if math.isfinite(baseline_ratio) and math.isfinite(fixed_ratio):
        if fixed_ratio >= baseline_ratio:
            failures.append("chemistry host/device ratio did not improve")

    status = "PASS" if not failures else "INCOMPLETE"
    print(f"STATUS {status}")
    for failure in failures:
        print(f"- {failure}")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
