#!/usr/bin/env python3
"""Summarize MC runtime placement, monitor, and GPU timeline evidence."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


CONFIG_RE = re.compile(r"\bscenarios=(\d+)")
ASSIGN_RE = re.compile(
    r"algorithmHEFT: Kernel (\d+) assigned to Rank (\d+) Proc (\d+) "
    r"NumParts (\d+)"
)
COMPLETION_DISPATCH_RE = re.compile(
    r"CompletionQueue: dispatch_order \d+ kernel (\d+) rank (\d+) "
    r"proc (\d+) parts (\d+)"
)
COMPLETION_DONE_RE = re.compile(
    r"CompletionQueue: complete kernel (\d+) device (\d+) parts (\d+)"
)
PROFILE_RE = re.compile(
    r"Offline profiling kernel_count: (\d+) device_index: (\d+) "
    r"num_parts: (\d+) start_ns: (\d+) end_ns: (\d+)"
)
MONITOR_RE = re.compile(
    r"ensureOfflineDeviceModel: Rank (\d+) Proc (\d+).*? Util ([0-9.eE+-]+) "
    r"FreshProfile (\d+) ServiceTimeScale ([0-9.eE+-]+) "
    r"AvailableTime ([0-9.eE+-]+)"
)
SUBMIT_RE = re.compile(
    r"Offline submit kernel_count: (\d+) host_duration_ns: (\d+)"
)


def epoch_and_scenario(kernel_count: int, scenarios: int) -> tuple[int, int]:
    return (kernel_count - 1) // scenarios + 1, (kernel_count - 1) % scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--scenarios",
        type=int,
        help="override scenarios when the CONFIG line is absent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.log.read_text(errors="replace")
    scenarios = args.scenarios
    config_match = CONFIG_RE.search(text)
    if scenarios is None and config_match:
        scenarios = int(config_match.group(1))
    if not scenarios or scenarios <= 0:
        raise SystemExit("scenarios is missing; pass --scenarios")

    placements: dict[int, tuple[int, int, int]] = {}
    for match in ASSIGN_RE.finditer(text):
        kernel, rank, proc, parts = map(int, match.groups())
        placements[kernel] = (rank, proc, parts)

    completion_placements: dict[int, tuple[int, int, int]] = {}
    for match in COMPLETION_DISPATCH_RE.finditer(text):
        kernel, rank, proc, parts = map(int, match.groups())
        completion_placements[kernel] = (rank, proc, parts)
    if completion_placements:
        placements = completion_placements

    completion_done = [
        tuple(map(int, match.groups()))
        for match in COMPLETION_DONE_RE.finditer(text)
    ]

    profiles: dict[int, tuple[int, int, int, int]] = {}
    for match in PROFILE_RE.finditer(text):
        kernel, proc, parts, start_ns, end_ns = map(int, match.groups())
        profiles[kernel] = (proc, parts, start_ns, end_ns)

    monitor_snapshots: list[dict[tuple[int, int], tuple[float, int, float, float]]] = []
    snapshot: dict[tuple[int, int], tuple[float, int, float, float]] = {}
    for match in MONITOR_RE.finditer(text):
        rank, proc = int(match.group(1)), int(match.group(2))
        key = (rank, proc)
        if key in snapshot:
            monitor_snapshots.append(snapshot)
            snapshot = {}
        snapshot[key] = (
            float(match.group(3)),
            int(match.group(4)),
            float(match.group(5)),
            float(match.group(6)),
        )
    if snapshot:
        monitor_snapshots.append(snapshot)

    submits = {
        int(match.group(1)): int(match.group(2))
        for match in SUBMIT_RE.finditer(text)
    }

    print(
        f"SUMMARY scenarios={scenarios} placements={len(placements)} "
        f"profiles={len(profiles)} monitor_snapshots={len(monitor_snapshots)} "
        f"completion_dispatch={len(completion_placements)} "
        f"completion_done={len(completion_done)}"
    )
    if completion_placements:
        done_counts = collections.Counter(kernel for kernel, _proc, _parts in completion_done)
        missing_done = sorted(set(completion_placements) - set(done_counts))
        duplicate_done = sorted(
            kernel for kernel, count in done_counts.items() if count != 1
        )
        unknown_done = sorted(set(done_counts) - set(completion_placements))
        print(
            "COMPLETION "
            f"missing_done={missing_done or '-'} "
            f"duplicate_done={duplicate_done or '-'} "
            f"unknown_done={unknown_done or '-'}"
        )

    max_kernel = max(placements.keys() | profiles.keys(), default=0)
    epoch_count = (max_kernel + scenarios - 1) // scenarios
    previous_scenario_placement: dict[int, tuple[int, int]] = {}
    total_migrations = 0
    total_split = 0

    for epoch in range(1, epoch_count + 1):
        gpu_counts: collections.Counter[str] = collections.Counter()
        migrations = 0
        split_count = 0
        for scenario in range(scenarios):
            kernel = (epoch - 1) * scenarios + scenario + 1
            if kernel not in placements:
                continue
            rank, proc, parts = placements[kernel]
            gpu_counts[f"r{rank}p{proc}"] += 1
            split_count += int(parts > 1)
            placement = (rank, proc)
            if (
                scenario in previous_scenario_placement
                and previous_scenario_placement[scenario] != placement
            ):
                migrations += 1
            previous_scenario_placement[scenario] = placement

        total_migrations += migrations
        total_split += split_count
        counts = ",".join(f"{gpu}:{gpu_counts[gpu]}" for gpu in sorted(gpu_counts))
        print(
            f"EPOCH {epoch} migrations={migrations} split={split_count} "
            f"gpu_counts={counts or '-'}"
        )

        epoch_profiles: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for scenario in range(scenarios):
            kernel = (epoch - 1) * scenarios + scenario + 1
            if kernel not in profiles:
                continue
            proc, _parts, start_ns, end_ns = profiles[kernel]
            epoch_profiles[proc].append((start_ns, end_ns))

        if epoch_profiles:
            epoch_start = min(start for items in epoch_profiles.values() for start, _ in items)
            for proc in sorted(epoch_profiles):
                intervals = sorted(epoch_profiles[proc])
                first_delay = (intervals[0][0] - epoch_start) / 1e9
                max_gap = max(
                    (intervals[index][0] - intervals[index - 1][1]) / 1e9
                    for index in range(1, len(intervals))
                ) if len(intervals) > 1 else 0.0
                print(
                    f"  GPU p{proc} kernels={len(intervals)} "
                    f"first_delay_sec={first_delay:.6f} "
                    f"max_internal_gap_sec={max_gap:.6f}"
                )

    print(f"TOTAL migrations={total_migrations} split={total_split}")

    for index, values in enumerate(monitor_snapshots, start=1):
        gpu_values = [value for (_rank, proc), value in values.items() if proc > 0]
        scales = sorted({value[2] for value in gpu_values})
        available = sorted({value[3] for value in gpu_values})
        utils = ",".join(str(value[0]) for value in gpu_values)
        fresh = ",".join(str(value[1]) for value in gpu_values)
        print(
            f"MONITOR {index} scales={scales} available={available} "
            f"utils={utils or '-'} fresh={fresh or '-'}"
        )

    if submits:
        slowest_kernel, slowest_ns = max(submits.items(), key=lambda item: item[1])
        print(
            f"SUBMIT samples={len(submits)} slowest_kernel={slowest_kernel} "
            f"max_host_duration_sec={slowest_ns / 1e9:.6f}"
        )
    else:
        print("SUBMIT samples=0 (rerun with PRINT_HANDLER_TRACE after rebuilding)")


if __name__ == "__main__":
    main()
