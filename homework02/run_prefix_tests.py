#!/usr/bin/env python3
"""
Utility for running the prefix benchmark locally across a range of thread counts.

This script repeatedly invokes ./prefix with --csv enabled, captures the runtimes,
and writes both the raw per-run data and aggregated averages to CSV files.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class RunResult:
    thread: int
    num_items: int
    serial_time: float
    nlogn_time: float
    n_time: float


def parse_threads(spec: str) -> List[int]:
    """Parse thread spec like '1-8,12,16' into a sorted unique list."""
    values = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            for val in range(start, end + step, step):
                values.add(val)
        else:
            values.add(int(part))
    if not values:
        raise ValueError("No thread counts parsed from specification.")
    return sorted(values)


def run_prefix_binary(
    binary: Path,
    size: int,
    seed: int,
    thread: int,
) -> RunResult:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(thread)

    completed = subprocess.run(
        [str(binary), str(size), str(seed), "--csv"],
        cwd=binary.parent,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Prefix binary produced no output.")
    last = lines[-1]
    parts = last.split(",")
    if len(parts) != 5:
        raise RuntimeError(f"Unexpected CSV line: {last!r}")

    return RunResult(
        thread=int(parts[0]),
        num_items=int(parts[1]),
        serial_time=float(parts[2]),
        nlogn_time=float(parts[3]),
        n_time=float(parts[4]),
    )


def aggregate_results(results: Iterable[RunResult]) -> List[RunResult]:
    buckets: Dict[int, List[RunResult]] = defaultdict(list)
    for res in results:
        buckets[res.thread].append(res)

    averaged: List[RunResult] = []
    for thread, bucket in sorted(buckets.items()):
        count = len(bucket)
        num_items = bucket[0].num_items
        averaged.append(
            RunResult(
                thread=thread,
                num_items=num_items,
                serial_time=sum(r.serial_time for r in bucket) / count,
                nlogn_time=sum(r.nlogn_time for r in bucket) / count,
                n_time=sum(r.n_time for r in bucket) / count,
            )
        )
    return averaged


def write_csv(path: Path, rows: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["num_threads", "num_items", "serial_time", "nlogn_time", "n_time"])
        for row in rows:
            writer.writerow(
                [row.thread, row.num_items, row.serial_time, row.nlogn_time, row.n_time]
            )


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True, help="Number of elements.")
    parser.add_argument(
        "--threads",
        type=str,
        default="1",
        help="Comma-separated thread counts, ranges allowed (e.g. '1-8,12').",
    )
    parser.add_argument("--runs", type=int, default=5, help="Runs per thread.")
    parser.add_argument("--seed", type=int, default=5, help="Random seed for prefix binary.")
    parser.add_argument(
        "--raw-out",
        type=Path,
        required=True,
        help="Path to write raw per-run CSV data.",
    )
    parser.add_argument(
        "--avg-out",
        type=Path,
        required=True,
        help="Path to write averaged CSV data.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("./prefix"),
        help="Path to the compiled prefix executable.",
    )

    args = parser.parse_args(argv)
    threads = parse_threads(args.threads)

    all_results: List[RunResult] = []
    for run_idx in range(1, args.runs + 1):
        for thread in threads:
            print(
                f"[run {run_idx}/{args.runs}] threads={thread} size={args.size}",
                file=sys.stderr,
            )
            result = run_prefix_binary(
                binary=args.binary.resolve(),
                size=args.size,
                seed=args.seed,
                thread=thread,
            )
            all_results.append(result)

    write_csv(args.raw_out, all_results)
    write_csv(args.avg_out, aggregate_results(all_results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
