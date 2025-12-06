#!/usr/bin/env python3
"""
Average the timing CSV rows stored in ./out/*.out and append num_threads.
Usage:
    python3 aggregate_timings.py [directory]
If no directory is provided, ./out (next to this script) is used.
Output order matches:
load_time,vec_bcast_time,matrix_scatter_time,lock_init_time,
coo_spmv_time,res_reduce_time,store_time,num_threads
"""

import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

COLUMN_NAMES = [
    "load_time",
    "vec_bcast_time",
    "matrix_scatter_time",
    "lock_init_time",
    "coo_spmv_time",
    "res_reduce_time",
    "store_time",
]

CPU_PATTERN = re.compile(r"(\d+)cpu", re.IGNORECASE)


def parse_file(path: Path) -> Optional[Tuple[List[float], int]]:
    match = CPU_PATTERN.search(path.stem)
    if not match:
        return None
    num_threads = int(match.group(1))
    sums = [0.0] * len(COLUMN_NAMES)
    count = 0
    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) < len(COLUMN_NAMES):
                continue
            values = parts[-len(COLUMN_NAMES) :]
            try:
                floats = [float(v) for v in values]
            except ValueError:
                continue
            for idx, value in enumerate(floats):
                sums[idx] += value
            count += 1
    if count == 0:
        return None
    averages = [total / count for total in sums]
    return averages, num_threads


def format_row(values: Sequence[float], num_threads: int) -> str:
    formatted = [f"{value:.6f}" for value in values]
    formatted.append(str(num_threads))
    return ",".join(formatted)


def find_directory(argv: Sequence[str]) -> Path:
    if len(argv) > 1:
        return Path(argv[1]).expanduser().resolve()
    return (Path(__file__).resolve().parent / "out").resolve()


def main(argv: Sequence[str]) -> None:
    directory = find_directory(argv)
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")
    rows: List[Tuple[int, List[float]]] = []
    for path in sorted(directory.glob("*cpu.out")):
        parsed = parse_file(path)
        if parsed is None:
            continue
        averages, num_threads = parsed
        rows.append((num_threads, averages))
    for num_threads, averages in sorted(rows, key=lambda item: item[0]):
        print(format_row(averages, num_threads))


if __name__ == "__main__":
    main(sys.argv)
