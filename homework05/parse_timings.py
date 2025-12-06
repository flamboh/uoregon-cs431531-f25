#!/usr/bin/env python3
"""
Convert the timing section printed by homework05/main.cc into CSV rows.
Usage:
    ./spmv ... | python3 parse_timings.py
The script emits one comma-separated row per timing block (no header). Columns:
load_time, vec_bcast_time, mat_scatter_time, lock_init_time,
coo_spmv_time, res_reduce_time, store_time
"""

import sys
from typing import Dict, List, Optional, Tuple

MODULES: List[Tuple[str, str]] = [
    ("Load", "load_time"),
    ("Vec Bcast", "vec_bcast_time"),
    ("Mat Scatter", "mat_scatter_time"),
    ("Lock Init", "lock_init_time"),
    ("COO SpMV", "coo_spmv_time"),
    ("Res Reduce", "res_reduce_time"),
    ("Store", "store_time"),
]

MODULE_NAMES = {name for name, _ in MODULES}


def parse_time_line(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        module, time_str = stripped.rsplit(maxsplit=1)
        module = module.strip()
        float(time_str)  # validate
    except ValueError:
        return None
    if module not in MODULE_NAMES:
        return None
    return module, time_str


def emit_row(times: Dict[str, str]) -> None:
    values = []
    for module_name, _ in MODULES:
        if module_name not in times:
            return
        values.append(times[module_name])
    print(",".join(values))


def main() -> None:
    collecting = False
    current_times: Dict[str, str] = {}
    for raw_line in sys.stdin:
        stripped = raw_line.strip()
        if stripped.startswith("Module") and "Time" in stripped:
            collecting = True
            current_times = {}
            continue
        parsed = parse_time_line(raw_line)
        if parsed is None:
            continue
        if not collecting:
            collecting = True
            current_times = {}
        module, time_str = parsed
        current_times[module] = time_str
        if len(current_times) == len(MODULES):
            emit_row(current_times)
            collecting = False


if __name__ == "__main__":
    main()
