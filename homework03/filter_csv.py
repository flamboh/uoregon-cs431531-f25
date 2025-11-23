#!/usr/bin/env python3
"""
Filter stdout from spmv runs to only include CSV timing lines.
Usage: ./spmv ... | python3 filter_csv.py
"""
import sys

EXPECTED_COLUMNS = 9


def is_header(parts):
    return parts[0] == "num_threads" and len(parts) == EXPECTED_COLUMNS


def is_data(parts):
    if len(parts) != EXPECTED_COLUMNS:
        return False
    try:
        int(parts[0])
        for value in parts[1:]:
            float(value)
    except ValueError:
        return False
    return True


def main() -> None:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if is_header(parts) or is_data(parts):
            print(stripped)


if __name__ == "__main__":
    main()
