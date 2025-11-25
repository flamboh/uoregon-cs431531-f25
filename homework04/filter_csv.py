#!/usr/bin/env python3
"""
Filter stdout from spmv runs to only include CSV timing lines.
Usage: ./spmv ... | python3 filter_csv.py
"""

import sys
from collections import deque


def is_data(parts):
    try:
        int(parts[0])
        for value in parts[1:]:
            float(value)
    except ValueError:
        return False
    return True


def main() -> None:
    q = deque(maxlen=7)
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split()]
        try:
            float(parts[-1])
            if parts[-1]:
                # print(parts)
                q.append(" ".join(parts[:-1]) + " " + str(float(parts[-1])))
                if len(q) > 7:
                    q.popleft()
        except Exception:
            continue
    for item in q:
        print(item.split()[-1], end=",")
    print()


if __name__ == "__main__":
    main()
