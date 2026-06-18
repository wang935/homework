from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import TestImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate no-header test prediction CSV.")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "Homework" / "Dog_Heart" / "Dog_Heart" / "Test" / "Images")
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs" / "results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = [name for _, name in TestImageDataset(args.test_dir, transform=None)]
    expected_set = set(expected)

    rows = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"Row {row_index} must have exactly 2 columns, got {len(row)}: {row}")
            name, label = row
            if label not in {"0", "1", "2"}:
                raise ValueError(f"Row {row_index} has invalid label {label!r}; expected 0, 1, or 2")
            rows.append((name, label))

    names = [name for name, _ in rows]
    missing = sorted(expected_set - set(names))
    extra = sorted(set(names) - expected_set)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} rows, found {len(rows)} rows")
    if missing:
        raise ValueError(f"Missing {len(missing)} test filenames, first examples: {missing[:10]}")
    if extra:
        raise ValueError(f"Found {len(extra)} unknown filenames, first examples: {extra[:10]}")
    if duplicates:
        raise ValueError(f"Found duplicate filenames, first examples: {duplicates[:10]}")

    print(f"OK: {args.csv} has {len(rows)} rows, exact test filename set, and labels in {{0,1,2}}.")


if __name__ == "__main__":
    main()
