#!/usr/bin/env python3
"""Filter parsed/materialized JSONL to keep only trajectories that were judged correct in evaluated.jsonl."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from tqdm import tqdm


def _line_count(path: Path) -> int:
    """Return number of lines in file using wc -l (faster than Python iteration)."""
    try:
        result = subprocess.run(
            ["wc", "-l", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return 0
        parts = result.stdout.strip().split()
        return int(parts[0]) if parts else 0
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter records by trajectory_id: keep only those with correct=True in evaluated.jsonl."
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("train/datasets/data/eval_work"),
        help="Directory containing evaluated.jsonl (default: train/datasets/data/eval_work)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL (parsed or materialized) with trajectory_id per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL: only records whose trajectory_id is in the correct set",
    )
    args = parser.parse_args()

    evaluated_path = args.eval_dir / "evaluated.jsonl"
    if not evaluated_path.exists():
        raise SystemExit(f"evaluated.jsonl not found: {evaluated_path}")

    correct_trajectory_ids: set[str] = set()
    with evaluated_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading evaluated.jsonl", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("correct") is True and item.get("trajectory_id"):
                correct_trajectory_ids.add(item["trajectory_id"])

    print(f"Correct trajectory IDs: {len(correct_trajectory_ids)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_lines = _line_count(args.input)
    kept = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        pbar = tqdm(
            fin,
            total=total_lines,
            desc="Filtering input",
            unit=" lines",
        )
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("trajectory_id")
            if tid in correct_trajectory_ids:
                fout.write(line + "\n")
                kept += 1
        pbar.close()

    print(f"Kept {kept} records -> {args.output}")


if __name__ == "__main__":
    main()
