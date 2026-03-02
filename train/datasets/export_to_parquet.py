#!/usr/bin/env python3
"""Export filtered trajectories back to parquet: one directory per seed (seed_42/, seed_43/, ...)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _seed_from_parquet_path(path: Path) -> str:
    """Extract seed from parquet path, e.g. .../seed_42/train-xxx.parquet -> seed_42."""
    parent = path.parent.name
    if parent.startswith("seed_"):
        return parent
    return parent or "seed_0"


def _make_trajectory_id(seed_str: str, record: dict[str, Any]) -> str:
    """Same formula as parse.py: seed + qid + chunk_idx + num_chunks."""
    qid = record.get("qid", 0)
    chunk_idx = record.get("chunk_idx", 0)
    num_chunks = record.get("num_chunks", 1)
    return f"{seed_str}_{qid}_{chunk_idx}_{num_chunks}"


def load_trajectory_id_allowlist(path: Path) -> set[str]:
    """Load set of trajectory_id from JSONL (each line has key 'trajectory_id')."""
    allowlist: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("trajectory_id")
            if tid:
                allowlist.add(tid)
    return allowlist


def discover_parquet_paths(dataset_dir: Path, seeds: list[int] | None) -> list[Path]:
    """List parquet files under dataset_dir/seed_* (or only given seeds)."""
    if seeds is not None and len(seeds) > 0:
        out: list[Path] = []
        for s in sorted(seeds):
            d = dataset_dir / f"seed_{s}"
            if d.is_dir():
                out.extend(sorted(d.rglob("*.parquet")))
        return out
    out = []
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name.startswith("seed_"):
            out.extend(sorted(d.rglob("*.parquet")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trajectories (by trajectory_id allowlist) to per-seed parquet dirs."
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        required=True,
        help="JSONL with trajectory_id per line (e.g. output of filter_correct_with_eval.py)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root dir containing seed_42/, seed_43/, ... with source parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root: will create output_dir/seed_42/, output_dir/seed_43/, ...",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="If set, only these seeds (e.g. 42 43). Otherwise all seed_* under dataset-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size when reading parquet (default 1024).",
    )
    args = parser.parse_args()

    allowlist = load_trajectory_id_allowlist(args.allowlist)
    print(f"Allowlist size: {len(allowlist)}")

    parquet_paths = discover_parquet_paths(args.dataset_dir.resolve(), args.seeds)
    if not parquet_paths:
        raise SystemExit("No parquet files found under dataset-dir.")

    # Collect kept rows by seed (seed_str -> list of row dicts)
    by_seed: dict[str, list[dict[str, Any]]] = {}
    labels: list[dict[str, Any]] = []

    for path in parquet_paths:
        seed_str = _seed_from_parquet_path(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=args.batch_size):
            for row in batch.to_pylist():
                record = dict(row)
                tid = _make_trajectory_id(seed_str, record)
                if tid not in allowlist:
                    continue
                if seed_str not in by_seed:
                    by_seed[seed_str] = []
                by_seed[seed_str].append(record)
                labels.append({
                    "trajectory_id": tid,
                    "seed": seed_str,
                    "qid": record.get("qid"),
                    "chunk_idx": record.get("chunk_idx"),
                    "num_chunks": record.get("num_chunks"),
                    "status": record.get("status"),
                    "label": record.get("correct"),
                    "included": True,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed_str, rows in by_seed.items():
        out_seed_dir = args.output_dir / seed_str
        out_seed_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        out_path = out_seed_dir / "train.parquet"
        pq.write_table(table, out_path)
        print(f"Wrote {len(rows)} rows -> {out_path}")

    labels_path = args.output_dir / "trajectory_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    print(f"Wrote {len(labels)} labels -> {labels_path}")


if __name__ == "__main__":
    main()
