#!/usr/bin/env python3
"""Export filtered trajectories back to parquet: one directory per seed (seed_42/, seed_43/, ...)."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


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
        for line in tqdm(f, desc="Loading allowlist", unit=" lines"):
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


def _process_parquet_chunk(
    paths: list[Path],
    allowlist: set[str],
    batch_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Process a chunk of parquet paths; return (by_seed, labels). Used in worker process."""
    by_seed: dict[str, list[dict[str, Any]]] = {}
    labels: list[dict[str, Any]] = []
    for path in paths:
        seed_str = _seed_from_parquet_path(path)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
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
    return by_seed, labels


def _merge_worker_results(
    results: list[tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Merge (by_seed, labels) from multiple workers."""
    merged_seed: dict[str, list[dict[str, Any]]] = {}
    merged_labels: list[dict[str, Any]] = []
    for by_seed, labels in results:
        for seed_str, rows in by_seed.items():
            if seed_str not in merged_seed:
                merged_seed[seed_str] = []
            merged_seed[seed_str].extend(rows)
        merged_labels.extend(labels)
    return merged_seed, merged_labels


def _write_one_seed(
    item: tuple[str, list[dict[str, Any]]],
    output_dir: Path,
) -> tuple[str, int]:
    """Write one seed's parquet; return (seed_str, row_count). Used in thread pool."""
    seed_str, rows = item
    out_seed_dir = output_dir / seed_str
    out_seed_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    out_path = out_seed_dir / "train.parquet"
    pq.write_table(table, out_path)
    return seed_str, len(rows)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for scanning parquet (default: 16).",
    )
    args = parser.parse_args()

    allowlist = load_trajectory_id_allowlist(args.allowlist)
    print(f"Allowlist size: {len(allowlist)}")

    parquet_paths = discover_parquet_paths(args.dataset_dir.resolve(), args.seeds)
    if not parquet_paths:
        raise SystemExit("No parquet files found under dataset-dir.")

    n_workers = args.workers
    if n_workers is None:
        n_workers = 16
    n_workers = max(1, n_workers)
    print(f"Using {n_workers} worker(s) for scanning.")

    # Distribute paths across workers
    chunk_size = max(1, (len(parquet_paths) + n_workers - 1) // n_workers)
    chunks: list[list[Path]] = []
    for i in range(0, len(parquet_paths), chunk_size):
        chunks.append(parquet_paths[i : i + chunk_size])

    by_seed: dict[str, list[dict[str, Any]]] = {}
    labels: list[dict[str, Any]] = []

    if len(chunks) == 1:
        by_seed, labels = _process_parquet_chunk(
            chunks[0], allowlist, args.batch_size
        )
    else:
        worker_results: list[tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]] = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _process_parquet_chunk,
                    chunk,
                    allowlist,
                    args.batch_size,
                ): i
                for i, chunk in enumerate(chunks)
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Scanning parquet",
                unit=" chunks",
            ):
                worker_results.append(fut.result())
        by_seed, labels = _merge_worker_results(worker_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parallel write per-seed parquet files
    write_workers = min(n_workers, len(by_seed))
    if write_workers <= 1:
        for seed_str, rows in tqdm(
            by_seed.items(), desc="Writing parquet", unit=" seeds"
        ):
            out_seed_dir = args.output_dir / seed_str
            out_seed_dir.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(rows)
            out_path = out_seed_dir / "train.parquet"
            pq.write_table(table, out_path)
            print(f"Wrote {len(rows)} rows -> {out_path}")
    else:
        items = list(by_seed.items())
        with ThreadPoolExecutor(max_workers=write_workers) as executor:
            write_futs = {
                executor.submit(_write_one_seed, item, args.output_dir): item[0]
                for item in items
            }
            for fut in tqdm(
                as_completed(write_futs),
                total=len(write_futs),
                desc="Writing parquet",
                unit=" seeds",
            ):
                seed_str, count = fut.result()
                print(f"Wrote {count} rows -> {args.output_dir / seed_str / 'train.parquet'}")

    labels_path = args.output_dir / "trajectory_labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    print(f"Wrote {len(labels)} labels -> {labels_path}")


if __name__ == "__main__":
    main()
