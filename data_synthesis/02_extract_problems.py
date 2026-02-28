#!/usr/bin/env python3
"""
步骤 2：从 OpenResearcher-Dataset 解析出 (qid, question, answer)，供 Deep Search 与 Judge 使用。

支持：
- 本地目录：--dataset_dir 下 seed_*/ 中的 *.parquet
- HuggingFace：--from_hf OpenResearcher/OpenResearcher-Dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def _discover_parquet_paths(dataset_dir: Path) -> list[Path]:
    """Find all parquet files under seed_* directories, sorted."""
    out: list[Path] = []
    for d in sorted(dataset_dir.iterdir()):
        if d.is_dir() and d.name.startswith("seed_"):
            out.extend(sorted(d.rglob("*.parquet")))
    return out


def extract_from_parquet(parquet_paths: list[Path], max_samples: int | None) -> list[dict]:
    """Read parquet row by row; each row has qid, question, answer (and optionally messages)."""
    records: list[dict] = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            for row in batch.to_pylist():
                r = dict(row)
                qid = r.get("qid")
                question = r.get("question")
                answer = r.get("answer")
                if qid is None or question is None or answer is None:
                    continue
                records.append({
                    "qid": int(qid) if not isinstance(qid, int) else qid,
                    "question": str(question),
                    "answer": str(answer),
                })
                if max_samples is not None and len(records) >= max_samples:
                    return records
    return records


def extract_from_hf(repo_id: str, split: str = "train", max_samples: int | None = None) -> list[dict]:
    """Load OpenResearcher-Dataset from HuggingFace and return list of {qid, question, answer}."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError as e:
        raise ImportError("HuggingFace datasets required: pip install datasets") from e

    ds = hf_load(repo_id, split=split, trust_remote_code=True)
    records = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        qid = row.get("qid")
        question = row.get("question")
        answer = row.get("answer")
        if qid is None or question is None or answer is None:
            continue
        records.append({
            "qid": int(qid) if not isinstance(qid, int) else qid,
            "question": str(question),
            "answer": str(answer),
        })
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Extract (qid, question, answer) from OpenResearcher-Dataset for data synthesis."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Local OpenResearcher-Dataset root (contains seed_42/, seed_43/, ... with .parquet).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Only include these seeds, e.g. --seeds 42 43. Default: all seed_* under dataset_dir.",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Include all seed_* under --dataset_dir (default when using --dataset_dir).",
    )
    parser.add_argument(
        "--from_hf",
        type=str,
        default=None,
        help="Load from HuggingFace instead of local dir, e.g. OpenResearcher/OpenResearcher-Dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace split when using --from_hf (default: train).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_synthesis/artifacts/problems.jsonl"),
        help="Output JSONL path. Each line: {\"qid\": int, \"question\": str, \"answer\": str}.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of problems to extract (for quick runs).",
    )
    args = parser.parse_args()

    if args.from_hf:
        records = extract_from_hf(
            args.from_hf,
            split=args.split,
            max_samples=args.max_samples,
        )
        print(f"Loaded {len(records)} problems from HuggingFace {args.from_hf} ({args.split}).")
    elif args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

        if args.seeds:
            parquet_paths = []
            for s in args.seeds:
                seed_dir = dataset_dir / f"seed_{s}"
                if seed_dir.is_dir():
                    parquet_paths.extend(sorted(seed_dir.rglob("*.parquet")))
        else:
            parquet_paths = _discover_parquet_paths(dataset_dir)

        if not parquet_paths:
            raise FileNotFoundError(
                f"No .parquet under {dataset_dir}. Check --dataset_dir and optional --seeds."
            )
        print(f"Found {len(parquet_paths)} parquet file(s).")
        records = extract_from_parquet(parquet_paths, args.max_samples)
        print(f"Extracted {len(records)} problems.")
    else:
        raise SystemExit(
            "Provide either --dataset_dir <path> or --from_hf <repo_id>. "
            "Example: --dataset_dir /path/to/OpenResearcher-Dataset --all_seeds"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
