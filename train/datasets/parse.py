#!/usr/bin/env python3
"""Convert OpenResearcher parquet rows to example-style JSONL."""

from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm


def load_template_tools(template_jsonl: Path) -> list[dict[str, Any]]:
    """Load `tools` field from the first line of a template JSONL file."""
    with template_jsonl.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return []
    obj = json.loads(first_line)
    tools = obj.get("tools", [])
    return tools if isinstance(tools, list) else []


def _extract_text(content: Any) -> str:
    """Extract plain text from source message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "\n".join(parts)
    return ""


def _new_message(
    role: str,
    content: str,
    reasoning_content: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build target-format message object."""
    return {
        "content": content,
        "reasoning_content": reasoning_content,
        "role": role,
        "tool_call_id": tool_call_id,
        "tool_calls": tool_calls,
    }


def convert_messages_to_target_format(
    src_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert Cursor-style messages to target OpenAI-style messages.

    Source format has `channel/recipient/content_type`; target uses
    `content/reasoning_content/tool_calls/tool_call_id`.
    """
    out_messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None
    pending_tool_call_ids: deque[str] = deque()
    call_idx = 0

    # Merge system + developer prompt into one `system` message.
    system_parts: list[str] = []
    for msg in src_messages:
        if msg.get("role") in {"system", "developer"}:
            text = _extract_text(msg.get("content"))
            if text.strip():
                system_parts.append(text)
    if system_parts:
        out_messages.append(
            _new_message(
                role="system",
                content="\n\n".join(system_parts),
                reasoning_content=None,
                tool_call_id=None,
                tool_calls=None,
            )
        )

    for msg in src_messages:
        role = msg.get("role")
        text = _extract_text(msg.get("content"))

        if role in {"system", "developer"}:
            continue

        if role == "user":
            out_messages.append(
                _new_message(
                    role="user",
                    content=text,
                    reasoning_content=None,
                    tool_call_id=None,
                    tool_calls=None,
                )
            )
            continue

        if role == "assistant":
            recipient = msg.get("recipient")
            channel = msg.get("channel")

            # Tool invocation: map to assistant message with tool_calls.
            if isinstance(recipient, str) and recipient:
                call_idx += 1
                call_id = f"call_{call_idx:08d}"
                pending_tool_call_ids.append(call_id)
                arguments = text if text else "{}"
                reasoning = pending_reasoning if pending_reasoning else None
                pending_reasoning = None

                out_messages.append(
                    _new_message(
                        role="assistant",
                        content="",
                        reasoning_content=reasoning,
                        tool_call_id=None,
                        tool_calls=[
                            {
                                "function": {
                                    "arguments": arguments,
                                    "name": recipient,
                                },
                                "id": call_id,
                                "type": "function",
                            }
                        ],
                    )
                )
                continue

            # Analysis messages are usually reasoning for the next tool call/final answer.
            if channel == "analysis":
                pending_reasoning = text if text else pending_reasoning
                continue

            # Final assistant response (no tool call).
            out_messages.append(
                _new_message(
                    role="assistant",
                    content=text,
                    reasoning_content=pending_reasoning,
                    tool_call_id=None,
                    tool_calls=None,
                )
            )
            pending_reasoning = None
            continue

        if role == "tool":
            tool_call_id = pending_tool_call_ids.popleft() if pending_tool_call_ids else None
            out_messages.append(
                _new_message(
                    role="tool",
                    content=text,
                    reasoning_content=None,
                    tool_call_id=tool_call_id,
                    tool_calls=None,
                )
            )
            continue

    return out_messages


def _count_parquet_rows(path: Path) -> int:
    """Return number of rows in a parquet file (from metadata)."""
    return pq.ParquetFile(path).metadata.num_rows


def _seed_from_parquet_path(path: Path) -> str:
    """Extract seed identifier from parquet path, e.g. .../seed_42/train-xxx.parquet -> seed_42."""
    parent = path.parent.name
    if parent.startswith("seed_"):
        return parent
    return parent or "seed_0"


def _make_trajectory_id(seed_str: str, record: dict[str, Any]) -> str:
    """Unique trajectory ID: seed + qid + chunk_idx + num_chunks."""
    qid = record.get("qid", 0)
    chunk_idx = record.get("chunk_idx", 0)
    num_chunks = record.get("num_chunks", 1)
    return f"{seed_str}_{qid}_{chunk_idx}_{num_chunks}"


def _convert_one_parquet(
    path: Path,
    template_tools: list[dict[str, Any]],
    default_correct: bool,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """Convert a single parquet file. Returns (records, n_skipped). Only keeps rows with status == 'success'. Adds trajectory_id from seed + qid + chunk."""
    records: list[dict[str, Any]] = []
    n_skipped = 0
    seed_str = _seed_from_parquet_path(path)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            record = dict(row)
            if record.get("status") != "success":
                n_skipped += 1
                continue
            record["trajectory_id"] = _make_trajectory_id(seed_str, record)
            record["messages"] = convert_messages_to_target_format(
                record.get("messages", [])
            )
            record.setdefault("correct", default_correct)
            record.setdefault("tools", template_tools)
            records.append(record)
    return (records, n_skipped)


def convert_parquet_to_jsonl(
    parquet_paths: list[Path],
    output_jsonl_path: Path,
    output_pretty_json_path: Path | None,
    template_jsonl_path: Path | None,
    default_correct: bool,
    batch_size: int = 128,
    pretty_limit: int = 50,
    workers: int = 1,
) -> int:
    """Convert parquet record(s) to one JSONL and optional pretty JSON array.
    Only rows with status == 'success' are written.
    """
    template_tools: list[dict[str, Any]] = []
    if template_jsonl_path is not None:
        template_tools = load_template_tools(template_jsonl_path)

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    pretty_rows = 0

    if workers <= 1:
        # Sequential: one progress bar over rows
        total_from_meta = sum(_count_parquet_rows(p) for p in parquet_paths)
        with output_jsonl_path.open("w", encoding="utf-8") as out_jsonl:
            out_pretty = None
            if output_pretty_json_path is not None:
                output_pretty_json_path.parent.mkdir(parents=True, exist_ok=True)
                out_pretty = output_pretty_json_path.open("w", encoding="utf-8")
                out_pretty.write("[\n")

            n_success = 0
            n_skipped = 0
            pbar = tqdm(total=total_from_meta, unit="row", desc="Converting")
            for parquet_path in parquet_paths:
                parquet = pq.ParquetFile(parquet_path)
                seed_str = _seed_from_parquet_path(parquet_path)
                for batch in parquet.iter_batches(batch_size=batch_size):
                    for row in batch.to_pylist():
                        record = dict(row)
                        if record.get("status") != "success":
                            n_skipped += 1
                            pbar.update(1)
                            pbar.set_postfix(success=n_success, skipped=n_skipped, refresh=True)
                            continue
                        record["trajectory_id"] = _make_trajectory_id(seed_str, record)
                        record["messages"] = convert_messages_to_target_format(
                            record.get("messages", [])
                        )
                        record.setdefault("correct", default_correct)
                        record.setdefault("tools", template_tools)
                        out_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

                        if out_pretty is not None and pretty_rows < pretty_limit:
                            if pretty_rows > 0:
                                out_pretty.write(",\n")
                            out_pretty.write(
                                json.dumps(record, ensure_ascii=False, indent=2)
                            )
                            pretty_rows += 1

                        total_rows += 1
                        n_success += 1
                        pbar.update(1)
                        pbar.set_postfix(success=n_success, skipped=n_skipped, refresh=True)
                    pbar.refresh()
            pbar.close()

            if out_pretty is not None:
                out_pretty.write("\n]\n")
                out_pretty.close()
    else:
        # Parallel: convert files in parallel, then write in order
        with output_jsonl_path.open("w", encoding="utf-8") as out_jsonl:
            out_pretty = None
            if output_pretty_json_path is not None:
                output_pretty_json_path.parent.mkdir(parents=True, exist_ok=True)
                out_pretty = output_pretty_json_path.open("w", encoding="utf-8")
                out_pretty.write("[\n")

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _convert_one_parquet,
                        p,
                        template_tools,
                        default_correct,
                        batch_size,
                    ): i
                    for i, p in enumerate(parquet_paths)
                }
                # Collect results in order of parquet_paths
                results_by_idx: dict[int, tuple[list[dict[str, Any]], int]] = {}
                with tqdm(
                    total=len(parquet_paths),
                    unit="file",
                    desc="Converting",
                ) as pbar:
                    for future in as_completed(futures):
                        idx = futures[future]
                        results_by_idx[idx] = future.result()
                        pbar.update(1)

            total_to_write = sum(len(results_by_idx[i][0]) for i in range(len(parquet_paths)))
            total_skipped = sum(results_by_idx[i][1] for i in range(len(parquet_paths)))
            with tqdm(total=total_to_write, unit="row", desc="Writing") as pbar:
                for i in range(len(parquet_paths)):
                    records_i = results_by_idx[i][0]
                    for record in records_i:
                        out_jsonl.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        if out_pretty is not None and pretty_rows < pretty_limit:
                            if pretty_rows > 0:
                                out_pretty.write(",\n")
                            out_pretty.write(
                                json.dumps(record, ensure_ascii=False, indent=2)
                            )
                            pretty_rows += 1
                        total_rows += 1
                        pbar.update(1)
                        pbar.set_postfix(success=total_rows, skipped=total_skipped, refresh=True)

            if out_pretty is not None:
                out_pretty.write("\n]\n")
                out_pretty.close()

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert parquet to example-style JSONL."
    )

    # Use paths relative to this file's directory (train/datasets)
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    dataset_dir = repo_root / "OpenResearcher-Dataset"
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=dataset_dir,
        help="Base directory containing seed_42, seed_43, ... (used with --seeds). Default: OpenResearcher-Dataset/",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Seed numbers to include, e.g. --seeds 42 43 44 45. Paths become <dataset-dir>/seed_N. Use --all-seeds to include every seed_* under dataset-dir.",
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Include all seed_* directories under --dataset-dir (discovered automatically). Overrides --seeds.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit parquet files or directories. Ignored when --seeds is set. Default when no --seeds: seed_42 + seed_43.",
    )
    parser.add_argument(
        "--template-jsonl",
        type=Path,
        default=this_dir / "templates/converted_gpt_oss_search_correct.example.jsonl",
        help="Template JSONL (first line used to copy `tools` structure).",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=this_dir / "data/converted_gpt_oss_search_correct.parsed.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--output-pretty-json",
        type=Path,
        default=this_dir / "data/converted_gpt_oss_search_correct.parsed.pretty.json",
        help="Pretty-printed JSON array output path.",
    )
    parser.add_argument(
        "--default-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default value for `correct` when source records do not have it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Parquet reading batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers to convert parquet files (default: 1 = sequential).",
    )
    return parser.parse_args()


def _discover_seed_dirs(dataset_dir: Path) -> list[Path]:
    """Find all seed_* directories under dataset_dir, sorted by seed number."""
    out: list[tuple[int, Path]] = []
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.startswith("seed_"):
            try:
                n = int(d.name[5:])
                out.append((n, d))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    # --all-seeds > --seeds > --parquet > 默认 seed_42 + seed_43
    if args.all_seeds:
        parquet_inputs = _discover_seed_dirs(dataset_dir)
        if not parquet_inputs:
            raise FileNotFoundError(
                f"No seed_* directories under {dataset_dir}. Check --dataset-dir."
            )
        print(f"Using all seeds under {dataset_dir}: {[p.name for p in parquet_inputs]}")
    elif args.seeds is not None and len(args.seeds) > 0:
        parquet_inputs = [dataset_dir / f"seed_{s}" for s in sorted(args.seeds)]
    elif args.parquet is not None and len(args.parquet) > 0:
        parquet_inputs = list(args.parquet)
    else:
        parquet_inputs = [
            dataset_dir / "seed_42",
            dataset_dir / "seed_43",
        ]

    parquet_paths: list[Path] = []
    for p in parquet_inputs:
        p = Path(p).resolve()
        if p.is_dir():
            found = sorted(p.rglob("*.parquet"))
            if not found:
                print(f"Warning: no .parquet under {p}, skipping.")
            else:
                parquet_paths.extend(found)
                print(f"Found {len(found)} parquet file(s) under {p}")
        elif p.is_file():
            parquet_paths.append(p)
        else:
            raise FileNotFoundError(f"Not a file or directory: {p}")

    if not parquet_paths:
        raise FileNotFoundError(
            "No .parquet files from any of the given paths. Check --dataset-dir/--seeds or --parquet."
        )
    parquet_paths.sort()

    rows = convert_parquet_to_jsonl(
        parquet_paths=parquet_paths,
        output_jsonl_path=args.output_jsonl,
        output_pretty_json_path=args.output_pretty_json,
        template_jsonl_path=args.template_jsonl,
        default_correct=args.default_correct,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    if args.output_pretty_json is not None:
        print(
            f"Done. Wrote {rows} rows to: {args.output_jsonl} "
            f"and {args.output_pretty_json}"
        )
    else:
        print(f"Done. Wrote {rows} rows to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
