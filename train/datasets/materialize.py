#!/usr/bin/env python3
"""Add num_generated_tokens to parse.py-style JSONL using the tokenizer (optionally with multiprocessing).

Reads JSONL with messages/tools, computes token count per record via apply_chat_template + encode,
writes each record unchanged except for the added num_generated_tokens field. Intended for use with
parse.py output so that sft_dataset.py can filter trajectories by token count.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm

try:
    import orjson

    def _loads(line: str):
        return orjson.loads(line)

    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")
except Exception:
    def _loads(line: str):
        return json.loads(line)

    def _dumps(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)


_WORKER_TOKENIZER = None


def _init_worker(
    tokenizer_model: str,
    chat_template_text: Optional[str],
    trust_remote_code: bool,
):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    if chat_template_text:
        _WORKER_TOKENIZER.chat_template = chat_template_text


def replace_json_args(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool-call function arguments from json string to dict if needed for apply_chat_template."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls", []) or []:
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    pass
    return messages


def compute_num_tokens(data: Dict[str, Any], tokenizer) -> Optional[int]:
    """Compute token count for one record. Returns None on failure."""
    messages = replace_json_args(data.get("messages", []))
    if not messages:
        return None
    tools = data.get("tools")
    has_thinking = any(
        m.get("reasoning_content") for m in messages if isinstance(m, dict)
    )
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs={"enable_thinking": has_thinking},
        )
    except Exception:
        return None
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    return len(token_ids)


def process_record(data: Dict[str, Any], tokenizer) -> List[Dict[str, Any]]:
    """Add num_generated_tokens to record and return single-element list (1:1 with input)."""
    num_tokens = compute_num_tokens(data, tokenizer)
    if num_tokens is None:
        return []
    out = dict(data)
    out["num_generated_tokens"] = num_tokens
    return [out]


def _process_task_mp(task: Tuple[int, str]):
    line_no, line = task
    try:
        data = _loads(line)
    except Exception:
        return line_no, [], None
    try:
        out_items = process_record(data, _WORKER_TOKENIZER)
        out_lines = [_dumps(x) for x in out_items]
        return line_no, out_lines, None
    except Exception:
        return line_no, [], None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path (e.g. parse.py output).")
    parser.add_argument("--output", required=True, help="Output JSONL path (same schema + num_generated_tokens).")
    parser.add_argument("--tokenizer-model", required=True, help="HF tokenizer model/path.")
    # Default to train/datasets/templates/all_assistant.jinja
    this_dir = Path(__file__).resolve().parent
    default_template = this_dir / "templates" / "all_assistant.jinja"
    parser.add_argument(
        "--chat-template-file",
        default=str(default_template),
        help="Jinja chat template file (defaults to templates/all_assistant.jinja).",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--chunksize", type=int, default=8, help="Pool imap chunksize.")
    args = parser.parse_args()

    chat_template_text = None
    if args.chat_template_file:
        chat_template_text = Path(args.chat_template_file).read_text(encoding="utf-8")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    if args.workers <= 1:
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer_model, trust_remote_code=args.trust_remote_code
        )
        if chat_template_text:
            tok.chat_template = chat_template_text
        with in_path.open("r", encoding="utf-8") as fin, out_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in tqdm(fin, desc="materialize"):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    data = _loads(line)
                    out_items = process_record(data, tok)
                    for item in out_items:
                        fout.write(_dumps(item) + "\n")
                        kept += 1
                except Exception:
                    pass
    else:
        with out_path.open("w", encoding="utf-8") as fout, mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                args.tokenizer_model,
                chat_template_text,
                args.trust_remote_code,
            ),
        ) as pool:

            def _tasks():
                with in_path.open("r", encoding="utf-8") as fin:
                    for line_no, line in enumerate(fin, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        yield (line_no, line)

            iterator = pool.imap_unordered(
                _process_task_mp, _tasks(), chunksize=max(1, args.chunksize)
            )
            results = []
            for t in tqdm(iterator, desc="materialize(mp)"):
                results.append(t)
            results.sort(key=lambda x: x[0])
            for _, out_lines, _ in results:
                total += 1
                for s in out_lines:
                    fout.write(s + "\n")
                    kept += 1

    print(f"materialize done: kept={kept} dropped={total - kept} total={total} output={out_path}")


if __name__ == "__main__":
    main()
