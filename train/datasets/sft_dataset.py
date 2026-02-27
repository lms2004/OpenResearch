import argparse
import json
import random
from pathlib import Path
import sys

# Ensure repo root (one level above project root) is on sys.path
this_dir = Path(__file__).resolve().parent
repo_root = this_dir.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_utils import get_browser_tools_json_schema

# 使用统一的工具 JSON schema 生成函数
TOOLS_JSON_SCHEMA_STR = get_browser_tools_json_schema()

# 与 materialize 流水线衔接：输入为 materialized.jsonl（含 num_generated_tokens），输出为 tools_sft.jsonl（训练集）+ 两份评估集（与训练集无交集）
input_path = this_dir / "data/converted_gpt_oss_search_correct.materialized.jsonl"
output_path = this_dir / "data/converted_gpt_oss_search_correct.tools_sft.jsonl"
# 评估集（与训练集列一致：messages + tools），供 train.py --eval_jsonl 使用
eval_sft_path = this_dir / "data/converted_gpt_oss_search_correct.eval_sft.jsonl"
# 评估集（含 num_generated_tokens），供 make_eval_turns.py 按 token 筛选使用
eval_sft_with_tokens_path = this_dir / "data/converted_gpt_oss_search_correct.eval_sft_with_tokens.jsonl"

# 默认按 token 数筛选：超过此值的轨迹不写入（仅当样本有 num_generated_tokens 时生效）
DEFAULT_MAX_TOKENS = 30000
# 默认从通过筛选的样本中划分 5% 作为评估池（与训练集无交集），供 make_eval_turns.py 使用
DEFAULT_EVAL_RATIO = 0.05
DEFAULT_EVAL_SEED = 42


def _normalize_args_id(obj):
    """Ensure arguments.id is always str so PyArrow/datasets see consistent type."""
    if isinstance(obj, dict):
        if "id" in obj and obj["id"] is not None:
            obj = {k: _normalize_args_id(v) if k != "id" else str(v) for k, v in obj.items()}
        else:
            obj = {k: _normalize_args_id(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj = [_normalize_args_id(x) for x in obj]
    return obj


def normalize_messages(raw_messages):
    """
    把 pretty.json / 原始 messages 转成类似：
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "name": "...", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    id_to_tool_name = {}
    normalized = []

    for m in raw_messages:
        role = m.get("role")
        tool_calls = m.get("tool_calls")

        # 带 tool_calls 的 assistant：只保留 role + tool_calls，arguments 解析成 dict
        if role == "assistant" and tool_calls:
            new_tool_calls = []
            for tc in tool_calls:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                # 原数据里 arguments 多半是 JSON str，这里解析成 dict
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except Exception:
                        parsed_args = args
                else:
                    parsed_args = args
                # 统一 arguments 内 id 为 str，避免 PyArrow 报 "changed from string to number"
                if isinstance(parsed_args, dict):
                    parsed_args = _normalize_args_id(parsed_args)
                # 统一写成 JSON 字符串，避免 PyArrow 报 "changed from object to string"
                if isinstance(parsed_args, dict):
                    arguments_out = json.dumps(parsed_args, ensure_ascii=False)
                else:
                    arguments_out = parsed_args if isinstance(parsed_args, str) else "{}"

                new_tc = {
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn.get("name"),
                        "arguments": arguments_out,
                    },
                }
                new_tool_calls.append(new_tc)

                tc_id = tc.get("id")
                if tc_id:
                    id_to_tool_name[tc_id] = fn.get("name")

            normalized.append(
                {
                    "role": "assistant",
                    "tool_calls": new_tool_calls,
                }
            )

        # tool 消息：根据 tool_call_id 找回 name 字段
        elif role == "tool":
            tool_call_id = m.get("tool_call_id")
            name = id_to_tool_name.get(tool_call_id)
            new_msg = {
                "role": "tool",
                "content": m.get("content", ""),
            }
            if name:
                new_msg["name"] = name
            normalized.append(new_msg)

        # 其他（system / user / 普通 assistant）：只保留 role + content
        else:
            content = m.get("content") or ""
            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return normalized


def convert(max_tokens: int | None, eval_ratio: float = 0.0, eval_seed: int = DEFAULT_EVAL_SEED):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读取并过滤
    samples = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            num_tokens = sample.get("num_generated_tokens")
            if max_tokens is not None and isinstance(num_tokens, int) and num_tokens > max_tokens:
                continue
            samples.append(sample)

    n = len(samples)
    if n == 0:
        print("No samples after filtering, abort.")
        return

    # 2) 划分 train / eval（固定 seed，保证可复现且评估集不与训练集重叠）
    rng = random.Random(eval_seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_eval = max(0, int(round(n * eval_ratio))) if eval_ratio > 0 else 0
    eval_indices = set(indices[:n_eval])
    train_indices = set(indices[n_eval:])

    # 3) 写训练集 tools_sft.jsonl（仅 train 部分）
    n_train = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for i in train_indices:
            sample = samples[i]
            raw_messages = sample["messages"]
            messages = normalize_messages(raw_messages)
            out_obj = {
                "messages": messages,
                "tools": TOOLS_JSON_SCHEMA_STR,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_train += 1

    # 4) 写评估集：两份文件，列与训练集一致的一份给 train.py，带 num_generated_tokens 的一份给 make_eval_turns.py
    if eval_indices:
        eval_sft_path.parent.mkdir(parents=True, exist_ok=True)
        # 4a) eval_sft.jsonl：仅 messages + tools，与训练集列一致，供 train.py --eval_jsonl
        with eval_sft_path.open("w", encoding="utf-8") as fout:
            for i in sorted(eval_indices):
                sample = samples[i]
                messages = normalize_messages(sample["messages"])
                out_obj = {
                    "messages": messages,
                    "tools": TOOLS_JSON_SCHEMA_STR,
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        # 4b) eval_sft_with_tokens.jsonl：含 num_generated_tokens，供 make_eval_turns.py 按 --max-tokens 筛选
        with eval_sft_with_tokens_path.open("w", encoding="utf-8") as fout:
            for i in sorted(eval_indices):
                sample = samples[i]
                messages = normalize_messages(sample["messages"])
                out_obj = {
                    "messages": messages,
                    "tools": TOOLS_JSON_SCHEMA_STR,
                }
                if "num_generated_tokens" in sample:
                    out_obj["num_generated_tokens"] = sample["num_generated_tokens"]
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        print(f"Converted {n_train} train samples -> {output_path}")
        print(f"Eval {len(eval_indices)} samples -> {eval_sft_path} (train.py --eval_jsonl, same columns as train)")
        print(f"Eval (with tokens) {len(eval_indices)} samples -> {eval_sft_with_tokens_path} (make_eval_turns.py --input)")
    else:
        print(f"Converted {n_train} train samples -> {output_path}")
        if eval_ratio > 0:
            print("Eval ratio yielded 0 eval samples; run make_eval_turns with --input to a dedicated eval JSONL if needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Build tools_sft.jsonl (train), eval_sft.jsonl (eval for train.py, same columns as train), and eval_sft_with_tokens.jsonl (eval for make_eval_turns.py)."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Drop trajectories with num_generated_tokens > this (default: {DEFAULT_MAX_TOKENS}). Set to 0 or negative to disable.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=DEFAULT_EVAL_RATIO,
        help=f"Fraction of (filtered) samples for eval set (default: {DEFAULT_EVAL_RATIO}). Written to eval_sft.jsonl and eval_sft_with_tokens.jsonl.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=f"Random seed for train/eval split (default: {DEFAULT_EVAL_SEED}).",
    )
    args = parser.parse_args()
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    convert(max_tokens=max_tokens, eval_ratio=args.eval_ratio, eval_seed=args.eval_seed)


if __name__ == "__main__":
    main()