#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从评估集（eval_sft_with_tokens.jsonl，含 num_generated_tokens）中拆分出多轮轨迹评估集，保证与训练集无交集：
- 必须先运行 sft_dataset.py 生成 eval_sft_with_tokens.jsonl，再运行本脚本。
- 单位：轨迹 sample_id × assistant turn_index；每条样本包含 messages 前缀 + gold_assistant。
- 可按 num_generated_tokens 筛选轨迹（--max-tokens）。
"""

import argparse
import json
import random
from pathlib import Path


this_dir = Path(__file__).resolve().parent
DATA_DIR = this_dir / "data"

# 默认从 sft_dataset.py 产出的带 token 的评估集读取（含 num_generated_tokens，供 --max-tokens 筛选）
INPUT_PATH = DATA_DIR / "converted_gpt_oss_search_correct.eval_sft_with_tokens.jsonl"
OUTPUT_PATH = DATA_DIR / "converted_gpt_oss_search_correct.tools_sft.eval_turns.jsonl"

MAX_EVAL_TRAJ = 10          # 最多选多少条轨迹（None 表示不过滤）
DEFAULT_MAX_TOKENS = 30000  # 默认：过滤 num_generated_tokens 超过此值的轨迹（0 或负数表示不按 token 筛选）
MAX_PREFIX_CHARS = 30000   # 前缀 messages 序列化后的最大字符数，超过则跳过该 turn


def normalize_messages(raw_messages):
    """
    简单归一化 messages 结构，方便后续直接送入 chat_template：
    - user/system/普通 assistant: 保留 role + content
    - tool: role + content
    - assistant+tool_calls: 保留 role + tool_calls（不在本脚本中复用，仅保持结构一致）
    """
    normalized = []
    for m in raw_messages:
        role = m.get("role")
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            normalized.append({"role": "assistant", "tool_calls": tool_calls})
        else:
            normalized.append(
                {
                    "role": role,
                    "content": m.get("content") or "",
                }
            )
    return normalized


def main(max_tokens: int | None, max_traj: int = MAX_EVAL_TRAJ, input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Eval pool not found: {input_path}\n"
            "Run sft_dataset.py first to produce eval_sft_with_tokens.jsonl (train/eval split)."
        )

    # 1) 读取评估池样本，并按 num_generated_tokens 过滤过长轨迹
    raw_samples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            nt = sample.get("num_generated_tokens")
            if max_tokens is not None and isinstance(nt, int) and nt > max_tokens:
                continue
            raw_samples.append(sample)

    num_traj = len(raw_samples)
    if num_traj == 0:
        print("No samples loaded, abort.")
        return

    # 2) 轨迹级随机采样
    max_traj = max_traj or num_traj
    if max_traj >= num_traj:
        keep_indices = list(range(num_traj))
    else:
        random.seed(42)  # 如需完全随机可移除
        keep_indices = random.sample(range(num_traj), k=max_traj)
    keep_indices_set = set(keep_indices)

    # 3) 展开为 (sample_id, turn_index) 级别的多轮评估样本
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_turns = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for sample_idx, sample in enumerate(raw_samples):
            if sample_idx not in keep_indices_set:
                continue

            raw_messages = sample["messages"]
            messages = normalize_messages(raw_messages)

            for turn_idx, m in enumerate(messages):
                if m.get("role") != "assistant":
                    continue

                # gold 文本：
                raw_m = raw_messages[turn_idx] if turn_idx < len(raw_messages) else {}
                gold_raw = (raw_m.get("content") or "").strip()
                if gold_raw:
                    gold = gold_raw
                else:
                    # 无文本内容，用 tool_calls 还原成 <tool_call> 块
                    tool_calls = raw_m.get("tool_calls") or []
                    tc_chunks = []
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            try:
                                parsed_args = json.loads(args)
                            except Exception:
                                parsed_args = args
                        else:
                            parsed_args = args
                        payload = {"name": name, "arguments": parsed_args}
                        tc_chunks.append(
                            "<tool_call>\n"
                            + json.dumps(payload, ensure_ascii=False)
                            + "\n</tool_call>"
                        )
                    gold = "\n\n".join(tc_chunks) if tc_chunks else "[tool_calls]"

                prefix_messages = messages[:turn_idx]
                if MAX_PREFIX_CHARS is not None:
                    if len(json.dumps(prefix_messages, ensure_ascii=False)) > MAX_PREFIX_CHARS:
                        continue

                eval_obj = {
                    "sample_id": sample_idx,
                    "turn_index": turn_idx,
                    "messages": prefix_messages,
                    "gold_assistant": gold,
                }
                fout.write(json.dumps(eval_obj, ensure_ascii=False) + "\n")
                n_turns += 1

    print(f"Picked {len(keep_indices_set)} trajectories out of {num_traj}")
    print(f"Wrote {n_turns} assistant turns -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build eval turns from eval_sft_with_tokens.jsonl (run sft_dataset.py first). Input has num_generated_tokens for --max-tokens filtering."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Input JSONL: eval set with num_generated_tokens (default: eval_sft_with_tokens.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output eval turns JSONL (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Drop trajectories with num_generated_tokens > this (default: {DEFAULT_MAX_TOKENS}). Set to 0 or negative to disable.",
    )
    parser.add_argument(
        "--max-traj",
        type=int,
        default=MAX_EVAL_TRAJ,
        help=f"Max number of trajectories to sample for eval (default: {MAX_EVAL_TRAJ}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    main(max_tokens=max_tokens, max_traj=args.max_traj, input_path=args.input, output_path=args.output)