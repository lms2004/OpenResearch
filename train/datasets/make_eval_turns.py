#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 materialized.jsonl 中单独拆分出多轮轨迹评估集：
- 单位：轨迹 sample_id × assistant turn_index
- 每条样本包含：到该轮为止的 messages 前缀 + gold_assistant
- 仅随机抽取最多 MAX_EVAL_TRAJ 条轨迹的全部轮次，方便在 eval_generate.py 中做多轮可视化评估。
"""

import json
import random
from pathlib import Path


this_dir = Path(__file__).resolve().parent
DATA_DIR = this_dir / "data"

INPUT_PATH = DATA_DIR / "converted_gpt_oss_search_correct.materialized.jsonl"
OUTPUT_PATH = DATA_DIR / "converted_gpt_oss_search_correct.tools_sft.eval_turns.jsonl"

MAX_EVAL_TRAJ = 10          # 最多选多少条轨迹（None 表示不过滤）
MAX_TOKENS_PER_TRAJ = 30000 # 可选：过滤特别长轨迹（基于 num_generated_tokens 字段）
MAX_PREFIX_CHARS = 30000    # 前缀 messages 序列化后的最大字符数，超过则跳过该 turn


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


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"INPUT not found: {INPUT_PATH}")

    # 1) 读取原始 materialized 样本，并按需要过滤过长轨迹
    raw_samples = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            nt = sample.get("num_generated_tokens")
            if isinstance(nt, int) and MAX_TOKENS_PER_TRAJ and nt > MAX_TOKENS_PER_TRAJ:
                continue
            raw_samples.append(sample)

    num_traj = len(raw_samples)
    if num_traj == 0:
        print("No samples loaded, abort.")
        return

    # 2) 轨迹级随机采样
    if MAX_EVAL_TRAJ is None or MAX_EVAL_TRAJ >= num_traj:
        keep_indices = list(range(num_traj))
    else:
        random.seed(42)  # 如需完全随机可移除
        keep_indices = random.sample(range(num_traj), k=MAX_EVAL_TRAJ)
    keep_indices_set = set(keep_indices)

    # 3) 展开为 (sample_id, turn_index) 级别的多轮评估样本
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_turns = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
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
    print(f"Wrote {n_turns} assistant turns -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()