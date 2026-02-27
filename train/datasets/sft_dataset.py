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

# 与 parse.py 流水线衔接：
# - 输入为 parse 输出的 materialized.jsonl
# - 输出 1（训练用）：tools_sft.jsonl（整段多轮对话）
# - 输出 2（评估用）：tools_sft.eval_turns.jsonl（按 assistant 轮次展开，便于多轮 gold 提取）
input_path = this_dir / "data/converted_gpt_oss_search_correct.materialized.jsonl"
output_path = this_dir / "data/converted_gpt_oss_search_correct.tools_sft.jsonl"
eval_turns_output_path = this_dir / "data/converted_gpt_oss_search_correct.tools_sft.eval_turns.jsonl"


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


MAX_EVAL_TRAJ = 10  # 多轮评估集最多包含多少条完整轨迹（None 表示不过滤）


def convert():
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_turns_output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_turns = 0
    # 先把 materialized.jsonl 全部读入内存，方便做轨迹级采样
    with input_path.open("r", encoding="utf-8") as fin:
        raw_samples = [json.loads(line) for line in fin if line.strip()]

    # 训练集仍使用全部样本；多轮评估集只从中随机抽取至多 MAX_EVAL_TRAJ 条轨迹
    num_traj = len(raw_samples)
    if MAX_EVAL_TRAJ is None or MAX_EVAL_TRAJ >= num_traj:
        eval_traj_indices = set(range(num_traj))
    else:
        # 固定一个种子，保证每次生成评估集可复现；如需完全随机，可去掉 seed 设置
        random.seed(42)
        eval_traj_indices = set(random.sample(range(num_traj), k=MAX_EVAL_TRAJ))

    with output_path.open("w", encoding="utf-8") as fout, \
            eval_turns_output_path.open("w", encoding="utf-8") as fout_turns:
        for sample_idx, sample in enumerate(raw_samples):
            # 如果上游 materialize 已经计算了 token 数量，这里再次做一道保险过滤
            num_tokens = sample.get("num_generated_tokens")
            if isinstance(num_tokens, int) and num_tokens > 30000:
                continue

            raw_messages = sample["messages"]
            messages = normalize_messages(raw_messages)

            out_obj = {
                "messages": messages,
                # 工具按照 transformers / TRL 推荐存成 JSON str
                "tools": TOOLS_JSON_SCHEMA_STR,
            }

            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n += 1

            # 只有被选中的轨迹才写入多轮评估集
            if sample_idx not in eval_traj_indices:
                continue

            # 额外构造「按 assistant 轮次展开」的评估集：
            # 对每条对话中的每个 assistant 轮（无论是文字还是 tool-calls-only），写一条样本：
            #   - messages: 截止该轮之前的上下文（不包含 gold 本轮）
            #   - gold_assistant: 该轮的 gold 文本；若无文本内容但有 tool_calls，则标记为 "[tool_calls]"
            #   - sample_id / turn_index: 便于在 eval 中对齐原始轨迹与轮次
            for turn_idx, m in enumerate(messages):
                if m.get("role") != "assistant":
                    continue
                gold_raw = (m.get("content") or "").strip()
                gold = gold_raw if gold_raw else "[tool_calls]"
                prefix_messages = messages[:turn_idx]
                eval_obj = {
                    "sample_id": sample_idx,
                    "turn_index": turn_idx,
                    "messages": prefix_messages,
                    "gold_assistant": gold,
                    "tools": TOOLS_JSON_SCHEMA_STR,
                }
                fout_turns.write(json.dumps(eval_obj, ensure_ascii=False) + "\n")
                n_turns += 1

    print(f"Converted {n} samples -> {output_path}")
    print(f"Converted {n_turns} assistant turns -> {eval_turns_output_path}")


if __name__ == "__main__":
    convert()