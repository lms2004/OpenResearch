#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _last_user_content(messages):
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return (m.get("content") or "")[:2000]
    return ""


def _assistant_turn_content(m):
    """单条 assistant 消息的展示内容（content 或 [tool_calls]）。"""
    if m.get("role") != "assistant":
        return ""
    c = m.get("content") or ""
    if c:
        return c[:2000]
    if m.get("tool_calls"):
        return "[tool_calls]"
    return ""


def _last_assistant_content(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = m.get("content") or ""
            if c:
                return c[:2000]
            if m.get("tool_calls"):
                return "[tool_calls]"
    return ""


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "从训练好的 SFT checkpoint 生成模型在 eval JSONL 上的回复，"
            "并把结果写到新的 JSONL 里，方便人工检查。"
        )
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="训练好的模型目录（train.py 的 output_dir）",
    )
    p.add_argument(
        "--eval_jsonl",
        type=str,
        required=True,
        help="评估输入 JSONL（格式与训练 JSONL 相同：包含 messages 与 tools 列）",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="保存带模型回复的 JSONL 路径",
    )
    p.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
        help="编码输入时的最大长度（与训练 max_seq_length 大致保持一致）",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="生成的最大新 token 数",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度（0 表示近似 greedy）",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="采样 top_p",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="只评估前 N 条样本（默认 None 表示全部）",
    )
    p.add_argument(
        "--vllm_batch_size",
        type=int,
        default=32,
        help="vLLM 并发生成时的批大小（提示条数），默认 32",
    )
    p.add_argument(
        "--log_wandb",
        action="store_true",
        help="将评估结果（含模型响应表格）记录到 wandb",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default="OpenResearch-SFT",
        help="wandb 项目名（与 train.py 一致）",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb 运行名（默认由 model_dir 与 eval_jsonl 推断）",
    )
    p.add_argument(
        "--per_turn_eval",
        action="store_true",
        help="按轮次评估：对每个 sample 的每个 assistant 回复，用「到该轮为止的 messages」分别生成并记录（否则整段对话只生成最后一轮）",
    )
    return p.parse_args()


def load_tools_field(sample):
    """把 sample['tools'] 统一转成 python 对象，用于传给 chat_template 的 tools 参数。"""
    tools = sample.get("tools")
    if tools is None:
        return None
    # 训练流水线中通常是 JSON 字符串
    if isinstance(tools, str):
        try:
            return json.loads(tools)
        except Exception:
            # 解析失败就直接返回原始字符串，后续可以忽略 tools
            return None
    return tools


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    eval_path = Path(args.eval_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not eval_path.is_file():
        raise FileNotFoundError(f"eval_jsonl 不存在: {eval_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir 不存在: {model_dir}")

    # ---------------- vLLM: 作为推理后端进行加速 ----------------
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError(
            "当前评估脚本已改为使用 vLLM 进行推理，请先安装 vllm：\n\n"
            "  pip install vllm\n"
        ) from e

    # 加载 tokenizer（其中已包含我们在 train.py 中设置的 chat_template）
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # vLLM 模型（直接加载你训练好的 checkpoint 目录）
    llm = LLM(model=str(model_dir), trust_remote_code=True)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # 用于 wandb 表格：仅记录 prompt / gold / model_response 三列
    table_rows = []
    total = 0
    total_response_chars = 0

    # 缓冲区：做批量并发生成
    batch_prompts = []
    # 每个元素: dict with
    #   sample_index: 当前对话在 eval_jsonl 中的索引（从 0 开始）
    #   sample: 原始 sample
    #   messages: 完整 messages
    #   turn_index: 当前评估的 assistant 轮次（整段评估时为 None）
    #   gold_display: 该轮的 gold assistant 文本
    batch_meta = []

    def flush_batch():
        nonlocal total, total_response_chars, table_rows
        if not batch_prompts:
            return
        outputs = llm.generate(batch_prompts, sampling_params)
        for i, out in enumerate(outputs):
            if not out.outputs:
                continue
            generated_text = out.outputs[0].text
            meta = batch_meta[i]
            sample_index = meta["sample_index"]
            sample = meta["sample"]
            messages = meta["messages"]
            prompt = batch_prompts[i]
            turn_index = meta.get("turn_index")
            gold_display = meta["gold_display"]

            out_obj = dict(sample)
            out_obj["model_response"] = generated_text
            if turn_index is not None:
                out_obj["eval_turn_index"] = turn_index
                out_obj["eval_gold_assistant_this_turn"] = gold_display
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            total += 1
            total_response_chars += len(generated_text)

            if args.log_wandb and _HAS_WANDB:
                # 记录 sample_id / turn_index + 核心三列：完整 prompt（经过 chat_template）、gold 回复、模型回复
                prompt_display = prompt  # 不再截断，便于在 wandb 中查看完整上下文
                resp_display = (generated_text[:3000] + "…") if len(generated_text) > 3000 else generated_text
                turn_id = -1 if turn_index is None else turn_index
                table_rows.append([sample_index, turn_id, prompt_display, gold_display, resp_display])

        batch_prompts.clear()
        batch_meta.clear()

    with eval_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        sample_count = 0
        for line in tqdm(fin, desc="eval-generate"):
            if args.num_samples is not None and sample_count >= args.num_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except Exception:
                continue

            messages = sample.get("messages")
            if not messages:
                continue

            # 当前对话在 eval_jsonl 中的索引（从 0 开始），用于 wandb 中区分不同轨迹
            sample_index = sample_count
            sample_count += 1

            tools = load_tools_field(sample)

            if args.per_turn_eval:
                # 按轮次：对每个 assistant 位置，用「到该轮为止的 messages」生成
                for i, m in enumerate(messages):
                    if m.get("role") != "assistant":
                        continue
                    prefix = messages[:i]
                    try:
                        prompt = tokenizer.apply_chat_template(
                            prefix,
                            tools=tools,
                            add_generation_prompt=True,
                            tokenize=False,
                            truncation=False,
                        )
                    except Exception:
                        continue
                    enc = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
                    if len(enc["input_ids"]) > 30000:
                        continue
                    gold_display = _assistant_turn_content(m)
                    batch_prompts.append(prompt)
                    batch_meta.append({
                        "sample_index": sample_index,
                        "sample": sample,
                        "messages": messages,
                        "turn_index": i,
                        "gold_display": gold_display,
                    })
                    if len(batch_prompts) >= args.vllm_batch_size:
                        flush_batch()
            else:
                # 整段对话只生成最后一轮
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        tokenize=False,
                        truncation=False,
                    )
                except Exception:
                    continue
                enc = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)
                if len(enc["input_ids"]) > 30000:
                    continue
                gold_display = _last_assistant_content(messages)
                batch_prompts.append(prompt)
                batch_meta.append({
                    "sample_index": sample_index,
                    "sample": sample,
                    "messages": messages,
                    "turn_index": None,
                    "gold_display": gold_display,
                })
                if len(batch_prompts) >= args.vllm_batch_size:
                    flush_batch()

        # 处理最后不足一批的样本
        flush_batch()

    if args.log_wandb and not _HAS_WANDB:
        print("⚠️ --log_wandb 已指定但未安装 wandb，跳过 wandb 记录。可执行: pip install wandb")
    if args.log_wandb and _HAS_WANDB and table_rows:
        run_name = args.wandb_run_name or f"eval_{model_dir.name}_{eval_path.stem}"
        wandb.init(project=args.wandb_project, name=run_name, job_type="eval")
        wandb.config.update({
            "model_dir": str(model_dir),
            "eval_jsonl": str(eval_path),
            "output_jsonl": str(out_path),
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_samples_evaluated": total,
            "per_turn_eval": args.per_turn_eval,
        })
        table = wandb.Table(
            columns=["sample_id", "turn_index", "prompt", "gold_assistant", "model_response"],
            data=table_rows,
        )
        wandb.log({"eval/responses": table})
        wandb.log({
            "eval/num_samples": total,
            "eval/avg_response_length_chars": total_response_chars / total if total else 0,
        })
        # 将完整输出 JSONL 作为 artifact，便于下载查看
        artifact = wandb.Artifact(name=f"eval_responses_{run_name}", type="eval_responses")
        artifact.add_file(str(out_path), name=out_path.name)
        wandb.log_artifact(artifact)
        wandb.finish()

    print(f"✅ eval_generate 完成，共生成 {total} 条样本，输出到: {out_path}")


if __name__ == "__main__":
    main()

