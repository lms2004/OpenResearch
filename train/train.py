#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练流程（推荐顺序）：
  1) 数据准备: 原始数据 → parse.py → materialize → sft_dataset.py → *.tools_sft.jsonl
  2) 训练: train.py --model_name_or_path <base> --train_jsonl <train.tools_sft.jsonl> [--eval_jsonl ...]
  3) 评估: eval_generate.py --model_dir <output_dir> --eval_jsonl ... --output_jsonl ... [--log_wandb]
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# 与 parse.py / sft_dataset.py 流水线一致：默认数据为 tools_sft.jsonl，输出到 train/outputs
_train_dir = Path(__file__).resolve().parent
DEFAULT_TRAIN_JSONL = _train_dir / "datasets/data/converted_gpt_oss_search_correct.tools_sft.jsonl"
DEFAULT_OUTPUT_DIR = _train_dir / "outputs/sft_full"


def parse_args():
    p = argparse.ArgumentParser(description="全参数 SFT 训练（无 LoRA），数据默认接 parse → sft_dataset 流水线输出。")
    p.add_argument("--model_name_or_path", type=str, required=True, help="基座模型路径或 HuggingFace 名")
    p.add_argument(
        "--train_jsonl",
        type=str,
        default=str(DEFAULT_TRAIN_JSONL),
        help=f"训练 JSONL（默认: {DEFAULT_TRAIN_JSONL}）",
    )
    p.add_argument(
        "--eval_jsonl",
        type=str,
        default=None,
        help="可选验证集 JSONL；若不提供，则从 train_jsonl 中按 --val_ratio 自动切分验证集",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.02,
        help="当未提供 --eval_jsonl 时，从训练集中划分为验证集的比例（默认 0.02，即 2%）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f" checkpoint 输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )

    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--run_name", type=str, default=None, help="wandb 运行名（默认由 output_dir 推断）")
    p.add_argument("--wandb_project", type=str, default="OpenResearch-SFT", help="wandb 项目名")

    # 工具调用数据一般不建议 packing（会把多个对话硬拼到一起，可能影响 tool 结构）
    p.add_argument("--packing", action="store_true", help="谨慎开启；tool calling 数据通常建议关闭")

    # 训练时在每次 eval 后采样若干条验证集，生成模型响应并记入 wandb 表格，便于查看
    p.add_argument("--log_eval_responses_wandb", action="store_true",
                   help="每次评估后对验证集采样生成回复，并记录到 wandb 表格（需提供 --eval_jsonl）")
    p.add_argument("--eval_response_samples", type=int, default=5,
                   help="每次评估时采样多少条验证样本做生成并记入 wandb（默认 5）")
    p.add_argument("--eval_response_max_new_tokens", type=int, default=256,
                   help="训练时记录响应生成的最大 token 数（默认 256）")

    return p.parse_args()


def ensure_tools_is_json_str(example):
    """
    TRL 推荐 tools 列存 JSON string（json.dumps([...])）。
    如果你数据里 tools 是 list[dict]（legacy），这里转一下。
    """
    if "tools" not in example:
        return example
    tools = example["tools"]
    if isinstance(tools, list):
        example["tools"] = json.dumps(tools, ensure_ascii=False)
    return example


def _load_tools_for_template(sample):
    tools = sample.get("tools")
    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            return json.loads(tools)
        except Exception:
            return None
    return tools


def _last_user_content(messages):
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return (m.get("content") or "")[:1500]
    return ""


def _last_assistant_content(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = m.get("content") or ""
            if c:
                return c[:1500]
            if m.get("tool_calls"):
                return "[tool_calls]"
    return ""


class EvalResponseLoggingCallback(TrainerCallback):
    """每次评估后对验证集采样做生成，并将结果记入 wandb 表格，便于查看模型响应。"""

    def __init__(self, tokenizer, eval_dataset, num_samples=5, max_new_tokens=256, max_length=4096):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = min(num_samples, len(eval_dataset)) if eval_dataset else 0
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

    def on_evaluate(self, args, state, model, **kwargs):
        if not _HAS_WANDB or self.num_samples <= 0 or state.global_step <= 0:
            return
        model.eval()
        device = next(model.parameters()).device
        table_rows = []
        with torch.no_grad():
            for i in range(self.num_samples):
                sample = self.eval_dataset[i]
                messages = sample.get("messages")
                if not messages:
                    continue
                tools = _load_tools_for_template(sample)
                try:
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        tools=tools,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    ).to(device)
                except Exception:
                    continue
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_ids = out[0, input_ids.shape[1]:]
                generated = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
                last_user = _last_user_content(messages)
                gold = _last_assistant_content(messages)
                resp_display = (generated[:2000] + "…") if len(generated) > 2000 else generated
                table_rows.append([state.global_step, i, last_user, gold, resp_display])
        if table_rows:
            table = wandb.Table(
                columns=["step", "sample_idx", "last_user", "gold_assistant", "model_response"],
                data=table_rows,
            )
            wandb.log({"train_eval/responses": table}, step=state.global_step)
        model.train()


def main():
    args = parse_args()
    if not os.path.isfile(args.train_jsonl):
        raise FileNotFoundError(
            f"训练数据不存在: {args.train_jsonl}\n"
            "请先运行: python train/datasets/parse.py && python train/datasets/sft_dataset.py"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 加载 jsonl
    data_files = {"train": args.train_jsonl}
    if args.eval_jsonl:
        data_files["validation"] = args.eval_jsonl
    ds = load_dataset("json", data_files=data_files)

    # 如果用户没有显式提供 eval_jsonl，就从训练集中按比例切出一部分作为验证集
    if not args.eval_jsonl:
        full_train = ds["train"]
        # 保底：样本太少时避免切出空验证集
        if len(full_train) >= 10 and 0.0 < args.val_ratio < 1.0:
            split = full_train.train_test_split(test_size=args.val_ratio, seed=42)
            ds_train = split["train"]
            ds_val = split["test"]
            print(f"自动从训练集中划分验证集: 总样本={len(full_train)}, "
                  f"train={len(ds_train)}, val={len(ds_val)}, 比例={args.val_ratio}")
        else:
            ds_train = full_train
            ds_val = None
            print("样本量过小或 val_ratio 非法，跳过自动划分验证集，仅使用训练集。")
    else:
        ds_train = ds["train"]
        ds_val = ds.get("validation", None)

    # 2) 统一把 tools 转成 JSON string（如果需要）
    ds_train = ds_train.map(ensure_tools_is_json_str)
    if ds_val is not None:
        ds_val = ds_val.map(ensure_tools_is_json_str)

    # 3) 基本字段检查
    cols = ds_train.column_names
    if "messages" not in cols:
        raise ValueError(f"你的数据缺少 'messages' 列，当前列：{cols}")
    # tools 列：强烈建议有（tool calling 场景）
    if "tools" not in cols:
        print("⚠️ 警告：你的数据没有 'tools' 列。tool calling SFT 通常需要 tools 列给 chat template 构造系统提示。")

    # 4) 模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 与数据流水线保持一致：自动加载 all_assistant.jinja 作为 chat template
    chat_template_path = _train_dir / "datasets/templates/all_assistant.jinja"
    if not chat_template_path.is_file():
        raise FileNotFoundError(f"chat template 不存在: {chat_template_path}")
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    # 避免某些模型没有 pad_token 导致 Trainer 报错
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
    )

    # 5) SFT 配置
    has_validation = ds_val is not None
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        packing=args.packing,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=args.eval_steps if has_validation else None,

        bf16=args.bf16,
        fp16=args.fp16,
        assistant_only_loss=True,
        report_to="wandb",
        run_name=args.run_name or Path(args.output_dir).name,
        project=args.wandb_project,
    )

    # 6) Trainer
    callbacks = []
    if args.log_eval_responses_wandb and has_validation and _HAS_WANDB:
        callbacks.append(
            EvalResponseLoggingCallback(
                tokenizer=tokenizer,
                eval_dataset=ds_val,
                num_samples=args.eval_response_samples,
                max_new_tokens=args.eval_response_max_new_tokens,
                max_length=args.max_seq_length,
            )
        )
    elif args.log_eval_responses_wandb and not has_validation:
        print("⚠️ --log_eval_responses_wandb 已指定但没有有效的验证集（既未提供 --eval_jsonl，也未成功自动划分），已忽略。")
    elif args.log_eval_responses_wandb and not _HAS_WANDB:
        print("⚠️ --log_eval_responses_wandb 已指定但未安装 wandb，已忽略。可执行: pip install wandb")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # 7) 开训
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()