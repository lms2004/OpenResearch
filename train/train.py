#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# 与 parse.py / sft_dataset.py 流水线一致：默认数据为 tools_sft.jsonl，输出到 train/outputs
_train_dir = Path(__file__).resolve().parent
DEFAULT_TRAIN_JSONL = _train_dir / "datasets/data/converted_gpt_oss_search_correct.tools_sft.jsonl"
DEFAULT_OUTPUT_DIR = _train_dir / "outputs/sft_full"


def parse_args():
    p = argparse.ArgumentParser(description="全参数 SFT 训练（无 LoRA），数据默认接 parse → sft_dataset 流水线输出。")
    p.add_argument("--model_name_or_path", type=str, required=True, help="基座模型路径或 HuggingFace 名")
    p.add_argument("--train_jsonl", type=str, default=str(DEFAULT_TRAIN_JSONL),
                   help=f"训练 JSONL（默认: {DEFAULT_TRAIN_JSONL}）")
    p.add_argument("--eval_jsonl", type=str, default=None, help="可选验证集 JSONL")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help=f" checkpoint 输出目录（默认: {DEFAULT_OUTPUT_DIR}）")

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

    # 2) 统一把 tools 转成 JSON string（如果需要）
    ds = ds.map(ensure_tools_is_json_str)

    # 3) 基本字段检查
    cols = ds["train"].column_names
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
        eval_strategy="steps" if args.eval_jsonl else "no",
        eval_steps=args.eval_steps if args.eval_jsonl else None,

        bf16=args.bf16,
        fp16=args.fp16,
        assistant_only_loss=True,
        report_to="wandb",
        run_name=args.run_name or Path(args.output_dir).name,
        project=args.wandb_project,
    )

    # 6) Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        processing_class=tokenizer,
    )

    # 7) 开训
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()