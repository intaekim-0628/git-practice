#!/usr/bin/env python3
"""Single-H200 QLoRA SFT for gpt-oss-120b Unsloth BnB 4-bit."""

import argparse
import json
import os
from pathlib import Path

# Import Unsloth before transformers/trl so its patches are applied.
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/gpt-oss-120b-unsloth-bnb-4bit")
    p.add_argument("--train-file", required=True)
    p.add_argument("--eval-file")
    p.add_argument("--output-dir", default="output-gpt-oss-120b-sql-lora")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume-from-checkpoint", nargs="?", const=True, default=False)
    return p.parse_args()


def normalize_messages(row):
    if isinstance(row.get("messages"), list):
        return row["messages"]
    if isinstance(row.get("conversations"), list):
        role_map = {"human": "user", "gpt": "assistant"}
        return [
            {"role": role_map.get(m.get("from"), m.get("role", "user")),
             "content": m.get("value", m.get("content", ""))}
            for m in row["conversations"]
        ]
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    answer = row.get("output") or row.get("response") or row.get("answer")
    if instruction is None or answer is None:
        raise ValueError("Each JSONL row needs messages/conversations or instruction+output.")
    user = str(instruction)
    if row.get("input"):
        user += "\n\n" + str(row["input"])
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": str(row["system"])})
    messages.extend((
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(answer)},
    ))
    return messages


def load_jsonl(path, tokenizer):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                messages = normalize_messages(raw)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                rows.append({"text": text})
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return Dataset.from_list(rows)


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This configuration expects exactly one CUDA GPU.")
    props = torch.cuda.get_device_properties(0)
    print(f"[GPU] {props.name}, VRAM={props.total_memory / 2**30:.1f} GiB")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU support is required.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        full_finetuning=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GPT-OSS MoE expert tensors are fused/stacked. Attention-only LoRA avoids
    # invalid gate/down projection targeting and is the merge-safe default.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    train_ds = load_jsonl(args.train_file, tokenizer)
    eval_ds = load_jsonl(args.eval_file, tokenizer) if args.eval_file else None
    cfg = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.save_steps if eval_ds is not None else None,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = dict(result.metrics)
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    print(f"[DONE] LoRA adapter: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
