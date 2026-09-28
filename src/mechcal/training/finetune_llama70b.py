"""Choice-only QLoRA SFT of the Llama-3.1-70B base model."""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path


CHOICE_PATTERN = re.compile(r"<<(.*?)>>")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-name",
        default="unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def _mask_choice_tokens(tokenizer, text: str, max_seq_length: int) -> dict[str, list[int]]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded.pop("offset_mapping")
    labels = [-100] * len(encoded["input_ids"])
    spans = [(match.start(1), match.end(1)) for match in CHOICE_PATTERN.finditer(text)]
    for token_index, (start, end) in enumerate(offsets):
        if start == end == 0:
            continue
        if any(start >= span_start and end <= span_end for span_start, span_end in spans):
            labels[token_index] = encoded["input_ids"][token_index]
    encoded["labels"] = labels
    encoded["n_choice_spans"] = len(spans)
    encoded["n_supervised_tokens"] = sum(label != -100 for label in labels)
    return encoded


def main() -> None:
    args = _arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")

    import unsloth  # noqa: F401 - must precede transformers imports
    import torch
    from datasets import load_dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from unsloth import FastLanguageModel, is_bfloat16_supported

    set_seed(args.seed)
    raw = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.val_file)},
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = raw.map(
        lambda row: _mask_choice_tokens(tokenizer, row["text"], args.max_seq_length),
        remove_columns=raw["train"].column_names,
        num_proc=1,
        desc="Tokenizing and applying choice-only labels",
    )
    audit = {}
    for split in ("train", "validation"):
        span_counts = tokenized[split]["n_choice_spans"]
        target_counts = tokenized[split]["n_supervised_tokens"]
        lengths = [len(values) for values in tokenized[split]["input_ids"]]
        audit[split] = {
            "n_sequences": len(tokenized[split]),
            "choice_span_min": min(span_counts),
            "choice_span_max": max(span_counts),
            "supervised_token_min": min(target_counts),
            "supervised_token_max": max(target_counts),
            "token_length_min": min(lengths),
            "token_length_max": max(lengths),
        }
        if min(span_counts) != 200 or max(span_counts) != 200:
            raise ValueError(f"{split} does not contain exactly 200 choice spans per sequence")
        if min(target_counts) != 200 or max(target_counts) != 200:
            raise ValueError(
                f"{split} choices do not each map to exactly one supervised token: "
                f"range={min(target_counts)}..{max(target_counts)}"
            )
        if max(lengths) >= args.max_seq_length:
            raise ValueError(f"{split} reaches truncation boundary {args.max_seq_length}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tokenization_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if args.smoke_only:
        return

    tokenized = tokenized.remove_columns(["n_choice_spans", "n_supervised_tokens"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=True,
        loftq_config=None,
    )

    kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.05,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    kwargs[eval_key] = "epoch"
    training_args = TrainingArguments(**kwargs)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        ],
    )
    model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model(str(args.output_dir / "best_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "best_adapter"))
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    metrics["best_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_metric"] = trainer.state.best_metric
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
