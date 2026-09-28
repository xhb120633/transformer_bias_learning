"""Original Centaur SFT pipeline with only the dataset source made local."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Pretrained model path or Hugging Face identifier"}
    )
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)


@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "Local train JSONL"})
    eval_file: str = field(metadata={"help": "Local validation JSONL"})
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=32768)
    early_stopping_patience: int = field(default=0)


def main(model_args, data_args, training_args) -> None:
    # Imports and calls below intentionally mirror the original Centaur finetune.py.
    import unsloth  # noqa: F401 - must precede transformers in the current environment
    from datasets import load_dataset
    from transformers import EarlyStoppingCallback, TrainingArguments, set_seed
    from trl import DataCollatorForCompletionOnlyLM
    from unsloth import (
        FastLanguageModel,
        UnslothTrainer,
        UnslothTrainingArguments,
        is_bfloat16_supported,
    )

    set_seed(training_args.seed)
    datasets = load_dataset(
        "json",
        data_files={"train": data_args.train_file, "test": data_args.eval_file},
    )
    train_dataset = datasets["train"].shuffle()
    eval_dataset = datasets["test"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=data_args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_args.seed,
        use_rslora=True,
        loftq_config=None,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    left_id = tokenizer(" <<").input_ids[1:]
    right_id = tokenizer(">>").input_ids[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=left_id,
        instruction_template=right_id,
        tokenizer=tokenizer,
    )

    eval_strategy = getattr(
        training_args,
        "evaluation_strategy",
        getattr(training_args, "eval_strategy", "steps"),
    )
    unsloth_args = UnslothTrainingArguments(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=training_args.warmup_steps,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        learning_rate=training_args.learning_rate,
        embedding_learning_rate=training_args.learning_rate / 10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        log_level=training_args.log_level,
        logging_strategy=training_args.logging_strategy,
        logging_steps=training_args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=training_args.eval_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        optim=training_args.optim,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type=training_args.lr_scheduler_type,
        seed=training_args.seed,
        output_dir=training_args.output_dir,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        greater_is_better=training_args.greater_is_better,
        save_total_limit=training_args.save_total_limit,
        report_to="none",
    )
    callbacks = []
    if data_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=data_args.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_num_proc=8,
        data_collator=collator,
        args=unsloth_args,
        callbacks=callbacks,
    )
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()


if __name__ == "__main__":
    import unsloth  # noqa: F401
    from transformers import HfArgumentParser, TrainingArguments

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
