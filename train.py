# -*- coding: utf-8 -*-
"""Training entry point for multi-GPU execution via ``accelerate launch``.

Usage::

    accelerate launch --multi_gpu --num_processes=2 train.py \\
        --config configs/default.yaml

The script is deliberately kept thin: all logic lives in the
``ner_re_pipeline`` package so it can be imported, tested, and reused
without executing a full training run.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from config import PipelineConfig, TrainingFormat
from data.registry import DatasetRegistry
from prompts.factory import PromptBuilderFactory
from training.collator import DataCollatorWithLossMask
from training.model_factory import ModelFactory
from training.trainer import NativeSafeTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing (tokenisation)
# ---------------------------------------------------------------------------


def make_preprocess_fn(tokenizer, config: PipelineConfig):
    """Return a ``datasets.Dataset.map``-compatible preprocessing function.

    For the **CAUSAL** format: tokenise the single ``"text"`` column.
    For the **INSTRUCTION** format: apply the chat template and store
    ``prompt_length`` so the collator can mask it from the loss.
    """
    max_len = config.model.max_seq_length

    if config.training.format == TrainingFormat.CAUSAL:
        def preprocess_causal(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_len,
                padding=False,
            )
        return preprocess_causal

    # INSTRUCTION format
    def preprocess_instruction(examples):
        full_texts = []
        prompt_lengths = []

        for sys_msg, usr_msg, ast_msg in zip(
            examples["system"], examples["user"], examples["assistant"]
        ):
            # Full sequence: system + user + assistant
            full_messages = [
                {"role": "system",    "content": sys_msg},
                {"role": "user",      "content": usr_msg},
                {"role": "assistant", "content": ast_msg},
            ]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )

            # Prompt-only sequence to compute prompt_length
            prompt_messages = full_messages[:2]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(
                tokenizer.encode(prompt_text, add_special_tokens=False)
            )

            full_texts.append(full_text)
            prompt_lengths.append(prompt_len)

        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        tokenized["prompt_length"] = prompt_lengths
        return tokenized

    return preprocess_instruction


# ---------------------------------------------------------------------------
# Training arguments helper
# ---------------------------------------------------------------------------


def build_training_args(config: PipelineConfig, num_samples: int) -> TrainingArguments:
    """Compute warmup steps and return a ``TrainingArguments`` instance."""
    tr = config.training
    total_steps = math.ceil(
        num_samples / (tr.batch_size * tr.gradient_accumulation_steps)
    ) * tr.epochs
    warmup_steps = max(1, int(total_steps * tr.warmup_ratio))

    return TrainingArguments(
        output_dir=tr.output_dir,
        per_device_train_batch_size=tr.batch_size,
        gradient_accumulation_steps=tr.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=tr.epochs,
        learning_rate=tr.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=tr.logging_steps,
        eval_strategy="steps",
        eval_steps=tr.eval_steps,
        save_strategy="steps",
        save_steps=tr.save_steps,
        optim="adamw_8bit",
        weight_decay=tr.weight_decay,
        lr_scheduler_type="linear",
        seed=config.model.seed,
        report_to="none",
        # Early stopping requires loading the best model
        load_best_model_at_end=(tr.early_stopping_patience > 0),
        metric_for_best_model="eval_loss" if (tr.early_stopping_patience > 0) else None,
        # DDP / gradient-checkpointing safety flags
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, # VERY IMPORTANT: prevent Trainer from dropping prompt_length
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM for clinical NER + RE."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a checkpoint. Use --resume to auto-detect the latest in output_dir, or --resume path/to/checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading config from: %s", args.config)
    config = PipelineConfig.from_yaml(args.config)
    set_seed(config.model.seed)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Aggregating training files …")
    train_docs = DatasetRegistry(config.data.train_files).aggregate()
    logger.info("Aggregating evaluation files …")
    eval_docs = DatasetRegistry(config.data.eval_files).aggregate()

    # ------------------------------------------------------------------
    # 2. Build prompt datasets
    # ------------------------------------------------------------------
    logger.info(
        "Building prompts — format=%s  task=%s  re_mode=%s",
        config.training.format.value,
        config.training.task_mode.value,
        config.training.re_mode.value,
    )
    builder = PromptBuilderFactory.create(config)
    train_raw = builder.generate_training_data(train_docs)
    eval_raw = builder.generate_training_data(eval_docs)
    
    if len(train_raw) == 0:
        raise ValueError("Training dataset is empty. Please check if the file paths in config are correct and files exist.")
        
    logger.info(
        "Train samples: %d  |  Eval samples: %d",
        len(train_raw),
        len(eval_raw),
    )

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    model, tokenizer = ModelFactory.load(config)

    # ------------------------------------------------------------------
    # 4. Tokenise
    # ------------------------------------------------------------------
    preprocess_fn = make_preprocess_fn(tokenizer, config)
    remove_cols = list(train_raw.column_names)

    logger.info("Tokenising training data …")
    train_dataset = train_raw.map(
        preprocess_fn,
        batched=True,
        remove_columns=remove_cols,
        load_from_cache_file=False,
    )
    logger.info("Tokenising evaluation data …")
    eval_dataset = eval_raw.map(
        preprocess_fn,
        batched=True,
        remove_columns=list(eval_raw.column_names),
        load_from_cache_file=False,
    )

    # Subset eval dataset to speed up validation during training
    subset_val = config.training.eval_subset_size
    if subset_val is not None and subset_val != 1.0:
        if isinstance(subset_val, float):
            subset_size = int(len(eval_dataset) * subset_val)
        else:
            subset_size = subset_val
            
        subset_size = max(1, min(subset_size, len(eval_dataset)))
        logger.info(
            "Subsetting eval dataset to %s (%d samples) for faster validation during training.",
            f"{int(subset_val * 100)}%%" if isinstance(subset_val, float) else f"{subset_val} items",
            subset_size
        )
        eval_dataset = eval_dataset.select(range(subset_size))

    # ------------------------------------------------------------------
    # 5. Data collator — choose based on format
    # ------------------------------------------------------------------
    if config.training.format == TrainingFormat.INSTRUCTION:
        data_collator = DataCollatorWithLossMask(tokenizer)
        logger.info("Using DataCollatorWithLossMask (prompt tokens masked from loss).")
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        logger.info("Using standard DataCollatorForLanguageModeling (causal).")

    # ------------------------------------------------------------------
    # 6. Build trainer & train
    # ------------------------------------------------------------------
    training_args = build_training_args(config, len(train_dataset))

    trainer_kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "args": training_args,
    }
    
    if config.training.early_stopping_patience > 0:
        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        ]
        logger.info("Early stopping enabled (patience=%d)", config.training.early_stopping_patience)

    trainer = NativeSafeTrainer(**trainer_kwargs)

    logger.info("Starting training …")
    
    resume_ckpt = None
    if args.resume:
        if args.resume == "latest":
            resume_ckpt = True # HF Trainer auto-detects latest in output_dir
            logger.info("Resuming from latest checkpoint in %s...", config.training.output_dir)
        else:
            resume_ckpt = args.resume
            logger.info("Resuming from specific checkpoint: %s...", resume_ckpt)
            
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ------------------------------------------------------------------
    # 7. Save (only on rank 0 to avoid file conflicts)
    # ------------------------------------------------------------------
    if trainer.is_world_process_zero():
        # ── 7a. LoRA adapter only (lightweight, ~100-400 MB) ──────────
        lora_path = os.path.join(config.training.output_dir, "final_lora_adapter")
        trainer.model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)
        logger.info("LoRA adapter saved to: %s", lora_path)

        # ── 7b. Merged model (base + LoRA, ready for inference) ───────
        # Unsloth provides a native method for merging 4-bit LoRA weights.
        # For standard HuggingFace + PEFT, we use merge_and_unload().
        merged_path = os.path.join(config.training.output_dir, "final_merged_model")
        if hasattr(trainer.model, "save_pretrained_merged"):
            # Unsloth path
            logger.info("Merging LoRA weights via Unsloth (16-bit) …")
            trainer.model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        else:
            # Standard PEFT path
            logger.info("Merging LoRA weights via PEFT merge_and_unload() …")
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
        logger.info("Merged model saved to: %s", merged_path)

        logger.info(
            "\n[DONE] Outputs in %s\n"
            "  ├── final_lora_adapter/   ← LoRA weights only\n"
            "  └── final_merged_model/   ← Full model (base + LoRA merged)",
            config.training.output_dir,
        )


if __name__ == "__main__":
    main()
