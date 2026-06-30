# -*- coding: utf-8 -*-
"""Model + LoRA adapter factory.

Supports loading via Unsloth (for GPU-optimised 4-bit training) with
automatic fallback to standard HuggingFace Transformers + PEFT when
Unsloth is not available.

This centralises all model-loading calls so the rest of the codebase
remains framework-agnostic and testable without a GPU.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from config import PipelineConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Creates a LoRA-wrapped causal language model.

    Attempts Unsloth first (2× faster training, lower VRAM), and falls
    back to standard HuggingFace ``AutoModelForCausalLM`` + PEFT if
    Unsloth is not installed.

    The factory pattern (SOLID – Single Responsibility) ensures model
    loading logic is isolated and can be swapped easily.
    """

    @staticmethod
    def load(config: PipelineConfig) -> Tuple[Any, Any]:
        """Load the backbone model and apply LoRA adapters.

        Args:
            config: Fully validated pipeline configuration.

        Returns:
            ``(model, tokenizer)`` — both ready for training.
        """
        try:
            return ModelFactory._load_unsloth(config)
        except ImportError:
            logger.warning(
                "Unsloth not found — falling back to HuggingFace + PEFT. "
                "Install Unsloth for 2× faster training."
            )
            return ModelFactory._load_hf(config)

    # ------------------------------------------------------------------
    # Unsloth backend
    # ------------------------------------------------------------------

    @staticmethod
    def _load_unsloth(config: PipelineConfig) -> Tuple[Any, Any]:
        """Load model via Unsloth's optimised path."""
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model_cfg = config.model
        lora_cfg = config.lora

        logger.info("Loading backbone via Unsloth: %s", model_cfg.name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.name,
            max_seq_length=model_cfg.max_seq_length,
            dtype=None,       # auto-detect
            load_in_4bit=True,
        )

        logger.info("Applying LoRA adapters (r=%d, alpha=%d)", lora_cfg.r, lora_cfg.alpha)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.r,
            target_modules=lora_cfg.target_modules,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=model_cfg.seed,
            use_rslora=False,
            loftq_config=None,
        )

        # Handle multimodal processors (e.g., Gemma4Processor).
        # These wrap a tokenizer internally but lack standard tokenizer
        # methods like .encode(), which breaks the rest of the pipeline.
        # We also need to preserve the chat_template from the processor
        # since the underlying tokenizer may not have it set.
        if hasattr(tokenizer, 'tokenizer') and not hasattr(tokenizer, 'encode'):
            logger.info(
                "Detected multimodal processor (%s) — extracting underlying tokenizer.",
                type(tokenizer).__name__,
            )
            processor_chat_template = getattr(tokenizer, 'chat_template', None)
            tokenizer = tokenizer.tokenizer
            if processor_chat_template and not tokenizer.chat_template:
                tokenizer.chat_template = processor_chat_template
                logger.info("Copied chat template from processor to tokenizer.")

        # Ensure the tokenizer has a chat template.
        if tokenizer.chat_template is None:
            model_name_lower = model_cfg.name.lower()
            if 'gemma' in model_name_lower:
                # Gemma uses <start_of_turn>/<end_of_turn> tokens which are
                # incompatible with Unsloth's get_chat_template("chatml").
                # Set the native Gemma chat template directly.
                logger.info("Gemma model detected — applying native Gemma chat template.")
                tokenizer.chat_template = (
                    "{{ bos_token }}"
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
                    "{% elif message['role'] == 'model' or message['role'] == 'assistant' %}"
                    "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
                    "{% elif message['role'] == 'system' %}"
                    "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "{{ '<start_of_turn>model\n' }}"
                    "{% endif %}"
                )
            else:
                logger.info("No chat template found — applying 'chatml'.")
                tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        else:
            logger.info("Using model's built-in chat template.")

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    # ------------------------------------------------------------------
    # Standard HuggingFace + PEFT backend
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf(config: PipelineConfig) -> Tuple[Any, Any]:
        """Load model via standard HuggingFace Transformers + PEFT."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model_cfg = config.model
        lora_cfg = config.lora

        logger.info("Loading backbone via HuggingFace: %s", model_cfg.name)

        # 4-bit quantisation config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        logger.info("Applying LoRA adapters (r=%d, alpha=%d)", lora_cfg.r, lora_cfg.alpha)
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.gradient_checkpointing_enable()

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Using model's built-in chat template.")
        return model, tokenizer
