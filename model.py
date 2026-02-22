"""Загрузка модели (base / fine-tuned) через Unsloth."""

import torch
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline

from config import ModelConfig, PipelineConfig


def load_unsloth_model(config: ModelConfig = None):
    """Загрузка base-модели через Unsloth для обучения."""
    if config is None:
        config = ModelConfig()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    return model, tokenizer


def load_finetuned_model(adapter_path: str, config: ModelConfig = None):
    """Загрузка fine-tuned модели (base + LoRA адаптеры) для inference."""
    if config is None:
        config = ModelConfig()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_langchain_llm(
    model,
    tokenizer,
    max_new_tokens: int = 512,
    pipeline_config: PipelineConfig = None,
):
    """Обёртка модели в HuggingFacePipeline для использования в LangChain LCEL."""
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        repetition_penalty=pipeline_config.repetition_penalty,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)
