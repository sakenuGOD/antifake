"""Загрузка модели (base / fine-tuned) через Unsloth (train) / transformers (inference)."""

import os
os.environ["XFORMERS_DISABLED"] = "1"

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
    """Загрузка fine-tuned модели (base + LoRA адаптеры) для inference.

    Используем plain transformers вместо Unsloth — Triton 3.2 не поддерживает
    Blackwell sm_120, а Unsloth патчит модель Triton-ядрами.
    """
    if config is None:
        config = ModelConfig()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Blackwell sm_120: отключаем Triton-based flash SDP
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # 4-bit NF4 квантизация через bitsandbytes (как при обучении)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Загрузка base-модели через plain transformers (без Unsloth/Triton)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # Загрузка и слияние LoRA адаптеров
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

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
