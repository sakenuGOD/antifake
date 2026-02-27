"""Загрузка модели (base / fine-tuned) через transformers + bitsandbytes.

Поддерживает:
  - Base модель (4-bit NF4) — для обучения и inference
  - SFT адаптеры — для inference после SFT
  - GRPO адаптеры — для inference после GRPO

RTX 5070 (Blackwell sm_120): используем SDPA вместо Triton-based flash attention.
Unsloth удалён — его Triton-ядра вызывают segfault на sm_120.
"""

import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline

from config import ModelConfig, PipelineConfig, PROJECT_ROOT


def find_best_adapter() -> str | None:
    """Автоматический выбор лучших адаптеров: GRPO > SFT > None."""
    grpo_path = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_grpo")
    sft_path = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_lora")

    if os.path.exists(grpo_path):
        return grpo_path
    elif os.path.exists(sft_path):
        return sft_path
    return None


def is_grpo_adapter(adapter_path: str) -> bool:
    """Проверяет, являются ли адаптеры GRPO (а не SFT)."""
    return adapter_path is not None and "grpo" in os.path.basename(adapter_path).lower()


def load_base_model(config: ModelConfig = None):
    """Загрузка base-модели через transformers + bitsandbytes (4-bit NF4).

    Используется для inference без адаптеров. Для обучения используется train_grpo.py.
    """
    if config is None:
        config = ModelConfig()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    base_model = config.base_model_name
    print(f"  Загрузка base модели: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model.eval()
    return model, tokenizer


def load_finetuned_model(adapter_path: str, config: ModelConfig = None):
    """Загрузка fine-tuned модели (base + LoRA адаптеры) для inference.

    Поддерживает два режима:
      1. SFT адаптеры: base → apply SFT → merge
      2. GRPO адаптеры: base → apply SFT → merge → apply GRPO → merge

    Используем plain transformers вместо Unsloth — Triton 3.2 не поддерживает
    Blackwell sm_120, а Unsloth патчит модель Triton-ядрами.
    """
    if config is None:
        config = ModelConfig()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Очистка VRAM перед загрузкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Blackwell sm_120: отключаем Triton-based flash SDP
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # Используем ту же pre-quantized модель что и при обучении (bnb-4bit).
    # Адаптеры обучены на ней → 100% совместимость весов.
    base_model = config.base_model_name
    print(f"  Загрузка base модели: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # GRPO адаптеры уже содержат SFT-веса (GRPO дообучает SFT LoRA напрямую),
    # поэтому загрузка одноэтапная — как и для SFT адаптеров.
    print(f"  Применение адаптеров из {adapter_path}")
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

    # Patch model's generation_config to remove conflicting defaults
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = None
        model.generation_config.temperature = None
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        max_length=None,
        repetition_penalty=pipeline_config.repetition_penalty,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)
