"""Загрузка модели (base / fine-tuned) через Unsloth (train) / transformers (inference).

Поддерживает:
  - Base модель (Unsloth) — для обучения
  - SFT адаптеры — для inference после SFT
  - GRPO адаптеры — для inference после SFT + GRPO (2-этапная загрузка)

RTX 5070 (Blackwell sm_120): используем SDPA вместо Triton-based flash attention.
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

    Поддерживает два режима:
      1. SFT адаптеры: base → apply SFT → merge
      2. GRPO адаптеры: base → apply SFT → merge → apply GRPO → merge

    Используем plain transformers вместо Unsloth — Triton 3.2 не поддерживает
    Blackwell sm_120, а Unsloth патчит модель Triton-ядрами.
    """
    if config is None:
        config = ModelConfig()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Очистка VRAM перед загрузкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    # Загрузка ОРИГИНАЛЬНОЙ модели через plain transformers (без Unsloth/Triton)
    # ВАЖНО: используем inference_model_name, НЕ base_model_name (Unsloth 4-bit)
    # Двойная квантизация уже квантованной модели → деградация весов
    print(f"  Загрузка base модели: {config.inference_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.inference_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.inference_model_name)

    # GRPO адаптеры: двухэтапная загрузка (SFT → merge → GRPO → merge)
    if is_grpo_adapter(adapter_path):
        sft_path = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_lora")
        if os.path.exists(sft_path):
            print(f"  Этап 1: Применение SFT адаптеров из {sft_path}")
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()

        print(f"  Этап 2: Применение GRPO адаптеров из {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    else:
        # SFT адаптеры: одноэтапная загрузка
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
