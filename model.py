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
from transformers import pipeline as hf_pipeline, StoppingCriteria, StoppingCriteriaList
from langchain_huggingface import HuggingFacePipeline

from config import ModelConfig, PipelineConfig, PROJECT_ROOT


class StopOnString(StoppingCriteria):
    """V11: Stop generation when a stop string is encountered in recent tokens."""
    def __init__(self, tokenizer, stop_strings: list):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(self, input_ids, scores, **kwargs):
        # Check last 20 tokens for stop strings (efficient)
        recent = self.tokenizer.decode(
            input_ids[0, -20:], skip_special_tokens=True
        )
        return any(s in recent for s in self.stop_strings)


def find_best_adapter() -> str | None:
    """Выбор лучшего адаптера: валидированный GRPO > SFT > None.

    V18: GRPO проверяется через train_metrics.json — если epoch < 1.0 или reward
    отрицательный, адаптер считается битым и игнорируется. Предыдущий checkpoint-100
    имел epoch=0.04 и correctness_reward=-0.5 — использование такого GRPO
    отравляло LLM parametric check на базовых фактах.
    """
    grpo_path = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_grpo")
    sft_path = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_lora")

    if os.path.exists(grpo_path) and _grpo_is_valid(grpo_path):
        return grpo_path
    if os.path.exists(sft_path):
        return sft_path
    return None


def _grpo_is_valid(grpo_path: str) -> bool:
    """GRPO считается валидным только если обучился хотя бы 1 эпоху."""
    metrics_file = os.path.join(grpo_path, "train_metrics.json")
    if not os.path.exists(metrics_file):
        return False
    try:
        import json
        with open(metrics_file, "r", encoding="utf-8") as f:
            m = json.load(f)
        return float(m.get("epoch", 0)) >= 1.0
    except Exception:
        return False


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
    # Do NOT call merge_and_unload() — merging LoRA into a 4-bit quantized base
    # round-trips through NF4 quantization and introduces rounding error
    # (PEFT warns: "Merge lora module to 4-bit linear may get different
    # generations due to rounding errors"). Keep LoRA as an overlay for
    # deterministic, drift-free inference. A small speed cost is acceptable.
    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def build_langchain_llm(
    model,
    tokenizer,
    max_new_tokens: int = 512,
    pipeline_config: PipelineConfig = None,
    stop_strings: list = None,
    sampling: bool = False,
    temperature: float = 0.3,
    top_p: float = 0.95,
):
    """Обёртка модели в HuggingFacePipeline для использования в LangChain LCEL.

    Args:
        stop_strings: Custom stop strings. Default (None) uses keyword-extraction
            stops: ["</answer>", "ВЕРДИКТ:", "</reasoning>\\n\\nВЕРДИКТ:"].
            Pass explicit list to override (e.g. ["</answer>"] for explain_llm).
        sampling: If True, enable do_sample + temperature (for Self-Consistency).
            Default False = deterministic greedy decoding (for keyword extraction
            and single-pass verdicts).
        temperature: Sampling temperature when sampling=True. 0.3 gives diversity
            without destroying coherence; 0.7+ tends to hallucinate on 7B models.
        top_p: Nucleus sampling threshold when sampling=True.
    """
    if pipeline_config is None:
        pipeline_config = PipelineConfig()

    # Set generation params directly on model.generation_config so hf_pipeline
    # doesn't double-pass them. Without this we get "multiple values for
    # keyword argument" on transformers 4.57+.
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.repetition_penalty = pipeline_config.repetition_penalty
        if sampling:
            model.generation_config.do_sample = True
            model.generation_config.temperature = temperature
            model.generation_config.top_p = top_p
        else:
            model.generation_config.do_sample = False
            # Setting temperature=None is required by transformers when do_sample=False,
            # otherwise it logs a spurious "temperature ignored" warning every call.
            model.generation_config.temperature = None
            model.generation_config.top_p = None

    if stop_strings is None:
        stop_strings = ["</answer>", "ВЕРДИКТ:", "</reasoning>\n\nВЕРДИКТ:"]
    stop_criteria = StoppingCriteriaList([
        StopOnString(tokenizer, stop_strings)
    ])

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        stopping_criteria=stop_criteria,
    )
    return HuggingFacePipeline(pipeline=pipe)
