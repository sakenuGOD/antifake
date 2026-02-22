"""Конфигурация: пути, API-ключи, гиперпараметры.

Оптимизировано под RTX 5070 (12GB GDDR7, Blackwell sm_120, bf16).
"""

from dataclasses import dataclass, field
from typing import List
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ModelConfig:
    base_model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    max_seq_length: int = 2048           # 2048 для длинных источников + reasoning
    dtype: str = None  # auto-detect (bf16 на Blackwell)
    load_in_4bit: bool = True


@dataclass
class LoraConfig:
    r: int = 16              # 16 для 12GB VRAM
    lora_alpha: int = 32     # alpha = 2*r
    lora_dropout: float = 0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class TrainingConfig:
    output_dir: str = "adapters"
    num_train_epochs: int = 1             # 1 эпоха — лосс сходится быстро
    per_device_train_batch_size: int = 2   # 2 — safe для 12GB
    gradient_accumulation_steps: int = 4   # эффективный batch = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10                # 10 шагов разогрева (Unsloth рекомендация)
    lr_scheduler_type: str = "linear"     # linear для коротких тренировок (1 эпоха)
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = True                     # нативная поддержка bf16 на Blackwell
    logging_steps: int = 10               # чаще логировать для мониторинга
    save_strategy: str = "epoch"            # сохранять только в конце
    save_steps: int = 5000
    save_total_limit: int = 1
    seed: int = 3407
    dataset_path: str = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
    dataloader_num_workers: int = 0       # 0 для Windows (multiprocessing incompatible with Unsloth)
    dataloader_pin_memory: bool = True    # pin memory для быстрой передачи в GPU
    tf32: bool = True                     # TF32 для матричных операций


@dataclass
class SearchConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("SERPAPI_API_KEY", ""))
    gl: str = "ru"
    hl: str = "ru"
    tbm: str = "nws"
    num_results: int = 5


@dataclass
class PipelineConfig:
    keyword_max_new_tokens: int = 128
    verdict_max_new_tokens: int = 1500  # увеличено для Chain-of-Thought рассуждений
    temperature: float = 0.1
    repetition_penalty: float = 1.15
