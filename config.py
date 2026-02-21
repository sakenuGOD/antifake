"""Конфигурация: пути, API-ключи, гиперпараметры.

Оптимизировано под RTX 5070 (12GB GDDR7, Blackwell sm_120, bf16).
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class ModelConfig:
    base_model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    max_seq_length: int = 2048
    dtype: str = None  # auto-detect (bf16 на Blackwell)
    load_in_4bit: bool = True


@dataclass
class LoraConfig:
    r: int = 32              # увеличен с 16 — 12GB VRAM позволяет
    lora_alpha: int = 64     # alpha = 2*r для стабильности
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
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8   # увеличен с 2 — 12GB + 4bit позволяет
    gradient_accumulation_steps: int = 2   # эффективный batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03            # 3% от шагов (лучше для больших датасетов)
    lr_scheduler_type: str = "cosine"     # cosine лучше linear для длинных тренировок
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = True                     # нативная поддержка bf16 на Blackwell
    logging_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 3             # хранить только 3 последних чекпоинта
    seed: int = 3407
    dataset_path: str = "data/train.jsonl"
    dataloader_num_workers: int = 4       # параллельная загрузка данных
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
    verdict_max_new_tokens: int = 1024
    temperature: float = 0.1
    repetition_penalty: float = 1.15
