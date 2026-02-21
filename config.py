"""Конфигурация: пути, API-ключи, гиперпараметры."""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class ModelConfig:
    base_model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    max_seq_length: int = 2048
    dtype: str = None  # auto-detect
    load_in_4bit: bool = True


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 16
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
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 1
    save_strategy: str = "epoch"
    seed: int = 3407
    dataset_path: str = "data/train.jsonl"


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
