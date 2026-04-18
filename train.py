"""
Fine-tuning скрипт (Unsloth + QLoRA).

Оптимизировано под RTX 5070 (12GB GDDR7, Blackwell, bf16, TF32).
"""

import argparse
import json
import os
import sys
import torch

# Fix для Windows spawn-воркеров: добавляем unsloth_compiled_cache в PYTHONPATH,
# чтобы subprocess мог импортировать UnslothSFTTrainer (иначе dill.loads падает).
_COMPILED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsloth_compiled_cache")
if os.path.isdir(_COMPILED_CACHE):
    if _COMPILED_CACHE not in sys.path:
        sys.path.insert(0, _COMPILED_CACHE)
    _existing_pp = os.environ.get("PYTHONPATH", "")
    if _COMPILED_CACHE not in _existing_pp:
        os.environ["PYTHONPATH"] = _COMPILED_CACHE + (os.pathsep + _existing_pp if _existing_pp else "")

from config import ModelConfig, LoraConfig, TrainingConfig


def setup_cuda_optimizations():
    """Включение CUDA-оптимизаций для Blackwell/Ampere+."""
    if not torch.cuda.is_available():
        print("CUDA недоступна!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    # TF32 — ускоряет matmul на Ampere+ (включая Blackwell)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # cuDNN benchmark — автоподбор оптимальных алгоритмов свёртки
    torch.backends.cudnn.benchmark = True

    # Оптимизация аллокации CUDA-памяти
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    # Принудительно SDPA (не xformers) — Blackwell sm_120
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    os.environ["XFORMERS_DISABLED"] = "1"

    # Очистка VRAM перед стартом
    torch.cuda.empty_cache()

    print("CUDA-оптимизации включены: TF32, cuDNN benchmark, SDPA flash, expandable segments")


def load_dataset_from_jsonl(path: str):
    """Загрузка датасета из JSONL файла."""
    from datasets import Dataset

    print(f"Чтение {path}...")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                data.append(json.loads(line))
            if (i + 1) % 100_000 == 0:
                print(f"  Загружено: {i + 1:,} строк")

    print(f"  Всего: {len(data):,} примеров")
    return Dataset.from_list(data)


def train(
    model_config: ModelConfig = None,
    lora_config: LoraConfig = None,
    training_config: TrainingConfig = None,
    resume_from: str = None,
):
    """Запуск fine-tuning через Unsloth + QLoRA.

    Args:
        resume_from: Путь к существующим LoRA адаптерам для дообучения.
                     Если указан — загружает адаптеры и продолжает обучение
                     с пониженным LR (без catastrophic forgetting).
    """
    if model_config is None:
        model_config = ModelConfig()
    if lora_config is None:
        lora_config = LoraConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # При дообучении — автоматически понижаем LR чтобы не стереть выученное
    if resume_from:
        original_lr = training_config.learning_rate
        training_config.learning_rate = min(training_config.learning_rate, 1e-4)
        if original_lr != training_config.learning_rate:
            print(f"Дообучение: LR снижен {original_lr} → {training_config.learning_rate}")
        else:
            print(f"Дообучение: LR = {training_config.learning_rate} (уже ниже порога)")

    # CUDA-оптимизации
    setup_cuda_optimizations()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig

    # 1. Загрузка модели
    if resume_from and os.path.exists(resume_from):
        print(f"\n[1/7] Загрузка fine-tuned модели из {resume_from} (дообучение)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=resume_from,
            max_seq_length=model_config.max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=model_config.load_in_4bit,
        )
    else:
        print("\n[1/7] Загрузка base модели...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config.base_model_name,
            max_seq_length=model_config.max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=model_config.load_in_4bit,
        )

    # 2. Настройка chat template
    print("[2/7] Настройка chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")

    # 3. Добавление LoRA адаптеров (при дообучении адаптеры уже загружены)
    if resume_from and os.path.exists(resume_from):
        print(f"[3/7] LoRA адаптеры загружены из {resume_from}")
    else:
        print(f"[3/7] Добавление LoRA адаптеров (r={lora_config.r}, alpha={lora_config.lora_alpha})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
            random_state=lora_config.random_state,
        )

    # Вывод количества обучаемых параметров
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Обучаемых параметров: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 4. Загрузка данных
    print(f"\n[4/7] Загрузка данных из {training_config.dataset_path}...")
    dataset = load_dataset_from_jsonl(training_config.dataset_path)

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            # Конвертация ShareGPT (from/value) -> HuggingFace (role/content)
            messages = []
            for msg in convo:
                role = msg.get("role") or msg.get("from", "")
                content = msg.get("content") or msg.get("value", "")
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                messages.append({"role": role, "content": content})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, num_proc=1)

    # 5. Настройка тренера
    print(f"\n[5/7] Настройка тренера...")
    print(f"  Packing: True (упаковка коротких примеров)")
    print(f"  Batch size: {training_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  LR: {training_config.learning_rate}")
    print(f"  BF16: {training_config.bf16}")
    print(f"  TF32: {training_config.tf32}")

    # Validation split (10% для мониторинга переобучения)
    split = dataset.train_test_split(test_size=0.1, seed=training_config.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            eval_accumulation_steps=training_config.eval_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            lr_scheduler_type=training_config.lr_scheduler_type,
            optim=training_config.optim,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            tf32=training_config.tf32,
            logging_steps=training_config.logging_steps,
            save_strategy=training_config.save_strategy,
            save_steps=training_config.save_steps,
            save_total_limit=training_config.save_total_limit,
            seed=training_config.seed,
            dataloader_num_workers=training_config.dataloader_num_workers,
            dataloader_pin_memory=training_config.dataloader_pin_memory,
            report_to="none",
            # Validation (eval_strategy должна совпадать с save_strategy для load_best_model_at_end)
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Packing параметры (в новом TRL — через SFTConfig)
            packing=True,
            dataset_text_field="text",
            max_seq_length=model_config.max_seq_length,
            dataset_num_proc=1,
        ),
    )

    # 6. train_on_responses_only — маскируем промпт, обучаем ТОЛЬКО на ответах
    print("[6/7] Маскировка промптов (train_on_responses_only)...")
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="[INST]",
        response_part="[/INST]",
    )

    # Callback: очистка CUDA-кэша перед eval (предотвращает OOM от фрагментации)
    from transformers import TrainerCallback, EarlyStoppingCallback

    class ClearCacheCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    trainer.add_callback(ClearCacheCallback())

    # Early stopping: остановка если eval_loss не улучшается 1 эпоху подряд
    # Предотвращает бесполезные эпохи при переобучении
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1))

    # 7. Запуск обучения
    print("\n[7/7] Запуск обучения...")
    print("=" * 60)
    stats = trainer.train()
    print("=" * 60)
    print(f"Обучение завершено!")
    print(f"  Train loss: {stats.training_loss:.4f}")
    print(f"  Время: {stats.metrics.get('train_runtime', 0):.0f} сек.")
    print(f"  Samples/sec: {stats.metrics.get('train_samples_per_second', 0):.1f}")

    # 8. Сохранение адаптеров
    save_path = os.path.join(training_config.output_dir, "fact_checker_lora")
    print(f"\nСохранение LoRA адаптеров в {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Готово!")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Mistral для проверки фактов (RTX 5070)")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Путь к файлу обучающих данных (JSONL)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Директория для сохранения адаптеров",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Количество эпох обучения",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size per device (по умолчанию: 2)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Путь к существующим адаптерам для дообучения (например: adapters/fact_checker_lora)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Обучение с нуля (игнорировать существующие адаптеры)",
    )
    args = parser.parse_args()

    training_config = TrainingConfig()

    if args.dataset:
        training_config.dataset_path = args.dataset
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.epochs:
        training_config.num_train_epochs = args.epochs
    if args.lr:
        training_config.learning_rate = args.lr
    if args.batch_size:
        training_config.per_device_train_batch_size = args.batch_size

    if not os.path.exists(training_config.dataset_path):
        print(f"Файл данных не найден: {training_config.dataset_path}")
        print("Сначала скачайте датасет: python download_dataset.py")
        return

    # Авто-обнаружение существующих адаптеров для дообучения
    resume_from = args.resume
    if args.no_resume:
        resume_from = None
    elif resume_from is None:
        default_adapter_path = os.path.join(training_config.output_dir, "fact_checker_lora")
        if os.path.exists(default_adapter_path):
            print(f"Обнаружены существующие адаптеры: {default_adapter_path}")
            print("Продолжаю обучение (--resume auto). Используйте --no-resume для обучения с нуля.")
            resume_from = default_adapter_path

    train(training_config=training_config, resume_from=resume_from)


if __name__ == "__main__":
    main()
