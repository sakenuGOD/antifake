"""
Fine-tuning скрипт (Unsloth + QLoRA).

Оптимизировано под RTX 5070 (12GB GDDR7, Blackwell, bf16, TF32).
"""

import argparse
import json
import os
import torch

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
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("CUDA-оптимизации включены: TF32, cuDNN benchmark, expandable segments")


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
):
    """Запуск fine-tuning через Unsloth + QLoRA."""
    if model_config is None:
        model_config = ModelConfig()
    if lora_config is None:
        lora_config = LoraConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # CUDA-оптимизации
    setup_cuda_optimizations()

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # 1. Загрузка модели (4-bit NF4 квантизация)
    print("\n[1/7] Загрузка модели...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.base_model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
    )

    # 2. Настройка chat template
    print("[2/7] Настройка chat template...")
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")

    # 3. Добавление LoRA адаптеров (r=32, alpha=64)
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

    dataset = dataset.map(formatting_func, batched=True, num_proc=4)

    # 5. Настройка тренера
    print(f"\n[5/7] Настройка тренера...")
    print(f"  Batch size: {training_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  LR: {training_config.learning_rate}")
    print(f"  BF16: {training_config.bf16}")
    print(f"  TF32: {training_config.tf32}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
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
        ),
    )

    # 6. Loss только на ответах модели
    print("[6/7] Настройка train_on_responses_only...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="[INST]",
        response_part="[/INST]",
    )

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
        help="Batch size per device (по умолчанию: 8)",
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

    train(training_config=training_config)


if __name__ == "__main__":
    main()
