"""Fine-tuning скрипт (Unsloth + QLoRA)."""

import argparse
import json
import os

from config import ModelConfig, LoraConfig, TrainingConfig


TEMPLATE_DATA = [
    {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Проверь достоверность утверждения.\n\n"
                    "Утверждение: Россия запустила новую космическую станцию в 2024 году.\n\n"
                    "Найденные новости:\n"
                    "1. Роскосмос объявил о планах создания новой орбитальной станции\n"
                    "   Источник: ТАСС\n"
                    "   Дата: 15 марта 2024\n"
                    "   Проект Российской орбитальной станции находится на этапе проектирования."
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "Вердикт: частично достоверно\n"
                    "Обоснование: Россия действительно разрабатывает новую орбитальную станцию "
                    "(РОС), однако по данным ТАСС проект находится на этапе проектирования, "
                    "а не запуска. Утверждение о запуске преждевременно."
                ),
            },
        ]
    },
    {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Проверь достоверность утверждения.\n\n"
                    "Утверждение: Уровень безработицы в России снизился до 2.9% в 2024 году.\n\n"
                    "Найденные новости:\n"
                    "1. Росстат: безработица в России обновила исторический минимум\n"
                    "   Источник: РБК\n"
                    "   Дата: 20 февраля 2024\n"
                    "   Уровень безработицы в январе 2024 года составил 2.9%."
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "Вердикт: достоверно\n"
                    "Обоснование: По данным Росстата, опубликованным РБК, уровень безработицы "
                    "в России действительно составил 2.9% в январе 2024 года, что является "
                    "историческим минимумом. Утверждение подтверждается официальной статистикой."
                ),
            },
        ]
    },
    {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Проверь достоверность утверждения.\n\n"
                    "Утверждение: Apple полностью прекратила продажи в России.\n\n"
                    "Найденные новости:\n"
                    "1. Apple приостановила продажи продукции в России\n"
                    "   Источник: Коммерсантъ\n"
                    "   Дата: 2 марта 2022\n"
                    "   Apple приостановила продажи через официальный магазин, "
                    "но продукция доступна через реселлеров."
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "Вердикт: недостоверно\n"
                    "Обоснование: Apple приостановила официальные продажи в России в 2022 году, "
                    "но не прекратила их полностью. Продукция Apple по-прежнему доступна "
                    "через авторизованных реселлеров и параллельный импорт."
                ),
            },
        ]
    },
]


def generate_template(output_path: str):
    """Генерация шаблона обучающих данных."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in TEMPLATE_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Шаблон обучающих данных сохранён в {output_path}")
    print(f"Количество примеров: {len(TEMPLATE_DATA)}")


def load_dataset_from_jsonl(path: str):
    """Загрузка датасета из JSONL файла."""
    from datasets import Dataset

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
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

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # 1. Загрузка модели
    print("Загрузка модели...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.base_model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
    )

    # 2. Настройка chat template
    tokenizer = get_chat_template(tokenizer, chat_template="mistral")

    # 3. Добавление LoRA адаптеров
    print("Добавление LoRA адаптеров...")
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

    # 4. Загрузка данных
    print(f"Загрузка данных из {training_config.dataset_path}...")
    dataset = load_dataset_from_jsonl(training_config.dataset_path)

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True)

    # 5. Настройка тренера
    print("Настройка тренера...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_steps,
            lr_scheduler_type=training_config.lr_scheduler_type,
            optim=training_config.optim,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            logging_steps=training_config.logging_steps,
            save_strategy=training_config.save_strategy,
            seed=training_config.seed,
        ),
    )

    # 6. Loss только на ответах модели
    trainer = train_on_responses_only(
        trainer,
        instruction_part="[INST]",
        response_part="[/INST]",
    )

    # 7. Запуск обучения
    print("Запуск обучения...")
    stats = trainer.train()
    print(f"Обучение завершено. Результаты: {stats}")

    # 8. Сохранение адаптеров
    save_path = os.path.join(training_config.output_dir, "fact_checker_lora")
    print(f"Сохранение LoRA адаптеров в {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Готово!")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Mistral для проверки фактов")
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Сгенерировать шаблон обучающих данных",
    )
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
    args = parser.parse_args()

    training_config = TrainingConfig()

    if args.generate_template:
        generate_template(training_config.dataset_path)
        return

    if args.dataset:
        training_config.dataset_path = args.dataset
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.epochs:
        training_config.num_train_epochs = args.epochs
    if args.lr:
        training_config.learning_rate = args.lr

    if not os.path.exists(training_config.dataset_path):
        print(f"Файл данных не найден: {training_config.dataset_path}")
        print("Используйте --generate-template для создания шаблона.")
        return

    train(training_config=training_config)


if __name__ == "__main__":
    main()
