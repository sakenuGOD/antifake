"""
Скрипт загрузки датасета фейковых новостей с Hugging Face
и конвертации в формат JSONL для обучения Mistral.

Источник: GonzaloA/fake_news (24 400 train / 8 120 val / 8 120 test)
Формат: title + text + label (0 = real, 1 = fake)
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Обрезка текста до max_chars символов по последнему предложению."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Обрезаем по последней точке, чтобы не рвать предложение
    last_dot = truncated.rfind(".")
    if last_dot > max_chars // 2:
        return truncated[:last_dot + 1]
    return truncated + "..."


def make_conversation(title: str, text: str, label: int) -> dict:
    """
    Конвертация одного примера в формат conversations для обучения.

    label: 0 = real (достоверно), 1 = fake (недостоверно)
    """
    snippet = truncate_text(text.strip(), max_chars=500)

    # Формируем блок "найденных новостей" из текста статьи
    news_block = (
        f"1. {title}\n"
        f"   Источник: News Agency\n"
        f"   {snippet}"
    )

    human_msg = (
        f"Проверь достоверность утверждения.\n\n"
        f"Утверждение: {title}\n\n"
        f"Найденные новости:\n{news_block}"
    )

    if label == 0:
        # Реальная новость
        verdict = (
            f"ДОСТОВЕРНОСТЬ: {random.randint(75, 98)}\n"
            f"ВЕРДИКТ: ПРАВДА\n"
            f"УВЕРЕННОСТЬ: {random.randint(80, 99)}\n"
            f"ОБОСНОВАНИЕ: Информация подтверждается данными из найденных "
            f"источников. Содержание статьи соответствует фактам, изложенным "
            f"в независимых новостных агентствах. Стилистика текста нейтральна, "
            f"отсутствуют маркеры манипулятивного воздействия.\n"
            f"ИСТОЧНИКИ: News Agency"
        )
    else:
        # Фейковая новость
        verdict = (
            f"ДОСТОВЕРНОСТЬ: {random.randint(2, 25)}\n"
            f"ВЕРДИКТ: ЛОЖЬ\n"
            f"УВЕРЕННОСТЬ: {random.randint(80, 99)}\n"
            f"ОБОСНОВАНИЕ: Информация не подтверждается проверенными источниками. "
            f"В тексте обнаружены признаки манипулятивного воздействия: "
            f"эмоционально окрашенная лексика и кликбейт-заголовки. "
            f"Фактологический анализ выявил несоответствия с данными из "
            f"независимых источников.\n"
            f"ИСТОЧНИКИ: Подтверждающие источники не найдены"
        )

    return {
        "conversations": [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": verdict},
        ]
    }


def download_and_convert(
    output_path: str = "data/train.jsonl",
    split: str = "train",
    limit: int = None,
    seed: int = 42,
):
    """Загрузка датасета с HF и конвертация в JSONL."""
    print(f"Загрузка датасета GonzaloA/fake_news (split={split})...")
    dataset = load_dataset("GonzaloA/fake_news", split=split)
    print(f"Загружено примеров: {len(dataset)}")

    # Фильтруем пустые примеры
    examples = []
    for row in dataset:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        label = row.get("label")
        if title and text and label is not None:
            examples.append((title, text, label))

    print(f"Валидных примеров: {len(examples)}")

    # Перемешиваем
    random.seed(seed)
    random.shuffle(examples)

    # Ограничиваем количество если указано
    if limit:
        # Берём сбалансированную выборку
        real = [e for e in examples if e[2] == 0]
        fake = [e for e in examples if e[2] == 1]
        half = limit // 2
        real = real[:half]
        fake = fake[:half]
        examples = real + fake
        random.shuffle(examples)
        print(f"Сбалансированная выборка: {len(real)} real + {len(fake)} fake = {len(examples)}")

    # Конвертируем
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for title, text, label in examples:
            conversation = make_conversation(title, text, label)
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            count += 1

    print(f"\nСохранено {count} примеров в {output_path}")
    print(f"Размер файла: {os.path.getsize(output_path) / 1024 / 1024:.1f} МБ")


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка датасета фейковых новостей с Hugging Face"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/train.jsonl",
        help="Путь для сохранения JSONL (по умолчанию: data/train.jsonl)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Сплит датасета (по умолчанию: train)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Максимальное количество примеров (сбалансировано по классам). "
             "Без ограничения: все ~24 400 примеров",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости",
    )
    args = parser.parse_args()

    download_and_convert(
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
