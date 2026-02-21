"""
Скрипт загрузки датасета фейковых новостей с Hugging Face
и конвертации в формат JSONL для обучения Mistral.

Источники:
  - Arko007/ultimate-fake-news-dataset (9.25M примеров, binary + 6-class)
  - GonzaloA/fake_news (40k, title + text + label)

Формат выхода: conversations [{from: human, value: ...}, {from: gpt, value: ...}]
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
    last_dot = truncated.rfind(".")
    if last_dot > max_chars // 2:
        return truncated[:last_dot + 1]
    return truncated + "..."


# Шаблоны обоснований для разнообразия ответов модели
REAL_REASONS = [
    (
        "Информация подтверждается данными из найденных источников. "
        "Содержание статьи соответствует фактам, изложенным в независимых "
        "новостных агентствах. Стилистика текста нейтральна, отсутствуют "
        "маркеры манипулятивного воздействия."
    ),
    (
        "Фактологический анализ подтверждает достоверность утверждения. "
        "Данные согласуются с информацией из независимых источников. "
        "Текст написан в нейтральном стиле без признаков манипуляции."
    ),
    (
        "Утверждение соответствует данным, представленным в найденных "
        "источниках. Семантический анализ не выявил противоречий. "
        "Стилистика публикации соответствует стандартам качественной журналистики."
    ),
    (
        "Проверка по трём направлениям подтверждает достоверность. "
        "Факты совпадают с данными из независимых источников, логических "
        "противоречий не обнаружено, текст стилистически нейтрален."
    ),
    (
        "Содержание статьи подтверждается множественными источниками. "
        "Отсутствуют признаки фабрикации данных. Изложение фактов "
        "последовательно и не содержит манипулятивных приёмов."
    ),
]

FAKE_REASONS = [
    (
        "Информация не подтверждается проверенными источниками. "
        "В тексте обнаружены признаки манипулятивного воздействия: "
        "эмоционально окрашенная лексика и кликбейт-заголовки. "
        "Фактологический анализ выявил несоответствия с данными из "
        "независимых источников."
    ),
    (
        "Фактологическая проверка выявила несоответствия с данными "
        "авторитетных источников. Текст содержит признаки манипуляции: "
        "гиперболизация, апелляция к эмоциям, отсутствие конкретных ссылок. "
        "Независимые источники опровергают основные тезисы."
    ),
    (
        "Анализ выявил множественные маркеры недостоверной информации. "
        "Утверждения противоречат данным проверенных источников. "
        "Стилистика текста указывает на манипулятивный характер публикации."
    ),
    (
        "Проверка по трём направлениям выявила признаки фейка. "
        "Факты не подтверждаются независимыми источниками, обнаружены "
        "логические противоречия, стиль текста содержит эмоциональные "
        "манипуляции и кликбейт."
    ),
    (
        "Семантический анализ выявил внутренние противоречия в тексте. "
        "Факты не находят подтверждения в авторитетных источниках. "
        "Обнаружены типичные маркеры дезинформации: сенсационные "
        "заголовки и бездоказательные утверждения."
    ),
]


def make_conversation(title: str, text: str, label: int, source: str = "News Agency") -> dict:
    """
    Конвертация одного примера в формат conversations для обучения.

    label: 0 = real (достоверно), 1 = fake (недостоверно)
    """
    snippet = truncate_text(text.strip(), max_chars=500)

    news_block = (
        f"1. {title}\n"
        f"   Источник: {source}\n"
        f"   {snippet}"
    )

    human_msg = (
        f"Проверь достоверность утверждения.\n\n"
        f"Утверждение: {title}\n\n"
        f"Найденные новости:\n{news_block}"
    )

    if label == 0:
        verdict = (
            f"ДОСТОВЕРНОСТЬ: {random.randint(75, 98)}\n"
            f"ВЕРДИКТ: ПРАВДА\n"
            f"УВЕРЕННОСТЬ: {random.randint(80, 99)}\n"
            f"ОБОСНОВАНИЕ: {random.choice(REAL_REASONS)}\n"
            f"ИСТОЧНИКИ: {source}"
        )
    else:
        verdict = (
            f"ДОСТОВЕРНОСТЬ: {random.randint(2, 25)}\n"
            f"ВЕРДИКТ: ЛОЖЬ\n"
            f"УВЕРЕННОСТЬ: {random.randint(80, 99)}\n"
            f"ОБОСНОВАНИЕ: {random.choice(FAKE_REASONS)}\n"
            f"ИСТОЧНИКИ: Подтверждающие источники не найдены"
        )

    return {
        "conversations": [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": verdict},
        ]
    }


def extract_title(text: str, max_len: int = 120) -> str:
    """Извлечение заголовка из текста, если title отсутствует."""
    text = text.strip()
    # Берём первое предложение
    for sep in [".\n", ". ", "\n"]:
        idx = text.find(sep)
        if 10 < idx < max_len:
            return text[:idx + 1].strip()
    # Fallback: первые max_len символов
    if len(text) > max_len:
        return text[:max_len].strip() + "..."
    return text


def download_and_convert(
    output_path: str = "data/train.jsonl",
    limit: int = 500_000,
    seed: int = 42,
):
    """Загрузка датасетов с HF и конвертация в JSONL."""
    random.seed(seed)
    examples = []

    # === Источник 1: Arko007/ultimate-fake-news-dataset (основной) ===
    print("=" * 60)
    print("Загрузка Arko007/ultimate-fake-news-dataset...")
    print("(это большой датасет, загрузка может занять несколько минут)")
    print("=" * 60)

    ds_ultimate = load_dataset(
        "Arko007/ultimate-fake-news-dataset",
        split="train",
        streaming=True,
    )

    # Собираем примеры из стриминга
    real_examples = []
    fake_examples = []
    target_per_class = limit // 2
    count = 0

    for row in ds_ultimate:
        text = (row.get("text") or "").strip()
        label = row.get("label_binary")
        source_name = row.get("source") or "News Agency"

        if not text or label is None or len(text) < 50:
            continue

        title = extract_title(text)

        if label == 1 and len(real_examples) < target_per_class:
            real_examples.append((title, text, 0, source_name))  # 1=REAL -> 0 в нашей системе
        elif label == 0 and len(fake_examples) < target_per_class:
            fake_examples.append((title, text, 1, source_name))  # 0=FAKE -> 1 в нашей системе

        count += 1
        if count % 100_000 == 0:
            print(f"  Обработано: {count:,} | real: {len(real_examples):,} | fake: {len(fake_examples):,}")

        if len(real_examples) >= target_per_class and len(fake_examples) >= target_per_class:
            break

    print(f"Из ultimate: real={len(real_examples):,}, fake={len(fake_examples):,}")

    # === Источник 2: GonzaloA/fake_news (дополнительный) ===
    need_more_real = target_per_class - len(real_examples)
    need_more_fake = target_per_class - len(fake_examples)

    if need_more_real > 0 or need_more_fake > 0:
        print(f"\nДозагрузка из GonzaloA/fake_news (нужно ещё: real={need_more_real}, fake={need_more_fake})...")
        ds_gonzalo = load_dataset("GonzaloA/fake_news", split="train")

        for row in ds_gonzalo:
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            label = row.get("label")

            if not title or not text or label is None:
                continue

            if label == 0 and need_more_real > 0:
                real_examples.append((title, text, 0, "News Agency"))
                need_more_real -= 1
            elif label == 1 and need_more_fake > 0:
                fake_examples.append((title, text, 1, "News Agency"))
                need_more_fake -= 1

            if need_more_real <= 0 and need_more_fake <= 0:
                break

        print(f"Итого: real={len(real_examples):,}, fake={len(fake_examples):,}")

    # Объединяем и перемешиваем
    examples = real_examples + fake_examples
    random.shuffle(examples)

    total = len(examples)
    print(f"\nВсего примеров: {total:,}")

    # Конвертируем и записываем
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for title, text, label, source in examples:
            conversation = make_conversation(title, text, label, source)
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            written += 1

            if written % 50_000 == 0:
                print(f"  Записано: {written:,}/{total:,}")

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nСохранено {written:,} примеров в {output_path}")
    print(f"Размер файла: {file_size:.1f} МБ")


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка датасета фейковых новостей с Hugging Face (до 500к примеров)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/train.jsonl",
        help="Путь для сохранения JSONL (по умолчанию: data/train.jsonl)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=500_000,
        help="Количество примеров (сбалансировано по классам, по умолчанию: 500000)",
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
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
