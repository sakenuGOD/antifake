"""
Генерация обучающих данных с <reasoning> тегами.

Конвертирует обычный train.jsonl → train_reasoning.jsonl с пошаговыми рассуждениями.
Для масштабной генерации можно заменить rule-based на Claude/GPT-4 API (distillation).

Использование:
    python generate_reasoning_data.py
    python generate_reasoning_data.py --input data/train.jsonl --output data/train_reasoning.jsonl --limit 10000
"""

import argparse
import json
import os
import random
import re
from typing import List, Dict, Optional

from config import PROJECT_ROOT


# === Шаблоны reasoning для ПРАВДА ===
TRUE_REASONING_TEMPLATES = [
    (
        "Шаг 1 — Анализ источников:\n"
        "- Источник ({source}): «{title}». Описывает то же событие, что и утверждение.\n"
        "- Ключевые детали совпадают: {detail}.\n\n"
        "Шаг 2 — Доказательства:\n"
        "{source} сообщает: «{snippet_short}». "
        "Это прямо подтверждает утверждение.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Числовые данные в утверждении и источнике совпадают.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Источник описывает ТО ЖЕ событие — проверено.\n"
        "Б) Подтверждение получено из конкретного источника.\n"
        "В) Вердикт логичен — ПРАВДА."
    ),
    (
        "Шаг 1 — Анализ источников:\n"
        "- {source}: Заголовок «{title}» напрямую связан с утверждением.\n"
        "- Контекст и факты совпадают.\n\n"
        "Шаг 2 — Доказательства:\n"
        "Цитата из {source}: «{snippet_short}».\n"
        "Факты в утверждении и источнике идентичны.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Все числовые данные ({detail}) согласуются.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Это не просто совпадение слов — событие то же.\n"
        "Б) Есть конкретная цитата из авторитетного источника.\n"
        "В) Через неделю я бы согласился: ПРАВДА."
    ),
    (
        "Шаг 1 — Анализ источников:\n"
        "- {source} подтверждает: «{title}».\n"
        "- Описывает именно то же событие с теми же деталями.\n\n"
        "Шаг 2 — Доказательства:\n"
        "«{snippet_short}» — прямое подтверждение утверждения.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Ключевые данные ({detail}) совпадают с источником.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Событие идентично, не просто похожие слова.\n"
        "Б) Подтверждение конкретное, с цитатой.\n"
        "В) Вердикт обоснован — ПРАВДА."
    ),
]

# === Шаблоны reasoning для ЛОЖЬ ===
FALSE_REASONING_TEMPLATES = [
    (
        "Шаг 1 — Анализ источников:\n"
        "- Найденные источники описывают похожую тему, но с ДРУГИМИ данными.\n"
        "- Ни один источник не подтверждает конкретные детали утверждения.\n\n"
        "Шаг 2 — Доказательства:\n"
        "Источники противоречат утверждению: {contradiction}.\n"
        "Утверждение содержит информацию, не подтверждённую ни одним надёжным источником.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Числовые данные в утверждении {num_issue}.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Это не отсутствие подтверждения — источники ПРЯМО ПРОТИВОРЕЧАТ.\n"
        "Б) Различие фактическое, не интерпретационное.\n"
        "В) Вердикт обоснован — ЛОЖЬ."
    ),
    (
        "Шаг 1 — Анализ источников:\n"
        "- Утверждение описывает событие, которое {impossibility}.\n"
        "- Независимые источники опровергают это.\n\n"
        "Шаг 2 — Доказательства:\n"
        "{contradiction}.\n"
        "Ни один авторитетный источник не подтверждает заявленное.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "{num_issue}.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Источники проанализированы — противоречие установлено.\n"
        "Б) Это опровержение, а не просто отсутствие данных.\n"
        "В) Вердикт однозначен — ЛОЖЬ."
    ),
    (
        "Шаг 1 — Анализ источников:\n"
        "- Проверка по нескольким источникам не подтвердила утверждение.\n"
        "- Найдены прямые опровержения.\n\n"
        "Шаг 2 — Доказательства:\n"
        "Ключевое противоречие: {contradiction}.\n"
        "Авторитетные СМИ сообщают иное.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Заявленные цифры {num_issue}.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Противоречие конкретное и верифицируемое.\n"
        "Б) Это не «не найдено» — найдено ОБРАТНОЕ.\n"
        "В) ЛОЖЬ — обоснованный вердикт."
    ),
]

# === Шаблоны reasoning для НЕ ПОДТВЕРЖДЕНО ===
UNVERIFIED_REASONING_TEMPLATES = [
    (
        "Шаг 1 — Анализ источников:\n"
        "- Найденные источники описывают похожую тему, но НЕ то же событие.\n"
        "- Совпадение слов не означает совпадение событий.\n\n"
        "Шаг 2 — Доказательства:\n"
        "Ни один источник не описывает точно то же событие с теми же деталями.\n"
        "Данных недостаточно для подтверждения или опровержения.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Невозможно сравнить — релевантные данные не найдены.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Похожие слова ≠ то же событие — проверено.\n"
        "Б) Не путаю «не найдено» с «опровергнуто» — это НЕ ПОДТВЕРЖДЕНО.\n"
        "В) Корректный вердикт при недостатке данных."
    ),
    (
        "Шаг 1 — Анализ источников:\n"
        "- По запросу не найдено релевантных источников.\n"
        "- Невозможно ни подтвердить, ни опровергнуть утверждение.\n\n"
        "Шаг 2 — Доказательства:\n"
        "Конкретных цитат, подтверждающих утверждение, не обнаружено.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Нет данных для сравнения.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Отсутствие подтверждения ≠ ложь.\n"
        "Б) Нельзя выносить вердикт без данных.\n"
        "В) НЕ ПОДТВЕРЖДЕНО — единственный честный ответ."
    ),
    (
        "Шаг 1 — Анализ источников:\n"
        "- Источники найдены, но описывают другие аспекты темы.\n"
        "- Ни один не подтверждает и не опровергает конкретное утверждение.\n\n"
        "Шаг 2 — Доказательства:\n"
        "Источники тематически связаны, но не касаются именно этого события.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Числовые данные из источников относятся к другим событиям.\n\n"
        "Шаг 4 — Самопроверка:\n"
        "А) Тематическая близость ≠ подтверждение.\n"
        "Б) Недостаточно данных для любого категоричного вердикта.\n"
        "В) НЕ ПОДТВЕРЖДЕНО — верно."
    ),
]

# Шаблоны для заполнения переменных
CONTRADICTIONS = [
    "данные в утверждении не соответствуют найденным фактам",
    "авторитетные источники сообщают противоположное",
    "официальные данные расходятся с заявленными",
    "хронология событий не совпадает с утверждением",
    "ключевые цифры в утверждении не подтверждены",
]

NUM_ISSUES = [
    "не соответствуют официальной статистике",
    "значительно преувеличены по сравнению с реальными",
    "не подтверждаются ни одним источником",
    "противоречат физически возможным значениям",
    "расходятся с данными профильных ведомств",
]

IMPOSSIBILITIES = [
    "не подтверждается ни одним авторитетным источником",
    "противоречит общеизвестным фактам",
    "содержит признаки типичной дезинформации",
    "является физически/логически невозможным",
]


def extract_info_from_conversation(conversation: dict) -> dict:
    """Извлечение информации из существующего примера."""
    convos = conversation.get("conversations", [])
    if len(convos) < 2:
        return {}

    human_msg = convos[0].get("value", "")
    gpt_msg = convos[1].get("value", "")

    # Извлекаем утверждение
    claim = ""
    claim_match = re.search(r"Утверждение:\s*(.+?)(?:\n|$)", human_msg)
    if claim_match:
        claim = claim_match.group(1).strip()

    # Извлекаем источник
    source = "News Agency"
    source_match = re.search(r"Источник:\s*(.+?)(?:\n|$)", human_msg)
    if source_match:
        source = source_match.group(1).strip()

    # Извлекаем заголовок новости
    title = claim[:80]
    title_match = re.search(r"Найденные новости:\n\d+\.\s*(.+?)(?:\n|$)", human_msg)
    if title_match:
        title = title_match.group(1).strip()

    # Извлекаем сниппет
    snippet = ""
    lines = human_msg.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("Источник:") and i + 1 < len(lines):
            snippet = lines[i + 1].strip()[:150]
            break

    # Вердикт
    verdict = "НЕ ПОДТВЕРЖДЕНО"
    if "ПРАВДА" in gpt_msg:
        verdict = "ПРАВДА"
    elif "ЛОЖЬ" in gpt_msg:
        verdict = "ЛОЖЬ"

    # Обоснование
    reasoning_text = ""
    reasoning_match = re.search(r"ОБОСНОВАНИЕ:\s*(.+?)(?:\nИСТОЧНИКИ:|$)", gpt_msg, re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()

    # Источники
    sources_text = ""
    sources_match = re.search(r"ИСТОЧНИКИ:\s*(.+)", gpt_msg, re.DOTALL)
    if sources_match:
        sources_text = sources_match.group(1).strip()

    return {
        "claim": claim,
        "source": source,
        "title": title,
        "snippet": snippet[:150] if snippet else title[:80],
        "verdict": verdict,
        "reasoning_text": reasoning_text,
        "sources_text": sources_text,
        "human_msg": human_msg,
    }


def generate_reasoning(info: dict) -> str:
    """Генерация reasoning на основе извлечённой информации."""
    verdict = info["verdict"]
    source = info.get("source", "News Agency")
    title = info.get("title", "")[:80]
    snippet_short = info.get("snippet", "")[:100]
    detail = title[:50] if title else "основные данные"

    if verdict == "ПРАВДА":
        template = random.choice(TRUE_REASONING_TEMPLATES)
        return template.format(
            source=source,
            title=title,
            snippet_short=snippet_short,
            detail=detail,
        )

    elif verdict == "ЛОЖЬ":
        template = random.choice(FALSE_REASONING_TEMPLATES)
        return template.format(
            contradiction=random.choice(CONTRADICTIONS),
            num_issue=random.choice(NUM_ISSUES),
            impossibility=random.choice(IMPOSSIBILITIES),
        )

    else:  # НЕ ПОДТВЕРЖДЕНО
        return random.choice(UNVERIFIED_REASONING_TEMPLATES)


def convert_to_reasoning_format(conversation: dict) -> Optional[dict]:
    """Конвертация одного примера в формат с <reasoning> тегами."""
    info = extract_info_from_conversation(conversation)
    if not info or not info.get("claim"):
        return None

    reasoning = generate_reasoning(info)
    verdict = info["verdict"]

    if verdict == "ПРАВДА":
        score = random.randint(75, 98)
        confidence = random.randint(80, 99)
    elif verdict == "ЛОЖЬ":
        score = random.randint(2, 25)
        confidence = random.randint(80, 99)
    else:
        score = random.randint(35, 65)
        confidence = random.randint(45, 75)

    reasoning_text = info.get("reasoning_text", "Анализ на основе найденных источников.")
    sources_text = info.get("sources_text", "Не найдены")

    gpt_response = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n"
        f"ДОСТОВЕРНОСТЬ: {score}\n"
        f"ВЕРДИКТ: {verdict}\n"
        f"УВЕРЕННОСТЬ: {confidence}\n"
        f"ОБОСНОВАНИЕ: {reasoning_text}\n"
        f"ИСТОЧНИКИ: {sources_text}\n"
        f"</answer>"
    )

    return {
        "conversations": [
            {"from": "human", "value": info["human_msg"]},
            {"from": "gpt", "value": gpt_response},
        ]
    }


def create_reasoning_dataset(
    input_path: str = None,
    output_path: str = None,
    limit: int = 10_000,
    add_unverified_ratio: float = 0.35,
):
    """Конвертация обычного датасета в формат с reasoning.

    Args:
        input_path: Путь к исходному JSONL
        output_path: Путь для сохранения
        limit: Максимум примеров
        add_unverified_ratio: Доля НЕ ПОДТВЕРЖДЕНО (генерируются из ПРАВДА путём модификации)
    """
    if input_path is None:
        input_path = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "data", "train_reasoning.jsonl")

    print(f"Чтение {input_path}...")
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Загружено: {len(examples)} примеров")
    random.shuffle(examples)

    # Ограничиваем выборку
    base_limit = int(limit * (1 - add_unverified_ratio))
    examples = examples[:base_limit]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    written = 0
    unverified_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Конвертируем оригинальный пример (ПРАВДА или ЛОЖЬ)
            converted = convert_to_reasoning_format(ex)
            if converted:
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                written += 1

            # Генерируем НЕ ПОДТВЕРЖДЕНО из каждого N-го ПРАВДА примера
            if (unverified_written < limit * add_unverified_ratio and
                    random.random() < add_unverified_ratio):
                unverified = _make_unverified_from(ex)
                if unverified:
                    f.write(json.dumps(unverified, ensure_ascii=False) + "\n")
                    unverified_written += 1

            if (written + unverified_written) % 1000 == 0:
                print(f"  Записано: {written + unverified_written} "
                      f"(base={written}, unverified={unverified_written})")

            if written + unverified_written >= limit:
                break

    total = written + unverified_written
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nСохранено {total} примеров в {output_path}")
    print(f"  ПРАВДА/ЛОЖЬ: {written}")
    print(f"  НЕ ПОДТВЕРЖДЕНО: {unverified_written}")
    print(f"  Размер: {file_size:.1f} МБ")


def _make_unverified_from(conversation: dict) -> Optional[dict]:
    """Создаёт НЕ ПОДТВЕРЖДЕНО из существующего примера.

    Берём утверждение и «портим» его — меняем числа/детали,
    чтобы источники описывали ДРУГОЕ событие.
    """
    info = extract_info_from_conversation(conversation)
    if not info or not info.get("claim"):
        return None

    original_claim = info["claim"]
    modified_claim = _modify_claim(original_claim)

    if modified_claim == original_claim:
        return None

    # Строим human_msg с изменённым утверждением, но ОРИГИНАЛЬНЫМИ источниками
    human_msg = info["human_msg"].replace(
        f"Утверждение: {original_claim}",
        f"Утверждение: {modified_claim}",
    )

    score = random.randint(35, 65)
    confidence = random.randint(45, 75)
    reasoning = random.choice(UNVERIFIED_REASONING_TEMPLATES)

    gpt_response = (
        f"<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n"
        f"ДОСТОВЕРНОСТЬ: {score}\n"
        f"ВЕРДИКТ: НЕ ПОДТВЕРЖДЕНО\n"
        f"УВЕРЕННОСТЬ: {confidence}\n"
        f"ОБОСНОВАНИЕ: Найденные источники описывают похожее, но не идентичное событие. "
        f"Данных недостаточно для подтверждения или опровержения.\n"
        f"ИСТОЧНИКИ: Подтверждающие источники не найдены\n"
        f"</answer>"
    )

    return {
        "conversations": [
            {"from": "human", "value": human_msg},
            {"from": "gpt", "value": gpt_response},
        ]
    }


def _modify_claim(claim: str) -> str:
    """Модификация утверждения для создания 'похожего но другого'."""

    # Замена чисел (21% → 47%, 100 → 350)
    def multiply_number(match):
        num = int(match.group())
        if num == 0:
            return str(random.randint(5, 50))
        return str(int(num * random.uniform(1.5, 4.0)))

    modified = re.sub(r'\d+', multiply_number, claim, count=1)

    if modified != claim:
        return modified

    # Добавляем усиление если числа не нашлись
    modifiers = [
        ("запустила", "экстренно запустила"),
        ("подписал", "отменил"),
        ("повысил", "снизил"),
        ("разрешил", "запретил"),
        ("открыл", "закрыл"),
    ]
    for original, replacement in modifiers:
        if original in claim.lower():
            return claim.replace(original, replacement, 1)

    # Fallback: добавляем сенсационный prefix
    prefixes = ["Экстренно: ", "Впервые в истории: ", "Шок: "]
    return random.choice(prefixes) + claim


def main():
    parser = argparse.ArgumentParser(
        description="Генерация обучающих данных с <reasoning> тегами"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Путь к исходному JSONL (по умолчанию: data/train.jsonl)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь для сохранения (по умолчанию: data/train_reasoning.jsonl)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10_000,
        help="Количество примеров (по умолчанию: 10000)",
    )
    args = parser.parse_args()

    create_reasoning_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
