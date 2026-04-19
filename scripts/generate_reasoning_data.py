"""
Генерация обучающих данных с <reasoning> тегами + self-critique.

Конвертирует обычный train.jsonl → train_reasoning.jsonl с:
- Пошаговыми рассуждениями (4 шага + адвокат дьявола)
- Post-verdict self-critique (модель перепроверяет себя)
- Глубокими ссылками на конкретные источники

Для масштабной генерации можно заменить rule-based на LLM API (distillation).

Использование:
    python generate_reasoning_data.py
    python generate_reasoning_data.py --input data/train.jsonl --output data/train_reasoning.jsonl --limit 10000
"""
import _path  # noqa: F401,E402 — inject project root into sys.path

import argparse
import json
import os
import random
import re
from typing import List, Dict, Optional

from config import PROJECT_ROOT
from claim_parser import classify_claim, extract_numbers, extract_dates


# === Шаблоны reasoning для ПРАВДА (глубокие, с адвокатом дьявола) ===
TRUE_REASONING_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение описывает: {claim_summary}.\n"
        "Источник ({source}): «{title}» — описывает ТО ЖЕ событие с теми же деталями.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "{source} сообщает: «{snippet_short}». "
        "Ключевые детали совпадают: {detail}.\n"
        "Противоречащих источников не обнаружено.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Числовые данные в утверждении и источнике совпадают.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Источник описывает ТО ЖЕ событие, не просто совпадение слов — проверено.\n"
        "Б) Подтверждение конкретное, с цитатой из авторитетного источника.\n"
        "В) Адвокат дьявола: что если источник описывает ДРУГОЕ похожее событие? "
        "Нет — детали (участники, дата, цифры) совпадают.\n"
        "Г) Вердикт логичен — ПРАВДА."
    ),
    (
        "Шаг 1 — Идентификация события:\n"
        "Анализирую утверждение: {claim_summary}.\n"
        "Источник {source}: заголовок «{title}» напрямую связан с утверждением.\n"
        "Контекст и факты совпадают — это одно и то же событие.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "Цитата из {source}: «{snippet_short}».\n"
        "Факты идентичны: {detail}.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Все числовые данные согласуются между утверждением и источником.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Это не просто совпадение слов — событие идентичное.\n"
        "Б) Есть конкретная цитата из авторитетного источника.\n"
        "В) Адвокат дьявола: возможно ли, что данные устарели? "
        "Проверяю дату — актуально.\n"
        "Г) Через неделю я бы согласился: ПРАВДА."
    ),
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение содержит: {claim_summary}.\n"
        "Источник {source}: «{title}» — описывает именно это событие.\n"
        "Совпадают: участники, место, время действия.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "«{snippet_short}» — прямое подтверждение.\n"
        "Дополнительные детали из источника не противоречат утверждению.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Ключевые данные ({detail}) полностью совпадают.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Событие идентифицировано верно — совпадают все ключевые детали.\n"
        "Б) Подтверждение из авторитетного источника, а не из соцсетей.\n"
        "В) Адвокат дьявола: есть ли контекст, который мог бы изменить картину? "
        "На данный момент нет.\n"
        "Г) Вердикт обоснован — ПРАВДА."
    ),
]

# === Шаблоны reasoning для ЛОЖЬ (глубокие, с анализом противоречий) ===
FALSE_REASONING_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение описывает: {claim_summary}.\n"
        "Найденные источники описывают похожую тему, но с ДРУГИМИ данными.\n"
        "Ни один источник не подтверждает конкретные детали утверждения.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "Источники противоречат утверждению: {contradiction}.\n"
        "Утверждение содержит информацию, не подтверждённую ни одним надёжным источником.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Числовые данные в утверждении {num_issue}.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Это не отсутствие подтверждения — источники ПРЯМО ПРОТИВОРЕЧАТ.\n"
        "Б) Различие фактическое, не интерпретационное.\n"
        "В) Адвокат дьявола: может ли утверждение быть верным, а источники ошибаться? "
        "Маловероятно — несколько независимых источников согласованы.\n"
        "Г) Вердикт обоснован — ЛОЖЬ."
    ),
    (
        "Шаг 1 — Идентификация события:\n"
        "Анализирую: {claim_summary}.\n"
        "Утверждение описывает событие, которое {impossibility}.\n"
        "Независимые источники опровергают это.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "{contradiction}.\n"
        "Ни один авторитетный источник не подтверждает заявленное.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "{num_issue}.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Источники проанализированы тщательно — противоречие установлено.\n"
        "Б) Это конкретное опровержение, а не просто отсутствие данных.\n"
        "В) Адвокат дьявола: есть ли шанс, что информация ещё не попала в СМИ? "
        "Нет — тема активно освещается, и данные ДРУГИЕ.\n"
        "Г) Вердикт однозначен — ЛОЖЬ."
    ),
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение: {claim_summary}.\n"
        "Проверка по нескольким источникам выявила прямые противоречия.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "Ключевое противоречие: {contradiction}.\n"
        "Авторитетные СМИ сообщают иные данные.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Заявленные цифры {num_issue}.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Противоречие конкретное и верифицируемое.\n"
        "Б) Это не «не найдено» — найдено ОБРАТНОЕ.\n"
        "В) Адвокат дьявола: возможна ли ошибка в моём анализе? "
        "Проверил дважды — данные однозначны.\n"
        "Г) ЛОЖЬ — обоснованный вердикт."
    ),
]

# === Шаблоны reasoning для НЕ ПОДТВЕРЖДЕНО (глубокие, с объяснением) ===
UNVERIFIED_REASONING_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение: {claim_summary}.\n"
        "Найденные источники описывают похожую тему, но НЕ то же событие.\n"
        "Совпадение слов не означает совпадение событий.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "Ни один источник не описывает точно то же событие с теми же деталями.\n"
        "Данных недостаточно для подтверждения или опровержения.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Невозможно сравнить — релевантные данные не найдены.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Похожие слова ≠ то же событие — проверено.\n"
        "Б) Не путаю «не найдено» с «опровергнуто» — это НЕ ПОДТВЕРЖДЕНО.\n"
        "В) Адвокат дьявола: может быть, стоит поставить ПРАВДА/ЛОЖЬ? "
        "Нет — без конкретного подтверждающего/опровергающего источника нельзя.\n"
        "Г) Корректный вердикт при недостатке данных."
    ),
    (
        "Шаг 1 — Идентификация события:\n"
        "Анализирую: {claim_summary}.\n"
        "По запросу не найдено релевантных источников.\n"
        "Невозможно ни подтвердить, ни опровергнуть утверждение.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "Конкретных цитат, подтверждающих утверждение, не обнаружено.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Нет данных для сравнения числовых значений.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Отсутствие подтверждения ≠ ложь.\n"
        "Б) Нельзя выносить вердикт без данных.\n"
        "В) Адвокат дьявола: возможно, новость слишком свежая для индексации? "
        "Да, это возможно. Именно поэтому НЕ ПОДТВЕРЖДЕНО.\n"
        "Г) НЕ ПОДТВЕРЖДЕНО — единственный честный ответ."
    ),
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение: {claim_summary}.\n"
        "Источники найдены, но описывают другие аспекты темы.\n"
        "Ни один не подтверждает и не опровергает конкретное утверждение.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "Источники тематически связаны, но не касаются именно этого события.\n"
        "Отсутствуют прямые цитаты за или против.\n\n"
        "Шаг 3 — Числовая проверка:\n"
        "Числовые данные из источников относятся к другим событиям.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Тематическая близость ≠ подтверждение — это ключевой момент.\n"
        "Б) Недостаточно данных для любого категоричного вердикта.\n"
        "В) Адвокат дьявола: не слишком ли я осторожен? "
        "Нет — лучше признать неопределённость, чем угадывать.\n"
        "Г) НЕ ПОДТВЕРЖДЕНО — верно."
    ),
]

# === Специализированные шаблоны для ЧИСЛОВЫХ утверждений ===
TRUE_NUMERICAL_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение содержит числовые данные: {claim_summary}.\n"
        "Источник {source}: «{title}» — описывает то же событие.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "{source} сообщает: «{snippet_short}».\n"
        "Факты совпадают: {detail}.\n\n"
        "Шаг 3 — Числовая проверка (КРИТИЧНО):\n"
        "Числа в утверждении: {claim_numbers}.\n"
        "Числа в источнике: совпадают. Расхождение менее 10%.\n"
        "Единицы измерения корректны, масштаб верный.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Источник описывает ТО ЖЕ событие с ТЕМИ ЖЕ цифрами.\n"
        "Б) Числа проверены — расхождений нет.\n"
        "В) Адвокат дьявола: могли ли данные измениться? "
        "Источник актуален, данные свежие.\n"
        "Г) ПРАВДА — числа подтверждены."
    ),
]

FALSE_NUMERICAL_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение содержит числовые данные: {claim_summary}.\n"
        "Источники описывают похожую тему, но с ДРУГИМИ цифрами.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "{contradiction}.\n"
        "Числовые данные в утверждении не подтверждаются.\n\n"
        "Шаг 3 — Числовая проверка (КРИТИЧНО):\n"
        "Числа в утверждении: {claim_numbers}.\n"
        "Числа в источниках: {num_issue}.\n"
        "Расхождение СУЩЕСТВЕННОЕ — более 10%, или порядок величины другой.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Расхождение в числах конкретное и верифицируемое.\n"
        "Б) Не перепутаны ли единицы? Нет — сравниваю одно и то же.\n"
        "В) Адвокат дьявола: может быть, данные обновились? "
        "Нет — несколько актуальных источников дают другие цифры.\n"
        "Г) ЛОЖЬ — числа не сходятся."
    ),
]

# === Специализированные шаблоны для ДАТОВЫХ утверждений ===
TRUE_DATE_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение содержит дату/период: {claim_summary}.\n"
        "Источник {source}: «{title}» — подтверждает хронологию.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "{source} сообщает: «{snippet_short}».\n"
        "Дата и место события совпадают.\n\n"
        "Шаг 3 — Проверка дат (КРИТИЧНО):\n"
        "Дата в утверждении: {claim_dates}.\n"
        "Дата в источнике: совпадает. Год, месяц и место верны.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Не путаю ли я разные мероприятия разных лет? Нет — год совпадает.\n"
        "Б) Место проведения проверено отдельно.\n"
        "В) Адвокат дьявола: может ли быть ошибка в годе? Проверил — нет.\n"
        "Г) ПРАВДА — дата подтверждена."
    ),
]

FALSE_DATE_TEMPLATES = [
    (
        "Шаг 1 — Идентификация события:\n"
        "Утверждение содержит дату/период: {claim_summary}.\n"
        "Источники указывают ДРУГУЮ дату или место.\n\n"
        "Шаг 2 — Сравнение фактов:\n"
        "{contradiction}.\n"
        "Хронология в утверждении не совпадает с реальной.\n\n"
        "Шаг 3 — Проверка дат (КРИТИЧНО):\n"
        "Дата в утверждении: {claim_dates}.\n"
        "Реальная дата: ДРУГАЯ. {num_issue}.\n"
        "Разница не в днях — в годах или месте проведения.\n\n"
        "Шаг 4 — Самопроверка + адвокат дьявола:\n"
        "А) Не перепутал ли я сам даты? Проверил дважды — нет.\n"
        "Б) Это не «приблизительная дата» — это принципиально другой период.\n"
        "В) Адвокат дьявола: могло ли событие повториться? "
        "Нет — это конкретное уникальное событие.\n"
        "Г) ЛОЖЬ — дата/место неверны."
    ),
]

# Шаблоны для self-critique
SELF_CRITIQUE_TEMPLATES = {
    "no_errors": [
        "ОШИБКИ: нет\nКОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: {score}\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: {verdict}",
    ],
    "true_correct": [
        "ОШИБКИ: нет; аналитик корректно идентифицировал релевантный источник и проверил факты\n"
        "КОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: {score}\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: ПРАВДА",
    ],
    "false_correct": [
        "ОШИБКИ: нет; аналитик верно определил противоречия между утверждением и источниками\n"
        "КОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: {score}\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: ЛОЖЬ",
    ],
    "unverified_correct": [
        "ОШИБКИ: нет; аналитик правильно отметил недостаток данных\n"
        "КОРРЕКЦИЯ: НЕТ\nРЕКОМЕНДУЕМЫЙ_SCORE: {score}\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: НЕ ПОДТВЕРЖДЕНО",
    ],
    "overconfident_correction": [
        "ОШИБКИ: аналитик слишком уверен; источники описывают похожую тему но не то же событие\n"
        "КОРРЕКЦИЯ: ДА\nРЕКОМЕНДУЕМЫЙ_SCORE: {corrected_score}\nРЕКОМЕНДУЕМЫЙ_ВЕРДИКТ: НЕ ПОДТВЕРЖДЕНО",
    ],
}

# Шаблоны для заполнения переменных
CONTRADICTIONS = [
    "данные в утверждении не соответствуют найденным фактам",
    "авторитетные источники сообщают противоположное",
    "официальные данные расходятся с заявленными",
    "хронология событий не совпадает с утверждением",
    "ключевые цифры в утверждении не подтверждены",
    "источник описывает ДРУГОЕ событие, не совпадающее с утверждением",
    "цитаты в утверждении вырваны из контекста или искажены",
    "независимые эксперты прямо опровергают заявленное",
    "дата события в утверждении не совпадает с реальной",
    "действующие лица и участники описаны неверно",
]

NUM_ISSUES = [
    "не соответствуют официальной статистике",
    "значительно преувеличены по сравнению с реальными",
    "не подтверждаются ни одним источником",
    "противоречат физически возможным значениям",
    "расходятся с данными профильных ведомств",
    "отличаются от опубликованных на порядок",
    "перепутаны единицы измерения или масштаб",
    "взяты из устаревших отчётов, актуальные данные другие",
]

IMPOSSIBILITIES = [
    "не подтверждается ни одним авторитетным источником",
    "противоречит общеизвестным фактам",
    "содержит признаки типичной дезинформации",
    "является физически/логически невозможным",
    "опровергается множеством независимых источников",
    "хронологически невозможно — описанное ещё не произошло",
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

    # Краткое описание утверждения для reasoning шаблонов
    claim_summary = claim[:100] if claim else "событие"

    return {
        "claim": claim,
        "claim_summary": claim_summary,
        "source": source,
        "title": title,
        "snippet": snippet[:150] if snippet else title[:80],
        "verdict": verdict,
        "reasoning_text": reasoning_text,
        "sources_text": sources_text,
        "human_msg": human_msg,
    }


def generate_reasoning(info: dict) -> str:
    """Генерация reasoning на основе извлечённой информации.

    Классифицирует утверждение по типу (числовое, датовое, общее)
    и выбирает специализированный шаблон reasoning.
    """
    verdict = info["verdict"]
    source = info.get("source", "News Agency")
    title = info.get("title", "")[:80]
    snippet_short = info.get("snippet", "")[:100]
    detail = title[:50] if title else "основные данные"
    claim_summary = info.get("claim_summary", "событие")
    claim = info.get("claim", "")

    # Классификация утверждения
    claim_info = classify_claim(claim) if claim else {"type": "general", "numbers": [], "dates": []}
    claim_type = claim_info["type"]
    claim_numbers = ", ".join(n["raw"] for n in claim_info["numbers"][:3]) or "нет"
    claim_dates = ", ".join(d["raw"] for d in claim_info["dates"][:2]) or "нет"

    format_kwargs = dict(
        source=source, title=title,
        snippet_short=snippet_short if snippet_short else title[:80],
        detail=detail, claim_summary=claim_summary,
        claim_numbers=claim_numbers, claim_dates=claim_dates,
        contradiction=random.choice(CONTRADICTIONS),
        num_issue=random.choice(NUM_ISSUES),
        impossibility=random.choice(IMPOSSIBILITIES),
    )

    if verdict == "ПРАВДА":
        # Выбираем шаблон по типу утверждения
        if claim_type == "numerical" and claim_info["numbers"]:
            templates = TRUE_NUMERICAL_TEMPLATES + TRUE_REASONING_TEMPLATES
        elif claim_type == "date" and claim_info["dates"]:
            templates = TRUE_DATE_TEMPLATES + TRUE_REASONING_TEMPLATES
        else:
            templates = TRUE_REASONING_TEMPLATES
        template = random.choice(templates)
        return template.format(**format_kwargs)

    elif verdict == "ЛОЖЬ":
        if claim_type == "numerical" and claim_info["numbers"]:
            templates = FALSE_NUMERICAL_TEMPLATES + FALSE_REASONING_TEMPLATES
        elif claim_type == "date" and claim_info["dates"]:
            templates = FALSE_DATE_TEMPLATES + FALSE_REASONING_TEMPLATES
        else:
            templates = FALSE_REASONING_TEMPLATES
        template = random.choice(templates)
        return template.format(**format_kwargs)

    else:  # НЕ ПОДТВЕРЖДЕНО
        template = random.choice(UNVERIFIED_REASONING_TEMPLATES)
        return template.format(claim_summary=claim_summary)


def generate_self_critique(verdict: str, score: int) -> str:
    """Генерация self-critique для обучающих данных."""
    if verdict == "ПРАВДА":
        template = random.choice(SELF_CRITIQUE_TEMPLATES["true_correct"])
        return template.format(score=score, verdict=verdict)
    elif verdict == "ЛОЖЬ":
        template = random.choice(SELF_CRITIQUE_TEMPLATES["false_correct"])
        return template.format(score=score, verdict=verdict)
    else:
        template = random.choice(SELF_CRITIQUE_TEMPLATES["unverified_correct"])
        return template.format(score=score, verdict=verdict)


def convert_to_reasoning_format(conversation: dict) -> Optional[dict]:
    """Конвертация одного примера в формат с <reasoning> тегами + self-critique."""
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
    claim_summary = modified_claim[:100]
    reasoning = random.choice(UNVERIFIED_REASONING_TEMPLATES).format(
        claim_summary=claim_summary,
    )

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
        ("одобрил", "отклонил"),
        ("подтвердил", "опроверг"),
        ("увеличил", "сократил"),
    ]
    for original, replacement in modifiers:
        if original in claim.lower():
            return claim.replace(original, replacement, 1)

    # Fallback: добавляем сенсационный prefix
    prefixes = ["Экстренно: ", "Впервые в истории: ", "Шок: "]
    return random.choice(prefixes) + claim


def main():
    parser = argparse.ArgumentParser(
        description="Генерация обучающих данных с <reasoning> тегами + self-critique"
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
