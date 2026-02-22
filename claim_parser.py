"""Утилиты для парсинга и классификации утверждений.

Извлекает числа, даты, проценты, денежные суммы и определяет тип утверждения.
Используется в pipeline, training data generation и evaluation.
"""

import re
from typing import List, Dict, Any, Optional


# === ТИПЫ УТВЕРЖДЕНИЙ ===
CLAIM_TYPE_NUMERICAL = "numerical"       # числа, проценты, суммы
CLAIM_TYPE_DATE = "date"                 # даты, сроки, "когда"
CLAIM_TYPE_PERSON = "person"             # персоны, назначения, действия людей
CLAIM_TYPE_EVENT = "event"               # события, запуски, аварии
CLAIM_TYPE_INSTITUTIONAL = "institutional"  # организации, решения органов
CLAIM_TYPE_GENERAL = "general"           # общие факты


def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Извлечение всех числовых значений из текста с контекстом.

    Возвращает список: [{"value": 25.5, "raw": "25.5%", "type": "percent", "context": "ставку до 25.5%"}, ...]
    """
    results = []

    # Проценты: 25%, 2.3%, 25,5%
    for m in re.finditer(r'(\d+[.,]?\d*)\s*%', text):
        val = float(m.group(1).replace(',', '.'))
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 20)
        results.append({
            "value": val, "raw": m.group(0), "type": "percent",
            "context": text[start:end].strip(),
        })

    # Денежные суммы: $100, 100 долларов, 5 млрд рублей, 1 триллион
    money_pattern = (
        r'(\d+[.,]?\d*)\s*'
        r'(долларов|доллар|руб(?:лей|ля|ль)?|евро|\$|€|₽|'
        r'(?:трлн|трилл?ион|млрд|миллиард|млн|миллион|тыс(?:яч)?)\.?\s*'
        r'(?:долларов|доллар|руб(?:лей|ля|ль)?|евро|\$|€|₽)?)'
    )
    for m in re.finditer(money_pattern, text, re.IGNORECASE):
        raw_num = m.group(1).replace(',', '.')
        val = float(raw_num)
        # Обработка множителей
        unit = m.group(2).lower()
        if any(x in unit for x in ['трлн', 'трилл']):
            val *= 1e12
        elif any(x in unit for x in ['млрд', 'миллиард']):
            val *= 1e9
        elif any(x in unit for x in ['млн', 'миллион']):
            val *= 1e6
        elif any(x in unit for x in ['тыс']):
            val *= 1e3
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        results.append({
            "value": val, "raw": m.group(0), "type": "money",
            "context": text[start:end].strip(),
        })

    # Просто числа с множителями (без валюты): "8 миллиардов человек", "146 миллионов"
    count_pattern = (
        r'(\d+[.,]?\d*)\s*'
        r'(трлн|трилл?ион|млрд|миллиард|млн|миллион|тыс(?:яч)?)'
        r'(?:ов|а|ы|ами)?\s*'
        r'(?!долларов|доллар|руб|евро|\$|€|₽)'
    )
    for m in re.finditer(count_pattern, text, re.IGNORECASE):
        raw_num = m.group(1).replace(',', '.')
        val = float(raw_num)
        unit = m.group(2).lower()
        if any(x in unit for x in ['трлн', 'трилл']):
            val *= 1e12
        elif any(x in unit for x in ['млрд', 'миллиард']):
            val *= 1e9
        elif any(x in unit for x in ['млн', 'миллион']):
            val *= 1e6
        elif any(x in unit for x in ['тыс']):
            val *= 1e3
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        # Не дублировать если уже поймано как money
        if not any(r["raw"] in m.group(0) or m.group(0) in r["raw"] for r in results):
            results.append({
                "value": val, "raw": m.group(0), "type": "count",
                "context": text[start:end].strip(),
            })

    # Простые числа (баллы, ранг): "магнитудой 15 баллов", "до 50"
    for m in re.finditer(r'(?<!\d)(\d+[.,]?\d*)(?!\d|%|\s*(?:трлн|млрд|млн|тыс|миллион|миллиард))', text):
        val = float(m.group(1).replace(',', '.'))
        # Пропускаем если уже поймано
        if any(m.start() >= r.get("_start", -1) and m.end() <= r.get("_end", -1)
               for r in results):
            continue
        start = max(0, m.start() - 25)
        end = min(len(text), m.end() + 25)
        ctx = text[start:end].strip()
        # Пропускаем годы — они обрабатываются в extract_dates
        if re.search(r'(?:19|20)\d{2}', m.group(1)) and len(m.group(1)) == 4:
            continue
        if not any(m.group(0) in r["raw"] or r["raw"] in m.group(0) for r in results):
            results.append({
                "value": val, "raw": m.group(0), "type": "number",
                "context": ctx,
            })

    return results


def extract_dates(text: str) -> List[Dict[str, Any]]:
    """Извлечение дат из текста.

    Поддерживает: "февраль 2025", "1 января 2025", "2025 году", "январе 2025"
    """
    results = []

    months_map = {
        'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4,
        'мая': 5, 'май': 5, 'июн': 6, 'июл': 7, 'август': 8,
        'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
    }

    # "1 января 2025", "15 февраля 2024 года"
    for m in re.finditer(
        r'(\d{1,2})\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[яй]\w*|'
        r'июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)'
        r'\s+(\d{4})',
        text, re.IGNORECASE
    ):
        day = int(m.group(1))
        month_str = m.group(2).lower()[:5]
        year = int(m.group(3))
        month = next((v for k, v in months_map.items() if month_str.startswith(k)), None)
        if month:
            results.append({
                "day": day, "month": month, "year": year,
                "raw": m.group(0), "type": "full_date",
            })

    # "февраль 2025", "январе 2025 года"
    for m in re.finditer(
        r'(январ\w*|феврал\w*|март\w*|апрел\w*|ма[яй]\w*|'
        r'июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)'
        r'\s+(\d{4})',
        text, re.IGNORECASE
    ):
        month_str = m.group(1).lower()[:5]
        year = int(m.group(2))
        month = next((v for k, v in months_map.items() if month_str.startswith(k)), None)
        if month and not any(m.group(0) in r["raw"] for r in results):
            results.append({
                "month": month, "year": year,
                "raw": m.group(0), "type": "month_year",
            })

    # "в 2025 году", "2024 года"
    for m in re.finditer(r'(?:в\s+)?(\d{4})\s*(?:год[уае]?|г\.)', text):
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            if not any(str(year) in r["raw"] for r in results):
                results.append({
                    "year": year, "raw": m.group(0), "type": "year",
                })

    return results


def classify_claim(claim: str) -> Dict[str, Any]:
    """Классификация утверждения по типу и извлечение ключевых элементов.

    Returns:
        {
            "type": "numerical" | "date" | "person" | "event" | "institutional" | "general",
            "numbers": [...],
            "dates": [...],
            "has_numbers": True/False,
            "has_dates": True/False,
            "verification_hints": ["Проверить: ..."],
        }
    """
    numbers = extract_numbers(claim)
    dates = extract_dates(claim)

    has_numbers = len(numbers) > 0
    has_dates = len(dates) > 0

    # Определение типа
    claim_lower = claim.lower()

    # Персоны
    person_indicators = [
        'президент', 'министр', 'глава', 'директор',
        'путин', 'маск', 'трамп', 'байден', 'навальный',
        'назначил', 'уволил', 'встретился', 'заявил',
        'подписал', 'объявил', 'выступил', 'заявление',
    ]
    is_person = any(kw in claim_lower for kw in person_indicators)

    # Институциональные
    inst_indicators = [
        'цб рф', 'центральный банк', 'правительство', 'госдума',
        'мид', 'мвд', 'фсб', 'ООН', 'воз', 'nasa', 'роскосмос',
        'евросоюз', 'нато', 'вто', 'опек',
        'закон', 'указ', 'постановление', 'решение', 'запрет',
    ]
    is_institutional = any(kw in claim_lower for kw in inst_indicators)

    # Событийные
    event_indicators = [
        'запуск', 'запустила', 'авария', 'катастрофа', 'взрыв',
        'землетрясение', 'наводнение', 'пожар', 'крушение',
        'олимпийские', 'чемпионат', 'выборы', 'саммит',
        'произошло', 'случилось', 'началось', 'завершилось',
    ]
    is_event = any(kw in claim_lower for kw in event_indicators)

    # Определяем основной тип
    if has_numbers and (any(n["type"] == "percent" for n in numbers) or
                        any(n["type"] == "money" for n in numbers)):
        claim_type = CLAIM_TYPE_NUMERICAL
    elif has_dates and not has_numbers:
        claim_type = CLAIM_TYPE_DATE
    elif has_numbers and has_dates:
        claim_type = CLAIM_TYPE_NUMERICAL  # числа + даты = числовая проверка
    elif is_person:
        claim_type = CLAIM_TYPE_PERSON
    elif is_institutional:
        claim_type = CLAIM_TYPE_INSTITUTIONAL
    elif is_event:
        claim_type = CLAIM_TYPE_EVENT
    else:
        claim_type = CLAIM_TYPE_GENERAL

    # Генерация подсказок для верификации
    hints = []
    for n in numbers:
        if n["type"] == "percent":
            hints.append(f"Проверить процент: {n['raw']} — совпадает ли с источниками?")
        elif n["type"] == "money":
            hints.append(f"Проверить сумму: {n['raw']} — подтверждается ли источниками?")
        elif n["type"] == "count":
            hints.append(f"Проверить количество: {n['raw']} — совпадает ли с данными?")
        else:
            hints.append(f"Проверить число: {n['raw']} — корректно ли?")

    for d in dates:
        if d["type"] == "full_date":
            hints.append(f"Проверить дату: {d['raw']} — совпадает ли с реальной?")
        elif d["type"] == "month_year":
            hints.append(f"Проверить период: {d['raw']} — верный ли месяц/год?")
        elif d["type"] == "year":
            hints.append(f"Проверить год: {d['raw']} — актуальны ли данные?")

    return {
        "type": claim_type,
        "numbers": numbers,
        "dates": dates,
        "has_numbers": has_numbers,
        "has_dates": has_dates,
        "verification_hints": hints,
    }


def compare_numbers(claim_numbers: List[Dict], source_numbers: List[Dict],
                    tolerance: float = 0.10) -> List[Dict[str, Any]]:
    """Сравнение числовых значений между утверждением и источником.

    Args:
        tolerance: допустимое отклонение (0.10 = 10%)

    Returns:
        [{"claim": ..., "source": ..., "match": True/False, "deviation": 0.05}, ...]
    """
    comparisons = []

    for cn in claim_numbers:
        best_match = None
        best_deviation = float('inf')

        for sn in source_numbers:
            # Сравниваем только одинаковые типы
            if cn["type"] != sn["type"]:
                continue

            if cn["value"] == 0 and sn["value"] == 0:
                deviation = 0.0
            elif cn["value"] == 0 or sn["value"] == 0:
                deviation = 1.0
            else:
                deviation = abs(cn["value"] - sn["value"]) / max(abs(cn["value"]), abs(sn["value"]))

            if deviation < best_deviation:
                best_deviation = deviation
                best_match = sn

        if best_match is not None:
            comparisons.append({
                "claim_number": cn,
                "source_number": best_match,
                "match": best_deviation <= tolerance,
                "deviation": round(best_deviation, 4),
            })
        else:
            comparisons.append({
                "claim_number": cn,
                "source_number": None,
                "match": False,
                "deviation": None,
            })

    return comparisons


def format_verification_hints(claim_info: Dict[str, Any]) -> str:
    """Форматирование подсказок для верификации в текст для промпта."""
    if not claim_info["verification_hints"]:
        return ""

    hints_text = "КЛЮЧЕВЫЕ ТОЧКИ ПРОВЕРКИ:\n"
    for hint in claim_info["verification_hints"]:
        hints_text += f"• {hint}\n"

    return hints_text
