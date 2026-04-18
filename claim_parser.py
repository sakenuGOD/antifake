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


# === СКАМ-ПАТТЕРНЫ ===
SCAM_PATTERNS = [
    'безопасный счёт', 'безопасного счёта', 'безопасный счет',
    'служба безопасности банка', 'служба безопасности',
    'сотрудники банка звонят', 'банк звонит', 'звонок из банка',
    'перевести деньги', 'перевод денег на',
    'ваша карта заблокирована', 'карта заблокирована',
    'код из смс', 'назовите код', 'продиктуйте код',
    'cvv', 'пин-код', 'пин код',
    'следователь звонит', 'фсб звонит', 'полиция звонит', 'мвд звонит',
    'ваши деньги в опасности', 'деньги под угрозой',
    'гарантированная доходность', 'гарантированная прибыль',
    '100% доходность', 'тысячи процентов прибыли',
    'быстрый заработок без вложений',
    'финансовая пирамида', 'инвестиционная пирамида',
    # --- Платёжные ловушки ---
    'страховой взнос', 'страхового взноса',
    'проверочный взнос', 'проверочного взноса',
    'верификационный взнос', 'верификационного взноса',
    'регистрационный взнос', 'регистрационного взноса',
    'перевести взнос', 'перевести проверочный взнос',
    'комиссия за перевод', 'комиссию за перевод',
    'оплатить верификацию', 'оплатить проверку',
    'подтвердить личность переводом',
    'закрытая выплата', 'закрытую выплату',
    'предоплата за получение', 'оплата за разблокировку',
    'взнос для получения', 'взнос за получение',
    'взнос в размере',
    # --- Telegram/мессенджеры ---
    'телеграм-бот', 'telegram-бот', 'телеграм бот', 'telegram бот',
    'написать в телеграм', 'написать в telegram',
    'бот для выплат', 'бот для получения',
    'перейти в бот', 'перейдите в бот',
    'связаться через бот', 'получить через бот',
    # --- Скрытые активы/наследство ---
    'скрытые активы', 'скрытых активов',
    'тайное наследство', 'тайного наследства',
    'невостребованные средства', 'невостребованных средств',
    'нераспределённые активы', 'нераспределенные активы',
    'секретный счёт', 'секретный счет', 'секретного счёта',
    'неизвестное наследство', 'скрытое состояние',
    # --- Давление и срочность ---
    'только сегодня', 'срочная выплата', 'срочную выплату',
    'до конца дня', 'последний шанс',
    'не упустите шанс', 'успейте получить',
    'количество мест ограничено',
    # --- Лже-авторитеты ---
    'нотариально заверен', 'нотариально заверено',
    # --- Верификация/КОД ---
    'расшифровать код', 'ввести код для получения',
    'подтвердить данные', 'подтвердите данные',
    # --- Крипто-скам ---
    'привязка кошелька', 'привязки кошелька', 'привязать кошелёк',
    'перевести eth', 'перевести btc', 'отправить eth', 'отправить btc',
    'цифровые облигации', 'цифровых облигаций',
    'digital bonds',
    # --- Фальшивые платформы ---
    'зарегистрируйтесь на платформе', 'регистрация на платформе',
    # V18: Government impersonation & link/data scams
    'перейти по ссылке', 'перейдите по ссылке', 'перейдёт по ссылке',
    'ввести данные карты', 'введите данные карты', 'данные карты',
    'до конца месяца', 'до конца недели',
    'зарегистрироваться на сайте', 'зарегистрируйтесь на сайте',
    'зарегистрируется на сайте',
    'на специальном сайте',
    'программа выплат', 'программе выплат',
]

# Compiled regex-паттерны для сложных скам-формул
SCAM_REGEX_PATTERNS = [
    re.compile(r'подтвердить\s+личность\s+.*(?:перевод|оплат|взнос)', re.IGNORECASE),
    re.compile(r'(?:скрыт|тайн|секретн)\w*\s+(?:актив|наследств|счёт|счет|средств|состояни)', re.IGNORECASE),
    re.compile(r'(?:выплат|получ)\w*\s+(?:через|в)\s+(?:телеграм|telegram|бот)', re.IGNORECASE),
    re.compile(r'(?:страхов|комисси|взнос)\w*\s+(?:для|за|перед)\s+(?:получени|выплат|перевод)', re.IGNORECASE),
    re.compile(r'осталось\s+.{0,15}\s*мест', re.IGNORECASE),
    re.compile(r'по\s+решению\s+суда\s*.*выплат', re.IGNORECASE),
    re.compile(r'согласно\s+закону\s*.*получить', re.IGNORECASE),
]

# Regex-паттерны для scam-звонков знаменитостей
SCAM_CALL_PATTERNS = [
    r'позвони\w+.{0,40}(?:перевед|перевести|переслат|переша)',
    r'написа\w+.{0,40}(?:безопасный счёт|срочно перевед)',
    r'(?:знаменитость|певец|певиц|актёр|актрис|спортсмен).{0,60}попросил\w*\s+перевести',
    r'(?:позвонил|написал|обратил\w*).{0,40}(?:на безопасный|на специальный)\s+счёт',
    r'(?:выиграл|получил|унаследовал).{0,40}(?:миллион|тысяч|долларов|рублей).{0,20}(?:перевед|оплат)',
    r'попросил\w*\s+перевести',
]


def detect_scam_patterns(text: str) -> Dict[str, Any]:
    """Детекция мошеннических паттернов в утверждении.

    Returns:
        {"is_scam": True/False, "patterns_found": [...], "hint": "..."}
    """
    text_lower = text.lower()
    found = [p for p in SCAM_PATTERNS if p in text_lower]

    if found:
        return {
            "is_scam": True,
            "patterns_found": found,
            "hint": (
                f"ВНИМАНИЕ: утверждение содержит паттерны мошенничества/дезинформации: "
                f"{', '.join(found[:3])}. "
                f"Если описывает мошенническую схему как легитимную — вердикт ЛОЖЬ (0-15)."
            ),
        }

    # Compiled regex-паттерны для сложных скам-формул
    for pattern in SCAM_REGEX_PATTERNS:
        m = pattern.search(text_lower)
        if m:
            return {
                "is_scam": True,
                "patterns_found": [m.group(0)],
                "hint": (
                    f"ОБНАРУЖЕН МОШЕННИЧЕСКИЙ ПАТТЕРН (regex): «{m.group(0)[:60]}». "
                    f"Вердикт ЛОЖЬ (0-15)."
                ),
            }

    # Regex-паттерны для scam-звонков знаменитостей
    for pattern in SCAM_CALL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return {
                "is_scam": True,
                "patterns_found": [pattern],
                "hint": "ОБНАРУЖЕН МОШЕННИЧЕСКИЙ ПАТТЕРН: звонок знаменитости с просьбой перевести деньги. Вердикт ЛОЖЬ (0-15).",
            }

    # Fallback: стем-концепт детектор (ловит морфологические варианты)
    concept_result = detect_scam_concepts(text)
    if concept_result["is_scam"]:
        return {
            "is_scam": True,
            "patterns_found": concept_result["groups_matched"],
            "hint": concept_result["hint"],
        }
    return {"is_scam": False, "patterns_found": [], "hint": ""}


# === СТЕМ-КОНЦЕПТ ДЕТЕКТОР ===

def _normalize_russian(text: str) -> str:
    """ё→е, lowercase."""
    return text.lower().replace('ё', 'е')


def _extract_stems(text: str) -> set:
    """5-char псевдо-стемы из русских и латинских слов (мин. 4 символа)."""
    words = re.findall(r'[а-яеa-z]{4,}', _normalize_russian(text))
    return {w[:5] for w in words}


SCAM_CONCEPT_STEMS = {
    "PAYMENT": {
        "взнос",  # взнос/а/у/ы
        "перев",  # перевод/ить/ести/едите
        "оплат",  # оплата/ить/ив
        "комис",  # комиссия/ию/ии
        "предо",  # предоплата
        "отпра",  # отправить/ьте (ETH/BTC)
    },
    "DECEPTIVE_FRAMING": {
        "безоп",  # безопасный/ые/ого/ом
        "вериф",  # верификация/онный
        "секре",  # секретный/ого
        "скрыт",  # скрытые/ых
    },
    "CHANNEL": {
        "телег",  # телеграм
        "платф",  # платформа/е
        "кошел",  # кошелёк/ька
        "привя",  # привязка/ать
        "регис",  # регистрация/руйтесь
        "зарег",  # V18: зарегистрироваться/уется (prefix за-)
        "ссылк",  # V18: по ссылке
    },
    "LURE": {
        "выпла",  # выплата/ы/у
        "получ",  # получить/ение
        "облиг",  # облигации/ий
        "актив",  # активы/ов
        "насле",  # наследство/а
        "доход",  # доходность/ы
        "прибы",  # прибыль/и
        "гаран",  # гарантированная
    },
    "URGENCY": {
        "срочн",  # срочно/ая
        "огран",  # ограничено/ный
        "успей",  # успейте
        "конца",  # V18: до конца месяца/недели/дня
    },
    "AUTHORITY_ABUSE": {
        "сотру",  # сотрудники (банка)
        "служб",  # служба безопасности
        "обяза",  # обязали
    },
}

# 4-символьные стемы для коротких корней
SCAM_SHORT_STEMS = {
    "DECEPTIVE_FRAMING": {"тайн", "счет"},  # тайный/ое/ого; счёт/а (после ё→е)
    "AUTHORITY_ABUSE": {"звон"},             # звонит/ят/ок
}

# Точные латинские термины (без стемминга)
SCAM_EXACT_TERMS = {
    "CHANNEL": {"telegram", "whatsapp"},
    "PAYMENT": {"eth", "btc", "usdt"},
    "LURE": {"digital bonds", "nft"},
}

# Мета-контекст: статья О мошенничестве, а не сам скам
SCAM_META_RE = re.compile(
    r'(?:предупрежда|предостерега|разоблач|мошенни|лохотрон|'
    r'не\s+(?:попадитесь|ведитесь|верьте)|осторожн|'
    r'как\s+распознать|признаки\s+(?:мошен|обман))',
    re.IGNORECASE
)


def detect_scam_concepts(text: str) -> Dict[str, Any]:
    """Стем-концепт детектор. Морфо-независимый.
    ≥2 группы → is_scam=True.
    """
    normalized = _normalize_russian(text)
    stems = _extract_stems(text)

    if SCAM_META_RE.search(normalized):
        return {"is_scam": False, "groups_matched": [], "n_groups": 0, "hint": ""}

    matched_groups = []
    for group, group_stems in SCAM_CONCEPT_STEMS.items():
        hits = stems & group_stems
        # Короткие стемы
        for short in SCAM_SHORT_STEMS.get(group, set()):
            if any(s.startswith(short) for s in stems):
                hits.add(short)
        # Точные латинские
        for term in SCAM_EXACT_TERMS.get(group, set()):
            if term in normalized:
                hits.add(term)
        if hits:
            matched_groups.append(group)

    n = len(matched_groups)
    return {
        "is_scam": n >= 2,
        "groups_matched": matched_groups,
        "n_groups": n,
        "hint": f"СКАМ-КОНЦЕПЦИИ ({n} групп): {', '.join(matched_groups)}" if n >= 2 else "",
    }


def detect_temporal_mismatch(claim: str, sources: List[Dict]) -> Dict[str, Any]:
    """Проверяет: если claim подаётся как 'сегодня/вчера', а источники старые — это фейк."""
    from datetime import datetime

    FRESH_MARKERS = ["сегодня", "вчера", "только что", "срочно", "breaking",
                     "прямо сейчас", "буквально час назад"]
    claim_is_fresh = any(m in claim.lower() for m in FRESH_MARKERS)

    if not claim_is_fresh:
        return {"temporal_mismatch": False}

    # Извлекаем даты из источников
    source_dates = []
    for s in sources:
        date_str = s.get("date", "")
        if date_str:
            try:
                # Попытка разобрать разные форматы дат
                for fmt in ("%Y-%m-%d", "%b %d, %Y", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = datetime.strptime(date_str.strip()[:19], fmt)
                        source_dates.append(dt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

    if not source_dates:
        return {"temporal_mismatch": False}

    # Если медиана дат > 6 месяцев назад — mismatch
    median_date = sorted(source_dates)[len(source_dates) // 2]
    if (datetime.now() - median_date).days > 180:
        return {
            "temporal_mismatch": True,
            "claim_freshness": "свежее событие",
            "median_source_date": median_date.isoformat(),
            "hint": f"ВНИМАНИЕ: утверждение подано как свежее, но источники от {median_date.year}!"
        }
    return {"temporal_mismatch": False}


def extract_locations(text: str) -> List[str]:
    """Извлечение географических объектов из текста (эвристически).
    
    Ищет слова с заглавной буквы после предлогов места.
    """
    # (?<![а-яёА-ЯЁ]) гарантирует что предлог — отдельное слово, не суффикс
    location_pattern = r'(?<![а-яёА-ЯЁ])(?:в|на|из|к)\s+([А-ЯЁ][а-яёА-ЯЁ]+(?:\s+[А-ЯЁ][а-яёА-ЯЁ]+)?)'
    locations = re.findall(location_pattern, text)
    
    non_locations = {
        'этом', 'том', 'связи', 'целом', 'основном', 'ходе', 'рамках',
        'результате', 'случае', 'частности', 'числе', 'которой',
        'которого', 'которых', 'этой', 'этого', 'нашей', 'нашего',
        'Российской', 'Советского', 'Советской',
    }
    return [loc for loc in locations if loc.lower() not in non_locations and len(loc) > 3]




def extract_numbers(text: str) -> List[Dict[str, Any]]:
    """Извлечение всех числовых значений из текста с контекстом.

    Возвращает список: [{"value": 25.5, "raw": "25.5%", "type": "percent", "context": "ставку до 25.5%"}, ...]
    """
    results = []

    # Температуры: +50°C, -20°C, 343 °C, +50 градусов, −63°C
    for m in re.finditer(r'([+\-−]?\s*\d+[.,]?\d*)\s*(?:°\s*[CСcс]|градус\w*(?:\s+(?:Цельси|цельси|по\s+Цельси))?)', text):
        raw_val = m.group(1).replace(',', '.').replace('−', '-').replace(' ', '')
        val = float(raw_val)
        start = max(0, m.start() - 25)
        end = min(len(text), m.end() + 20)
        results.append({
            "value": val, "raw": m.group(0), "type": "temperature",
            "context": text[start:end].strip(),
        })

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
        # V17: Tag year-range numbers explicitly as "year" type
        if re.search(r'(?:19|20)\d{2}', m.group(1)) and len(m.group(1)) == 4:
            val_year = float(m.group(1))
            if 1800 <= val_year <= 2100:
                results.append({
                    "value": val_year, "raw": m.group(0), "type": "year",
                    "context": ctx,
                })
            continue
        if not any(m.group(0) in r["raw"] or r["raw"] in m.group(0) for r in results):
            results.append({
                "value": val, "raw": m.group(0), "type": "number",
                "context": ctx,
            })

    # V19.1: Annotate one-sided bounds. A number preceded by "более/больше/свыше/
    # не менее" is a lower bound — source values >= claim are matches, and
    # source values > claim are NOT mismatches. "Не более/меньше" is an upper
    # bound (symmetric). Annotation is consumed by compare_numbers() downstream.
    _LOWER_BOUND_RE = re.compile(
        r'\b(более|больше|свыше|от|не\s+менее|не\s+меньше|минимум|как\s+минимум)\s*$',
        re.IGNORECASE,
    )
    _UPPER_BOUND_RE = re.compile(
        r'\b(менее|меньше|до|не\s+более|не\s+больше|максимум|как\s+максимум)\s*$',
        re.IGNORECASE,
    )
    for r in results:
        raw = r.get("raw", "")
        pos = text.find(raw)
        if pos < 0:
            continue
        prefix = text[max(0, pos - 30):pos].strip()
        if _LOWER_BOUND_RE.search(prefix):
            r["bound"] = "lower"
        elif _UPPER_BOUND_RE.search(prefix):
            r["bound"] = "upper"

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

    # Детекция скам-паттернов
    scam_info = detect_scam_patterns(claim)
    if scam_info["is_scam"]:
        hints.insert(0, scam_info["hint"])

    # Детекция локаций (для проверки места события)
    locations = extract_locations(claim)
    for loc in locations[:2]:
        hints.append(f"Проверить место/локацию: «{loc}» — подтверждается ли источниками?")

    # Синонимы для искусства — «написал картину» = «создал» = «автор»
    claim_lower = claim.lower()
    if any(kw in claim_lower for kw in ["картин", "написал", "создал", "автор"]):
        hints.append(
            'Примечание: "написал картину" = "создал картину" = "является автором картины" '
            '— это синонимы в контексте изобразительного искусства.'
        )

    return {
        "type": claim_type,
        "numbers": numbers,
        "dates": dates,
        "has_numbers": has_numbers,
        "has_dates": has_dates,
        "verification_hints": hints,
        "is_scam": scam_info["is_scam"],
        "scam_patterns": scam_info.get("patterns_found", []),
        "locations": locations,
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
            # V19.1: honour one-sided bounds annotated on claim numbers.
            # "Более 97%" is satisfied by any source value >= 97; exceeding the
            # claim is NOT a contradiction (whereas it would be for an equality
            # claim "Ровно 97%"). Same logic symmetric for upper bounds.
            bound = cn.get("bound")
            if bound == "lower":
                match = best_match["value"] >= cn["value"] * (1 - tolerance)
            elif bound == "upper":
                match = best_match["value"] <= cn["value"] * (1 + tolerance)
            else:
                match = best_deviation <= tolerance
            comparisons.append({
                "claim_number": cn,
                "source_number": best_match,
                "match": match,
                "deviation": round(best_deviation, 4),
                "bound": bound,
            })
        else:
            # V17: No comparable data of same type — skip, not a mismatch
            continue

    return comparisons


# A3: Role-attribution half-truth detection
_ROLE_VERBS = re.compile(
    r'(?:основал|создал|изобрёл|изобрел|открыл|founded|invented|created|discovered)',
    re.IGNORECASE,
)
_COUNTER_PATTERNS = re.compile(
    r'(?:сооснователь|соучредитель|со-основатель|co-?founder|'
    r'ранний\s+инвестор|early\s+investor|'
    r'не\s+является\s+основателем|не\s+основывал|'
    r'присоединился|joined|'
    r'chairman|CEO\s+(?:but|however)|(?:not|isn.t)\s+(?:the\s+)?founder)',
    re.IGNORECASE,
)


# A4: Extract technical designations (Model-Number patterns)
_DESIGNATION_RE = re.compile(
    r'(?:[А-Яа-яA-Za-z]+[-\s]?\d+[А-Яа-яA-Za-z]*)',
)


def extract_designations(text: str) -> List[str]:
    """Extract technical designations like Восток-2, Apollo-13, MiG-29, iPhone-16."""
    return [m.group(0).strip() for m in _DESIGNATION_RE.finditer(text) if len(m.group(0).strip()) >= 3]


def detect_counter_evidence(claim: str, sources: List[Dict]) -> Dict[str, Any]:
    """Detect role-attribution half-truths (e.g., 'Musk founded Tesla').

    Returns:
        {"half_truth": True/False, "signal": str}
    """
    if not _ROLE_VERBS.search(claim):
        return {"half_truth": False, "signal": ""}

    source_text = " ".join(
        (s.get("snippet", "") + " " + s.get("title", ""))
        for s in sources[:7]
    )
    if _COUNTER_PATTERNS.search(source_text):
        return {
            "half_truth": True,
            "signal": "role-attribution mismatch detected in sources",
        }
    return {"half_truth": False, "signal": ""}


_FOUNDER_RE = re.compile(
    r'([А-ЯЁA-Z][а-яёa-z]+(?:\s+[А-ЯЁA-Z][а-яёa-z]+)?)\s+'
    r'(?:основал|создал|изобрёл|изобрел|открыл|founded|invented|created)\s+'
    r'(.+)',
    re.IGNORECASE,
)


def detect_person_entity_mismatch(claim: str, sources: List[Dict]) -> Dict[str, Any]:
    """Detect person-entity substitution (e.g., 'Jobs founded Microsoft').

    If the claim says 'X founded/created Y' but sources say 'Z founded Y',
    return a mismatch signal.

    Returns:
        {"mismatch": True/False, "signal": str}
    """
    m = _FOUNDER_RE.search(claim)
    if not m:
        return {"mismatch": False, "signal": ""}

    claimed_person = m.group(1).strip().lower()
    entity = m.group(2).strip()
    # Extract first meaningful word of entity
    entity_words = re.findall(r'[а-яёa-z]{3,}', entity.lower())
    if not entity_words:
        return {"mismatch": False, "signal": ""}
    entity_key = entity_words[0][:5]

    source_text = " ".join(
        (s.get("snippet", "") + " " + s.get("title", ""))
        for s in sources[:7]
    ).lower()

    # Check if entity is mentioned in sources
    if entity_key not in source_text:
        return {"mismatch": False, "signal": ""}

    # Check if a DIFFERENT person is named as founder in sources
    _founder_patterns = [
        re.compile(
            r'([а-яёa-z]+(?:\s+[а-яёa-z]+)?)\s+'
            r'(?:основал|создал|основатель|founder|founded|co-?founded|создатель)',
            re.IGNORECASE,
        ),
        re.compile(
            r'(?:основатель|founder|создатель|co-?founder)\s+'
            r'([а-яёa-z]+(?:\s+[а-яёa-z]+)?)',
            re.IGNORECASE,
        ),
    ]
    found_persons = set()
    for pat in _founder_patterns:
        for pm in pat.finditer(source_text):
            person = pm.group(1).strip().lower()
            if len(person) > 2 and person not in ('компани', 'компания', 'организаци'):
                found_persons.add(person)

    if not found_persons:
        return {"mismatch": False, "signal": ""}

    # Check if claimed person is NOT among found persons
    claimed_stem = claimed_person[:5]
    person_found = any(claimed_stem in p for p in found_persons)
    if not person_found:
        return {
            "mismatch": True,
            "signal": f"Person mismatch: claim says '{claimed_person}', sources mention {found_persons}",
        }

    return {"mismatch": False, "signal": ""}


def format_verification_hints(claim_info: Dict[str, Any]) -> str:
    """Форматирование подсказок для верификации в текст для промпта."""
    if not claim_info["verification_hints"]:
        return ""

    hints_text = "КЛЮЧЕВЫЕ ТОЧКИ ПРОВЕРКИ:\n"
    for hint in claim_info["verification_hints"]:
        hints_text += f"• {hint}\n"

    return hints_text
