"""Оценка надёжности источников новостей.

Домены с высокой репутацией получают бонус при ранжировании,
неизвестные или сомнительные — нейтральный score.
"""

from urllib.parse import urlparse
from typing import List, Dict

# Рейтинг надёжности (0.0 - 1.0)
SOURCE_CREDIBILITY = {
    # Государственные/крупные информагентства РФ (0.80-0.95)
    "tass.ru": 0.90,
    "ria.ru": 0.88,
    "interfax.ru": 0.90,
    "rbc.ru": 0.85,
    "kommersant.ru": 0.87,
    "vedomosti.ru": 0.88,
    "iz.ru": 0.82,
    "gazeta.ru": 0.80,
    "lenta.ru": 0.78,
    "fontanka.ru": 0.78,
    "meduza.io": 0.82,
    "novayagazeta.ru": 0.80,
    "forbes.ru": 0.82,

    # Международные (0.85-0.95)
    "reuters.com": 0.95,
    "apnews.com": 0.93,
    "bbc.com": 0.90,
    "bbc.co.uk": 0.90,
    "nytimes.com": 0.90,
    "theguardian.com": 0.88,
    "france24.com": 0.85,
    "dw.com": 0.85,
    "aljazeera.com": 0.82,

    # Техно-СМИ (0.70-0.80)
    "habr.com": 0.78,
    "3dnews.ru": 0.75,
    "ixbt.com": 0.73,
    "cnews.ru": 0.72,

    # Официальные ведомства (0.90-0.98)
    "gov.ru": 0.95,
    "kremlin.ru": 0.95,
    "cbr.ru": 0.95,
    "who.int": 0.93,
    "un.org": 0.93,
    "nasa.gov": 0.95,

    # Социальные / агрегаторы (0.25-0.45)
    "t.me": 0.30,
    "vk.com": 0.25,
    "zen.yandex.ru": 0.35,
    "dzen.ru": 0.35,
    "ok.ru": 0.25,
    "pikabu.ru": 0.30,
}


def get_credibility(url: str) -> float:
    """Получить оценку надёжности домена (0.0-1.0)."""
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")

        # Точное совпадение
        if domain in SOURCE_CREDIBILITY:
            return SOURCE_CREDIBILITY[domain]

        # Поддомены (news.google.com → google.com)
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in SOURCE_CREDIBILITY:
                return SOURCE_CREDIBILITY[parent]

    except Exception:
        pass

    return 0.50  # Неизвестный источник — нейтральный


def boost_by_credibility(
    results: List[Dict],
    semantic_weight: float = 0.7,
    credibility_weight: float = 0.3,
) -> List[Dict]:
    """Бустирование результатов с учётом надёжности источника.

    final_score = semantic_weight * semantic_score + credibility_weight * credibility
    """
    for r in results:
        cred = get_credibility(r.get("link", ""))
        r["source_credibility"] = round(cred, 2)

        if "semantic_score" in r:
            r["final_score"] = round(
                semantic_weight * r["semantic_score"] +
                credibility_weight * cred,
                4,
            )
        else:
            r["final_score"] = round(cred, 4)

    results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    return results
