"""Детектор сатиры, кликбейта и сомнительных заголовков.

Используется для понижения веса таких источников в NLI и ранжировании.
"""

import re
from typing import Optional

SATIRE_PATTERNS = [
    r"великий советский",
    r"ошеломляющее открытие учёных",
    r"сенсаци[яю]",
    r"учёные доказали что",
    r"шок(?:ирующ|!)",
    r"вы не поверите",
    r"британские учёные",
    r"панорама",
    r"это должен знать каждый",
    r"врачи в шоке",
    r"[\!\?]{3,}",
    r"СРОЧНО\s*!",
]

SATIRE_DOMAINS = {
    "panorama.pub",
    "theonion.com",
    "babylonbee.com",
}

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in SATIRE_PATTERNS]


def is_satire(title: str, domain: Optional[str] = None) -> bool:
    """Проверяет заголовок на сатиру/clickbait."""
    if domain and domain in SATIRE_DOMAINS:
        return True
    for pat in _compiled_patterns:
        if pat.search(title):
            return True
    return False


def satire_penalty(title: str, domain: Optional[str] = None) -> float:
    """Возвращает штраф за сатиру (0.0 = нет, 0.5 = сатира)."""
    return 0.5 if is_satire(title, domain) else 0.0
