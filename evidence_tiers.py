"""Source authority classification and evidence weighting.

V19 architecture: each search result is assigned an authority tier based on domain.
NLI scores from different tiers are weighted differently in the final confidence
computation — a contradicting Wikipedia snippet counts more than a contradicting
Habr blog post.

Tiers:
- T1 (encyclopedic / peer-reviewed / fact-checker): authoritative, weight=1.0
- T2 (major news / state media): weight=0.7
- T3 (secondary / blogs / aggregators): weight=0.4
- T4 (social / unknown): weight=0.1 (rarely reaches pipeline — already filtered)

The tier map is intentionally conservative — add domains to T1 only when they
meet editorial / citation standards. Unknown domains default to T3.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from urllib.parse import urlparse

T1_DOMAINS = frozenset({
    # Encyclopedic
    "wikipedia.org", "ru.wikipedia.org", "en.wikipedia.org",
    "britannica.com",
    # Fact-checkers
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "factcheck.afp.com", "provereno.media", "stopfake.org",
    # Peer-reviewed / scientific
    "nature.com", "science.org", "cell.com", "nejm.org",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "arxiv.org",
    # Global news agencies / authoritative
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk",
})

T2_DOMAINS = frozenset({
    # Major Russian news
    "tass.ru", "ria.ru", "rbc.ru", "kommersant.ru", "rg.ru",
    "interfax.ru", "crimea.ria.ru", "trends.rbc.ru", "iy.kommersant.ru",
    # Major international business / news
    "bloomberg.com", "ft.com", "wsj.com", "nytimes.com",
    "theguardian.com", "aljazeera.com", "economist.com",
    # State / government
    "kremlin.ru", "gov.ru", "government.ru",
})

T3_DOMAINS = frozenset({
    # Tech / enthusiast / magazine-style
    "habr.com", "vc.ru", "medium.com", "3dnews.ru",
    "lenta.ru", "gazeta.ru", "mk.ru", "aif.ru",
    "kp.ru", "sobesednik.ru", "life.ru",
})

T4_DOMAINS = frozenset({
    # Social / user-generated — usually pre-filtered, listed for completeness
    "vk.com", "telegram.org", "t.me", "pikabu.ru",
    "dzen.ru", "yandex.ru", "twitter.com", "x.com",
    "facebook.com", "youtube.com", "tiktok.com", "instagram.com",
})

TIER_WEIGHTS = {
    "T1": 1.0,
    "T2": 0.7,
    "T3": 0.4,
    "T4": 0.1,
}


def _base_domain(url: str) -> str:
    """Extract the base domain (example.com from http://sub.example.com/path)."""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def classify_source(url: str) -> str:
    """Return the tier label (T1..T4) for a given source URL.

    Unknown domains default to T3 (secondary) rather than T4, to avoid silently
    discarding potentially-useful evidence. T4 is reserved for known low-quality
    social platforms.
    """
    domain = _base_domain(url)
    if not domain:
        return "T3"
    if domain in T1_DOMAINS:
        return "T1"
    # Allow exact match or suffix match (ru.wikipedia.org → wikipedia.org).
    for d in T1_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return "T1"
    if domain in T2_DOMAINS:
        return "T2"
    for d in T2_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return "T2"
    if domain in T4_DOMAINS:
        return "T4"
    for d in T4_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return "T4"
    if domain in T3_DOMAINS:
        return "T3"
    return "T3"


def source_weight(url: str) -> float:
    """Numeric authority weight in [0, 1] for a source URL."""
    return TIER_WEIGHTS[classify_source(url)]


def tier_summary(sources: List[Dict]) -> Dict[str, int]:
    """Count sources per tier, for telemetry / calibration thresholds."""
    counts = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    for s in sources:
        url = s.get("link") or s.get("url", "")
        counts[classify_source(url)] += 1
    return counts


def weighted_nli_scores(
    per_source_scores: List[Tuple[str, float, float]],
) -> Dict[str, float]:
    """Compute authority-weighted max entailment / contradiction.

    Args:
        per_source_scores: list of (url, entailment, contradiction) tuples.

    Returns:
        dict with keys:
          - weighted_ent: authority-weighted max entailment across sources
          - weighted_con: authority-weighted max contradiction
          - top_ent_tier, top_con_tier: tier of the source that produced each max
          - t1_ratio: fraction of sources from T1 (used for calibration)
    """
    if not per_source_scores:
        return {
            "weighted_ent": 0.0,
            "weighted_con": 0.0,
            "top_ent_tier": "T3",
            "top_con_tier": "T3",
            "t1_ratio": 0.0,
        }
    best_ent, best_ent_tier = 0.0, "T3"
    best_con, best_con_tier = 0.0, "T3"
    t1 = 0
    for url, ent, con in per_source_scores:
        tier = classify_source(url)
        w = TIER_WEIGHTS[tier]
        if tier == "T1":
            t1 += 1
        e = ent * w
        c = con * w
        if e > best_ent:
            best_ent, best_ent_tier = e, tier
        if c > best_con:
            best_con, best_con_tier = c, tier
    return {
        "weighted_ent": best_ent,
        "weighted_con": best_con,
        "top_ent_tier": best_ent_tier,
        "top_con_tier": best_con_tier,
        "t1_ratio": t1 / len(per_source_scores),
    }
