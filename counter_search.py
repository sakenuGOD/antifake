"""Multi-framed claim search with active counter-evidence gathering.

V19 architecture: given a claim, generate three framing variants and search all
three in parallel. This ensures:
- affirm_query: finds sources that support the claim (current behaviour)
- deny_query: finds sources that refute the claim (previously under-represented)
- debunk_query: specifically targets fact-check / myth-buster articles

The classical single-query approach biases retrieval toward sources that
syntactically match the claim — which for a false claim tends to return the
same false framing (e.g. "Солнце вращается вокруг Земли" → heliocentric debunk
articles using that exact phrase, but NLI entails because the phrase is present).

By explicitly seeking counter-evidence we:
1. Populate the source pool with both sides of the claim.
2. Raise debunk_count when the claim is a known myth/misconception.
3. Let NLI contradiction signals fire against genuine counter-evidence rather
   than against off-topic snippets.

The reformulation is rule-based first (cheap, deterministic) with optional
LLM augmentation when a generate_fn is provided. Rule-based queries cover
the common cases; LLM reformulation helps with nuanced claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class QueryFrame:
    """A single search query with its framing intent."""
    text: str                # the query string
    frame: str               # "affirm" | "deny" | "debunk" | "verify"
    weight: float = 1.0      # relative importance for downstream aggregation


# Debunk / verification keywords used to generate counter-queries.
# Chosen to match what fact-check and myth-buster articles actually publish.
_DEBUNK_SUFFIXES_RU = (
    "миф разоблачение",
    "правда ли это",
    "опровержение",
    "не соответствует действительности",
)

_DEBUNK_SUFFIXES_EN = (
    "myth debunked",
    "fact check",
    "is it true",
    "false claim",
)

# Domains to target with dedicated site: filters for the debunk query. These
# are fact-check specialists that are sparse in generic DDG results.
_FACTCHECK_SITES = (
    "snopes.com", "politifact.com", "factcheck.org",
    "fullfact.org", "factcheck.afp.com",
)

# Tokens that, when present in the claim, strongly suggest it is an assertion
# worth counter-checking (as opposed to a query / question).
_ASSERTION_TOKENS = re.compile(
    r"\b(является|был[аи]?|была|были|находится|составляет|"
    r"вращается|содержит|создан[аы]?|имеет|носил[аи]?)\b",
    re.IGNORECASE,
)


def _clean(claim: str) -> str:
    """Normalise a claim for search: strip quotes, collapse whitespace."""
    text = claim.strip().strip('«»"\'')
    text = re.sub(r"\s+", " ", text)
    # Strip trailing punctuation that search engines don't need.
    return text.rstrip(".?!,;:")


def build_query_frames(
    claim: str,
    max_frames: int = 4,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> List[QueryFrame]:
    """Build a list of search queries covering multiple framings of a claim.

    Rule-based (deterministic) by default. Adds LLM-generated paraphrases when
    generate_fn is supplied. Returns at most max_frames unique frames.

    Order of returned frames is meaningful: downstream aggregation may prefer
    the earlier frames when truncating results.
    """
    base = _clean(claim)
    frames: List[QueryFrame] = []

    # 1. Affirm: claim as-is. Always included.
    frames.append(QueryFrame(text=base, frame="affirm", weight=1.0))

    # 2. Verify: "правда ли, что {claim}" — finds sources that explicitly
    # address the veracity of the claim (both confirmations and debunks).
    frames.append(QueryFrame(
        text=f"правда ли что {base}",
        frame="verify",
        weight=0.9,
    ))

    # 3. Debunk: target fact-check keywords. Multiple suffixes tried as one
    # OR-joined query to cast a wide net without exploding the query count.
    debunk_terms = " OR ".join(f'"{s}"' for s in _DEBUNK_SUFFIXES_RU[:2])
    frames.append(QueryFrame(
        text=f"{base} {debunk_terms}",
        frame="debunk",
        weight=0.8,
    ))

    # 4. Factcheck site-filter: only if the claim is an assertion (not a
    # question / open-ended search). site: queries require reasonable claim
    # structure to return results.
    if _ASSERTION_TOKENS.search(base):
        sites = " OR ".join(f"site:{s}" for s in _FACTCHECK_SITES[:3])
        frames.append(QueryFrame(
            text=f"{base} ({sites})",
            frame="debunk",
            weight=0.7,
        ))

    # 5. V23c: Wikipedia's "List of common misconceptions" is a curated
    # structured source for myths (Vikings helmets, Napoleon height, 10%
    # brain, flat earth, etc.). The RU version is missing, so target EN
    # page directly. This helps claims where standard DDG doesn't surface
    # debunk articles because the myth is too mainstream for recent news.
    # General mechanism — works for ANY myth whose English name is in
    # the article, not specific to any test claim.
    frames.append(QueryFrame(
        text=f'"common misconceptions" {base} site:en.wikipedia.org',
        frame="debunk",
        weight=0.6,
    ))

    # Optional LLM-generated paraphrase. Kept narrow: one extra query at most,
    # guarded against returning an echo of the input.
    if generate_fn is not None and len(frames) < max_frames:
        paraphrase = _llm_paraphrase(base, generate_fn)
        if paraphrase and paraphrase.lower() != base.lower():
            frames.append(QueryFrame(
                text=paraphrase,
                frame="affirm",
                weight=0.6,
            ))

    # Dedup while preserving order.
    seen = set()
    unique: List[QueryFrame] = []
    for f in frames:
        key = f.text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
        if len(unique) >= max_frames:
            break
    return unique


def _llm_paraphrase(claim: str, generate_fn: Callable[[str], str]) -> str:
    """Ask the LLM for one search-friendly paraphrase of the claim.

    Failure modes (model refuses, returns junk, times out) fall back to
    empty string — caller handles that.
    """
    prompt = (
        "[INST]Перефразируй утверждение в короткий поисковый запрос "
        "(3-7 слов, ключевые сущности). Без кавычек, без пояснений.\n"
        f"Утверждение: {claim}\n"
        "Запрос:[/INST]"
    )
    try:
        out = generate_fn(prompt)
    except Exception:
        return ""
    if not out:
        return ""
    # Take first line, strip common wrappers.
    line = str(out).strip().splitlines()[0].strip()
    line = line.strip('«»"\'').strip()
    # Reject obvious failures: too long, echoes input, contains INST markers.
    if len(line) > 200 or "[" in line or "]" in line:
        return ""
    return line


def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]],
    k: int = 60,
) -> List[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document = sum over lists of 1 / (k + rank_in_list).
    k=60 is the standard default from the original paper (Cormack 2009) —
    large enough that top-ranked items dominate but low-ranked items still
    contribute.

    Each input list is a ranked list of source dicts with a "link" field
    (used as the dedup key). Returned list is sorted by fused score and
    retains the original metadata from the first occurrence of each URL.
    """
    scores: dict = {}
    docs: dict = {}
    for rlist in ranked_lists:
        for rank, doc in enumerate(rlist):
            url = (doc.get("link") or doc.get("url") or "").strip()
            if not url:
                continue
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank)
            if url not in docs:
                docs[url] = doc
    fused = sorted(
        ((url, s) for url, s in scores.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [dict(docs[url], _rrf_score=s) for url, s in fused]
