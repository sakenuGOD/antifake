"""V17: Evidence-first fact-checking pipeline.

7-stage linear pipeline: PARSE → DECOMPOSE → SEARCH → EVIDENCE → DECIDE → AGGREGATE → EXPLAIN.
No backward overrides, no guards-on-guards, no self-consistency re-runs.
LLM explains, never judges. Hard facts override soft signals.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from config import ModelConfig, PipelineConfig, SearchConfig
from prompts import KEYWORD_EXTRACTION_TEMPLATE, EXPLANATION_TEMPLATE, LLM_KNOWLEDGE_TEMPLATE
from model import load_base_model, load_finetuned_model, build_langchain_llm, is_grpo_adapter
from search import FactCheckSearcher, boost_factcheck_scores
from claim_parser import (
    classify_claim, format_verification_hints, extract_numbers, compare_numbers,
    detect_scam_patterns,
)
from fact_cache import FactCache
from utils import STOPWORDS

# V8: Natasha NER + pymorphy2
try:
    from nlp_russian import (
        extract_entities, extract_entity_names, lemmatize,
        lemmatize_text, extract_keywords_lemmatized, stems_match,
        words_overlap_lemmatized,
    )
    _HAS_NATASHA = True
except ImportError:
    _HAS_NATASHA = False
    print("[WARNING] natasha/pymorphy2 не установлены — используем fallback regex")

logger = logging.getLogger("fact_checker")

# Debunk/myth detection regex for source snippets
_DEBUNK_RE = re.compile(
    r'(?:миф|заблуждени[ея]|заблуждения|popular\s+misconception|urban\s+legend|'
    r'не\s+(?:соответствует|является)\s+действительности|'
    r'разоблачен|debunked|опроверг|опровержение|'
    r'на\s+самом\s+деле|in\s+fact|actually|'
    r'ошибка\s+(?:перевода|трактовки)|myth|false\s+claim|'
    r'ложн\w+\s+(?:утверждение|информация|факт)|'
    r'это\s+неправда|это\s+ложь|не\s+правда)',
    re.IGNORECASE
)

# Months and generic words for keyword cleanup
_MONTHS_RU = {
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
}

# Multi-word entity reassembly
_MULTI_ENTITIES = [
    "цб рф", "цб рф", "мвд рф", "мид рф", "фсб рф",
    "вс рф", "минобороны рф", "правительство рф",
]


def _rejoin_entities(keywords: List[str]) -> List[str]:
    """Rejoin known multi-word entities that were split by LLM."""
    text = " ".join(keywords).lower()
    for ent in _MULTI_ENTITIES:
        parts = ent.split()
        if all(any(p in kw.lower() for kw in keywords) for p in parts):
            keywords = [kw for kw in keywords if kw.lower().strip() not in parts]
            keywords.insert(0, ent.upper())
    return keywords


@dataclass
class PipelineContext:
    """Per-request context. Created fresh for each check() call."""
    raw_results: List[Dict] = field(default_factory=list)
    nli_result: Optional[Dict] = None
    num_comparisons: List[Dict] = field(default_factory=list)
    claim_info: Optional[Dict] = None
    claim_entities: List[str] = field(default_factory=list)
    is_scam: bool = False
    sub_claims: List[str] = field(default_factory=list)
    wikidata_result: Dict = field(default_factory=dict)


class FactCheckPipeline:
    """V17: Evidence-first fact-checking pipeline."""

    _COMMON_WORDS = STOPWORDS

    def __init__(
        self,
        adapter_path: str = None,
        model_config: ModelConfig = None,
        pipeline_config: PipelineConfig = None,
        search_config: SearchConfig = None,
    ):
        if model_config is None:
            model_config = ModelConfig()
        if pipeline_config is None:
            pipeline_config = PipelineConfig()
        if search_config is None:
            search_config = SearchConfig()

        self.pipeline_config = pipeline_config
        self.searcher = FactCheckSearcher(search_config)
        self._use_reasoning_template = is_grpo_adapter(adapter_path)
        self._ctx = PipelineContext()
        self.fact_cache = FactCache()

        # Cross-encoder re-ranker (CPU)
        self._reranker = None
        if pipeline_config.enable_cross_encoder:
            try:
                from embeddings import ReRanker
                _ce_model = getattr(pipeline_config, 'cross_encoder_model', None)
                print(f"Загрузка CrossEncoder re-ranker ({_ce_model or 'default'})...")
                self._reranker = ReRanker(model_name=_ce_model) if _ce_model else ReRanker()
            except Exception as e:
                print(f"CrossEncoder не загружен: {e}")

        # NLI checker (CPU)
        self.nli_checker = None
        if pipeline_config.enable_nli:
            try:
                from nli_checker import NLIChecker
                print("Загрузка NLI модели на CPU...")
                self.nli_checker = NLIChecker(device=pipeline_config.nli_device)
                print("NLI модель загружена.")
            except Exception as e:
                print(f"NLI модель не загружена: {e}")

        # Load LLM model
        if adapter_path and os.path.exists(adapter_path):
            adapter_type = "GRPO" if self._use_reasoning_template else "SFT"
            print(f"Загрузка {adapter_type} модели из {adapter_path}...")
            model, tokenizer = load_finetuned_model(adapter_path, model_config)
        else:
            print("Загрузка base модели...")
            model, tokenizer = load_base_model(model_config)

        # Two LLM wrappers: keywords (short) and explanation (medium)
        self.keyword_llm = build_langchain_llm(
            model, tokenizer,
            max_new_tokens=pipeline_config.keyword_max_new_tokens,
            pipeline_config=pipeline_config,
        )
        self.explain_llm = build_langchain_llm(
            model, tokenizer,
            max_new_tokens=pipeline_config.verdict_max_new_tokens,
            pipeline_config=pipeline_config,
        )

        # Connect LLM to query classifier
        self.searcher.set_generate_fn(lambda p: self.keyword_llm.invoke(p))

        adapter_type = "GRPO" if self._use_reasoning_template else ("SFT" if adapter_path else "Base")
        print(f"  [Pipeline V17] Адаптер: {adapter_type}")
        print(f"  [Pipeline V17] Evidence-first architecture (LLM explains, never judges)")

    # ======================================================================
    # STAGE 1: PARSE
    # ======================================================================

    def _parse_claim(self, claim: str) -> Dict[str, Any]:
        """Parse claim: classify type, extract numbers/dates, detect scam."""
        claim_info = classify_claim(claim)
        self._ctx.claim_info = claim_info
        self._ctx.is_scam = claim_info.get("is_scam", False)
        return claim_info

    # ======================================================================
    # STAGE 2: DECOMPOSE (rule-based only)
    # ======================================================================

    @staticmethod
    def _has_conjunction(claim: str) -> bool:
        """Check if claim has compound markers."""
        claim_lower = claim.lower()
        markers = [" и ", ", а также", ", при этом", ", причём", ", а "]
        if any(m in claim_lower for m in markers):
            return True
        numbers = re.findall(r'\d+(?:[.,]\d+)?', claim)
        if len(numbers) > 1:
            return True
        return False

    @staticmethod
    def _split_by_conjunctions(claim: str) -> list:
        """Rule-based split of composite claims by conjunctions."""
        parts = re.split(
            r',\s*(?:и|а\s+также|при\s+этом|причём|а)\s+|\s+(?:и|а\s+также)\s+',
            claim, flags=re.IGNORECASE)
        if len(parts) <= 1:
            return [claim]

        first = parts[0].strip()
        subj_match = re.match(
            r'^(.+?)\s+(?:имеет|является|находится|входит|использует|'
            r'расположен[аоы]?|основан[аоы]?|был[аоы]?|стал[аоы]?|'
            r'содержит|составляет|протекает|впадает|насчитывает|'
            r'включает|обладает|достигает|'
            r'родил(?:ся|ась)|получил[аоы]?|написал[аоы]?|'
            r'изобрел[аоы]?|изобрёл[аоы]?|открыл[аоы]?|создал[аоы]?|'
            r'совершил[аоы]?|выиграл[аоы]?|состоит|'
            r'вращается|обращается|летает|летит|'
            r'весит|длится|занимает|помнит|проглатывает|'
            r'вызывает|'
            r'—\s|–\s|-\s)', first, re.IGNORECASE)
        subject = subj_match.group(1).strip() if subj_match else ""

        result = [first]
        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
            if subject and not re.match(r'^[А-ЯЁA-Z]', part):
                part = f"{subject} {part}"
            result.append(part)

        return [p for p in result if len(p) > 5]

    def _decompose(self, claim: str) -> List[str]:
        """Decompose claim into sub-claims (rule-based only)."""
        if not self._has_conjunction(claim):
            return [claim]
        parts = self._split_by_conjunctions(claim)
        return parts if len(parts) > 1 else [claim]

    # ======================================================================
    # STAGE 3: SEARCH
    # ======================================================================

    def _extract_claim_entities(self, claim: str, keywords: List[str] = None) -> List[str]:
        """Extract named entities from claim for search & matching."""
        entities = []
        if _HAS_NATASHA:
            try:
                ner_entities = extract_entities(claim)
                for ent in ner_entities:
                    entities.append(ent["normal"])
                    entities.append(ent["text"].lower())
            except Exception:
                pass

        for m in re.finditer(r'[А-ЯЁA-Z][а-яёa-zA-Z]{2,}', claim):
            word = m.group(0)
            idx = m.start()
            if idx == 0 or claim[idx - 1] in '.!?\n':
                continue
            entities.append(word.lower())

        if keywords:
            for kw in keywords:
                caps = re.findall(r'[А-ЯЁA-Z][а-яёa-zA-Z]{2,}', kw)
                if len(caps) >= 2:
                    entities.append(kw.lower().strip())
                for c in caps:
                    entities.append(c.lower())

        for m in re.finditer(r'[«"]([^»"]+)[»"]', claim):
            entities.append(m.group(1).lower())

        for w in re.findall(r'[а-яёa-z]{4,}', claim.lower()):
            if w not in self._COMMON_WORDS:
                entities.append(w)

        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique

    @staticmethod
    def _extract_keywords_rule_based(claim: str) -> str:
        """Rule-based keyword extraction."""
        keywords = []

        for m in re.finditer(r'[«"]([^»"]+)[»"]', claim):
            keywords.append(m.group(1))

        for m in re.finditer(r'[А-ЯЁA-Z][а-яёa-zA-Z]+(?:\s+[А-ЯЁA-Z][а-яёa-zA-Z]+)+', claim):
            keywords.append(m.group(0))

        words = claim.split()
        for i, w in enumerate(words):
            if i == 0:
                continue
            clean = w.strip('.,!?;:«»"()-')
            if clean and clean[0].isupper() and len(clean) >= 3:
                if clean.lower() not in STOPWORDS:
                    keywords.append(clean)

        for m in re.finditer(r'\d+(?:[.,]\d+)?(?:\s*(?:%|млн|млрд|тыс|миллион|миллиард|трлн))', claim):
            keywords.append(m.group(0))

        for m in re.finditer(r'\b((?:19|20)\d{2})\b', claim):
            keywords.append(m.group(1))

        for w in re.findall(r'[а-яёa-z]{6,}', claim.lower()):
            if w not in STOPWORDS and w not in [k.lower() for k in keywords]:
                keywords.append(w)

        seen = set()
        unique = []
        for k in keywords:
            kl = k.lower().strip()
            if kl and kl not in seen:
                seen.add(kl)
                unique.append(k)

        return ", ".join(unique[:7])

    @staticmethod
    def _parse_keywords(raw_output: str) -> List[str]:
        """Parse keywords from LLM or rule-based output."""
        text = raw_output.strip()
        for prefix in ["Ключевые слова:", "Keywords:", "ключевые слова:",
                       "КЛЮЧЕВЫЕ СЛОВА:", "Ключевые сущности:",
                       "ключевые сущности:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        text = re.sub(r'^\d+[.)]\s*', '', text, flags=re.MULTILINE)

        if "," in text:
            raw_keywords = [kw.strip().strip('"').strip("'").strip("-—•") for kw in text.split(",")]
        elif "\n" in text:
            raw_keywords = [kw.strip().strip('"').strip("'").strip("-—•") for kw in text.split("\n")]
        else:
            raw_keywords = [text.strip()]

        expanded = []
        for kw in raw_keywords:
            if '\n' in kw:
                expanded.extend(part.strip().strip('"').strip("'").strip("-—•") for part in kw.split('\n'))
            else:
                expanded.append(kw)
        raw_keywords = [kw for kw in expanded if kw and len(kw) < 60]

        raw_keywords = _rejoin_entities(raw_keywords)

        cleaned = []
        for kw in raw_keywords:
            kw_lower = kw.lower().strip()
            if kw_lower in _MONTHS_RU:
                continue
            if re.fullmatch(r'\d+', kw_lower):
                continue
            if len(kw_lower) < 3:
                continue
            cleaned.append(kw.strip())

        seen = set()
        result = []
        for kw in cleaned:
            norm = kw.lower().strip()
            if norm not in seen:
                seen.add(norm)
                is_sub = False
                for existing_norm in list(seen):
                    if existing_norm != norm and norm in existing_norm:
                        is_sub = True
                        break
                if not is_sub:
                    result.append(kw)

        return result[:7] if result else [text.strip()[:60]]

    def _search_sources(self, claim: str, keywords: List[str]) -> List[Dict]:
        """Search for sources using Wikipedia + DDG."""
        claim_entities = self._extract_claim_entities(claim, keywords=keywords)
        self._ctx.claim_entities = claim_entities

        # NER entities for Wikipedia
        _wiki_entities = list(claim_entities)
        if _HAS_NATASHA:
            try:
                ner_names = extract_entity_names(claim)
                for name in ner_names:
                    if name.lower() not in [e.lower() for e in _wiki_entities]:
                        _wiki_entities.insert(0, name)
            except Exception:
                pass

        # Wikipedia PRIMARY
        wiki_results = self.searcher.wiki_entity_lookup(_wiki_entities + keywords[:3])
        results = list(wiki_results)
        existing_urls = {r.get("link", "") for r in results}

        # DDG supplementary
        ddg_results = self.searcher.search_all_keywords(keywords, claim=claim)
        for dr in ddg_results:
            url = dr.get("link", "")
            if url and url not in existing_urls:
                results.append(dr)
                existing_urls.add(url)

        # Verification queries
        try:
            _generate_fn = lambda p: self.keyword_llm.invoke(p)
            verif_queries = self.searcher.generate_verification_queries(
                claim, claim_entities[:5], generate_fn=_generate_fn)
            if verif_queries:
                verif_results = self.searcher.search_verification_queries(verif_queries)
                for vr in verif_results:
                    url = vr.get("link", "")
                    if url and url not in existing_urls:
                        results.append(vr)
                        existing_urls.add(url)
        except Exception:
            pass

        # Counter-entity search
        try:
            counter_results = self.searcher.search_counter_entities(claim, claim_entities[:5])
            for cr in counter_results:
                url = cr.get("link", "")
                if url and url not in existing_urls:
                    results.append(cr)
                    existing_urls.add(url)
        except Exception:
            pass

        wiki_cnt = sum(1 for r in results if 'wikipedia.org' in r.get('link', ''))
        print(f"  Найдено источников: {len(results)} (Wikipedia: {wiki_cnt})")

        # Rank by relevance
        results = self.searcher.rank_by_relevance(claim, results)

        # Entity pre-filter
        if claim_entities:
            distinctive = [e for e in claim_entities if ' ' in e or len(e) >= 5]
            filter_ents = distinctive if distinctive else claim_entities

            def _has_entity(r):
                text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                return any(e in text for e in filter_ents)

            filtered = [r for r in results if _has_entity(r)]
            if len(filtered) >= 5:
                results = filtered

        # Cross-encoder re-ranking
        if self._reranker and results:
            results = self._reranker.rerank(claim, results, top_k=min(7, len(results)))
            results = boost_factcheck_scores(results)

        return results

    # ======================================================================
    # STAGE 4: EVIDENCE (three independent signals)
    # ======================================================================

    def _check_wikidata(self, claim: str, entities: List[str]) -> int:
        """Check claim against Wikidata. Returns -1, 0, or +1."""
        try:
            from wikidata import check_structured_facts
            result = check_structured_facts(claim, entities)
            self._ctx.wikidata_result = result
            if not result.get("found"):
                return 0
            facts = result.get("facts", [])
            confirmed = sum(1 for f in facts if f.get("match") is True)
            contradicted = sum(1 for f in facts if f.get("match") is False)

            # Neutral facts matching claim text = implicit confirmation
            if _HAS_NATASHA:
                claim_lemmas = set(lemmatize(w) for w in re.findall(r'[а-яёa-z]{3,}', claim.lower()))
                for f in facts:
                    if f.get("match") is None:
                        for v in f.get("wikidata_values", []):
                            if lemmatize(v) in claim_lemmas or v.lower() in claim.lower():
                                confirmed += 1
                                break

            if contradicted > 0:
                return -1
            elif confirmed > 0:
                return +1
            return 0
        except Exception as e:
            print(f"  [Wikidata] Ошибка: {e}")
            self._ctx.wikidata_result = {"found": False, "facts": []}
            return 0

    def _check_numbers(self, claim: str, sources: List[Dict], claim_info: Dict) -> int:
        """Check numbers in claim vs sources. Returns -1, 0, or +1."""
        claim_numbers = claim_info.get("numbers", [])
        if not claim_numbers:
            return 0

        # Extract numbers from all source snippets
        source_numbers = []
        for src in sources[:7]:
            snippet = src.get("snippet", "")
            if snippet:
                source_numbers.extend(extract_numbers(snippet))

        if not source_numbers:
            return 0

        comparisons = compare_numbers(claim_numbers, source_numbers)
        self._ctx.num_comparisons = comparisons

        if not comparisons:
            return 0

        # V17.1: Year mismatches override matches (year in wrong context doesn't confirm)
        has_year_mismatch = any(
            not c["match"] for c in comparisons
            if c.get("source_number") and c.get("claim_number", {}).get("type") == "year"
        )
        if has_year_mismatch:
            return -1

        # Non-year: matches override mismatches
        has_match = any(c["match"] for c in comparisons if c.get("source_number"))
        has_mismatch = any(not c["match"] for c in comparisons if c.get("source_number"))

        if has_match:
            return +1
        elif has_mismatch:
            return -1
        return 0

    def _check_nli(self, claim: str, sources: List[Dict]) -> tuple:
        """Check claim vs sources using NLI. Returns (signal, scores)."""
        if not self.nli_checker or not sources:
            return 0, {"ent": 0.0, "con": 0.0}

        nli_result = self.nli_checker.check_claim(claim, sources[:7], snippet_key="snippet")
        self._ctx.nli_result = nli_result
        max_ent = nli_result.get("max_entailment", 0.0)
        max_con = nli_result.get("max_contradiction", 0.0)

        # Cross-encoder tiebreaker for close cases — REPLACE scores, not MAX
        if abs(max_ent - max_con) < 0.15:
            if self.nli_checker._cross_encoder is None:
                try:
                    self.nli_checker._init_cross_encoder()
                except Exception:
                    pass
            if self.nli_checker._cross_encoder is not None:
                try:
                    ce_result = self.nli_checker.check_claim_cross(claim, sources[:3], snippet_key="snippet")
                    ce_ent = ce_result.get("max_entailment", 0.0)
                    ce_con = ce_result.get("max_contradiction", 0.0)
                    # V17.1: Replace with CE scores when primary is ambiguous
                    max_ent = ce_ent
                    max_con = ce_con
                except Exception:
                    pass

        scores = {
            "ent": max_ent,
            "con": max_con,
            # Use FINAL (post-CE) scores for is_contested — pre-CE purity values can differ
            "is_contested": max_ent >= 0.40 and max_con >= 0.40,
        }

        # V17.1: Gap-based threshold — dominant signal wins if gap >= 0.15
        gap = abs(max_ent - max_con)
        if max_ent >= 0.50 and gap >= 0.20 and max_ent > max_con:
            return +1, scores
        elif max_con >= 0.50 and gap >= 0.20 and max_con > max_ent:
            return -1, scores
        else:
            return 0, scores

    def _check_debunk(self, claim: str, sources: List[Dict]) -> int:
        """Check if sources are debunking the claim. Returns count of debunk sources."""
        claim_words = set(re.findall(r'[а-яёa-z]{4,}', claim.lower()))
        count = 0
        for r in sources[:7]:
            text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
            src_words = set(re.findall(r'[а-яёa-z]{4,}', text))
            overlap = len(claim_words & src_words) / max(len(claim_words), 1)
            if _DEBUNK_RE.search(text) and overlap >= 0.3:
                count += 1
        return count

    # ======================================================================
    # STAGE 5: DECIDE (priority-based decision tree)
    # ======================================================================

    def _check_llm_knowledge(self, claim: str) -> int:
        """LLM parametric knowledge check — Tier 4 fallback.

        Использует fine-tuned Mistral 7B для проверки факта из параметрической
        памяти модели. Запускается только когда wikidata/NLI/numbers не дали сигнал.
        Returns: -1 (ЛОЖЬ), 0 (неизвестно), +1 (ПРАВДА)
        """
        try:
            prompt = LLM_KNOWLEDGE_TEMPLATE.format(claim=claim[:200])
            raw = self.keyword_llm.invoke(prompt)
            text = StrOutputParser().invoke(raw).strip().upper()[:50]
            if "ПРАВДА" in text or "\nTRUE" in text:
                return +1
            elif "ЛОЖЬ" in text or "FALSE" in text or "НЕПРАВДА" in text:
                return -1
            return 0
        except Exception as e:
            logger.debug(f"LLM knowledge check failed: {e}")
            return 0

    def _decide(self, wikidata_signal: int, numbers_signal: int,
                nli_signal: int, nli_scores: Dict, is_scam: bool,
                debunk_count: int = 0, llm_signal: int = 0) -> tuple:
        """Priority-based verdict. Hard facts override soft signals.

        Returns: (verdict, confidence)
        """
        # TIER 0: Scam
        if is_scam:
            return "ЛОЖЬ", 95

        # TIER 1: Wikidata (structured DB = highest authority)
        if wikidata_signal == -1:
            return "ЛОЖЬ", 90
        if wikidata_signal == +1 and numbers_signal != -1:
            return "ПРАВДА", 85

        # TIER 2: Numbers (deterministic)
        if numbers_signal == -1:
            return "ЛОЖЬ", 85
        if numbers_signal == +1 and wikidata_signal != -1:
            # NLI явно противоречит — число нашлось в другом контексте
            # (пример: '1939' есть в статьях о ВМВ, но WWI началась в 1914)
            if nli_signal == -1:
                return "ЛОЖЬ", 75
            _con = nli_scores.get("con", 0)
            _ent = nli_scores.get("ent", 0)
            # Слабое противоречие NLI (gap >= 0.10) тоже бьёт ложное числовое совпадение
            if _con >= 0.40 and _con > _ent + 0.10:
                return "ЛОЖЬ", 68
            # Оба NLI-скора высокие (genuinely conflicted) + LLM знает что факт ложный
            if _con >= 0.50 and _ent >= 0.50 and llm_signal == -1:
                return "ЛОЖЬ", 68
            return "ПРАВДА", 80

        # TIER 2.5: Contested topic — источники реально делятся (max_ent >= 0.40 И max_con >= 0.40)
        # Срабатывает для любой оспариваемой темы: мобильники+рак, ГМО, альт-медицина и т.д.
        # НЕ срабатывает для мифов (max_ent < 0.40) и для фактических ошибок (WD/NUM сигнал)
        if nli_scores.get("is_contested") and wikidata_signal == 0 and numbers_signal == 0:
            # LLM как арбитр: при оспариваемых темах параметрические знания решают
            if llm_signal == +1:
                return "ПРАВДА", 62
            elif llm_signal == -1:
                return "ЛОЖЬ", 58
            return "НЕ УВЕРЕНА", 50

        # TIER 3: NLI (statistical)
        if nli_signal == -1:
            _con = nli_scores.get("con", 0)
            _ent = nli_scores.get("ent", 0)
            # LLM veto: синоним/перефразировка (ent≈0 = источники не нашли эту формулировку)
            # ent < 0.05 означает: NO sources entail → чисто лексическая проблема, не фактическая
            # (Менделеев "изобрёл"→ent=0.02: нет entail из-за синонима "создал/открыл")
            # ent=0.11 (Сахар) или выше → есть реальное обсуждение, NLI противоречие не синоним
            if llm_signal == +1 and wikidata_signal == 0 and debunk_count == 0 and _ent < 0.05:
                return "ПРАВДА", 60
            # Мягкое противоречие с реальной поддержкой → оспариваемый факт (мобильники+рак)
            # Требует: ent >= 0.20 (реальные "за" источники), con >= 0.70, нет debunk-источников
            if _ent >= 0.20 and _con >= 0.70 and debunk_count == 0:
                return "НЕ УВЕРЕНА", 48
            return "ЛОЖЬ", 70
        if nli_signal == +1:
            return "ПРАВДА", 75

        # TIER 3.5: Debunk sources (myth detection)
        if debunk_count >= 2:
            return "ЛОЖЬ", 65
        if debunk_count == 1:
            # Require minimum gap to avoid float precision noise
            if nli_scores.get("con", 0) > nli_scores.get("ent", 0) + 0.05:
                return "ЛОЖЬ", 60

        # TIER 4: LLM parametric knowledge (last resort — fine-tuned model знает факты)
        if llm_signal == +1:
            return "ПРАВДА", 65
        elif llm_signal == -1:
            return "ЛОЖЬ", 60

        # TIER 5: No signal at all
        return "НЕ УВЕРЕНА", 45

    # ======================================================================
    # STAGE 6: AGGREGATE (for composite claims)
    # ======================================================================

    @staticmethod
    def _aggregate(sub_results: List[Dict]) -> str:
        """Aggregate sub-results into final verdict."""
        verdicts = [r["verdict"] for r in sub_results]
        if "ПРАВДА" in verdicts and "ЛОЖЬ" in verdicts:
            return "СОСТАВНОЕ"
        elif all(v == "ПРАВДА" for v in verdicts):
            return "ПРАВДА"
        elif any(v == "ЛОЖЬ" for v in verdicts):
            return "ЛОЖЬ"
        else:
            return "НЕ УВЕРЕНА"

    # ======================================================================
    # STAGE 7: EXPLAIN (LLM — explanation only, NOT verdict)
    # ======================================================================

    def _explain(self, claim: str, verdict: str, sub_results: List[Dict],
                 sources: List[Dict]) -> str:
        """Generate explanation using LLM. Verdict already decided."""
        # Build evidence summary
        evidence_lines = []

        # Wikidata evidence
        wd = self._ctx.wikidata_result
        if wd and wd.get("found"):
            for f in wd.get("facts", []):
                prop = f.get("property_label", "")
                vals = f.get("wikidata_values", [])
                match = f.get("match")
                if match is True:
                    evidence_lines.append(f"Wikidata подтверждает: {prop} = {', '.join(vals)}")
                elif match is False:
                    evidence_lines.append(f"Wikidata противоречит: {prop} = {', '.join(vals)}")

        # Number comparisons
        for nc in self._ctx.num_comparisons:
            if nc.get("source_number"):
                cn = nc["claim_number"]
                sn = nc["source_number"]
                if nc["match"]:
                    evidence_lines.append(f"Число подтверждено: {cn['raw']} ≈ {sn['raw']}")
                else:
                    evidence_lines.append(f"Числовое расхождение: {cn['raw']} ≠ {sn['raw']}")

        # NLI evidence
        nli = self._ctx.nli_result
        if nli:
            evidence_lines.append(
                f"NLI анализ: подтверждение={nli.get('max_entailment', 0):.2f}, "
                f"противоречие={nli.get('max_contradiction', 0):.2f}"
            )

        # Source summaries
        for src in sources[:3]:
            title = src.get("title", "")
            snippet = src.get("snippet", "")[:200]
            if title and snippet:
                evidence_lines.append(f"[{title}]: {snippet}")

        evidence_summary = "\n".join(evidence_lines) if evidence_lines else "Источники не найдены."

        # Sub-results for composite
        if len(sub_results) > 1:
            for sr in sub_results:
                evidence_summary += f"\nПодфакт: {sr['claim'][:80]} → {sr['verdict']}"

        try:
            prompt = EXPLANATION_TEMPLATE.format(
                claim=claim,
                verdict=verdict,
                evidence_summary=evidence_summary,
            )
            raw = self.explain_llm.invoke(prompt)
            explanation = StrOutputParser().invoke(raw)
            return explanation.strip()[:1500]
        except Exception as e:
            logger.warning(f"Explanation failed: {e}")
            return evidence_summary[:500]

    # ======================================================================
    # CONFIDENCE → SCORE mapping
    # ======================================================================

    @staticmethod
    def _confidence_to_score(verdict: str, confidence: int) -> int:
        """Map verdict + confidence to credibility score (0-100)."""
        if verdict == "ПРАВДА":
            return max(60, min(95, confidence))
        elif verdict == "ЛОЖЬ":
            return max(5, min(34, 100 - confidence))
        else:
            return 45

    # ======================================================================
    # MAIN CHECK FLOW
    # ======================================================================

    def check(self, claim: str) -> Dict[str, Any]:
        """Check a single claim. Returns structured result."""
        logger.info(f"Проверка: {claim[:120]}")
        print(f"\nПроверка: {claim}")

        # Cache check
        cached = self.fact_cache.get(claim)
        if cached:
            print(f"  [Cache] HIT — {cached['verdict']} (score={cached['credibility_score']})")
            return cached

        # Fresh context
        self._ctx = PipelineContext()
        total_start = time.time()

        try:
            # STAGE 1: PARSE
            claim_info = self._parse_claim(claim)
            print(f"  Тип: {claim_info['type']}, чисел: {len(claim_info['numbers'])}, "
                  f"скам: {claim_info.get('is_scam', False)}")

            # STAGE 2: DECOMPOSE
            sub_claims = self._decompose(claim)
            is_composite = len(sub_claims) > 1
            if is_composite:
                self._ctx.sub_claims = sub_claims
                print(f"  Декомпозиция: {len(sub_claims)} подфактов")
                for i, sc in enumerate(sub_claims, 1):
                    print(f"    {i}. {sc}")

            # Extract keywords (rule-based, LLM fallback)
            rule_kw = self._extract_keywords_rule_based(claim)
            keywords = self._parse_keywords(rule_kw)
            if len(keywords) < 2:
                try:
                    kw_prompt = PromptTemplate(template=KEYWORD_EXTRACTION_TEMPLATE, input_variables=["claim"])
                    llm_kw = (kw_prompt | self.keyword_llm | StrOutputParser()).invoke({"claim": claim})
                    keywords = self._parse_keywords(llm_kw)
                except Exception:
                    pass
            print(f"  Ключевые слова: {keywords}")

            # STAGE 3: SEARCH (global — shared across sub-claims)
            sources = self._search_sources(claim, keywords)
            self._ctx.raw_results = sources

            # Search per sub-claim if composite
            if is_composite:
                existing_urls = {r.get("link", "") for r in sources}
                for sc in sub_claims:
                    try:
                        sc_results = self.searcher.search_all_keywords([sc], claim=sc)
                        for r in sc_results:
                            url = r.get("link", "")
                            if url and url not in existing_urls:
                                sources.append(r)
                                existing_urls.add(url)
                    except Exception:
                        pass
                self._ctx.raw_results = sources

            # STAGE 4+5: EVIDENCE + DECIDE (per sub-claim)
            sub_results = []
            entities = self._ctx.claim_entities or []

            for sc in sub_claims:
                # Evidence signals
                wd_signal = self._check_wikidata(sc, entities)
                sc_claim_info = classify_claim(sc) if is_composite else claim_info
                num_signal = self._check_numbers(sc, sources, sc_claim_info)
                nli_signal, nli_scores = self._check_nli(sc, sources)
                debunk_count = self._check_debunk(sc, sources)

                # Tier 4: LLM knowledge
                # Вызываем если нет СТРОГИХ сигналов (wikidata/nli/debunk чистые).
                # num_signal=+1 НЕ блокирует: число может встречаться в ином контексте
                # (пример: '1943' есть в статьях о ВМВ, но не как дата окончания войны).
                llm_signal = 0
                ent = nli_scores.get("ent", 0)
                con = nli_scores.get("con", 0)
                nli_uncertain = abs(ent - con) < 0.30  # слабый или отсутствующий сигнал NLI
                nli_contradicts_no_wd = nli_signal == -1 and wd_signal == 0  # NLI-only без WD подтверждения
                if wd_signal == 0 and (nli_uncertain or nli_contradicts_no_wd):
                    llm_signal = self._check_llm_knowledge(sc)

                print(f"  [{sc[:50]}] WD={wd_signal:+d} NUM={num_signal:+d} "
                      f"NLI={nli_signal:+d} (ent={nli_scores['ent']:.2f}, con={nli_scores['con']:.2f}) "
                      f"debunk={debunk_count} LLM={llm_signal:+d}")

                # Decide
                verdict, confidence = self._decide(
                    wd_signal, num_signal, nli_signal, nli_scores,
                    self._ctx.is_scam, debunk_count, llm_signal)

                sub_results.append({
                    "claim": sc,
                    "verdict": verdict,
                    "confidence": confidence,
                    "status": verdict,  # for app.py sub_verdicts display
                    "wd_signal": wd_signal,
                    "num_signal": num_signal,
                    "nli_signal": nli_signal,
                    "nli_scores": nli_scores,
                    "debunk_count": debunk_count,
                })

            # STAGE 6: AGGREGATE
            if is_composite:
                final_verdict = self._aggregate(sub_results)
                # Confidence from worst sub-result
                final_confidence = min(sr["confidence"] for sr in sub_results)
            else:
                final_verdict = sub_results[0]["verdict"]
                final_confidence = sub_results[0]["confidence"]

            print(f"  VERDICT: {final_verdict} (confidence={final_confidence})")

            # STAGE 7: EXPLAIN
            explanation = self._explain(claim, final_verdict, sub_results, sources)

            # Build result
            credibility_score = self._confidence_to_score(final_verdict, final_confidence)

            # Ensemble features for app.py signal chips
            wd = self._ctx.wikidata_result
            wd_facts = wd.get("facts", []) if wd and wd.get("found") else []
            nli_result = self._ctx.nli_result or {}

            result = {
                "claim": claim,
                "credibility_score": credibility_score,
                "verdict": final_verdict,
                "confidence": final_confidence,
                "reasoning": explanation,
                "chain_of_thought": "",
                "sources": sources[:10],
                "sources_text": "\n".join(
                    f"- {s.get('title', '')}: {s.get('snippet', '')[:100]}"
                    for s in sources[:5]
                ),
                "keywords": keywords,
                "search_results_formatted": "",
                "raw_verdict": explanation,
                "self_critique": "",
                "self_critique_errors": "",
                "total_time": round(time.time() - total_start, 1),
                "sub_verdicts": [
                    {
                        "index": i + 1,
                        "claim": sr["claim"],
                        "status": sr["verdict"],
                        "citation": "",
                        "source": "",
                    }
                    for i, sr in enumerate(sub_results)
                ] if is_composite else [],
                "_composite": is_composite,
                "_ensemble_features": {
                    "wd_confirmed": sum(1 for f in wd_facts if f.get("match") is True),
                    "wd_contradicted": sum(1 for f in wd_facts if f.get("match") is False),
                    "nli_ent": nli_result.get("max_entailment", 0.0),
                    "nli_con": nli_result.get("max_contradiction", 0.0),
                    "nums_match": int(any(
                        c["match"] for c in self._ctx.num_comparisons if c.get("source_number")
                    )) if self._ctx.num_comparisons else 0,
                    "nums_mismatch": int(any(
                        not c["match"] for c in self._ctx.num_comparisons if c.get("source_number")
                    )) if self._ctx.num_comparisons else 0,
                    "num_sources": len(sources),
                },
                "_explanation": explanation if final_verdict == "НЕ УВЕРЕНА" else "",
            }

            # Cache result
            self.fact_cache.set(claim, result)

            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "claim": claim,
                "credibility_score": 50,
                "verdict": "НЕ УВЕРЕНА",
                "confidence": 0,
                "reasoning": f"Ошибка при анализе: {str(e)}",
                "chain_of_thought": "",
                "sources": [],
                "sources_text": "",
                "keywords": [],
                "search_results_formatted": "",
                "raw_verdict": "",
                "self_critique": "",
                "self_critique_errors": "",
                "total_time": round(time.time() - total_start, 1),
                "sub_verdicts": [],
                "_composite": False,
                "_ensemble_features": {},
                "_explanation": "",
            }
