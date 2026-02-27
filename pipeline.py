"""LangChain LCEL цепочка для проверки фактов.

v4: NLI-first ensemble — DeBERTa классифицирует, LLM объясняет.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import ModelConfig, PipelineConfig, SearchConfig
from prompts import (
    KEYWORD_EXTRACTION_TEMPLATE,
    CREDIBILITY_ASSESSMENT_TEMPLATE,
    CREDIBILITY_ASSESSMENT_REASONING_TEMPLATE,
    CREDIBILITY_ASSESSMENT_AUDIT_TEMPLATE,
    CREDIBILITY_ASSESSMENT_AUDIT_REASONING_TEMPLATE,
    SELF_CRITIQUE_TEMPLATE,
    FALLACY_DETECTION_TEMPLATE,
)
from model import load_base_model, load_finetuned_model, build_langchain_llm, is_grpo_adapter
from search import FactCheckSearcher, boost_factcheck_scores, validate_context_entities
from claim_parser import classify_claim, format_verification_hints, extract_numbers, compare_numbers
from fact_cache import FactCache
from utils import (
    TRANSLITERATION_MAP, STOPWORDS, COMMON_DETAIL_WORDS,
    NLI_STEM_TRANSLITERATION, check_locations_in_sources, location_found_in_text,
)
from source_credibility import get_credibility
from satire_detector import satire_penalty
from adversarial import AdversarialDebate

# V8: Natasha NER + pymorphy2 для русской морфологии
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

# Промпт для декомпозиции составных утверждений (Проблема 1)
_DECOMPOSE_TEMPLATE = """\
Разбей утверждение на атомарные факты. Каждый факт должен:
1. Содержать ровно ОДНО проверяемое утверждение
2. Быть понятным без оригинального контекста (восстанавливай подлежащее)
3. СОХРАНЯТЬ все отрицания ТОЧНО как в оригинале — НЕ добавлять "не"/"нет"/"без" если их нет в оригинале
4. НЕ менять смысл на противоположный
5. НЕ исправлять факты, даты, места, имена или числа — сохранять дословно
Если факт один — верни только его.

Утверждение: Норвегия входит в ЕС и использует евро
Факты:
Норвегия входит в ЕС
Норвегия использует евро

Утверждение: Страна входит в Содружество наций, а её официальная столица — Сидней
Факты:
Страна входит в Содружество наций
Официальная столица страны — Сидней

Утверждение: Tesla отозвала 2 млн авто, при этом акции компании упали на 5%
Факты:
Tesla отозвала 2 млн автомобилей
Акции Tesla упали на 5%

Утверждение: Река Амазонка протекает через Африку и впадает в Индийский океан
Факты:
Река Амазонка протекает через Африку
Река Амазонка впадает в Индийский океан

Утверждение: Путин подписал закон о повышении пенсий
Факты:
Путин подписал закон о повышении пенсий

Утверждение: {claim}
Факты:"""

# Месяцы и общие слова для программной очистки ключевых слов
_MONTHS_RU = {
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
}
_GENERIC_WORDS = STOPWORDS

# Известные многословные сущности (регулярные выражения)
# Если модель разбивает их на части — склеиваем обратно
_MULTI_WORD_ENTITIES = [
    (r"\bЦБ\b", r"\bРФ\b", "ЦБ РФ"),
    (r"\bМВД\b", r"\bРФ\b", "МВД РФ"),
    (r"\bМИД\b", r"\bРФ\b", "МИД РФ"),
    (r"\bФСБ\b", r"\bРФ\b", "ФСБ РФ"),
    (r"\bМинистр\b", r"\bобороны\b", "Министр обороны"),
    (r"\bСовет\b", r"\bФедерации\b", "Совет Федерации"),
    (r"\bГос\b", r"\bдума\b", "Госдума"),
    (r"\bключевая\b", r"\bставка\b", "ключевая ставка"),
]


@dataclass
class PipelineContext:
    """Per-request контекст. Создаётся заново для каждого вызова check().

    Заменяет thread-unsafe self._shared dict. Каждый запрос получает
    свой экземпляр — безопасно для multi-user (Streamlit).
    """
    search_hint: str = ""
    raw_results: List[Dict] = field(default_factory=list)
    wiki_results: List[Dict] = field(default_factory=list)
    nli_result: Optional[Dict] = None
    num_comparisons: List[Dict] = field(default_factory=list)
    claim_info: Optional[Dict] = None
    claim_locations: List[str] = field(default_factory=list)
    claim_entities: List[str] = field(default_factory=list)
    is_scam: bool = False
    date_mismatch: bool = False
    temporal_check: Dict = field(default_factory=dict)
    sub_claims: List[str] = field(default_factory=list)
    wikidata_result: Dict = field(default_factory=dict)
    sub_nli_results: List[Dict] = field(default_factory=list)  # V5: per-sub-claim NLI
    minicheck_result: Optional[Dict] = None  # V9: MiniCheck-FT5 verification


class FactCheckPipeline:
    """Пайплайн проверки достоверности утверждений.

    v4: NLI-first ensemble — DeBERTa классифицирует, LLM объясняет.
    """

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

        # Per-request context — создаётся заново в check()
        # Thread-safe: каждый запрос получает свой PipelineContext
        self._ctx = PipelineContext()

        # Fact cache (Redis, graceful degradation)
        self.fact_cache = FactCache()

        # Meta-classifier для ensemble verdict (если обучен)
        self._meta_classifier = None
        meta_path = os.path.join(os.path.dirname(__file__), "models", "meta_classifier.pkl")
        if os.path.exists(meta_path):
            try:
                import joblib
                self._meta_classifier = joblib.load(meta_path)
                print(f"Meta-classifier загружен из {meta_path}")
            except Exception as e:
                print(f"Meta-classifier не загружен: {e}")

        # Cross-encoder re-ranker (CPU, улучшает отбор источников для NLI)
        self._reranker = None
        if pipeline_config.enable_cross_encoder:
            try:
                from embeddings import ReRanker
                _ce_model = getattr(pipeline_config, 'cross_encoder_model', None)
                print(f"Загрузка CrossEncoder re-ranker ({_ce_model or 'default'})...")
                self._reranker = ReRanker(model_name=_ce_model) if _ce_model else ReRanker()
            except Exception as e:
                print(f"CrossEncoder не загружен: {e}")

        # MiniCheck-FT5 (CPU, ~1.7GB RAM — не занимает GPU)
        self.minicheck = None
        if getattr(pipeline_config, 'enable_minicheck', False):
            try:
                from minicheck_verifier import MiniCheckVerifier
                print("Загрузка MiniCheck-FT5 на CPU...")
                self.minicheck = MiniCheckVerifier(device="cpu")
            except Exception as e:
                print(f"MiniCheck не загружен: {e}")

        # NLI checker (CPU, не занимает GPU)
        self.nli_checker = None
        if pipeline_config.enable_nli:
            try:
                from nli_checker import NLIChecker
                print("Загрузка NLI модели на CPU...")
                self.nli_checker = NLIChecker(
                    device=pipeline_config.nli_device,
                    model_name=pipeline_config.nli_model_name,
                )
                print("NLI модель загружена.")
            except Exception as e:
                print(f"NLI модель не загружена: {e}")

        # Загрузка модели один раз
        if adapter_path and os.path.exists(adapter_path):
            adapter_type = "GRPO" if self._use_reasoning_template else "SFT"
            print(f"Загрузка {adapter_type} модели из {adapter_path}...")
            model, tokenizer = load_finetuned_model(adapter_path, model_config)
        else:
            print("Загрузка base модели...")
            model, tokenizer = load_base_model(model_config)

        # Три LLM-wrapper с разными max_new_tokens
        self.keyword_llm = build_langchain_llm(
            model, tokenizer,
            max_new_tokens=pipeline_config.keyword_max_new_tokens,
            pipeline_config=pipeline_config,
        )
        self.verdict_llm = build_langchain_llm(
            model, tokenizer,
            max_new_tokens=pipeline_config.verdict_max_new_tokens,
            pipeline_config=pipeline_config,
        )
        self.critique_llm = build_langchain_llm(
            model, tokenizer,
            max_new_tokens=pipeline_config.critique_max_new_tokens,
            pipeline_config=pipeline_config,
        )

        # Лог: какой адаптер и какой шаблон используется
        adapter_type = "GRPO" if self._use_reasoning_template else ("SFT" if adapter_path else "Base")
        template_name = "CREDIBILITY_ASSESSMENT_REASONING_TEMPLATE (<reasoning>/<answer>)" if self._use_reasoning_template else "CREDIBILITY_ASSESSMENT_TEMPLATE (стандартный)"
        print(f"  [Pipeline] Адаптер: {adapter_type}")
        print(f"  [Pipeline] Шаблон: {template_name}")
        if not self._use_reasoning_template:
            print("  [Pipeline] ВНИМАНИЕ: GRPO адаптер не найден — Chain-of-Thought отключён")

        # Подключаем Mistral к маршрутизатору запросов (QueryClassifier)
        # keyword_llm уже загружен → передаём как generate_fn для 1-токенной классификации
        self.searcher.set_generate_fn(lambda p: self.keyword_llm.invoke(p))

        # Сборка цепочки
        self.chain = self._build_chain()

    # ----------------------------------------------------------
    # Entity extraction & filtering (shared by Fixes 3, 4, 6)
    # ----------------------------------------------------------

    # Common Russian words — используем общий набор из utils.py
    _COMMON_WORDS = STOPWORDS

    def _extract_claim_entities(self, claim: str, keywords: List[str] = None) -> List[str]:
        """Extract key named entities from claim text and/or parsed keywords.

        V8: Uses Natasha NER as primary source (PER, LOC, ORG),
        falls back to regex for non-Russian text.
        Returns lowercased entity strings for matching.
        """
        entities = []

        # V8: Natasha NER — primary source for Russian entities
        if _HAS_NATASHA:
            try:
                ner_entities = extract_entities(claim)
                for ent in ner_entities:
                    entities.append(ent["normal"])
                    # Also add original text form for exact matching
                    entities.append(ent["text"].lower())
            except Exception:
                pass  # fallback to regex below

        # 1. Capitalized words from claim (works when user types properly)
        for m in re.finditer(r'[А-ЯЁA-Z][а-яёa-zA-Z]{2,}', claim):
            word = m.group(0)
            idx = m.start()
            if idx == 0 or claim[idx - 1] in '.!?\n':
                continue
            entities.append(word.lower())

        # 2. Entities from LLM-extracted keywords (works for lowercase input!)
        # Keywords like ['Валентин Стрикало', 'прекращение карьеры'] contain
        # proper nouns even when the original claim is all-lowercase.
        if keywords:
            for kw in keywords:
                # Multi-word keyword with capitalized parts = likely a name entity
                # Store as full phrase for better precision ("валентин стрикало"
                # won't match "валентина толкунова")
                caps = re.findall(r'[А-ЯЁA-Z][а-яёa-zA-Z]{2,}', kw)
                if len(caps) >= 2:
                    # Full multi-word entity (e.g. "Валентин Стрикало")
                    entities.append(kw.lower().strip())
                for c in caps:
                    entities.append(c.lower())

        # 3. Quoted terms from claim
        for m in re.finditer(r'[«"]([^»"]+)[»"]', claim):
            entities.append(m.group(1).lower())

        # 4. ALWAYS extract distinctive words from claim text
        # Critical for lowercase input where keywords may have LLM typos
        # (e.g. LLM writes "Стрикало" but claim has correct "стрыкало")
        for w in re.findall(r'[а-яёa-z]{4,}', claim.lower()):
            if w not in self._COMMON_WORDS:
                entities.append(w)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique

    @staticmethod
    def _prefilter_by_entities(
        results: List[Dict], entities: List[str], min_docs: int = 5
    ) -> List[Dict]:
        """Remove results where no distinctive claim entities appear in title+snippet.

        Uses distinctive entities (multi-word or 5+ chars) for filtering to avoid
        false matches like "валентин" matching Толкунова/Талызина instead of Стрыкало.
        Keeps at least min_docs results (falls back to unfiltered if too aggressive).
        """
        if not entities:
            return results

        # Prefer distinctive entities for filtering
        distinctive = [e for e in entities if ' ' in e or len(e) >= 5]
        filter_ents = distinctive if distinctive else entities

        def _has_entity(r: Dict) -> bool:
            text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
            return any(e in text for e in filter_ents)

        filtered = [r for r in results if _has_entity(r)]
        if len(filtered) >= min_docs:
            dropped = len(results) - len(filtered)
            if dropped:
                print(f"  [EntityPreFilter] Removed {dropped} results with 0 distinctive entity matches")
            return filtered
        return results

    @staticmethod
    def _extract_keywords_rule_based(claim: str) -> str:
        """B1: Rule-based keyword extraction — replaces LLM step (~15s savings).

        Extracts: capitalized words (names, places), numbers with context,
        quoted terms, distinctive long words. Returns comma-separated string.
        """
        keywords = []

        # 1. Quoted terms
        for m in re.finditer(r'[«"]([^»"]+)[»"]', claim):
            keywords.append(m.group(1))

        # 2. Capitalized multi-word entities (names, places)
        for m in re.finditer(r'[А-ЯЁA-Z][а-яёa-zA-Z]+(?:\s+[А-ЯЁA-Z][а-яёa-zA-Z]+)+', claim):
            keywords.append(m.group(0))

        # 3. Individual capitalized words (not at sentence start)
        words = claim.split()
        for i, w in enumerate(words):
            if i == 0:
                continue
            # Check if word starts with uppercase
            clean = w.strip('.,!?;:«»"()-')
            if clean and clean[0].isupper() and len(clean) >= 3:
                if clean.lower() not in STOPWORDS:
                    keywords.append(clean)

        # 4. Numbers with context
        for m in re.finditer(r'\d+(?:[.,]\d+)?(?:\s*(?:%|млн|млрд|тыс|миллион|миллиард|трлн))', claim):
            keywords.append(m.group(0))

        # 5. Years
        for m in re.finditer(r'\b((?:19|20)\d{2})\b', claim):
            keywords.append(m.group(1))

        # 6. Distinctive long words (>= 6 chars, not stopwords)
        for w in re.findall(r'[а-яёa-z]{6,}', claim.lower()):
            if w not in STOPWORDS and w not in [k.lower() for k in keywords]:
                keywords.append(w)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for k in keywords:
            kl = k.lower().strip()
            if kl and kl not in seen:
                seen.add(kl)
                unique.append(k)

        return ", ".join(unique[:7])

    def _build_chain(self):
        """Сборка LCEL цепочки с прогрессивным обогащением состояния.

        Цепочка:
        claim → +keywords_raw → +search_results → +search_hint,verification_hints
              → +nli_hint,number_comparison → +verdict [→ +self_critique]
        """
        keyword_prompt = PromptTemplate(
            template=KEYWORD_EXTRACTION_TEMPLATE,
            input_variables=["claim"],
        )
        # GRPO модель обучена с <reasoning>/<answer> форматом — используем соответствующий шаблон
        verdict_template = (
            CREDIBILITY_ASSESSMENT_REASONING_TEMPLATE
            if self._use_reasoning_template
            else CREDIBILITY_ASSESSMENT_TEMPLATE
        )
        verdict_prompt = PromptTemplate(
            template=verdict_template,
            input_variables=["claim_structured", "search_results", "search_hint",
                             "verification_hints", "nli_hint", "number_comparison"],
        )

        # B1: Rule-based keyword extraction with LLM fallback
        _llm_keyword_chain = keyword_prompt | self.keyword_llm | StrOutputParser()

        def keyword_step(state: dict) -> str:
            claim = state.get("claim", "")
            rule_kw = self._extract_keywords_rule_based(claim)
            # Fallback to LLM only if < 2 keywords
            if len([k.strip() for k in rule_kw.split(",") if k.strip()]) < 2:
                print("  [B1] Rule-based < 2 keywords, falling back to LLM")
                return _llm_keyword_chain.invoke(state)
            print(f"  [B1] Rule-based keywords: {rule_kw}")
            return rule_kw

        keyword_chain = RunnableLambda(keyword_step)

        # Шаг 2: Поиск новостей — сохраняет hint и raw_results на self
        def search_step(state: dict) -> str:
            keywords = self._parse_keywords(state["keywords_raw"])
            claim = state.get("claim", "")
            keywords = self._validate_keywords(keywords, claim)
            print(f"  Ключевые слова: {keywords}")

            # V8: Wikipedia as PRIMARY source (before DDG)
            # Энциклопедические факты проверяются через Wikipedia ПЕРВЫМИ.
            # DDG — только для новостей и актуальных событий.
            claim_entities_for_wiki = self._extract_claim_entities(claim, keywords=keywords)

            # V8: Natasha NER entities — более точные имена для Wikipedia lookup
            _wiki_entities = list(claim_entities_for_wiki)
            if _HAS_NATASHA:
                try:
                    ner_names = extract_entity_names(claim)
                    for name in ner_names:
                        if name.lower() not in [e.lower() for e in _wiki_entities]:
                            _wiki_entities.insert(0, name)
                except Exception:
                    pass

            wiki_entity_results = self.searcher.wiki_entity_lookup(
                _wiki_entities + keywords[:3]
            )

            # V8: Wikipedia PRIMARY — начинаем с Wikipedia, DDG дополняет
            results = list(wiki_entity_results)  # Wikipedia первая
            existing_urls = {r.get("link", "") for r in results}

            # DDG дополняет (новости, актуальные события)
            ddg_results = self.searcher.search_all_keywords(keywords, claim=claim)
            for dr in ddg_results:
                url = dr.get("link", "")
                if url and url not in existing_urls:
                    results.append(dr)
                    existing_urls.add(url)
                elif url in existing_urls:
                    # Обновляем snippet если DDG дал более полный текст
                    for ex in results:
                        if ex.get("link") == url:
                            if len(dr.get("snippet", "")) > len(ex.get("snippet", "")):
                                ex["snippet"] = dr["snippet"]
                            break

            # --- V5: Verification query search (Task 4) ---
            # Генерируем целевые вопросы для проверки ("кто основал Microsoft?")
            _generate_fn = None
            try:
                _generate_fn = lambda p: self.keyword_llm.invoke(p)
            except Exception:
                pass
            verif_queries = self.searcher.generate_verification_queries(
                claim, claim_entities_for_wiki, generate_fn=_generate_fn
            )
            if verif_queries:
                print(f"  [VerifQuery] Запросы: {verif_queries[:4]}")
                verif_results = self.searcher.search_verification_queries(verif_queries)
                for vr in verif_results:
                    url = vr.get("link", "")
                    if url and url not in existing_urls:
                        results.append(vr)
                        existing_urls.add(url)

            # --- V5: Multi-hop counter-entity search (Task 5) ---
            counter_results = self.searcher.search_counter_entities(claim, claim_entities_for_wiki)
            for cr in counter_results:
                url = cr.get("link", "")
                if url and url not in existing_urls:
                    results.append(cr)
                    existing_urls.add(url)

            wiki_cnt = sum(1 for r in results if 'wikipedia.org' in r.get('link', ''))
            print(f"  Найдено новостей: {len(results)} (Wikipedia: {wiki_cnt})")

            # Сохраняем Wikipedia ДО ранжирования — semantic ranker может их отсечь
            self._ctx.wiki_results = [r for r in results if 'wikipedia.org' in r.get('link', '')]

            # Дополнительный поиск для каждого под-утверждения (Проблема 1)
            sub_claims = self._ctx.sub_claims

            def _is_trivial_subclaim(sc: str) -> bool:
                """Отсеивает неверифицируемые под-утверждения."""
                # V8: {3,} вместо {4,} — не отсекать 3-буквенные слова (мёд, рак, мяч, рис)
                words = re.findall(r'[а-яёa-z]{3,}', sc.lower())
                if len(words) < 3:
                    return True
                # Нет именованных сущностей (заглавные буквы) и нет кавычек
                named = re.findall(r'[А-ЯЁ][а-яё]{3,}', sc)
                quoted = re.findall(r'«[^»]+»|"[^"]+"|«[^»]+»', sc)
                numbers = re.findall(r'\d+', sc)
                if not named and not quoted and not numbers and len(sc) < 30:
                    return True
                return False

            if len(sub_claims) > 1:
                existing_urls = {r.get("link", "") for r in results}
                for sc in sub_claims:
                    if _is_trivial_subclaim(sc):
                        print(f"  [Skip] Тривиальный sub-claim: '{sc[:50]}' → пропуск поиска")
                        continue
                    sc_results = self.searcher.search_all_keywords([sc], claim=sc)
                    for r in sc_results:
                        url = r.get("link", "")
                        if url and url not in existing_urls:
                            results.append(r)
                            existing_urls.add(url)
                print(f"  + под-утверждения: {len(results)} источников итого")

            # Targeted Wikipedia search per sub-claim (биографические факты)
            if sub_claims and len(sub_claims) > 1:
                BIO_KEYWORDS = ["похоронен", "умер", "родился", "написал", "создал",
                                "изобрел", "открыл", "получил", "завещал", "женился",
                                "развёлся", "учился", "окончил", "основал"]
                existing_urls = {r.get("link", "") for r in results}
                for sc in sub_claims:
                    if any(kw in sc.lower() for kw in BIO_KEYWORDS):
                        wiki_hits = self.searcher._search_wikipedia_with_extract(sc[:80])
                        for hit in wiki_hits:
                            if hit.get("link", "") not in existing_urls:
                                results.append(hit)
                                existing_urls.add(hit.get("link", ""))
                if len(results) > len(existing_urls):
                    print(f"  + Wikipedia по биофактам: {len(results)} итого")

            # Ранжирование по релевантности (bi-encoder)
            results = self.searcher.rank_by_relevance(claim, results)
            wiki_in_top = sum(1 for r in results if 'wikipedia.org' in r.get('link', ''))
            print(f"  После ранжирования: {len(results)} релевантных (Wikipedia в топ: {wiki_in_top}, топ-7 → модели)")

            # Entity pre-filter: remove results where zero claim entities appear
            claim_entities = self._extract_claim_entities(claim, keywords=keywords)
            self._ctx.claim_entities = claim_entities
            if claim_entities:
                results = self._prefilter_by_entities(results, claim_entities, min_docs=5)

            # Cross-encoder re-ranking поверх bi-encoder кандидатов (Проблема 3)
            if self._reranker is not None and results:
                results = self._reranker.rerank(claim, results, top_k=min(7, len(results)))

                # Direction 1: бустинг фактчек/debunk-источников (post-CrossEncoder).
                # Гарантирует, что статьи-разоблачения вытесняют лайфстайл-новости на Топ-1.
                results = boost_factcheck_scores(results)

                # Direction 2: entity-match penalty + filter.
                # Решает Semantic Overgeneralization: если из факта пропало ключевое слово
                # («математика») в контексте (только «вступительные экзамены») —
                # сначала снижаем скор пропорционально, затем фильтруем если penalty > порога.
                ENTITY_PENALTY_WEIGHT = 1.5   # снижение скора * penalty (0.0–1.0)
                ENTITY_FILTER_THRESHOLD = 0.35  # >35% ключевых слов факта отсутствует → кандидат на дроп
                MIN_DOCS_AFTER_FILTER = 3       # не фильтруем если остаётся < N документов

                any_penalized = False
                for r in results:
                    ctx = r.get("title", "") + " " + r.get("snippet", "")
                    penalty = validate_context_entities(claim, ctx)
                    r["_entity_penalty"] = penalty  # кэшируем для фильтрации ниже
                    if penalty > 0.0:
                        r["cross_encoder_score"] = (
                            r.get("cross_encoder_score", r.get("semantic_score", 0.0))
                            - penalty * ENTITY_PENALTY_WEIGHT
                        )
                        any_penalized = True

                if any_penalized:
                    # Пересортировка после entity-штрафа (Dir1 boost уже применён)
                    results.sort(
                        key=lambda x: x.get("cross_encoder_score", x.get("semantic_score", 0.0)),
                        reverse=True,
                    )
                    # Фильтрация: удаляем документы с высоким entity-штрафом,
                    # если у нас достаточно альтернатив с лучшим покрытием.
                    _good = [r for r in results if r.get("_entity_penalty", 0) <= ENTITY_FILTER_THRESHOLD]
                    if len(_good) >= MIN_DOCS_AFTER_FILTER:
                        _dropped = len(results) - len(_good)
                        if _dropped:
                            print(f"  [EntityFilter] Отфильтровано {_dropped} контекстов "
                                  f"(entity_penalty > {ENTITY_FILTER_THRESHOLD})")
                        results = _good

                # Убираем внутреннее поле перед передачей дальше
                for r in results:
                    r.pop("_entity_penalty", None)

                for r in results:
                    r["final_score"] = r.get("cross_encoder_score", r.get("semantic_score", 0))
                print(f"  После cross-encoder: {len(results)} источников")

            # Нейтральная подсказка — НЕ предрешаем за модель
            total = len(results)
            if not results:
                hint = (
                    "ВАЖНО: По запросу не найдено источников. "
                    "Ты не можешь подтвердить или опровергнуть утверждение. "
                    "Вердикт должен быть НЕ ПОДТВЕРЖДЕНО."
                )
            else:
                hint = (
                    f"ВАЖНО: Найдено {total} источников. Прочитай каждый "
                    f"ВНИМАТЕЛЬНО. Совпадение отдельных слов (президент, указ, "
                    f"подписал) НЕ означает подтверждение — источник должен "
                    f"описывать ТОЧНО ТО ЖЕ событие, что и утверждение."
                )

            # Сохраняем в thread-local (safe для multi-user)
            self._ctx.search_hint = hint
            self._ctx.raw_results = results

            # Передаём LLM только TOP-5 источников — меньше шума,
            # точнее reasoning. Полный набор остаётся в raw_results для NLI/ensemble.
            return self.searcher.format_results(results[:5])

        # Шаг 2.5: Извлечение search_hint из thread-local (отдельный assign)
        def get_search_hint(state: dict) -> str:
            return self._ctx.search_hint

        # Шаг 2.6: Классификация утверждения и генерация подсказок для проверки
        def get_verification_hints(state: dict) -> str:
            from claim_parser import detect_scam_patterns
            claim = state.get("claim", "")
            claim_info = classify_claim(claim)
            self._ctx.claim_info = claim_info
            self._ctx.claim_locations = claim_info.get("locations", [])
            self._ctx.is_scam = claim_info.get("is_scam", False)
            hints = format_verification_hints(claim_info)
            
            # Проверяем локации из claim в уже найденных источниках
            raw_results = self._ctx.raw_results
            claim_locs = claim_info.get("locations", [])
            if claim_locs and raw_results:
                loc_check = check_locations_in_sources(claim_locs, raw_results)

                if loc_check["all_confirmed"]:
                    conf_str = ", ".join(loc_check["confirmed"])
                    hints += (f"\nИНФО: локация [{conf_str}] "
                             f"присутствует в источниках.")
                elif loc_check["all_missing"]:
                    miss_str = ", ".join(loc_check["missing"])
                    hints += (f"\nИНФО: локация [{miss_str}] "
                             f"не найдена в источниках — возможно расхождение.")

            # Temporal mismatch check
            from claim_parser import detect_temporal_mismatch
            temporal = detect_temporal_mismatch(claim, raw_results)
            self._ctx.temporal_check = temporal
            if temporal.get("temporal_mismatch"):
                hints += f"\n{temporal['hint']}"

            # V5: Wikidata SPARQL для структурированных фактов (Task 6)
            try:
                from wikidata import check_structured_facts, format_wikidata_hint
                claim_entities = self._ctx.claim_entities or []
                wikidata_result = check_structured_facts(claim, claim_entities)
                self._ctx.wikidata_result = wikidata_result
                wiki_hint = format_wikidata_hint(wikidata_result)
                if wiki_hint:
                    hints += f"\n\n{wiki_hint}"
            except Exception as e:
                print(f"  [Wikidata] Ошибка: {e}")
                self._ctx.wikidata_result = {"found": False, "facts": [], "snippet": ""}

            if claim_info["type"] or hints:
                print(f"  Тип утверждения: {claim_info['type']}, "
                      f"чисел: {len(claim_info['numbers'])}, дат: {len(claim_info['dates'])}, "
                      f"локаций: {len(claim_locs)}, скам: {claim_info.get('is_scam', False)}")

            return hints

        # Шаг 2.7: NLI анализ — DeBERTa классифицирует claim vs TOP релевантных sources
        def nli_analysis_step(state: dict) -> str:
            if self.nli_checker is None:
                self._ctx.nli_result = None
                return ""

            raw_results = self._ctx.raw_results
            if not raw_results:
                print("  NLI: 0 источников после фильтрации — пропуск")
                self._ctx.nli_result = {
                    "max_entailment": 0.0, "max_contradiction": 0.0,
                    "entailment_count": 0, "contradiction_count": 0,
                }
                return "NLI АНАЛИЗ: источники не найдены."

            # ВАЖНО: передаём в NLI только TOP-5 по semantic_score.
            # 7 вредит: шумные 6-7 источники создают ложные contradiction для TRUE claims.
            # Pre-filter: отбрасываем источники без overlap с claim (спам: Facebook, otto, etc.)
            # Метод: извлекаем 5-char стемы из claim, требуем хотя бы 1 совпадение
            claim_stems = set()
            for _w in re.findall(r"[а-яёА-ЯЁa-zA-Z]{5,}", state.get("claim", "")):
                claim_stems.add(_w.lower()[:5])
            _expanded_stems = set(claim_stems)
            for _s, _alts in NLI_STEM_TRANSLITERATION.items():
                if _s in claim_stems:
                    _expanded_stems.update(_alts)

            def _has_claim_overlap(r: dict) -> bool:
                _txt = (r.get("snippet", "") + " " + r.get("title", "")).lower()
                return any(_st in _txt for _st in _expanded_stems)

            _relevant = [r for r in raw_results if _has_claim_overlap(r)]
            # Если релевантных достаточно — используем их, иначе fallback на всё
            _pool = _relevant if len(_relevant) >= 3 else raw_results

            # Wikipedia в NLI — ТОЛЬКО для локационных утверждений с явным подтверждением
            # года И локации в сниппете/заголовке. Это предотвращает:
            # 1) Ложные entailment для поддельных дат (Токио 2024 vs реальная статья о Париже 2024)
            # 2) Нарушение KEY_DETAIL_MISSING для PERSON-type (Цукерберг/Tesla)
            _wiki_all = self._ctx.wiki_results
            _wiki_relevant = []
            _claim_type = (self._ctx.claim_info or {}).get("type", "general")
            _claim_locations = self._ctx.claim_locations
            if _claim_locations and _claim_type in ("date", "date_loc"):
                # Строгий фильтр: год И локация должны быть в тексте статьи
                _claim_years = re.findall(r'\b(?:19|20)\d{2}\b', state.get("claim", ""))
                _loc_stems_strict = set()
                for _loc in _claim_locations:
                    for _w in re.findall(r"[а-яёА-ЯЁa-zA-Z]{4,}", _loc):
                        _loc_stems_strict.add(_w.lower()[:5])
                for _s, _alts in NLI_STEM_TRANSLITERATION.items():
                    if _s in _loc_stems_strict:
                        _loc_stems_strict.update(_alts)
                def _wiki_year_loc_match(r: dict) -> bool:
                    _t = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                    _year_ok = (not _claim_years) or any(_yr in _t for _yr in _claim_years)
                    _loc_ok = any(_ls in _t for _ls in _loc_stems_strict)
                    return _year_ok and _loc_ok
                _wiki_relevant = [r for r in _wiki_all if _wiki_year_loc_match(r)]

            if _wiki_relevant:
                _non_wiki = [r for r in _pool if 'wikipedia.org' not in r.get('link', '')]
                _non_wiki_sorted = sorted(
                    _non_wiki,
                    key=lambda x: x.get("final_score", x.get("semantic_score", 0)),
                    reverse=True,
                )[:max(2, 5 - len(_wiki_relevant[:3]))]
                nli_sources = _wiki_relevant[:3] + _non_wiki_sorted
                print(f"  NLI: Wikipedia с подтверждением года+локации ({len(_wiki_relevant[:3])} ст.)")
            else:
                nli_sources = sorted(
                    _pool,
                    key=lambda x: x.get("final_score", x.get("semantic_score", 0)),
                    reverse=True,
                )[:5]

            # NLI остаётся на CPU — перемещение GPU↔CPU каждый вызов
            # добавляет ~200ms задержки без пользы (GPU нужен для LLM)

            # Entity-aware NLI filtering: отбрасываем источники без entity coverage
            _pool_for_nli = []
            for r in nli_sources:
                ctx = r.get("snippet", "") + " " + r.get("title", "")
                penalty = validate_context_entities(state.get("claim", ""), ctx)
                if penalty <= 0.50:  # Мягче чем для LLM (0.35)
                    _pool_for_nli.append(r)
            if len(_pool_for_nli) >= 2:
                nli_sources = _pool_for_nli

            # NLI input quality guard: prefer sources matching distinctive entities.
            # "валентин" matches Толкунова/Талызина/Стрыкало — too broad.
            # "стрыкало" only matches Стрыкало — distinctive.
            _claim_entities = self._ctx.claim_entities
            if _claim_entities:
                # Distinctive = multi-word phrases or words 5+ chars from claim
                _distinctive = [e for e in _claim_entities
                                if ' ' in e or len(e) >= 5]
                # Use distinctive entities if available, otherwise all
                _filter_entities = _distinctive if _distinctive else _claim_entities

                def _entity_match_score(r):
                    _t = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                    return sum(1 for e in _filter_entities if e in _t)

                # Reorder NLI sources: those matching distinctive entities first
                nli_sources.sort(
                    key=lambda r: (-_entity_match_score(r),
                                   -r.get("final_score", r.get("semantic_score", 0))),
                )

                # Skip NLI if no sources match distinctive entities
                _matched = [r for r in nli_sources if _entity_match_score(r) > 0]
                if len(_matched) < 2:
                    print(f"  NLI: only {len(_matched)} sources match distinctive entities — skipping NLI")
                    self._ctx.nli_result = {
                        "max_entailment": 0.0, "max_contradiction": 0.0,
                        "entailment_count": 0, "contradiction_count": 0,
                    }
                    return "NLI АНАЛИЗ: источники не содержат ключевых сущностей из утверждения — NLI пропущен."

            # A1: Satire filtering — exclude or penalize satirical sources BEFORE NLI
            _pre_satire = len(nli_sources)
            _non_satire = []
            for r in nli_sources:
                _title = r.get("title", "")
                _domain = r.get("source", "")
                _sp = satire_penalty(_title, _domain)
                r["_satire_penalty"] = _sp
                if _sp < 0.5:
                    _non_satire.append(r)
            if len(_non_satire) >= 2:
                nli_sources = _non_satire
                if _pre_satire != len(nli_sources):
                    print(f"  NLI: excluded {_pre_satire - len(nli_sources)} satirical sources")

            # A1: Source credibility — annotate each source for weighted NLI
            for r in nli_sources:
                r["_credibility"] = get_credibility(r.get("link", ""))

            claim = state.get("claim", "")
            nli_result = self.nli_checker.check_claim(claim, nli_sources, snippet_key="snippet")

            # A1: Apply credibility weighting to NLI pair results
            if nli_result.get("pairs"):
                for i, pair in enumerate(nli_result["pairs"]):
                    cred = nli_sources[i].get("_credibility", 0.5) if i < len(nli_sources) else 0.5
                    pair["credibility"] = cred
                    pair["weighted_entailment"] = pair["entailment"] * cred
                    pair["weighted_contradiction"] = pair["contradiction"] * cred

                # Recount using credibility-weighted scores
                w_ent_scores = [p["weighted_entailment"] for p in nli_result["pairs"]]
                w_con_scores = [p["weighted_contradiction"] for p in nli_result["pairs"]]
                if w_ent_scores:
                    nli_result["max_entailment"] = round(max(w_ent_scores), 4)
                if w_con_scores:
                    nli_result["max_contradiction"] = round(max(w_con_scores), 4)

            self._ctx.nli_result = nli_result

            # V5: Per-sub-claim NLI (Task 7) — sharper signals for compound claims
            sub_claims = self._ctx.sub_claims
            if len(sub_claims) > 1 and nli_sources:
                sub_nli_results = []
                for sc in sub_claims:
                    sc_nli = self.nli_checker.check_claim(sc, nli_sources[:5], snippet_key="snippet")
                    sub_nli_results.append({
                        "sub_claim": sc,
                        "max_entailment": sc_nli.get("max_entailment", 0),
                        "max_contradiction": sc_nli.get("max_contradiction", 0),
                        "entailment_count": sc_nli.get("entailment_count", 0),
                        "contradiction_count": sc_nli.get("contradiction_count", 0),
                    })
                self._ctx.sub_nli_results = sub_nli_results
                print(f"  NLI per sub-claim:")
                for i, snr in enumerate(sub_nli_results, 1):
                    print(f"    SC{i}: ent={snr['max_entailment']:.2f} "
                          f"con={snr['max_contradiction']:.2f} | {snr['sub_claim'][:50]}")

                # Upgrade main NLI with sub-claim max signals
                sc_max_ent = max(snr["max_entailment"] for snr in sub_nli_results)
                sc_max_con = max(snr["max_contradiction"] for snr in sub_nli_results)
                if sc_max_ent > nli_result["max_entailment"]:
                    nli_result["max_entailment"] = round(sc_max_ent, 4)
                if sc_max_con > nli_result["max_contradiction"]:
                    nli_result["max_contradiction"] = round(sc_max_con, 4)

            ent = nli_result["max_entailment"]
            con = nli_result["max_contradiction"]
            ent_c = nli_result["entailment_count"]
            con_c = nli_result["contradiction_count"]
            print(f"  NLI: entailment={ent:.2f} ({ent_c} src), contradiction={con:.2f} ({con_c} src)")
            # DEBUG: показываем TOP-5 источники и их NLI-оценки
            for _i, _src in enumerate(nli_sources):
                _sc = _src.get("final_score", _src.get("semantic_score", 0))
                print(f"    NLI src {_i+1}: [{_sc:.2f}] {_src.get('source','')} | {_src.get('title','')[:60]}")

            # V5: Добавляем per-sub-claim NLI в hint
            _sub_nli_hint = ""
            if self._ctx.sub_nli_results:
                _sub_parts = []
                for i, snr in enumerate(self._ctx.sub_nli_results, 1):
                    sc_ent = snr["max_entailment"]
                    sc_con = snr["max_contradiction"]
                    if sc_con >= 0.50:
                        _sub_parts.append(f"  Пункт {i}: ВОЗМОЖНО ЛОЖЬ (con={sc_con:.2f})")
                    elif sc_ent >= 0.50:
                        _sub_parts.append(f"  Пункт {i}: ВОЗМОЖНО ПРАВДА (ent={sc_ent:.2f})")
                    else:
                        _sub_parts.append(f"  Пункт {i}: НЕЯСНО (ent={sc_ent:.2f}, con={sc_con:.2f})")
                if _sub_parts:
                    _sub_nli_hint = "\nNLI по пунктам:\n" + "\n".join(_sub_parts)

            # NLI hint как SUGGESTION, не как proof — LLM должен проверить сам
            if con >= 0.80 and ent < 0.35:
                return (f"NLI-ПОДСКАЗКА (проверь сам по источникам): автоанализ выявил "
                        f"возможное противоречие ({con_c} ист., con={con:.2f}). "
                        f"Это НЕ окончательный вывод — сверь с текстом источников."
                        + _sub_nli_hint)
            elif ent >= 0.70 and con < 0.30:
                return (f"NLI-ПОДСКАЗКА (проверь сам по источникам): автоанализ выявил "
                        f"возможное подтверждение ({ent_c} ист., ent={ent:.2f}). "
                        f"Это НЕ окончательный вывод — сверь с текстом источников."
                        + _sub_nli_hint)
            elif con >= 0.60 or ent >= 0.60:
                return (f"NLI-ПОДСКАЗКА: смешанные сигналы "
                        f"(ent={ent:.2f}×{ent_c}, con={con:.2f}×{con_c}). "
                        f"Полагайся на контент источников."
                        + _sub_nli_hint)
            else:
                return (f"NLI-ПОДСКАЗКА: слабые сигналы "
                        f"(ent={ent:.2f}, con={con:.2f}). "
                        f"Источники не дают чёткого сигнала."
                        + _sub_nli_hint)

        # Шаг 2.8: Детерминистическая проверка чисел И дат — claim vs sources
        def number_check_step(state: dict) -> str:
            from claim_parser import extract_dates
            claim = state.get("claim", "")
            claim_numbers = extract_numbers(claim)
            # Фильтр: маленькие числа (< 10) типа "number" — это шум
            # ("1 января" → 1, "2 раза" → 2), не годятся для фактчекинга
            claim_numbers = [n for n in claim_numbers
                            if not (n["type"] == "number" and n["value"] < 10)]
            
            # Также проверяем даты из claim
            claim_dates = extract_dates(claim)
            claim_years = [d["year"] for d in claim_dates if "year" in d]
            
            raw_results = self._ctx.raw_results
            date_hint = ""
            
            if claim_years and raw_results:
                # Собираем все годы из источников
                source_text_all = " ".join(
                    (r.get("snippet", "") + " " + r.get("title", ""))
                    for r in raw_results[:7]
                )
                source_dates = extract_dates(source_text_all)
                source_years = set(d["year"] for d in source_dates if "year" in d)
                
                for cy in claim_years:
                    if 1900 <= cy <= 2100:
                        if cy not in source_years and source_years:
                            near_years = [y for y in source_years if abs(y - cy) <= 5]
                            if near_years:
                                date_hint = (
                                    f"ПРОВЕРКА ДАТ: утверждение упоминает {cy} год, "
                                    f"но источники говорят о {sorted(near_years)[:3]} — "
                                    f"возможное расхождение в дате."
                                )
                                self._ctx.date_mismatch = True
                                print(f"  Дата: claim={cy}, sources={sorted(near_years)[:3]} — РАСХОЖДЕНИЕ")
                            else:
                                self._ctx.date_mismatch = False
                        else:
                            self._ctx.date_mismatch = False
            
            if not claim_numbers:
                self._ctx.num_comparisons = []

                # Добавляем location confirmation hint если локации найдены в источниках
                claim_locations = self._ctx.claim_locations
                if claim_locations and raw_results:
                    loc_check = check_locations_in_sources(claim_locations, raw_results)
                    if loc_check["all_confirmed"]:
                        loc_hint = (f"ИНФО: место «{', '.join(loc_check['confirmed'])}» "
                                   f"найдено в источниках.")
                        return (date_hint + " " + loc_hint).strip() if date_hint else loc_hint
                return date_hint

            raw_results = self._ctx.raw_results
            all_comparisons = []
            for source in raw_results:
                snippet = source.get("snippet", "")
                source_numbers = extract_numbers(snippet)
                if source_numbers:
                    comps = compare_numbers(claim_numbers, source_numbers)
                    all_comparisons.extend(comps)
            self._ctx.num_comparisons = all_comparisons

            mismatches = [c for c in all_comparisons if c["source_number"] and not c["match"]]
            matches = [c for c in all_comparisons if c["source_number"] and c["match"]]

            # КРИТИЧЕСКИ ВАЖНО: если есть совпадение (match), нерелевантные расхождения
            # игнорируем. Пример: claim "8 млрд" vs source "8.23 млрд" = match,
            # но source "2.47 млрд (1950 год)" = mismatch из другого контекста.
            # Если хотя бы один источник подтвердил число — считаем числа подтверждёнными.
            if matches:
                # Есть подтверждение — числа совпадают (mismatches из других контекстов игнорируем)
                parts = []
                for m in matches[:3]:
                    cv = m["claim_number"]["raw"]
                    sv = m["source_number"]["raw"]
                    parts.append(f"{cv}={sv}")
                print(f"  Числа: совпадают ({len(matches)} match, {len(mismatches)} mismatch из других контекстов)")
                return "ЧИСЛОВАЯ ПРОВЕРКА: числа совпадают: " + ", ".join(parts)
            elif mismatches:
                # Нет совпадений, только расхождения — реальное расхождение
                parts = []
                for m in mismatches[:3]:
                    cv = m["claim_number"]["raw"]
                    sv = m["source_number"]["raw"]
                    dev = m["deviation"]
                    parts.append(f"claim={cv}, источник={sv} (расхождение {dev*100:.0f}%)")
                hint = "ЧИСЛОВАЯ ПРОВЕРКА: РАСХОЖДЕНИЕ! " + "; ".join(parts)
                print(f"  Числа: РАСХОЖДЕНИЕ ({len(mismatches)} несовпадений, 0 совпадений)")
                return hint
            else:
                return ""

        # Шаг 2.9: Формирование claim для финального промпта (Проблема 1 — основной фикс)
        # Когда утверждение декомпозировано на N под-утверждений — передаём их модели
        # в виде нумерованного списка вместо исходного монолитного предложения.
        def build_claim_for_verdict(state: dict) -> str:
            sub_claims = self._ctx.sub_claims
            claim = state["claim"]
            if len(sub_claims) > 1:
                lines = "\n".join(f"{i}. {sc}" for i, sc in enumerate(sub_claims, 1))
                return (
                    f"СОСТАВНОЕ УТВЕРЖДЕНИЕ из {len(sub_claims)} независимых фактов "
                    f"(оцени каждый пункт ОТДЕЛЬНО):\n{lines}\n\n"
                    "Правило итогового вердикта: если хотя бы один пункт ЛОЖЬ — "
                    "весь вердикт ЛОЖЬ / ФЕЙК."
                )
            return claim

        # Шаг 3: Оценка достоверности
        # Для составных утверждений (sub_claims > 1) — аудиторский шаблон (матрица фактов).
        # Для простых — стандартный шаблон Chain-of-Thought.
        # Выбор происходит динамически на основе self._ctx.sub_claims.

        # Аудиторские промпты: base/SFT и GRPO версии
        audit_template = (
            CREDIBILITY_ASSESSMENT_AUDIT_REASONING_TEMPLATE
            if self._use_reasoning_template
            else CREDIBILITY_ASSESSMENT_AUDIT_TEMPLATE
        )
        audit_prompt = PromptTemplate(
            template=audit_template,
            input_variables=["claim_structured", "search_results", "search_hint",
                             "verification_hints", "nli_hint", "number_comparison"],
        )

        _verdict_prompt = verdict_prompt      # захватываем в closure
        _audit_prompt = audit_prompt

        def verdict_step(state: dict) -> str:
            """Динамически выбирает шаблон: аудиторский (N>1 фактов) или стандартный."""
            sub_claims = self._ctx.sub_claims
            if len(sub_claims) > 1:
                filled = _audit_prompt.format(**{
                    k: state.get(k, "") for k in _audit_prompt.input_variables
                })
                print(f"  [Verdict] Аудиторский шаблон ({len(sub_claims)} фактов)")
            else:
                filled = _verdict_prompt.format(**{
                    k: state.get(k, "") for k in _verdict_prompt.input_variables
                })
                print("  [Verdict] Стандартный шаблон")

            # V8: Минимум 1200 tokens для ВСЕХ шаблонов
            claim = state.get("claim", "")
            n_sub = len(sub_claims)
            claim_len = len(claim)
            if n_sub >= 3 or claim_len > 200:
                dyn_tokens = 1500  # complex
            elif n_sub == 2 or claim_len > 100:
                dyn_tokens = 1200  # medium (was 800)
            else:
                dyn_tokens = 1200  # simple (was 1000) — модель всегда заканчивает мысль
            try:
                self.verdict_llm.pipeline._forward_params["max_new_tokens"] = dyn_tokens
            except (AttributeError, TypeError):
                pass  # fallback to default if pipeline structure changed
            print(f"  [B2] Dynamic tokens: {dyn_tokens}")

            result = self.verdict_llm.invoke(filled)
            return StrOutputParser().invoke(result)

        verdict_chain = RunnableLambda(verdict_step)

        # Полная цепочка
        chain = (
            RunnablePassthrough.assign(
                keywords_raw=keyword_chain,
            )
            | RunnablePassthrough.assign(
                search_results=RunnableLambda(search_step),
            )
            | RunnablePassthrough.assign(
                search_hint=RunnableLambda(get_search_hint),
                verification_hints=RunnableLambda(get_verification_hints),
            )
            | RunnablePassthrough.assign(
                nli_hint=RunnableLambda(nli_analysis_step),
                number_comparison=RunnableLambda(number_check_step),
            )
            | RunnablePassthrough.assign(
                claim_structured=RunnableLambda(build_claim_for_verdict),
            )
            | RunnablePassthrough.assign(
                verdict=verdict_chain,
            )
        )

        # Шаг 4 (опционально): Post-verdict self-critique
        if self.pipeline_config.enable_self_critique:
            critique_prompt = PromptTemplate(
                template=SELF_CRITIQUE_TEMPLATE,
                input_variables=["claim", "search_results", "verdict"],
            )
            critique_chain = critique_prompt | self.critique_llm | StrOutputParser()
            chain = chain | RunnablePassthrough.assign(
                self_critique=critique_chain,
            )

        return chain

    def _decompose_claim(self, claim: str) -> List[str]:
        """LLM-декомпозиция составного утверждения на независимые под-утверждения.

        Пример: "Норвегия входит в ЕС и использует евро"
             → ["Норвегия входит в ЕС", "Норвегия использует евро"]
        Возвращает [claim] если утверждение уже простое или декомпозиция не удалась.
        """
        try:
            prompt = _DECOMPOSE_TEMPLATE.format(claim=claim)
            raw = self.keyword_llm.invoke(prompt)
            lines = [
                l.strip().lstrip("•-–—1234567890.)").strip()
                for l in raw.strip().split("\n")
                if l.strip()
            ]
            sub_claims = [
                l for l in lines
                if len(l) > 15
                and not l.lower().startswith(("факты:", "утверждение:", "ответ:", "пример"))
            ]
            if not sub_claims or len(sub_claims) > 5:
                return [claim]
            # Если все под-утверждения == оригинальный claim — это не декомпозиция
            if all(sc.strip().lower() == claim.strip().lower() for sc in sub_claims):
                return [claim]

            # Semantic validation: sub-claims must share words with original claim
            claim_words = set(re.findall(r'[а-яёa-z]{3,}', claim.lower()))
            validated = []
            for sc in sub_claims:
                sc_words = set(re.findall(r'[а-яёa-z]{3,}', sc.lower()))
                if not sc_words:
                    continue
                overlap = len(sc_words & claim_words) / len(sc_words)
                if overlap >= 0.25:
                    validated.append(sc)
                else:
                    logger.warning(f"Sub-claim rejected (overlap={overlap:.0%}): {sc[:60]}")

            # V8: Negation guard — reject sub-claims that invert meaning
            # If sub-claim contains negation words ("не", "нет", "без", "никогда")
            # that are NOT in the original claim → decomposer hallucinated a negation
            _NEGATION_WORDS = {"не", "нет", "без", "никогда", "ни", "нельзя", "невозможно", "никто", "ничто", "нигде"}
            claim_words_lower = set(re.findall(r'[а-яё]+', claim.lower()))
            negation_checked = []
            for sc in validated:
                sc_words_lower = set(re.findall(r'[а-яё]+', sc.lower()))
                added_negations = (sc_words_lower & _NEGATION_WORDS) - (claim_words_lower & _NEGATION_WORDS)
                if added_negations:
                    logger.warning(
                        f"Negation guard: sub-claim '{sc[:60]}' adds negation "
                        f"{added_negations} not in original → rejected"
                    )
                    print(f"  [NegGuard] Отклонён sub-claim с добавленным отрицанием: {added_negations}")
                    continue
                negation_checked.append(sc)
            validated = negation_checked if negation_checked else [claim]

            # Post-decomposition fidelity check:
            # If sub-claims introduce 3+ significant new words not in original → fallback
            if validated:
                all_sc_words = set()
                for sc in validated:
                    all_sc_words |= set(re.findall(r'[а-яёa-z]{4,}', sc.lower()))
                new_words = all_sc_words - claim_words
                # Filter out common grammatical words
                _fidelity_stopwords = {
                    'является', 'которая', 'которые', 'который', 'через',
                    'после', 'перед', 'также', 'этого', 'более', 'менее',
                    'около', 'всего', 'было', 'были', 'была', 'будет',
                    'может', 'могут', 'этот', 'того', 'кроме', 'между',
                }
                significant_new = new_words - _fidelity_stopwords
                if len(significant_new) >= 3:
                    logger.warning(
                        f"Fidelity check failed: {len(significant_new)} new words "
                        f"({', '.join(list(significant_new)[:5])}) → fallback to original"
                    )
                    return [claim]

            return validated if validated else [claim]
        except Exception as e:
            logger.warning(f"_decompose_claim failed: {e}")
            return [claim]

    @staticmethod
    def _parse_keywords(raw_output: str) -> List[str]:
        """Умный парсер ключевых слов с сохранением многословных сущностей.

        1. Извлекает сущности из вывода модели (запятые, нумерация, дефисы)
        2. Склеивает известные многословные сущности (ЦБ РФ, МВД РФ, ...)
        3. Программно удаляет: месяцы, числа без контекста, общие слова
        4. Дедупликация (подстроки убираются)
        5. Fallback если всё отфильтровано
        """
        text = raw_output.strip()

        # Убираем типичные префиксы
        for prefix in ["Ключевые слова:", "Keywords:", "ключевые слова:",
                       "КЛЮЧЕВЫЕ СЛОВА:", "Ключевые сущности:",
                       "ключевые сущности:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Убираем нумерацию (1. xxx, 2. yyy)
        text = re.sub(r'^\d+[.)]\s*', '', text, flags=re.MULTILINE)

        # Парсинг: приоритет — запятая, затем перенос строки
        if "," in text:
            raw_keywords = [kw.strip().strip('"').strip("'").strip("-—•")
                           for kw in text.split(",")]
        elif "\n" in text:
            raw_keywords = [kw.strip().strip('"').strip("'").strip("-—•")
                           for kw in text.split("\n")]
        else:
            # Единственная фраза — используем как есть
            raw_keywords = [text.strip()]

        # Разбиваем ключевые слова с встроенными переносами строк
        # (LLM иногда генерирует "50%\nSpaceX" как одно ключевое слово)
        expanded = []
        for kw in raw_keywords:
            if '\n' in kw:
                expanded.extend(part.strip().strip('"').strip("'").strip("-—•")
                                for part in kw.split('\n'))
            else:
                expanded.append(kw)
        raw_keywords = [kw for kw in expanded if kw and len(kw) < 60]

        # === Склеивание известных многословных сущностей ===
        raw_keywords = _rejoin_entities(raw_keywords)

        # === Программная очистка ===
        cleaned = []
        for kw in raw_keywords:
            kw_lower = kw.lower().strip()

            # Пропуск месяцев
            if kw_lower in _MONTHS_RU:
                continue

            # Пропуск чисто числовых БЕЗ единиц (e.g. "22", "33")
            # НО оставляем числа с % и единицами — они критичны для фактчекинга
            if re.match(r'^\d+$', kw_lower):
                continue

            # Пропуск общих слов
            if kw_lower in _GENERIC_WORDS:
                continue

            # Пропуск слишком коротких (1 символ)
            if len(kw_lower) < 2:
                continue

            cleaned.append(kw)

        # === Дедупликация: убираем подстроки ===
        # Если "ЦБ" и "ЦБ РФ" оба есть — убираем "ЦБ"
        deduplicated = _deduplicate_keywords(cleaned)

        # Fallback: если всё отфильтровано, вернуть оригинал
        result = deduplicated if deduplicated else raw_keywords
        return result[:5]

    def _validate_keywords(self, keywords: List[str], claim: str) -> List[str]:
        """Фильтрация keywords контаминированных из few-shot примеров.

        LLM иногда копирует keywords из примеров промпта вместо извлечения из claim.
        Проверяем: хотя бы одно слово keyword должно встречаться в claim.

        Используем stem-matching (первые 5 символов) для русской морфологии:
        "Австралия" → "австр" совпадает с "Австралии" → "австр".
        """
        claim_lower = claim.lower()
        claim_words = set(re.findall(r'[а-яёa-z0-9-]{3,}', claim_lower))
        # Stems: первые 5 символов для морфологического matching
        claim_stems = set(w[:5] for w in claim_words if len(w) >= 5)
        claim_stems.update(claim_words)  # короткие слова — как есть

        validated = []
        for kw in keywords:
            kw_words = set(re.findall(r'[а-яёa-z0-9-]{3,}', kw.lower()))
            kw_stems = set(w[:5] for w in kw_words if len(w) >= 5)
            kw_stems.update(kw_words)
            # Хотя бы одно слово/stem из keyword встречается в claim
            if kw_stems & claim_stems:
                validated.append(kw)
            else:
                logger.warning(f"Keyword '{kw}' не найден в claim — контаминация, пропуск")

        if not validated:
            # Fallback: разбиваем claim на 3-словные n-граммы
            words = claim.split()
            validated = [" ".join(words[i:i+2]) for i in range(0, min(len(words), 6), 2)]
            logger.info(f"Keyword fallback: {validated}")

        return validated

    @staticmethod
    def parse_verdict(raw_verdict: str) -> Dict[str, Any]:
        """Парсинг структурированного вердикта из ответа модели.

        Поддерживает три формата:
        1. Стандартный: ДОСТОВЕРНОСТЬ: ... ВЕРДИКТ: ... (SFT модель)
        2. XML: <reasoning>...</reasoning><answer>...</answer> (GRPO модель)
        3. Аудиторский: ПУНКТ N / ЦИТАТА / ИСТОЧНИК / СТАТУС (многофакторный)
        """
        result = {
            "credibility_score": 50,
            "verdict": "НЕ ПОДТВЕРЖДЕНО",
            "confidence": 50,
            "reasoning": "",
            "chain_of_thought": "",
            "sources": "",
            "sub_verdicts": [],   # список per-fact результатов из аудиторского шаблона
            "raw": raw_verdict,
        }

        # Извлечение chain-of-thought из <reasoning> тегов (если есть)
        cot_match = re.search(
            r"<reasoning>(.*?)</reasoning>", raw_verdict, re.DOTALL
        )
        if cot_match:
            cot_content = cot_match.group(1).strip()
            # Детектор: модель ничего не написала в reasoning.
            # С новым шаблоном инструкции вынесены ДО <reasoning>,
            # поэтому содержательным считается любой reasoning > 40 слов.
            _has_citation = bool(re.search(r'\[(?!Источник\b)[^\]]{3,60}\]\s*сообщает', cot_content))
            _has_real_content = _has_citation or len(cot_content.split()) > 40
            if not _has_real_content:
                print(f"  [WARNING:CoT] <reasoning> пустой или слишком короткий "
                      f"({len(cot_content)} символов) — модель не сгенерировала анализ")
                result["chain_of_thought"] = ""
                result["_empty_reasoning"] = True  # Флаг для retry в check()
            else:
                print(f"  [CoT] <reasoning> заполнен ({len(cot_content)} символов)")
                result["chain_of_thought"] = cot_content

        # Парсинг аудиторского протокола: ПУНКТ N / ЦИТАТА / ИСТОЧНИК / СТАТУС
        # Работает как с <reasoning> блоком (GRPO), так и с plain-текстом (base/SFT).
        # Ищем весь текст ответа, включая <reasoning> содержимое.
        _audit_search_text = raw_verdict
        _fact_block_pattern = re.compile(
            r"ПУНКТ\s+(\d+)[:\s]+(.+?)\n"
            r"ЦИТАТА[:\s]+(.+?)\n"
            r"ИСТОЧНИК[:\s]+(.+?)\n"
            r"СТАТУС[:\s]+(ПРАВДА|ЛОЖЬ|НЕТ ДАННЫХ|НЕТ)",
            re.DOTALL | re.IGNORECASE,
        )
        sub_verdicts = []
        for m in _fact_block_pattern.finditer(_audit_search_text):
            citation = m.group(3).strip()
            sub_verdicts.append({
                "index": int(m.group(1)),
                "claim": m.group(2).strip(),
                "citation": "" if "ЦИТАТА ОТСУТСТВУЕТ" in citation.upper() else citation,
                "has_citation": "ЦИТАТА ОТСУТСТВУЕТ" not in citation.upper(),
                "source": m.group(4).strip(),
                "status": m.group(5).strip().upper(),
            })
        if sub_verdicts:
            result["sub_verdicts"] = sub_verdicts
            result["_raw_sub_verdicts_count"] = len(sub_verdicts)
            statuses = [sv["status"] for sv in sub_verdicts]
            print(f"  [Audit] Распарсено {len(sub_verdicts)} пунктов: "
                  + ", ".join(f"П{sv['index']}={sv['status']}" for sv in sub_verdicts))
            # Audit matrix verdict вычисляется в check() после parse_verdict
            # (там он может быть скорректирован с учётом ensemble и scam override)

        # Парсинг структурированных полей (ищем и в <answer> и в основном тексте)
        patterns = {
            "credibility_score": r"ДОСТОВЕРНОСТЬ:\s*(\d+)",
            "verdict": r"ВЕРДИКТ:\s*(.+?)(?:\n|$)",
            "confidence": r"УВЕРЕННОСТЬ:\s*(\d+)",
            "reasoning": r"ОБОСНОВАНИЕ:\s*(.+?)(?:\nИСТОЧНИКИ:|$)",
            "sources": r"ИСТОЧНИКИ:\s*(.+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, raw_verdict, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if key in ("credibility_score", "confidence"):
                    try:
                        value = int(value)
                        value = max(0, min(100, value))
                    except ValueError:
                        continue
                result[key] = value

        # Audit matrix verdict вычисляется в check() на основе sub_verdicts

        # V7: Fallback verdict extraction when structured fields missing
        if result["verdict"] == "НЕ ПОДТВЕРЖДЕНО" and result["credibility_score"] == 50:
            # First try <answer> tag
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_verdict, re.DOTALL)
            answer_text = answer_match.group(1).strip() if answer_match else ""
            # If no <answer> tag AND reasoning was truncated (no </reasoning>),
            # try extracting verdict from partial reasoning text
            if not answer_text:
                has_reasoning_start = "<reasoning>" in raw_verdict
                has_reasoning_end = "</reasoning>" in raw_verdict
                if has_reasoning_start and not has_reasoning_end:
                    # Truncated output — model ran out of tokens
                    reasoning_text = raw_verdict.split("<reasoning>", 1)[1].lower()
                    # Look for clear verdict signals in reasoning
                    positive_signals = ["утверждение верно", "утверждение правдив",
                                        "подтверждается", "все источники подтверждают",
                                        "по данным всех источников"]
                    negative_signals = ["утверждение лож", "утверждение невер",
                                        "источники опровергают", "противоречит"]
                    has_positive = any(s in reasoning_text for s in positive_signals)
                    has_negative = any(s in reasoning_text for s in negative_signals)
                    if has_positive and not has_negative:
                        result["verdict"] = "ПРАВДА"
                        result["credibility_score"] = 75
                        result["_truncated_extraction"] = True
                        print("  [V7:Truncated] Reasoning truncated, extracted ПРАВДА from partial text")
                    elif has_negative and not has_positive:
                        result["verdict"] = "ЛОЖЬ"
                        result["credibility_score"] = 25
                        result["_truncated_extraction"] = True
                        print("  [V7:Truncated] Reasoning truncated, extracted ЛОЖЬ from partial text")
            if answer_text:
                # Try extracting verdict from content
                for v_label, v_name in [("ПРАВДА", "ПРАВДА"), ("ЛОЖЬ", "ЛОЖЬ"),
                                         ("НЕ ПОДТВЕРЖДЕНО", "НЕ ПОДТВЕРЖДЕНО"),
                                         ("МАНИПУЛЯЦИЯ", "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА")]:
                    if v_label in answer_text.upper():
                        result["verdict"] = v_name
                        # Try extracting score from answer
                        score_m = re.search(r"(\d{1,3})", answer_text)
                        if score_m:
                            s = int(score_m.group(1))
                            if 0 <= s <= 100:
                                result["credibility_score"] = s
                        elif "ПРАВДА" in v_name:
                            result["credibility_score"] = 80
                        elif "ЛОЖЬ" in v_name:
                            result["credibility_score"] = 20
                        break

        return result

    @staticmethod
    def _parse_self_critique(raw_critique: str) -> Dict[str, Any]:
        """Парсинг результата self-critique."""
        result = {
            "errors": "",
            "needs_correction": False,
            "recommended_score": None,
            "recommended_verdict": None,
            "raw": raw_critique,
        }

        # ОШИБКИ
        errors_match = re.search(r"ОШИБКИ:\s*(.+?)(?:\n|$)", raw_critique, re.DOTALL)
        if errors_match:
            errors = errors_match.group(1).strip()
            result["errors"] = errors

        # КОРРЕКЦИЯ
        correction_match = re.search(r"КОРРЕКЦИЯ:\s*(ДА|НЕТ)", raw_critique, re.IGNORECASE)
        if correction_match:
            result["needs_correction"] = correction_match.group(1).upper() == "ДА"

        # РЕКОМЕНДУЕМЫЙ_SCORE
        score_match = re.search(r"РЕКОМЕНДУЕМЫЙ_SCORE:\s*(\d+)", raw_critique)
        if score_match:
            result["recommended_score"] = max(0, min(100, int(score_match.group(1))))

        # РЕКОМЕНДУЕМЫЙ_ВЕРДИКТ
        verdict_match = re.search(
            r"РЕКОМЕНДУЕМЫЙ_ВЕРДИКТ:\s*(.+?)(?:\n|$)", raw_critique
        )
        if verdict_match:
            v = verdict_match.group(1).strip().upper()
            if v in ("ПРАВДА", "ЛОЖЬ", "НЕ ПОДТВЕРЖДЕНО"):
                result["recommended_verdict"] = v

        return result

    @staticmethod
    def _apply_self_critique(
        parsed_verdict: Dict[str, Any],
        critique: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Применение коррекций из self-critique к вердикту."""
        if not critique["needs_correction"]:
            return parsed_verdict

        rec_score = critique["recommended_score"]
        rec_verdict = critique["recommended_verdict"]

        if rec_score is not None and rec_verdict is not None:
            is_consistent = False
            if rec_verdict == "ПРАВДА" and 70 <= rec_score <= 100:
                is_consistent = True
            elif rec_verdict == "ЛОЖЬ" and 0 <= rec_score <= 29:
                is_consistent = True
            elif rec_verdict == "НЕ ПОДТВЕРЖДЕНО" and 30 <= rec_score <= 69:
                is_consistent = True
            # Fix: разрешить score 0-29 когда критик рекомендует снижение до ЛОЖЬ/НЕ ПОДТВЕРЖДЕНО
            elif rec_verdict == "НЕ ПОДТВЕРЖДЕНО" and 0 <= rec_score <= 29:
                # Критик рекомендует "НЕ ПОДТВЕРЖДЕНО" с очень низким score — маппируем в ЛОЖЬ
                rec_verdict = "ЛОЖЬ"
                is_consistent = True
            elif rec_verdict == "ЛОЖЬ" and 30 <= rec_score <= 69:
                # Критик говорит ЛОЖЬ, но score в зоне НЕ ПОДТВЕРЖДЕНО — доверяем вердикту
                rec_score = min(rec_score, 25)
                is_consistent = True

            if is_consistent:
                parsed_verdict["credibility_score"] = rec_score
                parsed_verdict["verdict"] = rec_verdict

        errors = critique.get("errors", "")
        if errors and errors.lower() != "нет":
            parsed_verdict["reasoning"] += f" [Самопроверка: {errors}]"

        return parsed_verdict

    @staticmethod
    def extract_sources(raw_search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Извлечение списка источников из результатов поиска."""
        sources = []
        for article in raw_search_results:
            sources.append({
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "link": article.get("link", ""),
                "date": article.get("date", ""),
            })
        return sources

    def _meta_classify(self, parsed: Dict[str, Any], features: Dict) -> Dict[str, Any]:
        """Вердикт через обученный meta-классификатор вместо ручных правил."""
        import numpy as np
        feature_names = [
            "nli_ent", "nli_con", "nli_ent_count", "nli_con_count",
            "llm_verdict", "llm_score", "nums_match", "nums_mismatch",
            "match_ratio", "num_sources", "nli_mixed", "nli_clean_ent", "nli_clean_con",
        ]
        X = np.array([[features[f] for f in feature_names]])
        pred = self._meta_classifier.predict(X)[0]
        proba = self._meta_classifier.predict_proba(X)[0]
        confidence = float(max(proba))

        if pred == 1:
            parsed["verdict"] = "ПРАВДА"
            parsed["credibility_score"] = int(confidence * 100)
        else:
            parsed["verdict"] = "ЛОЖЬ"
            parsed["credibility_score"] = int((1 - confidence) * 100)

        logger.info(f"Meta-classifier: {parsed['verdict']} (conf={confidence:.2f}, proba={proba})")
        return parsed

    @staticmethod
    def _check_zero_evidence(claim: str, nli_result: Dict[str, Any]) -> bool:
        """Если ни один источник реально не подтвердил скам-тему claim → True.

        Предотвращает ситуацию когда биографические источники дают entailment
        по имени персоны, но не по скам-компоненте (выплаты, бот, активы).
        """
        scam_markers = ["выплат", "взнос", "бот", "телеграм", "telegram",
                        "актив", "наследств", "верификац", "комисси", "страхов",
                        "безоп", "кошел", "привяз", "облиг", "секрет",
                        "проверочн", "регистрац", "платформ"]
        claim_lower = claim.lower()
        has_scam_terms = any(m in claim_lower for m in scam_markers)
        if not has_scam_terms:
            return False

        if not nli_result:
            return True

        for src in nli_result.get("pairs", []):
            if src.get("label") == "entailment" and src.get("entailment", 0) > 0.6:
                snippet_lower = src.get("source", "").lower()
                if any(m in snippet_lower for m in scam_markers):
                    return False  # нашли подтверждение скам-темы
        return True  # ни один source не подтвердил скам-тему

    def _sources_are_scam_warnings(self, claim, raw_results, nli_result):
        """True если entailing источники — анти-фрод статьи (предупреждения)."""
        WARNING_RE = re.compile(
            r'(?:предупрежда|мошенни|мошенн|осторожн|обман|'
            r'не\s+переводите|не\s+сообщайте|фрод|fraud|scam|'
            r'разоблач|фишинг|будьте\s+бдительн)',
            re.IGNORECASE
        )
        if not nli_result or not nli_result.get("pairs"):
            return False

        entailing = [p for p in nli_result["pairs"]
                     if p.get("label") == "entailment" and p.get("entailment", 0) > 0.5]
        if not entailing:
            return False

        warning_count = 0
        for pair in entailing:
            src_name = pair.get("source", "")
            for r in raw_results:
                if src_name and src_name in r.get("title", ""):
                    text = r.get("title", "") + " " + r.get("snippet", "")
                    if WARNING_RE.search(text):
                        warning_count += 1
                    break
        return warning_count > 0 and warning_count >= len(entailing) * 0.5

    def _ensemble_verdict(
        self,
        parsed: Dict[str, Any],
        nli_result: Dict[str, Any],
        num_comparisons: List[Dict],
        raw_results: List[Dict],
        claim: str = "",
    ) -> Dict[str, Any]:
        """Ensemble verdict v8 — упрощённый, 5 надёжных сигналов.

        Принцип: Wikidata(±3), Числа(±3), LLM(±2), NLI(±1.5), Скам(-4).
        Меньше конфликтов. Меньше багов. Доверяем модели.
        """
        claim_info = self._ctx.claim_info or {}
        claim_type = claim_info.get("type", "general")

        verdict = parsed["verdict"].upper().strip()
        score = parsed["credibility_score"]
        has_sources = len(raw_results) > 0

        # --- Извлечение сигналов ---
        nums_match = any(c for c in num_comparisons if c["source_number"] and c["match"])
        nums_mismatch = (
            any(c for c in num_comparisons if c["source_number"] and not c["match"])
            and not nums_match
        )

        nli_ent = nli_result.get("max_entailment", 0.0) if nli_result else 0.0
        nli_con = nli_result.get("max_contradiction", 0.0) if nli_result else 0.0
        nli_ent_count = nli_result.get("entailment_count", 0) if nli_result else 0
        nli_con_count = nli_result.get("contradiction_count", 0) if nli_result else 0

        # NLI purity filtering
        if nli_result and nli_result.get("pairs"):
            clean_pairs = [p for p in nli_result["pairs"] if p.get("purity", 0) > 0.3]
            if len(clean_pairs) >= 2:
                nli_ent_count = sum(1 for p in clean_pairs if p["label"] == "entailment")
                nli_con_count = sum(1 for p in clean_pairs if p["label"] == "contradiction")

        # Feature vector
        match_count = sum(1 for c in num_comparisons if c["source_number"] and c["match"])
        total_with_source = sum(1 for c in num_comparisons if c["source_number"])
        match_ratio = match_count / total_with_source if total_with_source > 0 else 0.0
        nli_mixed = nli_ent >= 0.40 and nli_con >= 0.40
        parsed["_ensemble_features"] = {
            "nli_ent": nli_ent, "nli_con": nli_con,
            "nli_ent_count": nli_ent_count, "nli_con_count": nli_con_count,
            "llm_verdict": 1 if verdict == "ПРАВДА" else (-1 if verdict == "ЛОЖЬ" else 0),
            "llm_score": score, "nums_match": int(nums_match),
            "nums_mismatch": int(nums_mismatch), "match_ratio": match_ratio,
            "num_sources": len(raw_results), "nli_mixed": int(nli_mixed),
            "nli_clean_ent": int(nli_ent >= 0.50 and nli_con < 0.35),
            "nli_clean_con": int(nli_con >= 0.50 and nli_ent < 0.35),
        }

        # META-CLASSIFIER (если обучен)
        if hasattr(self, '_meta_classifier') and self._meta_classifier is not None:
            return self._meta_classify(parsed, parsed["_ensemble_features"])

        # ============================================================
        # V8: SIMPLIFIED ENSEMBLE — 5 надёжных сигналов
        # ============================================================

        # Evidence sufficiency gating
        claim_entities = self._ctx.claim_entities or []
        if has_sources and claim_entities:
            src_text_all = " ".join(
                (r.get("snippet", "") + " " + r.get("title", "")).lower()
                for r in raw_results[:7]
            )
            entity_coverage = sum(1 for e in claim_entities if e in src_text_all) / max(len(claim_entities), 1)
        else:
            entity_coverage = 0.0
        nli_decisiveness = max(nli_ent, nli_con)
        source_count_norm = min(len(raw_results) / 5.0, 1.0)
        sufficiency = entity_coverage * 0.3 + nli_decisiveness * 0.4 + source_count_norm * 0.3
        if sufficiency < 0.30 and not self._ctx.is_scam:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["credibility_score"] = 50
            logger.info(f"V8: Evidence insufficient (sufficiency={sufficiency:.2f}) → НЕ ПОДТВЕРЖДЕНО")
            return parsed

        vote = 0.0

        # --- СИГНАЛ 1: Wikidata (±3) — самый надёжный ---
        wikidata = self._ctx.wikidata_result
        if wikidata and wikidata.get("found"):
            wd_facts = wikidata.get("facts", [])
            wd_confirmed = sum(1 for f in wd_facts if f.get("match") is True)
            wd_contradicted = sum(1 for f in wd_facts if f.get("match") is False)
            wd_neutral = sum(1 for f in wd_facts if f.get("match") is None)

            # Neutral facts matching claim text = implicit confirmation
            if wd_neutral > 0 and wd_contradicted == 0:
                claim_lower = claim.lower()
                if _HAS_NATASHA:
                    claim_lemmas = set(lemmatize(w) for w in re.findall(r'[а-яёa-z]{3,}', claim_lower))
                else:
                    claim_lemmas = set(w[:4] for w in re.findall(r'[а-яёa-z]{4,}', claim_lower))
                for f in wd_facts:
                    if f.get("match") is None:
                        for v in f.get("wikidata_values", []):
                            if _HAS_NATASHA:
                                if lemmatize(v) in claim_lemmas or v.lower() in claim_lower:
                                    wd_confirmed += 1
                                    break
                            else:
                                v_stem = v.lower()[:4] if len(v) >= 4 else v.lower()
                                if v_stem in claim_lemmas or v.lower() in claim_lower:
                                    wd_confirmed += 1
                                    break

            if wd_confirmed > 0 and wd_contradicted == 0:
                wd_vote = min(3.0, wd_confirmed * 1.0)
                vote += wd_vote
                logger.info(f"Vote: WIKIDATA_CONFIRMED → +{wd_vote:.1f}")
            elif wd_contradicted > 0:
                wd_vote = min(3.0, wd_contradicted * 1.5)
                vote -= wd_vote
                logger.info(f"Vote: WIKIDATA_CONTRADICTED → -{wd_vote:.1f}")

        # --- СИГНАЛ 2: Числа (±3) — детерминистический ---
        if nums_mismatch:
            vote -= 3.0
            logger.info("Vote: NUMS_MISMATCH → -3")
        elif nums_match:
            vote += 3.0
            logger.info("Vote: NUMS_MATCH → +3")

        # --- СИГНАЛ 3: LLM вердикт (±2) ---
        if verdict == "ПРАВДА":
            llm_vote = max(0.0, min(2.0, (score - 50) / 25.0))
            vote += llm_vote
            logger.info(f"Vote: LLM ПРАВДА (score={score}) → +{llm_vote:.2f}")
        elif verdict == "ЛОЖЬ":
            llm_vote = max(-2.0, min(0.0, (score - 50) / 25.0))
            vote += llm_vote
            logger.info(f"Vote: LLM ЛОЖЬ (score={score}) → {llm_vote:+.2f}")

        # --- СИГНАЛ 4: NLI (±1.5) — только при сильном сигнале ---
        if has_sources:
            # NLI только при сильном сигнале (>0.65) и не mixed
            if nli_ent >= 0.65 and nli_con < 0.35:
                nli_vote = min(1.5, nli_ent * 1.5)
                vote += nli_vote
                logger.info(f"Vote: NLI entailment={nli_ent:.2f} → +{nli_vote:.2f}")
            elif nli_con >= 0.65 and nli_ent < 0.35:
                nli_vote = min(1.5, nli_con * 1.5)
                vote -= nli_vote
                logger.info(f"Vote: NLI contradiction={nli_con:.2f} → -{nli_vote:.2f}")
            elif nli_mixed:
                logger.info(f"Vote: NLI mixed (ent={nli_ent:.2f}, con={nli_con:.2f}) → 0")

        # --- СИГНАЛ 5: Скам-паттерн (-4) ---
        if self._ctx.is_scam:
            vote -= 4.0
            logger.info("Vote: SCAM_PATTERN → -4.0")
        if self._check_zero_evidence(claim, nli_result):
            vote -= 3.0
            logger.info("Vote: ZERO_EVIDENCE_SCAM → -3.0")

        # --- СИГНАЛ 6: MiniCheck-FT5 (±1.5) — кросс-модельная верификация ---
        mc_result = self._ctx.minicheck_result
        if mc_result and mc_result.get("per_source"):
            mc_signal = mc_result.get("signal", 0)
            if mc_signal != 0:
                mc_vote = mc_signal * 0.75  # scale: ±2 → ±1.5
                vote += mc_vote
                mc_sup = mc_result["max_support"]
                logger.info(f"Vote: MINICHECK (support={mc_sup:.2f}) → {mc_vote:+.2f}")

        # --- Date mismatch (бонусный сигнал, -2) ---
        if self._ctx.date_mismatch:
            vote -= 2.0
            logger.info("Vote: DATE_MISMATCH → -2.0")

        # --- ИТОГОВОЕ РЕШЕНИЕ ---
        logger.info(f"V9 Vote TOTAL: {vote:+.2f}")
        print(f"  [V8:Ensemble] vote={vote:+.2f}")

        VOTE_THRESHOLD = 1.0

        if nli_mixed and -VOTE_THRESHOLD < vote < VOTE_THRESHOLD:
            parsed["verdict"] = "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА"
            parsed["credibility_score"] = max(30, min(45, int(50 + vote * 5)))
        elif vote >= VOTE_THRESHOLD:
            parsed["verdict"] = "ПРАВДА"
            parsed["credibility_score"] = min(95, int(50 + vote * 10))
        elif vote <= -VOTE_THRESHOLD:
            parsed["verdict"] = "ЛОЖЬ"
            parsed["credibility_score"] = max(5, int(50 + vote * 10))
        else:
            # V8: В зоне неопределённости — доверяем LLM с cap
            if verdict == "ПРАВДА":
                parsed["verdict"] = "ПРАВДА"
                parsed["credibility_score"] = min(score, 75)
            elif verdict == "ЛОЖЬ":
                parsed["verdict"] = "ЛОЖЬ"
                parsed["credibility_score"] = max(score, 25)
            else:
                parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = 50
            logger.info(f"V8: Uncertain (vote={vote:+.2f}) → LLM fallback: {parsed['verdict']}")

        return parsed

    def _self_consistency_vote(self, state: dict, first_parsed: Dict) -> Dict:
        """V9: Self-consistency — N прогонов LLM с temperature>0, majority vote.

        Если модель не уверена в себе (разные вердикты при разных прогонах),
        это сигнал неопределённости → НЕ ПОДТВЕРЖДЕНО.
        """
        n_runs = getattr(self.pipeline_config, 'self_consistency_n', 3)
        sc_temp = getattr(self.pipeline_config, 'self_consistency_temperature', 0.3)

        # First verdict already parsed
        verdicts = [first_parsed["verdict"]]

        # Temporarily enable sampling with temperature
        try:
            orig_do_sample = self.verdict_llm.pipeline._forward_params.get("do_sample", False)
            orig_temp = self.verdict_llm.pipeline._forward_params.get("temperature", None)
            self.verdict_llm.pipeline._forward_params["do_sample"] = True
            self.verdict_llm.pipeline._forward_params["temperature"] = sc_temp
        except (AttributeError, TypeError):
            print("  [SC] Cannot set temperature — skipping self-consistency")
            return first_parsed

        try:
            for i in range(n_runs - 1):
                try:
                    sc_result = self.chain.invoke(state)
                    sc_parsed = self.parse_verdict(sc_result.get("verdict", ""))
                    sc_verdict = sc_parsed["verdict"].upper().strip()
                    verdicts.append(sc_verdict)
                except Exception as e:
                    logger.warning(f"SC run {i+2} failed: {e}")
                    verdicts.append("НЕ ПОДТВЕРЖДЕНО")
        finally:
            # Restore original settings
            try:
                self.verdict_llm.pipeline._forward_params["do_sample"] = orig_do_sample
                if orig_temp is not None:
                    self.verdict_llm.pipeline._forward_params["temperature"] = orig_temp
                else:
                    self.verdict_llm.pipeline._forward_params.pop("temperature", None)
            except (AttributeError, TypeError):
                pass

        # Majority vote
        from collections import Counter
        vote_counts = Counter(verdicts)
        majority_verdict, majority_count = vote_counts.most_common(1)[0]

        print(f"  [SC] Verdicts: {verdicts} → majority: {majority_verdict} ({majority_count}/{len(verdicts)})")

        # If no majority (all different) → uncertainty
        if majority_count <= len(verdicts) // 2:
            first_parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            first_parsed["credibility_score"] = 50
            first_parsed["_sc_disagreement"] = True
            print(f"  [SC] No majority → НЕ ПОДТВЕРЖДЕНО")
        elif majority_verdict != first_parsed["verdict"]:
            # Majority disagrees with first run → use majority
            first_parsed["verdict"] = majority_verdict
            # Adjust score based on majority direction
            if majority_verdict == "ПРАВДА":
                first_parsed["credibility_score"] = max(first_parsed["credibility_score"], 70)
            elif majority_verdict == "ЛОЖЬ":
                first_parsed["credibility_score"] = min(first_parsed["credibility_score"], 30)
            print(f"  [SC] Majority override: {first_parsed['verdict']}")

        first_parsed["_sc_verdicts"] = verdicts
        first_parsed["_sc_agreement"] = majority_count / len(verdicts)
        return first_parsed

    def _detect_fallacy(self, claim: str, confirmed_facts: List[str]) -> Dict:
        """Проверяет логическую связность факт→вывод."""
        if len(confirmed_facts) < 2:
            return {"fallacy": None}
        prompt = FALLACY_DETECTION_TEMPLATE.format(
            claim=claim,
            confirmed_facts="\n".join(f"- {f}" for f in confirmed_facts)
        )
        try:
            raw = self.keyword_llm.invoke(prompt)
            if "НЕ ОБНАРУЖЕНА" in raw:
                return {"fallacy": None}
            return {"fallacy": raw, "penalty": -1.5}
        except Exception:
            return {"fallacy": None}

    def check(self, claim: str) -> Dict[str, Any]:
        """Проверка одного утверждения. Возвращает структурированный результат."""
        logger.info(f"Проверка: {claim[:120]}")
        print(f"\nПроверка: {claim}")

        # Fact Cache: проверяем кэш
        cached = self.fact_cache.get(claim)
        if cached:
            print(f"  [Cache] HIT — {cached['verdict']} (score={cached['credibility_score']})")
            return cached

        # Сброс промежуточного состояния (thread-safe per-request context)
        self._ctx = PipelineContext()

        # Декомпозиция составных утверждений (Проблема 1)
        if self.pipeline_config.enable_claim_decomposition:
            sub_claims = self._decompose_claim(claim)
            if len(sub_claims) > 1:
                print(f"  Декомпозиция: {len(sub_claims)} под-утверждений")
                for i, sc in enumerate(sub_claims, 1):
                    print(f"    {i}. {sc}")
                self._ctx.sub_claims = sub_claims

        state = {"claim": claim}

        try:
            # Запускаем полную цепочку
            total_start = time.time()
            raw_result = self.chain.invoke(state)
            total_time = time.time() - total_start
            # Лог сырого ответа модели (первые 300 символов)
            raw_v = raw_result.get("verdict", "")
            print(f"  [Model:raw] {len(raw_v)} символов | превью: {raw_v[:200].replace(chr(10), ' ')!r}")
            # [debug line removed]
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {
                "claim": claim,
                "credibility_score": 50,
                "verdict": "НЕ ПОДТВЕРЖДЕНО",
                "confidence": 0,
                "reasoning": f"Ошибка при анализе: {str(e)}. Попробуйте позже.",
                "sources": [],
                "sources_text": "",
                "keywords": [],
                "search_results_formatted": "",
                "raw_verdict": "",
                "self_critique": "",
                "total_time": 0,
            }

        # Парсинг вердикта
        raw_results = self._ctx.raw_results
        nli_result = self._ctx.nli_result
        num_comparisons = self._ctx.num_comparisons
        trusted_cnt = sum(1 for r in raw_results if not r.get("_unverified"))
        print(f"  [Sources] Итого после фильтра: {len(raw_results)} "
              f"(доверенных: {trusted_cnt})")
        parsed = self.parse_verdict(raw_result.get("verdict", ""))

        # V8: Retry при пустом/обрезанном reasoning
        _needs_retry = parsed.get("_empty_reasoning")
        # Также retry если reasoning обрезан (нет </reasoning>)
        raw_v = raw_result.get("verdict", "")
        if not _needs_retry and "<reasoning>" in raw_v and "</reasoning>" not in raw_v:
            _needs_retry = True
            print("  [V8:Truncated] Reasoning обрезан — retry с увеличенными токенами")

        if _needs_retry and self._use_reasoning_template:
            print("  [V8:Retry] Повторяем генерацию (1500 tokens)")
            try:
                # V8: Увеличиваем токены для retry
                try:
                    self.verdict_llm.pipeline._forward_params["max_new_tokens"] = 1500
                except (AttributeError, TypeError):
                    pass
                retry_result = self.chain.invoke(state)
                retry_parsed = self.parse_verdict(retry_result.get("verdict", ""))
                if not retry_parsed.get("_empty_reasoning"):
                    parsed = retry_parsed
                    raw_result = retry_result
                    print("  [V8:Retry] Успешно — reasoning заполнен")
                else:
                    print("  [V8:Retry] Повторно пустой — используем truncated extraction")
            except Exception as e:
                logger.warning(f"Retry failed: {e}")
        parsed.pop("_empty_reasoning", None)

        # V9: MiniCheck-FT5 verification (CPU, параллельно с парсингом)
        if self.minicheck and raw_results:
            try:
                mc_result = self.minicheck.verify_claim(
                    claim, raw_results[:5], snippet_key="snippet"
                )
                self._ctx.minicheck_result = mc_result
                print(f"  [MiniCheck] support={mc_result['max_support']:.2f}, "
                      f"signal={mc_result['signal']:+d}")
            except Exception as e:
                print(f"  [MiniCheck] Ошибка: {e}")
                self._ctx.minicheck_result = None

        # Claim-drift detection: if model restated a different claim in reasoning,
        # its verdict is unreliable. Check word overlap between original claim
        # and the model's restated claim (first sentence after "Утверждение:").
        cot = parsed.get("chain_of_thought", "")
        if cot:
            _restate_m = re.search(r'[Уу]тверждени[ея][:\s]+[«"]?(.+?)[»"]?(?:\.|Источник|\n)', cot)
            if _restate_m:
                _restated = _restate_m.group(1).lower().strip()
                _orig_words = set(re.findall(r'[а-яёa-z]{4,}', claim.lower()))
                _rest_words = set(re.findall(r'[а-яёa-z]{4,}', _restated))
                if _orig_words and _rest_words:
                    _drift = 1.0 - len(_orig_words & _rest_words) / max(len(_orig_words), 1)
                    if _drift > 0.6:
                        print(f"  [ClaimDrift] Model restated different claim (drift={_drift:.0%}): "
                              f"'{_restated[:60]}' — verdict unreliable")
                        parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                        parsed["credibility_score"] = 50
                        parsed["_claim_drift"] = True

        # Fix 3: Ограничение парсера — обрезаем sub_verdicts до количества входных фактов
        sub_claims = self._ctx.sub_claims
        _has_audit_matrix = False
        if sub_claims and parsed.get("sub_verdicts"):
            n_facts = len(sub_claims)
            raw_count = parsed.get("_raw_sub_verdicts_count", len(parsed["sub_verdicts"]))
            if len(parsed["sub_verdicts"]) > n_facts:
                print(f"  [Audit:Trim] Обрезка: {raw_count} распарсено → {n_facts} фактов на входе")
                parsed["sub_verdicts"] = parsed["sub_verdicts"][:n_facts]
            # A4: Post-audit designation verification
            # If sub-claim mentions "Восток-2" but source only has "Восток-1" → override to ЛОЖЬ
            from claim_parser import extract_designations
            for i, sv in enumerate(parsed["sub_verdicts"]):
                if sv.get("status") != "ПРАВДА":
                    continue
                sc_text = sv.get("text", sub_claims[i] if i < len(sub_claims) else "")
                citation = sv.get("citation", sv.get("quote", ""))
                sc_desig = extract_designations(sc_text)
                cite_desig = extract_designations(citation)
                if sc_desig and citation and citation != "ЦИТАТА ОТСУТСТВУЕТ":
                    for d in sc_desig:
                        # Extract the base name (letters before digits)
                        base = re.match(r'([А-Яа-яA-Za-z]+)', d)
                        if not base:
                            continue
                        base_str = base.group(1).lower()
                        # Check if this exact designation is in citation
                        if d.lower() not in citation.lower():
                            # Check if a different designation of same series exists
                            cite_same_series = [
                                cd for cd in cite_desig
                                if cd.lower() != d.lower()
                                and re.match(r'([А-Яа-яA-Za-z]+)', cd)
                                and re.match(r'([А-Яа-яA-Za-z]+)', cd).group(1).lower() == base_str
                            ]
                            if cite_same_series:
                                sv["status"] = "ЛОЖЬ"
                                print(f"  [A4:DesignCheck] Sub-fact {i+1}: '{d}' not in source, "
                                      f"found '{cite_same_series[0]}' instead → ЛОЖЬ")
                                break

            # A4: Per-sub-claim NLI cross-validation
            if self.nli_checker and len(sub_claims) > 1:
                for i, sv in enumerate(parsed["sub_verdicts"]):
                    if i >= len(sub_claims):
                        break
                    sc = sub_claims[i]
                    sc_nli = self.nli_checker.check_claim(sc, raw_results[:5], snippet_key="snippet")
                    sc_max_con = sc_nli.get("max_contradiction", 0)
                    sc_max_ent = sc_nli.get("max_entailment", 0)
                    # If LLM says ПРАВДА but NLI strongly contradicts → override
                    if sv["status"] == "ПРАВДА" and sc_max_con >= 0.70 and sc_max_ent < 0.40:
                        sv["status"] = "ЛОЖЬ"
                        print(f"  [A4:SubNLI] '{sc[:40]}': LLM=ПРАВДА but NLI con={sc_max_con:.2f} → ЛОЖЬ")
                    # If LLM says ЛОЖЬ but NLI strongly entails → override
                    elif sv["status"] == "ЛОЖЬ" and sc_max_ent >= 0.70 and sc_max_con < 0.30:
                        sv["status"] = "ПРАВДА"
                        print(f"  [A4:SubNLI] '{sc[:40]}': LLM=ЛОЖЬ but NLI ent={sc_max_ent:.2f} → ПРАВДА")

            # Post-audit numeric override: if num_comparisons show mismatch
            # and sub-claim contains that number → force ЛОЖЬ
            if num_comparisons:
                for nc in num_comparisons:
                    if nc.get("source_number") and not nc.get("match"):
                        mismatched_raw = nc["claim_number"].get("raw", "")
                        mismatched_val = str(nc["claim_number"].get("value", ""))
                        for sv in parsed["sub_verdicts"]:
                            if sv["status"] == "ПРАВДА":
                                sv_text = sv.get("text", "")
                                if mismatched_raw in sv_text or mismatched_val in sv_text:
                                    sv["status"] = "ЛОЖЬ"
                                    print(f"  [NumOverride] Sub-fact '{sv_text[:40]}' contains "
                                          f"mismatched number {mismatched_raw} → ЛОЖЬ")

            # Fix 4: Пересчитываем verdict по условной матрице
            statuses = [sv["status"] for sv in parsed["sub_verdicts"]]
            count_true = sum(1 for s in statuses if s == "ПРАВДА")
            count_false = sum(1 for s in statuses if s == "ЛОЖЬ")
            count_nodata = sum(1 for s in statuses if s in ("НЕТ ДАННЫХ", "НЕТ"))
            total = len(statuses)

            # V9: FActScore-style metric — % подтверждённых атомарных фактов
            # Вместо бинарной матрицы используем непрерывный FActScore
            if total > 0:
                factscore = count_true / total  # 0.0 - 1.0
                composite_score = int(
                    (count_true * 100 + count_nodata * 50 + count_false * 0) / total
                )
            else:
                factscore = 0.0
                composite_score = 50

            parsed["_factscore"] = round(factscore, 2)
            print(f"  [FActScore] {count_true}/{total} confirmed = {factscore:.0%}")

            # Core fact detection for severity assessment
            _core_fact_keywords = (
                'океан', 'море', 'город', 'стран', 'контин', 'планет',
                'основа', 'создал', 'родил', 'столиц', 'президент',
                'река', 'озеро', 'гора', 'остров',
            )
            _is_core_false = False
            for sv in parsed["sub_verdicts"]:
                if sv["status"] == "ЛОЖЬ":
                    sv_text = sv.get("claim", sv.get("text", "")).lower()
                    if any(kw in sv_text for kw in _core_fact_keywords):
                        _is_core_false = True
                        break

            # V9: Three-zone FActScore verdict
            if factscore >= 0.80:
                # >80% confirmed → ПРАВДА
                parsed["verdict"] = "ПРАВДА"
                parsed["credibility_score"] = max(80, composite_score)
            elif factscore <= 0.30:
                # <30% confirmed → ЛОЖЬ
                parsed["verdict"] = "ЛОЖЬ"
                parsed["credibility_score"] = max(5, composite_score)
                print(f"  [FActScore] ЛОЖЬ: {factscore:.0%} < 30%")
            elif _is_core_false:
                # Core fact false → ЛОЖЬ regardless of FActScore
                parsed["verdict"] = "ЛОЖЬ"
                parsed["credibility_score"] = max(5, composite_score)
                print(f"  [FActScore] ЛОЖЬ (core fact false): {count_true}/{total}")
            elif count_false > 0 and count_false >= count_true:
                # More false than true → ЛОЖЬ
                parsed["verdict"] = "ЛОЖЬ"
                parsed["credibility_score"] = max(5, composite_score)
                print(f"  [FActScore] ЛОЖЬ: {count_false} ложь >= {count_true} правда")
            elif count_false > 0 and count_true > count_false:
                # 30-80% confirmed, some false → ЧАСТИЧНО ВЕРНО / МАНИПУЛЯЦИЯ
                parsed["verdict"] = "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА"
                parsed["credibility_score"] = composite_score
                print(f"  [FActScore] МАНИПУЛЯЦИЯ: {factscore:.0%} ({count_true} правда + {count_false} ложь)")
            elif count_nodata == total:
                parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = 50
            elif count_true > 0 and count_nodata > 0:
                # Some confirmed, rest unknown → ЧАСТИЧНО ПОДТВЕРЖДЕНО
                parsed["verdict"] = "ЧАСТИЧНО ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = composite_score
            _has_audit_matrix = True
        parsed.pop("_raw_sub_verdicts_count", None)

        # Post-audit scam safety net: стем-концепт детектор на полном claim
        # даже если substring-based is_scam был False
        is_scam = self._ctx.is_scam
        if not is_scam:
            from claim_parser import detect_scam_concepts
            concept = detect_scam_concepts(claim)
            if concept["is_scam"]:
                zero_ev = self._check_zero_evidence(claim, nli_result)
                warning_src = self._sources_are_scam_warnings(claim, raw_results, nli_result)
                if zero_ev or warning_src:
                    is_scam = True
                    self._ctx.is_scam = True
                    reason = "zero evidence" if zero_ev else "sources are warnings"
                    print(f"  [ScamSafetyNet] {concept['n_groups']} концепт-групп + "
                          f"{reason} → override to ЛОЖЬ")

        # Scam override: принудительно ЛОЖЬ, независимо от audit matrix
        if is_scam:
            parsed["verdict"] = "ЛОЖЬ"
            parsed["credibility_score"] = max(5, min(parsed.get("credibility_score", 15), 20))
            print("  [Scam Override] is_scam=True → вердикт принудительно ЛОЖЬ")
        elif _has_audit_matrix and not parsed.get("_claim_drift"):
            # V8: Audit matrix + ensemble = единая система
            # Audit matrix дал вердикт, но проверяем Wikidata/числа/NLI — они могут override
            _audit_verdict = parsed["verdict"]
            _audit_score = parsed["credibility_score"]

            # Wikidata override: если Wikidata явно противоречит audit-вердикту
            wikidata = self._ctx.wikidata_result
            wd_contradicted = 0
            wd_confirmed = 0
            if wikidata and wikidata.get("found"):
                wd_facts = wikidata.get("facts", [])
                wd_contradicted = sum(1 for f in wd_facts if f.get("match") is False)
                wd_confirmed = sum(1 for f in wd_facts if f.get("match") is True)
                # Check neutral facts that actually match claim text
                claim_lower = claim.lower()
                claim_stems = set(w[:4] for w in re.findall(r'[а-яёa-z]{4,}', claim_lower))
                for f in wd_facts:
                    if f.get("match") is None:
                        for v in f.get("wikidata_values", []):
                            v_stem = v.lower()[:4] if len(v) >= 4 else v.lower()
                            if v_stem in claim_stems or v.lower() in claim_lower:
                                wd_confirmed += 1
                                break

            # Числа override
            nums_mismatch = (
                any(c for c in num_comparisons if c["source_number"] and not c["match"])
                and not any(c for c in num_comparisons if c["source_number"] and c["match"])
            )
            nums_match = any(c for c in num_comparisons if c["source_number"] and c["match"])

            # NLI override
            nli_ent = nli_result.get("max_entailment", 0.0) if nli_result else 0.0
            nli_con = nli_result.get("max_contradiction", 0.0) if nli_result else 0.0

            # MiniCheck override
            mc_result = self._ctx.minicheck_result
            mc_signal = mc_result.get("signal", 0) if mc_result else 0

            _overridden = False
            # Если audit говорит ПРАВДА но Wikidata/числа/NLI/MiniCheck противоречат
            if _audit_verdict == "ПРАВДА":
                override_signals = []
                if wd_contradicted > 0:
                    override_signals.append(f"Wikidata противоречит ({wd_contradicted})")
                if nums_mismatch:
                    override_signals.append("Числа расходятся")
                if nli_con >= 0.70 and nli_ent < 0.35:
                    override_signals.append(f"NLI contradiction={nli_con:.2f}")
                if mc_signal <= -2:
                    override_signals.append(f"MiniCheck не подтверждает (signal={mc_signal})")
                if override_signals:
                    parsed["verdict"] = "ЛОЖЬ"
                    parsed["credibility_score"] = max(5, min(35, _audit_score))
                    _overridden = True
                    print(f"  [V8:AuditOverride] Audit=ПРАВДА, но {'; '.join(override_signals)} → ЛОЖЬ")

            # Если audit говорит ЛОЖЬ/НЕ ПОДТВЕРЖДЕНО но Wikidata/числа подтверждают
            elif _audit_verdict in ("НЕ ПОДТВЕРЖДЕНО", "ЧАСТИЧНО ПОДТВЕРЖДЕНО"):
                confirm_signals = []
                if wd_confirmed > 0 and wd_contradicted == 0:
                    confirm_signals.append(f"Wikidata подтверждает ({wd_confirmed})")
                if nums_match:
                    confirm_signals.append("Числа совпадают")
                if nli_ent >= 0.70 and nli_con < 0.30:
                    confirm_signals.append(f"NLI entailment={nli_ent:.2f}")
                if mc_signal >= 2:
                    confirm_signals.append(f"MiniCheck подтверждает (signal={mc_signal})")
                if len(confirm_signals) >= 2:
                    parsed["verdict"] = "ПРАВДА"
                    parsed["credibility_score"] = max(70, _audit_score)
                    _overridden = True
                    print(f"  [V8:AuditOverride] Audit={_audit_verdict}, но {'; '.join(confirm_signals)} → ПРАВДА")

            if not _overridden:
                print(f"  [V8:Audit] Audit verdict сохранён: {_audit_verdict} (score={_audit_score})")
        elif not parsed.get("_claim_drift"):
            # NLI ensemble verdict (A5 sufficiency gating is inside _ensemble_verdict)
            parsed = self._ensemble_verdict(parsed, nli_result, num_comparisons, raw_results, claim=claim)
        print(f"  Ensemble verdict: {parsed['verdict']} (score={parsed['credibility_score']})")

        # V9: Self-consistency — majority vote из N прогонов LLM
        # Запускаем только когда вердикт не скам и не claim-drift
        if (getattr(self.pipeline_config, 'enable_self_consistency', False)
                and not self._ctx.is_scam
                and not parsed.get("_claim_drift")):
            try:
                parsed = self._self_consistency_vote(state, parsed)
                print(f"  SC verdict: {parsed['verdict']} "
                      f"(agreement={parsed.get('_sc_agreement', 1.0):.0%})")
            except Exception as e:
                logger.warning(f"Self-consistency failed: {e}")

        # Post-verdict self-critique
        critique_info = {}
        if self.pipeline_config.enable_self_critique and "self_critique" in raw_result:
            raw_critique = raw_result["self_critique"]
            critique_info = self._parse_self_critique(raw_critique)
            parsed = self._apply_self_critique(parsed, critique_info)
            logger.info(
                f"Self-critique: errors={critique_info.get('errors', 'нет')}, "
                f"correction={critique_info.get('needs_correction', False)}"
            )
            print(f"  Self-critique: коррекция={'да' if critique_info.get('needs_correction') else 'нет'}")

        # Извлечение источников из thread-local
        raw_search = self._ctx.raw_results
        sources = self.extract_sources(raw_search) if raw_search else []

        # Ключевые слова
        keywords = self._parse_keywords(raw_result.get("keywords_raw", ""))

        # V9: Three-zone confidence calibration
        # Зона 1 (>75): высокая уверенность → вердикт остаётся
        # Зона 2 (35-75): средняя уверенность → НЕ ПОДТВЕРЖДЕНО
        # Зона 3 (<35): высокая уверенность в противоположном → вердикт остаётся
        raw_score = parsed["credibility_score"]
        sc_agreement = parsed.get("_sc_agreement", 1.0)
        _calibrated = False

        # Calibration only when self-consistency shows disagreement
        if sc_agreement < 0.67:  # Less than 2/3 agree
            if 35 <= raw_score <= 75:
                parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = 50
                _calibrated = True
                print(f"  [Calibration] score={raw_score}, SC agreement={sc_agreement:.0%} "
                      f"→ НЕ ПОДТВЕРЖДЕНО (zone 2)")

        # Additional calibration: if NLI and LLM disagree, lower confidence
        if not _calibrated and nli_result:
            nli_ent = nli_result.get("max_entailment", 0.0)
            nli_con = nli_result.get("max_contradiction", 0.0)
            llm_verdict = parsed["verdict"]
            if llm_verdict == "ПРАВДА" and nli_con > nli_ent and nli_con >= 0.5:
                # LLM says true but NLI says contradiction
                parsed["credibility_score"] = min(raw_score, 65)
            elif llm_verdict == "ЛОЖЬ" and nli_ent > nli_con and nli_ent >= 0.5:
                # LLM says false but NLI says entailment
                parsed["credibility_score"] = max(raw_score, 35)

        logger.info(f"Вердикт: {parsed['verdict']} (score={parsed['credibility_score']})")
        logger.info(f"Источников: {len(sources)}, Ключевые слова: {keywords}")

        result = {
            "claim": claim,
            "sub_claims": self._ctx.sub_claims,
            "credibility_score": parsed["credibility_score"],
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "chain_of_thought": parsed.get("chain_of_thought", ""),
            "sub_verdicts": parsed.get("sub_verdicts", []),
            "factscore": parsed.get("_factscore"),
            "sources": sources,
            "sources_text": parsed["sources"],
            "keywords": keywords,
            "search_results_formatted": raw_result.get("search_results", ""),
            "raw_verdict": parsed["raw"],
            "self_critique": critique_info.get("raw", ""),
            "self_critique_errors": critique_info.get("errors", ""),
            "_ensemble_features": parsed.get("_ensemble_features", {}),
            "_sc_agreement": parsed.get("_sc_agreement"),
            "_sc_verdicts": parsed.get("_sc_verdicts"),
            "total_time": round(total_time, 2),
        }

        # Fact Cache: сохраняем результат
        self.fact_cache.set(claim, result)

        return result


# === Вспомогательные функции для парсинга ключевых слов ===

def _rejoin_entities(keywords: List[str]) -> List[str]:
    """Склеивание известных многословных сущностей.

    Если модель разбила "ЦБ РФ" на ["ЦБ", "РФ"] — склеиваем обратно.
    """
    if len(keywords) < 2:
        return keywords

    result = list(keywords)

    for pattern_a, pattern_b, joined in _MULTI_WORD_ENTITIES:
        # Ищем пару подряд идущих ключевых слов, которые нужно склеить
        i = 0
        while i < len(result) - 1:
            a_match = re.search(pattern_a, result[i], re.IGNORECASE)
            b_match = re.search(pattern_b, result[i + 1], re.IGNORECASE)
            if a_match and b_match:
                # Склеиваем
                result[i] = joined
                result.pop(i + 1)
            else:
                i += 1

    return result


def _deduplicate_keywords(keywords: List[str]) -> List[str]:
    """Дедупликация: убирает подстроки и дубликаты.

    Если "ЦБ" и "ЦБ РФ" оба есть — убираем "ЦБ".
    Если "путин" и "Путин" оба есть — оставляем один.
    """
    if len(keywords) < 2:
        return keywords

    # Сначала убираем точные дубликаты (case-insensitive)
    seen = {}
    unique = []
    for kw in keywords:
        key = kw.lower().strip()
        if key not in seen:
            seen[key] = True
            unique.append(kw)

    # Затем убираем подстроки: если "ЦБ" содержится в "ЦБ РФ" — убираем "ЦБ"
    result = []
    for i, kw in enumerate(unique):
        is_substring = False
        kw_lower = kw.lower().strip()
        for j, other in enumerate(unique):
            if i == j:
                continue
            other_lower = other.lower().strip()
            # kw является подстрокой other и kw короче other
            if kw_lower in other_lower and len(kw_lower) < len(other_lower):
                is_substring = True
                break
        if not is_substring:
            result.append(kw)

    return result
