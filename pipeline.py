"""LangChain LCEL цепочка для проверки фактов.

v4: NLI-first ensemble — DeBERTa классифицирует, LLM объясняет.
"""

import logging
import os
import re
import threading
import time
from typing import List, Dict, Any

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
)
from model import load_base_model, load_finetuned_model, build_langchain_llm, is_grpo_adapter
from search import FactCheckSearcher, boost_factcheck_scores, validate_context_entities
from claim_parser import classify_claim, format_verification_hints, extract_numbers, compare_numbers

logger = logging.getLogger("fact_checker")

# Промпт для декомпозиции составных утверждений (Проблема 1)
_DECOMPOSE_TEMPLATE = """\
Если утверждение содержит несколько отдельных фактов через союзы "и", "а", "а также", "при этом", "однако", "но", "кроме того", "где" — выпиши каждый факт ОТДЕЛЬНО, каждый с новой строки.
ВАЖНО: восстанавливай подлежащее в каждом предложении (если написано "а её столица — X" → пиши "Официальная столица страны — X").
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
_GENERIC_WORDS = {
    "женщина", "мужчина", "человек", "люди", "год", "день", "время",
    "место", "новость", "сообщение", "данные", "информация",
    "также", "который", "этот", "свой", "быть", "весь",
    "такой", "другой", "какой", "наш", "после", "перед",
}

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

        # Shared state dict для промежуточных данных между шагами LCEL цепочки.
        # ВАЖНО: НЕ используем threading.local() — LangChain RunnablePassthrough.assign
        # выполняет lambdas в worker-потоках (ThreadPoolExecutor), и thread-local
        # данные из main thread невидимы в worker threads.
        # Для multi-user (Streamlit) используем threading.Lock для синхронизации.
        self._shared = {}
        self._lock = threading.Lock()

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
                print("Загрузка CrossEncoder re-ranker...")
                self._reranker = ReRanker()
            except Exception as e:
                print(f"CrossEncoder не загружен: {e}")

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

        # Шаг 1: Извлечение ключевых слов
        keyword_chain = keyword_prompt | self.keyword_llm | StrOutputParser()

        # Шаг 2: Поиск новостей — сохраняет hint и raw_results на self
        def search_step(state: dict) -> str:
            keywords = self._parse_keywords(state["keywords_raw"])
            claim = state.get("claim", "")
            keywords = self._validate_keywords(keywords, claim)
            print(f"  Ключевые слова: {keywords}")
            results = self.searcher.search_all_keywords(keywords, claim=claim)
            wiki_cnt = sum(1 for r in results if 'wikipedia.org' in r.get('link', ''))
            print(f"  Найдено новостей: {len(results)} (Wikipedia: {wiki_cnt})")

            # Сохраняем Wikipedia ДО ранжирования — semantic ranker может их отсечь
            self._shared["wiki_results"] = [r for r in results if 'wikipedia.org' in r.get('link', '')]

            # Дополнительный поиск для каждого под-утверждения (Проблема 1)
            sub_claims = self._shared.get("sub_claims", [])
            if len(sub_claims) > 1:
                existing_urls = {r.get("link", "") for r in results}
                for sc in sub_claims:
                    sc_results = self.searcher.search_all_keywords([sc], claim=sc)
                    for r in sc_results:
                        url = r.get("link", "")
                        if url and url not in existing_urls:
                            results.append(r)
                            existing_urls.add(url)
                print(f"  + под-утверждения: {len(results)} источников итого")

            # Ранжирование по релевантности (bi-encoder)
            results = self.searcher.rank_by_relevance(claim, results)
            wiki_in_top = sum(1 for r in results if 'wikipedia.org' in r.get('link', ''))
            print(f"  После ранжирования: {len(results)} релевантных (Wikipedia в топ: {wiki_in_top}, топ-7 → модели)")

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
            self._shared["search_hint"] =hint
            self._shared["raw_results"] =results

            return self.searcher.format_results(results)

        # Шаг 2.5: Извлечение search_hint из thread-local (отдельный assign)
        def get_search_hint(state: dict) -> str:
            return self._shared.get("search_hint", "")

        # Шаг 2.6: Классификация утверждения и генерация подсказок для проверки
        def get_verification_hints(state: dict) -> str:
            from claim_parser import detect_scam_patterns
            claim = state.get("claim", "")
            claim_info = classify_claim(claim)
            self._shared["claim_info"] = claim_info
            self._shared["claim_locations"] = claim_info.get("locations", [])
            self._shared["is_scam"] = claim_info.get("is_scam", False)
            hints = format_verification_hints(claim_info)
            
            # Проверяем локации из claim в уже найденных источниках
            raw_results = self._shared.get("raw_results", [])
            claim_locs = claim_info.get("locations", [])
            if claim_locs and raw_results:
                _trans_map = {
                    'токио': ['tokyo', 'tokio'], 'париж': ['paris'],
                    'берлин': ['berlin'], 'лондон': ['london'],
                    'катар': ['qatar'], 'японии': ['japan'],
                    'германии': ['germany'], 'австралии': ['australia'],
                    'китае': ['china'], 'сочи': ['sochi'],
                    'пекин': ['beijing'], 'москва': ['moscow'],
                }
                src_text = " ".join(
                    (r.get("snippet", "") + " " + r.get("title", "")).lower()
                    for r in raw_results[:10]
                )
                confirmed = []
                missing_loc = []
                for loc in claim_locs:
                    ll = loc.lower()
                    found = ll in src_text or any(t in src_text for t in _trans_map.get(ll, []))
                    (confirmed if found else missing_loc).append(loc)
                
                if confirmed and not missing_loc:
                    conf_str = ", ".join(confirmed)
                    hints += (f"\nПОДТВЕРЖДЕНИЕ: ключевая локация [{conf_str}] "
                             f"ПРИСУТСТВУЕТ в источниках — место указано верно. "
                             f"Не доверяй NLI если он слабо противоречит.")
                elif missing_loc and not confirmed:
                    miss_str = ", ".join(missing_loc)
                    hints += (f"\nПРЕДУПРЕЖДЕНИЕ: локация [{miss_str}] "
                             f"НЕ НАЙДЕНА в источниках — место, возможно, неверно.")

            if claim_info["type"] or hints:
                print(f"  Тип утверждения: {claim_info['type']}, "
                      f"чисел: {len(claim_info['numbers'])}, дат: {len(claim_info['dates'])}, "
                      f"локаций: {len(claim_locs)}, скам: {claim_info.get('is_scam', False)}")

            return hints

        # Шаг 2.7: NLI анализ — DeBERTa классифицирует claim vs TOP релевантных sources
        def nli_analysis_step(state: dict) -> str:
            if self.nli_checker is None:
                self._shared["nli_result"] = None
                return ""

            raw_results = self._shared.get("raw_results", [])
            if not raw_results:
                print("  NLI: 0 источников после фильтрации — пропуск")
                self._shared["nli_result"] = {
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
            # Транслитерация для топонимов в стемах
            _stem_trans = {
                "париж": ["paris"], "токио": ["tokyo"], "лондо": ["londo"],
                "берли": ["berli"], "пекин": ["beiji", "pekin"], "катар": ["qatar"],
                "сидне": ["sydne"], "канбе": ["canbe"], "вашин": ["washi"],
            }
            _expanded_stems = set(claim_stems)
            for _s, _alts in _stem_trans.items():
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
            _wiki_all = self._shared.get("wiki_results", [])
            _wiki_relevant = []
            _claim_type = self._shared.get("claim_info", {}).get("type", "general") if self._shared.get("claim_info") else "general"
            _claim_locations = self._shared.get("claim_locations", [])
            if _claim_locations and _claim_type in ("date", "date_loc"):
                # Строгий фильтр: год И локация должны быть в тексте статьи
                _claim_years = re.findall(r'\b(?:19|20)\d{2}\b', state.get("claim", ""))
                _loc_stems_strict = set()
                for _loc in _claim_locations:
                    for _w in re.findall(r"[а-яёА-ЯЁa-zA-Z]{4,}", _loc):
                        _loc_stems_strict.add(_w.lower()[:5])
                for _s, _alts in _stem_trans.items():
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

            claim = state.get("claim", "")
            nli_result = self.nli_checker.check_claim(claim, nli_sources, snippet_key="snippet")
            self._shared["nli_result"] = nli_result

            ent = nli_result["max_entailment"]
            con = nli_result["max_contradiction"]
            ent_c = nli_result["entailment_count"]
            con_c = nli_result["contradiction_count"]
            print(f"  NLI: entailment={ent:.2f} ({ent_c} src), contradiction={con:.2f} ({con_c} src)")
            # DEBUG: показываем TOP-5 источники и их NLI-оценки
            for _i, _src in enumerate(nli_sources):
                _sc = _src.get("final_score", _src.get("semantic_score", 0))
                print(f"    NLI src {_i+1}: [{_sc:.2f}] {_src.get('source','')} | {_src.get('title','')[:60]}")

            if con >= 0.80 and ent < 0.35:
                # Чёткое противоречие: высокий con + очень низкий ent
                return (f"NLI АНАЛИЗ: {con_c} источник(ов) ПРОТИВОРЕЧАТ утверждению "
                        f"(contradiction={con:.2f}, entailment={ent:.2f}). "
                        f"Возможно, утверждение содержит неверные детали. Проверь сам.")
            elif ent >= 0.70 and con < 0.30:
                # Чёткое подтверждение без противоречий
                return (f"NLI АНАЛИЗ: {ent_c} источник(ов) ПОДТВЕРЖДАЮТ утверждение "
                        f"(entailment={ent:.2f}, contradiction={con:.2f}).")
            elif con >= 0.60 or ent >= 0.60:
                # Смешанный или умеренный сигнал — не делаем сильных выводов
                return (f"NLI АНАЛИЗ: смешанные сигналы "
                        f"(entailment={ent:.2f}×{ent_c}, contradiction={con:.2f}×{con_c}). "
                        f"Полагайся на контент источников, не только на NLI.")
            else:
                return (f"NLI АНАЛИЗ: слабые сигналы "
                        f"(entailment={ent:.2f}, contradiction={con:.2f}). "
                        f"Источники не дают чёткого сигнала.")

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
            
            raw_results = self._shared.get("raw_results", [])
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
                                self._shared["date_mismatch"] = True
                                print(f"  Дата: claim={cy}, sources={sorted(near_years)[:3]} — РАСХОЖДЕНИЕ")
                            else:
                                self._shared["date_mismatch"] = False
                        else:
                            self._shared["date_mismatch"] = False
            
            if not claim_numbers:
                self._shared["num_comparisons"] = []
                
                # Добавляем location confirmation hint если локации найдены в источниках
                claim_locations = self._shared.get("claim_locations", [])
                if claim_locations and raw_results:
                    transliteration_map = {
                        'токио': ['tokyo', 'tokio'], 'париж': ['paris'],
                        'берлин': ['berlin'], 'лондон': ['london'],
                        'катар': ['qatar'], 'японии': ['japan', 'japanese'],
                        'японии': ['japan'], 'германии': ['germany'],
                        'австралии': ['australia'], 'китае': ['china'],
                    }
                    source_text_all = " ".join(
                        (r.get("snippet", "") + " " + r.get("title", "")).lower()
                        for r in raw_results[:10]
                    )
                    confirmed_locs = []
                    for loc in claim_locations:
                        loc_lower = loc.lower()
                        found = loc_lower in source_text_all
                        if not found:
                            trans = transliteration_map.get(loc_lower, [])
                            found = any(t in source_text_all for t in trans)
                        if found:
                            confirmed_locs.append(loc)
                    
                    if confirmed_locs and len(confirmed_locs) == len(claim_locations):
                        loc_hint = (f"ПОДТВЕРЖДЕНИЕ ЛОКАЦИИ: место «{', '.join(confirmed_locs)}» "
                                   f"из утверждения найдено в источниках — это сигнал достоверности. "
                                   f"Если источники описывают это место в контексте события — склоняйся к ПРАВДА.")
                        return (date_hint + " " + loc_hint).strip() if date_hint else loc_hint       
                return date_hint

            raw_results = self._shared.get("raw_results", [])
            all_comparisons = []
            for source in raw_results:
                snippet = source.get("snippet", "")
                source_numbers = extract_numbers(snippet)
                if source_numbers:
                    comps = compare_numbers(claim_numbers, source_numbers)
                    all_comparisons.extend(comps)
            self._shared["num_comparisons"] =all_comparisons

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
            sub_claims = self._shared.get("sub_claims", [])
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
        # Выбор происходит динамически на основе self._shared["sub_claims"].

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
            sub_claims = self._shared.get("sub_claims", [])
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
            return sub_claims
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
        claim_words = set(re.findall(r'[а-яёa-z0-9]{3,}', claim_lower))
        # Stems: первые 5 символов для морфологического matching
        claim_stems = set(w[:5] for w in claim_words if len(w) >= 5)
        claim_stems.update(claim_words)  # короткие слова — как есть

        validated = []
        for kw in keywords:
            kw_words = set(re.findall(r'[а-яёa-z0-9]{3,}', kw.lower()))
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
            # Автовывод итогового вердикта: условная матрица (Fix 4)
            count_true = sum(1 for s in statuses if s == "ПРАВДА")
            count_false = sum(1 for s in statuses if s == "ЛОЖЬ")
            count_nodata = sum(1 for s in statuses if s in ("НЕТ ДАННЫХ", "НЕТ"))
            total = len(statuses)

            if count_true == total:
                # Все факты подтверждены
                result["_audit_verdict"] = "ПРАВДА"
                result["_audit_score"] = 85
            elif count_false > 0 and count_true > 0:
                # Смесь правды и лжи = МАНИПУЛЯЦИЯ
                true_pct = int(count_true / total * 100)
                result["_audit_verdict"] = "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА"
                result["_audit_score"] = true_pct
                print(f"  [Audit] МАНИПУЛЯЦИЯ: {count_true} правда + {count_false} ложь из {total}")
            elif count_false > 0 and count_true == 0:
                # Только ложь (и, возможно, НЕТ ДАННЫХ)
                result["_audit_verdict"] = "ЛОЖЬ"
                result["_audit_score"] = max(5, int(count_nodata / total * 20))
            elif count_nodata == total:
                # Всё неизвестно
                result["_audit_verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                result["_audit_score"] = 50
            elif count_true > 0 and count_nodata > 0 and count_false == 0:
                # Часть подтверждена, остальное неизвестно
                result["_audit_verdict"] = "ЧАСТИЧНО ПОДТВЕРЖДЕНО"
                result["_audit_score"] = int(count_true / total * 70 + 15)

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

        # Fallback: если модель не написала ДОСТОВЕРНОСТЬ/ВЕРДИКТ (аудиторский формат
        # иногда не дублирует их), берём _audit_verdict вычисленный из СТАТУС-пунктов.
        if result["verdict"] == "НЕ ПОДТВЕРЖДЕНО" and result["credibility_score"] == 50:
            audit_v = result.get("_audit_verdict")
            audit_s = result.get("_audit_score")
            if audit_v:
                result["verdict"] = audit_v
                result["credibility_score"] = audit_s
                print(f"  [Audit:Fallback] Вердикт из протокола: {audit_v} (score={audit_s})")

        # Убираем внутренние ключи перед возвратом
        result.pop("_audit_verdict", None)
        result.pop("_audit_score", None)

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

    def _ensemble_verdict(
        self,
        parsed: Dict[str, Any],
        nli_result: Dict[str, Any],
        num_comparisons: List[Dict],
        raw_results: List[Dict],
        claim: str = "",
    ) -> Dict[str, Any]:
        # Определяем тип утверждения для адаптации весов
        claim_info = self._shared.get("claim_info", {})
        claim_type = claim_info.get("type", "general") if claim_info else "general"

        """Ensemble verdict v5 — простой, робастный, универсальный.

        Принцип: NLI score — это голоса. Суммируем голоса и принимаем решение.
        Никаких хрупких порогов. Три сигнала голосуют: числа, NLI, LLM.
        """
        verdict = parsed["verdict"].upper().strip()
        score = parsed["credibility_score"]

        has_sources = len(raw_results) > 0

        # --- Извлечение сигналов ---
        nums_match = any(c for c in num_comparisons if c["source_number"] and c["match"])
        nums_mismatch = (
            any(c for c in num_comparisons if c["source_number"] and not c["match"])
            and not nums_match
        )
        match_count = sum(1 for c in num_comparisons if c["source_number"] and c["match"])
        total_with_source = sum(1 for c in num_comparisons if c["source_number"])

        nli_ent = nli_result.get("max_entailment", 0.0) if nli_result else 0.0
        nli_con = nli_result.get("max_contradiction", 0.0) if nli_result else 0.0
        nli_ent_count = nli_result.get("entailment_count", 0) if nli_result else 0
        nli_con_count = nli_result.get("contradiction_count", 0) if nli_result else 0

        # --- Feature vector (для логирования и будущего meta-classifier) ---
        match_ratio = match_count / total_with_source if total_with_source > 0 else 0.0
        nli_mixed = nli_ent >= 0.40 and nli_con >= 0.40
        nli_clean_ent = nli_ent >= 0.50 and nli_con < 0.35
        nli_clean_con = nli_con >= 0.50 and nli_ent < 0.35
        parsed["_ensemble_features"] = {
            "nli_ent": nli_ent, "nli_con": nli_con,
            "nli_ent_count": nli_ent_count, "nli_con_count": nli_con_count,
            "llm_verdict": 1 if verdict == "ПРАВДА" else (-1 if verdict == "ЛОЖЬ" else 0),
            "llm_score": score, "nums_match": int(nums_match),
            "nums_mismatch": int(nums_mismatch), "match_ratio": match_ratio,
            "num_sources": len(raw_results), "nli_mixed": int(nli_mixed),
            "nli_clean_ent": int(nli_clean_ent), "nli_clean_con": int(nli_clean_con),
        }

        # --- META-CLASSIFIER (если обучен — заменяет всю логику ниже) ---
        if hasattr(self, '_meta_classifier') and self._meta_classifier is not None:
            return self._meta_classify(parsed, parsed["_ensemble_features"])

        # ============================================================
        # SCORING SYSTEM: каждый сигнал голосует за ПРАВДА (+) или ЛОЖЬ (-)
        # Итоговый score определяет вердикт
        # ============================================================
        vote = 0.0  # положительный = ПРАВДА, отрицательный = ЛОЖЬ

        # --- ГОЛОС 1: Числа (самый надёжный, вес 3) ---
        if nums_mismatch:
            vote -= 3.0
            logger.info("Vote: NUMS_MISMATCH → -3")
        elif nums_match:
            vote += 3.0
            logger.info(f"Vote: NUMS_MATCH ({match_count}/{total_with_source}) → +3")

        # --- ГОЛОС 2: NLI (вес ±3 — может перебить LLM при сильном сигнале) ---
        # Для PERSON-типа снижаем вес NLI: модель путает "упоминаются вместе" с "роль подтверждена"
        # (Статья "Маск vs Цукерберг" упоминает обоих → NLI даёт entailment для "Цукерберг CEO Tesla")
        # NLI weight: снижаем для всех типов (NLI ненадёжен при смешанных сигналах)
        if claim_type == "person":
            nli_max_weight = 1.5  # Меньше для персон (co-occurrence ≠ role)
        else:
            nli_max_weight = 2.0  # Снижено с 3.0 для более осторожного голосования
        logger.info(f"Vote: NLI max_weight={nli_max_weight} (type={claim_type})")

        if has_sources and not nli_mixed:
            nli_score = (nli_ent * max(nli_ent_count, 1) - nli_con * max(nli_con_count, 1))
            nli_vote = max(-nli_max_weight, min(nli_max_weight, nli_score))
            vote += nli_vote
            logger.info(f"Vote: NLI ent={nli_ent:.2f}×{nli_ent_count} con={nli_con:.2f}×{nli_con_count} → {nli_vote:+.2f}")
        elif has_sources and nli_mixed:
            nli_vote = (nli_ent_count - nli_con_count) * 0.3
            nli_vote = max(-1.0, min(1.0, nli_vote))
            vote += nli_vote
            logger.info(f"Vote: NLI_MIXED ent_cnt={nli_ent_count} con_cnt={nli_con_count} → {nli_vote:+.2f}")

        # --- ГОЛОС 3: Detail matching (ключевые детали claim в источниках) ---
        # Проверяет что КОНКРЕТНЫЕ слова из claim упоминаются в source snippets.
        # Ловит подмену деталей: "юань" vs "иена", "Google" vs "OpenAI"
        if has_sources:
            # Извлекаем уникальные значимые слова из claim (> 3 символов)
            claim_detail_words = set(
                w.lower() for w in re.findall(r'[а-яёa-z]{4,}', claim)
            )
            # Собираем все слова из source snippets
            source_text = " ".join(
                (r.get("snippet", "") + " " + r.get("title", "")).lower()
                for r in raw_results[:7]
            )
            source_words = set(re.findall(r'[а-яёa-z]{4,}', source_text))

            # Проверяем ключевые уникальные детали claim в sources
            common_words = {
                "является", "составляет", "более", "менее", "около",
                "самый", "самая", "самое", "крупнейший", "крупнейшая",
                "первый", "первая", "второй", "третий", "мире", "мира",
                "страна", "город", "года", "году", "после", "стал",
                "стала", "имеет", "может", "будет", "были", "было",
                "также", "который", "которая", "которое", "одним",
            }
            if claim_detail_words:
                unique_details = claim_detail_words - common_words
                
                # Исключаем подтверждённые локации из проверки деталей:
                # они проверяются отдельно через транслитерацию (голос 6).
                # Используем 5-символьный стем для русских падежных форм:
                # "Париж"→стем "париж", "Париже" (локатив)→стем "париж" → совпадают
                confirmed_loc_stems = set()
                for loc in self._shared.get("claim_locations", []):
                    for w in re.findall(r"[а-яёА-ЯЁa-zA-Z]{4,}", loc):
                        confirmed_loc_stems.add(w.lower()[:5])
                # Исключаем слова чей 5-char стем совпадает со стемом локации
                unique_details_filtered = {
                    w for w in unique_details if w[:5] not in confirmed_loc_stems
                }
                
                missing = unique_details_filtered - source_words
                found = unique_details_filtered & source_words
                # Ключевые детали (без локаций) отсутствуют в sources → подмена
                if missing and len(missing) <= 3 and len(found) >= 2:
                    vote -= 2.0
                    logger.info(f"Vote: KEY_DETAIL_MISSING — '{', '.join(missing)}' "
                                f"not found in sources → -2.0")
                else:
                    overlap = claim_detail_words & source_words
                    overlap_ratio = len(overlap) / len(claim_detail_words)
                    if overlap_ratio < 0.40:
                        vote -= 1.0
                        logger.info(f"Vote: DETAIL_MISMATCH — {len(overlap)}/{len(claim_detail_words)} "
                                    f"claim words in sources → -1.0")

        # --- ГОЛОС 4: LLM (вес зависит от уверенности) ---
        if verdict == "ПРАВДА":
            # score 50-100 → вес 0-2
            llm_vote = (score - 50) / 25.0  # score=75 → +1, score=100 → +2
            llm_vote = max(0.0, min(2.0, llm_vote))
            vote += llm_vote
            logger.info(f"Vote: LLM ПРАВДА (score={score}) → +{llm_vote:.2f}")
        elif verdict == "ЛОЖЬ":
            # score 0-50 → вес -2 .. 0
            llm_vote = (score - 50) / 25.0  # score=25 → -1, score=0 → -2
            llm_vote = max(-2.0, min(0.0, llm_vote))
            vote += llm_vote
            logger.info(f"Vote: LLM ЛОЖЬ (score={score}) → {llm_vote:+.2f}")
        else:
            logger.info("Vote: LLM uncertain → 0")

        # --- ГОЛОС 5: Скам-паттерны (если обнаружены — сильный сигнал ЛОЖЬ) ---
        is_scam = self._shared.get("is_scam", False)
        if is_scam:
            vote -= 4.0
            logger.info("Vote: SCAM_PATTERN → -4.0 (мошеннический паттерн)")
        
        # --- ГОЛОС 5.5: Date mismatch (год в claim ≠ годам в источниках) ---
        date_mismatch = self._shared.get("date_mismatch", False)
        if date_mismatch:
            vote -= 2.0
            logger.info("Vote: DATE_YEAR_MISMATCH → -2.0")

        # --- ГОЛОС 6: Детекция несовпадения локаций (claim vs sources) ---
        # ВАЖНО: Не применяем если NLI entailment высокий (>= 0.50) — 
        # это означает что источники ПОДТВЕРЖДАЮТ claim, просто локация в другом языке
        # (напр: "Париж" в claim, но источник содержит "Paris" на английском)
        claim_locations = self._shared.get("claim_locations", [])
        # Только пропускаем location check при ЧЁТКОМ подтверждении (ent>=0.70, con<0.30)
        loc_guard_active = (nli_ent >= 0.70 and nli_con < 0.30)
        if claim_locations and has_sources and not loc_guard_active:
            source_text_all = " ".join(
                (r.get("snippet", "") + " " + r.get("title", "")).lower()
                for r in raw_results[:7]
            )
            # Транслитерация для основных топонимов (Рус → Лат)
            transliteration_map = {
                'токио': ['tokyo', 'tokio'],
                'париж': ['paris'],
                'берлин': ['berlin'],
                'лондон': ['london'],
                'москва': ['moscow', 'moskva'],
                'пекин': ['beijing', 'peking'],
                'катар': ['qatar'],
                'япония': ['japan', 'japanese'],
                'японии': ['japan', 'japanese'],
                'германии': ['germany', 'german'],
                'австралии': ['australia', 'australian'],
                'канада': ['canada', 'canadian'],
                'бразилия': ['brazil', 'brazilian'],
                'аргентина': ['argentina'],
                'китай': ['china', 'chinese'],
                'китае': ['china', 'chinese'],
                'корея': ['korea', 'korean'],
                'корее': ['korea', 'korean'],
            }

            # Для date/date_loc утверждений с явным годом — требуем
            # совместного появления года + локации в ОДНОМ сниппете.
            # Это исключает ложные "LOCATION_CONFIRMED" когда год и
            # локация упоминаются в разных источниках (напр: токио 2020 ≠ 2024).
            claim_years = re.findall(r'\b(?:19|20)\d{2}\b', claim)
            use_cooccurrence = bool(
                claim_years and claim_type in ("date", "date_loc")
            )
            # Список снипетов для поиска совместного вхождения
            source_snippets = [
                (r.get("snippet", "") + " " + r.get("title", "")).lower()
                for r in raw_results[:10]
            ]

            def _loc_found(loc_lower: str) -> bool:
                """Проверяет наличие локации (с транслитерацией) в источниках.

                Для date/date_loc утверждений требует HOSTING-контекст:
                "в {location}" рядом с годом, а не просто любое упоминание.
                Это исключает ложные срабатывания для команд-участников
                (напр: "Сборная Японии на ЧМ 2022" ≠ "прошёл в Японии 2022").
                """
                trans = transliteration_map.get(loc_lower, [])
                loc_variants = [loc_lower] + trans
                if not use_cooccurrence:
                    return any(v in source_text_all for v in loc_variants)
                # В контексте hosting: ищем "в {location}" рядом с годом
                # ("в Париже 2024", "2024 в Японии") — исключает роль участника
                hosting_pattern = f"в {loc_lower}"
                for snippet in source_snippets:
                    for yr in claim_years:
                        if yr not in snippet:
                            continue
                        yr_idx = snippet.index(yr)
                        window = snippet[max(0, yr_idx - 200): yr_idx + 200]
                        if hosting_pattern in window:
                            return True
                        # Транслитерации (для англоязычных источников: "in Tokyo")
                        for t in trans:
                            if f"in {t}" in window or f"в {t}" in window:
                                return True
                return False

            missing_locs = []
            for loc in claim_locations:
                loc_lower = loc.lower()
                found_in_sources = _loc_found(loc_lower)
                if not found_in_sources and len(loc_lower) > 3:
                    missing_locs.append(loc)
            if missing_locs and len(missing_locs) == len(claim_locations):
                vote -= 2.5
                logger.info(f"Vote: LOCATION_MISMATCH — локации {missing_locs} не в sources → -2.5")
            elif missing_locs:
                vote -= 1.0
                logger.info(f"Vote: LOCATION_PARTIAL_MISS — часть локаций не в sources → -1.0")
            else:
                # Все локации из claim найдены в sources → слабое подтверждение
                vote += 0.5
                logger.info(f"Vote: LOCATION_CONFIRMED — локации {claim_locations} найдены → +0.5")

        # --- ИТОГОВОЕ РЕШЕНИЕ ---
        logger.info(f"Vote TOTAL: {vote:+.2f}")

        if vote >= 1.5:
            parsed["verdict"] = "ПРАВДА"
            parsed["credibility_score"] = min(95, int(50 + vote * 10))
            logger.info(f"Ensemble: ПРАВДА (vote={vote:+.2f})")
        elif vote <= -1.5:
            parsed["verdict"] = "ЛОЖЬ"
            parsed["credibility_score"] = max(5, int(50 + vote * 10))
            logger.info(f"Ensemble: ЛОЖЬ (vote={vote:+.2f})")
        else:
            # Зона неопределённости — доверяем LLM, но с пониженной уверенностью
            if verdict == "ПРАВДА":
                parsed["credibility_score"] = min(score, 70)
            elif verdict == "ЛОЖЬ":
                parsed["credibility_score"] = max(score, 30)
            else:
                parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = 50
            logger.info(f"Ensemble: UNCERTAIN (vote={vote:+.2f}) → LLM fallback: {parsed['verdict']}")

        return parsed

    def check(self, claim: str) -> Dict[str, Any]:
        """Проверка одного утверждения. Возвращает структурированный результат."""
        logger.info(f"Проверка: {claim[:120]}")
        print(f"\nПроверка: {claim}")

        # Сброс промежуточного состояния (thread-local)
        self._shared["search_hint"] = ""
        self._shared["raw_results"] = []
        self._shared["nli_result"] = None
        self._shared["num_comparisons"] = []
        self._shared["claim_locations"] = []
        self._shared["is_scam"] = False
        self._shared["date_mismatch"] = False
        self._shared["sub_claims"] = []

        # Декомпозиция составных утверждений (Проблема 1)
        if self.pipeline_config.enable_claim_decomposition:
            sub_claims = self._decompose_claim(claim)
            if len(sub_claims) > 1:
                print(f"  Декомпозиция: {len(sub_claims)} под-утверждений")
                for i, sc in enumerate(sub_claims, 1):
                    print(f"    {i}. {sc}")
                self._shared["sub_claims"] = sub_claims

        state = {"claim": claim}

        try:
            # Запускаем полную цепочку
            total_start = time.time()
            raw_result = self.chain.invoke(state)
            total_time = time.time() - total_start
            # Лог сырого ответа модели (первые 300 символов)
            raw_v = raw_result.get("verdict", "")
            print(f"  [Model:raw] {len(raw_v)} символов | превью: {raw_v[:200].replace(chr(10), ' ')!r}")
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
        raw_results = self._shared.get("raw_results", [])
        nli_result = self._shared.get("nli_result", None)
        num_comparisons = self._shared.get("num_comparisons", [])
        trusted_cnt = sum(1 for r in raw_results if not r.get("_unverified"))
        print(f"  [Sources] Итого после фильтра: {len(raw_results)} "
              f"(доверенных: {trusted_cnt})")
        parsed = self.parse_verdict(raw_result.get("verdict", ""))

        # Fix 3: Ограничение парсера — обрезаем sub_verdicts до количества входных фактов
        sub_claims = self._shared.get("sub_claims", [])
        _has_audit_matrix = False
        if sub_claims and parsed.get("sub_verdicts"):
            n_facts = len(sub_claims)
            raw_count = parsed.get("_raw_sub_verdicts_count", len(parsed["sub_verdicts"]))
            if len(parsed["sub_verdicts"]) > n_facts:
                print(f"  [Audit:Trim] Обрезка: {raw_count} распарсено → {n_facts} фактов на входе")
                parsed["sub_verdicts"] = parsed["sub_verdicts"][:n_facts]
            # Fix 4: Пересчитываем verdict по условной матрице
            statuses = [sv["status"] for sv in parsed["sub_verdicts"]]
            count_true = sum(1 for s in statuses if s == "ПРАВДА")
            count_false = sum(1 for s in statuses if s == "ЛОЖЬ")
            count_nodata = sum(1 for s in statuses if s in ("НЕТ ДАННЫХ", "НЕТ"))
            total = len(statuses)

            if count_true == total:
                parsed["verdict"] = "ПРАВДА"
                parsed["credibility_score"] = 85
            elif count_false > 0 and count_true > 0:
                true_pct = int(count_true / total * 100)
                parsed["verdict"] = "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА"
                parsed["credibility_score"] = true_pct
                print(f"  [Audit:Matrix] МАНИПУЛЯЦИЯ: {count_true} правда + {count_false} ложь из {total}")
            elif count_false > 0 and count_true == 0:
                parsed["verdict"] = "ЛОЖЬ"
                parsed["credibility_score"] = max(5, int(count_nodata / total * 20))
            elif count_nodata == total:
                parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = 50
            elif count_true > 0 and count_nodata > 0:
                parsed["verdict"] = "ЧАСТИЧНО ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = int(count_true / total * 70 + 15)
            _has_audit_matrix = True
        parsed.pop("_raw_sub_verdicts_count", None)

        # NLI ensemble verdict (заменяет _apply_safety_net)
        # Для составных утверждений с audit матрицей — НЕ перезаписываем verdict через ensemble,
        # т.к. audit-verdict уже вычислен по per-fact результатам и более надёжен.
        if not _has_audit_matrix:
            parsed = self._ensemble_verdict(parsed, nli_result, num_comparisons, raw_results, claim=claim)
        print(f"  Ensemble verdict: {parsed['verdict']} (score={parsed['credibility_score']})")

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
        raw_search = self._shared.get("raw_results", [])
        sources = self.extract_sources(raw_search) if raw_search else []

        # Ключевые слова
        keywords = self._parse_keywords(raw_result.get("keywords_raw", ""))

        logger.info(f"Вердикт: {parsed['verdict']} (score={parsed['credibility_score']})")
        logger.info(f"Источников: {len(sources)}, Ключевые слова: {keywords}")

        return {
            "claim": claim,
            "sub_claims": self._shared.get("sub_claims", []),
            "credibility_score": parsed["credibility_score"],
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "chain_of_thought": parsed.get("chain_of_thought", ""),
            "sub_verdicts": parsed.get("sub_verdicts", []),
            "sources": sources,
            "sources_text": parsed["sources"],
            "keywords": keywords,
            "search_results_formatted": raw_result.get("search_results", ""),
            "raw_verdict": parsed["raw"],
            "self_critique": critique_info.get("raw", ""),
            "self_critique_errors": critique_info.get("errors", ""),
            "_ensemble_features": parsed.get("_ensemble_features", {}),
            "total_time": round(total_time, 2),
        }


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
