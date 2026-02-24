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
    SELF_CRITIQUE_TEMPLATE,
)
from model import load_base_model, load_finetuned_model, build_langchain_llm, is_grpo_adapter
from search import FactCheckSearcher
from claim_parser import classify_claim, format_verification_hints, extract_numbers, compare_numbers

logger = logging.getLogger("fact_checker")

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
            input_variables=["claim", "search_results", "search_hint",
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
            print(f"  Найдено новостей: {len(results)}")

            # Ранжирование по релевантности
            results = self.searcher.rank_by_relevance(claim, results)
            print(f"  После ранжирования: {len(results)} релевантных (топ-7 → модели)")

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
            claim = state.get("claim", "")
            claim_info = classify_claim(claim)
            self._shared["claim_info"] =claim_info
            hints = format_verification_hints(claim_info)
            if hints:
                print(f"  Тип утверждения: {claim_info['type']}, "
                      f"чисел: {len(claim_info['numbers'])}, дат: {len(claim_info['dates'])}")
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
            nli_sources = sorted(
                raw_results,
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

            if con >= 0.60:
                return (f"NLI АНАЛИЗ: {con_c} источник(ов) ОПРОВЕРГАЮТ утверждение "
                        f"(contradiction={con:.2f}). Есть прямое противоречие с фактами.")
            elif ent >= 0.60:
                return (f"NLI АНАЛИЗ: {ent_c} источник(ов) подтверждают утверждение "
                        f"(entailment={ent:.2f}).")
            else:
                return (f"NLI АНАЛИЗ: источники не дают чёткого сигнала "
                        f"(entailment={ent:.2f}, contradiction={con:.2f}).")

        # Шаг 2.8: Детерминистическая проверка чисел — claim vs sources
        def number_check_step(state: dict) -> str:
            claim = state.get("claim", "")
            claim_numbers = extract_numbers(claim)
            # Фильтр: маленькие числа (< 10) типа "number" — это шум
            # ("1 января" → 1, "2 раза" → 2), не годятся для фактчекинга
            claim_numbers = [n for n in claim_numbers
                            if not (n["type"] == "number" and n["value"] < 10)]
            if not claim_numbers:
                self._shared["num_comparisons"] =[]
                return ""

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

        # Шаг 3: Оценка достоверности
        verdict_chain = verdict_prompt | self.verdict_llm | StrOutputParser()

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

        Поддерживает два формата:
        1. Стандартный: ДОСТОВЕРНОСТЬ: ... ВЕРДИКТ: ... (SFT модель)
        2. XML: <reasoning>...</reasoning><answer>...</answer> (GRPO модель)
        """
        result = {
            "credibility_score": 50,
            "verdict": "НЕ ПОДТВЕРЖДЕНО",
            "confidence": 50,
            "reasoning": "",
            "chain_of_thought": "",
            "sources": "",
            "raw": raw_verdict,
        }

        # Извлечение chain-of-thought из <reasoning> тегов (если есть)
        cot_match = re.search(
            r"<reasoning>(.*?)</reasoning>", raw_verdict, re.DOTALL
        )
        if cot_match:
            result["chain_of_thought"] = cot_match.group(1).strip()

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
        # NLI score = (ent * ent_count - con * con_count), clamped to [-3, +3]
        if has_sources and not nli_mixed:
            # Чистый сигнал (не mixed) — полный вес
            nli_score = (nli_ent * max(nli_ent_count, 1) - nli_con * max(nli_con_count, 1))
            nli_vote = max(-3.0, min(3.0, nli_score))
            vote += nli_vote
            logger.info(f"Vote: NLI ent={nli_ent:.2f}×{nli_ent_count} con={nli_con:.2f}×{nli_con_count} → {nli_vote:+.2f}")
        elif has_sources and nli_mixed:
            # Mixed signal — NLI ненадёжен, только count-based с малым весом
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
                missing = unique_details - source_words
                found = unique_details & source_words
                # Ключевые детали отсутствуют в sources → подмена
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
        self._shared["search_hint"] =""
        self._shared["raw_results"] =[]
        self._shared["nli_result"] =None
        self._shared["num_comparisons"] =[]

        state = {"claim": claim}

        try:
            # Запускаем полную цепочку
            total_start = time.time()
            raw_result = self.chain.invoke(state)
            total_time = time.time() - total_start
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
        parsed = self.parse_verdict(raw_result.get("verdict", ""))

        # NLI ensemble verdict (заменяет _apply_safety_net)
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
            "credibility_score": parsed["credibility_score"],
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "chain_of_thought": parsed.get("chain_of_thought", ""),
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
