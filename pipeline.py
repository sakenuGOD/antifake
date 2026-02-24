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

        # Thread-local хранилище для промежуточных данных
        # (RunnablePassthrough.assign создаёт новый dict, мутации state не пробрасываются)
        # Используем threading.local() для thread-safety в multi-user сценариях (Streamlit)
        self._local = threading.local()

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
            print(f"  Ключевые слова: {keywords}")
            results = self.searcher.search_all_keywords(keywords, claim=claim)
            print(f"  Найдено новостей: {len(results)}")

            # Ранжирование по релевантности
            results = self.searcher.rank_by_relevance(claim, results)
            print(f"  Отсортировано по релевантности, топ-7 покажем модели")

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
            self._local.search_hint = hint
            self._local.raw_results = results

            return self.searcher.format_results(results)

        # Шаг 2.5: Извлечение search_hint из thread-local (отдельный assign)
        def get_search_hint(state: dict) -> str:
            return getattr(self._local, "search_hint", "")

        # Шаг 2.6: Классификация утверждения и генерация подсказок для проверки
        def get_verification_hints(state: dict) -> str:
            claim = state.get("claim", "")
            claim_info = classify_claim(claim)
            self._local.claim_info = claim_info
            hints = format_verification_hints(claim_info)
            if hints:
                print(f"  Тип утверждения: {claim_info['type']}, "
                      f"чисел: {len(claim_info['numbers'])}, дат: {len(claim_info['dates'])}")
            return hints

        # Шаг 2.7: NLI анализ — DeBERTa классифицирует claim vs каждый source
        def nli_analysis_step(state: dict) -> str:
            if self.nli_checker is None:
                self._local.nli_result = None
                return ""

            raw_results = getattr(self._local, "raw_results", [])
            if not raw_results:
                self._local.nli_result = {
                    "max_entailment": 0.0, "max_contradiction": 0.0,
                    "entailment_count": 0, "contradiction_count": 0,
                }
                return "NLI АНАЛИЗ: источники не найдены."

            claim = state.get("claim", "")
            nli_result = self.nli_checker.check_claim(claim, raw_results, snippet_key="snippet")
            self._local.nli_result = nli_result

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
            if not claim_numbers:
                self._local.num_comparisons = []
                return ""

            raw_results = getattr(self._local, "raw_results", [])
            all_comparisons = []
            for source in raw_results:
                snippet = source.get("snippet", "")
                source_numbers = extract_numbers(snippet)
                if source_numbers:
                    comps = compare_numbers(claim_numbers, source_numbers)
                    all_comparisons.extend(comps)
            self._local.num_comparisons = all_comparisons

            mismatches = [c for c in all_comparisons if c["source_number"] and not c["match"]]
            matches = [c for c in all_comparisons if c["source_number"] and c["match"]]

            if mismatches:
                parts = []
                for m in mismatches[:3]:
                    cv = m["claim_number"]["raw"]
                    sv = m["source_number"]["raw"]
                    dev = m["deviation"]
                    parts.append(f"claim={cv}, источник={sv} (расхождение {dev*100:.0f}%)")
                hint = "ЧИСЛОВАЯ ПРОВЕРКА: РАСХОЖДЕНИЕ! " + "; ".join(parts)
                print(f"  Числа: РАСХОЖДЕНИЕ ({len(mismatches)} несовпадений)")
                return hint
            elif matches:
                parts = []
                for m in matches[:3]:
                    cv = m["claim_number"]["raw"]
                    sv = m["source_number"]["raw"]
                    parts.append(f"{cv}={sv}")
                print(f"  Числа: совпадают ({len(matches)})")
                return "ЧИСЛОВАЯ ПРОВЕРКА: числа совпадают: " + ", ".join(parts)
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

        raw_keywords = [kw for kw in raw_keywords if kw and len(kw) < 60]

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

    def _ensemble_verdict(
        self,
        parsed: Dict[str, Any],
        nli_result: Dict[str, Any],
        num_comparisons: List[Dict],
        raw_results: List[Dict],
    ) -> Dict[str, Any]:
        """NLI-first ensemble verdict.

        Приоритет: числовая проверка > NLI DeBERTa > LLM reasoning.
        """
        verdict = parsed["verdict"].upper().strip()
        score = parsed["credibility_score"]

        has_sources = len(raw_results) > 0
        nums_mismatch = any(
            c for c in num_comparisons
            if c["source_number"] and not c["match"]
        )

        nli_ent = 0.0
        nli_con = 0.0
        if nli_result:
            nli_ent = nli_result.get("max_entailment", 0.0)
            nli_con = nli_result.get("max_contradiction", 0.0)

        # Rule 1: Числовое расхождение → ЛОЖЬ (детерминистика, 100%)
        if nums_mismatch:
            parsed["verdict"] = "ЛОЖЬ"
            parsed["credibility_score"] = min(score, 20)
            logger.info("Ensemble: Rule 1 — числовое расхождение → ЛОЖЬ")
            return parsed

        # Rule 2: NLI contradiction сильная → ЛОЖЬ
        if nli_con >= 0.70 and nli_ent < 0.40:
            parsed["verdict"] = "ЛОЖЬ"
            parsed["credibility_score"] = min(score, 25)
            logger.info(f"Ensemble: Rule 2 — NLI contradiction={nli_con:.2f} → ЛОЖЬ")
            return parsed

        # Rule 3: NLI entailment сильная + числа ok → ПРАВДА
        if nli_ent >= 0.65 and nli_con < 0.30 and not nums_mismatch:
            parsed["verdict"] = "ПРАВДА"
            parsed["credibility_score"] = max(score, 75)
            logger.info(f"Ensemble: Rule 3 — NLI entailment={nli_ent:.2f} → ПРАВДА")
            return parsed

        # Rule 4: Нет источников
        if not has_sources:
            if verdict == "ПРАВДА" and score > 70:
                # common knowledge, cap confidence
                parsed["credibility_score"] = min(score, 80)
            elif verdict == "ЛОЖЬ":
                pass  # LLM уверена что ложь — доверяем
            else:
                parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
                parsed["credibility_score"] = min(score, 55)
            logger.info("Ensemble: Rule 4 — нет источников")
            return parsed

        # Rule 5: LLM ПРАВДА но NLI не подтверждает
        if verdict == "ПРАВДА" and nli_ent < 0.40 and has_sources:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["credibility_score"] = min(score, 55)
            logger.info(f"Ensemble: Rule 5 — LLM ПРАВДА но NLI ent={nli_ent:.2f} → НЕ ПОДТВЕРЖДЕНО")
            return parsed

        # Rule 6: LLM ЛОЖЬ но NLI entailment (LLM ошиблась)
        if verdict == "ЛОЖЬ" and nli_con < 0.30 and nli_ent > 0.50:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["credibility_score"] = 45
            logger.info(f"Ensemble: Rule 6 — LLM ЛОЖЬ но NLI ent={nli_ent:.2f} → НЕ ПОДТВЕРЖДЕНО")
            return parsed

        # Default: trust LLM + basic consistency check
        if verdict == "ПРАВДА" and score < 40:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
        elif verdict == "ЛОЖЬ" and score > 60:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"

        return parsed

    def check(self, claim: str) -> Dict[str, Any]:
        """Проверка одного утверждения. Возвращает структурированный результат."""
        logger.info(f"Проверка: {claim[:120]}")
        print(f"\nПроверка: {claim}")

        # Сброс промежуточного состояния (thread-local)
        self._local.search_hint = ""
        self._local.raw_results = []
        self._local.nli_result = None
        self._local.num_comparisons = []

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
        raw_results = getattr(self._local, "raw_results", [])
        nli_result = getattr(self._local, "nli_result", None)
        num_comparisons = getattr(self._local, "num_comparisons", [])
        parsed = self.parse_verdict(raw_result.get("verdict", ""))

        # NLI ensemble verdict (заменяет _apply_safety_net)
        parsed = self._ensemble_verdict(parsed, nli_result, num_comparisons, raw_results)
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
        raw_search = getattr(self._local, "raw_results", [])
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
