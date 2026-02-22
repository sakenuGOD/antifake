"""LangChain LCEL цепочка для проверки фактов."""

import logging
import os
import re
import time
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import ModelConfig, PipelineConfig, SearchConfig
from prompts import KEYWORD_EXTRACTION_TEMPLATE, CREDIBILITY_ASSESSMENT_TEMPLATE
from model import load_unsloth_model, load_finetuned_model, build_langchain_llm
from search import FactCheckSearcher

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
}


class FactCheckPipeline:
    """Пайплайн проверки достоверности утверждений."""

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

        # Промежуточное хранилище (RunnablePassthrough.assign создаёт новый dict,
        # поэтому мутации state внутри функций НЕ пробрасываются в следующие шаги)
        self._last_search_hint = ""
        self._last_raw_results = []

        # Загрузка модели один раз
        if adapter_path and os.path.exists(adapter_path):
            print(f"Загрузка fine-tuned модели из {adapter_path}...")
            model, tokenizer = load_finetuned_model(adapter_path, model_config)
        else:
            print("Загрузка base модели...")
            model, tokenizer = load_unsloth_model(model_config)
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)

        # Два LLM-wrapper с разными max_new_tokens
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

        # Сборка цепочки
        self.chain = self._build_chain()

    def _build_chain(self):
        """Сборка LCEL цепочки с прогрессивным обогащением состояния.

        ВАЖНО: RunnablePassthrough.assign() создаёт НОВЫЙ dict — мутации
        исходного state внутри RunnableLambda не видны следующим шагам.
        Поэтому промежуточные данные (search_hint, raw_results) сохраняются
        на self, а для search_hint добавлен отдельный шаг assign().
        """
        keyword_prompt = PromptTemplate(
            template=KEYWORD_EXTRACTION_TEMPLATE,
            input_variables=["claim"],
        )
        verdict_prompt = PromptTemplate(
            template=CREDIBILITY_ASSESSMENT_TEMPLATE,
            input_variables=["claim", "search_results", "search_hint"],
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

            # Ранжирование по косинусному сходству (только для сортировки)
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

            # Сохраняем на self (не через мутацию state!)
            self._last_search_hint = hint
            self._last_raw_results = results

            return self.searcher.format_results(results)

        # Шаг 2.5: Извлечение search_hint из self (отдельный assign)
        def get_search_hint(state: dict) -> str:
            return self._last_search_hint

        # Шаг 3: Оценка достоверности (Quote-Then-Judge + Triple Self-Check)
        verdict_chain = verdict_prompt | self.verdict_llm | StrOutputParser()

        # Полная цепочка:
        # claim → +keywords_raw → +search_results → +search_hint → +verdict
        chain = (
            RunnablePassthrough.assign(
                keywords_raw=keyword_chain,
            )
            | RunnablePassthrough.assign(
                search_results=RunnableLambda(search_step),
            )
            | RunnablePassthrough.assign(
                search_hint=RunnableLambda(get_search_hint),
            )
            | RunnablePassthrough.assign(
                verdict=verdict_chain,
            )
        )
        return chain

    @staticmethod
    def _parse_keywords(raw_output: str) -> List[str]:
        """Парсер ключевых слов с программной очисткой.

        1. Извлекает слова из вывода модели
        2. Программно удаляет: месяцы, числа, общие слова
        3. Fallback если всё отфильтровано
        """
        text = raw_output.strip()

        # Убираем типичные префиксы
        for prefix in ["Ключевые слова:", "Keywords:", "ключевые слова:",
                       "КЛЮЧЕВЫЕ СЛОВА:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Парсинг по запятой
        raw_keywords = [kw.strip().strip('"').strip("'") for kw in text.split(",")]
        raw_keywords = [kw for kw in raw_keywords if kw and len(kw) < 50]

        # Fallback: если не разделено запятыми
        if not raw_keywords:
            raw_keywords = [w for w in text.split() if len(w) > 2][:5]

        # === Программная очистка (модель 7B не всегда слушается инструкций) ===
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

            # Пропуск слишком коротких
            if len(kw_lower) < 2:
                continue

            cleaned.append(kw)

        # Fallback: если всё отфильтровано, вернуть оригинал
        result = cleaned if cleaned else raw_keywords
        return result[:5]

    @staticmethod
    def parse_verdict(raw_verdict: str) -> Dict[str, Any]:
        """Парсинг структурированного вердикта из ответа модели."""
        result = {
            "credibility_score": 50,
            "verdict": "НЕ ПОДТВЕРЖДЕНО",
            "confidence": 50,
            "reasoning": "",
            "sources": "",
            "raw": raw_verdict,
        }

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

    @staticmethod
    def _apply_safety_net(
        parsed: Dict[str, Any],
        raw_results: List[Dict],
    ) -> Dict[str, Any]:
        """Минимальный safety net: проверка консистентности вердикта.

        НЕ использует cosine similarity labels (они ненадёжны для русского).
        Проверяет только внутреннюю логику ответа модели.

        Правила:
        1. Нет результатов поиска вообще → НЕ ПОДТВЕРЖДЕНО (нельзя судить без данных)
        2. Вердикт и score противоречат друг другу → приводим в соответствие
        """
        score = parsed["credibility_score"]
        verdict = parsed["verdict"].upper().strip()

        # Rule 1: Нет результатов поиска → нельзя сказать ПРАВДА или ЛОЖЬ
        if not raw_results and verdict in ("ПРАВДА", "ЛОЖЬ"):
            parsed["credibility_score"] = max(30, min(score, 60))
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["reasoning"] += (
                " [Коррекция: источники не найдены — "
                "невозможно подтвердить или опровергнуть]"
            )

        # Rule 2: Вердикт ПРАВДА но score низкий → несоответствие
        elif verdict == "ПРАВДА" and score < 60:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"

        # Rule 3: Вердикт ЛОЖЬ но score высокий → несоответствие
        elif verdict == "ЛОЖЬ" and score > 40:
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"

        return parsed

    def check(self, claim: str) -> Dict[str, Any]:
        """Проверка одного утверждения. Возвращает структурированный результат."""
        logger.info(f"Проверка: {claim[:120]}")
        print(f"\nПроверка: {claim}")

        # Сброс промежуточного состояния
        self._last_search_hint = ""
        self._last_raw_results = []

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
                "total_time": 0,
            }
        finally:
            # Гарантированная очистка промежуточного состояния
            pass

        # Парсинг вердикта + smart safety net
        parsed = self.parse_verdict(raw_result.get("verdict", ""))
        parsed = self._apply_safety_net(parsed, self._last_raw_results)

        # Извлечение источников из self (не из chain output)
        raw_search = self._last_raw_results
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
            "sources": sources,
            "sources_text": parsed["sources"],
            "keywords": keywords,
            "search_results_formatted": raw_result.get("search_results", ""),
            "raw_verdict": parsed["raw"],
            "total_time": round(total_time, 2),
        }
