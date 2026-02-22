"""LangChain LCEL цепочка для проверки фактов."""

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

            # Ранжирование по косинусному сходству
            results = self.searcher.rank_by_relevance(claim, results)
            confirming = [r for r in results if r.get("is_confirming")]
            related = [r for r in results if r.get("is_related")]
            print(f"  Подтверждающих: {len(confirming)}, близких: {len(related)}")

            # Генерация подсказки для LLM на основе статистики
            total = len(results)
            if not results:
                hint = (
                    "ПОДСКАЗКА: По запросу не найдено новостей. "
                    "Невозможно подтвердить или опровергнуть. "
                    "Оцени правдоподобность на основе общих знаний. "
                    "Вердикт: НЕ ПОДТВЕРЖДЕНО."
                )
            elif confirming:
                hint = (
                    f"ПОДСКАЗКА: {len(confirming)} из {total} источников "
                    f"подтверждают утверждение. Проанализируй их содержание."
                )
            elif related:
                hint = (
                    f"ПОДСКАЗКА: Прямых подтверждений нет, но {len(related)} "
                    f"из {total} источников обсуждают близкую тему. "
                    f"Внимательно прочитай их — возможно, они подтверждают "
                    f"утверждение другими словами."
                )
            else:
                hint = (
                    f"ПОДСКАЗКА: Найдено {total} источников, но ни один "
                    f"не имеет высокого тематического сходства. "
                    f"Вероятно, новость не освещена в СМИ. "
                    f"Это НЕ опровержение — вердикт: НЕ ПОДТВЕРЖДЕНО."
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

            # Пропуск чисто числовых или "число + слово" (e.g. "22 год", "33 бойца")
            if re.match(r'^\d+\s*\S*$', kw_lower):
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
        """Умный safety net: целевая коррекция только очевидных ошибок модели.

        Правила (минимальное вмешательство):
        1. Модель сказала ПРАВДА, но НЕТ подтверждающих источников → НЕ ПОДТВЕРЖДЕНО
        2. Модель сказала ЛОЖЬ, но ЕСТЬ подтверждающие источники → НЕ ПОДТВЕРЖДЕНО
        3. Модель сказала ЛОЖЬ, но поиск ВООБЩЕ не дал результатов → НЕ ПОДТВЕРЖДЕНО
        4. Во всех остальных случаях — доверяем модели
        """
        confirming = [r for r in raw_results if r.get("is_confirming")]
        score = parsed["credibility_score"]
        verdict = parsed["verdict"].upper().strip()

        # Rule 1: ПРАВДА без подтверждающих источников → снижаем
        if verdict == "ПРАВДА" and not confirming:
            parsed["credibility_score"] = min(score, 55)
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["reasoning"] += (
                " [Коррекция: вердикт снижен — подтверждающие источники не найдены]"
            )

        # Rule 2: ЛОЖЬ при наличии подтверждающих источников → повышаем
        elif verdict == "ЛОЖЬ" and confirming:
            parsed["credibility_score"] = max(score, 60)
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["reasoning"] += (
                " [Коррекция: найдены подтверждающие источники, "
                "вердикт ЛОЖЬ необоснован]"
            )

        # Rule 3: ЛОЖЬ без каких-либо результатов поиска → нельзя опровергнуть
        elif verdict == "ЛОЖЬ" and not raw_results:
            parsed["credibility_score"] = max(score, 40)
            parsed["verdict"] = "НЕ ПОДТВЕРЖДЕНО"
            parsed["reasoning"] += (
                " [Коррекция: поиск не дал результатов — "
                "невозможно подтвердить или опровергнуть]"
            )

        # Rule 4: всё остальное — доверяем модели
        return parsed

    def check(self, claim: str) -> Dict[str, Any]:
        """Проверка одного утверждения. Возвращает структурированный результат."""
        print(f"\nПроверка: {claim}")

        # Сброс промежуточного состояния
        self._last_search_hint = ""
        self._last_raw_results = []

        state = {"claim": claim}

        # Запускаем полную цепочку
        total_start = time.time()
        raw_result = self.chain.invoke(state)
        total_time = time.time() - total_start

        # Парсинг вердикта + smart safety net
        parsed = self.parse_verdict(raw_result.get("verdict", ""))
        parsed = self._apply_safety_net(parsed, self._last_raw_results)

        # Извлечение источников из self (не из chain output)
        raw_search = self._last_raw_results
        sources = self.extract_sources(raw_search) if raw_search else []

        # Ключевые слова
        keywords = self._parse_keywords(raw_result.get("keywords_raw", ""))

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
