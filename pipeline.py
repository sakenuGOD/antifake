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
        """Сборка LCEL цепочки с прогрессивным обогащением состояния."""
        keyword_prompt = PromptTemplate(
            template=KEYWORD_EXTRACTION_TEMPLATE,
            input_variables=["claim"],
        )
        verdict_prompt = PromptTemplate(
            template=CREDIBILITY_ASSESSMENT_TEMPLATE,
            input_variables=["claim", "search_results"],
        )

        # Шаг 1: Извлечение ключевых слов
        keyword_chain = keyword_prompt | self.keyword_llm | StrOutputParser()

        # Шаг 2: Поиск новостей (как функция)
        def search_step(state: dict) -> str:
            keywords = self._parse_keywords(state["keywords_raw"])
            print(f"  Ключевые слова: {keywords}")
            results = self.searcher.search_all_keywords(keywords)
            print(f"  Найдено новостей: {len(results)}")
            # Ранжирование по косинусному сходству (Раздел 2.4)
            claim = state.get("claim", "")
            results = self.searcher.rank_by_relevance(claim, results)
            confirming = [r for r in results if r.get("is_confirming")]
            print(f"  Подтверждающих (similarity >= 0.85): {len(confirming)}")
            # Сохраняем сырые результаты поиска в состояние
            state["_raw_search_results"] = results
            return self.searcher.format_results(results)

        # Шаг 3: Оценка достоверности
        verdict_chain = verdict_prompt | self.verdict_llm | StrOutputParser()

        # Полная цепочка через RunnablePassthrough.assign()
        chain = (
            RunnablePassthrough.assign(
                keywords_raw=keyword_chain,
            )
            | RunnablePassthrough.assign(
                search_results=RunnableLambda(search_step),
            )
            | RunnablePassthrough.assign(
                verdict=verdict_chain,
            )
        )
        return chain

    @staticmethod
    def _parse_keywords(raw_output: str) -> List[str]:
        """Парсер ключевых слов с fallback для нечистого вывода модели."""
        text = raw_output.strip()

        for prefix in ["Ключевые слова:", "Keywords:", "ключевые слова:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        keywords = [kw.strip().strip('"').strip("'") for kw in text.split(",")]
        keywords = [kw for kw in keywords if kw and len(kw) < 50]

        if not keywords:
            keywords = [w for w in text.split() if len(w) > 2][:5]

        return keywords[:5]

    @staticmethod
    def parse_verdict(raw_verdict: str) -> Dict[str, Any]:
        """Парсинг структурированного вердикта из ответа модели."""
        result = {
            "credibility_score": 50,
            "verdict": "НЕИЗВЕСТНО",
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

    def check(self, claim: str) -> Dict[str, Any]:
        """Проверка одного утверждения. Возвращает структурированный результат."""
        print(f"\nПроверка: {claim}")
        timings = {}

        # Шаг 1: Ключевые слова
        t0 = time.time()
        state = {"claim": claim}

        # Запускаем полную цепочку
        total_start = time.time()
        raw_result = self.chain.invoke(state)
        total_time = time.time() - total_start

        # Парсинг вердикта
        parsed = self.parse_verdict(raw_result.get("verdict", ""))

        # Извлечение источников из результатов поиска
        raw_search = raw_result.get("_raw_search_results", [])
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
