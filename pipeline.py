"""LangChain LCEL цепочка для проверки фактов."""

import os
from typing import List

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
        # Убираем возможные артефакты вывода
        text = raw_output.strip()

        # Убираем возможные маркеры формата
        for prefix in ["Ключевые слова:", "Keywords:", "ключевые слова:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Разделяем по запятой
        keywords = [kw.strip().strip('"').strip("'") for kw in text.split(",")]

        # Фильтруем пустые и слишком длинные (вероятно, мусор)
        keywords = [kw for kw in keywords if kw and len(kw) < 50]

        # Fallback: если парсинг не дал результатов, разбиваем по пробелам
        if not keywords:
            keywords = [w for w in text.split() if len(w) > 2][:5]

        return keywords[:5]

    def check(self, claim: str) -> dict:
        """Проверка одного утверждения. Возвращает полный словарь состояния."""
        print(f"\nПроверка: {claim}")
        result = self.chain.invoke({"claim": claim})
        return result
