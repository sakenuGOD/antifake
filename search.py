"""SerpAPI обёртка для поиска новостей."""

from typing import List, Dict
from serpapi import GoogleSearch

from config import SearchConfig


class FactCheckSearcher:
    """Поиск новостей через SerpAPI для проверки фактов."""

    def __init__(self, config: SearchConfig = None):
        if config is None:
            config = SearchConfig()
        self.config = config

    def search_keyword(self, keyword: str) -> List[Dict[str, str]]:
        """Поиск новостей по одному ключевому слову."""
        params = {
            "q": keyword.strip(),
            "engine": "google",
            "gl": self.config.gl,
            "hl": self.config.hl,
            "tbm": self.config.tbm,
            "num": self.config.num_results,
            "api_key": self.config.api_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        articles = []
        for item in results.get("news_results", []):
            articles.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "link": item.get("link", ""),
                "date": item.get("date", ""),
            })
        return articles

    def search_all_keywords(self, keywords: List[str]) -> List[Dict[str, str]]:
        """Поиск по всем ключевым словам с дедупликацией по URL."""
        seen_urls = set()
        all_results = []

        for keyword in keywords:
            if not keyword.strip():
                continue
            articles = self.search_keyword(keyword)
            for article in articles:
                url = article["link"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(article)

        return all_results

    @staticmethod
    def format_results(results: List[Dict[str, str]]) -> str:
        """Форматирование результатов в нумерованный текстовый блок для LLM."""
        if not results:
            return "Новости по данному запросу не найдены."

        lines = []
        for i, article in enumerate(results, 1):
            parts = [f"{i}. {article['title']}"]
            if article.get("source"):
                parts.append(f"   Источник: {article['source']}")
            if article.get("date"):
                parts.append(f"   Дата: {article['date']}")
            if article.get("snippet"):
                parts.append(f"   {article['snippet']}")
            lines.append("\n".join(parts))

        return "\n\n".join(lines)
