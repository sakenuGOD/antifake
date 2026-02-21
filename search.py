"""SerpAPI обёртка для поиска новостей + ранжирование по косинусному сходству."""

from typing import List, Dict, Tuple
from serpapi import GoogleSearch

from config import SearchConfig

# Порог релевантности (Раздел 2.4 документации: Threshold >= 0.85)
RELEVANCE_THRESHOLD = 0.85


def _simple_tokenize(text: str) -> Dict[str, int]:
    """Простая токенизация текста для вычисления TF-вектора."""
    words = text.lower().split()
    freq = {}
    for w in words:
        w = w.strip(".,!?;:\"'()-[]{}»«")
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    return freq


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Косинусное сходство между двумя текстами (bag-of-words)."""
    vec_a = _simple_tokenize(text_a)
    vec_b = _simple_tokenize(text_b)

    if not vec_a or not vec_b:
        return 0.0

    # Скалярное произведение
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)

    # Нормы
    norm_a = sum(v ** 2 for v in vec_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in vec_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


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
    def rank_by_relevance(
        claim: str,
        results: List[Dict[str, str]],
        threshold: float = RELEVANCE_THRESHOLD,
    ) -> List[Dict[str, str]]:
        """Ранжирование результатов по косинусному сходству с утверждением.

        Источники с similarity >= threshold считаются подтверждающими.
        Результаты сортируются по убыванию релевантности.
        """
        scored = []
        for article in results:
            text = f"{article.get('title', '')} {article.get('snippet', '')}"
            score = cosine_similarity(claim, text)
            article["relevance_score"] = round(score, 3)
            article["is_confirming"] = score >= threshold
            scored.append(article)

        # Сортировка по релевантности (убывание)
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored

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
