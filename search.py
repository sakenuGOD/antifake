"""SerpAPI обёртка для поиска новостей + ранжирование по косинусному сходству."""

from typing import List, Dict, Tuple
from serpapi import GoogleSearch

from config import SearchConfig

# Порог релевантности для bag-of-words cosine similarity.
# Для точного (embedding-based) сходства порог 0.85 корректен (Раздел 2.4),
# но bag-of-words без лемматизации даёт низкие скоры из-за морфологии русского языка
# (например, "снежного" != "снежный"), поэтому пороги снижены.
RELEVANCE_THRESHOLD = 0.35  # Подтверждающий источник
RELATED_THRESHOLD = 0.20    # Тематически близкий источник


def _simple_tokenize(text: str) -> Dict[str, int]:
    """Токенизация текста с псевдо-стеммингом для русского языка.

    Для каждого слова длиннее 4 символов добавляет его 4-символьный
    префикс как дополнительный токен. Это компенсирует отсутствие
    полноценной лемматизации: "снежного" и "снежный" → общий токен "снеж".
    """
    words = text.lower().split()
    freq = {}
    for w in words:
        w = w.strip(".,!?;:\"'()-[]{}»«")
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
            # Псевдо-стемминг: 4-символьный префикс для морфологических вариаций
            if len(w) > 4:
                stem = w[:4]
                freq[stem] = freq.get(stem, 0) + 1
    return freq


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Косинусное сходство между двумя текстами (bag-of-words + псевдо-стемминг)."""
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
        try:
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
        except Exception as e:
            print(f"  [SerpAPI] Ошибка при поиске '{keyword}': {e}")
            return []

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
        related_threshold: float = RELATED_THRESHOLD,
    ) -> List[Dict[str, str]]:
        """Ранжирование результатов по косинусному сходству с утверждением.

        Источники с similarity >= threshold считаются подтверждающими.
        Источники с similarity >= related_threshold считаются тематически близкими.
        Результаты сортируются по убыванию релевантности.
        """
        scored = []
        for article in results:
            text = f"{article.get('title', '')} {article.get('snippet', '')}"
            score = cosine_similarity(claim, text)
            article["relevance_score"] = round(score, 3)
            article["is_confirming"] = score >= threshold
            article["is_related"] = score >= related_threshold and score < threshold
            scored.append(article)

        # Сортировка по релевантности (убывание)
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored

    @staticmethod
    def format_results(results: List[Dict[str, str]]) -> str:
        """Форматирование результатов для LLM (топ-7, сниппеты до 200 символов)."""
        if not results:
            return "Новости по данному запросу не найдены."

        confirming = [r for r in results if r.get("is_confirming")]
        related = [r for r in results if r.get("is_related")]
        total = len(results)

        lines = [
            f"СТАТИСТИКА: найдено {total} источников, "
            f"подтверждающих: {len(confirming)}, "
            f"тематически близких: {len(related)}."
        ]
        lines.append("")

        for i, article in enumerate(results[:7], 1):
            score = article.get("relevance_score", 0)
            if article.get("is_confirming"):
                status = "ПОДТВЕРЖДАЕТ"
            elif article.get("is_related"):
                status = "БЛИЗКАЯ ТЕМА"
            else:
                status = "НЕ СВЯЗАН"
            parts = [f"{i}. [{status}, сходство: {score}] {article['title']}"]
            if article.get("source"):
                parts.append(f"   Источник: {article['source']}")
            if article.get("date"):
                parts.append(f"   Дата: {article['date']}")
            if article.get("snippet"):
                snippet = article["snippet"][:200]
                parts.append(f"   {snippet}")
            lines.append("\n".join(parts))

        return "\n\n".join(lines)
