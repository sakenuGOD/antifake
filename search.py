"""Поиск новостей: SerpAPI (основной) → DuckDuckGo (fallback) + ранжирование."""

from typing import List, Dict
from config import SearchConfig

# Пороги для bag-of-words cosine similarity с псевдо-стеммингом.
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

    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)

    norm_a = sum(v ** 2 for v in vec_a.values()) ** 0.5
    norm_b = sum(v ** 2 for v in vec_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class FactCheckSearcher:
    """Поиск новостей: SerpAPI (основной) → DuckDuckGo (бесплатный fallback)."""

    def __init__(self, config: SearchConfig = None):
        if config is None:
            config = SearchConfig()
        self.config = config
        self._serpapi_failed = False  # Если True — все запросы идут через DDG

    def _search_serpapi(self, keyword: str) -> List[Dict[str, str]]:
        """Поиск через SerpAPI (платный, основной)."""
        from serpapi import GoogleSearch

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

    def _search_ddg(self, keyword: str) -> List[Dict[str, str]]:
        """Поиск через DuckDuckGo (бесплатный fallback, без API-ключа)."""
        from duckduckgo_search import DDGS

        results = DDGS().news(
            keywords=keyword.strip(),
            region="ru-ru",
            max_results=self.config.num_results,
        )
        articles = []
        for item in results:
            articles.append({
                "title": item.get("title", ""),
                "snippet": item.get("body", ""),
                "source": item.get("source", ""),
                "link": item.get("url", ""),
                "date": item.get("date", ""),
            })
        return articles

    def search_keyword(self, keyword: str) -> List[Dict[str, str]]:
        """Поиск новостей с автоматическим fallback: SerpAPI → DuckDuckGo.

        Если SerpAPI падает (ошибка, исчерпан лимит), все последующие
        запросы в этой сессии автоматически идут через DuckDuckGo.
        """
        # 1. SerpAPI (если ключ есть и API не упал)
        if self.config.api_key and not self._serpapi_failed:
            try:
                return self._search_serpapi(keyword)
            except Exception as e:
                print(f"  [SerpAPI] Ошибка: {e}")
                print(f"  [SerpAPI] Переключаюсь на DuckDuckGo для всех запросов")
                self._serpapi_failed = True

        # 2. DuckDuckGo (fallback)
        try:
            return self._search_ddg(keyword)
        except Exception as e:
            print(f"  [DDG] Ошибка при поиске '{keyword}': {e}")
            return []

    def search_all_keywords(
        self, keywords: List[str], claim: str = ""
    ) -> List[Dict[str, str]]:
        """Поиск по всем ключевым словам + комбинированный запрос + прямой поиск.

        Стратегия поиска (три уровня для максимального recall):
        1. Комбинированный запрос: все ключевые слова вместе
        2. Прямой поиск: первые 120 символов утверждения целиком
        3. Поиск по отдельным ключевым словам
        """
        seen_urls = set()
        all_results = []

        def _add_articles(articles: List[Dict[str, str]]):
            for article in articles:
                url = article["link"]
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(article)

        # 1. Комбинированный запрос (все ключевые слова вместе — лучший recall)
        combined_query = " ".join(kw.strip() for kw in keywords if kw.strip())
        if combined_query:
            _add_articles(self.search_keyword(combined_query))

        # 2. Прямой поиск утверждения (первые 120 символов)
        if claim.strip():
            direct_query = claim.strip()[:120]
            if direct_query != combined_query:
                _add_articles(self.search_keyword(direct_query))

        # 3. Поиск по отдельным ключевым словам
        for keyword in keywords:
            if not keyword.strip():
                continue
            _add_articles(self.search_keyword(keyword))

        return all_results

    @staticmethod
    def rank_by_relevance(
        claim: str,
        results: List[Dict[str, str]],
        threshold: float = RELEVANCE_THRESHOLD,
        related_threshold: float = RELATED_THRESHOLD,
    ) -> List[Dict[str, str]]:
        """Ранжирование результатов по косинусному сходству с утверждением."""
        scored = []
        for article in results:
            text = f"{article.get('title', '')} {article.get('snippet', '')}"
            score = cosine_similarity(claim, text)
            article["relevance_score"] = round(score, 3)
            article["is_confirming"] = score >= threshold
            article["is_related"] = score >= related_threshold and score < threshold
            scored.append(article)

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored

    @staticmethod
    def format_results(results: List[Dict[str, str]]) -> str:
        """Форматирование результатов для LLM — НЕЙТРАЛЬНО, без предрешения.

        ВАЖНО: НЕ помечаем источники как "подтверждающие" или "не связанные" —
        bag-of-words cosine similarity не может отличить "Путин подписал указ
        о судьях" от "Путин подписал указ о крипте" (совпадают общие слова).
        Модель сама должна прочитать и определить релевантность.
        """
        if not results:
            return "Новости по данному запросу не найдены."

        lines = [f"Найдено {len(results)} источников. Прочитай каждый и определи, "
                 f"подтверждает ли он утверждение или нет."]
        lines.append("")

        for i, article in enumerate(results[:7], 1):
            parts = [f"{i}. {article['title']}"]
            if article.get("source"):
                parts.append(f"   Источник: {article['source']}")
            if article.get("date"):
                parts.append(f"   Дата: {article['date']}")
            if article.get("snippet"):
                snippet = article["snippet"][:300]
                parts.append(f"   {snippet}")
            lines.append("\n".join(parts))

        return "\n\n".join(lines)
