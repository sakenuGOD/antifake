"""Семантическое ранжирование с sentence-transformers.

Замена bag-of-words cosine similarity на нейронные эмбеддинги.
Правильно различает "Путин указ о судьях" vs "Путин указ о крипте".
"""

from typing import List, Dict

_ranker_instance = None
_reranker_instance = None


def get_ranker():
    """Ленивая инициализация — загружаем модель только при первом вызове."""
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = SemanticRanker()
    return _ranker_instance


class SemanticRanker:
    """Ранжирование результатов поиска по семантическому сходству.

    Модели (от быстрой к точной):
    - cointegrated/rubert-tiny2 (29M, быстрая, русский)
    - intfloat/multilingual-e5-base (278M, лучшее качество/скорость)
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (118M)
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer

        print(f"Загрузка модели эмбеддингов: {model_name}...")
        # Принудительно используем CPU для эмбеддингов — GPU нужен для LLM
        self.model = SentenceTransformer(model_name, device="cpu")
        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()
        print(f"Модель эмбеддингов загружена.")

    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        """Подготовка текста. E5 модели требуют prefix."""
        if self._is_e5:
            prefix = "query: " if is_query else "passage: "
            return prefix + text
        return text

    def rank_results(
        self,
        claim: str,
        results: List[Dict[str, str]],
        top_k: int = 10,
    ) -> List[Dict[str, str]]:
        """Ранжирование результатов по семантическому сходству с утверждением."""
        if not results:
            return []

        from sentence_transformers import util

        query = self._prepare_text(claim, is_query=True)
        passages = [
            self._prepare_text(
                f"{r.get('title', '')} {r.get('snippet', '')}"
            )
            for r in results
        ]

        query_emb = self.model.encode(query, convert_to_tensor=True)
        passage_embs = self.model.encode(passages, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, passage_embs)[0]

        for i, result in enumerate(results):
            result["semantic_score"] = round(scores[i].item(), 4)

        results.sort(key=lambda x: x["semantic_score"], reverse=True)
        return results[:top_k]

    def similarity(self, text_a: str, text_b: str) -> float:
        """B4/G4: Compute cosine similarity between two texts."""
        from sentence_transformers import util
        emb_a = self.model.encode(self._prepare_text(text_a, is_query=True), convert_to_tensor=True)
        emb_b = self.model.encode(self._prepare_text(text_b), convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b)[0][0])


def get_reranker():
    """Ленивая инициализация CrossEncoder — загружается только при первом вызове."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ReRanker()
    return _reranker_instance


class ReRanker:
    """Cross-encoder re-ranker поверх bi-encoder результатов.

    Паттерн: bi-encoder даёт N кандидатов (быстро), cross-encoder выбирает топ-K (точно).
    A7: Multilingual mmarco model (~134MB) — proper Russian text ranking.
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        from sentence_transformers import CrossEncoder

        print(f"Загрузка CrossEncoder: {model_name}...")
        self.cross_encoder = CrossEncoder(model_name, device="cpu")
        print("CrossEncoder загружен.")

    def rerank(self, claim: str, results: list, top_k: int = 5) -> list:
        """Re-rank результатов. Добавляет cross_encoder_score, возвращает top_k."""
        if not results:
            return []

        pairs = [
            (claim, r.get("title", "") + " " + r.get("snippet", ""))
            for r in results
        ]
        scores = self.cross_encoder.predict(pairs)

        for score, r in zip(scores, results):
            r["cross_encoder_score"] = float(score)

        ranked = sorted(results, key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
        return ranked[:top_k]
