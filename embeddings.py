"""Семантическое ранжирование с sentence-transformers.

Замена bag-of-words cosine similarity на нейронные эмбеддинги.
Правильно различает "Путин указ о судьях" vs "Путин указ о крипте".
"""

from typing import List, Dict

_ranker_instance = None


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
        self.model = SentenceTransformer(model_name)
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

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Вычисление семантического сходства между двумя текстами."""
        from sentence_transformers import util

        emb_a = self.model.encode(
            self._prepare_text(text_a, is_query=True),
            convert_to_tensor=True,
        )
        emb_b = self.model.encode(
            self._prepare_text(text_b),
            convert_to_tensor=True,
        )

        return util.cos_sim(emb_a, emb_b).item()
