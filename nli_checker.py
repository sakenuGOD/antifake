"""NLI (Natural Language Inference) cross-encoder для объективной проверки фактов.

Использует cross-encoder модель для определения отношения между
утверждением (claim) и каждым источником (evidence):
  - entailment: источник ПОДТВЕРЖДАЕТ утверждение
  - contradiction: источник ОПРОВЕРГАЕТ утверждение
  - neutral: источник НЕ СВЯЗАН с утверждением

Это даёт объективный сигнал вместо субъективного cosine similarity.

Модель: cross-encoder/nli-deberta-v3-base (multilingual)
VRAM: ~400MB (работает на CPU или GPU параллельно с основной моделью)

Использование:
    checker = NLIChecker()
    results = checker.check_claim("ЦБ РФ поднял ставку до 25%", sources)
    print(results["summary"])  # "2 подтверждают, 0 опровергают, 3 нейтральны"
"""

from typing import List, Dict, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIChecker:
    """Cross-encoder NLI для проверки фактов."""

    MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
    LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(self, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def check_pair(self, claim: str, evidence: str) -> Dict[str, float]:
        """Проверка одной пары (claim, evidence).

        Returns:
            {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10, "label": "entailment"}
        """
        inputs = self.tokenizer(
            claim, evidence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        scores = {label: round(prob, 4) for label, prob in zip(self.LABELS, probs)}
        scores["label"] = max(scores, key=scores.get)
        return scores

    def check_claim(
        self,
        claim: str,
        sources: List[Dict[str, str]],
        snippet_key: str = "snippet",
    ) -> Dict[str, Any]:
        """Проверка утверждения против списка источников.

        Args:
            claim: Утверждение для проверки
            sources: Список словарей с ключом snippet_key (текст источника)
            snippet_key: Ключ для извлечения текста из каждого источника

        Returns:
            {
                "pairs": [...],  # результат для каждого источника
                "entailment_count": 2,
                "contradiction_count": 0,
                "neutral_count": 3,
                "max_entailment": 0.92,
                "max_contradiction": 0.15,
                "summary": "2 подтверждают, 0 опровергают, 3 нейтральны",
                "nli_score": 75,  # 0-100, агрегированный score
            }
        """
        pairs = []
        for source in sources:
            evidence = source.get(snippet_key, "")
            if not evidence:
                continue
            result = self.check_pair(claim, evidence)
            result["source"] = source.get("title", source.get("source", ""))
            pairs.append(result)

        if not pairs:
            return {
                "pairs": [],
                "entailment_count": 0,
                "contradiction_count": 0,
                "neutral_count": 0,
                "max_entailment": 0.0,
                "max_contradiction": 0.0,
                "summary": "Источники не найдены",
                "nli_score": 50,
            }

        ent_count = sum(1 for p in pairs if p["label"] == "entailment")
        con_count = sum(1 for p in pairs if p["label"] == "contradiction")
        neu_count = sum(1 for p in pairs if p["label"] == "neutral")

        max_ent = max(p["entailment"] for p in pairs)
        max_con = max(p["contradiction"] for p in pairs)

        # Агрегированный NLI score: 0-100
        # entailment → высокий score, contradiction → низкий
        if ent_count > 0 and con_count == 0:
            nli_score = int(50 + max_ent * 50)
        elif con_count > 0 and ent_count == 0:
            nli_score = int(50 - max_con * 50)
        elif ent_count > 0 and con_count > 0:
            # Смешанные сигналы — ближе к неопределённости
            nli_score = int(50 + (max_ent - max_con) * 30)
        else:
            nli_score = 50  # Все нейтральны

        nli_score = max(0, min(100, nli_score))

        return {
            "pairs": pairs,
            "entailment_count": ent_count,
            "contradiction_count": con_count,
            "neutral_count": neu_count,
            "max_entailment": round(max_ent, 4),
            "max_contradiction": round(max_con, 4),
            "summary": f"{ent_count} подтверждают, {con_count} опровергают, {neu_count} нейтральны",
            "nli_score": nli_score,
        }
