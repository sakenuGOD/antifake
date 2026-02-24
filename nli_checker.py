"""NLI (Natural Language Inference) classifier for fact-checking.

Uses multilingual mDeBERTa model for Russian-native NLI:
  - entailment: source SUPPORTS the claim
  - contradiction: source CONTRADICTS the claim
  - neutral: source is UNRELATED to the claim

Sentence-level scoring (SummaC approach):
  - Split each snippet into sentences
  - NLI(claim, sentence) for each sentence
  - Aggregate max_entailment, max_contradiction across all sentence×source pairs

Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
  - 280M params, trained on XNLI + 2.7M multilingual pairs
  - Russian language natively supported
  - ~400MB RAM on CPU (does not occupy GPU)

IMPORTANT: Labels order for this model is ["entailment", "neutral", "contradiction"]
"""

import re
from typing import List, Dict, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIChecker:
    """Multilingual NLI classifier for fact-checking."""

    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    # CRITICAL: label order for mDeBERTa-xnli is ["entailment", "neutral", "contradiction"]
    LABELS = ["entailment", "neutral", "contradiction"]

    def __init__(self, device: str = "cpu", model_name: str = None):
        self.device = device
        model_name = model_name or self.DEFAULT_MODEL

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences for fine-grained NLI scoring."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @torch.no_grad()
    def check_pair(self, claim: str, evidence: str) -> Dict[str, float]:
        """Check a single (claim, evidence) pair.

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
        scores["label"] = max(scores, key=lambda k: scores[k] if k in self.LABELS else -1)
        return scores

    @torch.no_grad()
    def _check_batch(self, claim: str, sentences: List[str]) -> List[Dict[str, float]]:
        """Batch NLI check for multiple sentences against a single claim."""
        if not sentences:
            return []

        inputs = self.tokenizer(
            [claim] * len(sentences),
            sentences,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        results = []
        for prob_row in probs:
            scores = {label: round(p, 4) for label, p in zip(self.LABELS, prob_row)}
            scores["label"] = max(scores, key=lambda k: scores[k] if k in self.LABELS else -1)
            results.append(scores)
        return results

    def check_claim(
        self,
        claim: str,
        sources: List[Dict[str, str]],
        snippet_key: str = "snippet",
    ) -> Dict[str, Any]:
        """Check claim against list of sources with sentence-level scoring.

        For each source snippet:
          1. Split into sentences
          2. NLI(claim, sentence) for each sentence (batched)
          3. Track max_entailment, max_contradiction across ALL sentence×source

        Returns:
            {
                "pairs": [...],
                "entailment_count": 2,
                "contradiction_count": 0,
                "neutral_count": 3,
                "max_entailment": 0.92,
                "max_contradiction": 0.15,
                "summary": "2 подтверждают, 0 опровергают, 3 нейтральны",
                "nli_score": 75,
            }
        """
        all_sentence_scores = []
        source_level_results = []

        for source in sources:
            evidence = source.get(snippet_key, "")
            if not evidence:
                continue

            sentences = self._split_sentences(evidence)
            if not sentences:
                # Fallback: use whole snippet as one sentence
                sentences = [evidence]

            # Batch NLI for all sentences of this source
            sentence_results = self._check_batch(claim, sentences)
            all_sentence_scores.extend(sentence_results)

            # Source-level aggregation: best score per source
            if sentence_results:
                best_ent = max(r["entailment"] for r in sentence_results)
                best_con = max(r["contradiction"] for r in sentence_results)
                best_label = "entailment" if best_ent > best_con else (
                    "contradiction" if best_con > best_ent else "neutral"
                )
                source_level_results.append({
                    "entailment": best_ent,
                    "contradiction": best_con,
                    "label": best_label,
                    "source": source.get("title", source.get("source", "")),
                    "num_sentences": len(sentences),
                })

        if not source_level_results:
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

        ent_count = sum(1 for p in source_level_results if p["label"] == "entailment")
        con_count = sum(1 for p in source_level_results if p["label"] == "contradiction")
        neu_count = sum(1 for p in source_level_results if p["label"] == "neutral")

        # Max across ALL sentence×source pairs
        max_ent = max(s["entailment"] for s in all_sentence_scores) if all_sentence_scores else 0.0
        max_con = max(s["contradiction"] for s in all_sentence_scores) if all_sentence_scores else 0.0

        # Aggregated NLI score: 0-100
        if ent_count > 0 and con_count == 0:
            nli_score = int(50 + max_ent * 50)
        elif con_count > 0 and ent_count == 0:
            nli_score = int(50 - max_con * 50)
        elif ent_count > 0 and con_count > 0:
            nli_score = int(50 + (max_ent - max_con) * 30)
        else:
            nli_score = 50

        nli_score = max(0, min(100, nli_score))

        return {
            "pairs": source_level_results,
            "entailment_count": ent_count,
            "contradiction_count": con_count,
            "neutral_count": neu_count,
            "max_entailment": round(max_ent, 4),
            "max_contradiction": round(max_con, 4),
            "summary": f"{ent_count} подтверждают, {con_count} опровергают, {neu_count} нейтральны",
            "nli_score": nli_score,
        }

    def get_verdict_signal(self, claim: str, sources: List[Dict[str, str]],
                           snippet_key: str = "snippet") -> Dict[str, Any]:
        """Get a verdict recommendation from NLI analysis."""
        result = self.check_claim(claim, sources, snippet_key=snippet_key)
        if result["max_contradiction"] >= 0.70:
            return {"signal": "ЛОЖЬ", "confidence": result["max_contradiction"], "nli_result": result}
        elif result["max_entailment"] >= 0.65:
            return {"signal": "ПРАВДА", "confidence": result["max_entailment"], "nli_result": result}
        else:
            return {"signal": "NEUTRAL", "confidence": 0.5, "nli_result": result}
