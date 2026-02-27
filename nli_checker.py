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
from utils import STOPWORDS


class NLIChecker:
    """Multilingual NLI classifier for fact-checking."""

    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    # CRITICAL: label order for mDeBERTa-xnli is ["entailment", "neutral", "contradiction"]
    LABELS = ["entailment", "neutral", "contradiction"]

    def __init__(self, device: str = "cpu", model_name: str = None):
        self.device = device
        model_name = model_name or self.DEFAULT_MODEL
        self._use_onnx = False

        # B4: Try ONNX Runtime first for 2-3x speedup
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            import os
            onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nli_onnx")
            if os.path.exists(onnx_path):
                self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
                self.model = ORTModelForSequenceClassification.from_pretrained(onnx_path)
                self._use_onnx = True
                print(f"NLI: ONNX Runtime loaded from {onnx_path}")
            else:
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        except (ImportError, FileNotFoundError, Exception) as e:
            # Fallback to PyTorch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"NLI: PyTorch model loaded ({model_name})")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences for fine-grained NLI scoring.

        Improved Russian sentence splitting:
        - Split on sentence-ending punctuation followed by space + uppercase/digit
        - Also split on semicolons and long dashes (common in Russian news)
        - Filter out very short fragments (< 15 chars) for better NLI signal
        """
        # Split on .!? followed by space, also on ; and — (Russian style)
        sentences = re.split(r'(?<=[.!?;])\s+|(?<=\.)\s*\n|(?:\s+[—–]\s+)', text)
        # Additional split: numbered items "1) ...", "2. ..."
        expanded = []
        for s in sentences:
            parts = re.split(r'(?:^|\s)\d+[.)]\s+', s)
            expanded.extend(parts)
        return [s.strip() for s in expanded if len(s.strip()) > 15]

    def check_pair(self, claim: str, evidence: str) -> Dict[str, float]:
        """Check a single (claim, evidence) pair.

        Returns:
            {"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10, "label": "entailment"}
        """
        inputs = self.tokenizer(
            claim, evidence,
            return_tensors="pt" if not self._use_onnx else "np",
            truncation=True,
            max_length=512,
            padding=True,
        )

        if self._use_onnx:
            import numpy as np
            outputs = self.model(**{k: v for k, v in inputs.items()})
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if hasattr(logits, 'numpy'):
                logits_np = logits.numpy() if not isinstance(logits, np.ndarray) else logits
            else:
                logits_np = np.array(logits)
            # Softmax
            exp = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
            probs_all = exp / exp.sum(axis=-1, keepdims=True)
            probs = probs_all[0].tolist()
        else:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        scores = {label: round(prob, 4) for label, prob in zip(self.LABELS, probs)}
        scores["label"] = max(scores, key=lambda k: scores[k] if k in self.LABELS else -1)
        return scores

    # A2: Survey/misconception sentence filter patterns
    _SURVEY_PATTERNS = re.compile(
        r'(?:считают\s+что|полагают\s+что|по\s+данным\s+опроса|'
        r'%\s*респондентов|процент\w*\s+(?:россиян|людей|населения)\s+считают|'
        r'(?:миф|заблуждени[ея]|popular\s+misconception|urban\s+legend)\s+(?:что|о\s+том))',
        re.IGNORECASE,
    )

    @staticmethod
    def _is_survey_or_misconception(sentence: str) -> bool:
        """Check if a sentence reports survey results or labels something a misconception."""
        return bool(NLIChecker._SURVEY_PATTERNS.search(sentence))

    @staticmethod
    def _claim_term_guard(claim: str, premise: str, score: float,
                          symmetric: bool = False) -> float:
        """Если ключевые слова claim отсутствуют в premise, cap score.

        A2: Proportional cap instead of fixed 0.45. Applied symmetrically
        to both entailment and contradiction.
        """
        claim_tokens = set(re.findall(r'[а-яёa-z]{4,}', claim.lower()))
        premise_tokens = set(re.findall(r'[а-яёa-z]{4,}', premise.lower()))
        claim_unique = claim_tokens - premise_tokens - STOPWORDS
        missing_ratio = len(claim_unique) / max(len(claim_tokens), 1)
        if len(claim_unique) >= 3 and missing_ratio > 0.4:
            # A2: Proportional cap: more missing terms → lower cap
            cap = max(0.20, 1.0 - missing_ratio * 1.2)
            return min(score, cap)
        return score

    def _check_batch(self, claim: str, sentences: List[str]) -> List[Dict[str, float]]:
        """Batch NLI check for multiple sentences against a single claim."""
        if not sentences:
            return []

        inputs = self.tokenizer(
            [claim] * len(sentences),
            sentences,
            return_tensors="pt" if not self._use_onnx else "np",
            truncation=True,
            max_length=512,
            padding=True,
        )

        if self._use_onnx:
            # ONNX path — numpy tensors, no torch
            import numpy as np
            outputs = self.model(**{k: v for k, v in inputs.items()})
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            if hasattr(logits, 'numpy'):
                logits_np = logits.numpy() if not isinstance(logits, np.ndarray) else logits
            else:
                logits_np = np.array(logits)
            # Softmax
            exp = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
            probs_all = exp / exp.sum(axis=-1, keepdims=True)
        else:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs_all = torch.softmax(logits, dim=-1).cpu().tolist()

        results = []
        for prob_row in (probs_all if isinstance(probs_all, list) else probs_all.tolist()):
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
            # Apply claim-term guard: cap entailment AND contradiction if key claim terms missing
            for i, sr in enumerate(sentence_results):
                guarded_ent = self._claim_term_guard(claim, sentences[i], sr["entailment"])
                # A2: Symmetric guard — also cap contradiction
                guarded_con = self._claim_term_guard(claim, sentences[i], sr["contradiction"], symmetric=True)
                changed = False
                if guarded_ent < sr["entailment"]:
                    sr["entailment"] = guarded_ent
                    changed = True
                if guarded_con < sr["contradiction"]:
                    sr["contradiction"] = guarded_con
                    changed = True
                if changed:
                    # Recalculate label
                    sr["label"] = max(
                        (k for k in self.LABELS),
                        key=lambda k: sr.get(k, -1)
                    )
            # A2: Filter survey/misconception sentences BEFORE aggregation
            filtered_results = []
            for i, sr in enumerate(sentence_results):
                if self._is_survey_or_misconception(sentences[i]):
                    continue
                filtered_results.append(sr)
            if not filtered_results:
                filtered_results = sentence_results  # fallback to all if all filtered

            all_sentence_scores.extend(filtered_results)

            # Source-level aggregation: A2 — top-k mean instead of max
            if filtered_results:
                ent_scores = sorted([r["entailment"] for r in filtered_results], reverse=True)
                con_scores = sorted([r["contradiction"] for r in filtered_results], reverse=True)
                # Top-2 mean (more robust than single max)
                best_ent = sum(ent_scores[:2]) / min(2, len(ent_scores))
                best_con = sum(con_scores[:2]) / min(2, len(con_scores))
                # A2: Margin-based label — if difference < 0.15, label neutral
                if abs(best_ent - best_con) < 0.15:
                    best_label = "neutral"
                elif best_ent > best_con:
                    best_label = "entailment"
                else:
                    best_label = "contradiction"
                source_level_results.append({
                    "entailment": best_ent,
                    "contradiction": best_con,
                    "label": best_label,
                    "source": source.get("title", source.get("source", "")),
                    "num_sentences": len(sentences),
                    "purity": abs(best_ent - best_con),
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
