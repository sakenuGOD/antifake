"""MiniCheck-FT5 fact verification via Flan-T5-Large.

Lightweight implementation — loads lytang/MiniCheck-Flan-T5-Large directly
via transformers without the minicheck pip package.

Model: ~770M params, ~1.7GB on CPU (not GPU).
Prompt format: "predict: {document}</s>{claim}"
Output: softmax over token IDs [3 (no support), 209 (support)].

Reference: https://arxiv.org/abs/2404.10774
"""

import re
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MiniCheckVerifier:
    """MiniCheck-FT5: GPT-4 level fact-checking accuracy on CPU."""

    MODEL_NAME = "lytang/MiniCheck-Flan-T5-Large"
    # Token IDs in Flan-T5 vocabulary: 3 = "no support", 209 = "support"
    LABEL_TOKEN_IDS = [3, 209]
    MAX_INPUT_LEN = 512
    # Chunk size in words for long documents
    CHUNK_SIZE = 350
    CHUNK_OVERLAP = 50

    def __init__(self, device: str = "cpu", model_name: str = None):
        self.device = device
        model_name = model_name or self.MODEL_NAME
        print(f"[MiniCheck] Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[MiniCheck] Ready ({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)")

    def _chunk_document(self, document: str) -> List[str]:
        """Split long document into overlapping word-based chunks."""
        words = document.split()
        if len(words) <= self.CHUNK_SIZE:
            return [document]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.CHUNK_SIZE, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    def score_pair(self, document: str, claim: str) -> float:
        """Score a single (document, claim) pair.

        Returns: support probability (0.0 - 1.0).
        For long documents, returns max across chunks.
        """
        chunks = self._chunk_document(document)
        max_prob = 0.0

        for chunk in chunks:
            # MiniCheck-FT5 prompt format
            text = f"predict: {chunk}{self.tokenizer.eos_token}{claim}"
            inputs = self.tokenizer(
                text,
                max_length=self.MAX_INPUT_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Single forward pass — zero decoder input
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].size(0), 1),
                dtype=torch.long,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                )

            logits = outputs.logits.squeeze(1)  # [batch, vocab_size]
            label_logits = logits[:, torch.tensor(self.LABEL_TOKEN_IDS)].cpu()
            probs = torch.nn.functional.softmax(label_logits, dim=-1)
            support_prob = probs[0, 1].item()  # index 1 = "support" token
            max_prob = max(max_prob, support_prob)

        return max_prob

    def score_batch(self, documents: List[str], claims: List[str]) -> List[float]:
        """Score multiple (document, claim) pairs.

        No chunking — for batch efficiency, pass pre-chunked documents.
        """
        if not documents or not claims:
            return []

        texts = [
            f"predict: {doc}{self.tokenizer.eos_token}{cl}"
            for doc, cl in zip(documents, claims)
        ]
        inputs = self.tokenizer(
            texts,
            max_length=self.MAX_INPUT_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        decoder_input_ids = torch.zeros(
            (inputs["input_ids"].size(0), 1),
            dtype=torch.long,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

        logits = outputs.logits.squeeze(1)
        label_logits = logits[:, torch.tensor(self.LABEL_TOKEN_IDS)].cpu()
        probs = torch.nn.functional.softmax(label_logits, dim=-1)
        return probs[:, 1].tolist()

    def verify_claim(
        self,
        claim: str,
        sources: List[Dict[str, str]],
        snippet_key: str = "snippet",
    ) -> Dict[str, Any]:
        """Verify a claim against multiple sources.

        Returns:
            {
                "supported": bool,
                "max_support": float,  # max support prob across sources
                "mean_support": float,
                "per_source": [{"source": ..., "support_prob": ...}, ...],
                "signal": int,  # +2 if supported, -2 if contradicted, 0 otherwise
            }
        """
        per_source = []
        for src in sources:
            doc = src.get(snippet_key, "")
            if not doc or len(doc) < 20:
                continue
            prob = self.score_pair(doc, claim)
            per_source.append({
                "source": src.get("title", src.get("source", "")),
                "support_prob": round(prob, 4),
            })

        if not per_source:
            return {
                "supported": False,
                "max_support": 0.0,
                "mean_support": 0.0,
                "per_source": [],
                "signal": 0,
            }

        probs = [p["support_prob"] for p in per_source]
        max_support = max(probs)
        mean_support = sum(probs) / len(probs)

        # Decision: supported if max > 0.5
        supported = max_support > 0.5

        # V11: Spread guard — distinguish topic-match from fact-match
        spread = max_support - mean_support
        is_narrow = spread <= 0.25  # All sources agree → real support

        if max_support >= 0.80 and is_narrow:
            signal = 2
        elif max_support >= 0.70 and is_narrow:
            signal = 1  # Lower confidence
        elif max_support >= 0.70 and not is_narrow:
            signal = 0  # Topic-match, not fact-match
        elif max_support <= 0.2 and mean_support <= 0.3:
            signal = -2
        elif max_support <= 0.35:
            signal = -1
        else:
            signal = 0

        return {
            "supported": supported,
            "max_support": round(max_support, 4),
            "mean_support": round(mean_support, 4),
            "per_source": per_source,
            "signal": signal,
        }

    def verify_with_negation(
        self,
        claim: str,
        sources: List[Dict[str, str]],
        snippet_key: str = "snippet",
    ) -> int:
        """V11: Run MiniCheck on claim AND its negation to detect topic-match.

        If both positive and negated claim get high support → topic-match, signal=0.
        Only called conditionally (when positive signal > 0 and Wikidata contradicts).
        """
        pos_result = self.verify_claim(claim, sources, snippet_key)
        pos_max = pos_result["max_support"]

        if pos_max <= 0.5:
            return pos_result["signal"]  # Not supported, no need for negation check

        neg_claim = f"Неверно, что {claim[0].lower()}{claim[1:]}"
        neg_result = self.verify_claim(neg_claim, sources, snippet_key)
        neg_max = neg_result["max_support"]

        if pos_max > 0.5 and neg_max > 0.5:
            return 0  # Both supported → topic-match
        elif pos_max > 0.5 and neg_max < 0.3:
            return pos_result["signal"]  # Genuine support
        elif neg_max > 0.5 and pos_max < 0.3:
            return -abs(pos_result["signal"])  # Genuine contradiction
        else:
            return pos_result["signal"]
