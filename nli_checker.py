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
    """V13: Russian-native NLI classifier for fact-checking.

    PRIMARY: cointegrated/rubert-base-cased-nli-threeway (RuBERT, native Russian)
    FALLBACK: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
    """

    # V13: Primary model — RuBERT fine-tuned on Russian NLI (MNLI+SNLI+FEVER translated)
    PRIMARY_MODEL = "cointegrated/rubert-base-cased-nli-threeway"
    FALLBACK_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    # Label order for rubert-base-cased-nli-threeway: entailment(0), contradiction(1), neutral(2)
    # Label order for mDeBERTa: entailment(0), neutral(1), contradiction(2)
    # We normalize to a common order internally
    LABELS = ["entailment", "neutral", "contradiction"]

    def __init__(self, device: str = "cpu", model_name: str = None):
        self.device = device
        self._use_onnx = False
        self._label_order = None  # will be set based on loaded model
        # V17: Temperature scaling — 1.0 = no distortion
        self.temperature = 1.0

        # V11: Cross-encoder NLI tiebreaker for ambiguous cases
        self._cross_encoder = None

        # V13: Try primary RuBERT model first, fallback to mDeBERTa
        _model_to_load = model_name or self.PRIMARY_MODEL
        _loaded = False

        # B4: Try ONNX Runtime first for 2-3x speedup
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            import os
            onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "nli_onnx")
            if os.path.exists(onnx_path):
                self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
                self.model = ORTModelForSequenceClassification.from_pretrained(onnx_path)
                self._use_onnx = True
                self._label_order = ["entailment", "neutral", "contradiction"]  # mDeBERTa ONNX
                _loaded = True
                print(f"NLI: ONNX Runtime loaded from {onnx_path}")
            else:
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        except (ImportError, FileNotFoundError, Exception):
            pass

        if not _loaded:
            # Try primary model
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*position_ids.*")
                    self.tokenizer = AutoTokenizer.from_pretrained(_model_to_load)
                    self.model = AutoModelForSequenceClassification.from_pretrained(_model_to_load)
                self.model.to(self.device)
                self.model.eval()
                # Set label order based on model config
                if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                    id2l = self.model.config.id2label
                    self._label_order = [id2l[k] for k in sorted(id2l.keys(), key=int)]
                else:
                    self._label_order = self.LABELS
                _loaded = True
                print(f"NLI: PyTorch model loaded ({_model_to_load})")
            except Exception as e:
                print(f"NLI: Failed to load {_model_to_load}: {e}")

        # Fallback to mDeBERTa if primary failed
        if not _loaded:
            try:
                _model_to_load = self.FALLBACK_MODEL
                self.tokenizer = AutoTokenizer.from_pretrained(_model_to_load)
                self.model = AutoModelForSequenceClassification.from_pretrained(_model_to_load)
                self.model.to(self.device)
                self.model.eval()
                self._label_order = ["entailment", "neutral", "contradiction"]
                print(f"NLI: Fallback model loaded ({_model_to_load})")
            except Exception as e:
                raise RuntimeError(f"NLI: Cannot load any model: {e}")

    def _init_cross_encoder(self):
        """V17: Load cross-encoder/nli-deberta-v3-base (3-class NLI, not relevance scorer)."""
        if self._cross_encoder is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-base")
            self._ce_is_baai = False
            print("NLI: cross-encoder/nli-deberta-v3-base loaded")
            return True
        except Exception as e:
            print(f"NLI: Cross-encoder not available: {e}")
            return False

    def cross_nli(self, premise: str, hypothesis: str) -> dict:
        """Cross-encoder NLI: more accurate but slower. Used as tiebreaker."""
        if not self._init_cross_encoder():
            return {"contradiction": 0.0, "entailment": 0.0, "neutral": 1.0}
        scores = self._cross_encoder.predict([(premise, hypothesis)])
        labels = ["contradiction", "entailment", "neutral"]
        result = dict(zip(labels, scores[0].tolist()))
        return result

    def check_claim_cross(
        self,
        claim: str,
        sources: list,
        snippet_key: str = "snippet",
        top_k: int = 3,
    ) -> dict:
        """V12: CrossEncoder NLI for top-K sources — catches entity-swap contradictions.

        Unlike bi-encoder (encodes claim and premise separately), cross-encoder
        tokenizes (premise, hypothesis) together, so it sees entity differences directly.

        Returns: {"max_entailment": float, "max_contradiction": float, "ce_pairs": [...]}
        """
        if not self._init_cross_encoder():
            return {"max_entailment": 0.0, "max_contradiction": 0.0, "ce_pairs": []}

        ce_pairs = []
        max_ent = 0.0
        max_con = 0.0

        for source in sources[:top_k]:
            evidence = source.get(snippet_key, "")
            if not evidence or len(evidence) < 20:
                continue

            # Split into sentences for fine-grained scoring
            sentences = self._split_sentences(evidence)
            if not sentences:
                sentences = [evidence]

            # Build pairs for batch prediction
            pairs = [(sent, claim) for sent in sentences[:10]]  # cap at 10 sentences
            try:
                scores = self._cross_encoder.predict(pairs, apply_softmax=True)
                labels = ["contradiction", "entailment", "neutral"]
                best_ent = 0.0
                best_con = 0.0
                for score_row in scores:
                    result = dict(zip(labels, score_row.tolist()))
                    best_ent = max(best_ent, result["entailment"])
                    best_con = max(best_con, result["contradiction"])

                ce_pairs.append({
                    "source": source.get("title", source.get("source", "")),
                    "entailment": round(best_ent, 4),
                    "contradiction": round(best_con, 4),
                })
                max_ent = max(max_ent, best_ent)
                max_con = max(max_con, best_con)
            except Exception as e:
                print(f"  [CE-NLI] Error processing source: {e}")
                continue

        return {
            "max_entailment": round(max_ent, 4),
            "max_contradiction": round(max_con, 4),
            "ce_pairs": ce_pairs,
        }

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
            # E3: Temperature scaling before softmax
            logits_np = logits_np / self.temperature
            # Softmax
            exp = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
            probs_all = exp / exp.sum(axis=-1, keepdims=True)
            probs = probs_all[0].tolist()
        else:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            # E3: Temperature scaling before softmax
            logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        # V13: Use dynamic label order from model config
        _labels = self._label_order or self.LABELS
        raw_scores = {label: round(prob, 4) for label, prob in zip(_labels, probs)}
        # Normalize to standard LABELS order
        scores = {
            "entailment": raw_scores.get("entailment", 0.0),
            "neutral": raw_scores.get("neutral", 0.0),
            "contradiction": raw_scores.get("contradiction", 0.0),
        }
        scores["label"] = max(scores, key=lambda k: scores[k] if k in self.LABELS else -1)
        return scores

    # V11: Correction-pattern detection — "на самом деле", "однако", etc.
    # If source sentence contains a correction, boost contradiction score
    _CORRECTION_RE = re.compile(
        r'на\s+самом\s+деле|в\s+действительности|фактически|однако|'
        r'но\s+не\s+в|а\s+не\s+в|not\s+in|actually|in\s+fact',
        re.IGNORECASE)

    # A2: Survey/misconception sentence filter patterns
    # V8: расширенные паттерны для опросов/заблуждений
    _SURVEY_PATTERNS = re.compile(
        r'(?:считают\s+что|полагают\s+что|по\s+данным\s+опроса|'
        r'верят\s+что|убеждены\s+что|думают\s+что|'
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
        """Если ключевые слова claim отсутствуют в premise, penalize score.

        D3: Soft penalty (score *= 0.7) instead of hard cap when key
        claim terms are missing from premise.
        """
        claim_tokens = set(re.findall(r'[а-яёa-z]{4,}', claim.lower()))
        premise_tokens = set(re.findall(r'[а-яёa-z]{4,}', premise.lower()))
        claim_unique = claim_tokens - premise_tokens - STOPWORDS
        missing_ratio = len(claim_unique) / max(len(claim_tokens), 1)
        if len(claim_unique) >= 3 and missing_ratio > 0.4:
            # D3: Soft penalty instead of hard cap
            score *= 0.7
        return score

    # V20: Proper-noun sentence-initial filter — don't treat sentence-starting
    # capitalization as a proper noun. Covers:
    #   - Russian sentence-starters (Когда, Что, …)
    #   - Ordinals and their gender variants (Первый, Вторая, Третье, …) which
    #     often lead named events ("Вторая мировая война") where the ordinal is
    #     the only capitalized word and the rest of the name is lowercase. NLI
    #     on such claims is driven by the event's content words, not the ordinal.
    _SENTENCE_STARTERS = {
        # question / connector words
        "когда", "что", "как", "где", "почему", "нужно", "можно", "это",
        "существует", "возможно", "действительно", "например", "однако",
        "сейчас", "сегодня", "вчера", "раньше", "некоторые", "многие",
        "большинство", "именно", "кроме", "согласно", "после", "перед",
        "каждый", "каждая", "каждое", "весь", "вся", "все", "всё",
        # ordinals — Russian 1–10 across gender / number variants
        "первый", "первая", "первое", "первые", "первому", "первую", "первой",
        "второй", "вторая", "второе", "вторые", "второму", "вторую",
        "третий", "третья", "третье", "третьи", "третьему", "третью", "третьей",
        "четвёртый", "четвертый", "четвёртая", "четвертая", "четвёртое", "четвертое",
        "пятый", "пятая", "пятое", "пятые",
        "шестой", "шестая", "шестое",
        "седьмой", "седьмая", "седьмое",
        "восьмой", "восьмая", "восьмое",
        "девятый", "девятая", "девятое",
        "десятый", "десятая", "десятое",
    }

    @staticmethod
    def _extract_proper_noun_groups(text: str) -> List[List[str]]:
        """Extract proper-noun stem groups from text.

        Adjacent capitalized tokens (no intervening lowercase word) form a
        single group — this treats compound names like "Юрий Гагарин" or
        "Виталик Бутерин" as one entity. Separate groups mark distinct
        entities in the claim (e.g., "Биткоин ... Виталиком Бутериным" →
        two groups: [битко] and [витал, бутер]).

        Sentence-starter words that happen to be capitalized (Как, Что, etc.)
        are skipped.

        Returns a list of groups; each group is a list of 5-char lowercase
        stems preserving input order.
        """
        # Tokenize into (token, is_capitalized) pairs. Word = [A-Za-zА-ЯЁа-яё]{2,}
        word_re = re.compile(r'[А-ЯЁA-Zа-яёa-z]{2,}')
        groups: List[List[str]] = []
        current: List[str] = []
        for m in word_re.finditer(text):
            w = m.group(0)
            first = w[0]
            is_cap = first.isupper()
            if not is_cap:
                if current:
                    groups.append(current)
                    current = []
                continue
            low = w.lower()
            if low in NLIChecker._SENTENCE_STARTERS:
                if current:
                    groups.append(current)
                    current = []
                continue
            stem = low[:5]
            if stem not in current:
                current.append(stem)
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _subject_mention_penalty(claim_pn_groups: List[List[str]],
                                 sentence: str) -> float:
        """Return multiplier in [0.3, 1.0] for NLI sentence scores.

        Mechanism: NLI on a sentence that doesn't contain the claim's primary
        entity (first proper-noun group = the subject) is likely cross-topic
        noise — the signal comes from verb/object lexical echo, not the claim
        itself. Strong entailment from such a sentence is a spurious match.

        Compound names are handled via groups: "Юрий Гагарин" is one group,
        so a sentence mentioning only "Гагарин" still satisfies the primary
        subject.

        Rules:
          - No proper nouns in claim → no penalty (1.0).
          - Any stem of the PRIMARY group in sentence → 1.0.
          - Primary group absent, but some other-group stem present → 0.6.
          - No claim proper-noun stem in sentence at all → 0.3.
        """
        if not claim_pn_groups:
            return 1.0
        sent_lower = sentence.lower()
        primary = claim_pn_groups[0]
        primary_hit = any(s in sent_lower for s in primary)
        if primary_hit:
            return 1.0
        rest = [s for g in claim_pn_groups[1:] for s in g]
        any_other = any(s in sent_lower for s in rest)
        if any_other:
            # Sentence mentions a non-primary entity (e.g., claim's object)
            # but not the subject — classic entity-swap noise signal.
            return 0.30
        # Sentence references no claim entity at all — cross-topic echo.
        return 0.10

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
            # E3: Temperature scaling before softmax
            logits_np = logits_np / self.temperature
            # Softmax
            exp = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
            probs_all = exp / exp.sum(axis=-1, keepdims=True)
        else:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            # E3: Temperature scaling before softmax
            logits = logits / self.temperature
            probs_all = torch.softmax(logits, dim=-1).cpu().tolist()

        # V13: Use dynamic label order
        _labels = self._label_order or self.LABELS
        results = []
        for prob_row in (probs_all if isinstance(probs_all, list) else probs_all.tolist()):
            raw_scores = {label: round(p, 4) for label, p in zip(_labels, prob_row)}
            scores = {
                "entailment": raw_scores.get("entailment", 0.0),
                "neutral": raw_scores.get("neutral", 0.0),
                "contradiction": raw_scores.get("contradiction", 0.0),
            }
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

        # V20: pre-compute claim's proper-noun groups for subject-mention guard.
        # Adjacent capitalized tokens are grouped (compound names). Applied per
        # sentence below to downgrade NLI signals from sentences that don't
        # reference the claim's primary subject.
        claim_pn_groups = self._extract_proper_noun_groups(claim)

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
                # V20: subject-mention penalty. If the claim's primary proper-noun
                # subject is missing from the sentence, the NLI signal is likely
                # cross-topic noise (verb/object lexical echo). Applies symmetrically
                # to ent and con so both false entailments and false contradictions
                # get downgraded.
                subj_mult = self._subject_mention_penalty(claim_pn_groups, sentences[i])
                # Record whether this sentence references the claim's primary
                # subject — used in source-level aggregation to flag
                # signals driven purely by cross-topic lexical echo.
                sr["_subject_verified"] = (subj_mult >= 1.0)
                if subj_mult < 1.0:
                    guarded_ent *= subj_mult
                    guarded_con *= subj_mult
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
            # V17: Skip survey/misconception sentences entirely (they confuse NLI)
            sentence_results = [
                sr for i, sr in enumerate(sentence_results)
                if not self._is_survey_or_misconception(sentences[i])
            ]
            # Also filter the sentences list to keep indices aligned
            sentences = [
                s for s in sentences
                if not self._is_survey_or_misconception(s)
            ]

            # V11: Correction-pattern boost — "на самом деле" etc. → boost contradiction
            for i, sr in enumerate(sentence_results):
                if self._CORRECTION_RE.search(sentences[i]):
                    sr["contradiction"] = round(min(0.95, sr["contradiction"] * 1.5), 4)
                    sr["label"] = max(
                        (k for k in self.LABELS),
                        key=lambda k: sr.get(k, -1)
                    )

            all_sentence_scores.extend(sentence_results)

            # V17: Source-level aggregation — MAX single best sentence, no margin guard
            if sentence_results:
                # V20: track whether the sentence driving best_ent / best_con
                # referenced the claim's primary subject. If not, the source's
                # signal is likely cross-topic noise even after per-sentence
                # penalty (there may just be no on-topic sentence at all).
                best_ent_sr = max(sentence_results, key=lambda r: r["entailment"])
                best_con_sr = max(sentence_results, key=lambda r: r["contradiction"])
                best_ent = best_ent_sr["entailment"]
                best_con = best_con_sr["contradiction"]
                if best_ent > best_con:
                    best_label = "entailment"
                else:
                    best_label = "contradiction"
                source_level_results.append({
                    "entailment": best_ent,
                    "contradiction": best_con,
                    "subject_verified_ent": bool(best_ent_sr.get("_subject_verified", True)),
                    "subject_verified_con": bool(best_con_sr.get("_subject_verified", True)),
                    "label": best_label,
                    "source": source.get("title", source.get("source", "")),
                    "url": source.get("link") or source.get("url", ""),
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

        # V18: Purity-weighted aggregation — filter out "confused" sources
        # that give both high ent and high con (noise from mixed-content snippets).
        # purity = |best_ent - best_con|: low purity → source is ambiguous, exclude it.
        sorted_by_purity = sorted(source_level_results, key=lambda x: x["purity"], reverse=True)
        n_keep = max(1, len(sorted_by_purity) // 2)  # top 50% most decisive sources
        decisive = sorted_by_purity[:n_keep]

        max_ent = max(s["entailment"] for s in decisive)
        max_con = max(s["contradiction"] for s in decisive)

        # V20: subject-verified aggregates. A signal is "subject-verified" if at
        # least one decisive source driving it had its best-scoring sentence
        # mention the claim's primary subject. Signals without ANY subject-
        # verified source are unreliable (classic entity-swap noise) — the
        # decide-tree uses this to gate strong/moderate NLI verdicts.
        subj_verified_ent = any(
            s.get("subject_verified_ent", True)
            for s in decisive
            if s["entailment"] >= 0.50
        )
        subj_verified_con = any(
            s.get("subject_verified_con", True)
            for s in decisive
            if s["contradiction"] >= 0.50
        )

        ent_count = sum(1 for p in decisive if p["label"] == "entailment")
        con_count = sum(1 for p in decisive if p["label"] == "contradiction")
        neu_count = sum(1 for p in decisive if p["label"] == "neutral")

        # V18: Contested topic detection — sources genuinely divided on the claim.
        # Fires when BOTH ent and con signals are significant after purity filtering.
        # Distinguishes contested scientific claims (мобильники: ent=0.63, con=0.93)
        # from clear myths (сахар: ent=0.11 — almost no support) and synonym issues.
        is_contested = max_ent >= 0.40 and max_con >= 0.40

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
            "is_contested": is_contested,
            "subject_verified_ent": subj_verified_ent,
            "subject_verified_con": subj_verified_con,
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
