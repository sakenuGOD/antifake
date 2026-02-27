"""Русский NLP: NER + лемматизация через Natasha/pymorphy2.

Заменяет regex-based entity extraction и 4-char stem matching
на полноценный морфологический анализ русского языка.

CPU-only, ~30MB модели, не занимает GPU.
"""

import re
from typing import List, Dict, Set, Tuple
from functools import lru_cache

# Natasha NLP pipeline
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding,
    NewsMorphTagger, NewsNERTagger, Doc,
)
import pymorphy2

# Singleton — загружаем модели один раз
_segmenter = Segmenter()
_morph_vocab = MorphVocab()
_emb = NewsEmbedding()
_morph_tagger = NewsMorphTagger(_emb)
_ner_tagger = NewsNERTagger(_emb)
_morph = pymorphy2.MorphAnalyzer()

print("[NLP] Natasha + pymorphy2 загружены")


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Извлекает именованные сущности из текста.

    Returns:
        [{"text": "Нил Армстронг", "type": "PER", "normal": "нил армстронг"},
         {"text": "Луну", "type": "LOC", "normal": "луна"}, ...]
    """
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    doc.tag_ner(_ner_tagger)

    # Resolve spans to extract normal forms
    for span in doc.spans:
        span.normalize(_morph_vocab)

    entities = []
    seen = set()
    for span in doc.spans:
        normal = span.normal.lower() if span.normal else span.text.lower()
        if normal not in seen:
            seen.add(normal)
            entities.append({
                "text": span.text,
                "type": span.type,  # PER, LOC, ORG
                "normal": normal,
            })
    return entities


def extract_entity_names(text: str) -> List[str]:
    """Извлекает имена сущностей (для поиска в Wikipedia/Wikidata).

    Returns: ["Нил Армстронг", "Луна", "Microsoft"]
    """
    entities = extract_entities(text)
    return [e["text"] for e in entities]


@lru_cache(maxsize=2048)
def lemmatize(word: str) -> str:
    """Лемматизация одного слова через pymorphy2.

    "спутником" → "спутник"
    "Москвы" → "москва"
    "основал" → "основать"
    """
    parsed = _morph.parse(word.lower())
    if parsed:
        return parsed[0].normal_form
    return word.lower()


def lemmatize_text(text: str) -> List[str]:
    """Лемматизация всех значимых слов в тексте.

    Returns: ["нил", "армстронг", "высадиться", "луна", "июль", "1969", "год"]
    """
    words = re.findall(r'[а-яёА-ЯЁa-zA-Z]{2,}', text)
    return [lemmatize(w) for w in words]


def extract_keywords_lemmatized(text: str) -> List[str]:
    """Извлекает ключевые слова с лемматизацией.

    Фильтрует стоп-слова (предлоги, союзы, частицы).
    Returns: ["нил", "армстронг", "высадиться", "луна"]
    """
    _STOP_POS = {'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO'}  # служебные ЧР

    words = re.findall(r'[а-яёА-ЯЁ]{3,}', text)
    keywords = []
    seen = set()
    for w in words:
        parsed = _morph.parse(w.lower())
        if not parsed:
            continue
        p = parsed[0]
        if p.tag.POS in _STOP_POS:
            continue
        lemma = p.normal_form
        if lemma not in seen and len(lemma) >= 3:
            seen.add(lemma)
            keywords.append(lemma)
    return keywords


def stems_match(word1: str, word2: str) -> bool:
    """Проверяет совпадение слов через лемматизацию.

    "спутником" vs "спутник" → True (оба → "спутник")
    "Москвы" vs "Москве" → True (оба → "москва")
    """
    return lemmatize(word1) == lemmatize(word2)


def words_overlap_lemmatized(text1: str, text2: str) -> Tuple[Set[str], Set[str], float]:
    """Считает overlap двух текстов через лемматизацию.

    Returns: (common_lemmas, missing_from_text2, overlap_ratio)
    """
    lemmas1 = set(lemmatize_text(text1))
    lemmas2 = set(lemmatize_text(text2))

    if not lemmas1:
        return set(), set(), 0.0

    common = lemmas1 & lemmas2
    missing = lemmas1 - lemmas2
    ratio = len(common) / len(lemmas1)
    return common, missing, ratio
