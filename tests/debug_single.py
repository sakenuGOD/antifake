"""Диагностика одного claim: полная трассировка NLI, чисел, ensemble."""
import _path  # noqa: F401,E402 — inject project root into sys.path
import os
import sys
import logging

# Включаем подробное логирование ensemble
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from config import SearchConfig, PipelineConfig
from model import find_best_adapter

# Claim для тестирования — берём из аргумента или дефолт
claim = sys.argv[1] if len(sys.argv) > 1 else "Население Земли превысило 8 миллиардов человек"
print(f"\n{'='*60}")
print(f"CLAIM: {claim}")
print(f"{'='*60}\n")

serpapi_key = os.environ.get("SERPAPI_API_KEY", "")
search_config = SearchConfig(api_key=serpapi_key)
adapter_path = find_best_adapter()

print(f"Adapter: {adapter_path}")
print(f"Search: {'SerpAPI' if serpapi_key else 'DuckDuckGo'}\n")

from pipeline import FactCheckPipeline
pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=search_config)

# Прогоняем claim
result = pipeline.check(claim)

# Достаём NLI данные из thread-local
nli_result = pipeline._shared.get("nli_result", None)
num_comparisons = pipeline._shared.get("num_comparisons", [])
raw_results = pipeline._shared.get("raw_results", [])

print(f"\n{'='*60}")
print("ДИАГНОСТИКА")
print(f"{'='*60}")

print(f"\n--- ПОИСК ---")
print(f"Найдено источников: {len(raw_results)}")
for i, src in enumerate(raw_results[:7], 1):
    snippet = src.get("snippet", "")[:150]
    print(f"  {i}. {src.get('title', 'N/A')[:80]}")
    print(f"     snippet: {snippet}")

print(f"\n--- NLI РЕЗУЛЬТАТ ---")
if nli_result:
    print(f"  max_entailment:    {nli_result.get('max_entailment', 0):.4f}")
    print(f"  max_contradiction: {nli_result.get('max_contradiction', 0):.4f}")
    print(f"  entailment_count:  {nli_result.get('entailment_count', 0)}")
    print(f"  contradiction_count: {nli_result.get('contradiction_count', 0)}")
    print(f"  neutral_count:     {nli_result.get('neutral_count', 0)}")
    print(f"  nli_score:         {nli_result.get('nli_score', 50)}")
    if nli_result.get("pairs"):
        print(f"  --- Per-source ---")
        for p in nli_result["pairs"][:5]:
            print(f"    {p['label']:15s} ent={p['entailment']:.3f} con={p['contradiction']:.3f} src={p.get('source','')[:50]}")
else:
    print("  NLI: НЕ ЗАПУЩЕН или нет данных!")

print(f"\n--- ЧИСЛОВАЯ ПРОВЕРКА ---")
if num_comparisons:
    for c in num_comparisons:
        cm = c["claim_number"]
        sm = c["source_number"]
        if sm:
            print(f"  claim={cm['raw']}, source={sm['raw']}, match={c['match']}, dev={c['deviation']}")
        else:
            print(f"  claim={cm['raw']}, source=НЕ НАЙДЕНО")
else:
    print("  Нет числовых сравнений")

print(f"\n--- RAW VERDICT (LLM) ---")
raw_v = result.get("raw_verdict", "")
print(f"  {raw_v[:500]}")

print(f"\n--- ФИНАЛЬНЫЙ РЕЗУЛЬТАТ ---")
print(f"  Verdict:    {result['verdict']}")
print(f"  Score:      {result['credibility_score']}")
print(f"  Confidence: {result['confidence']}")
print(f"  Reasoning:  {result.get('reasoning', '')[:200]}")
print(f"  Time:       {result['total_time']} sec")
print(f"{'='*60}")
