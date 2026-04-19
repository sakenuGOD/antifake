"""Quick 15-claim test across difficulty levels."""
import _path  # noqa: F401,E402 — inject project root into sys.path
import time
import sys
import os

# 15 claims: 3 easy, 4 medium, 4 hard, 4 very hard
TEST_CLAIMS = [
    # === EASY (d=1) ===
    {"claim": "Москва является столицей России", "expected": "ПРАВДА", "diff": 1, "cat": "simple_true"},
    {"claim": "Париж является столицей Германии", "expected": "ЛОЖЬ", "diff": 1, "cat": "simple_false"},
    {"claim": "Амазонка протекает через Африку", "expected": "ЛОЖЬ", "diff": 1, "cat": "simple_false"},

    # === MEDIUM (d=2) ===
    {"claim": "Скорость звука в воздухе составляет примерно 343 метра в секунду", "expected": "ПРАВДА", "diff": 2, "cat": "numerical_true"},
    {"claim": "Эверест имеет высоту 3000 метров", "expected": "ЛОЖЬ", "diff": 2, "cat": "numerical_false"},
    {"claim": "Microsoft основана Стивом Джобсом", "expected": "ЛОЖЬ", "diff": 2, "cat": "person_swap"},
    {"claim": "Титаник затонул в Тихом океане", "expected": "ЛОЖЬ", "diff": 3, "cat": "location_swap"},

    # === HARD (d=3-4) ===
    {"claim": "Россия является самой большой страной мира и имеет население более 300 миллионов человек", "expected": "МАНИПУЛЯЦИЯ", "diff": 3, "cat": "composite"},
    {"claim": "Великая Китайская стена видна из космоса невооружённым глазом", "expected": "ЛОЖЬ", "diff": 4, "cat": "myths"},
    {"claim": "Мы используем только 10 процентов нашего мозга", "expected": "ЛОЖЬ", "diff": 4, "cat": "myths"},
    {"claim": "Марс имеет 2 спутника и температуру поверхности +50 градусов Цельсия", "expected": "МАНИПУЛЯЦИЯ", "diff": 4, "cat": "composite"},

    # === VERY HARD (d=4-5) ===
    {"claim": "Юрий Гагарин совершил первый полёт в космос в 1961 году и высадился на Луне", "expected": "МАНИПУЛЯЦИЯ", "diff": 3, "cat": "composite"},
    {"claim": "Кофе вреден для здоровья", "expected": "НЕ ПОДТВЕРЖДЕНО", "diff": 4, "cat": "ambiguous"},
    {"claim": "Быки приходят в ярость от красного цвета", "expected": "ЛОЖЬ", "diff": 3, "cat": "myths"},
    {"claim": "Наполеон Бонапарт был очень маленького роста", "expected": "ЛОЖЬ", "diff": 4, "cat": "myths"},
]

VERDICT_NORMALIZE = {
    # V13: Order matters! Check longer/more specific patterns FIRST to avoid
    # "ПРАВДА" matching inside "ПОЛУПРАВДА" or "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА"
    "МАНИПУЛЯЦИЯ / ПОЛУПРАВДА": "МАНИПУЛЯЦИЯ",
    "МАНИПУЛЯЦИЯ": "МАНИПУЛЯЦИЯ",
    "ПОЛУПРАВДА": "МАНИПУЛЯЦИЯ",
    "ЧАСТИЧНО ПОДТВЕРЖДЕНО": "МАНИПУЛЯЦИЯ",
    "ЧАСТИЧНО": "МАНИПУЛЯЦИЯ",
    "НЕ ПОДТВЕРЖДЕНО": "НЕ ПОДТВЕРЖДЕНО",
    "ЛОЖЬ": "ЛОЖЬ", "FALSE": "ЛОЖЬ", "ФЕЙК": "ЛОЖЬ",
    "ПРАВДА": "ПРАВДА", "TRUE": "ПРАВДА",
}

def norm(v):
    v = v.strip().upper()
    # V13: Check longer keys first to avoid substring false positives
    for key, val in VERDICT_NORMALIZE.items():
        if key in v:
            return val
    return "НЕ ПОДТВЕРЖДЕНО"

def main():
    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    adapter_path = find_best_adapter()
    print(f"Adapter: {adapter_path or 'base'}")
    pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=SearchConfig())

    results = []
    total_time = 0

    for i, tc in enumerate(TEST_CLAIMS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/15] (d={tc['diff']}, {tc['cat']}) {tc['claim']}")
        print(f"Expected: {tc['expected']}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            result = pipeline.check(tc["claim"])
            elapsed = time.time() - t0
            total_time += elapsed

            predicted = norm(result.get("verdict", ""))
            score = result.get("credibility_score", 50)
            correct = predicted == tc["expected"]

            status = "OK" if correct else "MISS"
            print(f"\n>>> RESULT: {predicted} (score={score}) [{status}] ({elapsed:.1f}s)")

            results.append({
                "claim": tc["claim"][:50],
                "expected": tc["expected"],
                "predicted": predicted,
                "score": score,
                "correct": correct,
                "time": round(elapsed, 1),
                "diff": tc["diff"],
                "cat": tc["cat"],
            })
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"\n>>> ERROR: {e} ({elapsed:.1f}s)")
            results.append({
                "claim": tc["claim"][:50],
                "expected": tc["expected"],
                "predicted": "ERROR",
                "score": 0,
                "correct": False,
                "time": round(elapsed, 1),
                "diff": tc["diff"],
                "cat": tc["cat"],
            })

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY: 15 claims test")
    print(f"{'='*70}")
    correct_count = sum(1 for r in results if r["correct"])
    print(f"Accuracy: {correct_count}/15 ({correct_count/15:.0%})")
    print(f"Avg time: {total_time/15:.1f}s")
    print(f"Total time: {total_time:.0f}s")

    print(f"\n{'Claim':<52s} {'Exp':<15s} {'Got':<15s} {'Score':>5s} {'Time':>6s} {'OK':>4s}")
    print("-" * 100)
    for r in results:
        ok = "OK" if r["correct"] else "MISS"
        print(f"{r['claim']:<52s} {r['expected']:<15s} {r['predicted']:<15s} {r['score']:>5d} {r['time']:>5.1f}s {ok:>4s}")

    # Per-difficulty
    print(f"\nPer difficulty:")
    for d in sorted(set(r["diff"] for r in results)):
        d_results = [r for r in results if r["diff"] == d]
        d_correct = sum(1 for r in d_results if r["correct"])
        print(f"  d={d}: {d_correct}/{len(d_results)}")

    # Per-category
    print(f"\nPer category:")
    for cat in sorted(set(r["cat"] for r in results)):
        c_results = [r for r in results if r["cat"] == cat]
        c_correct = sum(1 for r in c_results if r["correct"])
        print(f"  {cat}: {c_correct}/{len(c_results)}")

    # Critical errors (ПРАВДА<->ЛОЖЬ)
    critical = sum(1 for r in results
                   if (r["expected"] == "ПРАВДА" and r["predicted"] == "ЛОЖЬ")
                   or (r["expected"] == "ЛОЖЬ" and r["predicted"] == "ПРАВДА"))
    print(f"\nCritical errors (ПРАВДА<->ЛОЖЬ): {critical}")

if __name__ == "__main__":
    main()
