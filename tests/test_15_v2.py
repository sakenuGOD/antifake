"""Quick 15-claim test v2 — new claims, easy→hard.

V15: 3 verdicts (ПРАВДА / ЛОЖЬ / НЕ УВЕРЕНА) + СОСТАВНОЕ for composite claims.
"""
import _path  # noqa: F401,E402 — inject project root into sys.path
import time
import sys
import os

# 15 claims: 3 easy, 4 medium, 4 hard, 4 very hard
TEST_CLAIMS = [
    # === EASY (d=1) ===
    {"claim": "Земля вращается вокруг Солнца", "expected": "ПРАВДА", "diff": 1, "cat": "simple_true"},
    {"claim": "Австралия является частью Европы", "expected": "ЛОЖЬ", "diff": 1, "cat": "simple_false"},
    {"claim": "Токио является столицей Японии", "expected": "ПРАВДА", "diff": 1, "cat": "simple_true"},

    # === MEDIUM (d=2) ===
    {"claim": "Температура кипения воды составляет 100 градусов Цельсия", "expected": "ПРАВДА", "diff": 2, "cat": "numerical_true"},
    {"claim": "Расстояние от Земли до Солнца составляет 15 миллионов километров", "expected": "ЛОЖЬ", "diff": 2, "cat": "numerical_false"},
    {"claim": "Первая мировая война началась в 1939 году", "expected": "ЛОЖЬ", "diff": 2, "cat": "date_swap"},
    {"claim": "Александр Пушкин написал роман Война и мир", "expected": "ЛОЖЬ", "diff": 2, "cat": "person_swap"},

    # === HARD (d=3-4) ===
    {"claim": "Альберт Эйнштейн родился в Германии и получил Нобелевскую премию по математике",
     "expected": "СОСТАВНОЕ", "diff": 3, "cat": "composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},
    {"claim": "Молния никогда не бьёт в одно и то же место дважды", "expected": "ЛОЖЬ", "diff": 3, "cat": "myths"},
    {"claim": "Золотая рыбка помнит события только 3 секунды", "expected": "ЛОЖЬ", "diff": 3, "cat": "myths"},
    {"claim": "Земля имеет один естественный спутник и расстояние до Луны составляет около 38 тысяч километров",
     "expected": "СОСТАВНОЕ", "diff": 3, "cat": "composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},

    # === VERY HARD (d=4-5) ===
    {"claim": "Сахар вызывает гиперактивность у детей", "expected": "ЛОЖЬ", "diff": 4, "cat": "myths"},
    {"claim": "Мобильные телефоны вызывают рак мозга", "expected": "НЕ УВЕРЕНА", "diff": 4, "cat": "ambiguous"},
    {"claim": "Человек проглатывает в среднем 8 пауков в год во сне", "expected": "ЛОЖЬ", "diff": 4, "cat": "myths"},
    {"claim": "Дмитрий Менделеев изобрёл периодическую таблицу и открыл водку",
     "expected": "СОСТАВНОЕ", "diff": 4, "cat": "composite",
     "expected_subs": ["ПРАВДА", "ЛОЖЬ"]},
]

VERDICT_NORMALIZE = {
    "СОСТАВНОЕ": "СОСТАВНОЕ",
    "СОСТАВНОЕ УТВЕРЖДЕНИЕ": "СОСТАВНОЕ",
    "НЕ УВЕРЕНА": "НЕ УВЕРЕНА",
    "ЛОЖЬ": "ЛОЖЬ", "FALSE": "ЛОЖЬ", "ФЕЙК": "ЛОЖЬ",
    "ПРАВДА": "ПРАВДА", "TRUE": "ПРАВДА",
    # Legacy (backward compat)
    "МАНИПУЛЯЦИЯ": "СОСТАВНОЕ",
    "ПОЛУПРАВДА": "СОСТАВНОЕ",
    "ЧАСТИЧНО ПОДТВЕРЖДЕНО": "СОСТАВНОЕ",
    "НЕ ПОДТВЕРЖДЕНО": "НЕ УВЕРЕНА",
}

def norm(v):
    v = v.strip().upper()
    for key, val in VERDICT_NORMALIZE.items():
        if key in v:
            return val
    return "НЕ УВЕРЕНА"


def check_composite_subs(result, expected_subs):
    """Check if sub-verdicts match expected pattern (order-independent)."""
    sub_verdicts = result.get("sub_verdicts", [])
    if not sub_verdicts or not expected_subs:
        return False
    actual_statuses = [sv.get("status", "").upper() for sv in sub_verdicts]
    # Check that expected sub-verdicts are all present (order-independent)
    expected_sorted = sorted(expected_subs)
    actual_sorted = sorted(actual_statuses[:len(expected_subs)])
    return expected_sorted == actual_sorted


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
        if tc.get("expected_subs"):
            print(f"Expected subs: {tc['expected_subs']}")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            result = pipeline.check(tc["claim"])
            elapsed = time.time() - t0
            total_time += elapsed

            predicted = norm(result.get("verdict", ""))
            score = result.get("credibility_score", 50)

            # For composite claims, check sub-verdicts too
            if tc["expected"] == "СОСТАВНОЕ" and tc.get("expected_subs"):
                correct = predicted == "СОСТАВНОЕ" and check_composite_subs(result, tc["expected_subs"])
                subs_actual = [sv.get("status", "?") for sv in result.get("sub_verdicts", [])]
                print(f"  Sub-verdicts: {subs_actual} (expected: {tc['expected_subs']})")
            else:
                correct = predicted == tc["expected"]

            status = "OK" if correct else "MISS"
            print(f"\n>>> RESULT: {predicted} (score={score}) [{status}] ({elapsed:.1f}s)")

            # Show explanation for НЕ УВЕРЕНА
            if predicted == "НЕ УВЕРЕНА" and result.get("_explanation"):
                print(f"    Explanation: {result['_explanation']}")

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
    print(f"SUMMARY: 15 claims test v2 (V15: 3 verdicts + СОСТАВНОЕ)")
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
