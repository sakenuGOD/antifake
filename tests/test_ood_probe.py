"""V20 out-of-distribution probe — claims the model has NOT seen during
training or in hard10/manipulative. Checks that subject-mention guard
generalizes and doesn't break OOD.

Focus categories:
- Fresh person/entity swaps (not Бутерин/Сидней)
- Subject-only claims (no proper-noun entity swaps)
- Compound-name claims
- Ordinal/event-name claims (Русский XIX)
- Role/origin swaps
- Generalizations / superlatives

Short list (8 claims) — run quickly, catches regressions without 30-min wait.
"""
import _path  # noqa: F401,E402 — inject project root into sys.path

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OOD_CLAIMS = [
    # Fresh person swaps (not in training, different domain)
    {"claim": "Роман «Война и мир» написал Фёдор Достоевский",
     "expected": "ЛОЖЬ", "category": "person_swap_literature",
     "note": "Толстой написал Войну и мир, не Достоевский"},
    {"claim": "Теорию относительности сформулировал Исаак Ньютон",
     "expected": "ЛОЖЬ", "category": "person_swap_science",
     "note": "Эйнштейн, не Ньютон"},

    # Fresh location swaps
    {"claim": "Столица Канады — Торонто",
     "expected": "ЛОЖЬ", "category": "location_swap",
     "note": "Оттава"},
    {"claim": "Река Амазонка протекает в Африке",
     "expected": "ЛОЖЬ", "category": "location_continent_swap",
     "note": "Южная Америка"},

    # True claims with subject matching
    {"claim": "Москва является столицей России",
     "expected": "ПРАВДА", "category": "basic_true"},
    {"claim": "Эверест — самая высокая гора планеты",
     "expected": "ПРАВДА", "category": "superlative_true"},

    # Numerical / temporal
    {"claim": "Берлинская стена пала в 1979 году",
     "expected": "ЛОЖЬ", "category": "date_swap_event",
     "note": "1989"},

    # Science fact (no proper-noun subject — verifies no regression on no-PN claims)
    {"claim": "Скорость света в вакууме составляет примерно 300 000 километров в секунду",
     "expected": "ПРАВДА", "category": "science_numerical_true"},
]


def norm_verdict(v: str) -> str:
    v = (v or "").upper().strip()
    if "ПРАВДА" in v or v in ("TRUE", "ИСТИНА"):
        return "ПРАВДА"
    if "ЛОЖЬ" in v or v in ("FALSE", "СКАМ"):
        return "ЛОЖЬ"
    return "НЕ УВЕРЕНА"


def main():
    import os
    import argparse
    from pipeline import FactCheckPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    args, _ = parser.parse_known_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if args.adapter:
        adapter = args.adapter if os.path.isabs(args.adapter) else os.path.join(project_root, args.adapter)
    else:
        sft_path = os.path.join(project_root, "adapters", "fact_checker_lora")
        adapter = sft_path if os.path.exists(sft_path) else None

    print(f"\n=== V20 OOD PROBE ({len(OOD_CLAIMS)} claims) ===")
    print(f"Adapter: {adapter}\n")

    pipe = FactCheckPipeline(adapter_path=adapter)

    import gc
    try:
        import torch
    except Exception:
        torch = None

    results = []
    correct = 0
    for i, sample in enumerate(OOD_CLAIMS, 1):
        claim = sample["claim"]
        expected = sample["expected"]
        print(f"\n[{i}/{len(OOD_CLAIMS)}] {claim}", flush=True)
        t0 = time.time()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            r = pipe.check(claim)
            actual = norm_verdict(r.get("verdict", ""))
            score = r.get("credibility_score", 0)
            latency = time.time() - t0
            ok = (actual == expected)
            if ok:
                correct += 1
            status = "OK  " if ok else "FAIL"
            print(f"  [{status}] pred={actual:12s} | exp={expected:8s} | score={score:3d} | {latency:.1f}s", flush=True)
            results.append({
                "claim": claim,
                "expected": expected,
                "predicted": actual,
                "score": score,
                "correct": ok,
                "category": sample["category"],
                "note": sample.get("note", ""),
                "latency": round(latency, 2),
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"claim": claim, "expected": expected, "error": str(e), "correct": False})

    accuracy = correct / len(OOD_CLAIMS)
    print(f"\n{'='*60}")
    print(f"OOD ACCURACY: {correct}/{len(OOD_CLAIMS)} = {accuracy:.1%}")
    print(f"{'='*60}")

    out = Path(__file__).parent.parent / "data" / "ood_probe_results.json"
    out.write_text(json.dumps({
        "accuracy": accuracy,
        "correct": correct,
        "total": len(OOD_CLAIMS),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
