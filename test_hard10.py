"""V18 hard test: 10 adversarial claims covering all known failure patterns.

Target: >=9/10 (90%) accuracy.

Failure categories covered:
- Pseudo-authority / clickbait entailment (Солнце вокруг Земли)
- Well-known facts that were failing with NLI contradiction (Пушкин, Гагарин, ДНК)
- NUM+NLI coherence bug (Кислород 21%)
- Location/person swaps (Столица Австралии, Биткоин/Бутерин)
- Date swap (ВМВ 1943)
- НЕ ПОДТВЕРЖДЕНО fallback traps (Эйфелева, Сахара)
"""

import json
import sys
import time
from pathlib import Path

# Windows cp1251 can't encode Unicode glyphs (✓, checkmarks, etc.) emitted by
# search.py/pipeline.py. Force UTF-8 on stdout/stderr before any import.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HARD_CLAIMS = [
    {"claim": "Александр Пушкин написал роман в стихах Евгений Онегин",
     "expected": "ПРАВДА", "category": "well_known_fact"},
    {"claim": "Юрий Гагарин был первым человеком в космосе",
     "expected": "ПРАВДА", "category": "well_known_fact"},
    {"claim": "Солнце вращается вокруг Земли",
     "expected": "ЛОЖЬ", "category": "pseudo_authority_false"},
    {"claim": "Эйфелева башня находится в Париже",
     "expected": "ПРАВДА", "category": "location_true"},
    {"claim": "Кислород составляет примерно 21% атмосферы Земли",
     "expected": "ПРАВДА", "category": "numerical_true"},
    {"claim": "Вторая мировая война закончилась в 1943 году",
     "expected": "ЛОЖЬ", "category": "date_swap"},
    {"claim": "ДНК содержит генетическую информацию живых организмов",
     "expected": "ПРАВДА", "category": "science_fact"},
    {"claim": "Биткоин был создан Виталиком Бутериным",
     "expected": "ЛОЖЬ", "category": "person_swap"},
    {"claim": "Столица Австралии — Сидней",
     "expected": "ЛОЖЬ", "category": "location_trap"},
    {"claim": "Сахара является самой большой жаркой пустыней в мире",
     "expected": "ПРАВДА", "category": "general_true"},
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
    parser.add_argument("--adapter", type=str, default=None,
                        help="Путь к LoRA адаптеру (по умолчанию: adapters/fact_checker_lora)")
    args, _ = parser.parse_known_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if args.adapter:
        adapter = args.adapter if os.path.isabs(args.adapter) else os.path.join(project_root, args.adapter)
    else:
        sft_path = os.path.join(project_root, "adapters", "fact_checker_lora")
        adapter = sft_path if os.path.exists(sft_path) else None

    print(f"\n=== V18 HARD TEST (10 claims) ===")
    print(f"Adapter: {adapter}\n")

    pipe = FactCheckPipeline(adapter_path=adapter)

    import gc
    try:
        import torch
    except Exception:
        torch = None

    results = []
    correct = 0
    for i, sample in enumerate(HARD_CLAIMS, 1):
        claim = sample["claim"]
        expected = sample["expected"]
        print(f"[{i}/10] {claim}", flush=True)
        t0 = time.time()
        # Free VRAM between claims — NLI cross-encoder and LLM caches accumulate
        # and can crash subsequent iterations on 12GB (Blackwell/RTX 5070).
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
            if not ok:
                reasoning = (r.get("reasoning", "") or "")[:200].replace("\n", " | ")
                print(f"         reason: {reasoning}")
            results.append({
                "claim": claim,
                "expected": expected,
                "predicted": actual,
                "score": score,
                "correct": ok,
                "category": sample["category"],
                "latency": round(latency, 2),
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"claim": claim, "expected": expected, "error": str(e), "correct": False})

    accuracy = correct / len(HARD_CLAIMS)
    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{len(HARD_CLAIMS)} = {accuracy:.1%}")
    print(f"TARGET:   90% ({'PASS' if accuracy >= 0.9 else 'FAIL'})")
    print(f"{'='*60}")

    out = Path(__file__).parent / "data" / "hard10_results.json"
    out.write_text(json.dumps({
        "accuracy": accuracy,
        "correct": correct,
        "total": len(HARD_CLAIMS),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")

    sys.exit(0 if accuracy >= 0.9 else 1)


if __name__ == "__main__":
    main()
