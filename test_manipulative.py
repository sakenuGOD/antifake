"""Hard test: manipulative, conspiratorial and speculative claims.

Distinct from test_hard10 — these are not just factual errors but deliberate
disinformation / common myths / pseudo-scientific claims. Tests whether the
pipeline can detect:
- Conspiracy theories (Moon landing hoax, 5G→cancer, Covid bioweapon)
- Persistent myths (10% brain, horned Viking helmets, Great Wall from space)
- Manipulative framing of partial truth (Tesla founding, climate consensus)
- Debunked pseudo-science (flat Earth, vaccines→autism)

Target: ≥80% (harder than test_hard10 — these lean on myth/debunk detection
and less on rigid Wikidata facts).
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MANIPULATIVE_CLAIMS = [
    {"claim": "Высадка американских астронавтов на Луну в 1969 году была снята в голливудской студии",
     "expected": "ЛОЖЬ", "category": "conspiracy"},
    {"claim": "Мозг человека использует лишь 10 процентов своих возможностей",
     "expected": "ЛОЖЬ", "category": "myth"},
    {"claim": "Викинги носили шлемы с рогами в боевых сражениях",
     "expected": "ЛОЖЬ", "category": "myth"},
    {"claim": "Прививки вызывают у детей аутизм",
     "expected": "ЛОЖЬ", "category": "pseudoscience"},
    {"claim": "Излучение сетей 5G вызывает онкологические заболевания у людей",
     "expected": "ЛОЖЬ", "category": "conspiracy"},
    {"claim": "Земля имеет плоскую форму",
     "expected": "ЛОЖЬ", "category": "pseudoscience"},
    {"claim": "Великая Китайская стена является единственным рукотворным объектом, видимым с Луны невооружённым глазом",
     "expected": "ЛОЖЬ", "category": "myth"},
    {"claim": "COVID-19 был создан как биологическое оружие в лаборатории",
     "expected": "ЛОЖЬ", "category": "conspiracy"},
    {"claim": "Более 97 процентов климатологов согласны, что глобальное потепление вызвано деятельностью человека",
     "expected": "ПРАВДА", "category": "scientific_consensus"},
    {"claim": "Наполеон Бонапарт был человеком маленького роста по меркам своего времени",
     "expected": "ЛОЖЬ", "category": "myth"},
]


def norm_verdict(v: str) -> str:
    v = (v or "").upper().strip()
    if "ПРАВДА" in v or v in ("TRUE", "ИСТИНА"):
        return "ПРАВДА"
    if "ЛОЖЬ" in v or v in ("FALSE", "СКАМ"):
        return "ЛОЖЬ"
    return "НЕ УВЕРЕНА"


def main():
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

    print(f"\n=== MANIPULATIVE / SPECULATIVE TEST (10 claims) ===")
    print(f"Adapter: {adapter}\n")

    pipe = FactCheckPipeline(adapter_path=adapter)

    import gc
    try:
        import torch
    except Exception:
        torch = None

    results = []
    correct = 0
    # Save incrementally so crashes don't lose progress.
    out = Path(__file__).parent / "data" / "manipulative_results.json"

    for i, sample in enumerate(MANIPULATIVE_CLAIMS, 1):
        claim = sample["claim"]
        expected = sample["expected"]
        print(f"[{i}/10] {claim}", flush=True)
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
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
                print(f"         reason: {reasoning}", flush=True)
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
            print(f"  [ERROR] {e}", flush=True)
            results.append({"claim": claim, "expected": expected, "error": str(e), "correct": False,
                            "category": sample["category"]})
        # Incremental save
        out.write_text(json.dumps({
            "progress": f"{i}/10",
            "correct_so_far": correct,
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    accuracy = correct / len(MANIPULATIVE_CLAIMS)
    print(f"\n{'='*60}")
    print(f"ACCURACY: {correct}/{len(MANIPULATIVE_CLAIMS)} = {accuracy:.1%}")
    print(f"TARGET:   80% ({'PASS' if accuracy >= 0.8 else 'FAIL'})")
    print(f"{'='*60}")

    # By-category breakdown
    cats = {}
    for r in results:
        c = r.get("category", "?")
        cats.setdefault(c, []).append(r.get("correct", False))
    for c, arr in cats.items():
        acc = sum(arr) / len(arr) if arr else 0
        print(f"  {c:20s}: {sum(arr)}/{len(arr)} = {acc:.0%}")

    out.write_text(json.dumps({
        "accuracy": accuracy,
        "correct": correct,
        "total": len(MANIPULATIVE_CLAIMS),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    sys.exit(0 if accuracy >= 0.8 else 1)


if __name__ == "__main__":
    main()
