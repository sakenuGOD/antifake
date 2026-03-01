"""
Universal evaluation script for Antifake fact-checker.

Runs eval_dataset.jsonl through the pipeline and computes:
- Accuracy per category
- Macro-F1 (all verdicts equally weighted)
- Confusion matrix (ПРАВДА/ЛОЖЬ/МАНИПУЛЯЦИЯ/НЕ ПОДТВЕРЖДЕНО)
- Expected Calibration Error (ECE)
- Average time per claim
- Timeout/error rate
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Verdict labels in order
VERDICT_LABELS = ["ПРАВДА", "ЛОЖЬ", "МАНИПУЛЯЦИЯ", "НЕ ПОДТВЕРЖДЕНО"]


def load_dataset(path: str) -> List[dict]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                dataset.append(item)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON parse error: {e}")
    return dataset


def normalize_verdict(verdict: str) -> str:
    """Normalize verdict string to one of the 4 standard labels."""
    v = verdict.strip().upper()

    if v in ("ПРАВДА", "TRUE", "ДОСТОВЕРНО", "ПОДТВЕРЖДЕНО"):
        return "ПРАВДА"
    if v in ("ЛОЖЬ", "FALSE", "ФЕЙК", "НЕДОСТОВЕРНО"):
        return "ЛОЖЬ"
    if v in ("МАНИПУЛЯЦИЯ", "ПОЛУПРАВДА", "ЧАСТИЧНО", "ЧАСТИЧНО ПРАВДА"):
        return "МАНИПУЛЯЦИЯ"
    if v in ("НЕ ПОДТВЕРЖДЕНО", "UNVERIFIED", "НЕИЗВЕСТНО", "НЕ ОПРЕДЕЛЕНО"):
        return "НЕ ПОДТВЕРЖДЕНО"

    # Fallback: check substrings
    if "ПРАВДА" in v or "TRUE" in v:
        return "ПРАВДА"
    if "ЛОЖЬ" in v or "FALSE" in v or "ФЕЙК" in v:
        return "ЛОЖЬ"
    if "МАНИПУЛ" in v or "ЧАСТИЧ" in v or "ПОЛУПРАВД" in v:
        return "МАНИПУЛЯЦИЯ"
    return "НЕ ПОДТВЕРЖДЕНО"


def compute_confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict:
    """Compute confusion matrix for multi-class classification."""
    matrix = {}
    for true_label in VERDICT_LABELS:
        matrix[true_label] = {}
        for pred_label in VERDICT_LABELS:
            matrix[true_label][pred_label] = 0

    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1

    return matrix


def compute_per_class_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Compute precision, recall, F1 per class."""
    metrics = {}
    for label in VERDICT_LABELS:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if yt == label),
        }

    return metrics


def compute_macro_f1(per_class: Dict) -> float:
    """Compute macro-averaged F1 across all classes with support > 0."""
    f1s = [m["f1"] for m in per_class.values() if m["support"] > 0]
    return round(sum(f1s) / len(f1s), 4) if f1s else 0.0


def compute_ece(scores: List[int], y_true: List[str], y_pred: List[str], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    For each confidence bin, measures |accuracy - avg_confidence|.
    """
    if not scores:
        return 0.0

    bins = defaultdict(list)
    for score, yt, yp in zip(scores, y_true, y_pred):
        bin_idx = min(score // (100 // n_bins), n_bins - 1)
        correct = 1 if yt == yp else 0
        bins[bin_idx].append((score / 100.0, correct))

    ece = 0.0
    total = len(scores)
    for bin_idx, items in bins.items():
        if not items:
            continue
        avg_conf = sum(s for s, c in items) / len(items)
        avg_acc = sum(c for s, c in items) / len(items)
        ece += len(items) / total * abs(avg_acc - avg_conf)

    return round(ece, 4)


def is_critical_error(expected: str, predicted: str) -> bool:
    """ПРАВДА<->ЛОЖЬ confusion is the most critical error."""
    return (expected == "ПРАВДА" and predicted == "ЛОЖЬ") or \
           (expected == "ЛОЖЬ" and predicted == "ПРАВДА")


def run_evaluation(pipeline, dataset: List[dict], resume_from: int = 0) -> Dict:
    """Run the pipeline on all claims and collect results."""
    results = []
    errors = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        if i < resume_from:
            continue

        claim = item["claim"]
        expected = normalize_verdict(item["expected_verdict"])
        category = item.get("category", "unknown")
        difficulty = item.get("difficulty", 0)

        print(f"\n[{i+1}/{total}] ({category}, d={difficulty}) {claim[:70]}...")

        t0 = time.time()
        try:
            result = pipeline.check(claim)
            elapsed = time.time() - t0

            raw_verdict = result.get("verdict", "")
            predicted = normalize_verdict(raw_verdict)
            score = result.get("credibility_score", 50)

            correct = predicted == expected
            critical = is_critical_error(expected, predicted)

            status = "OK" if correct else ("CRITICAL" if critical else "MISS")
            print(f"  -> {predicted} (score={score}) expected={expected} [{status}] ({elapsed:.1f}s)")

            results.append({
                "index": i,
                "claim": claim,
                "category": category,
                "difficulty": difficulty,
                "expected_verdict": expected,
                "predicted_verdict": predicted,
                "raw_verdict": raw_verdict,
                "credibility_score": score,
                "correct": correct,
                "critical_error": critical,
                "elapsed_seconds": round(elapsed, 2),
                "reasoning": result.get("reasoning", "")[:500],
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> ERROR: {e} ({elapsed:.1f}s)")
            errors.append({
                "index": i,
                "claim": claim,
                "category": category,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(elapsed, 2),
            })
            results.append({
                "index": i,
                "claim": claim,
                "category": category,
                "difficulty": difficulty,
                "expected_verdict": expected,
                "predicted_verdict": "НЕ ПОДТВЕРЖДЕНО",
                "raw_verdict": f"ERROR: {e}",
                "credibility_score": 50,
                "correct": False,
                "critical_error": False,
                "elapsed_seconds": round(elapsed, 2),
                "reasoning": "",
            })

        # Save intermediate results every 10 claims
        if (i + 1) % 10 == 0:
            _save_intermediate(results, errors)

    return {"results": results, "errors": errors}


def _save_intermediate(results, errors):
    """Save intermediate results for crash recovery."""
    path = "data/eval_intermediate.json"
    os.makedirs("data", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "errors": errors,
                    "saved_at": datetime.now().isoformat()},
                   f, ensure_ascii=False, indent=2)


def compute_all_metrics(results: List[dict], errors: List[dict]) -> Dict:
    """Compute all evaluation metrics from results."""
    if not results:
        return {"error": "No results to evaluate"}

    y_true = [r["expected_verdict"] for r in results]
    y_pred = [r["predicted_verdict"] for r in results]
    scores = [r["credibility_score"] for r in results]
    times = [r["elapsed_seconds"] for r in results]

    # Overall accuracy
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = round(correct / total, 4) if total else 0

    # Critical errors (ПРАВДА<->ЛОЖЬ)
    critical_count = sum(1 for r in results if r["critical_error"])

    # Confusion matrix
    confusion = compute_confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred)
    macro_f1 = compute_macro_f1(per_class)

    # ECE
    ece = compute_ece(scores, y_true, y_pred)

    # Per-category accuracy
    per_category = {}
    cat_results = defaultdict(list)
    for r in results:
        cat_results[r["category"]].append(r)

    for cat, cat_items in sorted(cat_results.items()):
        cat_correct = sum(1 for r in cat_items if r["correct"])
        cat_total = len(cat_items)
        cat_y_true = [r["expected_verdict"] for r in cat_items]
        cat_y_pred = [r["predicted_verdict"] for r in cat_items]
        cat_per_class = compute_per_class_metrics(cat_y_true, cat_y_pred)
        per_category[cat] = {
            "accuracy": round(cat_correct / cat_total, 4) if cat_total else 0,
            "correct": cat_correct,
            "total": cat_total,
            "macro_f1": compute_macro_f1(cat_per_class),
            "avg_time": round(sum(r["elapsed_seconds"] for r in cat_items) / cat_total, 2),
        }

    # Per-difficulty accuracy
    per_difficulty = {}
    diff_results = defaultdict(list)
    for r in results:
        diff_results[r["difficulty"]].append(r)

    for diff, diff_items in sorted(diff_results.items()):
        d_correct = sum(1 for r in diff_items if r["correct"])
        d_total = len(diff_items)
        per_difficulty[str(diff)] = {
            "accuracy": round(d_correct / d_total, 4) if d_total else 0,
            "correct": d_correct,
            "total": d_total,
        }

    # Timing
    avg_time = round(sum(times) / len(times), 2) if times else 0
    timeout_count = sum(1 for t in times if t > 300)

    # Verdict distribution
    pred_dist = Counter(y_pred)
    true_dist = Counter(y_true)

    return {
        "timestamp": datetime.now().isoformat(),
        "total_claims": total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "critical_errors": critical_count,
        "critical_error_rate": round(critical_count / total, 4) if total else 0,
        "ece": ece,
        "avg_time_seconds": avg_time,
        "timeout_count": timeout_count,
        "timeout_rate": round(timeout_count / total, 4) if total else 0,
        "error_count": len(errors),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "per_category": per_category,
        "per_difficulty": per_difficulty,
        "predicted_distribution": dict(pred_dist),
        "expected_distribution": dict(true_dist),
    }


def print_report(metrics: Dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print("ANTIFAKE UNIVERSAL EVALUATION REPORT")
    print("=" * 70)

    print(f"\nTotal claims:        {metrics['total_claims']}")
    print(f"Accuracy:            {metrics['accuracy']:.1%}")
    print(f"Macro-F1:            {metrics['macro_f1']:.4f}")
    print(f"Critical errors:     {metrics['critical_errors']} ({metrics['critical_error_rate']:.1%})")
    print(f"ECE (calibration):   {metrics['ece']:.4f}")
    print(f"Avg time/claim:      {metrics['avg_time_seconds']:.1f}s")
    print(f"Timeouts (>300s):    {metrics['timeout_count']} ({metrics['timeout_rate']:.1%})")
    print(f"Errors:              {metrics['error_count']}")

    print(f"\n--- Per-class metrics ---")
    print(f"{'Verdict':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    for label in VERDICT_LABELS:
        m = metrics["per_class"].get(label, {})
        print(f"{label:<20s} {m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} "
              f"{m.get('f1',0):>10.4f} {m.get('support',0):>10d}")

    print(f"\n--- Per-category accuracy ---")
    print(f"{'Category':<20s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>10s} {'F1':>10s} {'Avg time':>10s}")
    for cat, m in sorted(metrics["per_category"].items()):
        print(f"{cat:<20s} {m['accuracy']:>10.1%} {m['correct']:>10d} {m['total']:>10d} "
              f"{m['macro_f1']:>10.4f} {m['avg_time']:>9.1f}s")

    print(f"\n--- Per-difficulty accuracy ---")
    print(f"{'Difficulty':<12s} {'Accuracy':>10s} {'Correct':>8s} {'Total':>8s}")
    for diff, m in sorted(metrics["per_difficulty"].items()):
        print(f"{'d=' + diff:<12s} {m['accuracy']:>10.1%} {m['correct']:>8d} {m['total']:>8d}")

    print(f"\n--- Confusion matrix ---")
    cm = metrics["confusion_matrix"]
    labels_short = ["ПРАВДА", "ЛОЖЬ", "МАНИПУЛ", "НЕ ПОДТВ"]
    print(f"{'Expected\\Predicted':<16s}", end="")
    for ls in labels_short:
        print(f"{ls:>10s}", end="")
    print()
    for i, label in enumerate(VERDICT_LABELS):
        print(f"{labels_short[i]:<16s}", end="")
        for pred_label in VERDICT_LABELS:
            val = cm.get(label, {}).get(pred_label, 0)
            print(f"{val:>10d}", end="")
        print()

    print(f"\n--- Verdict distribution ---")
    print(f"{'Verdict':<20s} {'Expected':>10s} {'Predicted':>10s}")
    for label in VERDICT_LABELS:
        exp = metrics["expected_distribution"].get(label, 0)
        pred = metrics["predicted_distribution"].get(label, 0)
        print(f"{label:<20s} {exp:>10d} {pred:>10d}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Universal Antifake Evaluation")
    parser.add_argument("--dataset", "-d", type=str,
                        default="data/eval_dataset.jsonl",
                        help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--output", "-o", type=str,
                        default="data/eval_universal_results.json",
                        help="Path to save results")
    parser.add_argument("--resume", "-r", type=int, default=0,
                        help="Resume from claim index (0-based)")
    parser.add_argument("--limit", "-l", type=int, default=0,
                        help="Limit number of claims (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dataset stats without running pipeline")
    args = parser.parse_args()

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if not os.path.exists(dataset_path):
        dataset_path = args.dataset
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} claims from {args.dataset}")

    if args.limit > 0:
        dataset = dataset[:args.limit]
        print(f"Limited to {len(dataset)} claims")

    # Dataset stats
    cats = Counter(d["category"] for d in dataset)
    vds = Counter(d["expected_verdict"] for d in dataset)
    print(f"\nCategories: {dict(cats)}")
    print(f"Verdicts:   {dict(vds)}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running pipeline.")
        return

    # Load pipeline
    print("\n" + "=" * 70)
    print("Loading Antifake pipeline...")
    print("=" * 70)

    from pipeline import FactCheckPipeline
    from config import SearchConfig
    from model import find_best_adapter

    serpapi_key = os.environ.get("SERPAPI_API_KEY", "")
    search_config = SearchConfig(api_key=serpapi_key)
    adapter_path = find_best_adapter()
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    else:
        print("No adapter found, using base model")

    pipeline = FactCheckPipeline(adapter_path=adapter_path, search_config=search_config)

    # Run evaluation
    print(f"\nStarting evaluation: {len(dataset)} claims, resume={args.resume}")
    print("=" * 70)

    eval_data = run_evaluation(pipeline, dataset, resume_from=args.resume)

    # Compute metrics
    metrics = compute_all_metrics(eval_data["results"], eval_data["errors"])

    # Print report
    print_report(metrics)

    # Save full results
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    full_output = {
        "metrics": metrics,
        "results": eval_data["results"],
        "errors": eval_data["errors"],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")

    # Also save metrics-only summary
    summary_path = output_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_path.replace(os.path.dirname(__file__) + '/', '')}")


if __name__ == "__main__":
    main()
