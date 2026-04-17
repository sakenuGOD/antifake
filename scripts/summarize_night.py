"""Aggregate all eval_*.json from night_run into a single markdown report."""

import argparse
import glob
import json
import os
import sys


def load_metrics(path):
    try:
        with open(path) as f:
            data = json.load(f)
        if "metrics" in data:
            return data["metrics"]
        return data
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        return None


def row_for(path, label):
    m = load_metrics(path)
    if not m:
        return f"| {label} | — | — | — | — | (failed) |\n"
    return (
        f"| {label} | {m.get('accuracy', 0):.1%} | {m.get('macro_f1', 0):.3f} "
        f"| {m.get('critical_errors', 0)} | {m.get('ece', 0):.3f} "
        f"| {m.get('avg_time_seconds', 0):.1f}s |\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    out = []
    out.append(f"# Night run report — {os.path.basename(args.log_dir)}\n\n")
    out.append("## Baseline (before retrain)\n\n")
    out.append("| Model | Accuracy | Macro-F1 | Critical | ECE | Avg time |\n")
    out.append("|-------|----------|----------|----------|-----|----------|\n")

    for label, fname in [
        ("base Mistral", "baseline_01_base.json"),
        ("SFT (fact_checker_lora)", "baseline_02_sft.json"),
        ("GRPO old (broken)", "baseline_03_grpo_broken.json"),
    ]:
        out.append(row_for(os.path.join(args.log_dir, fname), label))

    out.append("\n## GRPO v2 checkpoints\n\n")
    out.append("| Checkpoint | Accuracy | Macro-F1 | Critical | ECE | Avg time |\n")
    out.append("|------------|----------|----------|----------|-----|----------|\n")

    ckpt_files = sorted(glob.glob(os.path.join(args.log_dir, "eval_step*.json")),
                         key=lambda p: int(os.path.basename(p)
                                            .replace("eval_step", "").replace(".json", "")))
    for path in ckpt_files:
        step = os.path.basename(path).replace("eval_step", "").replace(".json", "")
        out.append(row_for(path, f"step-{step}"))

    final = os.path.join(args.log_dir, "eval_final.json")
    if os.path.exists(final):
        out.append(row_for(final, "final (saved)"))

    out.append("\n## Recommendation\n\n")

    best = None
    best_acc = 0
    # Find best checkpoint
    for path in ckpt_files + [final]:
        if not os.path.exists(path):
            continue
        m = load_metrics(path)
        if m and m.get("accuracy", 0) > best_acc:
            best_acc = m["accuracy"]
            best = path

    sft = load_metrics(os.path.join(args.log_dir, "baseline_02_sft.json"))
    sft_acc = sft.get("accuracy", 0) if sft else 0

    if best is None:
        out.append("Нет ни одного успешного eval — проверь ошибки в retrain log.\n")
    else:
        label = os.path.basename(best).replace("eval_", "").replace(".json", "")
        out.append(f"Best GRPO v2: **{label}** — accuracy {best_acc:.1%}\n")
        out.append(f"SFT baseline: accuracy {sft_acc:.1%}\n\n")
        if best_acc > sft_acc + 0.02:
            out.append(f"**GRPO v2 лучше SFT на {(best_acc-sft_acc)*100:.1f}pp.** "
                        f"Промоут: `mv adapters/fact_checker_grpo adapters/_fact_checker_grpo_broken && "
                        f"cp -r {best.replace('.json','').replace(args.log_dir+'/eval_','').replace('step','adapters/fact_checker_grpo_v2/checkpoint-')} adapters/fact_checker_grpo`\n")
        elif best_acc >= sft_acc - 0.01:
            out.append("GRPO v2 примерно равен SFT. Оставить SFT как основной адаптер.\n")
        else:
            out.append(f"GRPO v2 **хуже** SFT на {(sft_acc-best_acc)*100:.1f}pp. "
                        "Использовать SFT как основной. Отключить GRPO переименованием:\n"
                        "`mv adapters/fact_checker_grpo adapters/_fact_checker_grpo_broken`\n")

    print("".join(out))


if __name__ == "__main__":
    main()
