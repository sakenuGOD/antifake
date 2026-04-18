"""
Merge существующего train_russian.jsonl с failure-patterns (с upsampling).

Output: data/train_v2_combined.jsonl
  = train_russian.jsonl (2500) + train_v2_failures.jsonl x 6 (3000)
  = ~5500 examples, failure patterns составляют ~55% от общего объёма
"""

import json
import random
from pathlib import Path

random.seed(42)

DATA = Path(__file__).parent / "data"
SRC_EXISTING = DATA / "train_russian.jsonl"
SRC_FAILURES = DATA / "train_v2_failures.jsonl"
OUT = DATA / "train_v2_combined.jsonl"

UPSAMPLE_FAILURES = 6


def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print(f"Читаю {SRC_EXISTING.name}…")
    existing = read_jsonl(SRC_EXISTING)
    print(f"  {len(existing)} примеров")

    print(f"Читаю {SRC_FAILURES.name}…")
    failures = read_jsonl(SRC_FAILURES)
    print(f"  {len(failures)} примеров")

    print(f"Upsampling failures x {UPSAMPLE_FAILURES}…")
    upsampled = failures * UPSAMPLE_FAILURES
    print(f"  {len(upsampled)} examples после upsample")

    combined = existing + upsampled
    random.shuffle(combined)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    frac = len(upsampled) / len(combined) * 100
    print(f"\nЗаписано {len(combined)} в {OUT}")
    print(f"  failure patterns: {frac:.1f}%")
    print(f"  existing:         {100 - frac:.1f}%")


if __name__ == "__main__":
    main()
