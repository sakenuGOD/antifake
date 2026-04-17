"""Sanity check before night training: adapter loads, model generates, dataset is valid.

Run BEFORE night_run.sh to catch "ночь потеряна на падении на 3 шаге"-style disasters.

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --adapter adapters/fact_checker_lora
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_cuda():
    try:
        import torch
    except ImportError:
        print("[FAIL] torch не установлен")
        return False
    if not torch.cuda.is_available():
        print("[FAIL] CUDA недоступна — GRPO обучение невозможно")
        return False
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK]  GPU: {name} ({vram:.1f} GB)")
    if vram < 11:
        print(f"[WARN] VRAM < 11GB, max_completion_length=1024 + gen=2 может привести к OOM")
    return True


def check_adapter(adapter_path):
    if adapter_path is None:
        print("[OK]  Adapter: [base model]")
        return True
    if not os.path.isabs(adapter_path):
        abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 adapter_path)
    else:
        abs_path = adapter_path
    if not os.path.exists(abs_path):
        print(f"[FAIL] Adapter не найден: {abs_path}")
        return False
    cfg = os.path.join(abs_path, "adapter_config.json")
    sft = os.path.join(abs_path, "adapter_model.safetensors")
    if not os.path.exists(cfg):
        print(f"[FAIL] adapter_config.json отсутствует в {abs_path}")
        return False
    if not os.path.exists(sft):
        print(f"[FAIL] adapter_model.safetensors отсутствует в {abs_path}")
        return False
    with open(cfg) as f:
        data = json.load(f)
    print(f"[OK]  Adapter: {abs_path}")
    print(f"       r={data.get('r')}, alpha={data.get('lora_alpha')}, "
          f"targets={len(data.get('target_modules', []))}")
    return True


def check_dataset():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "train_russian.jsonl")
    if not os.path.exists(path):
        print(f"[FAIL] Dataset отсутствует: {path}")
        return False
    count = 0
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if len(ex.get("conversations", [])) < 2:
                    bad += 1
                count += 1
            except json.JSONDecodeError:
                bad += 1
                count += 1
    print(f"[{'OK' if bad == 0 else 'WARN'}]  Dataset: {count} примеров, "
          f"{bad} битых (conv < 2)")
    return True


def check_eval_dataset():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "eval_dataset.jsonl")
    if not os.path.exists(path):
        print(f"[WARN] Eval dataset отсутствует: {path} — eval часть night_run.sh не работает")
        return False
    count = sum(1 for line in open(path, encoding="utf-8") if line.strip())
    print(f"[OK]  Eval dataset: {count} claims")
    return True


def check_generation(adapter_path):
    """Minimal end-to-end: load model + adapter, run 1 generation. 3-5 min on RTX 5070."""
    print("\n--- Проверка загрузки модели и генерации (~3 мин) ---")
    try:
        from model import load_base_model, load_finetuned_model
    except Exception as e:
        print(f"[FAIL] import model: {e}")
        return False

    t0 = time.time()
    try:
        if adapter_path and adapter_path.lower() not in ("none", "base"):
            model, tokenizer = load_finetuned_model(adapter_path)
        else:
            model, tokenizer = load_base_model()
    except Exception as e:
        print(f"[FAIL] Загрузка модели: {type(e).__name__}: {e}")
        return False
    print(f"[OK]  Модель загружена за {time.time()-t0:.1f}s")

    try:
        import torch
        inputs = tokenizer("[INST]Скажи одно слово: тест.[/INST]",
                             return_tensors="pt").to(model.device)
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        txt = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"[OK]  Генерация за {time.time()-t1:.1f}s: {txt[:80]!r}")
    except Exception as e:
        print(f"[FAIL] Генерация: {type(e).__name__}: {e}")
        return False

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[OK]  Peak VRAM: {peak:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="adapters/fact_checker_lora",
                         help="Adapter path or 'base'/'none' (default: SFT)")
    parser.add_argument("--skip-generation", action="store_true",
                         help="Skip model load + generation (fast static checks only)")
    args = parser.parse_args()

    print("=" * 60)
    print("ANTIFAKE SANITY CHECK (ночной prep)")
    print("=" * 60)

    checks = []
    checks.append(check_cuda())
    checks.append(check_adapter(args.adapter))
    checks.append(check_dataset())
    check_eval_dataset()  # advisory only

    if not args.skip_generation:
        checks.append(check_generation(args.adapter))

    print("\n" + "=" * 60)
    if all(checks):
        print("READY. Можно запускать scripts/night_run.sh")
        sys.exit(0)
    else:
        print("FAIL. Почини ошибки выше перед ночным забегом.")
        sys.exit(1)


if __name__ == "__main__":
    main()
