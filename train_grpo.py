"""
GRPO (Group Relative Policy Optimization) — обучение модели рассуждать.

Модель генерирует N вариантов ответа, каждый оценивается reward-функциями,
и модель учится генерировать ответы с высоким reward.

Трёхэтапный план:
  1. SFT на reasoning данных (train.py --resume)
  2. GRPO поверх SFT (этот скрипт)

Требования: ~12GB VRAM (RTX 5070), Unsloth + TRL >= 0.7.4.

Использование:
    python train_grpo.py
    python train_grpo.py --dataset data/train_russian.jsonl --steps 300 --generations 2
    python train_grpo.py --load-adapter adapters/fact_checker_lora
"""

import argparse
import gc
import json
import os
import platform
import re

if platform.system() == "Windows":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch

if platform.system() == "Windows":
    try:
        torch._dynamo.config.disable = True
    except AttributeError:
        pass

from claim_parser import extract_numbers, extract_dates
from config import ModelConfig, PROJECT_ROOT


SYSTEM_PROMPT = """\
Ты — эксперт по проверке фактов. Проанализируй утверждение и источники.

ФОРМАТ ОТВЕТА (строго!):
<reasoning>
Шаг 1: Для каждого источника — описывает ли он ТО ЖЕ событие?
Шаг 2: Из релевантных источников выпиши конкретные цитаты.
Шаг 3: Числовая проверка — совпадают ли цифры?
Шаг 4: Тройная самопроверка + адвокат дьявола:
  А) Источник про ТО ЖЕ событие или только тему?
  Б) "Не найдено" ≠ "опровергнуто"?
  В) Аргументы ПРОТИВ моего вердикта?
  Г) Мой вердикт логичен?
</reasoning>
<answer>
ДОСТОВЕРНОСТЬ: [0-100]
ВЕРДИКТ: [ПРАВДА / ЛОЖЬ / НЕ ПОДТВЕРЖДЕНО]
УВЕРЕННОСТЬ: [0-100]
ОБОСНОВАНИЕ: [3-5 предложений с цитатами]
ИСТОЧНИКИ: [список через ;]
</answer>
"""


def _extract_contents(completions: list) -> list:
    """Извлечение текста из completions."""
    contents = []
    for completion in completions:
        if isinstance(completion, str):
            contents.append(completion)
        elif isinstance(completion, list) and len(completion) > 0:
            contents.append(completion[0].get("content", ""))
        else:
            contents.append("")
    return contents


# ===== REWARD-ФУНКЦИИ =====

def format_reward(completions: list, **kwargs) -> list:
    contents = _extract_contents(completions)
    scores = []
    for text in contents:
        score = 0.0

        has_r_open = "<reasoning>" in text
        has_r_close = "</reasoning>" in text
        has_a_open = "<answer>" in text
        has_a_close = "</answer>" in text

        if has_r_open and has_r_close:
            score += 0.5
        if has_a_open and has_a_close:
            score += 0.5

        if text.count("<reasoning>") > 1:
            score -= 0.5
        if text.count("<answer>") > 1:
            score -= 0.5

        scores.append(score)
    return scores


def reasoning_quality_reward(completions: list, **kwargs) -> list:
    contents = _extract_contents(completions)
    scores = []
    for text in contents:
        score = 0.0

        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
        )
        if not reasoning_match:
            scores.append(-1.0)
            continue

        reasoning = reasoning_match.group(1)
        word_count = len(reasoning.split())

        if 50 <= word_count <= 500:
            score += 0.5
        elif word_count < 20:
            score -= 0.5

        step_keywords = [
            ("шаг 1", "источник", "анализ", "идентификац"),
            ("шаг 2", "цитат", "доказательств", "факт"),
            ("шаг 3", "числ", "цифр", "проверк"),
            ("шаг 4", "самопроверк", "вывод", "адвокат"),
        ]
        for step_group in step_keywords:
            if any(kw in reasoning.lower() for kw in step_group):
                score += 0.3

        source_refs = re.findall(
            r"источник\s*\d|по данным|сообщает|согласно|цитата",
            reasoning.lower(),
        )
        score += min(len(source_refs) * 0.2, 1.0)

        devil_keywords = [
            "адвокат дьявола", "против моего вердикта",
            "что если", "а если", "может ли",
            "аргументы против", "контраргумент",
        ]
        if any(kw in reasoning.lower() for kw in devil_keywords):
            score += 0.5

        comparison_keywords = [
            "совпадает", "не совпадает", "расходится",
            "подтверждает", "противоречит",
            "то же событие", "другое событие",
        ]
        comparison_count = sum(
            1 for kw in comparison_keywords
            if kw in reasoning.lower()
        )
        score += min(comparison_count * 0.15, 0.6)

        template_phrases = [
            "множественные источники подтвердили",
            "стилистика нейтральна",
            "маркеры манипулятивного воздействия",
            "признаки фабрикации",
            "соответствует стандартам качественной журналистики",
            "мультиисточниковая проверка",
        ]
        for phrase in template_phrases:
            if phrase in reasoning.lower():
                score -= 0.3

        scores.append(max(-1.0, min(1.0, score)))
    return scores


def verdict_consistency_reward(completions: list, **kwargs) -> list:
    contents = _extract_contents(completions)
    scores = []
    for text in contents:
        score = 0.0

        answer_match = re.search(
            r"<answer>(.*?)</answer>", text, re.DOTALL
        )
        if not answer_match:
            scores.append(-1.0)
            continue

        answer = answer_match.group(1)

        score_match = re.search(r"ДОСТОВЕРНОСТЬ:\s*(\d+)", answer)
        verdict_match = re.search(r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", answer)

        if not score_match or not verdict_match:
            scores.append(-0.5)
            continue

        cred_score = int(score_match.group(1))
        verdict = verdict_match.group(1).upper().strip()

        if verdict == "ПРАВДА" and 70 <= cred_score <= 100:
            score += 1.0
        elif verdict == "ЛОЖЬ" and 0 <= cred_score <= 29:
            score += 1.0
        elif "НЕ ПОДТВЕРЖДЕНО" in verdict and 30 <= cred_score <= 69:
            score += 1.0
        else:
            score -= 1.0

        if "ОБОСНОВАНИЕ:" in answer:
            reasoning_text = answer.split("ОБОСНОВАНИЕ:")[1]
            if len(reasoning_text.split()) > 15:
                score += 0.5

        scores.append(score)
    return scores


def correctness_reward(completions: list, **kwargs) -> list:
    expected_verdicts = kwargs.get("expected_verdict", [])
    if not expected_verdicts:
        return [0.0] * len(completions)

    contents = _extract_contents(completions)

    if len(expected_verdicts) > 0 and len(contents) > len(expected_verdicts):
        num_gen = len(contents) // len(expected_verdicts)
        expanded_verdicts = []
        for v in expected_verdicts:
            expanded_verdicts.extend([v] * num_gen)
        expected_verdicts = expanded_verdicts

    scores = []
    for text, expected in zip(contents, expected_verdicts):
        answer_match = re.search(
            r"<answer>(.*?)</answer>", text, re.DOTALL
        )
        if not answer_match:
            scores.append(-1.0)
            continue

        verdict_match = re.search(
            r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", answer_match.group(1)
        )
        if not verdict_match:
            scores.append(-0.5)
            continue

        predicted = verdict_match.group(1).upper().strip()
        expected_upper = expected.upper().strip()

        if predicted == expected_upper:
            scores.append(1.0)
        elif "НЕ ПОДТВЕРЖДЕНО" in predicted:
            scores.append(-0.2)
        else:
            scores.append(-1.0)

    return scores


def devils_advocate_reward(completions: list, **kwargs) -> list:
    contents = _extract_contents(completions)
    scores = []
    for text in contents:
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
        )
        if not reasoning_match:
            scores.append(-0.5)
            continue

        reasoning = reasoning_match.group(1).lower()
        score = 0.0

        contra_keywords = [
            "адвокат дьявола", "против моего вердикта",
            "что если", "а если", "может ли",
            "контраргумент", "аргументы против",
            "может быть", "возможно ли",
            "не слишком ли", "с другой стороны",
        ]
        has_contra = sum(1 for kw in contra_keywords if kw in reasoning)

        if has_contra >= 2:
            score += 0.5
        elif has_contra == 1:
            score += 0.25

        rejection_keywords = [
            "нет —", "нет,", "маловероятно",
            "проверил", "проверено",
            "именно поэтому", "потому что",
        ]
        has_rejection = any(kw in reasoning for kw in rejection_keywords)
        if has_contra > 0 and has_rejection:
            score += 0.5

        if has_contra == 0:
            score -= 0.5

        scores.append(max(-1.0, min(1.0, score)))
    return scores


def numerical_accuracy_reward(completions: list, **kwargs) -> list:
    contents = _extract_contents(completions)
    
    # ИСПРАВЛЕНИЕ: TRL передает ключ как "prompt" (единственное число), а не "prompts"
    prompts = kwargs.get("prompt", kwargs.get("prompts", []))

    if len(prompts) > 0 and len(contents) > len(prompts):
        num_gen = len(contents) // len(prompts)
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_gen)
        prompts = expanded_prompts

    if not prompts:
        return [0.0] * len(contents)

    scores = []
    for text, prompt in zip(contents, prompts):
        score = 0.0

        if isinstance(prompt, list):
            prompt_text = " ".join(
                msg.get("content", "") for msg in prompt if isinstance(msg, dict)
            )
        else:
            prompt_text = str(prompt)

        claim_numbers = extract_numbers(prompt_text)
        claim_dates = extract_dates(prompt_text)

        if not claim_numbers and not claim_dates:
            scores.append(0.0)
            continue

        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
        )
        if not reasoning_match:
            scores.append(-0.5)
            continue

        reasoning = reasoning_match.group(1)
        reasoning_lower = reasoning.lower()

        if claim_numbers:
            mentioned_count = 0
            for num in claim_numbers:
                raw = num["raw"]
                if raw in reasoning or str(int(num["value"])) in reasoning:
                    mentioned_count += 1

            if mentioned_count > 0:
                score += min(mentioned_count * 0.4, 0.8)
            else:
                score -= 0.5

            comparison_patterns = [
                r"совпада\w*", r"не совпада\w*", r"расходи\w*",
                r"отличает\w*", r"соответств\w*", r"не соответств\w*",
                r"подтвержда\w*.*\d", r"\d.*процент\w*",
                r"больше|меньше|выше|ниже|превышает",
            ]
            has_comparison = any(
                re.search(pat, reasoning_lower) for pat in comparison_patterns
            )
            if has_comparison:
                score += 0.3

        if claim_dates:
            date_mentioned = False
            for date in claim_dates:
                raw = date["raw"]
                year = str(date.get("year", ""))
                if raw in reasoning or (year and year in reasoning):
                    date_mentioned = True
                    break

            if date_mentioned:
                score += 0.3
            else:
                score -= 0.3

            date_comparison = [
                r"в \d{4}.*а не в \d{4}",
                r"год[уае]?\s+(?:совпада|не совпада|расходи)",
                r"дат[аы]?\s+(?:совпада|не совпада|расходи|другая|верн|неверн)",
                r"(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*.*(?:а не|вместо|на самом деле)",
            ]
            has_date_comparison = any(
                re.search(pat, reasoning_lower) for pat in date_comparison
            )
            if has_date_comparison:
                score += 0.2

        scores.append(max(-1.0, min(1.0, score)))
    return scores


# ===== ПОДГОТОВКА ДАННЫХ =====

def prepare_grpo_dataset(jsonl_path: str):
    from datasets import Dataset

    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            example = json.loads(line)
            convos = example.get("conversations", [])
            if len(convos) < 2:
                continue

            human_msg = convos[0].get("value", "")
            gpt_msg = convos[1].get("value", "")

            verdict_match = re.search(r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", gpt_msg)
            expected = verdict_match.group(1).strip() if verdict_match else "НЕ ПОДТВЕРЖДЕНО"

            data.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": human_msg},
                ],
                "expected_verdict": expected,
            })

    print(f"Загружено {len(data)} примеров для GRPO")
    return Dataset.from_list(data)


# ===== ОБУЧЕНИЕ =====

def check_flash_attention():
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8:
        print(f"  Flash Attention 2 требует SM >= 80, текущая: SM {capability[0]}{capability[1]}")
        return False

    try:
        import flash_attn  # noqa: F401
        print(f"  Flash Attention 2 доступен (v{flash_attn.__version__})")
        return True
    except ImportError:
        print("  Flash Attention 2 не установлен (pip install flash-attn)")
        return False


def setup_cuda():
    if not torch.cuda.is_available():
        print("CUDA недоступна!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # expandable_segments:True вызывает Segfault на Linux/CUDA 12.8
    # при tight VRAM — cuMemAddressReserve падает вместо OOM.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:False,"
        "max_split_size_mb:256,"
        "garbage_collection_threshold:0.8"
    )

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ["XFORMERS_DISABLED"] = "1"

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def train_grpo(
    dataset_path: str = None,
    output_dir: str = None,
    max_steps: int = 500,
    num_generations: int = 2,
    load_adapter: str = None,
):
    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "data", "train_russian.jsonl")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_grpo")

    setup_cuda()

    # ИСПРАВЛЕНИЕ: КРИТИЧЕСКИ ВАЖНО импортировать PatchFastRL ДО импорта TRL
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    
    from trl import GRPOConfig, GRPOTrainer

    model_config = ModelConfig()

    max_prompt_len = 512
    max_completion_len = 512
    grpo_max_seq_length = max_prompt_len + max_completion_len  # 1024

    use_fa2 = check_flash_attention()

    if load_adapter is not None:
        adapter_path = load_adapter
        if not os.path.isabs(adapter_path):
            adapter_path = os.path.join(PROJECT_ROOT, adapter_path)

        if not os.path.exists(adapter_path):
            print(f"  ОШИБКА: директория {adapter_path} не существует!")
            return None

        print(f"\n[1/4] Загрузка SFT-адаптера из {adapter_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=grpo_max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=True,
            attn_implementation="flash_attention_2" if use_fa2 else None,
        )
        print("  SFT адаптер загружен, GRPO дообучит LoRA веса")
        print("[2/4] Используются существующие LoRA из SFT-адаптера")
    else:
        print("\n[1/4] Загрузка базовой модели...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config.base_model_name,
            max_seq_length=grpo_max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=True,
            attn_implementation="flash_attention_2" if use_fa2 else None,
        )
        print("  Инициализация с нуля (без SFT-адаптера)")

        print("[2/4] Настройка LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    # ИСПРАВЛЕНИЕ: Принудительно включаем градиенты для загруженного адаптера
    if not any(p.requires_grad for p in model.parameters()):
        print("  [ВНИМАНИЕ] Веса заморожены, включаем requires_grad для LoRA...")
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Обучаемых: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(f"[3/4] Загрузка датасета из {dataset_path}...")
    dataset = prepare_grpo_dataset(dataset_path)

    print(f"[4/4] Запуск GRPO (steps={max_steps}, generations={num_generations})...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        beta=0.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # gradient_checkpointing управляется Unsloth через PatchFastRL.
        # TRL с gradient_checkpointing=True ставит двойные хуки → segfault.
        gradient_checkpointing=False,
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=max_steps,
        save_steps=100,
        logging_steps=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.1,
        max_grad_norm=0.1,
        temperature=0.9,
        bf16=True,
        report_to="none",
        log_completions=True,
        use_vllm=False,
    )

    trainer = GRPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            reasoning_quality_reward,
            verdict_consistency_reward,
            correctness_reward,
            devils_advocate_reward,
            numerical_accuracy_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("\n" + "=" * 60)
    trainer.train()
    print("=" * 60)

    print(f"\nСохранение GRPO адаптеров в {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("GRPO обучение завершено!")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training — обучение модели рассуждать (RTX 5070)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Путь к данным с reasoning (по умолчанию: data/train_russian.jsonl)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Директория для сохранения (по умолчанию: adapters/fact_checker_grpo)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Количество шагов обучения (по умолчанию: 500)",
    )
    parser.add_argument(
        "--generations", type=int, default=2,
        help="Вариантов генерации на пример (по умолчанию: 2, для 12GB VRAM)",
    )
    parser.add_argument(
        "--load-adapter", type=str, default=None,
        help="Путь к SFT-адаптеру для инициализации LoRA весов (по умолчанию: с нуля)",
    )
    args = parser.parse_args()

    train_grpo(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.steps,
        num_generations=args.generations,
        load_adapter=args.load_adapter,
    )


if __name__ == "__main__":
    main()