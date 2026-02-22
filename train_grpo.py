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
    python train_grpo.py --dataset data/train_reasoning.jsonl --steps 500 --generations 4
"""

import argparse
import json
import os
import re
import torch

from config import ModelConfig, LoraConfig, PROJECT_ROOT


# ===== СИСТЕМА ПРОМПТОВ =====

SYSTEM_PROMPT = """\
Ты — эксперт по проверке фактов. Проанализируй утверждение и источники.

ФОРМАТ ОТВЕТА (строго!):
<reasoning>
Шаг 1: Для каждого источника — описывает ли он ТО ЖЕ событие?
Шаг 2: Из релевантных источников выпиши конкретные цитаты.
Шаг 3: Числовая проверка — совпадают ли цифры?
Шаг 4: Тройная самопроверка.
</reasoning>
<answer>
ДОСТОВЕРНОСТЬ: [0-100]
ВЕРДИКТ: [ПРАВДА / ЛОЖЬ / НЕ ПОДТВЕРЖДЕНО]
УВЕРЕННОСТЬ: [0-100]
ОБОСНОВАНИЕ: [3-5 предложений с цитатами]
ИСТОЧНИКИ: [список через ;]
</answer>
"""


# ===== REWARD-ФУНКЦИИ =====

def format_reward(completions: list, **kwargs) -> list:
    """Reward за правильный формат ответа (XML теги).

    +1.0 за <reasoning>...</reasoning>
    +1.0 за <answer>...</answer>
    -0.5 за дублирующиеся теги (галлюцинация)
    """
    scores = []
    for completion in completions:
        score = 0.0

        has_r_open = "<reasoning>" in completion
        has_r_close = "</reasoning>" in completion
        has_a_open = "<answer>" in completion
        has_a_close = "</answer>" in completion

        if has_r_open and has_r_close:
            score += 1.0
        if has_a_open and has_a_close:
            score += 1.0

        # Штраф за множественные теги
        if completion.count("<reasoning>") > 1:
            score -= 0.5
        if completion.count("<answer>") > 1:
            score -= 0.5

        scores.append(score)
    return scores


def reasoning_quality_reward(completions: list, **kwargs) -> list:
    """Reward за качество рассуждений.

    Оценивает: длину, наличие шагов, ссылки на источники.
    Штрафует: шаблонные фразы из синтетических данных.
    """
    scores = []
    for completion in completions:
        score = 0.0

        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", completion, re.DOTALL
        )
        if not reasoning_match:
            scores.append(-1.0)
            continue

        reasoning = reasoning_match.group(1)
        word_count = len(reasoning.split())

        # 1. Длина (оптимум 50-500 слов)
        if 50 <= word_count <= 500:
            score += 0.5
        elif word_count < 20:
            score -= 0.5

        # 2. Наличие шагов анализа
        step_keywords = [
            ("шаг 1", "источник", "анализ"),
            ("шаг 2", "цитат", "доказательств"),
            ("шаг 3", "числ", "цифр", "проверк"),
            ("шаг 4", "самопроверк", "вывод"),
        ]
        for step_group in step_keywords:
            if any(kw in reasoning.lower() for kw in step_group):
                score += 0.3

        # 3. Конкретные ссылки на источники
        source_refs = re.findall(
            r"источник\s*\d|по данным|сообщает|согласно|цитата",
            reasoning.lower(),
        )
        score += min(len(source_refs) * 0.2, 1.0)

        # 4. Штраф за шаблонные фразы (из синтетических данных)
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

        scores.append(score)
    return scores


def verdict_consistency_reward(completions: list, **kwargs) -> list:
    """Reward за консистентность вердикта и score.

    ПРАВДА → score 70-100
    ЛОЖЬ → score 0-29
    НЕ ПОДТВЕРЖДЕНО → score 30-69
    """
    scores = []
    for completion in completions:
        score = 0.0

        answer_match = re.search(
            r"<answer>(.*?)</answer>", completion, re.DOTALL
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

        # Проверка консистентности
        if verdict == "ПРАВДА" and 70 <= cred_score <= 100:
            score += 1.0
        elif verdict == "ЛОЖЬ" and 0 <= cred_score <= 29:
            score += 1.0
        elif "НЕ ПОДТВЕРЖДЕНО" in verdict and 30 <= cred_score <= 69:
            score += 1.0
        else:
            score -= 1.0  # Несоответствие

        # Бонус за наличие обоснования
        if "ОБОСНОВАНИЕ:" in answer:
            reasoning_text = answer.split("ОБОСНОВАНИЕ:")[1]
            if len(reasoning_text.split()) > 15:
                score += 0.5

        scores.append(score)
    return scores


def correctness_reward(completions: list, **kwargs) -> list:
    """Reward за правильный вердикт (если known ground truth).

    +2.0 за точное совпадение вердикта
    -1.5 за неверный вердикт
    -0.2 за НЕ ПОДТВЕРЖДЕНО когда ответ известен (мягкий штраф)
    """
    expected_verdicts = kwargs.get("expected_verdict", [])
    if not expected_verdicts:
        return [0.0] * len(completions)

    scores = []
    for completion, expected in zip(completions, expected_verdicts):
        answer_match = re.search(
            r"<answer>(.*?)</answer>", completion, re.DOTALL
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
            scores.append(2.0)
        elif "НЕ ПОДТВЕРЖДЕНО" in predicted:
            scores.append(-0.2)  # Мягкий штраф за осторожность
        else:
            scores.append(-1.5)

    return scores


# ===== ПОДГОТОВКА ДАННЫХ =====

def prepare_grpo_dataset(jsonl_path: str):
    """Подготовка датасета для GRPO."""
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

            # Извлекаем ожидаемый вердикт
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

def setup_cuda():
    """CUDA оптимизации."""
    if not torch.cuda.is_available():
        print("CUDA недоступна!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    os.environ["XFORMERS_DISABLED"] = "1"
    torch.cuda.empty_cache()


def train_grpo(
    dataset_path: str = None,
    output_dir: str = None,
    max_steps: int = 500,
    num_generations: int = 4,
):
    """Запуск GRPO обучения для reasoning.

    Args:
        dataset_path: Путь к данным с reasoning (train_reasoning.jsonl)
        output_dir: Директория для сохранения адаптеров
        max_steps: Количество шагов обучения
        num_generations: Сколько вариантов генерировать на пример (4 для 12GB VRAM)
    """
    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "data", "train_reasoning.jsonl")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_grpo")

    setup_cuda()

    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    model_config = ModelConfig()
    lora_config = LoraConfig()

    # 1. Загрузка модели
    print("\n[1/4] Загрузка модели...")

    # Попытка загрузить SFT адаптеры (если уже обучены)
    sft_adapter_path = os.path.join(PROJECT_ROOT, "adapters", "fact_checker_lora")
    if os.path.exists(sft_adapter_path):
        print(f"  Загружаю SFT адаптеры из {sft_adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=sft_adapter_path,
            max_seq_length=model_config.max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=True,
        )
    else:
        print(f"  SFT адаптеры не найдены, загружаю base модель")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config.base_model_name,
            max_seq_length=model_config.max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=True,
        )

    # 2. LoRA адаптеры
    print("[2/4] Настройка LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Обучаемых: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # 3. Датасет
    print(f"[3/4] Загрузка датасета из {dataset_path}...")
    dataset = prepare_grpo_dataset(dataset_path)

    # 4. GRPO Training
    print(f"[4/4] Запуск GRPO (steps={max_steps}, generations={num_generations})...")

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=num_generations,
        max_prompt_length=1024,
        max_completion_length=1024,
        max_steps=max_steps,
        save_steps=100,
        logging_steps=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.1,
        temperature=0.9,
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,
            reasoning_quality_reward,
            verdict_consistency_reward,
            correctness_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("\n" + "=" * 60)
    trainer.train()
    print("=" * 60)

    # Сохранение
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
        help="Путь к данным с reasoning (по умолчанию: data/train_reasoning.jsonl)",
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
        "--generations", type=int, default=4,
        help="Вариантов генерации на пример (по умолчанию: 4, уменьши если OOM)",
    )
    args = parser.parse_args()

    train_grpo(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.steps,
        num_generations=args.generations,
    )


if __name__ == "__main__":
    main()
