"""
GRPO (Group Relative Policy Optimization) — обучение модели рассуждать.

Модель генерирует N вариантов ответа, каждый оценивается reward-функциями,
и модель учится генерировать ответы с высоким reward.

Трёхэтапный план:
  1. SFT на reasoning данных (train.py --resume)
  2. GRPO поверх SFT (этот скрипт)

Reward-функции (6 штук):
  1. format_reward — правильный XML формат
  2. reasoning_quality_reward — глубина и качество рассуждений
  3. verdict_consistency_reward — согласованность вердикта и score
  4. correctness_reward — правильный вердикт (если известен ground truth)
  5. devils_advocate_reward — наличие "адвоката дьявола" в рассуждениях
  6. numerical_accuracy_reward — точная работа с числами, датами, процентами

Требования: ~12GB VRAM (RTX 5070), Unsloth + TRL >= 0.7.4.

Использование:
    python train_grpo.py
    python train_grpo.py --dataset data/train_russian.jsonl --steps 300 --generations 2
"""

import argparse
import json
import os
import re
import torch

from claim_parser import extract_numbers, extract_dates
from config import ModelConfig, LoraConfig, PROJECT_ROOT


# ===== СИСТЕМА ПРОМПТОВ =====

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


# ===== ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ =====

def _extract_contents(completions: list) -> list:
    """Извлечение текста из completions.

    TRL GRPOTrainer передаёт completions в conversational формате:
    list[list[dict]] — каждый completion = [{"role": "assistant", "content": "..."}]
    Эта функция извлекает content из каждого completion.
    """
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
    """Reward за правильный формат ответа (XML теги).

    +1.0 за <reasoning>...</reasoning>
    +1.0 за <answer>...</answer>
    -0.5 за дублирующиеся теги (галлюцинация)
    Нормализовано к [-1, +1]
    """
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

        # Штраф за множественные теги
        if text.count("<reasoning>") > 1:
            score -= 0.5
        if text.count("<answer>") > 1:
            score -= 0.5

        scores.append(score)
    return scores


def reasoning_quality_reward(completions: list, **kwargs) -> list:
    """Reward за качество рассуждений.

    Оценивает: длину, наличие шагов, ссылки на источники, адвокат дьявола.
    Штрафует: шаблонные фразы из синтетических данных.
    Нормализовано к [-1, +1]
    """
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

        # 1. Длина (оптимум 50-500 слов)
        if 50 <= word_count <= 500:
            score += 0.5
        elif word_count < 20:
            score -= 0.5

        # 2. Наличие шагов анализа
        step_keywords = [
            ("шаг 1", "источник", "анализ", "идентификац"),
            ("шаг 2", "цитат", "доказательств", "факт"),
            ("шаг 3", "числ", "цифр", "проверк"),
            ("шаг 4", "самопроверк", "вывод", "адвокат"),
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

        # 4. Бонус за адвоката дьявола (аргументы ПРОТИВ)
        devil_keywords = [
            "адвокат дьявола", "против моего вердикта",
            "что если", "а если", "может ли",
            "аргументы против", "контраргумент",
        ]
        if any(kw in reasoning.lower() for kw in devil_keywords):
            score += 0.5

        # 5. Бонус за конкретные сравнения (не абстрактные)
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

        # 6. Штраф за шаблонные фразы (из синтетических данных)
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

        # Нормализация к [-1, +1]
        scores.append(max(-1.0, min(1.0, score)))
    return scores


def verdict_consistency_reward(completions: list, **kwargs) -> list:
    """Reward за консистентность вердикта и score.

    ПРАВДА → score 70-100
    ЛОЖЬ → score 0-29
    НЕ ПОДТВЕРЖДЕНО → score 30-69
    """
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

    +1.0 за точное совпадение вердикта
    -1.0 за неверный вердикт
    -0.2 за НЕ ПОДТВЕРЖДЕНО когда ответ известен (мягкий штраф)
    Нормализовано к [-1, +1]

    ВАЖНО: GRPOTrainer генерирует num_generations completions на КАЖДЫЙ промпт,
    но expected_verdict — одно значение на промпт. Реплицируем expected_verdict.
    """
    expected_verdicts = kwargs.get("expected_verdict", [])
    if not expected_verdicts:
        return [0.0] * len(completions)

    contents = _extract_contents(completions)

    # GRPOTrainer: len(completions) = batch_size * num_generations
    # expected_verdicts: len = batch_size
    # Нужно реплицировать каждый expected_verdict на num_generations
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
    """Reward за наличие «адвоката дьявола» — аргументов ПРОТИВ своего вердикта.

    Учим модель не просто давать ответ, а критически анализировать собственные выводы.

    +0.5 за наличие контраргументов в reasoning
    +0.5 за явное отклонение контраргументов с обоснованием
    -0.5 за reasoning без самокритики
    Нормализовано к [-1, +1]
    """
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

        # Наличие контраргументов
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

        # Отклонение контраргументов с обоснованием
        rejection_keywords = [
            "нет —", "нет,", "маловероятно",
            "проверил", "проверено",
            "именно поэтому", "потому что",
        ]
        has_rejection = any(kw in reasoning for kw in rejection_keywords)
        if has_contra > 0 and has_rejection:
            score += 0.5

        # Штраф за отсутствие самокритики
        if has_contra == 0:
            score -= 0.5

        scores.append(max(-1.0, min(1.0, score)))
    return scores


def numerical_accuracy_reward(completions: list, **kwargs) -> list:
    """Reward за точную работу с числами, датами и процентами.

    Извлекает числа/даты из промпта (утверждение), затем проверяет,
    что модель явно упоминает и сравнивает их в reasoning.

    +0.4 за каждое число из утверждения, упомянутое в reasoning (max +0.8)
    +0.3 за явное сравнение чисел ("совпадает"/"не совпадает"/"расходится")
    +0.3 за упоминание дат из утверждения в reasoning
    -0.5 если в утверждении есть числа, но reasoning их игнорирует
    -0.3 если в утверждении есть даты, но reasoning их игнорирует
    Нормализовано к [-1, +1]
    """
    contents = _extract_contents(completions)

    # Извлекаем промпты для получения чисел/дат из утверждений
    prompts = kwargs.get("prompts", [])

    # Реплицируем промпты на num_generations (как в correctness_reward)
    if len(prompts) > 0 and len(contents) > len(prompts):
        num_gen = len(contents) // len(prompts)
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_gen)
        prompts = expanded_prompts

    # Если промпты недоступны — нейтральный reward
    if not prompts:
        return [0.0] * len(contents)

    scores = []
    for text, prompt in zip(contents, prompts):
        score = 0.0

        # Извлекаем текст промпта
        if isinstance(prompt, list):
            prompt_text = " ".join(
                msg.get("content", "") for msg in prompt if isinstance(msg, dict)
            )
        else:
            prompt_text = str(prompt)

        # Извлекаем числа и даты из утверждения
        claim_numbers = extract_numbers(prompt_text)
        claim_dates = extract_dates(prompt_text)

        # Если нет чисел и дат — нейтральный reward
        if not claim_numbers and not claim_dates:
            scores.append(0.0)
            continue

        # Извлекаем reasoning
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
        )
        if not reasoning_match:
            # Нет reasoning при наличии чисел/дат = штраф
            scores.append(-0.5)
            continue

        reasoning = reasoning_match.group(1)
        reasoning_lower = reasoning.lower()

        # --- Проверка чисел ---
        if claim_numbers:
            mentioned_count = 0
            for num in claim_numbers:
                # Проверяем упоминание числа в reasoning (raw или value)
                raw = num["raw"]
                if raw in reasoning or str(int(num["value"])) in reasoning:
                    mentioned_count += 1

            if mentioned_count > 0:
                score += min(mentioned_count * 0.4, 0.8)
            else:
                score -= 0.5  # Числа есть, но reasoning их игнорирует

            # Бонус за явное сравнение чисел
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

        # --- Проверка дат ---
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
                score -= 0.3  # Даты есть, но reasoning их игнорирует

            # Бонус за сравнение дат
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
    num_generations: int = 2,
):
    """Запуск GRPO обучения для reasoning.

    Args:
        dataset_path: Путь к данным с reasoning (train_russian.jsonl)
        output_dir: Директория для сохранения адаптеров
        max_steps: Количество шагов обучения
        num_generations: Сколько вариантов генерировать на пример (4 для 12GB VRAM)
    """
    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "data", "train_russian.jsonl")
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
    has_sft_adapters = os.path.exists(sft_adapter_path)

    if has_sft_adapters:
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
    if has_sft_adapters:
        # SFT адаптеры уже содержат LoRA — продолжаем обучение тех же весов
        # НЕ вызываем get_peft_model, иначе Unsloth выдаст ошибку
        print("  LoRA уже загружены из SFT адаптеров, продолжаю обучение...")
        FastLanguageModel.for_training(model)
    else:
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

    # Убеждаемся, что pad_token установлен
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO требует left-padding для генерации

    # Безопасные значения для RTX 5070 12GB (короче = меньше VRAM)
    max_prompt_len = 256
    max_completion_len = 512

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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
            devils_advocate_reward,
            numerical_accuracy_reward,
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
    args = parser.parse_args()

    train_grpo(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.steps,
        num_generations=args.generations,
    )


if __name__ == "__main__":
    main()
