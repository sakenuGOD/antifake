"""
GRPO (Group Relative Policy Optimization) — обучение модели рассуждать.

Модель генерирует N вариантов ответа, каждый оценивается reward-функциями,
и модель учится генерировать ответы с высоким reward.

Требования: ~12GB VRAM (RTX 5070), transformers + PEFT + TRL.

Использование:
    python train_grpo.py
    python train_grpo.py --dataset data/train_russian.jsonl --steps 300 --generations 2
    python train_grpo.py --load-adapter adapters/fact_checker_lora
"""

import argparse
import gc
import json
import os
import re
import sys
import time

# ============================================================
# Blackwell (sm_120) fix: Unsloth Triton kernels segfault
# на RTX 5070 при первом generate(). Используем чистый
# transformers + PEFT + TRL без Unsloth.
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from claim_parser import extract_numbers, extract_dates
from config import ModelConfig, PROJECT_ROOT


SYSTEM_PROMPT = """\
Ты — эксперт по проверке фактов. Проанализируй утверждение и источники.

ФОРМАТ ОТВЕТА (строго!):
<reasoning>
Шаг 1 — ИДЕНТИФИКАЦИЯ: Какое событие? Кто, что, где, когда, какие числа?
Для каждого источника: описывает ли он ТО ЖЕ событие с ТЕМИ ЖЕ деталями?
Шаг 2 — ФАКТЫ: Из релевантных источников выпиши цитаты.
Совпадающие: "[Источник] сообщает: ..."
Противоречащие: "[Источник] утверждает обратное: ..."
Шаг 3 — ВЫВОД: NLI и числовые данные проверены автоматически. Учитывай их как факт.
Если NLI contradiction — объясни почему утверждение ложно.
Если числа расходятся — это расхождение с реальностью.
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


# Adjacency for partial credit — avoids hard -1 when verdict is "close".
_VERDICT_ADJACENT = {
    "ПРАВДА": {"ЧАСТИЧНО ПОДТВЕРЖДЕНО", "ЧАСТИЧНО ПРАВДА"},
    "ЛОЖЬ": {"МАНИПУЛЯЦИЯ", "ПОЛУПРАВДА"},
    "МАНИПУЛЯЦИЯ": {"ПОЛУПРАВДА", "ЧАСТИЧНО ПОДТВЕРЖДЕНО", "ЧАСТИЧНО ПРАВДА", "ЛОЖЬ"},
    "ПОЛУПРАВДА": {"МАНИПУЛЯЦИЯ", "ЧАСТИЧНО ПОДТВЕРЖДЕНО"},
    "ЧАСТИЧНО ПОДТВЕРЖДЕНО": {"МАНИПУЛЯЦИЯ", "ПОЛУПРАВДА", "ПРАВДА"},
    "ЧАСТИЧНО ПРАВДА": {"МАНИПУЛЯЦИЯ", "ПОЛУПРАВДА", "ПРАВДА"},
    "НЕ ПОДТВЕРЖДЕНО": {"МАНИПУЛЯЦИЯ", "ПОЛУПРАВДА"},
}


def _verdicts_adjacent(predicted: str, expected: str) -> bool:
    return predicted in _VERDICT_ADJACENT.get(expected, set())


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
        # Softer penalty for truncation so length-discipline can be learned
        # without destroying the whole gradient when mask_truncated=False.
        if not answer_match:
            scores.append(-0.3)
            continue

        verdict_match = re.search(
            r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", answer_match.group(1)
        )
        if not verdict_match:
            scores.append(-0.3)
            continue

        predicted = verdict_match.group(1).upper().strip()
        expected_upper = expected.upper().strip()

        if predicted == expected_upper:
            scores.append(1.0)
        elif _verdicts_adjacent(predicted, expected_upper):
            scores.append(0.3)
        elif "НЕ ПОДТВЕРЖДЕНО" in predicted:
            scores.append(0.0)
        else:
            scores.append(-0.6)  # was -1.0; smoother gradient

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


def scam_detection_reward(completions: list, **kwargs) -> list:
    """Награда за правильное определение мошеннических схем.
    
    Если в промпте присутствуют скам-паттерны:
    - Модель поставила ЛОЖЬ → +1.0 (правильно)
    - Модель поставила НЕ ПОДТВЕРЖДЕНО → -0.5 (недостаточно решительно)
    - Модель поставила ПРАВДА → -2.0 (опасная ошибка — приняла скам как факт)
    """
    scam_patterns = [
        'безопасный счёт', 'безопасного счёта',
        'перевести деньги', 'перевод денег',
        'служба безопасности банка', 'банк звонит',
        'ваша карта заблокирована', 'код из смс',
        'назовите код', 'пин-код',
        'следователь звонит', 'фсб звонит',
        'гарантированная доходность', '100% доходность',
        'финансовая пирамида',
    ]
    
    contents = _extract_contents(completions)
    prompts = kwargs.get("prompt", kwargs.get("prompts", []))
    
    if len(prompts) > 0 and len(contents) > len(prompts):
        num_gen = len(contents) // len(prompts)
        prompts = [p for p in prompts for _ in range(num_gen)]
    
    scores = []
    for text, prompt in zip(contents, prompts if prompts else [''] * len(contents)):
        prompt_text = " ".join(
            msg.get("content", "") for msg in prompt if isinstance(msg, dict)
        ) if isinstance(prompt, list) else str(prompt)
        
        prompt_lower = prompt_text.lower()
        has_scam = any(p in prompt_lower for p in scam_patterns)
        
        if not has_scam:
            scores.append(0.0)
            continue
        
        # Есть скам-паттерн — проверяем вердикт
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not answer_match:
            scores.append(-0.5)
            continue
        
        verdict_match = re.search(r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", answer_match.group(1))
        if not verdict_match:
            scores.append(-0.5)
            continue
        
        predicted = verdict_match.group(1).upper().strip()
        if predicted == "ЛОЖЬ":
            scores.append(1.0)   # Правильно! Скам = ЛОЖЬ
        elif "НЕ ПОДТВЕРЖДЕНО" in predicted:
            scores.append(-0.5)  # Слишком осторожно
        else:
            scores.append(-2.0)  # Опасная ошибка
    
    return scores


def location_date_accuracy_reward(completions: list, **kwargs) -> list:
    """Награда за точную проверку дат и локаций.
    
    Если в reasoning есть явное сравнение год/место утверждения с источниками → reward.
    Если вердикт ЛОЖЬ когда источники говорят другую локацию/год → reward.
    """
    contents = _extract_contents(completions)
    
    location_comparison_patterns = [
        r"в токио.*?в париж|в париж.*?в токио",
        r"в катар.*?в япони|в япони.*?в катар",
        r"в \d{4}.*?а не в \d{4}|не в \d{4}.*?в \d{4}",
        r"город.*?другой|другой.*?город",
        r"прошл.*?в другом|другой стран",
        r"не \d{4} год|\d{4}.*?неверн|неверн.*?год",
        r"принимал.*?\d{4}|\d{4}.*?принимал",
        r"вышла.*?\d{4}|\d{4}.*?вышла",
        r"запущена в \d{4}|\d{4}.*?запущена",
    ]
    
    scores = []
    for text in contents:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if not reasoning_match:
            scores.append(0.0)
            continue
        
        reasoning = reasoning_match.group(1).lower()
        score = 0.0
        
        # Проверяем наличие явного сравнения дат/мест
        comparison_count = sum(
            1 for pat in location_comparison_patterns
            if re.search(pat, reasoning)
        )
        score += min(comparison_count * 0.3, 0.9)
        
        # Бонус если reasoning содержит конкретные годы и сравнение
        years = re.findall(r'\b(19|20)\d{2}\b', reasoning)
        if len(years) >= 2:
            score += 0.3  # Упомянуто несколько лет — сравнение идёт
        
        # Проверяем что вердикт ЛОЖЬ при явном противоречии дат/мест
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match and comparison_count > 0:
            verdict_match = re.search(r"ВЕРДИКТ:\s*(.+?)(?:\n|$)", answer_match.group(1))
            if verdict_match and verdict_match.group(1).upper().strip() == "ЛОЖЬ":
                score += 0.5  # Правильный вердикт при найденном противоречии
        
        scores.append(max(-1.0, min(1.0, score)))
    
    return scores


def audit_reward(completions: list, **kwargs) -> list:
    """Reward за правильный СТАТУС каждого ПУНКТА в audit протоколе."""
    contents = _extract_contents(completions)
    rewards = []
    for text in contents:
        reward = 0.0
        # Парсим СТАТУС: из каждого ПУНКТА
        statuses = re.findall(r"СТАТУС:\s*(ПРАВДА|ЛОЖЬ|НЕ УВЕРЕНА)", text, re.IGNORECASE)
        ground_truth = kwargs.get("ground_truth", {}).get("sub_verdicts", [])

        for i, status in enumerate(statuses):
            if i < len(ground_truth):
                expected = ground_truth[i].get("status", "")
                status_upper = status.strip().upper()
                if expected and status_upper == expected.upper():
                    reward += 0.25  # Правильный статус
                elif status_upper != "НЕ УВЕРЕНА" and expected.upper() == "НЕ УВЕРЕНА":
                    reward -= 0.1  # Галлюцинация хуже незнания
        rewards.append(reward)
    return rewards


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


# ===== CALLBACKS =====

class TimeLimitCallback(TrainerCallback):
    """Остановка обучения по таймеру. Сохраняет чекпоинт перед выходом."""

    def __init__(self, max_seconds: int = 10800):
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        hours = self.max_seconds / 3600
        print(f"  [TimeLimitCallback] Лимит: {hours:.1f} ч ({self.max_seconds} сек)")

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            print(f"\n  [TimeLimitCallback] Лимит времени достигнут "
                  f"({elapsed/3600:.1f} ч). Сохраняю и завершаю...")
            control.should_save = True
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time and logs:
            elapsed = time.time() - self.start_time
            remaining = self.max_seconds - elapsed
            logs["time_elapsed_min"] = round(elapsed / 60, 1)
            logs["time_remaining_min"] = round(remaining / 60, 1)


# ===== ОБУЧЕНИЕ =====

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

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:False,"
        "max_split_size_mb:256,"
        "garbage_collection_threshold:0.8"
    )

    # SDPA backends: flash отключен (нет flash-attn), mem_efficient + math
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

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

    from trl import GRPOConfig, GRPOTrainer

    model_config = ModelConfig()
    base_model_name = model_config.base_model_name

    max_prompt_len = 512
    # 512 caused 50-75% of completions to truncate before </answer>, producing
    # spurious negative correctness_reward. 1024 gives room for full reasoning
    # with ~2x step time; tradeoff is worth it on 12GB at num_generations=2.
    max_completion_len = 1024

    if load_adapter is not None:
        adapter_path = load_adapter
        if not os.path.isabs(adapter_path):
            adapter_path = os.path.join(PROJECT_ROOT, adapter_path)

        if not os.path.exists(adapter_path):
            print(f"  ОШИБКА: директория {adapter_path} не существует!")
            return None

        print(f"\n[1/4] Загрузка базовой модели + SFT-адаптера из {adapter_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        # enable_input_require_grads ПЕРЕД загрузкой адаптера —
        # без этого gradient_checkpointing не получает grad-enabled inputs
        for param in model.parameters():
            param.requires_grad = False
        model.enable_input_require_grads()

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print("  SFT адаптер загружен")
        print("[2/4] Используются существующие LoRA из SFT-адаптера")
    else:
        print(f"\n[1/4] Загрузка базовой модели: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("  Инициализация с нуля (без SFT-адаптера)")

        print("[2/4] Настройка LoRA...")
        # Ручная подготовка вместо prepare_model_for_kbit_training
        # (которая кастует lm_head/embed_tokens в float32, ломая generate())
        for param in model.parameters():
            param.requires_grad = False
        model.enable_input_require_grads()  # нужен для backward через bnb-4bit
        # gradient_checkpointing включается через GRPOConfig (не дублируем)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # Включаем градиенты для LoRA слоёв
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
        # === Оптимизированные гиперпараметры ===
        learning_rate=1e-5,           # агрессивнее (SFT фундамент позволяет)
        beta=0.04,                    # KL penalty — не забыть SFT знания
        loss_type="dr_grpo",          # Dr. GRPO: token-efficient loss
        num_iterations=2,             # 2 gradient update на 1 генерацию = 2x обучения бесплатно
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=num_generations,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        max_steps=max_steps,
        save_steps=25,                # чекпоинт каждые ~50 мин (max_len=1024)
        save_total_limit=20,           # держим больше чекпоинтов для выбора лучшего
        logging_steps=1,
        warmup_ratio=0.03,            # короткий warmup (SFT модель уже прогрета)
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.05,
        max_grad_norm=0.3,            # менее жёсткий клиппинг
        temperature=1.0,              # разнообразие генераций для GRPO exploration
        bf16=False,                   # native bfloat16 (Accelerator dtype fix)
        # mask_truncated_completions=True masked learning signal from cut-off
        # outputs, leaving GRPO with an effective batch of 0-2 valid completions
        # per step. With soft truncation penalty in correctness_reward, we keep
        # gradient from all completions including truncated ones.
        mask_truncated_completions=False,
        report_to="none",
        log_completions=True,
        use_vllm=False,
    )

    # Safety-net: 10 часов (36000 сек) — только защита от зависания.
    # Реальный ограничитель — max_steps. Качество модели важнее скорости.
    time_limit_cb = TimeLimitCallback(max_seconds=36000)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward,                 # structural tags
            reasoning_quality_reward,      # DISABLED: keyword farming ("шаг 1", "источник")
            verdict_consistency_reward,    # score <-> verdict coherence
            correctness_reward,            # DOMINANT: matches ground-truth (with partial credit)
            devils_advocate_reward,        # DISABLED: duplicates reasoning_quality, reward-hackable
            numerical_accuracy_reward,     # numbers mentioned in reasoning
            scam_detection_reward,         # scam pattern detection
            location_date_accuracy_reward, # date/location explicit comparison
        ],
        # reasoning_quality + devils_advocate are set to 0 because logs show
        # they both saturate to 1.0 within 5-10 steps (surface keyword match)
        # while correctness stays at -0.5. This is textbook reward hacking.
        # Weights rebalanced so correctness dominates and non-gameable signals
        # (format, verdict_consistency, numerical, scam, location) contribute.
        reward_weights=[0.2, 0.0, 0.4, 3.0, 0.0, 0.5, 0.7, 0.5],
        args=training_args,
        train_dataset=dataset,
        callbacks=[time_limit_cb],
    )

    # Страховка: оборачиваем generate() в autocast для совместимости dtype
    # (bnb-4bit отдаёт hidden_states в bf16, lm_head может быть в другом dtype)
    # Wrap _generate_single_turn with bfloat16 autocast for TRL dtype safety.
    # Signature differs across TRL versions: 0.24 took (prompts, images=None),
    # 1.x takes (prompt_ids, images, multimodal_fields). We use *args/**kwargs
    # to stay forward/backward compatible across TRL releases.
    _orig_generate = trainer._generate_single_turn

    def _patched_generate(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return _orig_generate(*args, **kwargs)

    trainer._generate_single_turn = _patched_generate

    print("\n" + "=" * 60)
    train_result = trainer.train()
    print("=" * 60)

    # Финальные метрики
    metrics = train_result.metrics
    elapsed_min = metrics.get("train_runtime", 0) / 60
    steps_done = metrics.get("train_steps", max_steps)
    print(f"\n--- Итоги обучения ---")
    print(f"  Шагов: {steps_done}")
    print(f"  Время: {elapsed_min:.1f} мин")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
    if elapsed_min > 0 and steps_done > 0:
        print(f"  Скорость: {elapsed_min / steps_done:.1f} мин/шаг")

    print(f"\nСохранение GRPO адаптеров в {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Сохраняем метрики в JSON для анализа
    import json as _json
    metrics_path = os.path.join(output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        _json.dump(metrics, f, indent=2)
    print(f"  Метрики сохранены: {metrics_path}")

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
        "--steps", type=int, default=300,
        help="Количество шагов обучения (по умолчанию: 300, ~10 часов при max_len=1024, gen=2)",
    )
    parser.add_argument(
        "--generations", type=int, default=2,
        help="Вариантов генерации на пример (по умолчанию: 2, для 12GB VRAM; 4 = OOM при max_len=1024)",
    )
    parser.add_argument(
        "--load-adapter", type=str, default="adapters/fact_checker_lora",
        help="SFT-адаптер как стартовая точка (по умолчанию: adapters/fact_checker_lora). "
             "Передайте 'none' для обучения с нуля (не рекомендуется — grad_norm>400 на step 1).",
    )
    args = parser.parse_args()

    # Auto-disable load-adapter if user passes 'none' or path does not exist
    if args.load_adapter and args.load_adapter.lower() == "none":
        args.load_adapter = None
    elif args.load_adapter and not os.path.isabs(args.load_adapter):
        abs_path = os.path.join(PROJECT_ROOT, args.load_adapter)
        if not os.path.exists(abs_path):
            print(f"[WARN] SFT adapter not found at {abs_path} — training from scratch")
            args.load_adapter = None

    train_grpo(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.steps,
        num_generations=args.generations,
        load_adapter=args.load_adapter,
    )


if __name__ == "__main__":
    main()
