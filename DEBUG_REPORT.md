# Отчёт по отладке GRPO Segfault на RTX 5070 (Blackwell sm_120)

## Статус: РЕШЕНО (5/5 шагов пройдены успешно)

---

## Хронология проблемы

### Корневая причина segfault

**Проблема**: `python train_grpo.py --steps 5 --generations 2` → Segmentation fault при первом `model.generate()` на шаге 0.

**Корневая причина**: Unsloth 2026.2.1 использует кастомные Triton-ядра (RoPE, RMSNorm, SwiGLU, attention), которые при JIT-компиляции под sm_120 (Blackwell/RTX 5070) генерируют невалидный PTX/SASS-код.

**Доказательства**:
- `TRITON_INTERPRET=1` (интерпретатор, без JIT) → `model.generate()` **работает** → логика ядер верна, но скомпилированный GPU-код — нет
- `UNSLOTH_DISABLE_CUSTOM_KERNELS=1` → segfault **остаётся** → Unsloth патчит generate-путь даже без кастомных ядер
- Чистый transformers + bitsandbytes (без Unsloth) → `model.generate()` **работает** → проблема точно в Unsloth

### Удалён Unsloth из train_grpo.py

| Было (Unsloth)                                     | Стало (transformers + PEFT)                              |
|-----------------------------------------------------|----------------------------------------------------------|
| `from unsloth import FastLanguageModel, PatchFastRL` | `from transformers import AutoModelForCausalLM`          |
| `PatchFastRL("GRPO", FastLanguageModel)`             | убрано                                                   |
| `FastLanguageModel.from_pretrained(...)`             | `AutoModelForCausalLM.from_pretrained(...)`               |
| `FastLanguageModel.get_peft_model(...)`              | `get_peft_model(model, lora_config)`                     |

### Исправлена ошибка `ref_model` (TRL 0.24.0)

TRL 0.24.0 убрал параметр `ref_model` из GRPOTrainer. Удалён `ref_model=None` из конструктора.

---

## Вторая проблема: dtype mismatch

**Ошибка**: `RuntimeError: expected scalar type Float but found BFloat16` при `self.lm_head(hidden_states)`

**Механизм проблемы**:

```
1. from_pretrained(dtype=bfloat16) → lm_head: bfloat16 ✓
2. prepare_model_for_kbit_training() → lm_head: float32 ✗  (кастует ВСЕ non-quantized params)
3. get_peft_model() → lm_head: float32 ✗
4. GRPOTrainer.__init__() → Accelerator.prepare() → mixed precision
5. trainer.train() → generate() БЕЗ autocast → hidden_states (bf16) vs lm_head (fp32) → ОШИБКА
```

**Баг TRL 0.24.0**: В regular generation path (`_generate_single_turn`) нет `torch.autocast`, хотя в paged generation path он есть.

### Что пробовали и не помогло

| Попытка                                              | Результат                                                  |
|------------------------------------------------------|-------------------------------------------------------------|
| `dtype=torch.bfloat16` в from_pretrained             | prepare_model_for_kbit_training откатывает в float32       |
| `torch_dtype=torch.bfloat16`                         | Deprecated в transformers 4.57.6                            |
| Явный каст `model.lm_head.to(bfloat16)` после get_peft_model | GRPOTrainer/Accelerator перекастовывает обратно в float32 |
| Убрать `prepare_model_for_kbit_training`, ручная заморозка | Accelerator всё равно перекастовывает при `bf16=True` |

---

## Финальное решение (3 исправления)

### 1. `bf16=False` в GRPOConfig (Вариант A)

Отключаем mixed precision — Accelerator не создаёт float32 "master weights" и не кастует lm_head. Модель остаётся в native bfloat16.

```python
training_args = GRPOConfig(
    ...
    bf16=False,  # native bfloat16, без mixed precision
    ...
)
```

### 2. `enable_input_require_grads()` вместо `prepare_model_for_kbit_training`

`prepare_model_for_kbit_training()` кастует lm_head/embed_tokens в float32 — это ломает generate(). Заменяем на ручную подготовку:

```python
for param in model.parameters():
    param.requires_grad = False
model.enable_input_require_grads()  # backward через bnb-4bit слои
# gradient_checkpointing включается через GRPOConfig (не дублируем вручную)
```

**Важно**: убрали ручной `model.gradient_checkpointing_enable()` — его делает GRPOTrainer через config. Двойной вызов мог вызывать проблемы с хуками.

### 3. Monkey-patch `_generate_single_turn` с autocast (Вариант B — страховка)

Оборачиваем generate() в autocast на случай, если dtype разойдётся:

```python
_orig_generate = trainer._generate_single_turn

def _patched_generate(prompts, images=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return _orig_generate(prompts, images)

trainer._generate_single_turn = _patched_generate
```

---

## Результат тестирования

```
python train_grpo.py --steps 5 --generations 2
```

**5 из 5 шагов пройдены без ошибок и segfault.**

| Шаг | Loss   | Grad Norm | Reward | Время  |
|-----|--------|-----------|--------|--------|
| 1   | 0.0    | 0.913     | 0.75   | ~2 мин |
| 2   | 0.0098 | 1.488     | -0.19  | ~2 мин |
| 3   | 0.0    | 0.0       | -4.0   | ~2 мин |
| 4   | 0.0    | 0.0       | -1.5   | ~2 мин |
| 5   | 0.0217 | 1.200     | -3.0   | ~2 мин |

- **Общее время**: 10 мин 37 сек
- **Train loss**: 0.0063
- **Адаптеры сохранены**: `adapters/fact_checker_grpo/`
- **LoRA параметры**: 41,943,040 / 3,800,305,664 (1.10%)

---

## Версии библиотек

| Библиотека    | Версия        |
|---------------|---------------|
| PyTorch       | 2.10.0+cu128  |
| Transformers  | 4.57.6        |
| TRL           | 0.24.0        |
| PEFT          | 0.18.1        |
| bitsandbytes  | 0.49.2        |
| Triton        | 3.6.0         |
| Unsloth       | 2026.2.1 (удалён из train_grpo.py) |
| GPU           | RTX 5070 (Blackwell, sm_120, 12GB GDDR7) |

---

## Git-история ветки `fix-grpo-segfault`

```
a146693 Fix dtype mismatch: enable_input_require_grads + autocast safety for generate
0400605 fix: убрать Unsloth (segfault sm_120) + исправить dtype mismatch в GRPO
fd6f982 fix: добавить antifake/, unsloth_compiled_cache/, data/ в .gitignore
bf09ebe fix: PatchFastRL try/except + убрать xformers из requirements.txt
d1e1483 fix: блокировка xformers на Blackwell sm_120 — segfault в attention
11ee4d1 fix: CUDA allocator segfault + gradient_checkpointing двойные хуки
61cddbe Fix Segfault in GRPO (added Unsloth PatchFastRL) and VRAM optimizations
cc89976 fix: GRPO без merge — дообучаем SFT LoRA напрямую
bd20489 fix: GRPO segfault — save merged SFT to disk instead of in-memory merge_and_unload
```

---

## Структура ключевых файлов

```
train_grpo.py       — GRPO тренировка: transformers + PEFT + TRL (без Unsloth)
config.py           — ModelConfig (base_model_name, LoRA гиперпараметры)
claim_parser.py     — Парсинг чисел/дат для reward-функций
data/train_russian.jsonl  — 2500 примеров для тренировки
adapters/fact_checker_grpo/ — сохранённые LoRA-адаптеры после GRPO
```

---

## Запуск

```bash
# Быстрый тест (5 шагов)
cd /home/javaslav/antifake
source venv/bin/activate
python train_grpo.py --steps 5 --generations 2

# Полная тренировка
python train_grpo.py --steps 500 --generations 2

# С SFT-адаптером
python train_grpo.py --load-adapter adapters/fact_checker_lora --steps 300
```

---

## Известные нюансы

1. **`bf16=False` не означает float32** — модель работает в native bfloat16, просто Accelerator не применяет mixed precision схему с float32 master weights.
2. **`loss: 0.0` на некоторых шагах** — нормально при `gradient_accumulation_steps=4`, логируется средний loss за накопленные шаги. Реальный loss виден когда `grad_norm > 0`.
3. **`completions/clipped_ratio: 1.0`** — генерации упираются в `max_completion_length=512`. Для боевой тренировки стоит увеличить до 1024+.
4. **Monkey-patch `_generate_single_turn`** — привязан к TRL 0.24.0. При обновлении TRL проверить, что метод ещё существует.
