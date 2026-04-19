# Night Run Overhaul (branch `perf/night-overhaul`)

## Диагноз текущего GRPO адаптера

`fact_checker_grpo/checkpoint-100` из `D:\afasdfasdf\antifake_weights`:

- `epoch = 0.04` — прошло **4% датасета** (100 шагов из нужных ~2500+)
- `correctness_reward/mean ≈ -0.5` на всех 100 шагах → модель **училась давать неверные вердикты**
- `completions/clipped_ratio = 0.25..1.0` → половина генераций обрезалась на 512 токенах
- `reasoning_quality_reward = 1.0` (максимум) через 5 шагов → **reward hacking** на ключевые слова
- GRPO запущен **без SFT init** (`grad_norm = 411, kl = 9.96` на step 1 = катастрофа)

Итог: текущий GRPO адаптер **хуже SFT**. Ночной прогон это докажет цифрами.

---

## Что изменено в этой ветке

### `train_grpo.py`

| Параметр | Было | Стало | Почему |
|---|---|---|---|
| `max_completion_len` | 512 | **1024** | Не обрезать до `</answer>` → убрать ложный correctness=-1 |
| `save_steps` | 10 | **25** | Чекпоинты каждые ~50 мин при max_len=1024 |
| `save_total_limit` | 5 | **20** | Держим все чекпоинты для выбора лучшего |
| `mask_truncated_completions` | True | **False** | Давать gradient signal с truncated, чтобы модель училась укладываться |
| `reward_weights[reasoning_quality]` | 0.1 | **0.0** | Keyword farming ("шаг 1", "источник") |
| `reward_weights[devils_advocate]` | 0.0 | 0.0 | Уже был 0 — оставлен |
| `reward_weights[verdict_consistency]` | 0.1 | **0.4** | Антигеймбл: score↔verdict coherence |
| `reward_weights` (full) | `[0.1, 0.1, 0.1, 3.0, 0.0, 0.2, 0.5, 0.3]` | `[0.2, 0.0, 0.4, 3.0, 0.0, 0.5, 0.7, 0.5]` | |
| `correctness_reward` для mismatch | −1.0 | **−0.6** (близкие) −0.6 (далёкие) +0.3 (adjacent class) | Smoother gradient, partial credit |
| `correctness_reward` для truncation | −1.0 | **−0.3** | Soft penalty, не уничтожать batch из-за усечения |
| default `--steps` | 500 | **300** | Реально влезает в 8-9 часов на RTX 5070 |
| default `--load-adapter` | None | **`adapters/fact_checker_lora`** | Без SFT init — kl=9.96 на step 1 |

### `model.py`

- **Убран `merge_and_unload()`** в `load_finetuned_model`: merge LoRA в 4-bit базу вызывает rounding errors (PEFT сам предупреждает). LoRA теперь overlay.
- **Гибкий `build_langchain_llm`**: параметры `sampling=False/True`, `temperature`, `top_p`. Раньше `do_sample=False` был захардкожен → Self-Consistency получал детерминистичные одинаковые выходы.

### `evaluate_universal.py`

- Добавлен флаг `--adapter PATH` (или `--adapter base`) — можно явно сравнивать разные адаптеры без перетасовки папок.

### Новое

- `scripts/sanity_check.py` — smoke test перед ночью (~4 мин): CUDA, adapter files, dataset, одна генерация. Если падает — не теряешь ночь.
- `scripts/night_run.sh` — 8-9 часовой pipeline: baseline eval → GRPO retrain → eval checkpoints → report.
- `scripts/summarize_night.py` — сводка результатов в markdown.

---

## Как запускать ночью

```bash
# 1. Проверка перед сном (4 мин)
cd ~/antifake
source venv/bin/activate
python scripts/sanity_check.py

# если READY → запускаем в nohup, чтобы ssh/VPN не убил:
nohup ./scripts/night_run.sh > night.out 2>&1 &
echo $! > night.pid
disown
```

Утром:

```bash
cat night_logs_*/final_report.md
```

Решение по результатам:

- **Best GRPO v2 > SFT на 2+ п.п.** → промоут лучшего чекпоинта в `adapters/fact_checker_grpo`
- **GRPO v2 ≈ SFT** → использовать SFT, GRPO отключить (переименовать папку)
- **GRPO v2 < SFT** → GRPO как подход не работает на этой задаче/данных. Фокус на pipeline fixes из `docs/PLAN.md` или augmentation training data

---

## Оценка времени

На RTX 5070 при max_len=1024, gen=2:

| Этап | Время |
|---|---|
| Baseline eval × 3 | ~1.5 ч |
| GRPO retrain 300 шагов | ~6-7 ч |
| Eval 12 чекпоинтов (50..300 + final) | ~2-3 ч |
| **Итого** | **~10-11 ч** |

Если timeout критичен — уменьши `--steps 200` в `night_run.sh` → суммарно ~7-8 ч.

---

## Что НЕ трогалось

- `pipeline.py` (V17 уже evidence-first, SC убран автором)
- `prompts.py`
- `search.py`, `nli_checker.py`, `wikidata.py`
- Адаптеры на диске `D:\afasdfasdf\antifake_weights` (только читались)

Эти изменения можно делать следующими ветками по результатам ночного прогона.
