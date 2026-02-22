#!/bin/bash
# Полный цикл обучения за ~1.5 часа (RTX 5070, 12GB VRAM)
# Только русский язык, оптимальный датасет 2500 примеров
# Запуск: bash run_training.sh

set -e  # Остановка при ошибке

echo "=========================================="
echo "  FACT-CHECKER: Обучение (< 1.5 часа)"
echo "  Русский язык | SFT + GRPO"
echo "=========================================="
START=$(date +%s)

# 1. Русский датасет с reasoning (2500 примеров, ~1 сек)
echo ""
echo "[1/3] Генерация русского датасета (2500: 35% правда + 35% ложь + 30% не подтверждено)..."
python generate_russian_data.py --limit 2500 -o data/train_russian.jsonl

# 2. SFT обучение — 4 эпохи с reasoning данными (~25 мин)
# Автоматически резюмирует с существующих адаптеров если есть
echo ""
echo "[2/3] SFT обучение (4 эпохи, reasoning данные)..."
python train.py --dataset data/train_russian.jsonl --epochs 4

# 3. GRPO обучение — 300 шагов с 6 reward-функциями (~50 мин)
echo ""
echo "[3/3] GRPO обучение (300 шагов, 6 reward-функций)..."
python train_grpo.py --dataset data/train_russian.jsonl --steps 300 --generations 2

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=========================================="
echo "  ГОТОВО! Время: ${ELAPSED} мин"
echo "=========================================="
echo ""
echo "  SFT адаптеры:  adapters/fact_checker_lora"
echo "  GRPO адаптеры: adapters/fact_checker_grpo"
echo ""
echo "Следующие шаги:"
echo "  python evaluate.py          - оценка точности"
echo "  streamlit run app.py        - веб-интерфейс"
