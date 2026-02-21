#!/bin/bash
echo "============================================"
echo " Fact-Checker: Установка и запуск тренировки"
echo " Требуется: Python 3.10+, NVIDIA GPU (CUDA)"
echo "============================================"
echo

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "ОШИБКА: Python3 не найден."
    exit 1
fi

# Создание venv
if [ ! -d "venv" ]; then
    echo "[1/4] Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация venv
source venv/bin/activate

# Установка PyTorch с CUDA
echo "[2/4] Установка PyTorch с CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Установка Unsloth и зависимостей
echo "[3/4] Установка Unsloth и зависимостей..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install transformers datasets sentencepiece protobuf xformers

# Запуск тренировки
echo "[4/4] Запуск тренировки..."
echo
python3 train.py --dataset data/train.jsonl

echo
echo "============================================"
echo " Тренировка завершена!"
echo " Адаптеры сохранены в: adapters/fact_checker_lora"
echo "============================================"
