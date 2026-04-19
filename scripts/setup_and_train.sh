#!/bin/bash
echo "============================================"
echo " Fact-Checker: Установка и запуск тренировки"
echo " RTX 5070 (12GB, Blackwell, bf16, CUDA 12.8)"
echo " Требуется: Python 3.10-3.12"
echo "============================================"
echo

# Поиск подходящего Python
PYTHON_CMD=""
for ver in python3.12 python3.11 python3.10; do
    if command -v "$ver" &> /dev/null; then
        PYTHON_CMD="$ver"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ОШИБКА: Python 3.10-3.12 не найден!"
    echo "Установите Python 3.11: https://www.python.org/downloads/release/python-31110/"
    exit 1
fi

echo "Используется: $PYTHON_CMD"
$PYTHON_CMD --version
echo

# Удаляем старый venv
if [ -d "venv" ]; then
    echo "Удаление старого venv..."
    rm -rf venv
fi

# Создание venv
echo "[1/5] Создание виртуального окружения..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# PyTorch с CUDA 12.8
echo "[2/5] Установка PyTorch с CUDA 12.8 (для RTX 5070)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Проверка GPU
echo
echo "=== Проверка GPU ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A')"
echo

# Unsloth
echo "[3/5] Установка Unsloth (QLoRA + Flash Attention)..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

echo "[4/5] Установка остальных зависимостей..."
pip install transformers datasets sentencepiece protobuf xformers

# Датасет
if [ ! -f "data/train.jsonl" ]; then
    echo
    echo "[DATASET] Датасет не найден, скачиваю 500к примеров..."
    python3 download_dataset.py --limit 500000
fi

# Тренировка
echo
echo "[5/5] Запуск тренировки (bf16, batch=8, LoRA r=32)..."
echo "============================================"
python3 train.py --dataset data/train.jsonl

echo
echo "============================================"
echo " Тренировка завершена!"
echo " Адаптеры сохранены в: adapters/fact_checker_lora"
echo "============================================"
