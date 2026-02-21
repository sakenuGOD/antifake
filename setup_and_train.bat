@echo off
chcp 65001 >nul
echo ============================================
echo  Fact-Checker: Установка и запуск тренировки
echo  Требуется: Python 3.10+, NVIDIA GPU (CUDA)
echo ============================================
echo.

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден. Установите Python 3.10+ с python.org
    pause
    exit /b 1
)

:: Проверка CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if errorlevel 1 (
    echo [INFO] PyTorch с CUDA не найден, будет установлен...
)

:: Создание venv
if not exist "venv" (
    echo [1/4] Создание виртуального окружения...
    python -m venv venv
)

:: Активация venv
call venv\Scripts\activate.bat

:: Установка PyTorch с CUDA
echo [2/4] Установка PyTorch с CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: Установка Unsloth и зависимостей
echo [3/4] Установка Unsloth и зависимостей...
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install transformers datasets sentencepiece protobuf xformers

:: Запуск тренировки
echo [4/4] Запуск тренировки...
echo.
python train.py --dataset data/train.jsonl

echo.
echo ============================================
echo  Тренировка завершена!
echo  Адаптеры сохранены в: adapters/fact_checker_lora
echo ============================================
pause
