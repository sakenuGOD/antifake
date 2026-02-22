@echo off
chcp 65001 >nul
echo ============================================
echo  Fact-Checker: Установка и запуск тренировки
echo  RTX 5070 (12GB, Blackwell, bf16, CUDA 12.8)
echo  Требуется: Python 3.10-3.12
echo ============================================
echo.

:: Поиск подходящей версии Python (3.12 > 3.11 > 3.10)
set PYTHON_CMD=
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py -3.12
        goto :found
    )
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py -3.11
        goto :found
    )
    py -3.10 --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=py -3.10
        goto :found
    )
)

:: Fallback: проверяем python напрямую
python --version 2>nul | findstr /R "3\.1[0-2]\." >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :found
)

echo ОШИБКА: Python 3.10-3.12 не найден!
echo Python 3.13/3.14 слишком новые для PyTorch/Unsloth.
echo.
echo Установите Python 3.11:
echo   https://www.python.org/downloads/release/python-31110/
echo   (Обязательно отметьте "Add Python to PATH")
echo.
pause
exit /b 1

:found
echo Используется: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: Используем существующий venv или создаём новый
if exist "venv\Scripts\activate.bat" (
    echo [1/5] venv уже существует, пропускаю создание...
    call venv\Scripts\activate.bat
) else (
    echo [1/5] Создание виртуального окружения...
    %PYTHON_CMD% -m venv venv
    call venv\Scripts\activate.bat

    :: Установка PyTorch с CUDA 12.8 (Blackwell sm_120)
    echo [2/5] Установка PyTorch с CUDA 12.8 (для RTX 5070^)...
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    :: Установка Unsloth и зависимостей
    echo [3/5] Установка Unsloth (QLoRA + Flash Attention^)...
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

    echo [4/5] Установка остальных зависимостей...
    pip install transformers datasets sentencepiece protobuf
)

:: Удаляем xformers если установлен (не поддерживает Blackwell sm_120)
pip uninstall xformers -y >nul 2>&1

:: Удаляем кэш Unsloth (может содержать старый код с xformers)
if exist "unsloth_compiled_cache" (
    echo Очистка кэша Unsloth...
    rmdir /s /q unsloth_compiled_cache
)

:: Проверка CUDA
echo.
echo === Проверка GPU ===
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB') if torch.cuda.is_available() else None"
echo.

:: Скачивание датасета если не скачан
if not exist "data\train.jsonl" (
    echo.
    echo [DATASET] Датасет не найден, скачиваю 10к примеров...
    python download_dataset.py --limit 10000
)

:: Запуск тренировки
echo.
echo [5/5] Запуск тренировки (bf16, batch=2x4, seq=512, LoRA r=16)...
echo ============================================
python train.py --dataset data/train.jsonl

echo.
echo ============================================
echo  Тренировка завершена!
echo  Адаптеры сохранены в: adapters\fact_checker_lora
echo ============================================
pause
