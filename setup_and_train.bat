@echo off
chcp 65001 >nul
echo ============================================
echo  Fact-Checker: Установка и запуск тренировки
echo  RTX 5070 (12GB, Blackwell, bf16, CUDA 12.8)
echo  Требуется: Python 3.10-3.12
echo ============================================
echo.
echo Используйте run_training.bat для полного цикла обучения.
echo Этот скрипт только устанавливает зависимости.
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
echo Установите Python 3.11: https://www.python.org/downloads/release/python-31110/
pause
exit /b 1

:found
echo Используется: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: Используем существующий venv или создаём новый
if exist "venv\Scripts\activate.bat" (
    echo [1/4] venv уже существует, пропускаю создание...
    call venv\Scripts\activate.bat
) else (
    echo [1/4] Создание виртуального окружения...
    %PYTHON_CMD% -m venv venv
    call venv\Scripts\activate.bat

    echo [2/4] Установка PyTorch с CUDA 12.8 (для RTX 5070^)...
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

    echo [3/4] Установка Unsloth (QLoRA + Flash Attention^)...
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes

    echo [4/4] Установка остальных зависимостей...
    pip install transformers datasets sentencepiece protobuf
)

:: Удаляем xformers если установлен (не поддерживает Blackwell sm_120)
pip uninstall xformers -y >nul 2>&1

if exist "unsloth_compiled_cache" (
    echo Очистка кэша Unsloth...
    rmdir /s /q unsloth_compiled_cache
)

:: Проверка CUDA
echo.
echo === Проверка GPU ===
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB') if torch.cuda.is_available() else None"
echo.

echo ============================================
echo  Установка завершена!
echo  Для запуска обучения: run_training.bat
echo ============================================
pause
