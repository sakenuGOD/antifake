@echo off
chcp 65001 >nul
echo ============================================
echo  FACT-CHECKER: Полный цикл обучения
echo  Русский язык, SFT + GRPO (~1.5 часа)
echo  RTX 5070 (12GB, Blackwell, bf16, CUDA 12.8)
echo ============================================
echo.

set START_TIME=%TIME%

:: ===== ПОИСК PYTHON =====
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
echo Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: ===== VENV =====
if exist "venv\Scripts\activate.bat" (
    echo [SETUP] venv найден, активирую...
    call venv\Scripts\activate.bat
) else (
    echo [SETUP] Создание venv + установка зависимостей...
    %PYTHON_CMD% -m venv venv
    call venv\Scripts\activate.bat

    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps trl peft accelerate bitsandbytes
    pip install transformers datasets sentencepiece protobuf
)

:: Удаляем xformers (Blackwell sm_120 не поддерживает)
pip uninstall xformers -y >nul 2>&1
if exist "unsloth_compiled_cache" rmdir /s /q unsloth_compiled_cache

:: ===== ПРОВЕРКА GPU =====
echo.
echo === Проверка GPU ===
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB') if torch.cuda.is_available() else None"
echo.

:: ============================================
:: ЭТАП 1: ГЕНЕРАЦИЯ РУССКОГО ДАТАСЕТА
:: 2500 примеров, все с <reasoning> тегами
:: 35%% ПРАВДА + 35%% ЛОЖЬ + 30%% НЕ ПОДТВЕРЖДЕНО
:: ============================================
echo ============================================
echo [1/3] Генерация русского датасета (2500 примеров с reasoning)...
echo ============================================
python generate_russian_data.py --limit 2500 -o data/train_russian.jsonl
if errorlevel 1 (
    echo ОШИБКА: Генерация данных не удалась!
    pause
    exit /b 1
)
echo.

:: ============================================
:: ЭТАП 2: SFT ОБУЧЕНИЕ
:: 4 эпохи, LoRA r=16, bf16, train_on_responses_only
:: Резюмирует с существующих адаптеров если есть
:: ============================================
echo ============================================
echo [2/3] SFT обучение (4 эпохи, reasoning данные)...
echo ============================================
python train.py --dataset data/train_russian.jsonl --epochs 4
if errorlevel 1 (
    echo ОШИБКА: SFT обучение не удалось!
    pause
    exit /b 1
)
echo.

:: ============================================
:: ЭТАП 3: GRPO ОБУЧЕНИЕ
:: 300 шагов, 2 генерации, 6 reward-функций
:: Учит модель рассуждать качественно
:: ============================================
echo ============================================
echo [3/3] GRPO обучение (300 шагов, reasoning rewards)...
echo ============================================
python train_grpo.py --dataset data/train_russian.jsonl --steps 300 --generations 2
if errorlevel 1 (
    echo ОШИБКА: GRPO обучение не удалось!
    pause
    exit /b 1
)

:: ============================================
:: ГОТОВО
:: ============================================
echo.
echo ============================================
echo  ОБУЧЕНИЕ ЗАВЕРШЕНО!
echo  Начало: %START_TIME%
echo  Конец:  %TIME%
echo.
echo  SFT адаптеры:  adapters\fact_checker_lora
echo  GRPO адаптеры: adapters\fact_checker_grpo
echo ============================================
echo.
echo  Следующие шаги:
echo    python evaluate.py          - оценка точности
echo    streamlit run app.py        - веб-интерфейс
echo.
pause
