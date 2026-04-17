# night_run.ps1 -- Windows PowerShell overnight runner.
# Mirrors night_run.sh but runs natively on Windows (no WSL required).
#
# Run from the activated venv:
#   .\scripts\night_run.ps1
#
# Prevents Windows from sleeping via SetThreadExecutionState.
# Writes logs to night_logs_<timestamp>\.

$ErrorActionPreference = 'Continue'

$dateTag = Get-Date -Format 'yyyyMMdd_HHmm'
$logDir  = "night_logs_$dateTag"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$mainLog = Join-Path $logDir 'progress.log'

function Log {
    param($msg)
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    $line | Out-File -FilePath $mainLog -Append -Encoding utf8
}

function Run-Eval {
    param($adapter, $outName)
    Log "Eval: adapter=$adapter -> $outName"
    $outPath = Join-Path $logDir "$outName.json"
    $logPath = Join-Path $logDir "$outName.log"
    python evaluate_universal.py --adapter "$adapter" --dataset data/eval_dataset.jsonl --output "$outPath" 2>&1 |
        Tee-Object -FilePath $logPath
    Log "Eval $outName finished (exit=$LASTEXITCODE)"
}

# Keep system awake during training.
Add-Type -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -Name 'Power' -Namespace 'Win32' -ErrorAction SilentlyContinue

# ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001) | ES_AWAYMODE_REQUIRED (0x00000040)
[Win32.Power]::SetThreadExecutionState(0x80000041) | Out-Null

Log "=== NIGHT RUN START ==="
Log "LOG_DIR: $logDir"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | Tee-Object "$logDir\gpu.log"

# =====================================================================
# STAGE 1: BASELINE SFT EVAL
# =====================================================================
Log ""
Log "=== STAGE 1/3: BASELINE SFT EVAL (~30-60 min) ==="
Run-Eval "adapters/fact_checker_lora" "baseline_sft"

# =====================================================================
# STAGE 2: GRPO RETRAIN (6-8h)
# =====================================================================
Log ""
Log "=== STAGE 2/3: GRPO RETRAIN 300 steps, max_len=1024, SFT-init ==="

python train_grpo.py `
    --load-adapter adapters/fact_checker_lora `
    --output-dir adapters/fact_checker_grpo_v2 `
    --steps 300 `
    --generations 2 2>&1 |
    Tee-Object "$logDir\grpo_retrain.log"
Log "GRPO retrain finished (exit=$LASTEXITCODE)"

# =====================================================================
# STAGE 3: EVAL EVERY CHECKPOINT
# =====================================================================
Log ""
Log "=== STAGE 3/3: EVAL CHECKPOINTS ==="

$ckptDir = "adapters/fact_checker_grpo_v2"
if (Test-Path $ckptDir) {
    $checkpoints = Get-ChildItem -Path $ckptDir -Directory -Filter "checkpoint-*" |
                    Sort-Object { [int]($_.Name -replace 'checkpoint-', '') }

    foreach ($ckpt in $checkpoints) {
        $step = $ckpt.Name -replace 'checkpoint-', ''
        Run-Eval $ckpt.FullName "eval_step$step"
    }

    # Final saved adapter (post-training)
    Run-Eval $ckptDir "eval_final"
} else {
    Log "[ERROR] $ckptDir does not exist -- retrain failed."
}

# =====================================================================
# REPORT
# =====================================================================
Log ""
Log "=== SUMMARY ==="

$reportPath = Join-Path $logDir "final_report.md"
python scripts/summarize_night.py --log-dir $logDir | Out-File -FilePath $reportPath -Encoding utf8

if (Test-Path $reportPath) {
    Log "Summary -> $reportPath"
    Get-Content $reportPath | Tee-Object -FilePath $mainLog -Append
}

# Restore normal power state
[Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null

Log "=== NIGHT RUN END ==="
Log "In the morning: Get-Content $logDir\final_report.md"
