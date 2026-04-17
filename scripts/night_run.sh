#!/bin/bash
# night_run.sh — запуск на ночь (~8-9 часов).
#
# Что делает:
#   1. Baseline eval:      base Mistral vs fact_checker_lora (SFT) vs fact_checker_grpo (broken)
#   2. GRPO retrain:       fact_checker_lora (SFT init) -> fact_checker_grpo_v2, 300 шагов
#   3. Eval каждого checkpoint из v2 (50, 100, 150, 200, 250, 300)
#   4. Сводка в final_report.md
#
# Запуск:
#   cd ~/antifake && source venv/bin/activate && ./scripts/night_run.sh
#
# Проверить перед сном:
#   python scripts/sanity_check.py   # ~4 мин
#
# ВАЖНО: .env должен содержать SERPAPI_API_KEY (DDG работает без ключа).

set -u  # unset variables = error (BUT no -e: мы хотим продолжать при неудаче одного этапа)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="$(date +%Y%m%d_%H%M)"
LOG_DIR="night_logs_${DATE_TAG}"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/progress.log"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"
}

run_eval() {
    local adapter="$1"
    local out_name="$2"
    log "Eval: adapter=$adapter  out=$out_name"
    python evaluate_universal.py \
        --adapter "$adapter" \
        --dataset data/eval_dataset.jsonl \
        --output "$LOG_DIR/${out_name}.json" \
        2>&1 | tee "$LOG_DIR/${out_name}.log"
    local rc=${PIPESTATUS[0]}
    log "Eval $out_name finished with rc=$rc"
    return $rc
}

log "=== NIGHT RUN START ==="
log "ROOT: $ROOT_DIR"
log "LOG_DIR: $LOG_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$MAIN_LOG"

# ========================================================================
# STAGE 1: BASELINE EVAL (3x30min = 1.5h)
# ========================================================================
log ""
log "=== STAGE 1/3: BASELINE EVAL ==="

run_eval "base" "baseline_01_base" || log "[WARN] base eval failed"
run_eval "adapters/fact_checker_lora" "baseline_02_sft" || log "[WARN] SFT eval failed"
run_eval "adapters/fact_checker_grpo" "baseline_03_grpo_broken" || log "[WARN] GRPO eval failed"

# ========================================================================
# STAGE 2: GRPO RETRAIN (6-7h)
# ========================================================================
log ""
log "=== STAGE 2/3: GRPO RETRAIN (SFT init, 300 steps, max_len=1024) ==="

python train_grpo.py \
    --load-adapter adapters/fact_checker_lora \
    --output-dir adapters/fact_checker_grpo_v2 \
    --steps 300 \
    --generations 2 \
    2>&1 | tee "$LOG_DIR/grpo_retrain.log"
GRPO_RC=${PIPESTATUS[0]}
log "GRPO retrain finished with rc=$GRPO_RC"

# ========================================================================
# STAGE 3: EVAL CHECKPOINTS (6x30min = 3h, can be cut off if time runs out)
# ========================================================================
log ""
log "=== STAGE 3/3: EVAL CHECKPOINTS ==="

CKPT_DIR="adapters/fact_checker_grpo_v2"
if [ -d "$CKPT_DIR" ]; then
    # Sorted by step number numerically
    for ckpt in $(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -V); do
        step=$(basename "$ckpt" | sed 's/checkpoint-//')
        run_eval "$ckpt" "eval_step${step}" || log "[WARN] eval step $step failed"
    done
    # Final (post-training save, not a checkpoint-*)
    run_eval "$CKPT_DIR" "eval_final" || log "[WARN] final eval failed"
else
    log "[ERROR] $CKPT_DIR не существует — retrain провалился"
fi

# ========================================================================
# REPORT
# ========================================================================
log ""
log "=== SUMMARY ==="

python scripts/summarize_night.py --log-dir "$LOG_DIR" > "$LOG_DIR/final_report.md" \
    2>&1 || log "[WARN] summarize failed"

if [ -f "$LOG_DIR/final_report.md" ]; then
    log "Summary written to $LOG_DIR/final_report.md"
    cat "$LOG_DIR/final_report.md" | tee -a "$MAIN_LOG"
fi

log "=== NIGHT RUN END ==="
