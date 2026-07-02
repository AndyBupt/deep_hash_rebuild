#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# run_additional_experiments.sh
# One-click runner for the new supplementary experiments.
#
# Usage:
#   bash run_additional_experiments.sh
#   FVC2004_ROOT=/path/to/FVC2004 RGSS_MODEL_PATH=/path/to/final_model.pth bash run_additional_experiments.sh
#   RUN_ONLY="multi_helper public_r" bash run_additional_experiments.sh
#
# Optional env vars:
#   FVC2004_ROOT       Data root for FVC2004
#   RGSS_MODEL_PATH    Model checkpoint path
#   PYTHON_BIN         Python executable, default: python
#   RUN_ONLY           Space-separated short names to run a subset
#                      Supported: multi_helper public_r iom irreversibility
#   LOG_ROOT           Log directory, default: logs_additional_experiments
# =========================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
FVC2004_ROOT="${FVC2004_ROOT:-/root/autodl-tmp/FVC2004}"
RGSS_MODEL_PATH="${RGSS_MODEL_PATH:-checkpoints/final_model.pth}"
LOG_ROOT="${LOG_ROOT:-logs_additional_experiments}"
RUN_ONLY="${RUN_ONLY:-multi_helper public_r iom irreversibility}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/$TIMESTAMP"
mkdir -p "$RUN_LOG_DIR"

export FVC2004_ROOT
export RGSS_MODEL_PATH

printf '\n============================================================\n'
printf 'Additional Experiments Runner\n'
printf '============================================================\n'
printf 'Project root      : %s\n' "$PROJECT_ROOT"
printf 'Python            : %s\n' "$PYTHON_BIN"
printf 'FVC2004_ROOT      : %s\n' "$FVC2004_ROOT"
printf 'RGSS_MODEL_PATH   : %s\n' "$RGSS_MODEL_PATH"
printf 'RUN_ONLY          : %s\n' "$RUN_ONLY"
printf 'Log dir           : %s\n' "$RUN_LOG_DIR"
printf 'Started at        : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf '============================================================\n\n'

if [[ ! -d "$FVC2004_ROOT" ]]; then
  echo "[ERROR] FVC2004_ROOT does not exist: $FVC2004_ROOT" >&2
  exit 1
fi

if [[ ! -f "$RGSS_MODEL_PATH" ]]; then
  echo "[ERROR] RGSS_MODEL_PATH does not exist: $RGSS_MODEL_PATH" >&2
  exit 1
fi

run_one() {
  local short_name="$1"
  local script_name="$2"
  local output_dir="$3"
  local desc="$4"
  local log_file="$RUN_LOG_DIR/${short_name}.log"

  if [[ " $RUN_ONLY " != *" $short_name "* ]]; then
    echo "[SKIP] $short_name not requested"
    return 0
  fi

  if [[ ! -f "$script_name" ]]; then
    echo "[ERROR] script not found: $script_name" >&2
    exit 1
  fi

  printf '\n------------------------------------------------------------\n'
  printf '[RUN] %s\n' "$desc"
  printf 'Script      : %s\n' "$script_name"
  printf 'Output dir  : %s\n' "$output_dir"
  printf 'Log file    : %s\n' "$log_file"
  printf 'Start time  : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%s\n' '------------------------------------------------------------'

  "$PYTHON_BIN" "$script_name" 2>&1 | tee "$log_file"

  printf '[DONE] %s\n' "$desc"
  printf 'Finish time : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Artifacts   : %s\n' "$output_dir"
}

run_one "multi_helper" \
  "evaluate_multi_helper_leakage.py" \
  "results_multi_helper_leakage/" \
  "Multi-helper / multi-revocation leakage"

run_one "public_r" \
  "evaluate_public_r_linkability.py" \
  "results_public_r_linkability/" \
  "Public-R vs protected-R linkability"

run_one "iom" \
  "evaluate_iom_baseline.py" \
  "results_iom_baseline/" \
  "IoM-style stronger cancelable baseline"

run_one "irreversibility" \
  "evaluate_irreversibility_attack.py" \
  "results_irreversibility_attack/" \
  "Irreversibility / reconstruction attack"

printf '\n============================================================\n'
printf 'All requested experiments finished.\n'
printf 'Finished at       : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
printf 'Logs available in : %s\n' "$RUN_LOG_DIR"
printf '============================================================\n\n'

printf 'Quick result locations:\n'
printf '  - results_multi_helper_leakage/\n'
printf '  - results_public_r_linkability/\n'
printf '  - results_iom_baseline/\n'
printf '  - results_irreversibility_attack/\n\n'
