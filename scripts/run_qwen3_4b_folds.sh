#!/usr/bin/env bash
# 5-fold mixing of the pool-trained Qwen3-4B-Base adapter (pool + train.csv x5), grouped by prompt; about 4.2 h on the RTX 3090.
# Waits for step 1 of scripts/run_qwen3_4b.sh when that job is still running.
# Outputs: oof_mix_Qwen3-4B-Base-5fold_{half,full}.npy, logs/qwen_mix.jsonl, and one deployable adapter per fold in
# models/Qwen3-4B-Base-lora-mix-5fold-fold<k>/. Safe to rerun after a crash: finished folds are skipped.
# GPU/CPU sensors go to logs/sensors_qwen3_4b_folds.csv.
#
#   nohup setsid scripts/run_qwen3_4b_folds.sh > logs/qwen3_4b_folds.out 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false  # the machine has reset under many-thread load
PY=.venv/bin/python
BASE=Qwen/Qwen3-4B-Base

while pgrep -f "^bash scripts/run_qwen3_4b.sh" > /dev/null; do sleep 20; done  # GPU still in use by step 1
if [[ ! -f logs/qwen3_4b.step1.done ]]; then
    echo "step 1 did not finish (no logs/qwen3_4b.step1.done): rerun scripts/run_qwen3_4b.sh first, then this script" >&2
    exit 1
fi

$PY src/sensor_log.py logs/sensors_qwen3_4b_folds.csv 5 &
SENSORS=$!
trap 'kill $SENSORS 2>/dev/null || true' EXIT

echo "== 5-fold mixing: start $(date '+%F %T')"
$PY src/qwen_mix_folds.py --base $BASE --adapter models/Qwen3-4B-Base-lora-full \
    --folds 5 --save_folds --repeat 5 --pool_rows 20000 --lr 3e-5 --tag=-5fold
echo "== 5-fold mixing: all done $(date '+%F %T')"
