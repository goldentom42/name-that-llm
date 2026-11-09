#!/usr/bin/env bash
# Qwen3-4B-Base LoRA classifier, the same recipe as the deployed Qwen2.5-1.5B (about 7.5 h on the RTX 3090):
#   1. one epoch over the whole open pool (~5.5 h), resume checkpoint every 500 steps (~12 min)
#   2. 2-fold out-of-fold test of the mixing step (pool + train.csv x5), for the blend search (~1.3 h)
#   3. final mixing run on all train.csv rows -> models/Qwen3-4B-Base-lora-mix (~50 min)
# Safe to rerun after a crash or a kill: finished steps are skipped (logs/qwen3_4b.stepN.done) and
# step 1 continues from its last checkpoint. GPU/CPU sensors go to logs/sensors_qwen3_4b.csv.
#
#   nohup scripts/run_qwen3_4b.sh > logs/qwen3_4b_run.out 2>&1 &
#   tail -f logs/qwen3_4b_run.out
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BASE=Qwen/Qwen3-4B-Base

$PY src/sensor_log.py logs/sensors_qwen3_4b.csv 5 &
SENSORS=$!
trap 'kill $SENSORS 2>/dev/null || true' EXIT

step() {  # step <n> <description> <command...>
    local n=$1 desc=$2; shift 2
    if [[ -f logs/qwen3_4b.step$n.done ]]; then echo "== step $n ($desc): already done, skipping"; return; fi
    echo "== step $n ($desc): start $(date '+%F %T')"
    "$@"
    touch logs/qwen3_4b.step$n.done
    echo "== step $n ($desc): done $(date '+%F %T')"
}

step 1 "pool LoRA, full pool" \
    $PY src/train_qwen_lora.py $BASE --rows 0 --epochs 1 --evals 8 --tag=-full --ckpt_every 500 --resume
step 2 "2-fold mixing test" \
    $PY src/qwen_mix_folds.py --base $BASE --adapter models/Qwen3-4B-Base-lora-full --repeat 5 --pool_rows 20000 --lr 3e-5
step 3 "final mixing run" \
    $PY src/qwen_mix_folds.py --base $BASE --adapter models/Qwen3-4B-Base-lora-full --repeat 5 --pool_rows 20000 --lr 3e-5 --final
echo "== all done $(date '+%F %T')"
