#!/usr/bin/env bash
set -euo pipefail
cd /homes/ramneet/DICE_Harald
source .venv/bin/activate
mkdir -p llm_exp/results_v3/logs
exec python llm_exp/run_experiments.py \
  --quick \
  --n 10 25 50 100 \
  --rounds 8 \
  --output llm_exp/results_v3 \
  2>&1 | tee llm_exp/results_v3/run.log
