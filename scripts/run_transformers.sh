#!/usr/bin/env bash
# Main transformer experiments: 3 models x seed 42 (saved), run sequentially on MPS.
set -e
cd "$(dirname "$0")/.."
for M in castorini/afriberta_base Davlan/afro-xlmr-base FacebookAI/xlm-roberta-base; do
  N=$(basename $M)
  .venv/bin/python -u -m src.finetune --model $M --seeds 42 --save_dir models/ft-$N > logs/$N.log 2>&1
  grep "seed=" logs/$N.log
done
