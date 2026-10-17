#!/usr/bin/env bash
# Main transformer experiments (seed 42, saved to models/ft-<name>), run one at a time on MPS.
# Usage: scripts/run_transformers.sh [model ...]   (default: all three)
# The 278M-parameter XLM-R models use batch 4 x 4 accumulation steps (effective batch 16)
# to fit in 16 GB of unified memory; AfriBERTa (111M) uses 8 x 2.
# NOTE: run nothing else memory-heavy at the same time. Swapping slows MPS training ~50x.
set -e
cd "$(dirname "$0")/.."
MODELS=${@:-castorini/afriberta_base Davlan/afro-xlmr-base FacebookAI/xlm-roberta-base}
for M in $MODELS; do
  N=$(basename $M)
  BS=8; [[ $N == *xlm* ]] && BS=4
  .venv/bin/python -u -m src.finetune --model $M --seeds 42 --bs $BS --save_dir models/ft-$N > logs/$N.log 2>&1
  grep "seed=" logs/$N.log
done
