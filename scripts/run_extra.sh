#!/usr/bin/env bash
# Follow-up transformer experiments (run after scripts/run_transformers.sh).
#  1) extra seeds for the two best models -> mean ± std over 3 seeds
#  2) XLM-R retries: is its failure a hyper-parameter problem or a language-coverage problem?
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u -m src.finetune"
$PY --model castorini/afriberta_base --seeds 43 44                      > logs/afriberta_seeds.log 2>&1; grep "seed=" logs/afriberta_seeds.log
$PY --model Davlan/afro-xlmr-base --seeds 43 44                         > logs/afroxlmr_seeds.log 2>&1;  grep "seed=" logs/afroxlmr_seeds.log
$PY --model FacebookAI/xlm-roberta-base --lr 5e-5 --epochs 15 --tag -lr5e-5-ep15 > logs/xlmr_lr.log 2>&1;     grep "seed=" logs/xlmr_lr.log
$PY --model FacebookAI/xlm-roberta-base --no_freeze_embeddings --tag -fullFT   > logs/xlmr_full.log 2>&1;   grep "seed=" logs/xlmr_full.log
