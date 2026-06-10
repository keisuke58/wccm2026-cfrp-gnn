#!/bin/bash
# ─── DDPM anomaly detection training on Vancouver ──────────────────────────
# Usage: bash scripts/run_diffusion.sh [--eval]
# Prerequisites (on vancouver):
#   - CFRP data mounted at /home/nishioka/CFRP (same path as local)
#   - conda activate defect_hole_env  (or any env with torch, scipy, sklearn)
# ─────────────────────────────────────────────────────────────────────────────
set -eu
cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

MODE=${1:-train}

if [ "$MODE" = "--eval" ]; then
  python diffusion_anomaly.py \
    --mode eval \
    --ckpt results/diffusion/ddpm_best.pt \
    --base_ch 32
else
  python diffusion_anomaly.py \
    --mode train \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-4 \
    --base_ch 32 \
    --max_train 2000
fi
