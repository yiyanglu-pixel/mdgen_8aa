#!/bin/bash
# =============================================================================
# 8AA Octapeptide Training Pipeline
# =============================================================================
# This script runs the complete pipeline:
#   Step 1: Generate split CSVs from preprocessed .npy files
#   Step 2: Launch training
#
# Usage:
#   bash scripts/run_8AA_train.sh
#
# Prerequisites:
#   - Preprocessed .npy files in DATA_DIR (from prep_sims.py)
#   - Raw data in RAW_DATA_DIR (only needed if splits don't exist yet)
# =============================================================================

set -e

# ========================= USER CONFIG =========================
# Path to preprocessed .npy files (e.g., opep_0000_i1000.npy)
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"

# Path to raw octapeptides data (for generating splits)
RAW_DATA_DIR="/localhome3/lyy/8pep_gb_sim/octapeptides_data/ONE_octapeptides"

# .npy file suffix (must match what prep_sims.py used)
SUFFIX="_i1000"

# Training run name
RUN_NAME="8AA_sim_912"

# Number of trajectory frames per sample
NUM_FRAMES=100

# Training epochs
EPOCHS=1000

# Batch size (reduce if GPU OOM)
BATCH_SIZE=8

# Learning rate
LR=1e-4

# Enable Weights & Biases logging (set to "" to disable)
USE_WANDB=""  # set to "--wandb" to enable

# Checkpoint save frequency (epochs)
CKPT_FREQ=10

# GPU device
export CUDA_VISIBLE_DEVICES=0
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  8AA Octapeptide Training Pipeline"
echo "============================================"

# --- Step 0: Verify data exists ---
echo ""
echo "[Step 0] Verifying data..."
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
if [ "$NPY_COUNT" -eq 0 ]; then
    echo "ERROR: No .npy files found in ${DATA_DIR} with suffix ${SUFFIX}"
    echo "Run prep_sims.py first. Example:"
    echo "  python -m scripts.prep_sims \\"
    echo "    --split splits/8AA.csv \\"
    echo "    --sim_dir ${RAW_DATA_DIR} \\"
    echo "    --outdir ${DATA_DIR} \\"
    echo "    --num_workers 8 \\"
    echo "    --suffix ${SUFFIX} \\"
    echo "    --stride 1000 \\"
    echo "    --octapeptides"
    exit 1
fi
echo "  Found ${NPY_COUNT} .npy files in ${DATA_DIR}"

# --- Step 1: Generate splits if needed ---
echo ""
echo "[Step 1] Checking split files..."
if [ -f "splits/8AA_train.csv" ] && [ -f "splits/8AA_val.csv" ]; then
    TRAIN_COUNT=$(tail -n +2 splits/8AA_train.csv | wc -l)
    VAL_COUNT=$(tail -n +2 splits/8AA_val.csv | wc -l)
    echo "  Splits already exist: train=${TRAIN_COUNT}, val=${VAL_COUNT}"
    echo "  Delete splits/8AA_*.csv to regenerate"
else
    echo "  Generating splits from ${RAW_DATA_DIR}..."
    python -m scripts.generate_8AA_splits \
        --data_dir "${RAW_DATA_DIR}" \
        --outdir splits \
        --train_frac 0.8 \
        --val_frac 0.1 \
        --test_frac 0.1

    # Verify generated splits only contain peptides that have .npy files
    echo "  Validating splits against available .npy files..."
    python -c "
import pandas as pd
import os, sys

suffix = '${SUFFIX}'
data_dir = '${DATA_DIR}'

for split_name in ['8AA_train', '8AA_val', '8AA_test']:
    path = f'splits/{split_name}.csv'
    df = pd.read_csv(path, index_col='name')
    missing = [n for n in df.index if not os.path.exists(f'{data_dir}/{n}{suffix}.npy')]
    if missing:
        print(f'  WARNING: {len(missing)} entries in {split_name} have no .npy file')
        # Filter to only entries with data
        df = df.loc[[n for n in df.index if os.path.exists(f'{data_dir}/{n}{suffix}.npy')]]
        df.to_csv(path)
        print(f'  Filtered {split_name}: {len(df)} entries')
    else:
        print(f'  {split_name}: {len(df)} entries - all OK')
"
fi

# --- Step 2: Train ---
echo ""
echo "[Step 2] Starting training..."
echo "  Run name:   ${RUN_NAME}"
echo "  Data dir:   ${DATA_DIR}"
echo "  Suffix:     ${SUFFIX}"
echo "  Num frames: ${NUM_FRAMES}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs:     ${EPOCHS}"
echo "  LR:         ${LR}"
echo ""

export MODEL_DIR="workdir/${RUN_NAME}"
mkdir -p "${MODEL_DIR}"

python train.py \
    --sim_condition \
    --train_split splits/8AA_train.csv \
    --val_split splits/8AA_val.csv \
    --data_dir "${DATA_DIR}" \
    --crop 8 \
    --abs_pos_emb \
    --num_frames ${NUM_FRAMES} \
    --prepend_ipa \
    --suffix "${SUFFIX}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --ckpt_freq ${CKPT_FREQ} \
    --run_name "${RUN_NAME}" \
    ${USE_WANDB}

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints saved to: ${MODEL_DIR}"
echo "============================================"
