#!/bin/bash
# =============================================================================
# 8AA Octapeptide Multi-GPU Training Pipeline (7 GPUs: 1-7)
# =============================================================================
# Usage:
#   bash scripts/run_8AA_train_multi.sh
# =============================================================================

set -e

# Prevent system mpi4py (Python 3.12) from conflicting with conda env (Python 3.9)
export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
RAW_DATA_DIR="/localhome3/lyy/octapeptides_data"
SUFFIX="_i100"
RUN_NAME="8AA_sim_912_multi"
NUM_FRAMES=100

# --- Multi-GPU config ---
# Use GPUs 1-7 (7 cards), leave GPU 0 free
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NUM_GPUS=7

# --- Training hyperparams ---
# Per-GPU batch size: 16 (up from 8, ~8.4GB/10.2GB per card)
# Effective batch size = 16 * 7 = 112
BATCH_SIZE=16

# Epochs
EPOCHS=2000

# Learning rate: sqrt-scaled for larger effective batch
# Original: 1e-4 at bs=8 -> sqrt(112/8) * 1e-4 ≈ 3.7e-4
LR=3.7e-4

# Mixed precision for speed + memory savings
PRECISION="bf16-mixed"

CKPT_FREQ=50
USE_WANDB=""  # set to "--wandb" to enable
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  8AA Multi-GPU Training (${NUM_GPUS} GPUs)"
echo "============================================"

# --- Step 0: Verify data exists ---
echo ""
echo "[Step 0] Verifying data..."
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
if [ "$NPY_COUNT" -eq 0 ]; then
    echo "ERROR: No .npy files found in ${DATA_DIR} with suffix ${SUFFIX}"
    exit 1
fi
echo "  Found ${NPY_COUNT} .npy files in ${DATA_DIR}"

# --- Step 1: Check splits ---
echo ""
echo "[Step 1] Checking split files..."
if [ -f "splits/8AA_train.csv" ] && [ -f "splits/8AA_val.csv" ]; then
    TRAIN_COUNT=$(tail -n +2 splits/8AA_train.csv | wc -l)
    VAL_COUNT=$(tail -n +2 splits/8AA_val.csv | wc -l)
    echo "  Splits exist: train=${TRAIN_COUNT}, val=${VAL_COUNT}"
else
    echo "  Generating splits..."
    python -m scripts.generate_8AA_splits \
        --data_dir "${RAW_DATA_DIR}" \
        --outdir splits \
        --train_frac 0.8 \
        --val_frac 0.1 \
        --test_frac 0.1
fi

# --- Step 2: Train (multi-GPU DDP) ---
echo ""
echo "[Step 2] Starting multi-GPU training..."
echo "  GPUs:            ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} cards)"
echo "  Per-GPU batch:   ${BATCH_SIZE}"
echo "  Effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo "  Epochs:          ${EPOCHS}"
echo "  LR:              ${LR}"
echo "  Precision:       ${PRECISION}"
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
    --precision ${PRECISION} \
    --ckpt_freq ${CKPT_FREQ} \
    --run_name "${RUN_NAME}" \
    ${USE_WANDB}

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints saved to: ${MODEL_DIR}"
echo "============================================"
