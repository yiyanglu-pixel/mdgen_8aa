#!/bin/bash
# =============================================================================
# 8AA Phase 1 Training: 2000 epochs, 8 GPUs, fp32
# =============================================================================
# Usage:
#   bash scripts/run_8AA_phase1.sh
#
# Or run in background:
#   nohup bash scripts/run_8AA_phase1.sh > workdir/8AA_sim_phase1/train_phase1.log 2>&1 &
# =============================================================================

set -e

# Prevent system mpi4py (Python 3.12) from conflicting with conda env (Python 3.9)
export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
SUFFIX="_i1000"
RUN_NAME="8AA_sim_phase1"
NUM_FRAMES=100

# 8 GPUs (0-7)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Per-GPU batch size: 4, effective batch = 4 * 8 = 32
BATCH_SIZE=4

# Total epochs for Phase 1
EPOCHS=2000

# Learning rate: sqrt-scaled for effective batch size
# sqrt(32/8) * 1e-4 = 2 * 1e-4 = 2e-4
LR=2e-4

# fp32 precision
PRECISION="32-true"

# Save checkpoint every 50 epochs (40 checkpoints total)
CKPT_FREQ=50

USE_WANDB=""  # set to "--wandb" to enable W&B logging
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  8AA Phase 1 Training (2000 epochs, 8 GPUs)"
echo "============================================"

# --- Verify data ---
echo ""
echo "[Step 0] Verifying data..."
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
if [ "$NPY_COUNT" -eq 0 ]; then
    echo "ERROR: No .npy files found in ${DATA_DIR} with suffix ${SUFFIX}"
    exit 1
fi
echo "  Found ${NPY_COUNT} .npy files in ${DATA_DIR}"

# --- Verify splits ---
echo ""
echo "[Step 1] Checking split files..."
if [ ! -f "splits/8AA_train.csv" ] || [ ! -f "splits/8AA_val.csv" ]; then
    echo "ERROR: splits/8AA_train.csv or splits/8AA_val.csv not found"
    echo "Run generate_8AA_splits.py first:"
    echo "  python -m scripts.generate_8AA_splits \\"
    echo "    --data_dir /localhome3/lyy/octapeptides_data \\"
    echo "    --outdir splits \\"
    echo "    --npy_dir ${DATA_DIR} \\"
    echo "    --suffix ${SUFFIX}"
    exit 1
fi
TRAIN_COUNT=$(tail -n +2 splits/8AA_train.csv | wc -l)
VAL_COUNT=$(tail -n +2 splits/8AA_val.csv | wc -l)
echo "  Splits: train=${TRAIN_COUNT}, val=${VAL_COUNT}"

# --- Train ---
echo ""
echo "[Step 2] Starting training..."
echo "  Run name:        ${RUN_NAME}"
echo "  GPUs:            ${CUDA_VISIBLE_DEVICES} (8 cards)"
echo "  Per-GPU batch:   ${BATCH_SIZE}"
echo "  Effective batch: $((BATCH_SIZE * 8))"
echo "  Epochs:          ${EPOCHS}"
echo "  LR:              ${LR}"
echo "  Precision:       ${PRECISION}"
echo "  Ckpt freq:       ${CKPT_FREQ}"
echo ""

mkdir -p "workdir/${RUN_NAME}"

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
echo "  Phase 1 Training complete!"
echo "  Checkpoints: workdir/${RUN_NAME}/"
echo "  Plot loss:   python plot_loss.py workdir/${RUN_NAME}/log.out --save"
echo "============================================"
