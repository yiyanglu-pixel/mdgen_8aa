#!/bin/bash
# =============================================================================
# 8AA Short Test Training (50 epochs) - Debug NaN issue
# =============================================================================
# Uses default params (lr=1e-4, batch_size=8) to verify loss doesn't go NaN.
# If this also produces NaN, the bug is in the code (t=0 division), not hyperparams.
#
# Usage:
#   bash scripts/run_8AA_train_test50.sh
# =============================================================================

set -e

# Prevent system mpi4py (Python 3.12) from conflicting with conda env (Python 3.9)
export MPI4PY_RC_INITIALIZE=0
# Remove /apps paths that inject incompatible system packages
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
SUFFIX="_i1000"
RUN_NAME="8AA_sim_test50"
NUM_FRAMES=100
EPOCHS=50
BATCH_SIZE=8
LR=1e-4
CKPT_FREQ=10
export CUDA_VISIBLE_DEVICES=0
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  8AA Test Training (50 epochs, default params)"
echo "  Purpose: Verify loss doesn't go NaN"
echo "============================================"

# Verify data
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
if [ "$NPY_COUNT" -eq 0 ]; then
    echo "ERROR: No .npy files found in ${DATA_DIR} with suffix ${SUFFIX}"
    exit 1
fi
echo "  Found ${NPY_COUNT} .npy files"

# Verify splits exist
if [ ! -f "splits/8AA_train.csv" ] || [ ! -f "splits/8AA_val.csv" ]; then
    echo "ERROR: splits/8AA_train.csv or splits/8AA_val.csv not found"
    echo "Run scripts/run_8AA_train.sh first to generate splits"
    exit 1
fi

export MODEL_DIR="workdir/${RUN_NAME}"
mkdir -p "${MODEL_DIR}"

echo ""
echo "  Run name:   ${RUN_NAME}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  Epochs:     ${EPOCHS}"
echo "  Output:     ${MODEL_DIR}"
echo ""

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
    --run_name "${RUN_NAME}"

echo ""
echo "============================================"
echo "  Test training complete!"
echo "  Check loss: python plot_loss.py ${MODEL_DIR}/log.out --save"
echo "============================================"
