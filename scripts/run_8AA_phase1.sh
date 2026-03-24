#!/bin/bash
# =============================================================================
# Phase 1: 8AA Training — 2000 epochs (initial training)
# =============================================================================
# Run this first. After loss converges, proceed to Phase 2 (10000 epochs).
#
# Usage:
#   bash scripts/run_8AA_phase1.sh          # single GPU
#   bash scripts/run_8AA_phase1.sh --multi  # multi-GPU (GPUs 1-7)
# =============================================================================

set -e

export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')
# Use conda's libstdc++ to avoid GLIBCXX version mismatch with system /lib64
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
RAW_DATA_DIR="/localhome3/lyy/octapeptides_data"
SUFFIX="_i100"
RUN_NAME="8AA_sim_phase1"
NUM_FRAMES=100
EPOCHS=2000
CKPT_FREQ=50

# Single-GPU defaults
BATCH_SIZE=2
LR=1e-4
PRECISION="32-true"
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU override
if [[ "$1" == "--multi" ]]; then
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    NUM_GPUS=7
    BATCH_SIZE=16
    # sqrt-scaled LR: sqrt(16*7 / 8) * 1e-4 ≈ 3.7e-4
    LR=3.7e-4
    PRECISION="bf16-mixed"
    RUN_NAME="8AA_sim_phase1_multi"
    echo "Multi-GPU mode: ${NUM_GPUS} GPUs, effective batch=$((BATCH_SIZE * NUM_GPUS))"
fi

USE_WANDB=""  # set to "--wandb" to enable
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Phase 1: 8AA Training (${EPOCHS} epochs)"
echo "============================================"

# --- Step 0: Verify data ---
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
if [ "$NPY_COUNT" -eq 0 ]; then
    echo "ERROR: No .npy files in ${DATA_DIR} with suffix ${SUFFIX}"
    echo "Run prep_sims.py first."
    exit 1
fi
echo "  Found ${NPY_COUNT} .npy files"

# --- Step 1: Generate splits if needed ---
if [ -f "splits/8AA_train.csv" ] && [ -f "splits/8AA_val.csv" ]; then
    TRAIN_COUNT=$(tail -n +2 splits/8AA_train.csv | wc -l)
    VAL_COUNT=$(tail -n +2 splits/8AA_val.csv | wc -l)
    echo "  Splits: train=${TRAIN_COUNT}, val=${VAL_COUNT}"
else
    echo "  Generating splits..."
    python -m scripts.generate_8AA_splits \
        --data_dir "${RAW_DATA_DIR}" \
        --outdir splits \
        --train_frac 0.8 \
        --val_frac 0.1 \
        --test_frac 0.1

    # Validate splits against available .npy files
    python -c "
import pandas as pd, os
suffix, data_dir = '${SUFFIX}', '${DATA_DIR}'
for s in ['8AA_train', '8AA_val', '8AA_test']:
    df = pd.read_csv(f'splits/{s}.csv', index_col='name')
    missing = [n for n in df.index if not os.path.exists(f'{data_dir}/{n}{suffix}.npy')]
    if missing:
        df = df.loc[[n for n in df.index if os.path.exists(f'{data_dir}/{n}{suffix}.npy')]]
        df.to_csv(f'splits/{s}.csv')
        print(f'  {s}: filtered to {len(df)} (removed {len(missing)} missing)')
    else:
        print(f'  {s}: {len(df)} entries - all OK')
"
fi

# --- Step 2: Train Phase 1 ---
echo ""
echo "  Run name:   ${RUN_NAME}"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  LR:         ${LR}"
echo "  Precision:  ${PRECISION}"
echo "  Ckpt freq:  every ${CKPT_FREQ} epochs"
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
echo "  Phase 1 complete! (${EPOCHS} epochs)"
echo "  Checkpoints: ${MODEL_DIR}"
echo ""
echo "  Next steps:"
echo "    1. Check loss:  python plot_loss.py ${MODEL_DIR}/log.out --save"
echo "    2. Quick infer: python sim_inference.py --sim_ckpt ${MODEL_DIR}/best.ckpt \\"
echo "           --data_dir ${DATA_DIR} --split splits/8AA_test.csv \\"
echo "           --num_frames 100 --num_rollouts 10 --suffix ${SUFFIX} \\"
echo "           --xtc --out_dir results/8AA_phase1_test"
echo "    3. If loss OK:  bash scripts/run_8AA_phase2.sh"
echo "============================================"
