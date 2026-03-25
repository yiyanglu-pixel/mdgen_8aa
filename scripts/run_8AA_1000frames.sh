#!/bin/bash
# =============================================================================
# 8AA Training — 1000 frames per sample (high-resolution trajectory)
# =============================================================================
# Uses longer trajectories (1000 frames) vs the default 100-frame job.
# Heavier per-sample cost, so batch_size is reduced to 4 per GPU.
#
# Usage:
#   bash scripts/run_8AA_1000frames.sh
# =============================================================================

set -e

export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
RAW_DATA_DIR="/localhome3/lyy/octapeptides_data"
SUFFIX="_i100"
RUN_NAME="8AA_sim_1000frames"
NUM_FRAMES=1000
EPOCHS=2000
CKPT_FREQ=50

# Multi-GPU: GPUs 1-7 (7 cards), leave GPU 0 free
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
NUM_GPUS=7
# bs=2 per GPU → effective batch = 2*7 = 14
# LR sqrt-scaled from (bs=8, lr=1e-4): 1e-4 * sqrt(14/56) = 5e-5
BATCH_SIZE=2
LR=5e-5
PRECISION="bf16-mixed"

USE_WANDB=""  # set to "--wandb" to enable
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  8AA Training — 1000 frames (${EPOCHS} epochs)"
echo "  ${NUM_GPUS} GPUs, bs=${BATCH_SIZE}, ${PRECISION}"
echo "  Effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo "============================================"

# --- Step 0: Verify data ---
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
if [ "$NPY_COUNT" -eq 0 ]; then
    echo "ERROR: No .npy files in ${DATA_DIR} with suffix ${SUFFIX}"
    echo "Run prep_sims.py first."
    exit 1
fi
echo "  Found ${NPY_COUNT} .npy files"

# Verify at least one file has enough frames for 1000-frame sampling
SAMPLE_FILE=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | head -1)
SAMPLE_FRAMES=$(python -c "import numpy as np; a=np.lib.format.open_memmap('${SAMPLE_FILE}','r'); print(a.shape[0])")
if [ "$SAMPLE_FRAMES" -lt "$NUM_FRAMES" ]; then
    echo "ERROR: Sample file has only ${SAMPLE_FRAMES} frames, need >= ${NUM_FRAMES}"
    exit 1
fi
echo "  Frame check OK: sample file has ${SAMPLE_FRAMES} frames"

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

# --- Step 2: Train ---
echo ""
echo "  Run name:        ${RUN_NAME}"
echo "  Num frames:      ${NUM_FRAMES}"
echo "  GPUs:            ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} cards)"
echo "  Per-GPU batch:   ${BATCH_SIZE}"
echo "  Effective batch: $((BATCH_SIZE * NUM_GPUS))"
echo "  LR:              ${LR}"
echo "  Precision:       ${PRECISION}"
echo "  Ckpt freq:       every ${CKPT_FREQ} epochs"
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
echo "  Training complete! (${EPOCHS} epochs)"
echo "  Checkpoints: ${MODEL_DIR}"
echo ""
echo "  Inference:"
echo "    python sim_inference.py --sim_ckpt ${MODEL_DIR}/best.ckpt \\"
echo "        --data_dir ${DATA_DIR} --split splits/8AA_test.csv \\"
echo "        --num_frames ${NUM_FRAMES} --num_rollouts 10 --suffix ${SUFFIX} \\"
echo "        --xtc --out_dir results/8AA_1000frames_test"
echo "============================================"
