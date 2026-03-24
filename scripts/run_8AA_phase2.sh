#!/bin/bash
# =============================================================================
# Phase 2: 8AA Training — 10000 epochs (resume from Phase 1 checkpoint)
# =============================================================================
# Resumes from Phase 1 best checkpoint and continues to 10000 epochs.
# PyTorch Lightning automatically resumes optimizer state, LR scheduler, etc.
#
# Usage:
#   bash scripts/run_8AA_phase2.sh          # single GPU
#   bash scripts/run_8AA_phase2.sh --multi  # multi-GPU (GPUs 1-7)
#
# Custom checkpoint:
#   CKPT=workdir/8AA_sim_phase1/epoch=1999.ckpt bash scripts/run_8AA_phase2.sh
# =============================================================================

set -e

export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')
# Use conda's libstdc++ to avoid GLIBCXX version mismatch with system /lib64
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
SUFFIX="_i100"
RUN_NAME="8AA_sim_phase2"
NUM_FRAMES=100
EPOCHS=10000
CKPT_FREQ=200

# Phase 1 checkpoint to resume from
PHASE1_DIR="workdir/8AA_sim_phase1"

# Single-GPU defaults
BATCH_SIZE=8
LR=1e-4
PRECISION="32-true"
export CUDA_VISIBLE_DEVICES=0

# Multi-GPU override
if [[ "$1" == "--multi" ]]; then
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    NUM_GPUS=7
    BATCH_SIZE=16
    LR=3.7e-4
    PRECISION="bf16-mixed"
    RUN_NAME="8AA_sim_phase2_multi"
    PHASE1_DIR="workdir/8AA_sim_phase1_multi"
    echo "Multi-GPU mode: ${NUM_GPUS} GPUs, effective batch=$((BATCH_SIZE * NUM_GPUS))"
fi

USE_WANDB=""  # set to "--wandb" to enable

# Find checkpoint: use CKPT env var, or last epoch checkpoint from Phase 1
if [ -z "${CKPT}" ]; then
    CKPT=$(ls -t "${PHASE1_DIR}"/epoch=*.ckpt 2>/dev/null | head -1)
    if [ -z "${CKPT}" ]; then
        echo "ERROR: No checkpoint found in ${PHASE1_DIR}"
        echo "Run Phase 1 first: bash scripts/run_8AA_phase1.sh"
        exit 1
    fi
fi
echo "Resuming from: ${CKPT}"
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  Phase 2: 8AA Training (${EPOCHS} epochs)"
echo "  Resuming from Phase 1 checkpoint"
echo "============================================"

# Verify data and splits
NPY_COUNT=$(ls "${DATA_DIR}/"*"${SUFFIX}.npy" 2>/dev/null | wc -l)
echo "  .npy files:  ${NPY_COUNT}"

if [ ! -f "splits/8AA_train.csv" ]; then
    echo "ERROR: splits/8AA_train.csv not found. Run Phase 1 first."
    exit 1
fi
TRAIN_COUNT=$(tail -n +2 splits/8AA_train.csv | wc -l)
VAL_COUNT=$(tail -n +2 splits/8AA_val.csv | wc -l)
echo "  Splits:      train=${TRAIN_COUNT}, val=${VAL_COUNT}"

echo ""
echo "  Run name:    ${RUN_NAME}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  LR:          ${LR}"
echo "  Precision:   ${PRECISION}"
echo "  Resume from: ${CKPT}"
echo "  Ckpt freq:   every ${CKPT_FREQ} epochs"
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
    --ckpt "${CKPT}" \
    --run_name "${RUN_NAME}" \
    ${USE_WANDB}

echo ""
echo "============================================"
echo "  Phase 2 complete! (${EPOCHS} epochs)"
echo "  Checkpoints: ${MODEL_DIR}"
echo ""
echo "  Next steps:"
echo "    1. Check loss:  python plot_loss.py ${MODEL_DIR}/log.out --save"
echo "    2. Quick test:  python sim_inference.py --sim_ckpt ${MODEL_DIR}/best.ckpt \\"
echo "           --data_dir ${DATA_DIR} --split splits/8AA_test.csv \\"
echo "           --num_frames 100 --num_rollouts 10 --suffix ${SUFFIX} \\"
echo "           --xtc --out_dir results/8AA_phase2_test"
echo "    3. Full eval:   python sim_inference.py --sim_ckpt ${MODEL_DIR}/best.ckpt \\"
echo "           --data_dir ${DATA_DIR} --split splits/8AA_test.csv \\"
echo "           --num_frames 1000 --num_rollouts 10 --suffix ${SUFFIX} \\"
echo "           --xtc --out_dir results/8AA_phase2_full"
echo "    4. Analysis:    python scripts/analyze_8AA_sim.py \\"
echo "           --pdbdir results/8AA_phase2_full \\"
echo "           --mddir /localhome3/lyy/octapeptides_data \\"
echo "           --split splits/8AA_test.csv --save --plot"
echo "============================================"
