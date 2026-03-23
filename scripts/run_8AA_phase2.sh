#!/bin/bash
# =============================================================================
# 8AA Phase 2 Training: Resume from Phase 1 (2000ep) to 12000 total epochs
# =============================================================================
# Prerequisites:
#   - Phase 1 training completed (workdir/8AA_sim_phase1/ has checkpoints)
#
# Usage:
#   bash scripts/run_8AA_phase2.sh
#
# Or run in background:
#   nohup bash scripts/run_8AA_phase2.sh > workdir/8AA_sim_phase1/train_phase2.log 2>&1 &
# =============================================================================

set -e

# Prevent system mpi4py (Python 3.12) from conflicting with conda env (Python 3.9)
export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')

# ========================= USER CONFIG =========================
DATA_DIR="/localhome3/lyy/mdgen_8aa/data/8AA_data"
SUFFIX="_i1000"
RUN_NAME="8AA_sim_phase1"    # Same run_name as Phase 1 to keep logs continuous
NUM_FRAMES=100

# 8 GPUs (0-7) - must match Phase 1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# All hyperparams must match Phase 1
BATCH_SIZE=4
LR=2e-4
PRECISION="32-true"

# TOTAL target epochs (not additional)
# PyTorch Lightning resumes from checkpoint epoch and continues to this number
EPOCHS=12000

# Save checkpoint every 200 epochs (50 new checkpoints for 10000 additional epochs)
CKPT_FREQ=200

USE_WANDB=""  # set to "--wandb" to enable W&B logging
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  8AA Phase 2 Training (resume to 12000 epochs)"
echo "============================================"

# --- Find the last Phase 1 checkpoint ---
echo ""
echo "[Step 0] Finding last checkpoint..."
CKPT_DIR="workdir/${RUN_NAME}"
if [ ! -d "${CKPT_DIR}" ]; then
    echo "ERROR: ${CKPT_DIR} not found. Run Phase 1 first."
    exit 1
fi

CKPT=$(ls "${CKPT_DIR}"/epoch=*.ckpt 2>/dev/null | sort -t= -k2 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoints found in ${CKPT_DIR}"
    exit 1
fi
echo "  Resuming from: ${CKPT}"

# --- Verify splits ---
echo ""
echo "[Step 1] Checking split files..."
if [ ! -f "splits/8AA_train.csv" ] || [ ! -f "splits/8AA_val.csv" ]; then
    echo "ERROR: splits/8AA_train.csv or splits/8AA_val.csv not found"
    exit 1
fi
TRAIN_COUNT=$(tail -n +2 splits/8AA_train.csv | wc -l)
VAL_COUNT=$(tail -n +2 splits/8AA_val.csv | wc -l)
echo "  Splits: train=${TRAIN_COUNT}, val=${VAL_COUNT}"

# --- Resume training ---
echo ""
echo "[Step 2] Resuming training..."
echo "  Run name:        ${RUN_NAME}"
echo "  Resume ckpt:     ${CKPT}"
echo "  GPUs:            ${CUDA_VISIBLE_DEVICES} (8 cards)"
echo "  Per-GPU batch:   ${BATCH_SIZE}"
echo "  Effective batch: $((BATCH_SIZE * 8))"
echo "  Total epochs:    ${EPOCHS}"
echo "  LR:              ${LR}"
echo "  Precision:       ${PRECISION}"
echo "  Ckpt freq:       ${CKPT_FREQ}"
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
    --precision ${PRECISION} \
    --ckpt_freq ${CKPT_FREQ} \
    --ckpt "${CKPT}" \
    --run_name "${RUN_NAME}" \
    ${USE_WANDB}

echo ""
echo "============================================"
echo "  Phase 2 Training complete! (12000 total epochs)"
echo "  Checkpoints: workdir/${RUN_NAME}/"
echo "  Plot loss:   python plot_loss.py workdir/${RUN_NAME}/log.out --save"
echo ""
echo "  Next steps:"
echo "    1. Find best checkpoint:"
echo "       python -c \"import re; best_e,best_l=-1,float('inf')"
echo "       with open('workdir/${RUN_NAME}/log.out') as f:"
echo "         for l in f:"
echo "           m=re.search(r\\\"'epoch': (\d+).*'val_loss': ([0-9.e+-]+)\\\",l)"
echo "           if m:"
echo "             e,v=int(m.group(1)),float(m.group(2))"
echo "             if v<best_l: best_e,best_l=e,v"
echo "       print(f'Best epoch: {best_e}, val_loss: {best_l:.6f}')\""
echo ""
echo "    2. Create best.ckpt symlink:"
echo "       ln -sf 'epoch=XXXX-step=YYYY.ckpt' workdir/${RUN_NAME}/best.ckpt"
echo ""
echo "    3. Run inference:"
echo "       python sim_inference.py --sim_ckpt workdir/${RUN_NAME}/best.ckpt \\"
echo "         --data_dir ${DATA_DIR} --split splits/8AA_test.csv \\"
echo "         --num_frames 100 --num_rollouts 10 --suffix ${SUFFIX} \\"
echo "         --xtc --out_dir results/8AA_12000ep"
echo ""
echo "    4. Run analysis:"
echo "       python scripts/analyze_8AA_sim.py \\"
echo "         --pdbdir results/8AA_12000ep \\"
echo "         --mddir /localhome3/lyy/octapeptides_data \\"
echo "         --split splits/8AA_test.csv --save --plot"
echo "============================================"
