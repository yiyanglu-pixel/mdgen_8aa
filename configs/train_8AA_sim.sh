#!/bin/bash
# Train 8AA forward simulation model
# Assumes splits and preprocessed data are already generated
#
# Usage:
#   bash configs/train_8AA_sim.sh

set -e

# Environment setup (avoid GLIBCXX and PYTHONPATH conflicts)
export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

SUFFIX=${SUFFIX:-_i100}

MODEL_DIR=workdir/8AA_sim python train.py \
    --sim_condition \
    --train_split splits/8AA_train.csv \
    --val_split splits/8AA_val.csv \
    --data_dir data/8AA_data \
    --crop 8 \
    --abs_pos_emb \
    --prepend_ipa \
    --num_frames 100 \
    --batch_size 8 \
    --suffix $SUFFIX \
    --ckpt_freq 40 \
    --val_repeat 25 \
    --epochs 1000 \
    --wandb \
    --run_name 8AA_sim
