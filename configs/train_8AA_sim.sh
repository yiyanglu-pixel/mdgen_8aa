#!/bin/bash
# Train 8AA forward simulation model
# Assumes splits and preprocessed data are already generated
#
# SUFFIX must match preprocessing:
#   10ns high-freq data (stride=1000): --suffix _i1000
#   100ns production data (stride=100): --suffix _i100

SUFFIX=${SUFFIX:-_i1000}  # default to 10ns test data

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
