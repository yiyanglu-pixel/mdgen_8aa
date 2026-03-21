#!/bin/bash
# Run 8AA forward simulation inference

SUFFIX=${SUFFIX:-_i100}

python sim_inference.py \
    --sim_ckpt workdir/8AA_sim/best.ckpt \
    --data_dir data/8AA_data \
    --split splits/8AA_test.csv \
    --num_frames 100 \
    --num_rollouts 10 \
    --suffix $SUFFIX \
    --xtc \
    --out_dir results/8AA_sim
