#!/bin/bash
# Run 8AA forward simulation inference
#
# SUFFIX must match preprocessing/training:
#   10ns high-freq data: --suffix _i1000
#   100ns production data: --suffix _i100

SUFFIX=${SUFFIX:-_i1000}  # default to 10ns test data

python sim_inference.py \
    --sim_ckpt workdir/8AA_sim/best.ckpt \
    --data_dir data/8AA_data \
    --split splits/8AA_test.csv \
    --num_frames 100 \
    --num_rollouts 10 \
    --suffix $SUFFIX \
    --xtc \
    --out_dir results/8AA_sim
