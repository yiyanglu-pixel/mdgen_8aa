#!/bin/bash
# Run 8AA forward simulation inference
#
# Usage:
#   bash configs/infer_8AA_sim.sh
#
# Custom checkpoint:
#   CKPT=workdir/8AA_sim_phase2/best.ckpt bash configs/infer_8AA_sim.sh

set -e

# Environment setup (avoid GLIBCXX and PYTHONPATH conflicts)
export MPI4PY_RC_INITIALIZE=0
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/apps/' | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

SUFFIX=${SUFFIX:-_i100}
CKPT=${CKPT:-workdir/8AA_sim/best.ckpt}

python sim_inference.py \
    --sim_ckpt $CKPT \
    --data_dir data/8AA_data \
    --split splits/8AA_test.csv \
    --num_frames 100 \
    --num_rollouts 10 \
    --suffix $SUFFIX \
    --xtc \
    --out_dir results/8AA_sim
