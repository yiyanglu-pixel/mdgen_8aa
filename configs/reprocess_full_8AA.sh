#!/bin/bash
# Full 8AA dataset reprocessing pipeline
#
# Generates splits for all 1100 octapeptides and preprocesses them from scratch.
# Uses AMBER prmtop (not topology.pdb) for both split generation and preprocessing,
# ensuring consistent atom sets between topology and XTC trajectories.
#
# Usage:
#   bash configs/reprocess_full_8AA.sh
#
# Optional overrides:
#   SIM_DIR       path to ONE_octapeptides  (default: /localhome3/lyy/8pep_gb_sim/octapeptides_data/ONE_octapeptides)
#   OUTDIR        output dir for .npy files  (default: data/8AA_data)
#   SUFFIX        .npy filename suffix       (default: _i1000)
#   STRIDE        frame stride              (default: 1000)
#   NUM_WORKERS   parallel workers          (default: 8)

set -e

SIM_DIR=${SIM_DIR:-/localhome3/lyy/8pep_gb_sim/octapeptides_data/ONE_octapeptides}
OUTDIR=${OUTDIR:-data/8AA_data}
SUFFIX=${SUFFIX:-_i1000}
STRIDE=${STRIDE:-1000}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "=== Step 1: Generate splits for all 1100 octapeptides ==="
echo "  SIM_DIR     = $SIM_DIR"
echo "  OUTDIR      = $OUTDIR"
echo "  SUFFIX      = $SUFFIX"
echo "  STRIDE      = $STRIDE"
echo "  NUM_WORKERS = $NUM_WORKERS"
echo ""

# Generate splits from prmtop files (no --npy_dir filter → all 1100 included)
python -m scripts.generate_8AA_splits \
    --data_dir "$SIM_DIR" \
    --outdir splits

echo ""
echo "=== Step 2: Preprocess all octapeptides → .npy ==="
mkdir -p "$OUTDIR"

python -m scripts.prep_sims \
    --split splits/8AA.csv \
    --sim_dir "$SIM_DIR" \
    --outdir "$OUTDIR" \
    --num_workers "$NUM_WORKERS" \
    --suffix "$SUFFIX" \
    --stride "$STRIDE" \
    --octapeptides

echo ""
echo "=== Step 3: Verify preprocessed data ==="
python scripts/verify_data.py \
    --data_dir "$OUTDIR" \
    --suffix "$SUFFIX"

echo ""
echo "=== Step 4: Regenerate splits filtered to successfully preprocessed files ==="
python -m scripts.generate_8AA_splits \
    --data_dir "$SIM_DIR" \
    --outdir splits \
    --npy_dir "$OUTDIR" \
    --suffix "$SUFFIX"

echo ""
echo "=== Done ==="
echo "Split files: splits/8AA.csv  splits/8AA_train.csv  splits/8AA_val.csv  splits/8AA_test.csv"
echo "Data files:  $OUTDIR/opep_XXXX${SUFFIX}.npy"
