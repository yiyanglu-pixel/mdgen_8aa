#!/usr/bin/env python
"""Diagnose NaN/inf in 8AA .npy data files.

Usage:
    python scripts/diagnose_npy_data.py --data_dir /localhome3/lyy/mdgen_8aa/data/8AA_data --suffix _i100
"""
import argparse
import glob
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--suffix', default='_i100')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

files = sorted(glob.glob(f'{args.data_dir}/*{args.suffix}.npy'))
print(f"Checking {len(files)} .npy files in {args.data_dir}\n")

total_files = 0
nan_files = 0
inf_files = 0
dtype_issues = 0
coord_range_issues = 0

for f in files:
    arr = np.load(f, mmap_mode='r')
    total_files += 1
    name = os.path.basename(f).replace(args.suffix + '.npy', '')
    issues = []

    if arr.dtype == np.float16:
        pass  # expected
    else:
        issues.append(f"dtype={arr.dtype}")
        dtype_issues += 1

    arr32 = arr.astype(np.float32)

    n_nan = np.isnan(arr32).sum()
    if n_nan > 0:
        issues.append(f"NaN count={n_nan}")
        nan_files += 1
        # Which frames/residues/atoms have NaN?
        if args.verbose:
            nan_frames = np.any(np.isnan(arr32), axis=(1, 2, 3))
            issues.append(f"NaN in {nan_frames.sum()}/{arr32.shape[0]} frames")

    n_inf = np.isinf(arr32).sum()
    if n_inf > 0:
        issues.append(f"inf count={n_inf}")
        inf_files += 1

    # Check backbone atoms (N=0, CA=1, C=2) for zero coordinates
    backbone = arr32[:, :, :3, :]  # (frames, residues, 3_backbone_atoms, 3_xyz)
    zero_backbone = np.all(backbone == 0, axis=-1)  # (frames, residues, 3)
    n_zero_bb = zero_backbone.any(axis=0).sum()  # any frame with zero backbone
    if n_zero_bb > 0:
        issues.append(f"zero backbone atoms={n_zero_bb}")

    # Check coordinate range
    valid = arr32[~np.isnan(arr32) & ~np.isinf(arr32)]
    if len(valid) > 0:
        vmin, vmax = valid.min(), valid.max()
        if abs(vmax) > 1000 or abs(vmin) > 1000:
            issues.append(f"extreme coords: [{vmin:.1f}, {vmax:.1f}]")
            coord_range_issues += 1

    if issues:
        print(f"  {name}: {', '.join(issues)}")

print(f"\n{'='*60}")
print(f"Summary:")
print(f"  Total files:          {total_files}")
print(f"  Files with NaN:       {nan_files}")
print(f"  Files with inf:       {inf_files}")
print(f"  Files with dtype!=f16:{dtype_issues}")
print(f"  Extreme coordinates:  {coord_range_issues}")
print(f"{'='*60}")

if nan_files == 0 and inf_files == 0:
    print("\nNo NaN/inf in .npy files.")
    print("NaN likely introduced during frame/torsion computation from zero-coordinate atoms.")
    print("Check: does atom14_to_frames get (0,0,0) backbone atoms?")
