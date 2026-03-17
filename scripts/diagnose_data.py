"""Comprehensive data diagnostics for 8AA .npy files.

Checks for:
1. NaN / inf values
2. Extreme coordinate values (potential float16 overflow)
3. Degenerate backbone geometry (N/CA/C collinear or coincident)
4. Zero-coordinate atoms that should have valid positions
5. Per-frame statistics

Usage:
    python scripts/diagnose_data.py \
        --data_dir /localhome3/lyy/mdgen_8aa/data/8AA_data \
        --splits_dir splits \
        --suffix _i1000
"""
import argparse
import os
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--splits_dir', type=str, default='splits')
parser.add_argument('--suffix', type=str, default='_i1000')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# atom14 indices: N=0, CA=1, C=2, O=3
N_IDX, CA_IDX, C_IDX, O_IDX = 0, 1, 2, 3

files = sorted(glob.glob(f'{args.data_dir}/opep_*{args.suffix}.npy'))
print(f'Scanning {len(files)} files...\n')

issues = {
    'has_nan': [],
    'has_inf': [],
    'extreme_coords': [],       # coords > 500 Å (likely simulation blowup)
    'zero_backbone': [],        # N/CA/C at (0,0,0)
    'degenerate_backbone': [],  # N/CA/C nearly collinear (cross product < threshold)
    'bad_frames': {},           # per-file: list of bad frame indices
}

for f in files:
    name = os.path.basename(f).replace(f'{args.suffix}.npy', '')
    arr = np.load(f)  # (T, L, 14, 3)
    T, L, A, D = arr.shape

    file_issues = []

    # 1. NaN check
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        n_nan_frames = np.any(nan_mask.reshape(T, -1), axis=1).sum()
        issues['has_nan'].append((name, int(np.sum(nan_mask)), n_nan_frames, T))
        file_issues.append('nan')

    # 2. Inf check
    inf_mask = np.isinf(arr)
    if np.any(inf_mask):
        n_inf_frames = np.any(inf_mask.reshape(T, -1), axis=1).sum()
        issues['has_inf'].append((name, int(np.sum(inf_mask)), n_inf_frames, T))
        file_issues.append('inf')

    # Work with finite values for remaining checks
    arr_clean = np.where(np.isfinite(arr), arr, 0.0)

    # 3. Extreme coordinate values (> 500 Å, suggests simulation blowup)
    max_abs = np.max(np.abs(arr_clean))
    if max_abs > 500:
        # Find which frames have extreme values
        frame_max = np.max(np.abs(arr_clean).reshape(T, -1), axis=1)
        bad_frame_count = int(np.sum(frame_max > 500))
        issues['extreme_coords'].append((name, float(max_abs), bad_frame_count, T))
        file_issues.append(f'extreme(max={max_abs:.0f})')

    # 4. Zero backbone atoms (N, CA, C should never be at origin)
    backbone = arr_clean[:, :, :3, :]  # (T, L, 3, 3) - N, CA, C
    backbone_norms = np.linalg.norm(backbone, axis=-1)  # (T, L, 3)
    zero_backbone = backbone_norms < 1e-6  # atoms essentially at origin
    if np.any(zero_backbone):
        n_zero_frames = np.any(zero_backbone.reshape(T, -1), axis=1).sum()
        which_atoms = []
        for a, aname in enumerate(['N', 'CA', 'C']):
            if np.any(zero_backbone[:, :, a]):
                which_atoms.append(aname)
        issues['zero_backbone'].append((name, ','.join(which_atoms), int(n_zero_frames), T))
        file_issues.append(f'zero_backbone({",".join(which_atoms)})')

    # 5. Degenerate backbone geometry: N-CA-C angle near 0° or 180°
    #    Cross product ||(CA-C) × (CA-N)|| should be > threshold
    ca = arr_clean[:, :, CA_IDX, :]  # (T, L, 3)
    n = arr_clean[:, :, N_IDX, :]
    c = arr_clean[:, :, C_IDX, :]

    v1 = c - ca   # CA→C vector
    v2 = n - ca   # CA→N vector
    cross = np.cross(v1, v2)  # (T, L, 3)
    cross_norm = np.linalg.norm(cross, axis=-1)  # (T, L)
    v1_norm = np.linalg.norm(v1, axis=-1)
    v2_norm = np.linalg.norm(v2, axis=-1)

    # sin(angle) = |cross| / (|v1| * |v2|)
    denom = v1_norm * v2_norm
    denom = np.where(denom > 1e-8, denom, 1.0)  # avoid division by zero
    sin_angle = cross_norm / denom

    # Degenerate if sin(angle) < 0.01 (~0.6°) - backbone should have ~111° angle
    degenerate = sin_angle < 0.01
    if np.any(degenerate):
        n_degen_frames = np.any(degenerate, axis=1).sum()
        min_sin = float(np.min(sin_angle))
        issues['degenerate_backbone'].append((name, float(min_sin), int(n_degen_frames), T))
        file_issues.append(f'degenerate(min_sin={min_sin:.6f})')

    # Track per-file bad frames
    if file_issues:
        bad_frame_mask = np.zeros(T, dtype=bool)
        if np.any(nan_mask):
            bad_frame_mask |= np.any(nan_mask.reshape(T, -1), axis=1)
        if np.any(inf_mask):
            bad_frame_mask |= np.any(inf_mask.reshape(T, -1), axis=1)
        frame_max = np.max(np.abs(arr_clean).reshape(T, -1), axis=1)
        bad_frame_mask |= frame_max > 500
        if np.any(zero_backbone):
            bad_frame_mask |= np.any(zero_backbone.reshape(T, -1), axis=1)
        if np.any(degenerate):
            bad_frame_mask |= np.any(degenerate, axis=1)

        bad_indices = np.where(bad_frame_mask)[0]
        issues['bad_frames'][name] = {
            'count': int(len(bad_indices)),
            'total': T,
            'issues': file_issues,
            'first_bad': int(bad_indices[0]) if len(bad_indices) > 0 else -1,
            'last_bad': int(bad_indices[-1]) if len(bad_indices) > 0 else -1,
        }

# Print results
print("=" * 80)
print("DATA DIAGNOSTIC REPORT")
print("=" * 80)

print(f"\n--- Files with NaN ({len(issues['has_nan'])}) ---")
for name, count, n_frames, total in issues['has_nan']:
    print(f"  {name}: {count} NaN values in {n_frames}/{total} frames")

print(f"\n--- Files with Inf ({len(issues['has_inf'])}) ---")
for name, count, n_frames, total in issues['has_inf']:
    print(f"  {name}: {count} inf values in {n_frames}/{total} frames")

print(f"\n--- Files with extreme coords >500Å ({len(issues['extreme_coords'])}) ---")
for name, maxval, n_frames, total in issues['extreme_coords']:
    print(f"  {name}: max={maxval:.1f}Å in {n_frames}/{total} frames")

print(f"\n--- Files with zero backbone atoms ({len(issues['zero_backbone'])}) ---")
for name, atoms, n_frames, total in issues['zero_backbone']:
    print(f"  {name}: zero {atoms} in {n_frames}/{total} frames")

print(f"\n--- Files with degenerate backbone (<0.6° N-CA-C) ({len(issues['degenerate_backbone'])}) ---")
for name, min_sin, n_frames, total in sorted(issues['degenerate_backbone'], key=lambda x: x[1]):
    print(f"  {name}: min_sin={min_sin:.6f} in {n_frames}/{total} frames")

print(f"\n--- Summary ---")
all_bad = set()
for cat in ['has_nan', 'has_inf', 'extreme_coords', 'zero_backbone', 'degenerate_backbone']:
    names = [x[0] for x in issues[cat]]
    all_bad.update(names)
print(f"Total files with ANY issue: {len(all_bad)} / {len(files)}")
print(f"Bad file names: {sorted(all_bad)}")

# Check which split they're in
import pandas as pd
for split_name in ['8AA_train', '8AA_val']:
    path = os.path.join(args.splits_dir, f'{split_name}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, index_col='name')
    in_split = [n for n in all_bad if n in df.index]
    print(f"  In {split_name}: {len(in_split)} - {in_split}")

print(f"\n--- Per-file details ---")
for name, info in sorted(issues['bad_frames'].items()):
    print(f"  {name}: {info['count']}/{info['total']} bad frames "
          f"(range {info['first_bad']}-{info['last_bad']}), issues: {info['issues']}")
