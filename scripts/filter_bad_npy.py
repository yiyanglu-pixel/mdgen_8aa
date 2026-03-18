"""Filter out .npy files containing NaN/inf from split CSVs.

Usage:
    python -m scripts.filter_bad_npy \
        --data_dir /localhome3/lyy/mdgen_8aa/data/8AA_data \
        --splits_dir splits \
        --suffix _i1000
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Filter bad .npy files from split CSVs')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--splits_dir', type=str, default='splits')
parser.add_argument('--suffix', type=str, default='_i1000')
parser.add_argument('--dry_run', action='store_true', help='Only report, do not modify CSVs')
args = parser.parse_args()

# Step 1: Scan all .npy files for NaN/inf
files = sorted(glob.glob(f'{args.data_dir}/opep_*{args.suffix}.npy'))
print(f'Scanning {len(files)} .npy files for NaN/inf...')

bad_names = set()
for f in files:
    arr = np.load(f)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        name = os.path.basename(f).replace(f'{args.suffix}.npy', '')
        n_nan = int(np.sum(np.isnan(arr)))
        n_inf = int(np.sum(np.isinf(arr)))
        total = arr.size
        print(f'  BAD: {name} - shape {arr.shape}, NaN={n_nan}, inf={n_inf} ({(n_nan+n_inf)/total*100:.1f}%)')
        bad_names.add(name)

if not bad_names:
    print('All files OK! No filtering needed.')
    exit(0)

print(f'\nFound {len(bad_names)} bad files: {sorted(bad_names)}')

# Step 2: Filter split CSVs
for split_name in ['8AA', '8AA_train', '8AA_val', '8AA_test']:
    path = os.path.join(args.splits_dir, f'{split_name}.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, index_col='name')
    before = len(df)
    in_bad = [n for n in df.index if n in bad_names]
    if not in_bad:
        print(f'{split_name}: {before} entries, 0 bad - no change')
        continue
    df = df.drop(in_bad)
    print(f'{split_name}: {before} -> {len(df)} entries (removed {len(in_bad)}: {in_bad})')
    if not args.dry_run:
        df.to_csv(path)
        print(f'  Saved to {path}')

if args.dry_run:
    print('\nDry run - no files modified. Remove --dry_run to apply changes.')
else:
    print('\nDone! Split CSVs updated. You can now retrain.')
