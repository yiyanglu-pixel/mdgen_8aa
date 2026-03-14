import argparse
import os
import numpy as np
import pandas as pd
import mdtraj

from mdgen.residue_constants import restype_3to1

parser = argparse.ArgumentParser(description='Generate 8AA split CSVs from octapeptides data')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to octapeptides data (e.g., octapeptides_data/ONE_octapeptides)')
parser.add_argument('--outdir', type=str, default='splits')
parser.add_argument('--train_frac', type=float, default=0.8)
parser.add_argument('--val_frac', type=float, default=0.1)
parser.add_argument('--test_frac', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

entries = []
dirs = sorted([d for d in os.listdir(args.data_dir)
               if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('opep_')])

for name in dirs:
    pdb_path = os.path.join(args.data_dir, name, 'topology.pdb')
    if not os.path.exists(pdb_path):
        print(f'Warning: {pdb_path} not found, skipping')
        continue
    traj = mdtraj.load(pdb_path)
    residues = [r for r in traj.top.residues if r.name in restype_3to1]
    if len(residues) != 8:
        print(f'Warning: {name} has {len(residues)} standard residues (expected 8), skipping')
        continue
    seqres = ''.join(restype_3to1[r.name] for r in residues)
    entries.append({'name': name, 'seqres': seqres})

df = pd.DataFrame(entries)
print(f'Found {len(df)} valid octapeptides')

# Save full split
df.to_csv(os.path.join(args.outdir, '8AA.csv'), index=False)

# Random split
rng = np.random.RandomState(args.seed)
indices = rng.permutation(len(df))
n_train = int(len(df) * args.train_frac)
n_val = int(len(df) * args.val_frac)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

df.iloc[train_idx].to_csv(os.path.join(args.outdir, '8AA_train.csv'), index=False)
df.iloc[val_idx].to_csv(os.path.join(args.outdir, '8AA_val.csv'), index=False)
df.iloc[test_idx].to_csv(os.path.join(args.outdir, '8AA_test.csv'), index=False)

print(f'Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')
print(f'Saved to {args.outdir}/8AA*.csv')
