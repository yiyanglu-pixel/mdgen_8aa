"""Check raw octapeptides data completeness before preprocessing.

Scans all opep_XXXX directories and verifies required files exist.
Optionally spot-checks a few trajectories with mdtraj.

Usage:
    python -m scripts.check_raw_data --data_dir /localhome3/lyy/octapeptides_data
    python -m scripts.check_raw_data --data_dir /localhome3/lyy/octapeptides_data --spot_check 10
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Check raw octapeptides data completeness')
parser.add_argument('--data_dir', type=str, default='/localhome3/lyy/octapeptides_data',
                    help='Path to octapeptides data root')
parser.add_argument('--expected', type=int, default=1100,
                    help='Expected number of peptides (default: 1100)')
parser.add_argument('--spot_check', type=int, default=5,
                    help='Number of peptides to spot-check with mdtraj (0 to skip)')
args = parser.parse_args()

REQUIRED_FILES = [
    '{name}_noH.pdb',
    '{name}_noH.xtc',
    'topology.pdb',
    'prmtop',
]

# Scan directories
all_dirs = sorted([d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('opep_')])

print(f'Data directory: {args.data_dir}')
print(f'Found {len(all_dirs)} opep_* directories (expected {args.expected})')

# Check for missing directories
expected_names = [f'opep_{i:04d}' for i in range(args.expected)]
found_set = set(all_dirs)
missing_dirs = [n for n in expected_names if n not in found_set]
extra_dirs = [n for n in all_dirs if n not in set(expected_names)]

if missing_dirs:
    print(f'\nMISSING directories ({len(missing_dirs)}):')
    for n in missing_dirs[:20]:
        print(f'  {n}')
    if len(missing_dirs) > 20:
        print(f'  ... and {len(missing_dirs) - 20} more')

if extra_dirs:
    print(f'\nEXTRA directories ({len(extra_dirs)}): {extra_dirs[:10]}')

# Check required files in each directory
missing_files = {}
for name in all_dirs:
    d = os.path.join(args.data_dir, name)
    missing = []
    for pattern in REQUIRED_FILES:
        fname = pattern.format(name=name)
        if not os.path.exists(os.path.join(d, fname)):
            missing.append(fname)
    if missing:
        missing_files[name] = missing

if missing_files:
    print(f'\nDirectories with MISSING required files ({len(missing_files)}):')
    for name, files in sorted(missing_files.items())[:20]:
        print(f'  {name}: {", ".join(files)}')
    if len(missing_files) > 20:
        print(f'  ... and {len(missing_files) - 20} more')
else:
    print(f'\nAll {len(all_dirs)} directories have required files: OK')

# Spot-check with mdtraj
if args.spot_check > 0:
    try:
        import mdtraj
        import numpy as np
        rng = np.random.RandomState(42)
        # Pick from directories that have all required files
        valid_dirs = [n for n in all_dirs if n not in missing_files]
        check_names = rng.choice(valid_dirs, size=min(args.spot_check, len(valid_dirs)), replace=False)

        CAPPING_RESIDUES = {'ACE', 'NME', 'NHE'}
        from mdgen.residue_constants import restype_3to1

        print(f'\nSpot-checking {len(check_names)} peptides with mdtraj...')
        for name in sorted(check_names):
            d = os.path.join(args.data_dir, name)
            pdb = os.path.join(d, f'{name}_noH.pdb')
            xtc = os.path.join(d, f'{name}_noH.xtc')
            traj = mdtraj.load(xtc, top=pdb)
            n_res_total = traj.n_residues
            n_frames = traj.n_frames
            n_atoms = traj.n_atoms
            # Count standard AA residues (exclude ACE/NME capping groups)
            std_residues = [r for r in traj.top.residues if r.name in restype_3to1]
            cap_residues = [r for r in traj.top.residues if r.name in CAPPING_RESIDUES]
            n_std = len(std_residues)
            cap_names = ','.join(r.name for r in cap_residues)
            status = 'OK' if n_std == 8 else f'WARN: {n_std} standard residues!'
            print(f'  {name}: {n_frames} frames, {n_atoms} atoms, '
                  f'{n_res_total} residues ({n_std} AA + caps [{cap_names}]) - {status}')
    except ImportError:
        print('\nSkipping spot-check: mdtraj not available')

# Summary
print(f'\n{"=" * 60}')
print(f'SUMMARY')
print(f'{"=" * 60}')
print(f'  Total directories:     {len(all_dirs)} / {args.expected}')
print(f'  Missing directories:   {len(missing_dirs)}')
print(f'  Missing files:         {len(missing_files)} directories')
complete = len(all_dirs) - len(missing_files) - len(missing_dirs)
print(f'  Ready for preprocessing: {len(all_dirs) - len(missing_files)}')

if len(missing_dirs) == 0 and len(missing_files) == 0:
    print(f'\n  All {args.expected} peptides ready! You can proceed with:')
    print(f'    python -m scripts.generate_8AA_splits --data_dir {args.data_dir} --outdir splits')
    print(f'    python -m scripts.prep_sims --split splits/8AA.csv --sim_dir {args.data_dir} \\')
    print(f'        --outdir data/8AA_data --num_workers 8 --suffix _i100 --stride 100 --octapeptides')
    sys.exit(0)
else:
    print(f'\n  Some data is missing. Fix before proceeding.')
    sys.exit(1)
