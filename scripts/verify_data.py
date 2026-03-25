"""Verify preprocessed 8AA data before training.

Usage:
    python -m scripts.verify_data \
        --data_dir /localhome3/lyy/mdgen_8aa/data/8AA_data \
        --suffix _i100
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Verify preprocessed 8AA .npy files')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--suffix', type=str, default='_i100')
args = parser.parse_args()

files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(f'{args.suffix}.npy')])
print(f'Found {len(files)} .npy files in {args.data_dir}')

if not files:
    print('ERROR: No files found!')
    exit(1)

errors = []
shapes = {}
for i, f in enumerate(files):
    path = os.path.join(args.data_dir, f)
    try:
        arr = np.lib.format.open_memmap(path, 'r')
        shape = arr.shape
        if len(shape) != 4 or shape[1] != 8 or shape[2] != 14 or shape[3] != 3:
            errors.append(f'{f}: unexpected shape {shape} (expected (N, 8, 14, 3))')
        shapes[shape] = shapes.get(shape, 0) + 1
        if arr.dtype != np.float16:
            errors.append(f'{f}: unexpected dtype {arr.dtype} (expected float16)')
    except Exception as e:
        errors.append(f'{f}: failed to load - {e}')

    if (i + 1) % 100 == 0:
        print(f'  Checked {i + 1}/{len(files)}...')

print(f'\nShape distribution:')
for shape, count in sorted(shapes.items(), key=lambda x: -x[1]):
    print(f'  {shape}: {count} files')

if errors:
    print(f'\n{len(errors)} ERRORS:')
    for e in errors[:20]:
        print(f'  {e}')
    if len(errors) > 20:
        print(f'  ... and {len(errors) - 20} more')
else:
    print(f'\nAll {len(files)} files OK!')

# Check a sample file in detail
sample = os.path.join(args.data_dir, files[0])
arr = np.lib.format.open_memmap(sample, 'r')
print(f'\nSample file: {files[0]}')
print(f'  Shape: {arr.shape}')
print(f'  Dtype: {arr.dtype}')
print(f'  Frames: {arr.shape[0]}')
print(f'  Min coord: {np.min(arr):.2f} A')
print(f'  Max coord: {np.max(arr):.2f} A')
print(f'  Has NaN: {np.any(np.isnan(arr))}')

min_frames = min(np.lib.format.open_memmap(os.path.join(args.data_dir, f), 'r').shape[0] for f in files)
print(f'\nMin frames across all files: {min_frames}')
if min_frames < 100:
    print(f'  WARNING: some files have <100 frames. Use --num_frames < {min_frames} for training.')
else:
    print(f'  OK for --num_frames 100')
