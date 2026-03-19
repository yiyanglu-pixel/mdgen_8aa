"""Diagnostic: print all residue names from prmtop for peptides that fail the 8-residue check."""
import argparse, os, mdtraj

from mdgen.residue_constants import restype_3to1

AMBER_TO_STANDARD = {
    'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS',
    'CYX': 'CYS', 'CYM': 'CYS',
    'ASH': 'ASP',
    'GLH': 'GLU',
    'LYN': 'LYS',
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--names', type=str, nargs='+', default=['opep_0006', 'opep_0007', 'opep_0008', 'opep_0015'])
args = parser.parse_args()

for name in args.names:
    prmtop = os.path.join(args.data_dir, name, 'prmtop')
    if not os.path.exists(prmtop):
        print(f'{name}: prmtop not found')
        continue
    top = mdtraj.load_prmtop(prmtop)
    print(f'\n{name}: {top.n_residues} total residues')
    for r in top.residues:
        std = AMBER_TO_STANDARD.get(r.name, r.name)
        in_map = std in restype_3to1
        print(f'  {r.index:2d}  {r.name:5s} -> {std:5s}  {"OK" if in_map else "MISSING"}')
