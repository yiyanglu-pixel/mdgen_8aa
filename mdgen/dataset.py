import os
import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
        self._validate_data()
    def _validate_data(self):
        """Check that .npy files exist and have consistent shapes."""
        missing = []
        bad_shape = []
        ref_shape = None
        for name in self.df.index:
            full_name = name
            path = f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy'
            if not os.path.exists(path):
                missing.append(name)
                continue
            arr = np.lib.format.open_memmap(path, 'r')
            # Check residue/atom dimensions (skip frame dim which can vary)
            shape_tail = arr.shape[1:]  # (L, 14, 3)
            if ref_shape is None:
                ref_shape = shape_tail
            elif shape_tail != ref_shape:
                bad_shape.append((name, arr.shape, ref_shape))
            # Check minimum frames
            if arr.shape[0] < self.args.num_frames:
                bad_shape.append((name, f'{arr.shape[0]} frames < num_frames={self.args.num_frames}', None))
        if missing:
            print(f'WARNING: {len(missing)} peptides missing .npy files (first 5: {missing[:5]})')
        if bad_shape:
            print(f'ERROR: {len(bad_shape)} peptides have inconsistent shapes:')
            for item in bad_shape[:10]:
                print(f'  {item[0]}: got {item[1]}, expected (*,{ref_shape})')
            raise ValueError(
                f'Inconsistent .npy shapes detected! {len(bad_shape)} files differ. '
                f'Expected shape (*, {ref_shape}). '
                f'Re-run: python -m scripts.verify_data --data_dir {self.args.data_dir} --suffix {self.args.suffix}'
            )
        if ref_shape is not None:
            print(f'Dataset validated: {len(self.df)} peptides, shape (*, {ref_shape})')

    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        else:
            full_name = name
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        frame_start = int(np.random.randint(0, arr.shape[0] - self.args.num_frames))
        if self.args.overfit_frame:
            frame_start = 0
        end = frame_start + self.args.num_frames
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr[frame_start:end]).astype(np.float32) # / 10.0 # convert to nm
        if np.any(np.isinf(arr)) or np.any(np.isnan(arr)):
            print(f'WARNING: {full_name} has inf/NaN in frames {frame_start}:{end}, replacing with zeros')
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.args.copy_frames:
            arr[1:] = arr[0]

        # arr should be in ANGSTROMS
        frames = atom14_to_frames(torch.from_numpy(arr))
        # Guard against NaN from degenerate backbone geometry in from_3_points
        if torch.any(torch.isnan(frames._trans)):
            frames = Rigid(
                torch.nan_to_num(frames._trans, nan=0.0),
                frames._rots,
            )
        if torch.any(torch.isnan(frames._rots._rot_mats)):
            from .rigid_utils import Rotation
            frames = Rigid(
                frames._trans,
                Rotation(rot_mats=torch.nan_to_num(frames._rots._rot_mats, nan=0.0)),
            )
        seqres = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        # Zero out NaN torsions from degenerate geometry (e.g. masked atoms at origin)
        # NaN * 0 = NaN in PyTorch, so nan_to_num must come before any masking
        torsions = torch.nan_to_num(torsions, nan=0.0)

        torsion_mask = torsion_mask[0]
        
        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                
            
            elif L < self.args.crop:
                pad = self.args.crop - L
                frames = Rigid.cat([
                    frames, 
                    Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
                ], 1)
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask, # (L,)
        }

