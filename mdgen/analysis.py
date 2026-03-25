import json
import os

import numpy as np
# Restore deprecated numpy aliases removed in numpy 1.24+
# (required by pyemma internals for TICA/MSM)
if not hasattr(np, 'bool'):
    np.bool = bool
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.str = str
import pyemma
from tqdm import tqdm

def get_featurizer(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    return feat


class _FeatureDescriptor(list):
    """Lightweight replacement for pyemma MDFeaturizer.

    Extends list so isinstance(..., list) checks pass (required by
    pyemma.plots.plot_feature_histograms). Also provides describe()
    and dimension() for API compat with pyemma MDFeaturizer.
    """

    def __init__(self, names):
        super().__init__(names)

    def describe(self):
        return list(self)

    def dimension(self):
        return len(self)


def _featurize_traj_mdtraj(traj, sidechains=False, cossin=True):
    """Compute torsion features from an mdtraj.Trajectory using mdtraj directly.

    Produces features in the same order as pyemma:
      backbone: [phi_1..N-1, psi_0..N-2]
      sidechain: [chi1_all, chi2_all, chi3_all, chi4_all]
      cossin: [cos(all), sin(all)] per group

    Returns (_FeatureDescriptor, np.ndarray of shape (n_frames, n_features)).
    """
    import mdtraj

    arrays = []
    feat_names = []

    phi_idx, phi = mdtraj.compute_phi(traj)
    psi_idx, psi = mdtraj.compute_psi(traj)

    for idx in phi_idx:
        res = traj.topology.atom(idx[1]).residue
        feat_names.append(f'PHI {res.name} {res.index}')
    for idx in psi_idx:
        res = traj.topology.atom(idx[1]).residue
        feat_names.append(f'PSI {res.name} {res.index}')

    bb = np.hstack([phi, psi])
    if cossin:
        feat_names = [f'COS({n})' for n in feat_names] + \
                     [f'SIN({n})' for n in feat_names]
        bb = np.hstack([np.cos(bb), np.sin(bb)])
    arrays.append(bb)

    if sidechains:
        sc_names = []
        sc_parts = []
        for chi_func, chi_label in [(mdtraj.compute_chi1, 'CHI1'),
                                     (mdtraj.compute_chi2, 'CHI2'),
                                     (mdtraj.compute_chi3, 'CHI3'),
                                     (mdtraj.compute_chi4, 'CHI4')]:
            chi_idx, chi = chi_func(traj)
            if chi.size > 0:
                sc_parts.append(chi)
                for idx in chi_idx:
                    res = traj.topology.atom(idx[1]).residue
                    sc_names.append(f'{chi_label} {res.name} {res.index}')
        if sc_parts:
            sc = np.hstack(sc_parts)
            if cossin:
                sc_names = [f'COS({n})' for n in sc_names] + \
                           [f'SIN({n})' for n in sc_names]
                sc = np.hstack([np.cos(sc), np.sin(sc)])
            feat_names.extend(sc_names)
            arrays.append(sc)

    traj_feat = np.hstack(arrays) if arrays else np.empty((traj.n_frames, 0))
    return _FeatureDescriptor(feat_names), traj_feat


def get_featurized_traj(name, sidechains=False, cossin=True):
    import mdtraj
    traj = mdtraj.load(name + '.xtc', top=name + '.pdb')
    return _featurize_traj_mdtraj(traj, sidechains=sidechains, cossin=cossin)


def get_featurized_traj_octapeptide(md_dir, name, sidechains=False, cossin=True):
    """Load featurized reference MD trajectory for 8AA octapeptides.

    Strips ACE/NME capping groups, then computes torsion features with mdtraj
    directly (completely bypasses pyemma to avoid numpy >= 1.24 compat issues).
    """
    import mdtraj

    base = os.path.join(md_dir, name)
    pdb_noH = os.path.join(base, f'{name}_noH.pdb')
    xtc_noH = os.path.join(base, f'{name}_noH.xtc')

    traj = mdtraj.load(xtc_noH, top=pdb_noH)

    # Strip ACE/NME capping groups
    standard_res = [r.index for r in traj.topology.residues
                    if r.name not in ('ACE', 'NME')]
    if len(standard_res) < traj.topology.n_residues:
        atom_indices = traj.topology.select(
            ' or '.join(f'resid {r}' for r in standard_res))
        traj = traj.atom_slice(atom_indices)

    return _featurize_traj_mdtraj(traj, sidechains=sidechains, cossin=cossin)


def get_featurized_atlas_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'_prod_R1_fit.xtc', features=feat)
    return feat, traj

def get_tica(traj, lag=1000):
    tica = pyemma.coordinates.tica(traj, lag=lag, kinetic_map=True)
    # lag time 100 ps = 0.1 ns
    return tica, tica.transform(traj)

def get_kmeans(traj):
    kmeans = pyemma.coordinates.cluster_kmeans(traj, k=100, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj)[:,0]

def get_msm(traj, lag=1000, nstates=10):
    msm = pyemma.msm.estimate_markov_model(traj, lag=lag)
    pcca = msm.pcca(nstates)
    assert len(msm.metastable_assignments) == 100
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj], lag=lag)
    return msm, pcca, cmsm

def discretize(traj, kmeans, msm):
    return msm.metastable_assignments[kmeans.transform(traj)[:,0]]

def load_tps_ensemble(name, directory):
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_state_probs(tp, num_states=10):
    stationary = np.bincount(tp.reshape(-1), minlength=num_states)
    return stationary / stationary.sum()