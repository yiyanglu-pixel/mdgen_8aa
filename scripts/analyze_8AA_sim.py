"""
Analyze 8AA octapeptide forward simulation results.

Computes:
  - Jensen-Shannon divergence on all torsion angle marginals
  - Ramachandran (phi/psi pair) JSD
  - TICA free energy surface comparison
  - Torsion autocorrelation / decorrelation
  - Markov state model (MSM) metastable state distributions

Usage:
  python scripts/analyze_8AA_sim.py \
      --pdbdir results/8AA_multi_2000ep \
      --mddir /localhome3/lyy/8pep_gb_sim/octapeptides_data/ONE_octapeptides \
      --split splits/8AA_test.csv \
      --save --plot

  # Analyze specific peptides:
  python scripts/analyze_8AA_sim.py \
      --pdbdir results/8AA_multi_2000ep \
      --mddir /path/to/octapeptides \
      --pdb_id opep_0000 opep_0001 \
      --save --plot
"""
import argparse
parser = argparse.ArgumentParser(description='Analyze 8AA forward simulation results')
parser.add_argument('--mddir', type=str,
                    default='/localhome3/lyy/8pep_gb_sim/octapeptides_data/ONE_octapeptides',
                    help='Directory containing reference MD data (opep_XXXX/ subdirs)')
parser.add_argument('--pdbdir', type=str, required=True,
                    help='Directory containing generated PDB/XTC files from sim_inference.py')
parser.add_argument('--split', type=str, default=None,
                    help='CSV split file (e.g. splits/8AA_test.csv). '
                         'If provided, analyze all peptides in the split.')
parser.add_argument('--save', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--save_name', type=str, default='out_8AA.pkl')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--no_msm', action='store_true')
parser.add_argument('--no_decorr', action='store_true')
parser.add_argument('--no_traj_msm', action='store_true')
parser.add_argument('--truncate', type=int, default=None)
parser.add_argument('--msm_lag', type=int, default=10)
parser.add_argument('--tica_lag', type=int, default=100,
                    help='TICA lag time (default 100, suitable for 8AA 10ps frames)')
parser.add_argument('--num_workers', type=int, default=1)

args = parser.parse_args()

import mdgen.analysis
import pyemma, tqdm, os, pickle
from scipy.spatial.distance import jensenshannon
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acovf
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def main(name):
    out = {}
    np.random.seed(137)

    # --- Load generated trajectory (standard format: {name}.pdb + {name}.xtc) ---
    gen_path = f'{args.pdbdir}/{name}'

    # --- Load reference MD trajectory (8AA format) ---
    # Use the octapeptide-specific loader from analysis.py
    try:
        feats_ref, ref = mdgen.analysis.get_featurized_traj_octapeptide(
            args.mddir, name, sidechains=True, cossin=False)
    except Exception as e:
        print(f'ERROR loading reference for {name}: {e}', flush=True)
        return name, {'error': str(e)}

    try:
        feats, traj = mdgen.analysis.get_featurized_traj(
            gen_path, sidechains=True, cossin=False)
    except Exception as e:
        print(f'ERROR loading generated trajectory for {name}: {e}', flush=True)
        return name, {'error': str(e)}

    if args.truncate:
        traj = traj[:args.truncate]

    feat_names = feats.describe()
    out['features'] = feat_names
    n_feats = len(feat_names)

    # --- Determine subplot layout based on feature count ---
    n_cols = 4
    n_rows = max(4, (n_feats // n_cols) + 3)
    if args.plot:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # --- Backbone torsion marginals (plot only) ---
    if args.plot:
        feats_bb, traj_bb = mdgen.analysis.get_featurized_traj(
            gen_path, sidechains=False, cossin=False)
        feats_bb_ref, ref_bb = mdgen.analysis.get_featurized_traj_octapeptide(
            args.mddir, name, sidechains=False, cossin=False)
        if args.truncate:
            traj_bb = traj_bb[:args.truncate]

        pyemma.plots.plot_feature_histograms(
            ref_bb, feature_labels=feats_bb_ref, ax=axs[0, 0], color=colors[0])
        pyemma.plots.plot_feature_histograms(
            traj_bb, ax=axs[0, 0], color=colors[1])
        axs[0, 0].set_title('BB torsions (blue=MD, orange=gen)')

    # === JENSEN-SHANNON DISTANCES ON ALL TORSIONS ===
    out['JSD'] = {}
    for i, feat_name in enumerate(feat_names):
        ref_p = np.histogram(ref[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj[:, i], range=(-np.pi, np.pi), bins=100)[0]
        out['JSD'][feat_name] = jensenshannon(ref_p, traj_p)

    # --- Ramachandran (phi/psi pair) JSD ---
    # Dynamically find consecutive phi/psi pairs
    bb_feat_names = [f for f in feat_names if 'PHI' in f or 'PSI' in f]
    for i in range(len(feat_names) - 1):
        f1, f2 = feat_names[i], feat_names[i + 1]
        if ('PHI' in f1 and 'PSI' in f2) or ('PSI' in f1 and 'PHI' in f2):
            ref_p = np.histogram2d(
                *ref[:, i:i+2].T,
                range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50)[0]
            traj_p = np.histogram2d(
                *traj[:, i:i+2].T,
                range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50)[0]
            pair_name = f'{f1}|{f2}'
            out['JSD'][pair_name] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    # === TORSION DECORRELATION ===
    if not args.no_decorr:
        out['md_decorrelation'] = {}
        for i, feat_name in enumerate(feat_names):
            autocorr = (acovf(np.sin(ref[:, i]), demean=False, adjusted=True, nlag=100000) +
                        acovf(np.cos(ref[:, i]), demean=False, adjusted=True, nlag=100000))
            baseline = np.sin(ref[:, i]).mean()**2 + np.cos(ref[:, i]).mean()**2
            lags = 1 + np.arange(len(autocorr))
            normed = (autocorr - baseline) / (1 - baseline)
            out['md_decorrelation'][feat_name] = normed.astype(np.float16)

            if args.plot:
                ax_idx = 1 if ('PHI' in feat_name or 'PSI' in feat_name) else 2
                axs[0, ax_idx].plot(lags, normed, color=colors[i % len(colors)])

        if args.plot:
            axs[0, 1].set_title('MD BB decorrelation')
            axs[0, 2].set_title('MD SC decorrelation')
            axs[0, 1].set_xscale('log')
            axs[0, 2].set_xscale('log')

        out['our_decorrelation'] = {}
        for i, feat_name in enumerate(feat_names):
            autocorr = (acovf(np.sin(traj[:, i]), demean=False, adjusted=True, nlag=1000) +
                        acovf(np.cos(traj[:, i]), demean=False, adjusted=True, nlag=1000))
            baseline = np.sin(traj[:, i]).mean()**2 + np.cos(traj[:, i]).mean()**2
            lags = 1 + np.arange(len(autocorr))
            normed = (autocorr - baseline) / (1 - baseline)
            out['our_decorrelation'][feat_name] = normed.astype(np.float16)

            if args.plot:
                ax_idx = 1 if ('PHI' in feat_name or 'PSI' in feat_name) else 2
                axs[1, ax_idx].plot(lags, normed, color=colors[i % len(colors)])

        if args.plot:
            axs[1, 1].set_title('Gen BB decorrelation')
            axs[1, 2].set_title('Gen SC decorrelation')
            axs[1, 1].set_xscale('log')
            axs[1, 2].set_xscale('log')

    # === TICA ===
    feats_cs, traj_cs = mdgen.analysis.get_featurized_traj(
        gen_path, sidechains=True, cossin=True)
    if args.truncate:
        traj_cs = traj_cs[:args.truncate]
    feats_cs_ref, ref_cs = mdgen.analysis.get_featurized_traj_octapeptide(
        args.mddir, name, sidechains=True, cossin=True)

    tica, _ = mdgen.analysis.get_tica(ref_cs, lag=args.tica_lag)
    ref_tica = tica.transform(ref_cs)
    traj_tica = tica.transform(traj_cs)

    tica_0_min = min(ref_tica[:, 0].min(), traj_tica[:, 0].min())
    tica_0_max = max(ref_tica[:, 0].max(), traj_tica[:, 0].max())
    tica_1_min = min(ref_tica[:, 1].min(), traj_tica[:, 1].min())
    tica_1_max = max(ref_tica[:, 1].max(), traj_tica[:, 1].max())

    ref_p = np.histogram(ref_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    out['JSD']['TICA-0'] = jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(
        *ref_tica[:, :2].T,
        range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50)[0]
    traj_p = np.histogram2d(
        *traj_tica[:, :2].T,
        range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50)[0]
    out['JSD']['TICA-0,1'] = jensenshannon(ref_p.flatten(), traj_p.flatten())

    # --- TICA FES plot ---
    if args.plot:
        pyemma.plots.plot_free_energy(*ref_tica[::100, :2].T, ax=axs[2, 0], cbar=False)
        pyemma.plots.plot_free_energy(*traj_tica[:, :2].T, ax=axs[2, 1], cbar=False)
        axs[2, 0].set_title('TICA FES (MD ref)')
        axs[2, 1].set_title('TICA FES (generated)')

    # --- TICA decorrelation ---
    if not args.no_decorr:
        autocorr = acovf(ref_tica[:, 0], nlag=100000, adjusted=True, demean=False)
        out['md_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[0, 3].plot(autocorr)
            axs[0, 3].set_title('MD TICA autocorr')

        autocorr = acovf(traj_tica[:, 0], nlag=1000, adjusted=True, demean=False)
        out['our_decorrelation']['tica'] = autocorr.astype(np.float16)
        if args.plot:
            axs[1, 3].plot(autocorr)
            axs[1, 3].set_title('Gen TICA autocorr')

    # === MARKOV STATE MODEL ===
    if not args.no_msm:
        kmeans, ref_kmeans = mdgen.analysis.get_kmeans(tica.transform(ref_cs))
        try:
            msm, pcca, cmsm = mdgen.analysis.get_msm(ref_kmeans, nstates=10)

            out['kmeans'] = kmeans
            out['msm'] = msm
            out['pcca'] = pcca
            out['cmsm'] = cmsm

            traj_discrete = mdgen.analysis.discretize(tica.transform(traj_cs), kmeans, msm)
            ref_discrete = mdgen.analysis.discretize(tica.transform(ref_cs), kmeans, msm)
            out['traj_metastable_probs'] = (traj_discrete == np.arange(10)[:, None]).mean(1)
            out['ref_metastable_probs'] = (ref_discrete == np.arange(10)[:, None]).mean(1)

            msm_transition_matrix = np.eye(10)
            for a, i in enumerate(cmsm.active_set):
                for b, j in enumerate(cmsm.active_set):
                    msm_transition_matrix[i, j] = cmsm.transition_matrix[a, b]
            out['msm_transition_matrix'] = msm_transition_matrix

            out['pcca_pi'] = pcca._pi_coarse

            msm_pi = np.zeros(10)
            msm_pi[cmsm.active_set] = cmsm.pi
            out['msm_pi'] = msm_pi

            if not args.no_traj_msm:
                traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=args.msm_lag)
                out['traj_msm'] = traj_msm

                traj_transition_matrix = np.eye(10)
                for a, i in enumerate(traj_msm.active_set):
                    for b, j in enumerate(traj_msm.active_set):
                        traj_transition_matrix[i, j] = traj_msm.transition_matrix[a, b]
                out['traj_transition_matrix'] = traj_transition_matrix

                traj_pi = np.zeros(10)
                traj_pi[traj_msm.active_set] = traj_msm.pi
                out['traj_pi'] = traj_pi

        except Exception as e:
            print(f'MSM ERROR for {name}: {e}', flush=True)

    if args.plot:
        fig.suptitle(f'{name}', fontsize=16, y=1.0)
        fig.tight_layout()
        fig.savefig(f'{args.pdbdir}/{name}_analysis.pdf', bbox_inches='tight')
        plt.close(fig)

    return name, out


# --- Determine which peptides to analyze ---
if args.pdb_id:
    pdb_id = args.pdb_id
elif args.split:
    df = pd.read_csv(args.split, index_col='name')
    pdb_id = list(df.index)
else:
    pdb_id = [nam.split('.')[0] for nam in os.listdir(args.pdbdir)
              if nam.endswith('.pdb') and '_traj' not in nam]

# Filter to only those with generated XTC
pdb_id = [nam for nam in pdb_id if os.path.exists(f'{args.pdbdir}/{nam}.xtc')]
print(f'Analyzing {len(pdb_id)} trajectories', flush=True)

if len(pdb_id) == 0:
    print('ERROR: No trajectories found. Check --pdbdir and that inference has completed.')
    exit(1)

if args.num_workers > 1:
    p = Pool(args.num_workers)
    p.__enter__()
    __map__ = p.imap
else:
    __map__ = map

out = dict(tqdm.tqdm(__map__(main, pdb_id), total=len(pdb_id)))

if args.num_workers > 1:
    p.__exit__(None, None, None)

# --- Print summary ---
print('\n' + '=' * 60)
print('JSD Summary (lower is better)')
print('=' * 60)

all_jsds = {}
for name, result in out.items():
    if 'error' in result:
        print(f'  {name}: SKIPPED ({result["error"]})')
        continue
    for feat_name, jsd_val in result.get('JSD', {}).items():
        if feat_name not in all_jsds:
            all_jsds[feat_name] = []
        all_jsds[feat_name].append(jsd_val)

if all_jsds:
    print(f'\n{"Feature":<40} {"Mean JSD":>10} {"Std":>10} {"N":>5}')
    print('-' * 67)
    for feat_name in sorted(all_jsds.keys()):
        vals = np.array(all_jsds[feat_name])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            print(f'{feat_name:<40} {vals.mean():>10.4f} {vals.std():>10.4f} {len(vals):>5d}')

    # Overall mean JSD across torsion features (excluding TICA)
    torsion_jsds = []
    for k, v in all_jsds.items():
        if 'TICA' not in k and '|' not in k:
            torsion_jsds.extend(v)
    torsion_jsds = np.array(torsion_jsds)
    torsion_jsds = torsion_jsds[~np.isnan(torsion_jsds)]
    if len(torsion_jsds) > 0:
        print(f'\n  Overall torsion JSD:  {torsion_jsds.mean():.4f} +/- {torsion_jsds.std():.4f}')

    for tica_key in ['TICA-0', 'TICA-0,1']:
        if tica_key in all_jsds:
            vals = np.array(all_jsds[tica_key])
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                print(f'  {tica_key} JSD:         {vals.mean():.4f} +/- {vals.std():.4f}')

# --- Metastable state probability comparison ---
traj_probs_all = []
ref_probs_all = []
for name, result in out.items():
    if 'traj_metastable_probs' in result and 'ref_metastable_probs' in result:
        traj_probs_all.append(result['traj_metastable_probs'])
        ref_probs_all.append(result['ref_metastable_probs'])

if traj_probs_all:
    print(f'\nMSM metastable state prob MAE: '
          f'{np.abs(np.array(traj_probs_all) - np.array(ref_probs_all)).mean():.4f}')

if args.save:
    save_path = f"{args.pdbdir}/{args.save_name}"
    with open(save_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'\nResults saved to {save_path}')
