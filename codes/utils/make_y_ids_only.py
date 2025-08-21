import argparse, os
import numpy as np
import scipy.io as sio

def load_stims(subject: str, use_stim: str):
    """Load stimulus ID sequence used to stack rows in subjfeat."""
    base = f'../../mrifeat/{subject}/{subject}'
    if use_stim == 'ave':
        path = f'{base}_stims_ave.npy'
    elif use_stim == 'each':
        path = f'{base}_stims.npy'
    else:
        raise ValueError("--use_stim must be 'ave' or 'each'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    return np.load(path)

def compute_tr_te_ids(stims: np.ndarray, nsd_root='../../nsd/'):
    """Split the stims sequence into train/test by sharedix from nsd_expdesign.mat."""
    mat = sio.loadmat(os.path.join(
        nsd_root, 'nsddata/experiments/nsd/nsd_expdesign.mat'))
    sharedix = mat['sharedix'].astype(np.int64).ravel() - 1  # 1-based -> 0-based
    shared_set = set(int(x) for x in sharedix)
    # train = 非 shared；test = shared（与原仓库脚本一致）
    is_train = np.array([int(s) not in shared_set for s in stims], dtype=bool)
    ids_tr = stims[is_train]
    ids_te = stims[~is_train]
    return ids_tr.astype(np.int64), ids_te.astype(np.int64)

def maybe_check_length(subject, use_stim, featname, ids_tr, ids_te):
    """若 subjfeat 已有特征矩阵，做一次长度一致性校验（没有就跳过）。"""
    featdir = '../../nsdfeat/subjfeat'
    f_tr = f'{featdir}/{subject}_{use_stim}_{featname}_tr.npy'
    f_te = f'{featdir}/{subject}_{use_stim}_{featname}_te.npy'
    if os.path.exists(f_tr):
        n = np.load(f_tr, mmap_mode='r').shape[0]
        if n != len(ids_tr):
            print(f'[WARN] row mismatch: {os.path.basename(f_tr)} has {n}, '
                  f'but ids_tr has {len(ids_tr)}')
    if os.path.exists(f_te):
        n = np.load(f_te, mmap_mode='r').shape[0]
        if n != len(ids_te):
            print(f'[WARN] row mismatch: {os.path.basename(f_te)} has {n}, '
                  f'but ids_te has {len(ids_te)}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', required=True, type=str, help='e.g. subj01')
    ap.add_argument('--use_stim', required=True, choices=['ave', 'each'])
    ap.add_argument('--featname', nargs='+', required=True,
                    help='one or more names, e.g. c init_latent')
    ap.add_argument('--nsd_root', type=str, default='../../nsd/',
                    help='path to nsd root (contains nsddata/...)')
    args = ap.parse_args()

    stims = load_stims(args.subject, args.use_stim)
    ids_tr, ids_te = compute_tr_te_ids(stims, nsd_root=args.nsd_root)

    outdir = '../../nsdfeat/subjfeat'
    os.makedirs(outdir, exist_ok=True)

    for fname in args.featname:
        maybe_check_length(args.subject, args.use_stim, fname, ids_tr, ids_te)
        out_tr = f'{outdir}/{args.subject}_{args.use_stim}_{fname}_tr_ids.npy'
        out_te = f'{outdir}/{args.subject}_{args.use_stim}_{fname}_te_ids.npy'
        np.save(out_tr, ids_tr)
        np.save(out_te, ids_te)
        print(f'[OK] wrote: {out_tr}  (len={len(ids_tr)})')
        print(f'[OK] wrote: {out_te}  (len={len(ids_te)})')

if __name__ == '__main__':
    main()
