import argparse, os
import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def reorder_to(ids_src, ids_tgt, A):
    """把 A 的行从 ids_src 顺到 ids_tgt；A.shape[0] == len(ids_src)"""
    pos = {int(s): i for i, s in enumerate(ids_src)}
    idx = [pos[int(s)] for s in ids_tgt]  # 若 KeyError 说明集合不一致
    return A[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='', help="c / init_latent / ...")
    parser.add_argument("--roi", required=True, type=str, nargs="*", help="e.g. early ventral")
    parser.add_argument("--subject", type=str, default=None, help="subj01 / subj02 / subj05 / subj07")
    parser.add_argument("--tbatch", type=int, default=2048, help="targets per batch (chunk size)")
    parser.add_argument("--train_stim", type=str, default="each", choices=["each","ave"],
                        help="训练集使用 each(默认) 或 ave（与你如何做 X_tr 对齐一致）")
    parser.add_argument("--no_assert", action="store_true", help="跳过对齐断言（不建议）")
    opt = parser.parse_args()

    target    = opt.target
    roi       = opt.roi
    subject   = opt.subject
    tbatch    = opt.tbatch
    train_stim = opt.train_stim
    
    # use numpy backend
    _ = set_backend("numpy", on_error="warn")

    # alphas
    if target in ("c", "init_latent"):
        alpha = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    else:
        alpha = [1e4, 2e4, 4e4]

    ridge = RidgeCV(alphas=alpha)

    preprocess = make_pipeline(
        StandardScaler(with_mean=True, with_std=True, copy=False),
    )
    pipe = make_pipeline(preprocess, ridge)

    mridir  = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../../decoded/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    # -----------------------
    # 组装 X / X_te
    # -----------------------
    X_list, Xte_list = [], []
    for r in roi:
        if 'conv' in target:
            X_list.append(  np.load(f'{mridir}/{subject}_{r}_betas_ave_tr.npy').astype('float32'))
        else:
            # 你当前是 each 做训练
            if train_stim == "each":
                X_list.append(np.load(f'{mridir}/{subject}_{r}_betas_tr.npy').astype('float32'))
            else:
                X_list.append(np.load(f'{mridir}/{subject}_{r}_betas_ave_tr.npy').astype('float32'))
        Xte_list.append(np.load(f'{mridir}/{subject}_{r}_betas_ave_te.npy').astype('float32'))

    X    = np.hstack(X_list)     # (N_tr, V)
    X_te = np.hstack(Xte_list)   # (N_te, V)

    # -----------------------
    # 读取“权威 ID 顺序”
    # -----------------------
    if train_stim == "each":
        ids_X_tr_path = f'{mridir}/{subject}_ids_each_tr.npy'
    else:
        ids_X_tr_path = f'{mridir}/{subject}_ids_ave_tr.npy'
    ids_X_te_path = f'{mridir}/{subject}_ids_ave_te.npy'

    ids_X_tr = np.load(ids_X_tr_path)
    ids_X_te = np.load(ids_X_te_path)

    # -----------------------
    # 组装 Y / Y_te（先不 reshape）+ 尝试按 ids 重排
    # -----------------------
    Y_tr_path   = f'{featdir}/{subject}_{train_stim}_{target}_tr.npy'
    Y_tr_ids_p  = f'{featdir}/{subject}_{train_stim}_{target}_tr_ids.npy'
    Y_te_path   = f'{featdir}/{subject}_ave_{target}_te.npy'
    Y_te_ids_p  = f'{featdir}/{subject}_ave_{target}_te_ids.npy'

    Y_tr = np.load(Y_tr_path).astype('float32')
    Y_te = np.load(Y_te_path).astype('float32')

    # 如果有 _ids.npy 就重排；否则给出警告
    if os.path.exists(Y_tr_ids_p) and os.path.exists(Y_te_ids_p):
        ids_Y_tr = np.load(Y_tr_ids_p)
        ids_Y_te = np.load(Y_te_ids_p)

        # 重排到与 X 的顺序一致
        Y_tr = reorder_to(ids_Y_tr, ids_X_tr, Y_tr)
        Y_te = reorder_to(ids_Y_te, ids_X_te, Y_te)

        if not opt.no_assert:
            assert np.array_equal(ids_X_tr, ids_Y_tr), "TRAIN ids mismatch!"
            assert np.array_equal(ids_X_te, ids_Y_te), "TEST ids mismatch!"
    else:
        print("[WARN] *_ids.npy 不存在，跳过重排。在 make_subjstim.py 同步保存 *_ids.npy 以保证对齐。")

    # 真正 reshape（与 X 行数对齐）
    Y    = Y_tr.reshape([X.shape[0],   -1])
    Y_te = Y_te.reshape([X_te.shape[0], -1])
    tbatch=Y.shape[1]
    n_tr, n_te = X.shape[0], X_te.shape[0]
    D = Y.shape[1]   # 例如 59136
    print(f'Now making decoding model for... {subject}:  ROI={roi}, target={target}, train_stim={train_stim}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}, tbatch={tbatch}')

    # 预分配输出
    scores_te = np.empty((n_te, D), dtype='float32')
    scores_tr = np.empty((n_tr, D), dtype='float32')

    corr_te_all, corr_tr_all = [], []

    # ===== 分块拟合与预测 =====
    for start in range(0, D, tbatch):
        end = min(start + tbatch, D)
        Y_chunk    = Y[:,    start:end]        # (N_tr, tbatch)
        Yte_chunk  = Y_te[:, start:end]        # (N_te, tbatch)

        pipe.fit(X, Y_chunk)

        pred_te = pipe.predict(X_te).astype('float32')  # (N_te, tbatch)
        pred_tr = pipe.predict(X   ).astype('float32')  # (N_tr, tbatch)

        scores_te[:, start:end] = pred_te
        scores_tr[:, start:end] = pred_tr

        corr_te = correlation_score(Yte_chunk.T, pred_te.T).astype('float32')
        corr_tr = correlation_score(Y_chunk.T,   pred_tr.T).astype('float32')
        corr_te_all.append(corr_te)
        corr_tr_all.append(corr_tr)

        print(f'  chunk {start:5d}-{end:5d}: test r̄={float(np.mean(corr_te)):.3f}, '
              f'train r̄={float(np.mean(corr_tr)):.3f}')

    corr_te_all = np.concatenate(corr_te_all)  # (D,)
    corr_tr_all = np.concatenate(corr_tr_all)  # (D,)
    print(f'[TEST]  mean correlation: {float(np.mean(corr_te_all)):.3f}')
    print(f'[TRAIN] mean correlation: {float(np.mean(corr_tr_all)):.3f}')

    tag = f'{subject}_{"_".join(roi)}'
    np.save(f'{savedir}/{tag}_scores_{target}.npy',    scores_te)  # 测试集预测（给解码用）
    np.save(f'{savedir}/{tag}_scores_{target}_tr.npy', scores_tr)  # 训练集预测（给对角仿射用）
    np.save(f'{savedir}/{tag}_corr_{target}.npy',      corr_te_all)
    np.save(f'{savedir}/{tag}_corr_{target}_tr.npy',   corr_tr_all)

if __name__ == "__main__":
    main()
