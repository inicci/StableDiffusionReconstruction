import numpy as np, scipy.io, argparse, os
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--nsd_root", default='../../nsd')
    ap.add_argument("--mrifeat_root", default='../../mrifeat')
    args = ap.parse_args()

    subj = args.subject
    mridir = os.path.join(args.mrifeat_root, subj)
    os.makedirs(mridir, exist_ok=True)

    # 读取 split（sharedix 是 1-based，要减 1）
    exp = os.path.join(args.nsd_root, "nsddata/experiments/nsd/nsd_expdesign.mat")
    sharedix = scipy.io.loadmat(exp)["sharedix"].ravel() - 1

    stims_all = np.load(os.path.join(mridir, f"{subj}_stims.npy"))
    stims_ave = np.load(os.path.join(mridir, f"{subj}_stims_ave.npy"))

    is_test_each = np.isin(stims_all, sharedix)
    is_test_ave  = np.isin(stims_ave,  sharedix)

    np.save(os.path.join(mridir, f"{subj}_ids_each_tr.npy"), stims_all[~is_test_each])
    np.save(os.path.join(mridir, f"{subj}_ids_each_te.npy"), stims_all[ is_test_each])
    np.save(os.path.join(mridir, f"{subj}_ids_ave_tr.npy"),  stims_ave[ ~is_test_ave])
    np.save(os.path.join(mridir, f"{subj}_ids_ave_te.npy"),  stims_ave[  is_test_ave])

    print("Saved ids to:", mridir)

if __name__ == "__main__":
    main()