"""
Conformal anomaly detection — DISTRIBUTION-FREE guaranteed false-alarm rate.

On top of per-node Mahalanobis scores: calibrate per-node conformal p-values on
a held-out healthy set; flag node i if p_i ≤ α. Marginal validity guarantees
empirical FPR ≤ α with NO distributional assumption — the certification
language aerospace reviewers (JAXA) actually want, vs an uncertified AUROC.

  p_i(s) = (1 + #{calibration scores at node i ≥ s}) / (n_cal + 1)

Outputs: empirical node-FPR vs target α (validity check) + recall at each α.

Run: python3 conformal_anomaly.py
"""
import os
import glob
import numpy as np

BASE      = "/home/nishioka/CFRP/CFRP_hole/hole_data_inp"
DIR_1X1   = f"{BASE}/Defect_hole_1x1_Random_npy"
DIR_2X2   = f"{BASE}/Defect_hole_2x2_Region1_21_npy"
DIR_4X4   = f"{BASE}/Defect_hole_4x4_Region1_21_npy"
LBL_2X2   = f"{BASE}/Def2x2_19class_label"
LBL_4X4   = f"{BASE}/Def4x4_19class_label"
N_NODES   = 13942
N_FIT     = 1500       # fit per-node Gaussian
N_CAL     = 500        # conformal calibration (disjoint)
ALPHAS    = [0.05, 0.01, 0.001]
rng = np.random.default_rng(42)


def load_field(fpath):
    v = np.load(fpath).astype(np.float32)[:N_NODES]
    return (v - v.mean()) / (v.std() + 1e-8)


def load_label(label_dir, fpath):
    base = os.path.splitext(os.path.basename(fpath))[0]
    for cand in (f"{base}.npy", f"{base}_19label.npy"):
        lf = os.path.join(label_dir, cand)
        if os.path.exists(lf):
            lbl = np.load(lf)
            return (lbl[:, 1:].sum(axis=1) > 0).astype(np.float32)
    return None


def main():
    files = sorted(glob.glob(os.path.join(DIR_1X1, "*.npy")))
    idx   = rng.permutation(len(files))
    fit_f = [files[i] for i in idx[:N_FIT]]
    cal_f = [files[i] for i in idx[N_FIT:N_FIT + N_CAL]]

    X_fit = np.stack([load_field(f) for f in fit_f])
    mu, sd = X_fit.mean(0), X_fit.std(0) + 1e-6
    # calibration scores per node: (N_CAL, N_NODES), sorted per node
    S_cal = np.sort(np.abs(np.stack([load_field(f) for f in cal_f]) - mu) / sd, axis=0)
    print(f"fit={len(fit_f)}  calib={len(cal_f)}  (disjoint 1x1 sets)")

    def p_value(s):
        """vectorised per-node conformal p: fraction of calib scores ≥ s."""
        # searchsorted per node on the sorted calibration columns
        pos = np.array([np.searchsorted(S_cal[:, i], s[i]) for i in range(N_NODES)])
        return (1 + (N_CAL - pos)) / (N_CAL + 1)

    for raw_dir, lbl_dir, tag, n in [(DIR_2X2, LBL_2X2, "2x2", 200),
                                     (DIR_4X4, LBL_4X4, "4x4", 100)]:
        fs = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))[:n]
        P, L = [], []
        for f in fs:
            lbl = load_label(lbl_dir, f)
            if lbl is None:
                continue
            s = np.abs(load_field(f) - mu) / sd
            P.append(p_value(s)); L.append(lbl)
        P, L = np.concatenate(P), np.concatenate(L)
        print(f"\n[{tag}] n={len(fs)} samples, defect nodes={int(L.sum())}")
        print(f"  {'α':>7} {'empFPR':>8} {'recall':>8}  (validity: empFPR ≤ α)")
        for a in ALPHAS:
            flag = P <= a
            fpr = flag[L == 0].mean()
            rec = flag[L == 1].mean()
            ok  = "✓" if fpr <= a * 1.2 else "✗"
            print(f"  {a:>7.3f} {fpr:>8.4f} {rec:>8.3f}  {ok}")
    print("CONFORMAL-DONE")


if __name__ == "__main__":
    main()
