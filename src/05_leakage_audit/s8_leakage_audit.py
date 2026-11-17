"""S8 - LEAKAGE AUDIT: how much does each evaluation mistake inflate R2?  Explains the April result (0.52) vs the strict result (~0.2-0.4).
Same data, same model family, same target; only the PROTOCOL changes. Each protocol is repeated --reps times with different folds.
It is also run on a NULL target (targets permuted across SUBJECTS, within-subject structure preserved): a valid protocol must give R2 <= 0.

Protocols
  P0_correct        subject-level folds; scaler+PCA fitted inside training folds only; fixed k (set a priori)           <- the honest number
  P0b_select_in     as P0 but top-m feature selection by |corr| with y done INSIDE the training fold (the correct way to select)
  P1_scan_split     KFold over SCANS (a subject's scans fall in train and test)
  P2_pca_all        PCA/scaler fitted on ALL rows (train+test) before subject-level CV (transductive leakage)
  P3_select_all     top-m feature selection by |corr| with y computed on ALL rows before subject-level CV
  P4_best_of_grid   choose the best (k) per repetition BY TEST SCORE (optimistic model selection)
  P5_april_like     scan split + PCA on all rows + selection on all rows (everything wrong at once)
Models: ridge (RidgeCV) and gbm (xgboost if installed, else HistGradientBoosting).

Run (real, ~20-40 min):  python s8_leakage_audit.py --scans scan_table.csv --target MMSCORE --reps 30 --null-reps 15 --out-dir results_final/leakage
Needs only numpy/pandas/scikit-learn (xgboost optional).
"""
import argparse, os, warnings, time
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_XGB = False

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv"); ap.add_argument("--target", default="MMSCORE")
ap.add_argument("--reps", type=int, default=30); ap.add_argument("--null-reps", type=int, default=15)
ap.add_argument("--folds", type=int, default=5); ap.add_argument("--k", type=int, default=20, help="a-priori PCA components")
ap.add_argument("--m", type=int, default=50, help="features kept by selection protocols")
ap.add_argument("--grid", default="5,10,20,40,80"); ap.add_argument("--models", default="ridge,gbm")
ap.add_argument("--out-dir", default="results_final/leakage"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

d = pd.read_csv(a.scans)
DEEP = [f"f_{i}" for i in range(1536)]; DIRECT = ["age", "male", "e4", "f_1536", "f_1537"]
d = d.dropna(subset=[a.target] + DIRECT).reset_index(drop=True)
G = d.PTID.values; Y = d[a.target].values.astype(float); XD = d[DEEP].values.astype(float); XC = d[DIRECT].values.astype(float)
n = len(d); GRID = [int(x) for x in a.grid.split(",")]
print(f"{n} scans / {len(np.unique(G))} subjects | target {a.target} | xgboost {HAVE_XGB}")


def null_target(seed):
    """permute subject-level mean across subjects, keep each subject's within-subject deviations."""
    rng = np.random.RandomState(seed); s = pd.Series(Y); m = s.groupby(G).transform("mean").values
    subj = np.unique(G); perm = dict(zip(subj, rng.permutation(subj)))
    mean_by = pd.Series(m).groupby(G).first()
    return np.array([mean_by[perm[g]] for g in G]) + (Y - m)


def fit_predict(model, Xtr, ytr, Xte):
    if model == "ridge": return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, ytr).predict(Xte)
    m = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbosity=0) if HAVE_XGB else \
        HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200)
    return m.fit(Xtr, ytr).predict(Xte)


def pca_feats(tr_fit_idx, k, rows):
    sc = StandardScaler().fit(XD[tr_fit_idx]); p = PCA(min(k, len(tr_fit_idx) - 1), random_state=0).fit(sc.transform(XD[tr_fit_idx]))
    return p.transform(sc.transform(XD[rows])) / np.sqrt(p.explained_variance_[0])


def select(idx, y, m):
    z = StandardScaler().fit_transform(XD[idx]); yc = (y[idx] - y[idx].mean()) / (y[idx].std() + 1e-9)
    c = np.abs((z * yc[:, None]).mean(0)); return np.argsort(-c)[:m]


def r2(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def splits(scan_level, rep):
    if scan_level: return list(KFold(a.folds, shuffle=True, random_state=a.seed + rep).split(np.arange(n)))
    rng = np.random.RandomState(a.seed + rep); u = rng.permutation(np.unique(G)); fo = {g: i % a.folds for i, g in enumerate(u)}
    f = np.array([fo[g] for g in G]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(a.folds)]


def run(protocol, y, rep, model, cache):
    scan_level = protocol in ("P1_scan_split", "P5_april_like"); sp = splits(scan_level, rep)
    allidx = np.arange(n); out = np.zeros(n)
    kk = GRID if protocol == "P4_best_of_grid" else [a.k]; outs = {k: np.zeros(n) for k in kk}
    if protocol in ("P2_pca_all", "P5_april_like"):
        for k in kk:
            key = ("pca", k)
            if key not in cache: cache[key] = pca_feats(allidx, k, allidx)
    if protocol in ("P3_select_all", "P5_april_like"):
        if "sel" not in cache or cache["sel_y"] is not y: cache["sel"] = select(allidx, y, a.m); cache["sel_y"] = y
    for tr, te in sp:
        for k in kk:
            if protocol in ("P2_pca_all", "P5_april_like"):
                Z = cache[("pca", k)]; Xtr, Xte = np.hstack([Z[tr], XC[tr]]), np.hstack([Z[te], XC[te]])
            elif protocol == "P3_select_all":
                S = cache["sel"]; sc = StandardScaler().fit(XD[tr][:, S]); Xtr = np.hstack([sc.transform(XD[tr][:, S]), XC[tr]]); Xte = np.hstack([sc.transform(XD[te][:, S]), XC[te]])
            elif protocol == "P0b_select_in":
                S = select(tr, y, a.m); sc = StandardScaler().fit(XD[tr][:, S]); Xtr = np.hstack([sc.transform(XD[tr][:, S]), XC[tr]]); Xte = np.hstack([sc.transform(XD[te][:, S]), XC[te]])
            else:  # P0_correct, P1_scan_split, P4_best_of_grid
                sc = StandardScaler().fit(XD[tr]); p = PCA(min(k, len(tr) - 1), random_state=0).fit(sc.transform(XD[tr])); s0 = np.sqrt(p.explained_variance_[0])
                Xtr = np.hstack([p.transform(sc.transform(XD[tr])) / s0, XC[tr]]); Xte = np.hstack([p.transform(sc.transform(XD[te])) / s0, XC[te]])
            cs = StandardScaler().fit(Xtr[:, -XC.shape[1]:]); Xtr[:, -XC.shape[1]:] = cs.transform(Xtr[:, -XC.shape[1]:]); Xte[:, -XC.shape[1]:] = cs.transform(Xte[:, -XC.shape[1]:])
            outs[k][te] = fit_predict(model, Xtr, y[tr], Xte)
    return max(r2(y, outs[k]) for k in kk)       # P4 -> best-of-grid on the TEST predictions; others have a single k


PROTOCOLS = ["P0_correct", "P0b_select_in", "P1_scan_split", "P2_pca_all", "P3_select_all", "P4_best_of_grid", "P5_april_like"]
rows = []; t0 = time.time()
for label, reps in (("real", a.reps), ("NULL(permuted)", a.null_reps)):
    for model in a.models.split(","):
        for proto in PROTOCOLS:
            if model == "gbm" and proto == "P4_best_of_grid": continue
            vals = []
            for rep in range(reps):
                y = Y if label == "real" else null_target(a.seed + 1000 + rep); cache = {}
                vals.append(run(proto, y, rep, model, cache))
            vals = np.array(vals)
            rows.append(dict(target_type=label, model=model, protocol=proto, R2_mean=vals.mean(), R2_sd=vals.std(), p2_5=np.percentile(vals, 2.5), p97_5=np.percentile(vals, 97.5), reps=reps))
            print(f"[{time.time() - t0:6.0f}s] {label:15s} {model:5s} {proto:16s} R2={vals.mean():+.3f}  (sd {vals.std():.3f})", flush=True)
R = pd.DataFrame(rows)
base = R[R.protocol == "P0_correct"].set_index(["target_type", "model"]).R2_mean
R["inflation_vs_P0"] = [r.R2_mean - base[(r.target_type, r.model)] for r in R.itertuples()]
R.to_csv(os.path.join(a.out_dir, "leakage_audit.csv"), index=False)
pd.set_option("display.width", 200); print("\n", R.round(3).to_string(index=False))
print("\nHow to read it: P0 is the honest estimate. Any protocol far above P0 is inflated by leakage; on the NULL target a valid protocol stays <= 0.")
