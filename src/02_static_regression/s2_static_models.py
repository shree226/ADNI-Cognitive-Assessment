"""S2 - Static prediction on scan_table.csv: April-style baseline vs ROI-centric / APOE-conditioned / multi-task models.
All arms use the SAME scans/subjects, repeated subject-level GroupKFold, subject-bootstrap CIs, and paired deltas vs the April baseline.
Run:  python s2_static_models.py --scans scan_table.csv --out-dir results_final --reps 5
ASSUMPTION: deep features f_0..f_1535 = 3 blocks of 512 (whole brain / hippocampus / entorhinal, in the order your extractor
wrote them). Check your extraction script and fix --blocks if the order or sizes differ. f_1536, f_1537 = your hippocampus/entorhinal volumes.
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv")
ap.add_argument("--out-dir", default="results_final")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--k", type=int, default=10, help="PCA components per block")
ap.add_argument("--boot", type=int, default=500)
ap.add_argument("--blocks", default="0:512,512:1024,1024:1536")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

d = pd.read_csv(a.scans)
deep_all = [f"f_{i}" for i in range(1536)]
blocks = [[f"f_{i}" for i in range(*map(int, b.split(":")))] for b in a.blocks.split(",")]
CLIN = ["age", "male", "e4"]
OWNVOL = ["f_1536", "f_1537"]
TARGETS = ["MMSCORE", "CDRSB", "TOTAL13"]

d = d.dropna(subset=CLIN + TARGETS).copy()
d["ucsf_h"] = d.ucsf_hippo / d.ucsf_icv * 1e3; d["ucsf_e"] = d.ucsf_ento / d.ucsf_icv * 1e3
UCSF = ["ucsf_h", "ucsf_e"]
use_ucsf = d[UCSF].notna().all(axis=1).mean() >= 0.8
if use_ucsf:
    d = d.dropna(subset=UCSF)
d = d.reset_index(drop=True)
print(f"common set: {len(d)} scans / {d.PTID.nunique()} subjects | UCSF arm: {use_ucsf} | xgboost: {HAVE_XGB}")


class Feat(BaseEstimator, TransformerMixin):
    """per-block PCA (+ standardised clinical/volume columns, + optional APOE x PCA interactions)."""
    def __init__(self, blocks=(), k=10, extra=(), inter=False):
        self.blocks, self.k, self.extra, self.inter = blocks, k, extra, inter

    def fit(self, X, y=None):
        self.p_ = []
        for b in self.blocks:
            s = StandardScaler().fit(X[b]); pca = PCA(min(self.k, len(X) - 1), random_state=0).fit(s.transform(X[b]))
            self.p_.append((b, s, pca))
        self.es_ = StandardScaler().fit(X[list(self.extra)]) if self.extra else None
        self.cs_ = StandardScaler().fit(X[["e4"]])
        return self

    def transform(self, X):
        parts = []
        for b, s, pca in self.p_:
            z = pca.transform(s.transform(X[b])); z = z / np.sqrt(pca.explained_variance_[0])
            parts.append(z)
        e4 = self.cs_.transform(X[["e4"]])
        if self.inter:
            parts += [z * e4 for z in parts[:]]
        if self.es_ is not None:
            parts.append(self.es_.transform(X[list(self.extra)]))
        return np.hstack(parts) if parts else np.zeros((len(X), 0))


def ridge(): return RidgeCV(alphas=np.logspace(-2, 4, 13))
def logit(): return LogisticRegressionCV(Cs=10, max_iter=2000)
def gbm_r():
    return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbosity=0) if HAVE_XGB \
        else HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200)
def gbm_c():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, verbosity=0) if HAVE_XGB \
        else HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200)


# arm -> (Feat kwargs, model kind)   kind: 'lin' | 'gbm' | 'pls'
ARMS = {
    "clinical": (dict(blocks=[], extra=CLIN), "lin"),
    "april_pca_gbm": (dict(blocks=[deep_all], k=20, extra=CLIN), "gbm"),
    "own_volumes": (dict(blocks=[], extra=CLIN + OWNVOL), "lin"),
    "deep_pca_ridge": (dict(blocks=[deep_all], k=a.k, extra=CLIN), "lin"),
    "roi_pca_ridge": (dict(blocks=blocks, k=a.k, extra=CLIN), "lin"),
    "roi_apoe_cond": (dict(blocks=blocks, k=a.k, extra=CLIN, inter=True), "lin"),
    "roi_vol_apoe_cond": (dict(blocks=blocks, k=a.k, extra=CLIN + OWNVOL, inter=True), "lin"),
    "roi_pls_multitask": (dict(blocks=blocks, k=a.k, extra=CLIN + OWNVOL, inter=True), "pls"),
}
if use_ucsf:
    ARMS["ucsf_volumes"] = (dict(blocks=[], extra=CLIN + UCSF), "lin")
    ARMS["roi_ucsf_apoe_cond"] = (dict(blocks=blocks, k=a.k, extra=CLIN + UCSF, inter=True), "lin")
BASE = "april_pca_gbm"


def folds(groups, rep):
    rng = np.random.RandomState(a.seed + rep)
    u = rng.permutation(np.unique(groups)); fold_of = {g: i % a.folds for i, g in enumerate(u)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(a.folds)]


def oof(df, arm, tasks, classify):
    kw, kind = ARMS[arm]; P = {t: np.zeros(len(df)) for t in tasks}
    for rep in range(a.reps):
        for tr, te in folds(df.PTID.values, rep):
            F = Feat(**kw).fit(df.iloc[tr]); Xtr, Xte = F.transform(df.iloc[tr]), F.transform(df.iloc[te])
            if kind == "pls" and not classify:
                Y = df.iloc[tr][tasks].values; mu, sd = Y.mean(0), Y.std(0)
                m = PLSRegression(n_components=min(2, Xtr.shape[1]), scale=False).fit(Xtr, (Y - mu) / sd)
                pr = m.predict(Xte) * sd + mu
                for j, t in enumerate(tasks): P[t][te] += pr[:, j] / a.reps
                continue
            if kind == "pls":
                continue
            for t in tasks:
                y = df.iloc[tr][t].values
                if Xtr.shape[1] == 0: pr = np.full(len(te), y.mean())
                else:
                    m = (gbm_c() if classify else gbm_r()) if kind == "gbm" else (logit() if classify else ridge())
                    m.fit(Xtr, y)
                    pr = m.predict_proba(Xte)[:, 1] if classify else m.predict(Xte)
                P[t][te] += pr / a.reps
    return P


def metric(y, p, classify): return roc_auc_score(y, p) if classify else r2_score(y, p)


def boot(df, y, p_arm, p_base, classify, rng):
    idx = df.groupby("PTID").indices; keys = list(idx); est, dl = [], []
    for _ in range(a.boot):
        s = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        if classify and len(np.unique(y[s])) < 2: continue
        e = metric(y[s], p_arm[s], classify); est.append(e)
        if p_base is not None: dl.append(e - metric(y[s], p_base[s], classify))
    return np.percentile(est, [2.5, 97.5]), (np.percentile(dl, [2.5, 97.5]) if dl else (np.nan, np.nan))


rows = []
rng = np.random.RandomState(a.seed)
dxs = d.dropna(subset=["dx"]); dxs = dxs[dxs.dx.isin([1, 3])].assign(ad=lambda x: (x.dx == 3).astype(int)).reset_index(drop=True)
jobs = [("regression", d, TARGETS, False)]
if dxs.ad.nunique() == 2 and dxs.ad.sum() >= 10:
    jobs.append(("CN_vs_AD", dxs, ["ad"], True))
for name, df, tasks, classify in jobs:
    print(f"\n=== {name}: {len(df)} scans / {df.PTID.nunique()} subjects")
    preds = {arm: oof(df, arm, tasks, classify) for arm in ARMS if not (classify and ARMS[arm][1] == "pls")}
    for arm, P in preds.items():
        for t in tasks:
            y = df[t].values; lo_hi, dl = boot(df, y, P[t], preds[BASE][t] if arm != BASE else None, classify, rng)
            rows.append(dict(task=name, target=t, arm=arm, metric="AUC" if classify else "R2", est=metric(y, P[t], classify),
                             lo=lo_hi[0], hi=lo_hi[1], delta_vs_april=(metric(y, P[t], classify) - metric(y, preds[BASE][t], classify)) if arm != BASE else 0,
                             d_lo=dl[0], d_hi=dl[1], n_scans=len(df), n_subj=df.PTID.nunique()))
res = pd.DataFrame(rows); res.to_csv(os.path.join(a.out_dir, "static_results.csv"), index=False)
pd.set_option("display.width", 200)
print(res.round(3).drop(columns=["metric", "n_scans", "n_subj"]).to_string(index=False))
print(f"\nsaved {a.out_dir}/static_results.csv  (delta CI excluding 0 = difference vs April baseline is reliable)")
