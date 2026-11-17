"""S3 - Temporal prediction from pairs of own scans of the same subject (consecutive scans, gap >= --min-gap years).
Targets (per year): change in MMSE / CDR-SB / ADAS13, and change in the UCSF FreeSurfer hippocampal volume (independent reference,
so predicting it from YOUR embedding change tests whether your features track real atrophy).
Arms: April-style raw feature-difference Ridge vs fold-PCA / ROI / APOE-conditioned difference models. Subject-level folds, subject-bootstrap CIs.
Run:  python s3_temporal_models.py --scans scan_table.csv --out-dir results_final
"""
import argparse, os, warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv"); ap.add_argument("--out-dir", default="results_final")
ap.add_argument("--min-gap", type=float, default=0.25); ap.add_argument("--k", type=int, default=10)
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--boot", type=int, default=500); ap.add_argument("--blocks", default="0:512,512:1024,1024:1536")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

d = pd.read_csv(a.scans, parse_dates=["EXAMDATE"]).sort_values(["PTID", "EXAMDATE"]).reset_index(drop=True)
deep_all = [f"f_{i}" for i in range(1536)]
blocks = [[f"f_{i}" for i in range(*map(int, b.split(":")))] for b in a.blocks.split(",")]
d["ucsf_h"] = d.ucsf_hippo / d.ucsf_icv * 1e3
SC = {"MMSCORE": "dMMSE_yr", "CDRSB": "dCDRSB_yr", "TOTAL13": "dADAS13_yr", "ucsf_h": "dUCSFhippo_pct_yr"}

# ---- consecutive pairs
rows_i, rows_j, gaps = [], [], []
for pid, g in d.groupby("PTID"):
    idx = g.index.values
    for i in range(len(idx) - 1):
        gap = (d.EXAMDATE[idx[i + 1]] - d.EXAMDATE[idx[i]]).days / 365.25
        if gap >= a.min_gap: rows_i.append(idx[i]); rows_j.append(idx[i + 1]); gaps.append(gap)
Pi, Pj = d.loc[rows_i].reset_index(drop=True), d.loc[rows_j].reset_index(drop=True)
gaps = np.array(gaps)
T = pd.DataFrame({"PTID": Pi.PTID, "gap": gaps, "age": Pi.age, "male": Pi.male, "e4": Pi.e4})
for c, n in SC.items():
    T[n] = (Pj[c] - Pi[c]) / gaps
    if c == "ucsf_h": T[n] = T[n] / Pi[c] * 100
T = T.dropna(subset=["age", "male", "e4"]); keep = T.index.values
Pi, Pj = Pi.loc[keep].reset_index(drop=True), Pj.loc[keep].reset_index(drop=True); T = T.reset_index(drop=True)
print(f"pairs: {len(T)} from {T.PTID.nunique()} subjects | median gap {T.gap.median():.2f} y")
for n in SC.values(): print(f"  {n:20s} available: {int(T[n].notna().sum())}")


class Emb(BaseEstimator, TransformerMixin):
    def __init__(self, blocks=(), k=10): self.blocks, self.k = blocks, k
    def fit(self, X, y=None):
        self.p_ = [(b, StandardScaler().fit(X[b])) for b in self.blocks]
        self.p_ = [(b, s, PCA(min(self.k, len(X) - 1), random_state=0).fit(s.transform(X[b]))) for b, s in self.p_]
        return self
    def transform(self, X):
        return np.hstack([p.transform(s.transform(X[b])) / np.sqrt(p.explained_variance_[0]) for b, s, p in self.p_])


def feats(arm, tr, te):
    clin = lambda i: np.c_[T.gap[i], T.age[i], T.male[i], T.e4[i]]
    cs = StandardScaler().fit(clin(tr)); Ctr, Cte = cs.transform(clin(tr)), cs.transform(clin(te))
    if arm == "clinical_gap": return Ctr, Cte
    if arm == "april_raw_diff":
        s = StandardScaler().fit(pd.concat([Pi.loc[tr, deep_all], Pj.loc[tr, deep_all]]))
        f = lambda i: np.c_[s.transform(Pj.loc[i, deep_all]) - s.transform(Pi.loc[i, deep_all])]
        return np.c_[f(tr), Ctr], np.c_[f(te), Cte]
    bl = [deep_all] if arm == "deep_pca_diff" else blocks
    E = Emb(bl, a.k).fit(pd.concat([Pi.loc[tr], Pj.loc[tr]]))
    dz = lambda i: E.transform(Pj.loc[i]) - E.transform(Pi.loc[i])
    Dtr, Dte = dz(tr), dz(te)
    if arm == "roi_diff_apoe_cond":
        e = cs.transform(clin(tr))[:, 3:4], cs.transform(clin(te))[:, 3:4]
        return np.c_[Dtr, Dtr * e[0], Ctr], np.c_[Dte, Dte * e[1], Cte]
    return np.c_[Dtr, Ctr], np.c_[Dte, Cte]


ARMS = ["clinical_gap", "april_raw_diff", "deep_pca_diff", "roi_pca_diff", "roi_diff_apoe_cond"]
BASE = "april_raw_diff"


def folds(rep):
    rng = np.random.RandomState(a.seed + rep); u = rng.permutation(T.PTID.unique()); fo = {g: i % a.folds for i, g in enumerate(u)}
    f = T.PTID.map(fo).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(a.folds)]


res = []; rng = np.random.RandomState(a.seed)
for tname in SC.values():
    ok = T[tname].notna().values & np.isfinite(T[tname].values)
    if ok.sum() < 40: print("skip", tname, "(too few pairs)"); continue
    sub = np.where(ok)[0]; y = T[tname].values
    P = {}
    for arm in ARMS:
        p = np.zeros(len(T))
        for rep in range(a.reps):
            for tr, te in folds(rep):
                tr, te = np.intersect1d(tr, sub), np.intersect1d(te, sub)
                if len(te) == 0: continue
                Xtr, Xte = feats(arm, tr, te); m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, y[tr]); p[te] += m.predict(Xte) / a.reps
        P[arm] = p
    idx = {k: np.intersect1d(v, sub) for k, v in T.groupby("PTID").indices.items()}; keys = [k for k in idx if len(idx[k])]
    for arm in ARMS:
        e, dl = [], []
        for _ in range(a.boot):
            s = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))]); r = r2_score(y[s], P[arm][s]); e.append(r)
            dl.append(r - r2_score(y[s], P[BASE][s]))
        r0 = r2_score(y[sub], P[arm][sub])
        res.append(dict(target=tname, arm=arm, R2=r0, lo=np.percentile(e, 2.5), hi=np.percentile(e, 97.5),
                        delta_vs_april=0 if arm == BASE else r0 - r2_score(y[sub], P[BASE][sub]),
                        d_lo=np.nan if arm == BASE else np.percentile(dl, 2.5), d_hi=np.nan if arm == BASE else np.percentile(dl, 97.5),
                        n_pairs=len(sub), n_subj=len(keys)))
R = pd.DataFrame(res); R.to_csv(os.path.join(a.out_dir, "temporal_results.csv"), index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False))
print(f"\nsaved {a.out_dir}/temporal_results.csv")
