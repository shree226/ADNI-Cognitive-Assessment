"""S17 - TEMPORAL analyses in ONE file (needs common.py next to it, plus scan_table.csv, richer_features.csv, rich_cols.txt and your ADNI csv folder for the visit tables).
Anchor = each participant's FIRST own scan. All inputs come from that scan (+ optional change to the 2nd scan); targets come from LATER clinical visits.

1. FORECAST regression   target = (score at ~h years after anchor) - (score at anchor)   [h = 1 and 2 years, +-win]; R2 on the CHANGE (the honest measure) and R2_level (how much of the future score's variance is explained).
2. SLOPE regression      target = per-participant OLS slope (points/year) over all visits after the anchor (>= min-visits visits spanning >= min-span years).
3. DECLINE classification label = clinically meaningful worsening over ~1 year (MMSE drop >= 2, CDR-SB rise >= 1.0, ADAS-Cog13 rise >= 3; fixed in advance, change with flags); AUC.
4. GENETICS x TIME       does APOE e4 change the rate of decline, and is that effect carried by baseline medial-temporal atrophy?  (two-stage OLS with HC3; linear mixed model too if statsmodels is installed)
Arms (same participants, same folds): A current score | B +demographics+APOE | C +imaging (no APOE) | D imaging+APOE | E no current score (screening) | F = D + scan-to-scan change.
Paired subject-bootstrap deltas answer: what do demographics+APOE add, what does imaging add, what does APOE add on top of imaging.  Models: ridge and ExtraTrees.
Run:  python s17_temporal.py --scans scan_table.csv --rich richer_features.csv --adni-dir "C:\\Users\\Student\\Downloads\\project\\adni_csvs" --out-dir results_final/S17
Smoke test: add --quick
"""
import argparse, os, re, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from common import find_csv, norm_id
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv"); ap.add_argument("--rich", default="richer_features.csv"); ap.add_argument("--rec", default="rich_cols.txt")
ap.add_argument("--adni-dir", required=True); ap.add_argument("--out-dir", default="results_final/S17")
ap.add_argument("--horizons", default="1,2"); ap.add_argument("--win", type=float, default=0.5); ap.add_argument("--min-visits", type=int, default=3); ap.add_argument("--min-span", type=float, default=1.5)
ap.add_argument("--decline-mmse", type=float, default=2.0); ap.add_argument("--decline-cdr", type=float, default=1.0); ap.add_argument("--decline-adas", type=float, default=3.0)
ap.add_argument("--reps", type=int, default=10); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
if a.quick: a.reps, a.boot = 2, 100
OUT = {"MMSCORE": ("MMSE", "MMSE", -1), "CDRSB": ("CDR", "CDR-SB", 1), "TOTAL13": ("ADAS", "ADAS-Cog13", 1)}     # col -> (table, nice name, direction of WORSENING)
OWN = ["f_1536", "f_1537"]

# ------------------------------------------------------------------ anchor table (one row per participant)
d = pd.read_csv(a.scans, parse_dates=["EXAMDATE"]).sort_values(["PTID", "EXAMDATE"])
if os.path.exists(a.rich):
    R_ = pd.read_csv(a.rich); d = d.merge(R_[[c for c in R_.columns if c not in d.columns or c == "Image_ID"]], on="Image_ID", how="left")
REC = [c for c in open(a.rec).read().strip().split(",") if c in d.columns] if os.path.exists(a.rec) else []
d["icv"] = d["icv_proxy"] if "icv_proxy" in d else d["ucsf_icv"]
IMG = list(dict.fromkeys(OWN + REC + ["icv"])); DEMO = ["age", "male", "edu"]; GEN = ["e4", "e4c"]; d["e4c"] = (d.e4 > 0).astype(float)
DELTA_SRC = [c for c in OWN + REC if c.startswith(("f_", "nv_", "hip_", "ent_"))][:40]
rows = []
for pid, g in d.groupby("PTID"):
    f = g.iloc[0]; r = f[["PTID", "EXAMDATE", "dx"] + DEMO + GEN + IMG].to_dict(); r["has_delta"] = 0.0
    if len(g) > 1:
        l = g.iloc[-1]; gap = (l.EXAMDATE - f.EXAMDATE).days / 365.25
        if gap >= 0.25:
            r["has_delta"] = 1.0
            for c in DELTA_SRC: r["d_" + c] = (l[c] - f[c]) / gap
    rows.append(r)
A = pd.DataFrame(rows); DELTA = [c for c in A.columns if c.startswith("d_")] + ["has_delta"]
for c in DELTA:
    if c != "has_delta": A[c] = A[c].fillna(0.0)
A = A.dropna(subset=["age", "male", "e4"]).reset_index(drop=True); N = len(A)
mt = [c for c in ("nv_hippocampus", "nv_amygdala", "nv_entorhinal", "nv_parahippocampal") if c in A]
A["mtl"] = A[mt].apply(lambda s: (s - s.mean()) / s.std()).sum(1) if len(mt) == 4 else A[OWN].apply(lambda s: (s - s.mean()) / s.std()).sum(1)
print(f"{N} participants | anchor = first own scan | imaging features {len(IMG) - 1}+ICV | delta features {len(DELTA) - 1}+flag ({int(A.has_delta.sum())} have a 2nd scan)")

# ------------------------------------------------------------------ visits -> current / future / slope / decline
visits = {}
for col, (tab, nice, _) in OUT.items():
    v = pd.read_csv(find_csv(a.adni_dir, tab), low_memory=False); v["PTID"] = v.PTID.map(norm_id); v["VISDATE"] = pd.to_datetime(v.VISDATE, errors="coerce")
    visits[col] = v[v[col] >= 0].dropna(subset=["VISDATE", col])[["PTID", "VISDATE", col]]
HZ = [float(x) for x in a.horizons.split(",")]
for col in OUT:
    v = visits[col]; cur, fut, slope, nv, span = {}, {h: {} for h in HZ}, {}, {}, {}
    for pid, t0 in zip(A.PTID, A.EXAMDATE):
        g = v[v.PTID == pid]
        if g.empty: continue
        yrs = (g.VISDATE - t0).dt.days.values / 365.25; val = g[col].values; k = np.argmin(np.abs(yrs))
        if abs(yrs[k]) <= 90 / 365.25: cur[pid] = val[k]
        for h in HZ:
            m = (yrs >= h - a.win) & (yrs <= h + a.win)
            if m.any(): j = np.where(m)[0][np.argmin(np.abs(yrs[m] - h))]; fut[h][pid] = val[j]
        m = yrs >= -90 / 365.25
        if m.sum() >= a.min_visits and np.ptp(yrs[m]) >= a.min_span: slope[pid] = np.polyfit(yrs[m], val[m], 1)[0]; nv[pid] = m.sum(); span[pid] = np.ptp(yrs[m])
    A["cur_" + col] = A.PTID.map(cur)
    for h in HZ: A[f"chg_{col}_{h:g}y"] = A.PTID.map(fut[h]) - A["cur_" + col]
    A["slope_" + col] = A.PTID.map(slope); A["nvis_" + col] = A.PTID.map(nv)
A.to_csv(os.path.join(a.out_dir, "temporal_subject_table.csv"), index=False)
cnt = {f"{OUT[c][1]}": {**{f"chg@{h:g}y": int(A[f"chg_{c}_{h:g}y"].notna().sum()) for h in HZ}, "slope": int(A["slope_" + c].notna().sum())} for c in OUT}
print("usable participants per target:", cnt)

# ------------------------------------------------------------------ CV machinery (subject-level, same folds for all arms)
def folds(n, rep):
    r = np.random.RandomState(a.seed + rep); p = r.permutation(n); f = np.empty(n, int); f[p] = np.arange(n) % a.folds
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(a.folds)]


def mk(model, clf):
    if model == "ridge": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000) if clf else RidgeCV(alphas=np.logspace(-2, 4, 13)))
    return make_pipeline(SimpleImputer(strategy="median"), (ExtraTreesClassifier if clf else ExtraTreesRegressor)(n_estimators=100 if a.quick else 250, max_features=0.3, min_samples_leaf=3, random_state=0, n_jobs=1))


def oof(X, y, model, clf):
    P = np.zeros(len(y))
    for rep in range(a.reps):
        for tr, te in folds(len(y), rep):
            m = mk(model, clf).fit(X[tr], y[tr]); P[te] += (m.predict_proba(X[te])[:, 1] if clf else m.predict(X[te])) / a.reps
    return P


def arms(col):
    c = ["cur_" + col]
    return {"A_current": c, "B_cur+demo+APOE": c + DEMO + GEN, "C_cur+demo+img": c + DEMO + IMG, "D_cur+demo+img+APOE": c + DEMO + IMG + GEN, "E_no_current(demo+img+APOE)": DEMO + IMG + GEN,
            "F_D+change": c + DEMO + IMG + GEN + DELTA}


PAIRS = [("B_cur+demo+APOE", "A_current", "demo+APOE | current score"), ("C_cur+demo+img", "B_cur+demo+APOE", "imaging | current+demo+APOE"),
         ("D_cur+demo+img+APOE", "C_cur+demo+img", "APOE | imaging+current+demo"), ("F_D+change", "D_cur+demo+img+APOE", "scan-to-scan change | D")]
rng = np.random.RandomState(a.seed)


def evaluate(task, label, df, col, target, clf, level=None):
    """df rows = participants with target; returns list of result rows (+ prints)."""
    y = df[target].values.astype(float); n = len(df); out = []
    if n < 30 or (clf and (y.sum() < 10 or (1 - y).sum() < 10)): print(f"  skip {task} {label}: n={n}" + (f", positives={int(y.sum())}" if clf else "")); return out
    boots = [rng.randint(0, n, n) for _ in range(a.boot)]
    metric = (lambda yy, pp: roc_auc_score(yy, pp)) if clf else (lambda yy, pp: 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    for model in ("ridge", "etr"):
        P = {}; B = {}
        for arm, cols in arms(col).items():
            P[arm] = oof(df[cols].values.astype(float), y, model, clf); B[arm] = np.array([metric(y[s], P[arm][s]) for s in boots if (not clf) or len(np.unique(y[s])) > 1])
            r = dict(task=task, target=label, n=n, model=model, arm=arm, metric="AUC" if clf else "R2", value=metric(y, P[arm]), lo=np.percentile(B[arm], 2.5), hi=np.percentile(B[arm], 97.5))
            if level is not None: fl = df["cur_" + col].values + y; r["R2_level"] = 1 - ((y - P[arm]) ** 2).sum() / ((fl - fl.mean()) ** 2).sum()
            out.append(r)
        for A_, B_, name in PAIRS:
            k = min(len(B[A_]), len(B[B_])); dl = B[A_][:k] - B[B_][:k]
            out.append(dict(task=task, target=label, n=n, model=model, arm="PAIR: " + name, metric="delta", value=metric(y, P[A_]) - metric(y, P[B_]), lo=np.percentile(dl, 2.5), hi=np.percentile(dl, 97.5),
                            p_boot=min(1, max(2 * min((dl <= 0).mean(), (dl >= 0).mean()), 1 / (k + 1)))))
    return out


res = []
for col, (tab, nice, sign) in OUT.items():
    for h in HZ:
        df = A.dropna(subset=["cur_" + col, f"chg_{col}_{h:g}y"]).reset_index(drop=True); print(f"\n[{nice}] forecast change at {h:g}y: n={len(df)}")
        res += evaluate("forecast", f"{nice} change@{h:g}y", df, col, f"chg_{col}_{h:g}y", False, level=True)
    df = A.dropna(subset=["cur_" + col, "slope_" + col]).reset_index(drop=True); print(f"[{nice}] slope: n={len(df)}")
    res += evaluate("slope", f"{nice} slope/yr", df, col, "slope_" + col, False)
    df = A.dropna(subset=["cur_" + col, f"chg_{col}_{HZ[0]:g}y"]).reset_index(drop=True)
    thr = {"MMSCORE": a.decline_mmse, "CDRSB": a.decline_cdr, "TOTAL13": a.decline_adas}[col]; df["decline"] = ((df[f"chg_{col}_{HZ[0]:g}y"] * sign) >= thr).astype(float)
    print(f"[{nice}] decline (worsening >= {thr} points over ~{HZ[0]:g}y): n={len(df)}, decliners={int(df.decline.sum())}")
    res += evaluate("decline-classification", f"{nice} decline>={thr}", df, col, "decline", True)
R = pd.DataFrame(res); R.to_csv(os.path.join(a.out_dir, "temporal_results.csv"), index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 30)
if len(R):
    for task in R.task.unique():
        T = R[(R.task == task) & (~R.arm.str.startswith("PAIR"))]
        print(f"\n=== {task.upper()}: {'AUC' if task.startswith('decline') else 'R2'} by arm ==="); print(T.pivot_table(index=["target", "model"], columns="arm", values="value").round(3).to_string())
        P_ = R[(R.task == task) & R.arm.str.startswith("PAIR")]
        print(f"--- paired differences ({task}), 95% subject-bootstrap CI ---"); print(P_[["target", "model", "arm", "value", "lo", "hi", "p_boot"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ genetics x time
def ols_hc3(y, X):
    X1 = np.column_stack([np.ones(len(y)), X]); XtXi = np.linalg.pinv(X1.T @ X1); b = XtXi @ X1.T @ y; e = y - X1 @ b; h = np.einsum("ij,jk,ik->i", X1, XtXi, X1)
    cov = XtXi @ (X1.T * (e / (1 - h)) ** 2) @ X1 @ XtXi; se = np.sqrt(np.diag(cov)); p = 2 * stats.t.sf(np.abs(b / se), len(y) - X1.shape[1]); return b, se, p


grow = []
Z = lambda s: (s - s.mean()) / (s.std() + 1e-12)
for col, (tab, nice, sign) in OUT.items():
    for label, sub in (("all", A), ("CN+MCI", A[A.dx.isin([1, 2])])):
        s = sub.dropna(subset=["slope_" + col] + DEMO + ["e4", "mtl"]).reset_index(drop=True)
        if len(s) < 30: continue
        y = s["slope_" + col].values; cov = DEMO
        for model, extra in (("e4 only", []), ("e4 + baseline MTL atrophy", ["mtl"])):
            b, se, p = ols_hc3(y, s[["e4"] + extra + cov].values.astype(float) * 1.0)
            grow.append(dict(outcome=nice, sample=label, n=len(s), model=model, e4_effect_on_slope_per_year=b[1], lo=b[1] - 1.96 * se[1], hi=b[1] + 1.96 * se[1], p=p[1],
                             direction="worse with e4" if b[1] * sign > 0 else "better with e4"))
G = pd.DataFrame(grow)
if len(G):
    G["attenuation_by_MTL_pct"] = np.nan
    for (o, s_), g in G.groupby(["outcome", "sample"]):
        if len(g) == 2 and abs(g.iloc[0].e4_effect_on_slope_per_year) > 1e-9: G.loc[g.index[1], "attenuation_by_MTL_pct"] = 100 * (1 - g.iloc[1].e4_effect_on_slope_per_year / g.iloc[0].e4_effect_on_slope_per_year)
    G.to_csv(os.path.join(a.out_dir, "genetics_time_twostage.csv"), index=False); print("\n=== GENETICS x TIME (two-stage: participant slope ~ e4 + covariates; points/year per e4 allele) ===")
    print(G.round(4).to_string(index=False)); print("attenuation = how much of the e4 effect on decline disappears once baseline medial-temporal atrophy is in the model (descriptive; cross-sectional ordering is assumed).")
try:
    import statsmodels.formula.api as smf
    mm = []
    for col, (tab, nice, sign) in OUT.items():
        v = visits[col].merge(A[["PTID", "EXAMDATE", "age", "male", "edu", "e4"]], on="PTID"); v["t"] = (v.VISDATE - v.EXAMDATE).dt.days / 365.25; v = v[v.t >= -90 / 365.25].dropna()
        v = v[v.groupby("PTID").PTID.transform("size") >= 2].rename(columns={col: "y"})
        try:
            f = smf.mixedlm("y ~ t*e4 + t*age + t*male + edu", v, groups=v["PTID"], re_formula="~t").fit(reml=True, method="lbfgs"); ci = f.conf_int().loc["t:e4"]
            mm.append(dict(outcome=nice, n_subj=v.PTID.nunique(), n_visits=len(v), e4_x_time_per_year=f.params["t:e4"], lo=ci[0], hi=ci[1], p=f.pvalues["t:e4"]))
        except Exception as e: print("mixed model failed for", nice, type(e).__name__)
    if mm: pd.DataFrame(mm).to_csv(os.path.join(a.out_dir, "genetics_time_mixedmodel.csv"), index=False); print("\n=== LINEAR MIXED MODEL (random intercept+slope): e4 x time ==="); print(pd.DataFrame(mm).round(4).to_string(index=False))
except ImportError:
    print("\n(statsmodels not installed: skipped the mixed model; the two-stage table above is the primary genetics-by-time result. pip install statsmodels to add it.)")
print(f"\nsaved to {a.out_dir}")