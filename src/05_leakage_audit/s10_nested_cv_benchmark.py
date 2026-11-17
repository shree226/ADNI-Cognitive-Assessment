"""S10 - NESTED cross-validation benchmark (the heavy, defensible version of the model comparison).
Outer loop : repeated subject-level K-fold (all scans of a subject together) - used ONLY for scoring.
Inner loop : subject-level 3-fold on the outer-TRAINING subjects, random search over hyperparameters (optuna if installed, else random search).
All preprocessing (scaler, PCA, target standardisation) is refitted inside every inner/outer training set.
Feature sets : F0_clin (age,sex,APOE)  F1_clin+ownvol  F2_clin+ownvol+deepPCA (k tuned)  F3_clin+ucsf_vol  (UCSF only if available)
Models       : ridge | svr (RBF) | gbm (xgboost if installed else HistGradientBoosting)
Targets      : MMSCORE, CDRSB, TOTAL13.   Paired subject-bootstrap deltas: F1-F0, F2-F1, F1-F3 per model and target.
This takes hours on purpose (it is the real thing).  Rough cost: reps x 5 folds x 4 sets x 3 models x 3 targets x trials x 3 inner fits.
Run:  python s10_nested_cv_benchmark.py --scans scan_table.csv --reps 3 --trials 30 --jobs 4 --out-dir results_final/nested
Quick check:  --reps 1 --trials 5 --targets MMSCORE --models ridge
"""
import argparse, os, warnings, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
warnings.filterwarnings("ignore")
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_XGB = False

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv"); ap.add_argument("--out-dir", default="results_final/nested")
ap.add_argument("--targets", default="MMSCORE,CDRSB,TOTAL13"); ap.add_argument("--models", default="ridge,svr,gbm")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--inner", type=int, default=3)
ap.add_argument("--trials", type=int, default=30); ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

d = pd.read_csv(a.scans); T = a.targets.split(","); MODELS = a.models.split(",")
CLIN = ["age", "male", "e4"]; OWN = ["f_1536", "f_1537"]; DEEP = [f"f_{i}" for i in range(1536)]
d["ucsf_h"] = d.ucsf_hippo / d.ucsf_icv * 1e3; d["ucsf_e"] = d.ucsf_ento / d.ucsf_icv * 1e3
d = d.dropna(subset=T + CLIN + OWN).reset_index(drop=True)
FSETS = {"F0_clin": (CLIN, False), "F1_clin_ownvol": (CLIN + OWN, False), "F2_clin_ownvol_deep": (CLIN + OWN, True)}
if d[["ucsf_h", "ucsf_e"]].notna().all(axis=1).mean() > 0.8:
    d = d.dropna(subset=["ucsf_h", "ucsf_e"]).reset_index(drop=True); FSETS["F3_clin_ucsfvol"] = (CLIN + ["ucsf_h", "ucsf_e"], False)
G = d.PTID.values; Yall = d[T].values.astype(float); XD = d[DEEP].values.astype(float); n = len(d)
print(f"{n} scans / {len(np.unique(G))} subjects | sets {list(FSETS)} | models {MODELS} | xgboost {HAVE_XGB}")
try:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING); HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False


def build(cols, use_deep, k, tr, te):
    sc = StandardScaler().fit(d.loc[tr, cols].values); Xtr, Xte = sc.transform(d.loc[tr, cols].values), sc.transform(d.loc[te, cols].values)
    if use_deep:
        s2 = StandardScaler().fit(XD[tr]); p = PCA(min(k, len(tr) - 1), random_state=0).fit(s2.transform(XD[tr])); s0 = np.sqrt(p.explained_variance_[0])
        Xtr = np.hstack([Xtr, p.transform(s2.transform(XD[tr])) / s0]); Xte = np.hstack([Xte, p.transform(s2.transform(XD[te])) / s0])
    return Xtr, Xte


def make_model(model, p):
    if model == "ridge": return Ridge(alpha=p["alpha"])
    if model == "svr": return SVR(C=p["C"], gamma=p["gamma"], epsilon=p["eps"])
    if HAVE_XGB: return XGBRegressor(n_estimators=p["n"], max_depth=p["depth"], learning_rate=p["lr"], subsample=p["sub"], verbosity=0)
    return HistGradientBoostingRegressor(max_iter=p["n"], max_depth=p["depth"], learning_rate=p["lr"])


def sample(model, use_deep, g):
    p = {"k": int(g.choice([4, 8, 16, 32, 64]))} if use_deep else {"k": 0}
    if model == "ridge": p["alpha"] = 10 ** g.uniform(-2, 4)
    elif model == "svr": p.update(C=10 ** g.uniform(-1, 2), gamma=10 ** g.uniform(-4, -1), eps=10 ** g.uniform(-2, -0.3))
    else: p.update(n=int(g.uniform(50, 400)), depth=int(g.choice([2, 3, 4])), lr=10 ** g.uniform(-2, -0.7), sub=g.uniform(0.7, 1.0))
    return p


def inner_cv(model, cols, use_deep, p, tr, y, seed):
    g = np.random.RandomState(seed); u = g.permutation(np.unique(G[tr])); fo = {x: i % a.inner for i, x in enumerate(u)}
    f = np.array([fo[x] for x in G[tr]]); err = 0.0
    for i in range(a.inner):
        itr, ite = tr[f != i], tr[f == i]; Xtr, Xte = build(cols, use_deep, p["k"], itr, ite); m_, s_ = y[itr].mean(), y[itr].std() + 1e-9
        err += ((make_model(model, p).fit(Xtr, (y[itr] - m_) / s_).predict(Xte) * s_ + m_ - y[ite]) ** 2).mean()
    return err / a.inner


def tune(model, cols, use_deep, tr, y, seed):
    if HAVE_OPTUNA:
        def objective(trial):
            g = np.random.RandomState(trial.number + seed); p = {"k": trial.suggest_categorical("k", [4, 8, 16, 32, 64]) if use_deep else 0}
            if model == "ridge": p["alpha"] = trial.suggest_float("alpha", 1e-2, 1e4, log=True)
            elif model == "svr": p.update(C=trial.suggest_float("C", 0.1, 100, log=True), gamma=trial.suggest_float("gamma", 1e-4, 1e-1, log=True), eps=trial.suggest_float("eps", 0.01, 0.5, log=True))
            else: p.update(n=trial.suggest_int("n", 50, 400), depth=trial.suggest_int("depth", 2, 4), lr=trial.suggest_float("lr", 0.01, 0.2, log=True), sub=trial.suggest_float("sub", 0.7, 1.0))
            return inner_cv(model, cols, use_deep, p, tr, y, seed)
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(objective, n_trials=a.trials)
        p = dict(st.best_params); p.setdefault("k", 0); return p
    g = np.random.RandomState(seed); best, bp = 1e18, None
    for _ in range(a.trials):
        p = sample(model, use_deep, g); e = inner_cv(model, cols, use_deep, p, tr, y, seed)
        if e < best: best, bp = e, p
    return bp


def outer_task(rep, fold, tr, te):
    out = {}
    for fs, (cols, use_deep) in FSETS.items():
        for model in MODELS:
            for j, t in enumerate(T):
                y = Yall[:, j]; p = tune(model, cols, use_deep, tr, y, a.seed + rep * 100 + fold)
                Xtr, Xte = build(cols, use_deep, p["k"], tr, te); m_, s_ = y[tr].mean(), y[tr].std() + 1e-9
                out[(fs, model, t)] = make_model(model, p).fit(Xtr, (y[tr] - m_) / s_).predict(Xte) * s_ + m_
    return rep, te, out


def folds(rep):
    g = np.random.RandomState(a.seed + rep); u = g.permutation(np.unique(G)); fo = {x: i % a.folds for i, x in enumerate(u)}
    f = np.array([fo[x] for x in G]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(a.folds)]


t0 = time.time()
tasks = [(rep, i, tr, te) for rep in range(a.reps) for i, (tr, te) in enumerate(folds(rep))]
res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(outer_task)(*tk) for tk in tasks)
P = {(fs, m, t): np.zeros(n) for fs in FSETS for m in MODELS for t in T}
for rep, te, out in res:
    for k, v in out.items(): P[k][te] += v / a.reps
print(f"done in {(time.time() - t0) / 60:.1f} min")

idx = pd.Series(np.arange(n)).groupby(G).apply(list).to_dict(); keys = list(idx); rng = np.random.RandomState(a.seed)
boots = [np.concatenate([idx[keys[i]] for i in rng.randint(0, len(keys), len(keys))]) for _ in range(a.boot)]
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
rows = []
for (fs, m, t), p in P.items():
    y = Yall[:, T.index(t)]; b = [r2(y[s], p[s]) for s in boots]
    rows.append(dict(featureset=fs, model=m, target=t, R2=r2(y, p), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5), MAE=float(np.abs(y - p).mean())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out_dir, "nested_results.csv"), index=False)
pairs = [("F1_clin_ownvol", "F0_clin"), ("F2_clin_ownvol_deep", "F1_clin_ownvol")] + ([("F1_clin_ownvol", "F3_clin_ucsfvol")] if "F3_clin_ucsfvol" in FSETS else [])
dr = []
for m in MODELS:
    for t in T:
        y = Yall[:, T.index(t)]
        for A_, B_ in pairs:
            dl = [r2(y[s], P[(A_, m, t)][s]) - r2(y[s], P[(B_, m, t)][s]) for s in boots]
            pv = min(1.0, max(2 * min(np.mean(np.array(dl) <= 0), np.mean(np.array(dl) >= 0)), 1 / (a.boot + 1)))
            dr.append(dict(model=m, target=t, comparison=f"{A_} - {B_}", delta=r2(y, P[(A_, m, t)]) - r2(y, P[(B_, m, t)]), lo=np.percentile(dl, 2.5), hi=np.percentile(dl, 97.5), p_boot=pv))
D = pd.DataFrame(dr); D.to_csv(os.path.join(a.out_dir, "nested_paired.csv"), index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False)); print("\nPAIRED"); print(D.round(3).to_string(index=False))
