#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Part A: leakage-aware ADNI tabular feature and ensemble experiments.

Primary analysis uses the earliest eligible scan per PTID (one row/person).
Every arm shares identical repeated subject-level outer folds. Imputation,
scaling, MedNet PCA, hyperparameter selection, and blend weights are learned
inside each outer-training split only.

Optional FastSurfer surface features are parsed from:
  <surface-root>/<Image_ID>/stats/lh.aparc.DKTatlas.mapped.stats
  <surface-root>/<Image_ID>/stats/rh.aparc.DKTatlas.mapped.stats
with fallbacks to lh/rh.aparc.DKTatlas.stats. These files are produced by the
FastSurfer surface module; --seg_only outputs do not contain them.

Example:
  python partA_tabular_improvement.py --scans scan_table.csv \
      --rich richer_features.csv --surface-root FastSurfer_output_from_HDBET \
      --out-dir results_final/PartA --folds 5 --reps 3 --boot 1000

Permutation/null run:
  python partA_tabular_improvement.py --scans scan_table.csv \
      --rich richer_features.csv --surface-root FastSurfer_output_from_HDBET \
      --out-dir results_final/PartA_null --permute-targets --reps 1 --boot 500

Do not interpret the highest observed R2 as a guaranteed or expected target.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

TARGETS = ["MMSCORE", "CDRSB", "TOTAL13"]
LABELS = {"MMSCORE": "MMSE", "CDRSB": "CDR-SB", "TOTAL13": "ADAS-Cog13"}
# A priori DKT regions commonly implicated in AD-related morphometry.
AD_REGIONS = [
    "entorhinal", "parahippocampal", "fusiform", "inferiortemporal",
    "middletemporal", "superiortemporal", "temporalpole", "bankssts",
    "precuneus", "posteriorcingulate", "isthmuscingulate", "insula",
    "inferiorparietal", "superiorparietal", "supramarginal", "angular",
    "rostralmiddlefrontal", "caudalmiddlefrontal", "rostralanteriorcingulate",
    "caudalanteriorcingulate", "lateralorbitofrontal", "medialorbitofrontal",
]
SURF_MEASURES = ["thickness", "area", "gmvol", "thickstd", "foldind"]


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-aware tabular ADNI modelling with FastSurfer surface features.")
    p.add_argument("--scans", default="scan_table.csv")
    p.add_argument("--rich", default="richer_features.csv")
    p.add_argument("--surface-root", default="", help="FastSurfer subject root containing <Image_ID>/stats/")
    p.add_argument("--fold-map", default="", help="Optional S16_outer_folds.csv; uses exactly those participant folds (forces one repetition).")
    p.add_argument("--surface-left", default="lh.aparc.DKTatlas.mapped.stats")
    p.add_argument("--surface-right", default="rh.aparc.DKTatlas.mapped.stats")
    p.add_argument("--out-dir", default="results_final/PartA")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--mednet-pca", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1, help="GridSearchCV workers; use 1 for predictable Windows behaviour.")
    p.add_argument("--models", default="ridge,extratrees,histgb", help="Comma list: ridge,extratrees,histgb,elastic,svr")
    p.add_argument("--include-mednet", action="store_true", help="Include f_0..f_1535 as a PCA-reduced feature family.")
    p.add_argument("--no-surface", action="store_true", help="Do not search for or parse FastSurfer surface stats.")
    p.add_argument("--permute-targets", action="store_true", help="Shuffle target vectors across participants for a null run.")
    p.add_argument("--quick", action="store_true", help="Smoke test: 3 folds, 1 repeat, restricted grids, fewer trees.")
    return p.parse_args()


def numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def make_male(d: pd.DataFrame) -> pd.Series:
    if "male" in d:
        x = pd.to_numeric(d["male"], errors="coerce")
        vals = set(x.dropna().unique())
        if vals.issubset({0, 1}):
            return x
    if "sex" in d:
        s = d["sex"]
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            vals = set(x.dropna().unique())
            if vals.issubset({0, 1}):
                return x
            return x.map({1: 1, 2: 0})
        return s.astype(str).str.strip().str.lower().map({"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "2": 0})
    raise ValueError("scan_table.csv must contain 'male' or 'sex'.")


def merge_inputs(scans_path: str, rich_path: str) -> pd.DataFrame:
    if not os.path.isfile(scans_path):
        raise FileNotFoundError(f"Cannot find scan table: {scans_path}")
    d = pd.read_csv(scans_path, low_memory=False)
    required = ["PTID", "Image_ID"] + TARGETS + ["age", "e4"]
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise ValueError(f"scan_table.csv missing required columns: {miss}")
    d["PTID"] = d["PTID"].astype(str).str.strip()
    d["Image_ID"] = d["Image_ID"].astype(str).str.strip()
    if os.path.isfile(rich_path):
        r = pd.read_csv(rich_path, low_memory=False)
        if "Image_ID" not in r:
            raise ValueError("richer_features.csv must contain Image_ID for a scan-level merge.")
        r["Image_ID"] = r["Image_ID"].astype(str).str.strip()
        if r["Image_ID"].duplicated().any():
            raise ValueError("richer_features.csv has duplicate Image_ID values; resolve these before merging.")
        rcols = [c for c in r.columns if c != "Image_ID"]
        d = d.merge(r, on="Image_ID", how="left", suffixes=("", "__rich"), validate="many_to_one")
        for c in rcols:
            if c + "__rich" in d:
                d[c] = d[c + "__rich"]
                d.drop(columns=[c + "__rich"], inplace=True)
        print(f"Merged {len(rcols)} refreshed feature columns from {rich_path}.")
    else:
        print(f"WARNING: rich file not found ({rich_path}); using scan-table features only.")
    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["male"] = make_male(d)
    d["e4"] = pd.to_numeric(d["e4"], errors="coerce").clip(0, 2)
    for c in TARGETS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "edu" in d:
        d["edu"] = pd.to_numeric(d["edu"], errors="coerce")
    # A complete common outcome cohort prevents outcome-specific case selection.
    d = d.dropna(subset=["PTID", "Image_ID", "age", "male", "e4"] + TARGETS).copy()
    if "EXAMDATE" in d:
        d["EXAMDATE"] = pd.to_datetime(d["EXAMDATE"], errors="coerce")
        d = d.sort_values(["PTID", "EXAMDATE", "Image_ID"], kind="mergesort")
    else:
        d = d.sort_values(["PTID", "Image_ID"], kind="mergesort")
    n_scan = len(d)
    d = d.groupby("PTID", sort=False, as_index=False).head(1).reset_index(drop=True)
    print(f"Cohort: {n_scan} eligible scan rows -> {len(d)} baseline participants (one earliest scan/PTID).")
    return d


def find_subject_stats(root: Path, image_id: str, filename: str) -> Path | None:
    direct_candidates = [
        root / image_id / "stats" / filename,
        root / image_id / filename,
    ]
    for q in direct_candidates:
        if q.is_file():
            return q
    # Search only folders with the exact Image_ID path component to avoid loose ID matches.
    for p in root.rglob(filename):
        if image_id in p.parts:
            return p
    return None


def parse_freesurfer_stats(path: Path) -> dict[str, dict[str, float]]:
    """Read FreeSurfer/FastSurfer aparc stats: one row per cortical parcel."""
    headers = None
    rows = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            st = line.strip()
            if not st:
                continue
            if st.startswith("#"):
                if "ColHeaders" in st:
                    # Typical: '# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ...'
                    headers = st.split("ColHeaders", 1)[1].strip().split()
                continue
            parts = st.split()
            if len(parts) < 5:
                continue
            if headers is None:
                headers = ["StructName", "NumVert", "SurfArea", "GrayVol", "ThickAvg", "ThickStd", "MeanCurv", "GausCurv", "FoldInd", "CurvInd"]
            if len(parts) < len(headers):
                continue
            rec = {}
            for h, v in zip(headers[1:], parts[1:]):
                try:
                    rec[h] = float(v)
                except ValueError:
                    rec[h] = np.nan
            rows[parts[0].lower()] = rec
    # Standardise known header aliases.
    out = {}
    for region, rec in rows.items():
        out[region] = {
            "area": rec.get("SurfArea", rec.get("surfarea", np.nan)),
            "gmvol": rec.get("GrayVol", rec.get("grayvol", np.nan)),
            "thickness": rec.get("ThickAvg", rec.get("thickavg", np.nan)),
            "thickstd": rec.get("ThickStd", rec.get("thickstd", np.nan)),
            "foldind": rec.get("FoldInd", rec.get("foldind", np.nan)),
        }
    return out


def extract_surface_table(df: pd.DataFrame, surface_root: str, left_name: str, right_name: str, out_path: Path) -> pd.DataFrame:
    if not surface_root:
        return pd.DataFrame({"Image_ID": df["Image_ID"].astype(str)})
    root = Path(surface_root)
    if not root.is_dir():
        print(f"WARNING: surface root does not exist: {root}; cortical surface features skipped.")
        return pd.DataFrame({"Image_ID": df["Image_ID"].astype(str)})
    rows, n_ok = [], 0
    for image_id in df["Image_ID"].astype(str):
        row = {"Image_ID": image_id}
        lf = find_subject_stats(root, image_id, left_name)
        rf = find_subject_stats(root, image_id, right_name)
        # Fallback to non-mapped DKT stats, but never mix arbitrary atlases.
        if lf is None:
            lf = find_subject_stats(root, image_id, "lh.aparc.DKTatlas.stats")
        if rf is None:
            rf = find_subject_stats(root, image_id, "rh.aparc.DKTatlas.stats")
        L = parse_freesurfer_stats(lf) if lf else {}
        R = parse_freesurfer_stats(rf) if rf else {}
        if L or R:
            n_ok += 1
        for region in AD_REGIONS:
            l = L.get(region, {})
            r = R.get(region, {})
            for measure in SURF_MEASURES:
                lv = l.get(measure, np.nan)
                rv = r.get(measure, np.nan)
                row[f"surf_lh_{region}_{measure}"] = lv
                row[f"surf_rh_{region}_{measure}"] = rv
                if measure in ("area", "gmvol"):
                    row[f"surf_{region}_{measure}_sum"] = np.nansum([lv, rv]) if np.isfinite(lv) or np.isfinite(rv) else np.nan
                elif measure == "thickness":
                    la, ra = l.get("area", np.nan), r.get("area", np.nan)
                    vals = [(lv, la), (rv, ra)]
                    vals = [(v, w) for v, w in vals if np.isfinite(v) and np.isfinite(w) and w > 0]
                    row[f"surf_{region}_thickness"] = (sum(v*w for v, w in vals) / sum(w for _, w in vals)) if vals else np.nan
                    row[f"surf_{region}_thickness_asym"] = ((lv-rv)/(abs(lv)+abs(rv)+1e-8)) if np.isfinite(lv) and np.isfinite(rv) else np.nan
                elif measure == "thickstd":
                    row[f"surf_{region}_thickstd_mean"] = np.nanmean([lv, rv]) if np.isfinite(lv) or np.isfinite(rv) else np.nan
                elif measure == "foldind":
                    row[f"surf_{region}_foldind_sum"] = np.nansum([lv, rv]) if np.isfinite(lv) or np.isfinite(rv) else np.nan
        rows.append(row)
    surf = pd.DataFrame(rows)
    if n_ok == 0:
        # Do not expose an all-empty surface feature family to the estimators.
        surf = pd.DataFrame({"Image_ID": df["Image_ID"].astype(str)})
        print("WARNING: no DKT surface stats found. Check that FastSurfer surface reconstruction ran (not --seg_only) and folder structure.")
    surf.to_csv(out_path, index=False)
    print(f"FastSurfer surface stats found for {n_ok}/{len(df)} participants; saved {out_path}.")
    return surf


def build_features(d: pd.DataFrame, surface_cols: list[str], include_mednet: bool):
    demo = [c for c in ["age", "male", "edu", "e4"] if c in d.columns]
    own = [c for c in ["f_1536", "f_1537", "ucsf_hippo", "ucsf_ento"] if c in d.columns]
    # Additional UCSF/ADNI reference volumes are allowed as an imaging comparator,
    # but not dates, diagnoses, or any cognitive test-derived covariate.
    ucsf = [c for c in d.columns if c.lower().startswith("ucsf_") and c not in own]
    rich = [c for c in d.columns if c.startswith(("nv_", "asym_", "hip_", "ent_", "av_", "sym_"))]
    # Derive ICV-normalised label volumes from S11 vol_<label> if av_* were not merged.
    icv_col = next((c for c in ["icv_proxy", "ucsf_icv", "icv"] if c in d.columns), None)
    if icv_col:
        icv = pd.to_numeric(d[icv_col], errors="coerce").replace(0, np.nan)
        for c in [x for x in d.columns if re.fullmatch(r"vol_\d+", x)]:
            av = "av_" + c[4:]
            if av not in d:
                d[av] = pd.to_numeric(d[c], errors="coerce") / icv * 1e3
            rich.append(av)
    # Pre-defined ratios (pure arithmetic from MRI features; no outcomes used).
    def ratio(name, num, den):
        if all(c in d for c in num) and all(c in d for c in den):
            d[name] = d[num].sum(axis=1) / (d[den].sum(axis=1) + 1e-6)
            rich.append(name)
    ratio("sym_hip_vent", ["nv_hippocampus"], ["nv_lateral_ventricle"])
    ratio("sym_mtl_vent", ["nv_hippocampus", "nv_amygdala", "nv_entorhinal", "nv_parahippocampal"], ["nv_lateral_ventricle", "nv_inf_lat_ventricle"])
    if "nv_hippocampus" in d:
        d["sym_hip_asym_abs"] = pd.to_numeric(d.get("asym_hippocampus"), errors="coerce").abs() if "asym_hippocampus" in d else np.nan
        rich.append("sym_hip_asym_abs")
    rich = list(dict.fromkeys(rich + own + ucsf))
    mednet = []
    if include_mednet:
        mednet = [c for c in d.columns if re.fullmatch(r"f_\d+", c) and int(c.split("_")[1]) <= 1535]
    # Never include diagnosis, visit dates, identifiers, targets, or clinical test subscales.
    banned = set(TARGETS + ["PTID", "Image_ID", "EXAMDATE", "dx", "DX", "scan_idx", "uid", "Subject_ID", "age_at_scan"])
    def numeric_keep(cols):
        # Feature membership is defined from source/name a priori. Do not use
        # whole-cohort missingness or variance to select predictors: imputation
        # and scaling are learned separately inside each training fold.
        result = []
        for c in cols:
            if c in d.columns and c not in banned:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                result.append(c)
        return result
    demo = numeric_keep(demo)
    rich = numeric_keep(rich)
    surface_cols = numeric_keep(surface_cols)
    mednet = numeric_keep(mednet)
    # The `rich` family includes clinical covariates and own volumes to retain the
    # same core information for all incremental comparisons.
    core = list(dict.fromkeys(demo + rich))
    arms = {"rich": {"regular": core, "mednet": []}}
    if surface_cols:
        arms["rich_surface"] = {"regular": list(dict.fromkeys(core + surface_cols)), "mednet": []}
    if mednet:
        arms["rich_mednet"] = {"regular": core, "mednet": mednet}
        if surface_cols:
            arms["rich_surface_mednet"] = {"regular": list(dict.fromkeys(core + surface_cols)), "mednet": mednet}
    return arms, demo, rich, surface_cols, mednet


def make_preprocessor(regular_cols, mednet_cols, n_comp):
    trans = []
    if regular_cols:
        trans.append(("regular", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), regular_cols))
    if mednet_cols:
        trans.append(("mednet_pca", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_comp, svd_solver="auto", random_state=0)),
        ]), mednet_cols))
    return ColumnTransformer(trans, remainder="drop", sparse_threshold=0.0)


def model_spec(name: str, seed: int, quick: bool):
    if name == "ridge":
        est = Ridge()
        grid = {"model__alpha": ([1.0, 10.0] if quick else [0.1, 1.0, 10.0, 100.0, 300.0])}
    elif name == "extratrees":
        est = ExtraTreesRegressor(n_estimators=80 if quick else 300, random_state=seed, n_jobs=1)
        grid = ({"model__min_samples_leaf": [3], "model__max_features": [0.8]} if quick else
                {"model__min_samples_leaf": [2, 3, 5], "model__max_features": [0.6, 0.9, 1.0]})
    elif name == "histgb":
        est = HistGradientBoostingRegressor(random_state=seed, early_stopping=False)
        grid = ({"model__max_iter": [80], "model__max_leaf_nodes": [5], "model__l2_regularization": [1.0]} if quick else
                {"model__max_iter": [80, 140], "model__max_leaf_nodes": [5, 10], "model__l2_regularization": [1.0, 5.0]})
    elif name == "elastic":
        est = ElasticNet(max_iter=12000, random_state=seed)
        grid = ({"model__alpha": [0.03], "model__l1_ratio": [0.5]} if quick else
                {"model__alpha": [0.003, 0.01, 0.03, 0.1], "model__l1_ratio": [0.2, 0.5, 0.8]})
    elif name == "svr":
        est = SVR(kernel="rbf")
        grid = ({"model__C": [1.0], "model__gamma": ["scale"]} if quick else
                {"model__C": [0.3, 1.0, 3.0, 10.0], "model__gamma": ["scale", 0.01, 0.05]})
    else:
        raise ValueError(f"Unknown model: {name}")
    return est, grid


def grouped_splits(groups: np.ndarray, k: int, seed: int):
    unique = np.unique(groups)
    k = min(k, len(unique))
    if k < 2:
        raise ValueError("At least two unique participants are required for grouped CV.")
    # GroupKFold has deterministic membership; shuffle IDs first so repetitions vary.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(unique)
    mapping = {g: i % k for i, g in enumerate(perm)}
    fold = np.array([mapping[g] for g in groups])
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)], fold


def simplex_weights(P: np.ndarray, y: np.ndarray, penalty: float = 0.01) -> np.ndarray:
    """Non-negative convex blend; weights are fitted only on inner OOF predictions."""
    P = np.asarray(P, float)
    y = np.asarray(y, float)
    good = np.isfinite(y) & np.isfinite(P).all(axis=1)
    P, y = P[good], y[good]
    m = P.shape[1]
    if len(y) < max(8, m + 2):
        return np.ones(m) / m
    uni = np.ones(m) / m
    scale = np.std(y) + 1e-8
    def obj(w):
        return np.mean((P @ w - y) ** 2) / (scale ** 2) + penalty * np.sum((w - uni) ** 2)
    res = minimize(obj, uni, method="SLSQP", bounds=[(0, 1)] * m,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    if not res.success or not np.isfinite(res.x).all():
        return uni
    return res.x / res.x.sum()


def fit_arm_fold(d, feature_spec, groups, y, tr, te, model_names, args, seed, arm_name):
    reg_cols = feature_spec["regular"]
    med_cols = feature_spec["mednet"]
    Xcols = list(dict.fromkeys(reg_cols + med_cols))
    X = d[Xcols]
    inner_splits, _ = grouped_splits(groups[tr], min(args.inner_folds, len(np.unique(groups[tr]))), seed + 101)
    min_inner_train = min(len(i) for i, _ in inner_splits)
    n_comp = min(args.mednet_pca, max(1, min_inner_train - 1), len(med_cols)) if med_cols else 0
    pre = make_preprocessor(reg_cols, med_cols, n_comp)
    pred_test = {}
    pred_inner = {}
    choices = []
    for model_name in model_names:
        est, grid = model_spec(model_name, seed, args.quick)
        pipe = Pipeline([("prep", pre), ("model", est)])
        n_inner = min(len(inner_splits), len(np.unique(groups[tr])))
        # Convert our explicit group splits to local indices for GridSearchCV.
        gkf = GroupKFold(n_splits=n_inner)
        search = GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", cv=gkf,
                              n_jobs=args.jobs, refit=True, error_score="raise")
        search.fit(X.iloc[tr], y[tr], groups=groups[tr])
        best = search.best_estimator_
        pred_test[model_name] = best.predict(X.iloc[te])
        oof = np.full(len(tr), np.nan, dtype=float)
        for itr, iva in inner_splits:
            mdl = clone(best)
            mdl.fit(X.iloc[tr[itr]], y[tr[itr]])
            oof[iva] = mdl.predict(X.iloc[tr[iva]])
        pred_inner[model_name] = oof
        choices.append({"arm": arm_name, "model": model_name, "best_params": json.dumps(search.best_params_, default=str)})
    mat_names = list(pred_inner)
    P_in = np.column_stack([pred_inner[m] for m in mat_names])
    P_te = np.column_stack([pred_test[m] for m in mat_names])
    w = simplex_weights(P_in, y[tr], penalty=0.02)
    pred_test["blend"] = P_te @ w
    return pred_test, {m: float(v) for m, v in zip(mat_names, w)}, choices


def r2_safe(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(r2_score(y, p)) if len(y) > 1 and np.nanvar(y) > 1e-12 else np.nan


def bootstrap_metrics(y, p, boot, seed):
    rng = np.random.RandomState(seed)
    n = len(y)
    r2s, maes = [], []
    for _ in range(boot):
        ix = rng.randint(0, n, n)
        if np.var(y[ix]) < 1e-12:
            continue
        r2s.append(r2_safe(y[ix], p[ix]))
        maes.append(mean_absolute_error(y[ix], p[ix]))
    return (
        np.percentile(r2s, [2.5, 97.5]).tolist() if r2s else [np.nan, np.nan],
        np.percentile(maes, [2.5, 97.5]).tolist() if maes else [np.nan, np.nan],
    )


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty(len(p), dtype=float)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def evaluate(pred_store, y_true, ptids, reps, boot, seed):
    metrics = []
    average_preds = {}
    for model, arr in pred_store.items():
        pooled_pred = np.nanmean(arr, axis=0)
        average_preds[model] = pooled_pred
        per_rep = np.array([[r2_safe(y_true[:, j], arr[r, :, j]) for j in range(3)] for r in range(reps)])
        for j, target in enumerate(TARGETS):
            ci, mci = bootstrap_metrics(y_true[:, j], pooled_pred[:, j], boot, seed + j + (zlib.crc32(model.encode("utf-8")) % 10000))
            metrics.append({"model": model, "endpoint": LABELS[target], "n_subjects": len(y_true),
                            "R2_pooled_OOF": r2_safe(y_true[:, j], pooled_pred[:, j]),
                            "R2_mean_per_repetition": float(np.nanmean(per_rep[:, j])),
                            "R2_boot_lo": ci[0], "R2_boot_hi": ci[1],
                            "MAE": float(mean_absolute_error(y_true[:, j], pooled_pred[:, j])),
                            "MAE_boot_lo": mci[0], "MAE_boot_hi": mci[1]})
        macro_rep = np.nanmean(per_rep, axis=1)
        macro_ci = []
        rng = np.random.RandomState(seed + (zlib.crc32(model.encode("utf-8")) % 10000))
        for _ in range(boot):
            ix = rng.randint(0, len(y_true), len(y_true))
            macro_ci.append(np.nanmean([r2_safe(y_true[ix, j], pooled_pred[ix, j]) for j in range(3)]))
        metrics.append({"model": model, "endpoint": "MEAN_R2", "n_subjects": len(y_true),
                        "R2_pooled_OOF": float(np.nanmean([r2_safe(y_true[:, j], pooled_pred[:, j]) for j in range(3)])),
                        "R2_mean_per_repetition": float(np.nanmean(macro_rep)),
                        "R2_boot_lo": float(np.nanpercentile(macro_ci, 2.5)), "R2_boot_hi": float(np.nanpercentile(macro_ci, 97.5)),
                        "MAE": np.nan, "MAE_boot_lo": np.nan, "MAE_boot_hi": np.nan})
    return pd.DataFrame(metrics), average_preds


def paired_deltas(avg_preds, y, baseline, boot, seed):
    if baseline not in avg_preds:
        raise ValueError(f"Baseline {baseline} not found in predictions: {list(avg_preds)}")
    rng = np.random.RandomState(seed)
    n = len(y)
    samples = [rng.randint(0, n, n) for _ in range(boot)]
    rows = []
    for model, p in avg_preds.items():
        if model == baseline:
            continue
        for j, target in enumerate(TARGETS):
            delta = r2_safe(y[:, j], p[:, j]) - r2_safe(y[:, j], avg_preds[baseline][:, j])
            b = [r2_safe(y[ix, j], p[ix, j]) - r2_safe(y[ix, j], avg_preds[baseline][ix, j]) for ix in samples]
            b = np.asarray(b, float); b = b[np.isfinite(b)]
            lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
            pval = min(1.0, max(2 * min(np.mean(b <= 0), np.mean(b >= 0)), 1 / (boot + 1))) if len(b) else 1.0
            rows.append({"model": model, "baseline": baseline, "endpoint": LABELS[target], "delta_R2": delta,
                         "CI_lo": lo, "CI_hi": hi, "p_boot": pval})
        # Macro delta is calculated on identical bootstrap subject samples.
        dpoint = np.mean([r2_safe(y[:, j], p[:, j]) - r2_safe(y[:, j], avg_preds[baseline][:, j]) for j in range(3)])
        bmacro = []
        for ix in samples:
            bmacro.append(np.mean([r2_safe(y[ix, j], p[ix, j]) - r2_safe(y[ix, j], avg_preds[baseline][ix, j]) for j in range(3)]))
        bmacro = np.asarray(bmacro, float); bmacro = bmacro[np.isfinite(bmacro)]
        lo, hi = np.percentile(bmacro, [2.5, 97.5]) if len(bmacro) else (np.nan, np.nan)
        pval = min(1.0, max(2 * min(np.mean(bmacro <= 0), np.mean(bmacro >= 0)), 1 / (boot + 1))) if len(bmacro) else 1.0
        rows.append({"model": model, "baseline": baseline, "endpoint": "MEAN_R2", "delta_R2": dpoint,
                     "CI_lo": lo, "CI_hi": hi, "p_boot": pval})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_boot"].values)
    return out


def main():
    args = parse_args()
    if args.quick:
        args.folds = 3
        args.reps = 1
        args.boot = min(args.boot, 200)
        args.inner_folds = 2
    os.makedirs(args.out_dir, exist_ok=True)
    d = merge_inputs(args.scans, args.rich)
    surf = pd.DataFrame({"Image_ID": d["Image_ID"].astype(str)})
    if not args.no_surface and args.surface_root:
        surf = extract_surface_table(d, args.surface_root, args.surface_left, args.surface_right,
                                     Path(args.out_dir) / "A_surface_features.csv")
        if len(surf.columns) > 1:
            d = d.merge(surf, on="Image_ID", how="left", validate="one_to_one")
    elif not args.no_surface:
        print("Surface extraction skipped: pass --surface-root to enable cortical thickness/area features.")
    surf_cols = [c for c in d.columns if c.startswith("surf_")]
    arms, demo, rich, surf_cols, mednet = build_features(d, surf_cols, args.include_mednet)
    model_names = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    allowed = {"ridge", "extratrees", "histgb", "elastic", "svr"}
    bad = set(model_names) - allowed
    if bad:
        raise ValueError(f"Unknown models {bad}; choose from {sorted(allowed)}")
    if not model_names:
        raise ValueError("No models selected.")
    # One row per PTID by construction.
    groups = d["PTID"].astype(str).to_numpy()
    fixed_fold_ids = None
    if args.fold_map:
        if not os.path.isfile(args.fold_map):
            raise FileNotFoundError(f"Fold map not found: {args.fold_map}")
        fm = pd.read_csv(args.fold_map, low_memory=False)
        if not {"PTID", "outer_fold"}.issubset(fm.columns):
            raise ValueError("--fold-map must contain PTID and outer_fold columns.")
        fm["PTID"] = fm.PTID.astype(str).str.strip()
        # Accept scan-level sidecars as long as all scans of a participant
        # were assigned to the same outer fold. Collapse to one mapping/PTID.
        fold_counts = fm.groupby("PTID")["outer_fold"].nunique(dropna=True)
        inconsistent = fold_counts[fold_counts > 1]
        if len(inconsistent):
            raise ValueError(
                "The supplied fold map assigns some PTIDs to multiple outer folds; "
                f"examples: {inconsistent.index[:5].tolist()}"
            )
        fm = fm.dropna(subset=["outer_fold"]).drop_duplicates("PTID").set_index("PTID")
        common = set(groups) & set(fm.index.astype(str))
        if len(common) < max(20, int(0.8 * len(d))):
            raise ValueError(f"Only {len(common)}/{len(d)} participants match the fold map; verify S16 cohort alignment.")
        if len(common) < len(d):
            print(f"Restricting Part A to {len(common)} participants present in the supplied S16 fold map.")
            d = d[d.PTID.astype(str).isin(common)].reset_index(drop=True)
            groups = d.PTID.astype(str).to_numpy()
        fold_lookup = fm["outer_fold"].to_dict()
        fixed_fold_ids = np.array([fold_lookup[g] for g in groups])
        args.reps = 1
        if len(np.unique(fixed_fold_ids)) < 2:
            raise ValueError("The supplied fold map must contain at least two outer folds.")
        print(f"Using supplied S16 outer folds: {sorted(np.unique(fixed_fold_ids).tolist())}; one repetition.")
    y_true = d[TARGETS].to_numpy(float)
    if args.permute_targets:
        rng = np.random.RandomState(args.seed + 909)
        y_true = y_true[rng.permutation(len(y_true))]
        print("*** PERMUTED-TARGET NULL RUN: outcomes shuffled across participants; never report as a trained result. ***")
    print(f"Final participants={len(d)}; feature arms={list(arms)}; surface features={len(surf_cols)}; MedicalNet features={len(mednet)}")
    manifest = []
    for arm, spec in arms.items():
        cols = list(dict.fromkeys(spec["regular"] + spec["mednet"]))
        manifest.append({"arm": arm, "n_regular": len(spec["regular"]), "n_mednet_raw": len(spec["mednet"]),
                         "n_total_raw": len(cols), "regular_columns": ";".join(spec["regular"]),
                         "mednet_columns": ";".join(spec["mednet"])})
    pd.DataFrame(manifest).to_csv(Path(args.out_dir) / "A_feature_manifest.csv", index=False)

    pred_store = {}
    fold_rows, weight_rows, choice_rows = [], [], []
    reps = args.reps
    for rep in range(reps):
        if fixed_fold_ids is None:
            splits, fold_assignment = grouped_splits(groups, args.folds, args.seed + rep * 37)
        else:
            fold_assignment = fixed_fold_ids.copy()
            splits = [(np.where(fold_assignment != f)[0], np.where(fold_assignment == f)[0])
                      for f in sorted(np.unique(fold_assignment))]
        for i, f in enumerate(fold_assignment):
            fold_rows.append({"PTID": groups[i], "rep": rep, "outer_fold": int(f)})
        for arm, spec in arms.items():
            for model in model_names + ["blend"]:
                pred_store.setdefault(f"{arm}__{model}", np.full((reps, len(d), 3), np.nan, dtype=float))
            for fold, (tr, te) in enumerate(splits):
                for j, target in enumerate(TARGETS):
                    # All three endpoints use exactly the same train/test participants.
                    out, w, choices = fit_arm_fold(d, spec, groups, y_true[:, j], tr, te,
                                                    model_names, args, args.seed + rep*1000 + fold*31 + j,
                                                    arm)
                    for m, values in out.items():
                        pred_store[f"{arm}__{m}"][rep, te, j] = values
                    for m, val in w.items():
                        weight_rows.append({"rep": rep, "outer_fold": fold, "arm": arm,
                                            "endpoint": LABELS[target], "base_model": m, "blend_weight": val})
                    for c in choices:
                        c.update({"rep": rep, "outer_fold": fold, "endpoint": LABELS[target]})
                        choice_rows.append(c)
                print(f"[Part A] rep {rep+1}/{reps}, fold {fold+1}/{len(splits)}, arm={arm}: completed", flush=True)
    baseline = "rich__blend"
    if baseline not in pred_store:
        raise RuntimeError("The required rich__blend baseline could not be created.")
    metrics, avg_preds = evaluate(pred_store, y_true, groups, reps, args.boot, args.seed)
    metrics.to_csv(Path(args.out_dir) / "A_metrics.csv", index=False)
    delta = paired_deltas(avg_preds, y_true, baseline, args.boot, args.seed + 1)
    delta.to_csv(Path(args.out_dir) / "A_paired_deltas_vs_rich_blend.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(Path(args.out_dir) / "A_subject_folds.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(Path(args.out_dir) / "A_blend_weights.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(Path(args.out_dir) / "A_inner_model_choices.csv", index=False)

    oof = d[["PTID", "Image_ID"] + TARGETS].copy()
    for model, arr in pred_store.items():
        avg = np.nanmean(arr, axis=0)
        for j, t in enumerate(TARGETS):
            oof[f"{model}__pred_{LABELS[t]}"] = avg[:, j]
    oof.to_csv(Path(args.out_dir) / "A_oof_predictions.csv", index=False)

    print("\n=== Part A: pooled OOF R2; conservative mean-per-repetition R2; MAE ===")
    view = metrics[metrics.endpoint.isin(["MMSE", "CDR-SB", "ADAS-Cog13", "MEAN_R2"])].copy()
    print(view[["model", "endpoint", "R2_pooled_OOF", "R2_mean_per_repetition", "R2_boot_lo", "R2_boot_hi", "MAE"]].round(3).to_string(index=False))
    print(f"\nPaired deltas are relative to {baseline}; Holm-adjusted p-values saved.")
    if args.permute_targets:
        worst = metrics[metrics.endpoint == "MEAN_R2"].R2_pooled_OOF.max()
        print(f"Null run: highest pooled mean R2 across arms={worst:.3f}.")
        if worst > 0.05:
            print("WARNING: positive null R2 is larger than expected; inspect fold assignment, duplicate subjects, target columns, and preprocessing.")
        else:
            print("Null R2 values near/below zero are reassuring, but one permutation does not prove absence of leakage.")
    print(f"\nSaved Part A results to: {args.out_dir}")


if __name__ == "__main__":
    main()
