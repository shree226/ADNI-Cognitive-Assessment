#!/usr/bin/env python
"""S16: subject-level MRI + APOE modelling for ADAS-Cog13, CDR-SB and MMSE.

Models
------
1) Tabular reference models: Ridge, ExtraTrees, HistGradientBoosting.
2) FiLM-conditioned 3D ROI ResNet: T1 + FastSurfer medial-temporal masks, conditioned
   by age/sex/APOE (and education when available).
3) 3D ViT with genetic/demographic conditioning: same MRI ROI input.
4) Anatomical graph-attention model: FastSurfer regional features as brain nodes,
   conditioned by age/sex/APOE.

Design choices intended to reduce optimistic estimates
-------------------------------------------------------
- One earliest available scan per participant for the primary cross-sectional analysis.
- Outer GroupKFold by participant; no participant appears in train and test.
- Inner group split for neural-network early stopping; scaling/imputation is fit on
  training rows only.
- Same complete-case cohort and outer folds for all requested models.
- No diagnosis, IDs, dates, or cognitive scores are used as predictors.
- All three cognitive outcomes are reported; higher ADAS-Cog13/CDR-SB means worse,
  while higher MMSE means better.

IMPORTANT
---------
This script is an experimental research pipeline, not a promise of R2 >= 0.80.
The 3D models are trained from scratch on a small cohort (~200 people); treat them as
exploratory. The small ViT is NOT pretrained. Do not describe it as pretrained.
"""
import argparse
import copy
import json
import os
import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

TARGETS = ["MMSCORE", "CDRSB", "TOTAL13"]
LABELS = {"MMSCORE": "MMSE", "CDRSB": "CDR-SB", "TOTAL13": "ADAS-Cog13"}
NODE_NAMES = [
    "hippocampus", "amygdala", "entorhinal", "parahippocampal", "fusiform",
    "inferiortemporal", "middletemporal", "isthmuscingulate", "precuneus",
    "lateral_ventricle", "inf_lat_ventricle", "thalamus", "caudate", "putamen",
    "pallidum", "accumbens", "cerebral_wm", "third_ventricle", "fourth_ventricle",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def numeric(s):
    return pd.to_numeric(s, errors="coerce")


def make_male(df):
    if "male" in df.columns:
        return numeric(df["male"])
    if "sex" not in df.columns:
        raise ValueError("scan_table.csv needs either 'male' or 'sex'.")
    s = df["sex"]
    if pd.api.types.is_numeric_dtype(s):
        # ADNI sex is often 1=male, 2=female; also allow already binary 0/1.
        vals = set(pd.Series(s).dropna().unique().tolist())
        if vals.issubset({0, 1}):
            return numeric(s)
        return numeric(s).map({1: 1, 2: 0})
    return s.astype(str).str.strip().str.lower().map(
        {"m": 1, "male": 1, "man": 1, "1": 1, "f": 0, "female": 0, "woman": 0, "2": 0}
    )


def resolve_cohort(scans_path, rich_path, rec_path):
    if not os.path.exists(scans_path):
        raise FileNotFoundError(
            f"Could not find {scans_path}. Put scan_table.csv in this folder or pass --scans."
        )
    d = pd.read_csv(scans_path)
    required = ["PTID", "Image_ID"] + TARGETS + ["age", "e4"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"scan_table.csv is missing required columns: {missing}")
    d["PTID"] = d["PTID"].astype(str)
    d["Image_ID"] = d["Image_ID"].astype(str)
    d["age"] = numeric(d["age"])
    d["male"] = make_male(d)
    d["e4"] = numeric(d["e4"])
    if "edu" in d.columns:
        d["edu"] = numeric(d["edu"])
    for t in TARGETS:
        d[t] = numeric(d[t])

    rich_cols_available = []
    if rich_path and os.path.exists(rich_path):
        r = pd.read_csv(rich_path)
        if "Image_ID" not in r.columns:
            raise ValueError("richer_features.csv must contain Image_ID.")
        r["Image_ID"] = r["Image_ID"].astype(str)
        rich_cols_available = [c for c in r.columns if c != "Image_ID"]
        # Prefer the freshly extracted richer_features.csv columns over stale columns
        # with the same name in scan_table.csv.
        d = d.merge(r, on="Image_ID", how="left", suffixes=("", "__rich"))
        for c in rich_cols_available:
            rc = c + "__rich"
            if rc in d.columns:
                d[c] = d[rc]
                d.drop(columns=[rc], inplace=True)
    elif rich_path:
        print(f"WARNING: rich file not found: {rich_path}; using features already in scans table.")

    rec = []
    if rec_path and os.path.exists(rec_path):
        try:
            rec = [x.strip() for x in open(rec_path, "r", encoding="utf-8").read().strip().split(",") if x.strip()]
        except OSError:
            rec = []
    rec = [c for c in rec if c in d.columns]
    if not rec:
        rec = [c for c in d.columns if c.startswith(("nv_", "asym_", "hip_", "ent_"))]

    # Reproduce S14's all-region ICV-normalised volumes and compact anatomical ratios.
    # These transformations are per-scan arithmetic, not target-fitted preprocessing.
    icv_source = next((c for c in ("icv_proxy", "ucsf_icv", "icv") if c in d.columns), None)
    av_cols = []
    if icv_source is not None:
        icv_values = numeric(d[icv_source]).replace(0, np.nan)
        for c in [x for x in d.columns if re.fullmatch(r"vol_\d+", x)]:
            av = "av_" + c[4:]
            d[av] = numeric(d[c]) / icv_values * 1e3
            if d[av].notna().mean() >= 0.80 and d[av].nunique(dropna=True) > 1:
                av_cols.append(av)
    sym_cols = []
    def add_ratio(name, numerator, denominator=None):
        if numerator and all(c in d.columns for c in numerator):
            value = d[numerator].sum(axis=1)
            if denominator and all(c in d.columns for c in denominator):
                value = value / (d[denominator].sum(axis=1) + 1e-6)
            d[name] = value
            if d[name].notna().mean() >= 0.80 and d[name].nunique(dropna=True) > 1:
                sym_cols.append(name)
    add_ratio("sym_hip_vent", ["nv_hippocampus"], ["nv_lateral_ventricle"])
    add_ratio("sym_mtl", ["nv_hippocampus", "nv_amygdala", "nv_entorhinal", "nv_parahippocampal"])
    add_ratio("sym_vent", [c for c in ["nv_lateral_ventricle", "nv_inf_lat_ventricle"] if c in d.columns])
    if "sym_mtl" in d.columns:
        add_ratio("sym_mtl_vent", ["sym_mtl"], [c for c in ["sym_vent"] if c in d.columns])
    add_ratio("sym_temporal", [c for c in ["nv_fusiform", "nv_inferiortemporal", "nv_middletemporal"] if c in d.columns])
    if "asym_hippocampus" in d.columns:
        d["sym_hip_asym_abs"] = numeric(d["asym_hippocampus"]).abs()
        if d["sym_hip_asym_abs"].notna().mean() >= 0.80 and d["sym_hip_asym_abs"].nunique(dropna=True) > 1:
            sym_cols.append("sym_hip_asym_abs")

    # Keep only targets that are actually observed, then take the earliest scan per person.
    d = d.dropna(subset=["PTID", "Image_ID"] + TARGETS).copy()
    if "EXAMDATE" in d.columns:
        d["EXAMDATE"] = pd.to_datetime(d["EXAMDATE"], errors="coerce")
        d = d.sort_values(["PTID", "EXAMDATE", "Image_ID"], kind="mergesort")
    else:
        d = d.sort_values(["PTID", "Image_ID"], kind="mergesort")
    n_scans = len(d)
    d = d.groupby("PTID", as_index=False, sort=False).head(1).reset_index(drop=True)

    # Remove rows lacking core genetic/demographic inputs; MRI features are imputed inside folds.
    d = d.dropna(subset=["age", "male", "e4"]).reset_index(drop=True)
    d["e4"] = d["e4"].clip(lower=0, upper=2)
    d["e4"] = d["e4"].astype(float)
    if "edu" in d.columns and d["edu"].notna().mean() < 0.80:
        # Avoid losing many participants for partially recorded education.
        rec_demographic = ["age", "male", "e4"]
    else:
        rec_demographic = ["age", "male", "e4"] + (["edu"] if "edu" in d.columns else [])

    # Candidate tabular MRI features: the pre-specified rich_cols list if present;
    # otherwise the S11 named features. Include the two original volume fields if available.
    own = [c for c in ["f_1536", "f_1537"] if c in d.columns]
    rich_numeric = [c for c in rec if c in d.columns]
    if not rich_numeric:
        rich_numeric = [c for c in d.columns if c.startswith(("nv_", "asym_", "hip_", "ent_"))]
    icv_cols = [c for c in ["icv_proxy", "ucsf_icv", "icv"] if c in d.columns]
    tabular_cols = list(dict.fromkeys(rec_demographic + own + rich_numeric + av_cols + sym_cols + icv_cols))

    # Strictly exclude identifiers, dates, diagnosis, and outcomes even if rec list contains them.
    banned = set(TARGETS + ["PTID", "Image_ID", "EXAMDATE", "dx", "DX", "scan_idx", "uid", "Subject_ID"])
    tabular_cols = [c for c in tabular_cols if c not in banned and pd.api.types.is_numeric_dtype(numeric(d[c]))]
    if not tabular_cols:
        raise ValueError("No numeric tabular features were found. Check rich_cols.txt and the input CSVs.")

    # Drop numeric features that are entirely missing or constant in this cohort.
    keep = []
    for c in tabular_cols:
        x = numeric(d[c])
        if x.notna().sum() >= max(10, int(0.25 * len(d))) and x.nunique(dropna=True) > 1:
            d[c] = x
            keep.append(c)
    tabular_cols = keep

    # Conditions for neural models: education is used only if sufficiently complete.
    cond_cols = ["age", "male", "e4"] + (["edu"] if "edu" in d.columns and d["edu"].notna().mean() >= 0.80 else [])
    d = d.reset_index(drop=True)
    print(f"Loaded {n_scans} usable scan rows -> {len(d)} baseline participants.")
    print(f"Tabular predictors ({len(tabular_cols)}): {', '.join(tabular_cols)}")
    print(f"Genetic/demographic conditioning inputs: {cond_cols}")
    return d, tabular_cols, cond_cols, rich_numeric


def make_node_array(df):
    """Return [N, nodes, channels] raw anatomy features; standardise within each fold."""
    n = len(df)
    arr = np.zeros((n, len(NODE_NAMES), 4), dtype=np.float32)
    for j, name in enumerate(NODE_NAMES):
        vol = f"nv_{name}"
        asym = f"asym_{name}"
        if vol in df.columns:
            arr[:, j, 0] = numeric(df[vol]).to_numpy(dtype=float, na_value=np.nan)
        else:
            arr[:, j, 0] = np.nan
        if asym in df.columns:
            arr[:, j, 1] = numeric(df[asym]).to_numpy(dtype=float, na_value=np.nan)
        else:
            arr[:, j, 1] = np.nan
        if name == "hippocampus":
            for k, c in enumerate(["hip_surf_to_vol", "hip_int_entropy"], start=2):
                arr[:, j, k] = numeric(df[c]).to_numpy(dtype=float, na_value=np.nan) if c in df else np.nan
        elif name == "entorhinal":
            for k, c in enumerate(["ent_surf_to_vol", "ent_int_entropy"], start=2):
                arr[:, j, k] = numeric(df[c]).to_numpy(dtype=float, na_value=np.nan) if c in df else np.nan
        else:
            arr[:, j, 2:] = np.nan
    return arr


def make_adjacency():
    groups = {
        "mtl": {"hippocampus", "amygdala", "entorhinal", "parahippocampal"},
        "temporal": {"fusiform", "inferiortemporal", "middletemporal"},
        "limbic": {"isthmuscingulate", "precuneus"},
        "ventricle": {"lateral_ventricle", "inf_lat_ventricle", "third_ventricle", "fourth_ventricle"},
        "subcortical": {"thalamus", "caudate", "putamen", "pallidum", "accumbens"},
        "white_matter": {"cerebral_wm"},
    }
    a = np.eye(len(NODE_NAMES), dtype=np.float32)
    for i, x in enumerate(NODE_NAMES):
        for j, y in enumerate(NODE_NAMES):
            if i == j:
                continue
            gx = next((g for g, v in groups.items() if x in v), "other")
            gy = next((g for g, v in groups.items() if y in v), "other")
            if gx == gy or (gx == "mtl" and gy in {"temporal", "limbic"}) or (gy == "mtl" and gx in {"temporal", "limbic"}):
                a[i, j] = 1.0
    return a


def standardise_nodes(train, other):
    # Per-node, per-channel imputation and scaling using TRAINING PARTICIPANTS only.
    med = np.zeros(train.shape[1:], dtype=np.float32)
    for i in range(train.shape[1]):
        for j in range(train.shape[2]):
            values = train[:, i, j]
            values = values[np.isfinite(values)]
            med[i, j] = np.median(values) if len(values) else 0.0
    tr = np.where(np.isfinite(train), train, med)
    ot = np.where(np.isfinite(other), other, med)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return ((tr - mu) / sd).astype(np.float32), ((ot - mu) / sd).astype(np.float32)


def make_inner_split(groups, seed, val_fraction=0.18):
    unique = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    nv = max(1, int(round(len(unique) * val_fraction)))
    val_groups = set(unique[:nv])
    is_val = np.array([g in val_groups for g in groups])
    return np.where(~is_val)[0], np.where(is_val)[0]


def fit_tabular_models(df, cols, groups, targets, folds, out_dir, seed, quick=False, model_prefix="tabular"):

    from sklearn.base import clone
    model_specs = {
        "ridge": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0, 300.0]},
        ),
        "extratrees": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(
                n_estimators=60 if quick else 250, random_state=seed, n_jobs=-1
            ))]),
            ({"model__min_samples_leaf": [2], "model__max_features": [0.8]} if quick else
             {"model__min_samples_leaf": [2, 5], "model__max_features": [0.8, 1.0]}),
        ),
        "histgb": (
            Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(
                random_state=seed, early_stopping=False
            ))]),
            ({"model__max_iter": [80], "model__max_leaf_nodes": [5], "model__l2_regularization": [1.0]} if quick else
             {"model__max_iter": [100, 160], "model__max_leaf_nodes": [5, 10], "model__l2_regularization": [1.0]}),
        ),
    }
    models = {f"{model_prefix}_{key}": spec for key, spec in model_specs.items()}
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = df[targets].to_numpy(dtype=float)
    predictions = {name: np.full_like(y, np.nan, dtype=float) for name in models}
    chosen = []
    for fold_id, (tr, te) in enumerate(folds, start=1):
        inner_groups = groups[tr]
        n_inner = min(3, len(np.unique(inner_groups)))
        if n_inner < 2:
            raise ValueError("Not enough training participants for inner grouped CV.")
        inner_cv = GroupKFold(n_splits=n_inner)
        for name, (pipe, grid) in models.items():
            for j, target in enumerate(targets):
                search = GridSearchCV(
                    clone(pipe), grid, scoring="neg_mean_squared_error", cv=inner_cv,
                    n_jobs=1, refit=True,
                )
                search.fit(X[tr], y[tr, j], groups=inner_groups)
                predictions[name][te, j] = search.predict(X[te])
                chosen.append({"fold": fold_id, "model": name, "target": target, "best_params": str(search.best_params_)})
        print(f"[tabular] finished outer fold {fold_id}/{len(folds)}", flush=True)
    pd.DataFrame(chosen).to_csv(os.path.join(out_dir, f"{model_prefix}_inner_model_choices.csv"), index=False)
    return predictions


def build_image_index(root, image_ids, seg_name, t1_name):
    """Index FastSurfer files using the same ID-in-path convention as S11."""
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"--image-root does not exist: {root}")
    ids_by_key = {}
    for sid in image_ids:
        key = re.sub(r"\D", "", str(sid))
        if key:
            ids_by_key.setdefault(key, []).append(str(sid))
    found = {str(sid): {} for sid in image_ids}
    wanted = {seg_name, t1_name}
    for dp, _, files in os.walk(root):
        path_text = dp.replace("\\", "/")
        for fname in files:
            if fname not in wanted:
                # Also allow a file whose name contains the image ID and matching extension.
                ext_ok = fname.endswith(".mgz") or fname.endswith(".nii") or fname.endswith(".nii.gz")
                if not ext_ok:
                    continue
            full = os.path.join(dp, fname)
            for key, sids in ids_by_key.items():
                pat = re.compile(r"(?<!\d)" + re.escape(key) + r"(?!\d)")
                if fname == seg_name or fname == t1_name:
                    match = bool(pat.search(path_text))
                else:
                    match = bool(pat.search(fname))
                if match:
                    for sid in sids:
                        kind = "seg" if (fname == seg_name or (fname.endswith(".mgz") and "aseg" in fname.lower())) else "t1"
                        # Prefer exact requested filenames.
                        if fname == seg_name:
                            found[sid]["seg"] = full
                        elif fname == t1_name:
                            found[sid]["t1"] = full
                        elif kind not in found[sid]:
                            found[sid][kind] = full
    return found


def extract_roi_tensor(seg_path, t1_path, roi_size=48):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from scipy.ndimage import zoom

    seg_img = nib.load(seg_path)
    t1_img = nib.load(t1_path)
    if t1_img.shape[:3] != seg_img.shape[:3] or not np.allclose(t1_img.affine, seg_img.affine, atol=1e-3):
        t1_img = resample_from_to(t1_img, seg_img, order=1)
    seg = np.asanyarray(seg_img.dataobj).astype(np.int32)
    t1 = np.asanyarray(t1_img.dataobj).astype(np.float32)
    labels = {
        "hip": [17, 53], "amy": [18, 54], "ent": [1006, 2006], "para": [1016, 2016]
    }
    masks = {k: np.isin(seg, v) for k, v in labels.items()}
    union = masks["hip"] | masks["amy"] | masks["ent"] | masks["para"]
    if int(union.sum()) < 80:
        raise ValueError("Too few medial-temporal ROI voxels; segmentation labels may not match.")
    coords = np.where(union)
    pad = 8
    lo = [max(0, int(c.min()) - pad) for c in coords]
    hi = [min(seg.shape[i], int(coords[i].max()) + 1 + pad) for i in range(3)]
    sl = tuple(slice(lo[i], hi[i]) for i in range(3))
    t = t1[sl].copy()
    roi = {k: v[sl].astype(np.float32) for k, v in masks.items()}
    u = union[sl]
    vals = t[u]
    q1, q99 = np.percentile(vals, [1, 99])
    t = np.clip(t, q1, q99)
    mu, sd = float(t[u].mean()), float(t[u].std())
    t = np.clip((t - mu) / (sd + 1e-6), -5, 5).astype(np.float32)

    # Centre-pad the bounding box to a cube before isotropic resizing; this limits distortion.
    shape = np.array(t.shape)
    side = int(shape.max())
    def cube(a, is_mask=False):
        pads = []
        for dim in a.shape:
            total = side - dim
            pads.append((total // 2, total - total // 2))
        a = np.pad(a, pads, mode="constant", constant_values=0)
        order = 0 if is_mask else 1
        zf = [roi_size / side] * 3
        a = zoom(a, zf, order=order)
        # scipy zoom can be off by one depending on rounding.
        out = np.zeros((roi_size, roi_size, roi_size), dtype=np.float32)
        sl2 = tuple(slice(0, min(roi_size, a.shape[i])) for i in range(3))
        out[sl2] = a[sl2].astype(np.float32)
        return out
    chans = [cube(t)] + [cube(roi[k], is_mask=True) for k in ("hip", "amy", "ent", "para")]
    return np.stack(chans, axis=0).astype(np.float32)


def prepare_roi_cache(df, image_root, seg_name, t1_name, cache_dir, roi_size):
    os.makedirs(cache_dir, exist_ok=True)
    index = build_image_index(image_root, df["Image_ID"].astype(str).tolist(), seg_name, t1_name)
    paths, keep, failures = [], [], []
    for i, row in df.iterrows():
        sid = str(row["Image_ID"])
        item = index.get(sid, {})
        sp, tp = item.get("seg"), item.get("t1")
        if not sp or not tp:
            failures.append((sid, "missing T1 or segmentation"))
            continue
        out = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_-]", "_", sid) + f"_{roi_size}.npy")
        try:
            if not os.path.exists(out):
                arr = extract_roi_tensor(sp, tp, roi_size=roi_size)
                np.save(out, arr)
            paths.append(out)
            keep.append(i)
        except Exception as e:
            failures.append((sid, str(e)))
        if (i + 1) % 25 == 0:
            print(f"[MRI preprocessing] visited {i + 1}/{len(df)}; usable={len(keep)}", flush=True)
    pd.DataFrame(failures, columns=["Image_ID", "reason"]).to_csv(os.path.join(cache_dir, "missing_or_failed_scans.csv"), index=False)
    return np.asarray(keep, dtype=int), paths


# --------------------------- PyTorch architectures ---------------------------
def get_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("PyTorch is required for film3d, vit3d and gat models. Install torch first.") from e
    return torch, nn, Dataset, DataLoader, TensorDataset


def build_models():
    torch, nn, Dataset, DataLoader, TensorDataset = get_torch()

    class Block3D(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.conv1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.norm1 = nn.GroupNorm(8, cout)
            self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
            self.norm2 = nn.GroupNorm(8, cout)
            self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False), nn.GroupNorm(8, cout)
            )
            self.act = nn.GELU()
        def forward(self, x):
            z = self.act(self.norm1(self.conv1(x)))
            z = self.norm2(self.conv2(z))
            return self.act(z + self.skip(x))

    class FiLMResNet3D(nn.Module):
        def __init__(self, cond_dim, nout=3):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv3d(5, 16, 5, stride=2, padding=2, bias=False), nn.GroupNorm(8, 16), nn.GELU(), nn.MaxPool3d(2))
            self.l1 = nn.Sequential(Block3D(16, 16), Block3D(16, 16))
            self.l2 = nn.Sequential(Block3D(16, 32, 2), Block3D(32, 32))
            self.l3 = nn.Sequential(Block3D(32, 64, 2), Block3D(64, 64))
            self.l4 = nn.Sequential(Block3D(64, 96, 2), Block3D(96, 96))
            self.cond = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Dropout(0.15))
            self.film3 = nn.Linear(32, 128)
            self.film4 = nn.Linear(32, 192)
            self.head = nn.Sequential(nn.Linear(96 + 32, 64), nn.GELU(), nn.Dropout(0.35), nn.Linear(64, nout))
        def modulate(self, x, film):
            gamma, beta = film.chunk(2, dim=1)
            return x * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        def forward(self, x, c):
            c = self.cond(c)
            x = self.stem(x); x = self.l1(x); x = self.l2(x)
            x = self.modulate(self.l3(x), self.film3(c))
            x = self.modulate(self.l4(x), self.film4(c))
            x = x.mean(dim=(2, 3, 4))
            return self.head(torch.cat([x, c], dim=1))

    class SmallViT3D(nn.Module):
        def __init__(self, cond_dim, roi_size=48, patch=8, dim=96, nout=3):
            super().__init__()
            n_tokens = (roi_size // patch) ** 3
            self.patch = nn.Conv3d(5, dim, kernel_size=patch, stride=patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
            layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 2,
                                               dropout=0.2, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=3, norm=nn.LayerNorm(dim))
            self.cond = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Dropout(0.15))
            self.film = nn.Linear(32, dim * 2)
            self.head = nn.Sequential(nn.Linear(dim + 32, 64), nn.GELU(), nn.Dropout(0.35), nn.Linear(64, nout))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)
        def forward(self, x, c):
            z = self.patch(x).flatten(2).transpose(1, 2)
            cls = self.cls.expand(z.size(0), -1, -1)
            z = self.encoder(torch.cat([cls, z], dim=1) + self.pos)
            c = self.cond(c)
            gamma, beta = self.film(c).chunk(2, dim=1)
            z = z * (1 + gamma[:, None, :]) + beta[:, None, :]
            return self.head(torch.cat([z[:, 0], c], dim=1))

    class GraphAttention(nn.Module):
        def __init__(self, dim, adj, dropout=0.2):
            super().__init__()
            self.register_buffer("adj", torch.as_tensor(adj, dtype=torch.bool))
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.out = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)
            self.scale = dim ** -0.5
        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
            scores = scores.masked_fill(~self.adj[None, :, :], -1e4)
            attn = torch.softmax(scores, dim=-1)
            z = torch.matmul(attn, v)
            return self.norm(x + self.drop(self.out(z)))

    class GenotypeGraphNet(nn.Module):
        def __init__(self, cond_dim, adj, nout=3):
            super().__init__()
            self.node_in = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Dropout(0.15), nn.Linear(32, 64))
            self.cond = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Dropout(0.15))
            self.film = nn.Linear(32, 128)
            self.g1 = GraphAttention(64, adj)
            self.g2 = GraphAttention(64, adj)
            self.head = nn.Sequential(nn.Linear(64 + 32, 48), nn.GELU(), nn.Dropout(0.35), nn.Linear(48, nout))
        def forward(self, nodes, c):
            c = self.cond(c)
            h = self.node_in(nodes)
            gamma, beta = self.film(c).chunk(2, dim=1)
            h = h * (1 + gamma[:, None, :]) + beta[:, None, :]
            h = self.g1(h); h = self.g2(h)
            pooled = h.mean(dim=1)
            return self.head(torch.cat([pooled, c], dim=1))

    return {
        "film3d": FiLMResNet3D,
        "vit3d": SmallViT3D,
        "gat": GenotypeGraphNet,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "TensorDataset": TensorDataset,
    }


class NpyMRIDataset:
    """Factory is defined after importing torch in train_torch_model."""
    pass


def train_neural_fold(model_name, all_x, all_cond, all_y, groups, tr_idx, te_idx,
                      fold_seed, args, device, paths=None, adjacency=None, quick=False, drop_apoe=False):
    torch, nn, Dataset, DataLoader, TensorDataset = get_torch()
    constructors = build_models()
    rng = np.random.RandomState(fold_seed)
    local_groups = groups[tr_idx]
    inner_tr_local, inner_val_local = make_inner_split(local_groups, fold_seed)
    tr = tr_idx[inner_tr_local]
    va = tr_idx[inner_val_local]

    # Fold-only preprocessing for conditioning variables and targets.
    c_train = all_cond[tr].astype(np.float32)
    c_mu = np.nanmean(c_train, axis=0)
    c_sd = np.nanstd(c_train, axis=0)
    c_sd[c_sd < 1e-6] = 1.0
    cond = np.where(np.isfinite(all_cond), all_cond, c_mu)
    if drop_apoe:
        # Remove APOE using the TRAINING-fold mean, not a test-set statistic.
        # The column index is passed through args.condition_cols.
        if "e4" in args.condition_cols:
            cond[:, args.condition_cols.index("e4")] = c_mu[args.condition_cols.index("e4")]
    cond = ((cond - c_mu) / c_sd).astype(np.float32)

    y_train = all_y[tr].astype(np.float32)
    y_mu = y_train.mean(axis=0)
    y_sd = y_train.std(axis=0)
    y_sd[y_sd < 1e-6] = 1.0
    y_scaled = ((all_y - y_mu) / y_sd).astype(np.float32)

    if model_name == "gat":
        xtr, xrest = standardise_nodes(all_x[tr], all_x[np.arange(len(all_x))])
        # standardise_nodes returns train then all rows; the second array is aligned to full cohort.
        x_all = xrest
        Xtr = torch.as_tensor(x_all[tr], dtype=torch.float32)
        Xva = torch.as_tensor(x_all[va], dtype=torch.float32)
        Xte = torch.as_tensor(x_all[te_idx], dtype=torch.float32)
        model = constructors[model_name](cond.shape[1], adjacency).to(device)
        train_ds = TensorDataset(Xtr, torch.as_tensor(cond[tr]), torch.as_tensor(y_scaled[tr]))
        val_ds = TensorDataset(Xva, torch.as_tensor(cond[va]), torch.as_tensor(y_scaled[va]))
        test_ds = TensorDataset(Xte, torch.as_tensor(cond[te_idx]))
    else:
        class MRISet(Dataset):
            def __init__(self, ids, with_y=True):
                self.ids = np.asarray(ids, dtype=int)
                self.with_y = with_y
            def __len__(self): return len(self.ids)
            def __getitem__(self, ix):
                rid = int(self.ids[ix])
                arr = np.load(paths[rid], mmap_mode="r").astype(np.float32, copy=False)
                x = torch.from_numpy(np.array(arr, copy=True))
                c = torch.from_numpy(cond[rid])
                if self.with_y:
                    return x, c, torch.from_numpy(y_scaled[rid])
                return x, c
        model = constructors[model_name](cond.shape[1], roi_size=args.roi_size).to(device) if model_name == "vit3d" else constructors[model_name](cond.shape[1]).to(device)
        train_ds = MRISet(tr, True)
        val_ds = MRISet(va, True)
        test_ds = MRISet(te_idx, False)

    batch_size = max(1, args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    best_loss = float("inf"); best_state = None; patience = 0
    max_epochs = 4 if quick else args.epochs
    max_patience = 2 if quick else args.patience

    for epoch in range(max_epochs):
        model.train(); tr_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if model_name == "gat":
                xb, cb, yb = [z.to(device) for z in batch]
                pred = model(xb, cb)
            else:
                xb, cb, yb = [z.to(device) for z in batch]
                pred = model(xb, cb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "gat":
                    xb, cb, yb = [z.to(device) for z in batch]
                else:
                    xb, cb, yb = [z.to(device) for z in batch]
                pr = model(xb, cb)
                val_losses.append(float(loss_fn(pr, yb).detach().cpu()))
        vl = float(np.mean(val_losses)) if val_losses else float(np.mean(tr_losses))
        scheduler.step(vl)
        if vl < best_loss - 1e-5:
            best_loss = vl; patience = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= max_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval(); pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            if model_name == "gat":
                xb, cb = [z.to(device) for z in batch]
            else:
                xb, cb = [z.to(device) for z in batch]
            pr = model(xb, cb).detach().cpu().numpy()
            pred_list.append(pr)
    pred_scaled = np.concatenate(pred_list, axis=0)
    pred = pred_scaled * y_sd + y_mu
    del model, optimizer, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred


def bootstrap_r2(y, p, n_boot=1000, seed=123):
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        ix = rng.randint(0, n, n)
        if np.nanstd(y[ix]) < 1e-9:
            continue
        vals.append(r2_score(y[ix], p[ix]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise_predictions(df, pred_map, out_dir, boot=1000, seed=0):
    rows = []; pred_rows = df[["PTID", "Image_ID"] + TARGETS].copy()
    for model, pred in pred_map.items():
        if pred is None or np.isnan(pred).all():
            continue
        for j, t in enumerate(TARGETS):
            y = df[t].to_numpy(dtype=float)
            p = pred[:, j]
            lo, hi = bootstrap_r2(y, p, n_boot=boot, seed=seed + j)
            rows.append({
                "model": model, "endpoint": LABELS[t], "n_subjects": len(df),
                "R2": float(r2_score(y, p)), "R2_boot_lo": lo, "R2_boot_hi": hi,
                "MAE": float(mean_absolute_error(y, p)),
                "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            })
            pred_rows[f"{model}__pred_{LABELS[t]}"] = p
    metrics = pd.DataFrame(rows).sort_values(["endpoint", "R2"], ascending=[True, False])
    metrics.to_csv(os.path.join(out_dir, "S16_metrics.csv"), index=False)
    pred_rows.to_csv(os.path.join(out_dir, "S16_oof_predictions.csv"), index=False)

    # Paired bootstrap: does adding APOE improve held-out R2 on the exact same people?
    pairs = []
    pair_map = {}
    for name in pred_map:
        if name.startswith("tabular_no_apoe_"):
            pair_map[name] = "tabular_" + name.replace("tabular_no_apoe_", "", 1)
        elif name == "gat_no_apoe":
            pair_map[name] = "genotype_gat"
        elif name.endswith("_no_apoe"):
            pair_map[name] = name[:-len("_no_apoe")]
    rng = np.random.RandomState(seed + 991)
    n = len(df)
    for no_name, yes_name in pair_map.items():
        if yes_name not in pred_map:
            continue
        p0, p1 = pred_map[no_name], pred_map[yes_name]
        for j, t in enumerate(TARGETS):
            y = df[t].to_numpy(dtype=float)
            delta = r2_score(y, p1[:, j]) - r2_score(y, p0[:, j])
            bs = []
            for _ in range(boot):
                ix = rng.randint(0, n, n)
                if np.std(y[ix]) > 1e-9:
                    bs.append(r2_score(y[ix], p1[ix, j]) - r2_score(y[ix], p0[ix, j]))
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
            p_boot = min(1.0, 2 * min(np.mean(np.asarray(bs) <= 0), np.mean(np.asarray(bs) >= 0))) if bs else np.nan
            pairs.append({"model_with_APOE": yes_name, "model_without_APOE": no_name,
                          "endpoint": LABELS[t], "delta_R2_with_minus_without": delta,
                          "delta_lo": lo, "delta_hi": hi, "p_boot_approx": p_boot})
    pd.DataFrame(pairs).to_csv(os.path.join(out_dir, "S16_apoe_ablation.csv"), index=False)
    print("\n=== OUT-OF-FOLD RESULTS (one baseline scan per participant) ===")
    if len(metrics):
        print(metrics[["model", "endpoint", "R2", "R2_boot_lo", "R2_boot_hi", "MAE", "RMSE"]].round(3).to_string(index=False))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="S16 APOE-conditioned MRI models with subject-level CV")
    ap.add_argument("--scans", default="scan_table.csv")
    ap.add_argument("--rich", default="richer_features.csv")
    ap.add_argument("--rec", default="rich_cols.txt")
    ap.add_argument("--out-dir", default="results_final/S16")
    ap.add_argument("--models", default="tabular,gat,film3d,vit3d",
                    help="comma list: tabular,gat,film3d,vit3d")
    ap.add_argument("--image-root", default="", help="FastSurfer output root; required for film3d/vit3d")
    ap.add_argument("--seg-name", default="aparc.DKTatlas+aseg.deep.mgz")
    ap.add_argument("--t1-name", default="orig.mgz")
    ap.add_argument("--cache-dir", default="roi_cache_s16")
    ap.add_argument("--roi-size", type=int, default=48)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-group-col", default="PTID",
                    help="PTID for participant-grouped CV; optionally use a site column for site-held-out sensitivity")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--apoe-ablation", action="store_true",
                    help="also run matched models with APOE dosage removed (recommended for the genetics claim)")
    ap.add_argument("--quick", action="store_true", help="2 folds and 4 epochs for smoke testing only")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    requested = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    if args.roi_size <= 0 or args.roi_size % 8 != 0:
        raise ValueError("--roi-size must be a positive multiple of 8 (e.g. 48).")
    allowed = {"tabular", "gat", "film3d", "vit3d"}
    bad = set(requested) - allowed
    if bad:
        raise ValueError(f"Unknown models: {sorted(bad)}. Allowed: {sorted(allowed)}")
    df, tab_cols, cond_cols, rich_cols = resolve_cohort(args.scans, args.rich, args.rec)

    # Save the exact feature set for reproducibility. The final cohort is saved after
    # optional image-availability filtering below.
    pd.DataFrame({"tabular_feature": tab_cols}).to_csv(os.path.join(args.out_dir, "S16_feature_manifest.csv"), index=False)

    # Image models require both original T1 and segmentation. Restrict EVERY model to
    # the same complete MRI cohort when image models are requested.
    roi_paths = None
    image_models = [m for m in requested if m in {"film3d", "vit3d"}]
    if image_models:
        if not args.image_root:
            print("WARNING: --image-root not provided. Skipping film3d/vit3d; tabular and GAT can still run.")
            requested = [m for m in requested if m not in {"film3d", "vit3d"}]
        else:
            keep, roi_paths = prepare_roi_cache(df, args.image_root, args.seg_name, args.t1_name,
                                                args.cache_dir, args.roi_size)
            if len(keep) < 30:
                raise RuntimeError(f"Only {len(keep)} usable MRI crops found. Check --image-root, --seg-name, --t1-name and Image_ID matching.")
            if len(keep) != len(df):
                print(f"WARNING: dropping {len(df)-len(keep)} participants without usable MRI; all models will use identical n={len(keep)} cohort.")
                df = df.iloc[keep].reset_index(drop=True)
                # prepare_roi_cache returns paths in the same order as keep, matching df.iloc[keep].
            else:
                df = df.reset_index(drop=True)

    if args.cv_group_col not in df.columns:
        raise ValueError(f"--cv-group-col {args.cv_group_col!r} is not a column in scan_table.csv")
    groups = df[args.cv_group_col].astype(str).to_numpy()
    df[["PTID", "Image_ID"] + TARGETS].to_csv(os.path.join(args.out_dir, "S16_cohort.csv"), index=False)
    if len(np.unique(groups)) < args.folds:
        raise ValueError("Fewer participants than CV folds.")
    n_folds = 2 if args.quick else args.folds
    if len(np.unique(groups)) < n_folds:
        raise ValueError("Not enough unique participants for requested folds.")
    splitter = GroupKFold(n_splits=n_folds)
    folds = list(splitter.split(np.zeros(len(df)), df[TARGETS[0]].to_numpy(), groups))
    fold_rows = []
    for f, (_, te) in enumerate(folds, start=1):
        for ix in te:
            fold_rows.append({"PTID": groups[ix], "Image_ID": df.loc[ix, "Image_ID"], "outer_fold": f})
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out_dir, "S16_outer_folds.csv"), index=False)

    y = df[TARGETS].to_numpy(dtype=float)
    pred_map = {}
    if "tabular" in requested:
        pred_map.update(fit_tabular_models(df, tab_cols, groups, TARGETS, folds, args.out_dir, args.seed, args.quick))
        if args.apoe_ablation:
            tab_no_apoe = [c for c in tab_cols if c != "e4"]
            if len(tab_no_apoe) > 0:
                pred_map.update(fit_tabular_models(df, tab_no_apoe, groups, TARGETS, folds,
                                                   args.out_dir, args.seed, args.quick,
                                                   model_prefix="tabular_no_apoe"))

    cond = df[cond_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    args.condition_cols = cond_cols
    # The mandatory conditioning inputs should already be complete; fill optional education fold-safely.
    if np.isnan(cond).any():
        print("WARNING: missing optional conditioning values will be imputed using each fold's training mean.")

    if "gat" in requested:
        try:
            node_all = make_node_array(df)
            adj = make_adjacency()
            torch, _, _, _, _ = get_torch()
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is not available in this Python environment.")
            device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
            p = np.full_like(y, np.nan, dtype=float)
            for f, (tr, te) in enumerate(folds, start=1):
                p[te] = train_neural_fold("gat", node_all, cond, y, groups, tr, te,
                                          args.seed + f * 11, args, device, adjacency=adj, quick=args.quick)
                print(f"[gat] finished outer fold {f}/{len(folds)}", flush=True)
            pred_map["genotype_gat"] = p
            if args.apoe_ablation:
                p0 = np.full_like(y, np.nan, dtype=float)
                for f, (tr, te) in enumerate(folds, start=1):
                    p0[te] = train_neural_fold("gat", node_all, cond, y, groups, tr, te,
                                               args.seed + f * 11, args, device, adjacency=adj,
                                               quick=args.quick, drop_apoe=True)
                pred_map["gat_no_apoe"] = p0
        except Exception as e:
            print(f"ERROR: GAT model failed and will be skipped: {e}")

    image_requested = [m for m in requested if m in {"film3d", "vit3d"}]
    if image_requested:
        try:
            torch, _, _, _, _ = get_torch()
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is not available in this Python environment.")
            device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
            print(f"Neural models running on: {device}")
            for model in image_requested:
                p = np.full_like(y, np.nan, dtype=float)
                for f, (tr, te) in enumerate(folds, start=1):
                    p[te] = train_neural_fold(model, None, cond, y, groups, tr, te,
                                              args.seed + f * 101, args, device,
                                              paths=roi_paths, quick=args.quick)
                    print(f"[{model}] finished outer fold {f}/{len(folds)}", flush=True)
                model_label = "film_resnet3d_genotype" if model == "film3d" else "vit3d_genotype_fusion"
                pred_map[model_label] = p
                if args.apoe_ablation:
                    p0 = np.full_like(y, np.nan, dtype=float)
                    for f, (tr, te) in enumerate(folds, start=1):
                        p0[te] = train_neural_fold(model, None, cond, y, groups, tr, te,
                                                   args.seed + f * 101, args, device,
                                                   paths=roi_paths, quick=args.quick, drop_apoe=True)
                    pred_map[model_label + "_no_apoe"] = p0
        except Exception as e:
            print(f"ERROR: 3D MRI model failed: {e}")
            raise

    # Re-run summary on the final common cohort.
    metrics = summarise_predictions(df, pred_map, args.out_dir, boot=100 if args.quick else args.boot, seed=args.seed)
    metadata = {
        "script": "S16 genotype-conditioned models",
        "n_participants": int(len(df)),
        "n_unique_ptid": int(df.PTID.nunique()),
        "models_requested": requested,
        "models_completed": list(pred_map.keys()),
        "targets": TARGETS,
        "target_labels": LABELS,
        "primary_scan_policy": "earliest available scan per PTID",
        "outer_cv": f"GroupKFold({n_folds}) by {args.cv_group_col}",
        "cv_group_column": args.cv_group_col,
        "tabular_features": tab_cols,
        "conditioning_features": cond_cols,
        "rich_file": args.rich,
        "scans_file": args.scans,
        "image_root": args.image_root,
        "note": "Small ViT is trained from scratch; no pretrained checkpoint is loaded.",
    }
    with open(os.path.join(args.out_dir, "S16_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved results in: {os.path.abspath(args.out_dir)}")
    print("Important: report the held-out subject-level R2 and confidence intervals, not the best fold or training R2.")


if __name__ == "__main__":
    main()
