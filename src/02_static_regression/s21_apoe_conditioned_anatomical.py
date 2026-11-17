#!/usr/bin/env python
"""S21: APOE-conditioned anatomical representation learning for ADNI cognition.

Primary scientific question
---------------------------
Does APOE-epsilon4 dosage change how regional structural MRI information is
represented and weighted when estimating MMSE, CDR-SB and ADAS-Cog13?

Cohort/evaluation
-----------------
* Earliest scan with all three cognitive targets observed per PTID.
* Participant-level outer folds; accepts the same S16 fold-map CSV as GATv3.
* MRI node imputation/scaling, demographic imputation/scaling and target scaling
  are fitted on each training fold only.
* Diagnosis is used only to balance inner/outer folds; it is never a predictor.
* APOE is an explicit 0/1/2 dosage embedding in the additive and proposed models.

Models
------
Tabular contribution controls (same ExtraTrees hyperparameters):
  tab_demo_ET          age/sex/education
  tab_genetic_ET       demographics + APOE e4 dosage
  tab_mri_ET           demographics + FastSurfer MRI features
  tab_full_ET          demographics + MRI + APOE (main tabular reference)

Anatomical graph models (36 FastSurfer nodes, fixed anatomical-prior adjacency):
  graph_mri_control        demographics + MRI; APOE-free control only
  graph_apoe_additive      MRI graph + APOE at the final readout
  graph_apoe_conditioned   proposed model: genotype FiLM + explicit low-rank
                           APOE x regional representation interactions +
                           genotype-conditioned regional attention.

Interpretation exports
----------------------
* Per-subject predictions under APOE dosage 0, 1, 2 (model sensitivity only;
  NOT causal counterfactual effects).
* Per-subject regional attention, APOE interaction activation and FiLM
  modulation magnitude. Attention/activation is descriptive, not causal proof.

Important
---------
APOE here is APOE-epsilon4 copy dosage (0/1/2), NOT genome-wide PRS or GWAS.
The graph edges are expert-defined anatomical priors derived from T1/FastSurfer
regions; they are NOT DTI tractography or functional connectivity.

Example:
python apoe_conditioned_anatomical_s21.py --scans scan_table.csv \
  --rich richer_features.csv --fold-map results_final/S16/S16_outer_folds.csv \
  --out-dir results_final/S21_APOEConditioned --device cuda \
  --epochs 150 --patience 20 --batch-size 32 --lr 0.001 --boot 2000

--quick is a smoke test only and is not reportable.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TARGETS = ["MMSCORE", "CDRSB", "TOTAL13"]
ENDPOINTS = ["MMSE", "CDR-SB", "ADAS-Cog13"]

# FreeSurfer aseg / DKT segmentation labels. These define nodes, not connections.
BILATERAL = [
    ("hippocampus", 17, 53, "mtl"),
    ("amygdala", 18, 54, "mtl"),
    ("entorhinal", 1006, 2006, "mtl"),
    ("parahippocampal", 1016, 2016, "mtl"),
    ("fusiform", 1007, 2007, "temporal"),
    ("inferiortemporal", 1009, 2009, "temporal"),
    ("middletemporal", 1015, 2015, "temporal"),
    ("isthmuscingulate", 1010, 2010, "limbic"),
    ("precuneus", 1025, 2025, "limbic"),
    ("lateral_ventricle", 4, 43, "ventricle"),
    ("inf_lat_ventricle", 5, 44, "ventricle"),
    ("thalamus", 10, 49, "subcortical"),
    ("caudate", 11, 50, "subcortical"),
    ("putamen", 12, 51, "subcortical"),
    ("pallidum", 13, 52, "subcortical"),
    ("accumbens", 26, 58, "subcortical"),
    ("cerebral_wm", 2, 41, "white_matter"),
]
MIDLINE = [("third_ventricle", 14, "ventricle"), ("fourth_ventricle", 15, "ventricle")]
NODE_NAMES = [f"{name}_{side}" for name, *_ in BILATERAL for side in ("L", "R")] + [x[0] for x in MIDLINE]
NODE_GROUPS = [g for _, _, _, g in BILATERAL for _ in (0, 1)] + [g for _, _, g in MIDLINE]
NODE_LOOKUP = {}
for k, (name, _, _, _) in enumerate(BILATERAL):
    NODE_LOOKUP[(name, "L")] = 2 * k
    NODE_LOOKUP[(name, "R")] = 2 * k + 1
for j, (name, _, _) in enumerate(MIDLINE):
    NODE_LOOKUP[(name, "M")] = 2 * len(BILATERAL) + j
N_NODES = len(NODE_NAMES)


def seed_all(seed: int):
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


def num(s):
    return pd.to_numeric(s, errors="coerce")


def make_male(d):
    if "male" in d:
        return num(d["male"])
    if "sex" not in d:
        raise ValueError("Need either 'male' or 'sex' in scan_table.csv")
    s = d["sex"]
    if pd.api.types.is_numeric_dtype(s):
        vals = set(num(s).dropna().unique().tolist())
        return num(s) if vals.issubset({0, 1}) else num(s).map({1: 1, 2: 0})
    return s.astype(str).str.strip().str.lower().map({
        "m": 1, "male": 1, "man": 1, "1": 1,
        "f": 0, "female": 0, "woman": 0, "2": 0,
    })


def build_node_features(d, include_intensity=False):
    channels = ["log1p_norm_volume_x1000", "signed_asymmetry", "surface_to_volume", "compactness"]
    if include_intensity:
        channels += ["intensity_std", "intensity_entropy", "intensity_skew"]
    arr = np.full((len(d), N_NODES, len(channels)), np.nan, dtype=np.float32)
    icv_col = next((c for c in ["icv_proxy", "ucsf_icv", "icv"] if c in d), None)
    icv = num(d[icv_col]).replace(0, np.nan) if icv_col else pd.Series(np.nan, index=d.index)

    for name, left_lab, right_lab, _ in BILATERAL:
        for side, lab in [("L", left_lab), ("R", right_lab)]:
            ni = NODE_LOOKUP[(name, side)]
            vcol = f"vol_{lab}"
            if vcol in d and icv_col:
                v = num(d[vcol]) / icv * 1e3
            elif f"av_{lab}" in d:
                v = num(d[f"av_{lab}"])
            elif f"nv_{name}" in d:
                # Fallback only if hemisphere-specific label volumes are unavailable.
                v = num(d[f"nv_{name}"]) / 2.0
            else:
                v = pd.Series(np.nan, index=d.index)
            arr[:, ni, 0] = np.log1p(np.clip(v.to_numpy(dtype=float, na_value=np.nan), 0, None))
            asym_col = f"asym_{name}"
            if asym_col in d:
                asym = num(d[asym_col]).to_numpy(dtype=float, na_value=np.nan)
                arr[:, ni, 1] = asym if side == "L" else -asym
            if name in ("hippocampus", "entorhinal"):
                prefix = "hip" if name == "hippocampus" else "ent"
                for idx, col in [(2, f"{prefix}_surf_to_vol"), (3, f"{prefix}_compactness")]:
                    if col in d:
                        arr[:, ni, idx] = num(d[col]).to_numpy(dtype=float, na_value=np.nan)
                if include_intensity:
                    for idx, suffix in zip(range(4, 7), ["int_std", "int_entropy", "int_skew"]):
                        col = f"{prefix}_{suffix}"
                        if col in d:
                            arr[:, ni, idx] = num(d[col]).to_numpy(dtype=float, na_value=np.nan)

    for name, lab, _ in MIDLINE:
        ni = NODE_LOOKUP[(name, "M")]
        if f"vol_{lab}" in d and icv_col:
            v = num(d[f"vol_{lab}"]) / icv * 1e3
        elif f"nv_{name}" in d:
            v = num(d[f"nv_{name}"])
        else:
            v = pd.Series(np.nan, index=d.index)
        arr[:, ni, 0] = np.log1p(np.clip(v.to_numpy(dtype=float, na_value=np.nan), 0, None))
    return arr, channels


def prepare_data(scans_path, rich_path, include_intensity=False):
    if not os.path.isfile(scans_path):
        raise FileNotFoundError(f"Scan table not found: {scans_path}")
    d = pd.read_csv(scans_path)
    required = ["PTID", "Image_ID", "age", "e4"] + TARGETS
    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError(f"scan table missing columns: {missing}")
    d["PTID"] = d["PTID"].astype(str)
    d["Image_ID"] = d["Image_ID"].astype(str)
    if "EXAMDATE" in d:
        d["EXAMDATE"] = pd.to_datetime(d["EXAMDATE"], errors="coerce")
    d["age"] = num(d["age"])
    d["male"] = make_male(d)
    d["edu"] = num(d["edu"]) if "edu" in d else np.nan
    d["e4"] = num(d["e4"])
    for c in TARGETS:
        d[c] = num(d[c])

    rich_merged = False
    if rich_path:
        if not os.path.isfile(rich_path):
            raise FileNotFoundError(f"Rich feature file not found: {rich_path}")
        r = pd.read_csv(rich_path)
        if "Image_ID" not in r:
            raise ValueError("richer_features.csv needs Image_ID")
        r["Image_ID"] = r["Image_ID"].astype(str)
        overlap = [c for c in r.columns if c in d.columns and c != "Image_ID"]
        d = d.merge(r, on="Image_ID", how="left", suffixes=("", "__rich"), validate="many_to_one")
        for c in overlap:
            rc = c + "__rich"
            if rc in d:
                d[c] = d[rc]
                d.drop(columns=[rc], inplace=True)
        rich_merged = True

    icv_col = next((c for c in ["icv_proxy", "ucsf_icv", "icv"] if c in d), None)
    if icv_col:
        icv = num(d[icv_col]).replace(0, np.nan)
        for c in [c for c in d if re.fullmatch(r"vol_\d+", c)]:
            d["av_" + c[4:]] = num(d[c]) / icv * 1e3

    def safe_sum(cols):
        return d[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(cols))
    if all(c in d for c in ["nv_hippocampus", "nv_lateral_ventricle"]):
        d["sym_hip_vent"] = num(d["nv_hippocampus"]) / (num(d["nv_lateral_ventricle"]) + 1e-6)
    mtl_cols = [c for c in ["nv_hippocampus", "nv_amygdala", "nv_entorhinal", "nv_parahippocampal"] if c in d]
    vent_cols = [c for c in ["nv_lateral_ventricle", "nv_inf_lat_ventricle", "nv_third_ventricle", "nv_fourth_ventricle"] if c in d]
    if len(mtl_cols) >= 3:
        d["sym_mtl"] = safe_sum(mtl_cols)
    if vent_cols:
        d["sym_vent"] = safe_sum(vent_cols)
    if "sym_mtl" in d and "sym_vent" in d:
        d["sym_mtl_vent"] = d["sym_mtl"] / (d["sym_vent"] + 1e-6)

    # One earliest complete visit per participant for cross-sectional analysis.
    d = d.dropna(subset=["PTID", "Image_ID", "age", "male", "e4"] + TARGETS).copy()
    d = d.sort_values(["PTID", "EXAMDATE", "Image_ID"] if "EXAMDATE" in d else ["PTID", "Image_ID"], kind="mergesort")
    n_complete_scans = len(d)
    d = d.groupby("PTID", as_index=False, sort=False).head(1).reset_index(drop=True)
    d["e4"] = d["e4"].clip(0, 2)
    e4_round = np.rint(d["e4"].to_numpy(dtype=float))
    if np.max(np.abs(d["e4"].to_numpy(dtype=float) - e4_round)) > 1e-4:
        raise ValueError("e4 must contain APOE-epsilon4 copy dosage values 0, 1, or 2.")
    d["e4"] = e4_round.astype(int)
    if len(d) < 40:
        raise ValueError(f"Only {len(d)} participants after filtering; check targets and APOE data.")

    # Include available education only if sufficiently complete.
    demo_cols = ["age", "male"]
    if d["edu"].notna().mean() >= 0.65:
        demo_cols.append("edu")

    node_x, node_channels = build_node_features(d, include_intensity)
    tab_cols = [c for c in ["age", "male", "edu", "e4", "f_1536", "f_1537", icv_col] if c and c in d]
    for c in d.columns:
        if c.startswith(("av_", "nv_", "asym_", "hip_", "ent_", "sym_")):
            tab_cols.append(c)
    tab_cols = list(dict.fromkeys(tab_cols))
    banned = set(TARGETS + ["PTID", "Image_ID", "EXAMDATE", "dx", "DX", "uid", "Subject_ID", "scan_idx", "PHASE", "FIELD_STRENGTH"])
    tab_cols = [c for c in tab_cols if c not in banned and pd.api.types.is_numeric_dtype(num(d[c]))]
    tab_cols = [c for c in tab_cols if num(d[c]).notna().sum() >= 10 and num(d[c]).nunique(dropna=True) > 1]

    # The full feature set must include age, sex, APOE dosage and MRI features.
    # MRI controls intentionally exclude e4 but retain demographics.
    genetic_cols = list(dict.fromkeys(demo_cols + ["e4"]))
    mri_cols = [c for c in tab_cols if c not in set(demo_cols + ["e4"])]
    feature_sets = {
        "tab_demo_ET": demo_cols,
        "tab_genetic_ET": genetic_cols,
        "tab_mri_ET": list(dict.fromkeys(demo_cols + mri_cols)),
        "tab_full_ET": list(dict.fromkeys(demo_cols + ["e4"] + mri_cols)),
    }
    # Fail early if any expected inputs have silently disappeared.
    for key, cols in feature_sets.items():
        absent = [c for c in cols if c not in d]
        if absent:
            raise ValueError(f"Feature set {key} missing columns: {absent}")
    if not mri_cols:
        raise ValueError("No FastSurfer MRI features were found. Check the rich feature merge.")

    print(f"Cohort: {n_complete_scans} complete scan rows -> {len(d)} baseline participants; rich file merged={rich_merged}.")
    print(f"MRI feature count={len(mri_cols)}; tabular full feature count={len(feature_sets['tab_full_ET'])}; graph nodes={N_NODES}.")
    print(f"Demographic covariates={demo_cols}; APOE dosage counts={d['e4'].value_counts().sort_index().to_dict()}")
    if "dx" in d:
        print("Diagnosis distribution (split diagnostics only):", num(d.dx).value_counts(dropna=False).to_dict())
    return d, node_x, node_channels, demo_cols, feature_sets


def build_sparse_adjacency():
    A = np.eye(N_NODES, dtype=np.float32)
    def link(a, b):
        i, j = a, b
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Retain left-right homologous links from the GATv3 sparse anatomical prior.
    for name, *_ in BILATERAL:
        link(NODE_LOOKUP[(name, "L")], NODE_LOOKUP[(name, "R")])
    mtl_edges = [("hippocampus", "entorhinal"), ("hippocampus", "parahippocampal"),
                 ("hippocampus", "amygdala"), ("entorhinal", "parahippocampal"),
                 ("entorhinal", "amygdala"), ("parahippocampal", "amygdala")]
    for side in ("L", "R"):
        for a, b in mtl_edges:
            link(NODE_LOOKUP[(a, side)], NODE_LOOKUP[(b, side)])
        for a, b in [("fusiform", "inferiortemporal"), ("fusiform", "middletemporal"),
                     ("inferiortemporal", "middletemporal"), ("entorhinal", "fusiform"),
                     ("entorhinal", "inferiortemporal"), ("parahippocampal", "fusiform"),
                     ("parahippocampal", "inferiortemporal"), ("middletemporal", "isthmuscingulate"),
                     ("middletemporal", "precuneus"), ("isthmuscingulate", "precuneus")]:
            link(NODE_LOOKUP[(a, side)], NODE_LOOKUP[(b, side)])
        for a, b in [("thalamus", "caudate"), ("caudate", "putamen"), ("putamen", "pallidum"),
                     ("pallidum", "accumbens"), ("thalamus", "putamen"), ("caudate", "accumbens")]:
            link(NODE_LOOKUP[(a, side)], NODE_LOOKUP[(b, side)])
        for a, b in [("lateral_ventricle", "caudate"), ("lateral_ventricle", "thalamus"),
                     ("inf_lat_ventricle", "hippocampus"), ("inf_lat_ventricle", "amygdala")]:
            link(NODE_LOOKUP[(a, side)], NODE_LOOKUP[(b, side)])
        for b in ("thalamus", "caudate", "putamen"):
            link(NODE_LOOKUP[("cerebral_wm", side)], NODE_LOOKUP[(b, side)])
        link(NODE_LOOKUP[("third_ventricle", "M")], NODE_LOOKUP[("thalamus", side)])
    return A


def make_folds(df, fold_map_path, n_folds, repeats, seed):
    ptids = df.PTID.astype(str).to_numpy()
    fold_sets = []
    if fold_map_path:
        if not os.path.isfile(fold_map_path):
            raise FileNotFoundError(f"Fold map not found: {fold_map_path}")
        fmap = pd.read_csv(fold_map_path)
        if not {"PTID", "outer_fold"}.issubset(fmap.columns):
            raise ValueError("Fold map must contain PTID,outer_fold (Image_ID optional).")
        fmap["PTID"] = fmap.PTID.astype(str)
        if "Image_ID" in fmap:
            fmap["Image_ID"] = fmap.Image_ID.astype(str)
            by_ptid = fmap.groupby("PTID").outer_fold.nunique()
            if (by_ptid > 1).any():
                raise ValueError("Some PTIDs have multiple outer_fold values.")
            exact = fmap.drop_duplicates(["PTID", "Image_ID"]).set_index(["PTID", "Image_ID"])["outer_fold"]
            fallback = fmap.groupby("PTID").outer_fold.first().to_dict()
            assigned = [exact.get((str(r.PTID), str(r.Image_ID)), fallback.get(str(r.PTID), np.nan)) for r in df.itertuples()]
        else:
            fallback = fmap.groupby("PTID").outer_fold.first().to_dict()
            assigned = [fallback.get(p, np.nan) for p in ptids]
        assigned = np.asarray(assigned)
        if pd.isna(assigned).any():
            missing = sorted(set(ptids[pd.isna(assigned)]))
            raise ValueError(f"Fold map misses {len(missing)} participants, e.g. {missing[:5]}")
        vals = sorted(pd.unique(assigned))
        if len(vals) < 3:
            raise ValueError("Fold map must contain at least 3 folds.")
        fold_sets.append([(np.where(assigned != v)[0], np.where(assigned == v)[0]) for v in vals])
        print(f"Fold map supplied: {len(vals)} folds; participant coverage={len(assigned)}/{len(df)}")
    dx = num(df["dx"]).fillna(-1).astype(int).to_numpy() if "dx" in df else np.zeros(len(df), dtype=int)
    for rep in range(len(fold_sets), repeats):
        _, counts = np.unique(dx, return_counts=True)
        positive_counts = counts[counts > 0]
        nf = min(n_folds, int(positive_counts.min())) if len(positive_counts) else n_folds
        nf = max(3, min(n_folds, nf))
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=seed + rep)
        fold_sets.append(list(skf.split(np.zeros(len(df)), dx)))
    return fold_sets


def fit_node_transform(X, idx):
    tr = X[idx]
    med = np.zeros(X.shape[1:], dtype=np.float32)
    for n in range(X.shape[1]):
        for c in range(X.shape[2]):
            vals = tr[:, n, c]
            vals = vals[np.isfinite(vals)]
            med[n, c] = np.median(vals) if len(vals) else 0.0
    fill = np.where(np.isfinite(tr), tr, med[None])
    mu = fill.mean(axis=0)
    sd = fill.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return med, mu, sd


def apply_node_transform(X, prep):
    med, mu, sd = prep
    missing = (~np.isfinite(X)).astype(np.float32)
    fill = np.where(np.isfinite(X), X, med[None])
    z = (fill - mu[None]) / sd[None]
    return np.concatenate([z, missing], axis=-1).astype(np.float32)


def fit_demo_transform(D, idx):
    X = D.to_numpy(dtype=float)
    med = np.nanmedian(X[idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isfinite(X[idx]), X[idx], med[None])
    mu = filled.mean(axis=0)
    sd = filled.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return med, mu, sd


def apply_demo_transform(D, prep):
    med, mu, sd = prep
    X = D.to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, med[None])
    return ((X - mu[None]) / sd[None]).astype(np.float32)


def inner_split(tr_idx, dx, seed, val_fraction=0.20):
    loc = np.arange(len(tr_idx))
    local_dx = dx[tr_idx]
    unique, counts = np.unique(local_dx, return_counts=True)
    can_stratify = len(unique) > 1 and counts.min() >= 2
    try:
        a, b = train_test_split(loc, test_size=val_fraction, random_state=seed,
                                shuffle=True, stratify=local_dx if can_stratify else None)
    except ValueError:
        a, b = train_test_split(loc, test_size=val_fraction, random_state=seed, shuffle=True)
    return tr_idx[a], tr_idx[b]


def import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("PyTorch is required. Install a CUDA-compatible torch build.") from e
    return torch, nn, DataLoader, TensorDataset


def build_models(adjacency, node_dim, hidden=64, dropout=0.30):
    torch, nn, _, _ = import_torch()
    import math

    class GraphAttentionLayer(nn.Module):
        def __init__(self, dim, adj, drop):
            super().__init__()
            self.register_buffer("adj", torch.as_tensor(adj > 0, dtype=torch.bool))
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.out = nn.Linear(dim, dim)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(drop), nn.Linear(dim * 2, dim))
            self.drop = nn.Dropout(drop)
            self.scale = dim ** -0.5

        def forward(self, x):
            q, k, v = self.q(x), self.k(x), self.v(x)
            scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
            scores = scores.masked_fill(~self.adj[None, :, :], -1e4)
            weights = torch.softmax(scores, dim=-1)
            z = torch.matmul(weights, v)
            x = self.norm1(x + self.drop(self.out(z)))
            x = self.norm2(x + self.drop(self.ff(x)))
            return x

    class APOEAnatomicalNet(nn.Module):
        def __init__(self, demo_dim, mode):
            super().__init__()
            self.mode = mode
            self.hidden = hidden
            self.gene_dim = 9
            self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU()
            )
            self.node_identity = nn.Parameter(torch.randn(N_NODES, hidden) * 0.02)
            self.demo_encoder = nn.Sequential(nn.Linear(demo_dim, 16), nn.GELU(), nn.Dropout(dropout))
            self.e4_embedding = nn.Embedding(3, self.gene_dim - 1)
            nn.init.normal_(self.e4_embedding.weight, mean=0.0, std=0.05)

            # Genotype-conditioned modulation and explicit low-rank interactions.
            self.film = nn.Sequential(nn.Linear(self.gene_dim, 24), nn.GELU(), nn.Linear(24, hidden * 2))
            nn.init.normal_(self.film[-1].weight, mean=0.0, std=0.005)
            nn.init.zeros_(self.film[-1].bias)
            self.node_low_rank = nn.Linear(hidden, 24, bias=False)
            self.gene_low_rank = nn.Linear(self.gene_dim, 24, bias=False)
            self.interaction_project = nn.Sequential(nn.Linear(24, hidden), nn.Tanh())

            self.g1 = GraphAttentionLayer(hidden, adjacency, dropout)
            self.g2 = GraphAttentionLayer(hidden, adjacency, dropout)
            self.node_pool = nn.Sequential(nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1))
            self.node_key = nn.Linear(hidden, 24, bias=False)
            self.gene_query = nn.Linear(self.gene_dim, 24, bias=False)

            # Readout has two pooled MRI summaries + demographics + APOE +
            # (for conditioned model) an explicit interaction summary.
            readout_dim = hidden * 2 + 16 + self.gene_dim + (hidden if mode == "graph_apoe_conditioned" else 0)
            self.head = nn.Sequential(
                nn.Linear(readout_dim, 96), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(96, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 3)
            )

        def forward(self, nodes, demo, e4, return_aux=False):
            d = self.demo_encoder(demo)
            base = self.node_encoder(nodes) + self.node_identity[None, :, :]
            genotype = (torch.cat([self.e4_embedding(e4.long()), (e4.float() / 2.0).unsqueeze(1)], dim=1)
                        if self.mode != "graph_mri_control" else torch.zeros(
                            (nodes.shape[0], self.gene_dim), dtype=nodes.dtype, device=nodes.device
                        ))
            interaction_norm = torch.zeros((nodes.shape[0], N_NODES), dtype=nodes.dtype, device=nodes.device)
            modulation_norm = torch.zeros_like(interaction_norm)
            h = base

            if self.mode == "graph_apoe_conditioned":
                gamma, beta = self.film(genotype).chunk(2, dim=1)
                gamma = 0.35 * torch.tanh(gamma)
                beta = 0.15 * torch.tanh(beta)
                h_film = base * (1.0 + gamma[:, None, :]) + beta[:, None, :]
                low_node = self.node_low_rank(base)
                low_gene = self.gene_low_rank(genotype)[:, None, :]
                interaction = low_node * low_gene
                interaction_h = self.interaction_project(interaction)
                h = h_film + interaction_h
                interaction_norm = torch.linalg.vector_norm(interaction_h, dim=-1)
                modulation_norm = torch.linalg.vector_norm(h - base, dim=-1)

            h = self.g1(h)
            h = self.g2(h)
            pool_logits = self.node_pool(h).squeeze(-1)
            if self.mode == "graph_apoe_conditioned":
                qg = self.gene_query(genotype)
                kn = self.node_key(h)
                genotype_node_score = (kn * qg[:, None, :]).sum(dim=-1) / math.sqrt(24.0)
                pool_logits = pool_logits + genotype_node_score
            attention = torch.softmax(pool_logits, dim=1)
            pooled_attn = torch.sum(h * attention[:, :, None], dim=1)
            pooled_mean = h.mean(dim=1)
            parts = [pooled_attn, pooled_mean, d, genotype]
            if self.mode == "graph_apoe_conditioned":
                parts.append(interaction_h.mean(dim=1))
            pred = self.head(torch.cat(parts, dim=1))
            if return_aux:
                return pred, {
                    "attention": attention,
                    "interaction_norm": interaction_norm,
                    "modulation_norm": modulation_norm,
                }
            return pred

    class GeneticDemoMLP(nn.Module):
        def __init__(self, demo_dim):
            super().__init__()
            self.demo = nn.Sequential(nn.Linear(demo_dim, 16), nn.GELU())
            self.e4 = nn.Embedding(3, 8)
            self.head = nn.Sequential(nn.Linear(25, 48), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 3))
        def forward(self, demo, e4):
            gene = torch.cat([self.e4(e4.long()), (e4.float() / 2.0).unsqueeze(1)], dim=1)
            return self.head(torch.cat([self.demo(demo), gene], dim=1))

    return APOEAnatomicalNet, GeneticDemoMLP


def fit_tree_fold(X, Y, tr, te, seed, estimators=300, min_leaf=2, max_features=0.8):
    """Identical ExtraTrees settings across all feature sets; imputation is fold-local."""
    pred = np.zeros((len(te), Y.shape[1]), dtype=float)
    for j in range(Y.shape[1]):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", ExtraTreesRegressor(
                n_estimators=estimators, min_samples_leaf=min_leaf,
                max_features=max_features, random_state=seed + j * 31,
                n_jobs=-1, bootstrap=False
            ))
        ])
        pipe.fit(X[tr], Y[tr, j])
        pred[:, j] = pipe.predict(X[te])
    return pred


def train_neural_fold(mode, X_raw, D_frame, E4, Y, dx, tr, te, seed, args, device):
    torch, nn, DataLoader, TensorDataset = import_torch()
    GraphModel, GeneticMLP = build_models(build_sparse_adjacency(), X_raw.shape[2] * 2, args.hidden, args.dropout)
    inner_tr, inner_va = inner_split(tr, dx, seed + 17)

    def fit_preps(fit_idx):
        nprep = fit_node_transform(X_raw, fit_idx)
        xall = apply_node_transform(X_raw, nprep)
        dprep = fit_demo_transform(D_frame, fit_idx)
        dall = apply_demo_transform(D_frame, dprep)
        y_mu = Y[fit_idx].mean(axis=0)
        y_sd = Y[fit_idx].std(axis=0)
        y_sd[y_sd < 1e-6] = 1.0
        yall = ((Y - y_mu) / y_sd).astype(np.float32)
        return xall, dall, yall, y_mu, y_sd

    def construct(model_seed):
        seed_all(model_seed)
        if mode == "genetic_demo_mlp":
            return GeneticMLP(D_frame.shape[1]).to(device)
        return GraphModel(D_frame.shape[1], mode).to(device)

    def forward(model, xb, db, eb, aux=False):
        if mode == "genetic_demo_mlp":
            return model(db, eb)
        return model(xb, db, eb, return_aux=aux)

    def train_once(train_idx, val_idx, xall, dall, yall, max_epochs, patience, early, model_seed):
        model = construct(model_seed)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs), eta_min=args.lr * 0.03)
        loss_fn = nn.SmoothL1Loss(beta=1.0)
        ds = TensorDataset(
            torch.from_numpy(xall[train_idx]), torch.from_numpy(dall[train_idx]),
            torch.from_numpy(E4[train_idx].astype(np.int64)), torch.from_numpy(yall[train_idx])
        )
        loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, num_workers=0)
        if early:
            vx = torch.from_numpy(xall[val_idx]).to(device)
            vd = torch.from_numpy(dall[val_idx]).to(device)
            ve = torch.from_numpy(E4[val_idx].astype(np.int64)).to(device)
            vy = torch.from_numpy(yall[val_idx]).to(device)
        best = float("inf")
        best_epoch = max_epochs
        best_state = None
        bad = 0
        hist = []
        for epoch in range(1, max_epochs + 1):
            model.train()
            batch_losses = []
            for xb, db, eb, yb in loader:
                xb, db, eb, yb = xb.to(device), db.to(device), eb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                pr = forward(model, xb, db, eb)
                loss = loss_fn(pr, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                batch_losses.append(float(loss.detach().cpu()))
            scheduler.step()
            val_loss = np.nan
            if early:
                model.eval()
                with torch.no_grad():
                    val_pr = forward(model, vx, vd, ve)
                    val_loss = float(loss_fn(val_pr, vy).detach().cpu())
                hist.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": val_loss})
                if val_loss < best - args.min_delta:
                    best, best_epoch, bad = val_loss, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                    if bad >= patience:
                        break
            else:
                hist.append({"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": np.nan})
        if early and best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, hist

    # Inner split selects epoch count; preprocess using inner-train only.
    xin, din, yin, _, _ = fit_preps(inner_tr)
    inner_model, best_epoch, hist = train_once(inner_tr, inner_va, xin, din, yin,
                                                args.epochs, args.patience, True, seed)
    del inner_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Refit from scratch on all outer-train participants using fold-local statistics.
    xout, dout, yout, y_mu, y_sd = fit_preps(tr)
    final_model, _, _ = train_once(tr, np.array([], dtype=int), xout, dout, yout,
                                    max(1, best_epoch), max(1, best_epoch), False, seed)
    final_model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(xout[te]).to(device)
        dt = torch.from_numpy(dout[te]).to(device)
        et = torch.from_numpy(E4[te].astype(np.int64)).to(device)
        if mode == "genetic_demo_mlp":
            pred_scaled = final_model(dt, et).cpu().numpy()
            aux = None
        else:
            pred_scaled_t, aux_t = final_model(xt, dt, et, return_aux=True)
            pred_scaled = pred_scaled_t.cpu().numpy()
            aux = {k: v.cpu().numpy() for k, v in aux_t.items()}
        pred = pred_scaled * y_sd + y_mu

        # APOE dosage sensitivity for the additive and conditioned models only.
        dose_predictions = None
        if mode in ("graph_apoe_additive", "graph_apoe_conditioned", "genetic_demo_mlp"):
            dose_predictions = {}
            for dose in (0, 1, 2):
                alt_e = torch.full((len(te),), dose, dtype=torch.long, device=device)
                if mode == "genetic_demo_mlp":
                    alt = final_model(dt, alt_e)
                else:
                    alt = final_model(xt, dt, alt_e)
                dose_predictions[dose] = alt.cpu().numpy() * y_sd + y_mu

    del final_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred, best_epoch, hist, aux, dose_predictions


def bootstrap_indices(n, n_boot, seed):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(n_boot)]


def r2_vec(y, p):
    return np.asarray([r2_score(y[:, j], p[:, j]) for j in range(y.shape[1])], dtype=float)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[ix]))
        out[ix] = running
    return out


def save_metrics(predictions, Y, out_dir, boot, seed):
    rows = []
    averaged = {k: v.copy() for k, v in predictions.items()}
    boots = bootstrap_indices(len(Y), boot, seed + 7001)
    for model, pred in averaged.items():
        score = r2_vec(Y, pred)
        mae = np.mean(np.abs(Y - pred), axis=0)
        rmse = np.sqrt(np.mean((Y - pred) ** 2, axis=0))
        bs = np.asarray([r2_vec(Y[ix], pred[ix]) for ix in boots])
        for j, endpoint in enumerate(ENDPOINTS):
            rows.append({"model": model, "endpoint": endpoint, "n_subjects": len(Y),
                         "R2_pooled_OOF": score[j], "R2_boot_lo": np.percentile(bs[:, j], 2.5),
                         "R2_boot_hi": np.percentile(bs[:, j], 97.5), "MAE": mae[j], "RMSE": rmse[j]})
        rows.append({"model": model, "endpoint": "MEAN_R2", "n_subjects": len(Y),
                     "R2_pooled_OOF": score.mean(), "R2_boot_lo": np.percentile(bs.mean(axis=1), 2.5),
                     "R2_boot_hi": np.percentile(bs.mean(axis=1), 97.5), "MAE": mae.mean(), "RMSE": rmse.mean()})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out_dir, "S21_metrics.csv"), index=False)
    return metrics


def save_paired_deltas(predictions, Y, out_dir, boot, seed):
    candidate_comparisons = [
        ("tab_genetic_ET", "tab_demo_ET", "genetic_increment_over_demographics"),
        ("tab_mri_ET", "tab_demo_ET", "mri_increment_over_demographics"),
        ("tab_full_ET", "tab_mri_ET", "apoe_increment_over_mri_tabular"),
        ("graph_apoe_additive", "graph_mri_control", "apoe_increment_in_graph_readout"),
        ("graph_apoe_conditioned", "graph_apoe_additive", "conditioning_interaction_over_additive"),
        ("graph_apoe_conditioned", "tab_full_ET", "conditioned_graph_vs_full_ET"),
        ("graph_apoe_additive", "tab_full_ET", "additive_graph_vs_full_ET"),
    ]
    comparisons = [(m, b, label) for m, b, label in candidate_comparisons
                   if m in predictions and b in predictions]
    boots = bootstrap_indices(len(Y), boot, seed + 7331)
    rows = []
    for model, base, comparison in comparisons:
        point = r2_vec(Y, predictions[model]) - r2_vec(Y, predictions[base])
        b = np.asarray([r2_vec(Y[ix], predictions[model][ix]) - r2_vec(Y[ix], predictions[base][ix]) for ix in boots])
        for j, endpoint in enumerate(ENDPOINTS + ["MEAN_R2"]):
            vals = b[:, j] if j < 3 else b.mean(axis=1)
            delta = point[j] if j < 3 else point.mean()
            pval = min(1.0, max(2 * min(np.mean(vals <= 0), np.mean(vals >= 0)), 1 / (boot + 1)))
            rows.append({"comparison": comparison, "model": model, "baseline": base,
                         "endpoint": endpoint, "delta_R2": delta,
                         "CI_lo": np.percentile(vals, 2.5), "CI_hi": np.percentile(vals, 97.5), "p_boot": pval})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for endpoint, ids in out.groupby("endpoint").groups.items():
        out.loc[ids, "p_holm"] = holm(out.loc[ids, "p_boot"].to_numpy())
    out.to_csv(os.path.join(out_dir, "S21_paired_deltas.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="APOE-conditioned anatomical representation learning (S21)")
    ap.add_argument("--scans", default="scan_table.csv")
    ap.add_argument("--rich", default="richer_features.csv")
    ap.add_argument("--fold-map", default="", help="Same S16 participant-level fold map; PTID,outer_fold; Image_ID optional.")
    ap.add_argument("--out-dir", default="results_final/S21_APOEConditioned")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.30)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tree-estimators", type=int, default=300)
    ap.add_argument("--tree-min-leaf", type=int, default=2)
    ap.add_argument("--tree-max-features", type=float, default=0.8)
    ap.add_argument("--include-intensity", action="store_true")
    ap.add_argument("--no-tree-controls", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--quick", action="store_true", help="Smoke test only; not reportable.")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch, _, _, _ = import_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable in this Python environment.")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if args.quick:
        args.epochs = 3
        args.patience = 2
        args.boot = min(args.boot, 100)
        args.tree_estimators = min(args.tree_estimators, 40)
        args.repeats = 1
        print("*** QUICK SMOKE TEST: results are NOT scientific results. ***")
    seed_all(args.seed)

    df, Xnodes, node_channels, demo_cols, feature_sets = prepare_data(args.scans, args.rich, args.include_intensity)
    ptids = df.PTID.astype(str).to_numpy()
    image_ids = df.Image_ID.astype(str).to_numpy()
    e4 = df.e4.to_numpy(dtype=np.int64)
    D = df[demo_cols].apply(pd.to_numeric, errors="coerce")
    Y = df[TARGETS].to_numpy(dtype=float)
    dx = num(df["dx"]).fillna(-1).astype(int).to_numpy() if "dx" in df else np.zeros(len(df), dtype=int)
    fold_sets = make_folds(df, args.fold_map, args.folds, args.repeats, args.seed)
    if len(fold_sets) > 1 and args.fold_map:
        print("WARNING: only the first repetition matches the supplied fold map; later repeats use newly generated folds.")

    # All models are evaluated on exactly the same out-of-fold participant set.
    tree_names = list(feature_sets) if not args.no_tree_controls else []
    neural_names = ["genetic_demo_mlp", "graph_mri_control", "graph_apoe_additive", "graph_apoe_conditioned"]
    model_names = tree_names + neural_names
    pred_store = {name: np.full((len(fold_sets), len(df), 3), np.nan, dtype=float) for name in model_names}
    choice_rows, curve_rows = [], []
    attention_rows, modulation_rows, dose_rows = [], [], []
    fold_rows, fold_dose_rows = [], []
    for rep, folds in enumerate(fold_sets, start=1):
        for fold_num, (tr, te) in enumerate(folds, start=1):
            for ix in te:
                fold_rows.append({"repetition": rep, "PTID": ptids[ix], "Image_ID": image_ids[ix], "outer_fold": fold_num})
            fold_seed = args.seed + 1000 * (rep - 1) + 29 * fold_num
            for dose in (0, 1, 2):
                fold_dose_rows.append({"repetition": rep, "fold": fold_num, "e4_dosage": dose,
                                       "train_n": int((e4[tr] == dose).sum()),
                                       "test_n": int((e4[te] == dose).sum())})
            print(f"\nOuter fold {fold_num}/{len(folds)}; train={len(tr)} test={len(te)}", flush=True)

            if not args.no_tree_controls:
                for name, cols in feature_sets.items():
                    Xtab = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                    pred = fit_tree_fold(Xtab, Y, tr, te, fold_seed,
                                         estimators=args.tree_estimators,
                                         min_leaf=args.tree_min_leaf,
                                         max_features=args.tree_max_features)
                    pred_store[name][rep - 1, te] = pred
                    choice_rows.append({"repetition": rep, "fold": fold_num, "model": name,
                                        "method": "ExtraTrees", "n_features": len(cols),
                                        "estimators": args.tree_estimators,
                                        "min_samples_leaf": args.tree_min_leaf,
                                        "max_features": args.tree_max_features})
                    print(f"[{name}] done", flush=True)

            for model_i, name in enumerate(neural_names):
                model_seed = fold_seed + (20000 if name == "genetic_demo_mlp" else 10000)
                pred, best_epoch, hist, aux, dose_pred = train_neural_fold(
                    name, Xnodes, D, e4, Y, dx, tr, te, model_seed, args, device
                )
                pred_store[name][rep - 1, te] = pred
                choice_rows.append({"repetition": rep, "fold": fold_num, "model": name,
                                    "method": "PyTorch", "best_epoch": best_epoch,
                                    "seed": model_seed, "hidden": args.hidden, "dropout": args.dropout})
                curve_rows.extend([{"repetition": rep, "fold": fold_num, "model": name, **r} for r in hist])
                if aux is not None:
                    for local_i, global_i in enumerate(te):
                        for node_i, node_name in enumerate(NODE_NAMES):
                            attention_rows.append({
                                "repetition": rep, "fold": fold_num, "PTID": ptids[global_i],
                                "Image_ID": image_ids[global_i], "e4_observed": int(e4[global_i]),
                                "model": name, "node": node_name,
                                "attention_weight": float(aux["attention"][local_i, node_i]),
                            })
                            modulation_rows.append({
                                "repetition": rep, "fold": fold_num, "PTID": ptids[global_i],
                                "Image_ID": image_ids[global_i], "e4_observed": int(e4[global_i]),
                                "model": name, "node": node_name,
                                "interaction_activation_norm": float(aux["interaction_norm"][local_i, node_i]),
                                "film_plus_interaction_modulation_norm": float(aux["modulation_norm"][local_i, node_i]),
                            })
                if dose_pred is not None:
                    for local_i, global_i in enumerate(te):
                        row = {"repetition": rep, "fold": fold_num, "PTID": ptids[global_i],
                               "Image_ID": image_ids[global_i], "e4_observed": int(e4[global_i]), "model": name}
                        for dose in (0, 1, 2):
                            for j, endpoint in enumerate(ENDPOINTS):
                                row[f"pred_{endpoint.replace('-', '_').replace('+', '')}_e4_{dose}"] = float(dose_pred[dose][local_i, j])
                        for j, endpoint in enumerate(ENDPOINTS):
                            key = endpoint.replace("-", "_").replace("+", "")
                            row[f"effect_{key}_e4_1_vs_0"] = float(dose_pred[1][local_i, j] - dose_pred[0][local_i, j])
                            row[f"effect_{key}_e4_2_vs_0"] = float(dose_pred[2][local_i, j] - dose_pred[0][local_i, j])
                            row[f"effect_{key}_e4_2_vs_1"] = float(dose_pred[2][local_i, j] - dose_pred[1][local_i, j])
                            row[f"observed_prediction_{key}"] = float(pred[local_i, j])
                            row[f"observed_target_{key}"] = float(Y[global_i, j])
                        dose_rows.append(row)
                print(f"[{name}] best_epoch={best_epoch}", flush=True)

    # Catch accidental missing fold coverage before saving any metrics.
    pred_mean = {}
    for model, pred_repeats in pred_store.items():
        if np.isnan(pred_repeats).any():
            raise RuntimeError(f"Incomplete OOF predictions for {model}; inspect fold map coverage.")
        pred_mean[model] = np.mean(pred_repeats, axis=0)

    out_pred = df[["PTID", "Image_ID"] + TARGETS + ["e4"]].copy()
    for name, pred in pred_mean.items():
        for j, endpoint in enumerate(ENDPOINTS):
            out_pred[f"{name}__pred_{endpoint}"] = pred[:, j]
        for rep_i in range(pred_store[name].shape[0]):
            for j, endpoint in enumerate(ENDPOINTS):
                out_pred[f"{name}__rep{rep_i+1}__pred_{endpoint}"] = pred_store[name][rep_i, :, j]
    out_pred.to_csv(os.path.join(args.out_dir, "S21_oof_predictions.csv"), index=False)
    metrics = save_metrics(pred_mean, Y, args.out_dir, args.boot, args.seed)
    paired = save_paired_deltas(pred_mean, Y, args.out_dir, args.boot, args.seed)
    pd.DataFrame(choice_rows).to_csv(os.path.join(args.out_dir, "S21_model_choices.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(args.out_dir, "S21_training_curves.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out_dir, "S21_outer_folds.csv"), index=False)
    pd.DataFrame(fold_dose_rows).to_csv(os.path.join(args.out_dir, "S21_fold_apoe_counts.csv"), index=False)
    if attention_rows:
        pd.DataFrame(attention_rows).to_csv(os.path.join(args.out_dir, "S21_region_attention_oof.csv"), index=False)
    if modulation_rows:
        pd.DataFrame(modulation_rows).to_csv(os.path.join(args.out_dir, "S21_region_modulation_oof.csv"), index=False)
    if dose_rows:
        pd.DataFrame(dose_rows).to_csv(os.path.join(args.out_dir, "S21_apoe_dose_sensitivity.csv"), index=False)

    edge_rows = []
    A = build_sparse_adjacency()
    for i in range(N_NODES):
        for j in range(i, N_NODES):
            if A[i, j] > 0:
                edge_rows.append({"node_i": NODE_NAMES[i], "node_j": NODE_NAMES[j], "edge_type": "expert_defined_anatomical_prior"})
    pd.DataFrame(edge_rows).to_csv(os.path.join(args.out_dir, "S21_anatomical_graph_edges.csv"), index=False)
    pd.DataFrame([{"node_index": i, "node": n, "group": NODE_GROUPS[i]} for i, n in enumerate(NODE_NAMES)]).to_csv(
        os.path.join(args.out_dir, "S21_node_manifest.csv"), index=False)
    pd.DataFrame([{"channel_index": i, "channel": c} for i, c in enumerate(node_channels)]).to_csv(
        os.path.join(args.out_dir, "S21_node_feature_manifest.csv"), index=False)
    pd.DataFrame([{"e4_dosage": k, "n_participants": int((e4 == k).sum())} for k in (0, 1, 2)]).to_csv(
        os.path.join(args.out_dir, "S21_apoe_dosage_counts.csv"), index=False)

    metadata = {
        "script": "apoe_conditioned_anatomical_s21.py",
        "n_participants": int(len(df)), "n_repeats": int(len(fold_sets)),
        "device": str(device), "fold_map": args.fold_map,
        "targets": TARGETS, "endpoint_labels": ENDPOINTS,
        "apoe_variable": "APOE-epsilon4 copy dosage (0, 1, 2); not genome-wide PRS",
        "demographic_columns": demo_cols,
        "tabular_feature_sets": feature_sets,
        "models": model_names,
        "proposed_model": "graph_apoe_conditioned",
        "graph_model_initialization": "same fold-specific seed and same module dimensions for MRI-control, additive-APOE and APOE-conditioned graph models",
        "proposed_mechanisms": [
            "APOE-dosage embedding and FiLM modulation of each regional MRI representation",
            "low-rank multiplicative APOE x regional representation interaction",
            "APOE-conditioned attention over anatomical nodes",
        ],
        "graph_caveat": "Expert-defined anatomical prior based on T1/FastSurfer labels; not DTI or fMRI connectivity.",
        "counterfactual_caveat": "Predictions under alternate e4 dosage are model sensitivity analyses, not causal effects.",
        "preprocessing": "Node, demographic and target transforms fitted within each training fold only.",
        "diagnosis_use": "Fold stratification only; dx is not a predictor.",
        "tree_settings": {"n_estimators": args.tree_estimators, "min_samples_leaf": args.tree_min_leaf,
                          "max_features": args.tree_max_features},
        "quick_smoke_test": bool(args.quick),
    }
    with open(os.path.join(args.out_dir, "S21_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== S21 pooled OOF metrics ===")
    print(metrics.round(3).to_string(index=False))
    print("\n=== Paired OOF R2 differences (positive favours first model) ===")
    print(paired.round(4).to_string(index=False))
    print(f"\nFinished. Results saved to: {args.out_dir}")
    if args.quick:
        print("IMPORTANT: --quick was used. Do NOT report these results; they are smoke tests only.")
    print("Reminder: alternate-dosage predictions are model sensitivity analyses, not causal estimates.")


if __name__ == "__main__":
    main()
