"""S11 - Extract RICHER image-derived features from YOUR OWN FastSurfer segmentations (the most plausible source of a real R2 gain).
Per scan it computes:
  vol_<label>      volume (mm3) of every label present in the segmentation (aseg + DKT cortical labels if the file has them)
  nv_<name>        ICV-normalised bilateral volumes (x1e3) of ~20 Alzheimer-relevant structures (hippocampus, amygdala, entorhinal, ventricles, ...)
  asym_<name>      (L-R)/(L+R) asymmetry for hippocampus, amygdala, entorhinal
  hip_*/ent_*      shape + intensity radiomics of the bilateral hippocampus and entorhinal masks: voxel-face surface area, surface/volume, bbox extents,
                   and (if a T1 is supplied) intensity mean/std/skew/kurtosis/entropy inside the mask
Writes richer_features.csv (+ rich_cols.txt = the compact recommended column list) and, with --merge, a merged scan table.

LAYOUT: I have NOT seen your folder structure.  The script searches --seg-root recursively for files named --seg-name whose full path contains the scan's
image id (e.g. 'I293719' or '293719').  Set --seg-name / --t1-name to what FastSurfer wrote (e.g. aparc.DKTatlas+aseg.deep.mgz, aseg.mgz, orig_nu.mgz).
It prints how many scans were matched; if that is low, fix the patterns first.
Needs: pip install nibabel
Run:  python s11_extract_richer_features.py --scans scan_table.csv --seg-root D:\\fastsurfer_out --seg-name aparc.DKTatlas+aseg.deep.mgz --t1-name orig_nu.mgz --merge scan_table.csv --out-table scan_table_rich.csv
Then:  python s7_rigorous_AB.py --scans scan_table_rich.csv --direct age,male,e4,f_1536,f_1537,<columns from rich_cols.txt> ...
"""
import argparse, os, re, warnings
import numpy as np, pandas as pd
import nibabel as nib
from scipy import stats
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv"); ap.add_argument("--seg-root", required=True)
ap.add_argument("--seg-name", default="aparc.DKTatlas+aseg.deep.mgz"); ap.add_argument("--t1-name", default="")
ap.add_argument("--id-col", default="Image_ID"); ap.add_argument("--out", default="richer_features.csv")
ap.add_argument("--merge", default=""); ap.add_argument("--out-table", default="scan_table_rich.csv")
a = ap.parse_args()

NAMES = {  # name: (left labels, right labels)  -- FreeSurfer aseg ids / DKT cortical ids (1000+ left, 2000+ right)
    "hippocampus": ([17], [53]), "amygdala": ([18], [54]), "entorhinal": ([1006], [2006]), "parahippocampal": ([1016], [2016]),
    "fusiform": ([1007], [2007]), "inferiortemporal": ([1009], [2009]), "middletemporal": ([1015], [2015]), "isthmuscingulate": ([1010], [2010]),
    "precuneus": ([1025], [2025]), "lateral_ventricle": ([4], [43]), "inf_lat_ventricle": ([5], [44]), "thalamus": ([10], [49]),
    "caudate": ([11], [50]), "putamen": ([12], [51]), "pallidum": ([13], [52]), "accumbens": ([26], [58]), "cerebral_wm": ([2], [41])}
MID = {"third_ventricle": [14], "fourth_ventricle": [15]}
ASYM = ["hippocampus", "amygdala", "entorhinal"]


def find(root, name, key):
    """first file called `name` (or ending with name's extension if the image id is in the FILE name) whose path contains the exact image-id digits."""
    pat = re.compile(r"(?<!\d)" + re.escape(key) + r"(?!\d)")
    ext = name.split(".", 1)[-1] if "." in name else name
    hits = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            full = os.path.join(dp, f)
            if (f == name and pat.search(dp)) or (pat.search(f) and f.endswith(ext)): hits.append(full)
    return sorted(hits)[0] if hits else None


def faces(mask):
    m = np.pad(mask, 1); s = 0
    for ax in range(3):
        s += (m != np.roll(m, 1, ax)).sum()
    return s / 2.0  # number of exposed voxel faces (each boundary counted twice by the roll difference)


def roi_feats(prefix, mask, vox, img):
    n = int(mask.sum()); f = {f"{prefix}_vox_vol": n * vox}
    if n == 0: return f
    area = faces(mask) * (vox ** (2 / 3)); idx = np.argwhere(mask); ext = (idx.max(0) - idx.min(0) + 1)
    f.update({f"{prefix}_surf_area": area, f"{prefix}_surf_to_vol": area / (n * vox), f"{prefix}_ext_x": ext[0], f"{prefix}_ext_y": ext[1], f"{prefix}_ext_z": ext[2],
              f"{prefix}_compactness": (n * vox) ** (2 / 3) / (area + 1e-9)})
    if img is not None:
        v = img[mask].astype(float)
        if v.size > 10:
            h, _ = np.histogram(v, bins=32); pr = h / h.sum(); pr = pr[pr > 0]
            f.update({f"{prefix}_int_mean": v.mean(), f"{prefix}_int_std": v.std(), f"{prefix}_int_skew": float(stats.skew(v)), f"{prefix}_int_kurt": float(stats.kurtosis(v)),
                      f"{prefix}_int_entropy": float(-(pr * np.log2(pr)).sum())})
    return f


scans = pd.read_csv(a.scans); ids = scans[a.id_col].astype(str).tolist(); rows = []; miss = 0
for i, sid in enumerate(ids):
    key = re.sub(r"\D", "", sid); sp = find(a.seg_root, a.seg_name, key)
    if sp is None: miss += 1; continue
    seg_img = nib.load(sp); seg = np.asarray(seg_img.dataobj).astype(int); vox = float(np.prod(seg_img.header.get_zooms()[:3]))
    t1 = None
    if a.t1_name:
        tp = find(a.seg_root, a.t1_name, key)
        if tp and nib.load(tp).shape == seg.shape: t1 = np.asarray(nib.load(tp).dataobj)
    r = {a.id_col: sid}; labs, cnt = np.unique(seg[seg > 0], return_counts=True)
    for l, c in zip(labs, cnt): r[f"vol_{l}"] = c * vox
    icv = float((seg > 0).sum() * vox); r["icv_proxy"] = icv
    for nm, (L, R) in NAMES.items():
        l = sum(r.get(f"vol_{x}", 0.0) for x in L); rr = sum(r.get(f"vol_{x}", 0.0) for x in R); r[f"nv_{nm}"] = (l + rr) / icv * 1e3
        if nm in ASYM: r[f"asym_{nm}"] = (l - rr) / (l + rr + 1e-9)
    for nm, L in MID.items(): r[f"nv_{nm}"] = sum(r.get(f"vol_{x}", 0.0) for x in L) / icv * 1e3
    for nm, (L, R) in (("hip", NAMES["hippocampus"]), ("ent", NAMES["entorhinal"])):
        r.update(roi_feats(nm, np.isin(seg, L + R), vox, t1))
    rows.append(r)
    if (i + 1) % 25 == 0: print(f"  {i + 1}/{len(ids)} scans, matched {len(rows)}", flush=True)
F = pd.DataFrame(rows); print(f"matched {len(F)}/{len(ids)} scans ({miss} without a segmentation file)")
if len(F) == 0: raise SystemExit("No segmentation files matched: fix --seg-root / --seg-name (see docstring).")
F.to_csv(a.out, index=False)
rec = [c for c in F.columns if c.startswith(("nv_", "asym_")) or c.startswith(("hip_surf", "hip_compact", "hip_int", "ent_surf", "ent_compact", "ent_int"))]
open("rich_cols.txt", "w").write(",".join(rec)); print(f"saved {a.out} ({F.shape[1]} columns) and rich_cols.txt ({len(rec)} recommended columns)")
if a.merge:
    m = pd.read_csv(a.merge).merge(F[[a.id_col] + rec], on=a.id_col, how="left"); m.to_csv(a.out_table, index=False)
    print(f"merged -> {a.out_table}: {int(m[rec[0]].notna().sum())}/{len(m)} scans have the new features")
