"""S1 - One row per OWN scan, with targets measured nearest to THAT scan (fixes the baseline-anchor mismatch).
Needs common.py in the same folder.
Run:
  python s1_build_scan_table.py --adni-dir "C:\\Users\\Student\\Downloads\\project\\adni_csvs" ^
     --mri mri_features_with_subjects_overlap.csv --cohort cohort_overlap.csv --out scan_table.csv
"""
import argparse
import numpy as np
import pandas as pd
from common import find_csv, norm_id

ap = argparse.ArgumentParser()
ap.add_argument("--adni-dir", required=True)
ap.add_argument("--mri", default="mri_features_with_subjects_overlap.csv")
ap.add_argument("--cohort", default="cohort_overlap.csv")
ap.add_argument("--out", default="scan_table.csv")
ap.add_argument("--tol-score", type=int, default=90, help="max days between scan and clinical score")
ap.add_argument("--tol-dx", type=int, default=180)
ap.add_argument("--tol-fs", type=int, default=31)
a = ap.parse_args()

raw = pd.read_csv(a.mri)
featcols = [c for c in raw.columns if c.startswith("f_")]
m = raw[["Image_ID", "Subject_ID", "EXAMDATE"]].copy()
m["PTID"] = m.Subject_ID.map(norm_id)
m["EXAMDATE"] = pd.to_datetime(m.EXAMDATE)
m["uid"] = m.Image_ID.astype(str).str.extract(r"(\d+)")[0].astype(int)
c = pd.read_csv(a.cohort)
c["PTID"] = c.PTID.map(norm_id)
c["EXAMDATE_c"] = pd.to_datetime(c.EXAMDATE)

# static covariates from the cohort; age moved forward to the date of each own scan
cov = c[["PTID", "PHASE", "FIELD_STRENGTH", "male", "edu", "e4", "age", "EXAMDATE_c"]]
m = m.merge(cov, on="PTID", how="inner")
m["age"] = m.age + (m.EXAMDATE - m.EXAMDATE_c).dt.days / 365.25
m = m.drop(columns="EXAMDATE_c")
m["_row"] = m.index
m = m.sort_values("EXAMDATE").reset_index(drop=True)


def nearest(left, right, rdate, cols, tol):
    r = right.dropna(subset=[rdate]).sort_values(rdate)
    out = pd.merge_asof(left[["PTID", "EXAMDATE"]].sort_values("EXAMDATE"), r[["PTID", rdate] + cols],
                        left_on="EXAMDATE", right_on=rdate, by="PTID", direction="nearest",
                        tolerance=pd.Timedelta(days=tol))
    return out.drop(columns=[rdate]).drop_duplicates(["PTID", "EXAMDATE"])


for table, col in [("MMSE", "MMSCORE"), ("CDR", "CDRSB"), ("ADAS", "TOTAL13")]:
    d = pd.read_csv(find_csv(a.adni_dir, table), low_memory=False)
    d["PTID"] = d.PTID.map(norm_id); d["VISDATE"] = pd.to_datetime(d.VISDATE, errors="coerce")
    d = d[d[col] >= 0].dropna(subset=[col])
    m = m.merge(nearest(m, d, "VISDATE", [col], a.tol_score), on=["PTID", "EXAMDATE"], how="left")

dx = pd.read_csv(find_csv(a.adni_dir, "DXSUM"), low_memory=False)
dx["PTID"] = dx.PTID.map(norm_id); dx["EXAMDATE"] = pd.to_datetime(dx.EXAMDATE, errors="coerce")
dx = dx.dropna(subset=["DIAGNOSIS"]).rename(columns={"EXAMDATE": "DXDATE"})
m = m.merge(nearest(m, dx, "DXDATE", ["DIAGNOSIS"], a.tol_dx).rename(columns={"DIAGNOSIS": "dx"}),
            on=["PTID", "EXAMDATE"], how="left")      # 1=CN 2=MCI 3=AD

# UCSF FreeSurfer reference volumes (by image id, else nearest date)
f = pd.read_csv(find_csv(a.adni_dir, "UCSFFSX7"), low_memory=False)
f["PTID"] = f.PTID.map(norm_id); f["FDATE"] = pd.to_datetime(f.EXAMDATE, errors="coerce")
f = f.dropna(subset=["ST29SV", "ST88SV", "ST24CV", "ST83CV", "ST10CV"])
f["ucsf_hippo"] = (f.ST29SV + f.ST88SV) / 2; f["ucsf_ento"] = (f.ST24CV + f.ST83CV) / 2; f["ucsf_icv"] = f.ST10CV
by_id = f.drop_duplicates("IMAGEUID").set_index("IMAGEUID")[["ucsf_hippo", "ucsf_ento", "ucsf_icv"]]
m = m.join(by_id, on="uid")
miss = m.ucsf_hippo.isna()
if miss.any():
    near = nearest(m[miss], f, "FDATE", ["ucsf_hippo", "ucsf_ento", "ucsf_icv"], a.tol_fs).set_index(["PTID", "EXAMDATE"])
    key = pd.MultiIndex.from_frame(m.loc[miss, ["PTID", "EXAMDATE"]])
    for col in ["ucsf_hippo", "ucsf_ento", "ucsf_icv"]:
        m.loc[miss, col] = near[col].reindex(key).values

m["t_years"] = m.groupby("PTID").EXAMDATE.transform(lambda s: (s - s.min()).dt.days / 365.25)
m["scan_idx"] = m.groupby("PTID").cumcount()
m = m.drop_duplicates(["PTID", "EXAMDATE", "Image_ID"])
m = m.merge(raw[["Image_ID"] + featcols], on="Image_ID", how="left")
m.drop(columns="_row").to_csv(a.out, index=False)

print(f"saved {a.out}: {len(m)} scans / {m.PTID.nunique()} subjects")
for c_ in ["MMSCORE", "CDRSB", "TOTAL13", "dx", "ucsf_hippo"]:
    print(f"  {c_:11s} available on {int(m[c_].notna().sum())} scans / {m[m[c_].notna()].PTID.nunique()} subjects")
print("dx (1=CN 2=MCI 3=AD), per subject baseline scan:", m.groupby("PTID").dx.first().value_counts().to_dict())
