"""S4 - Forecasting table: one row per subject.
Inputs  = LAST own scan (deep feats f_*, volumes), CHANGE between first and last own scan per year (df_*), current scores, APOE/age/sex.
Targets = annualised change from the current score (at the last scan) to a FUTURE visit 0.5-2 years AFTER the last scan
          (nearest to +1 year).  y_MMSCORE, y_CDRSB, y_TOTAL13.   Subjects with no such future visit are dropped.
Run:  python s4_build_forecast_table.py --scans scan_table.csv --adni-dir "C:\\...\\adni_csvs" --out forecast_table.csv
"""
import argparse
import numpy as np, pandas as pd
from common import find_csv, norm_id

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="scan_table.csv"); ap.add_argument("--adni-dir", required=True)
ap.add_argument("--out", default="forecast_table.csv")
ap.add_argument("--min-ahead", type=float, default=0.5); ap.add_argument("--max-ahead", type=float, default=2.0)
ap.add_argument("--target-ahead", type=float, default=1.0)
a = ap.parse_args()

s = pd.read_csv(a.scans, parse_dates=["EXAMDATE"]).sort_values(["PTID", "EXAMDATE"])
fc = [c for c in s.columns if c.startswith("f_")]
SC = {"MMSCORE": "MMSCORE", "CDRSB": "CDRSB", "TOTAL13": "TOTAL13"}
tabs = {}
for table, col in [("MMSE", "MMSCORE"), ("CDR", "CDRSB"), ("ADAS", "TOTAL13")]:
    d = pd.read_csv(find_csv(a.adni_dir, table), low_memory=False)
    d["PTID"] = d.PTID.map(norm_id); d["VISDATE"] = pd.to_datetime(d.VISDATE, errors="coerce")
    tabs[col] = d[d[col] >= 0].dropna(subset=["VISDATE", col])[["PTID", "VISDATE", col]]

rows = []
for pid, g in s.groupby("PTID"):
    first, last = g.iloc[0], g.iloc[-1]
    gap = (last.EXAMDATE - first.EXAMDATE).days / 365.25
    r = {"PTID": pid, "age": last.age, "male": last.male, "e4": last.e4, "gap": gap, "n_scans": len(g),
         "last_scan": last.EXAMDATE, "has_delta": int(gap >= 0.25)}
    for c in fc:
        r[c] = last[c]; r["df_" + c[2:]] = (last[c] - first[c]) / gap if gap >= 0.25 else 0.0
    ok = False
    for col in SC:
        r["cur_" + col] = last[col]
        t = tabs[col]; t = t[t.PTID == pid]
        ahead = (t.VISDATE - last.EXAMDATE).dt.days / 365.25
        t = t[(ahead >= a.min_ahead) & (ahead <= a.max_ahead)]
        if len(t) and pd.notna(last[col]):
            ah = (t.VISDATE - last.EXAMDATE).dt.days / 365.25
            j = (ah - a.target_ahead).abs().idxmin()
            r["y_" + col] = (t.loc[j, col] - last[col]) / ah.loc[j]; r["ahead_" + col] = ah.loc[j]; ok = True
    if ok: rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(a.out, index=False)
print(f"saved {a.out}: {len(out)} subjects with a future visit {a.min_ahead}-{a.max_ahead} y after their last own scan")
for col in SC: print(f"  y_{col}: {int(out['y_' + col].notna().sum()) if 'y_' + col in out else 0} subjects")
print(f"  with a first-to-last scan change (gap >= 0.25 y): {int(out.has_delta.sum())}")
print("  complete targets (all three):", int(out[[f'y_{c}' for c in SC]].notna().all(axis=1).sum()))
