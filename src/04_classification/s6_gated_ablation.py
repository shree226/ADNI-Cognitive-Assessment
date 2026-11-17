"""S6 - (extends S5) APOE-conditioned anatomical gating with ABLATIONS, paired comparisons, and optional CN-vs-AD classification.
Extra arms: nn_gated_noapoe (gate sees no genetic/clinical context), nn_gated_wb (whole-brain block only), nn_gated_roi (ROI blocks only).
Paired differences reported: nn_gated vs nn_concat / noapoe / wb / roi.  Add --classify to predict CN(1) vs AD(3) from the 'dx' column (AUC).
"""
"""S5 - Small neural fusion models on your extracted features (static OR forecasting), same folds/subjects for every arm.

Arms
  ridge_direct     RidgeCV on direct columns (volumes + clinical)            <- the baseline to beat
  ridge_concat     RidgeCV on direct + per-block PCA                         <- plain concatenation
  nn_concat        per-block projection -> concat with direct -> small MLP
  nn_gated         per-block projection -> APOE/age/sex-conditioned softmax gate over blocks -> MLP (+direct)
  nn_resid_gated   PROPOSED: linear anchor on direct columns (volumes+clinical) + gated deep residual, residual head zero-initialised,
                   so the deep branch can only ADD to the volume model.  Multi-task (all targets predicted jointly).
Design for ~200 subjects: PCA inside each fold, tiny widths, dropout, weight decay, fixed epochs, seed-ensembles. No tuning on test folds.

STATIC   (scan_table.csv):
  python s5_deep_fusion.py --data scan_table.csv --targets MMSCORE,CDRSB,TOTAL13 ^
     --blocks f_:0:512,f_:512:1024,f_:1024:1536 --direct age,male,e4,f_1536,f_1537 --cond age,male,e4 --out-dir results_final/static_nn
FORECAST (forecast_table.csv from s4):
  python s5_deep_fusion.py --data forecast_table.csv --targets y_MMSCORE,y_CDRSB,y_TOTAL13 ^
     --blocks f_:0:512,f_:512:1024,f_:1024:1536,df_:0:512,df_:512:1024,df_:1024:1536 ^
     --direct age,male,e4,cur_MMSCORE,cur_CDRSB,cur_TOTAL13,f_1536,f_1537,df_1536,df_1537,gap --cond age,male,e4 --out-dir results_final/forecast_nn
Needs: pip install torch   (CPU is fine)
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--targets", required=True)
ap.add_argument("--blocks", required=True); ap.add_argument("--direct", required=True); ap.add_argument("--cond", required=True)
ap.add_argument("--group", default="PTID"); ap.add_argument("--out-dir", default="results_final/nn")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--pcs", type=int, default=16); ap.add_argument("--d", type=int, default=8)
ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--classify", action="store_true"); ap.add_argument("--boot", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

T = a.targets.split(","); DIRECT = a.direct.split(","); COND = a.cond.split(",")
BLOCKS = []
for item in a.blocks.split(","):
    pre, s, e = item.split(":"); BLOCKS.append([f"{pre}{i}" for i in range(int(s), int(e))])
d = pd.read_csv(a.data)
if a.classify:
    d = d[d.dx.isin([1, 3])].copy(); d["ad"] = (d.dx == 3).astype(float); T = ["ad"]
d = d.dropna(subset=T + [c for c in COND if c in d]).reset_index(drop=True)
print(f"{len(d)} rows / {d[a.group].nunique()} subjects | targets {T} | blocks {len(BLOCKS)} | direct {len(DIRECT)}")
ARMS = ["ridge_direct", "ridge_concat", "nn_concat", "nn_gated", "nn_gated_noapoe", "nn_gated_wb", "nn_gated_roi", "nn_resid_gated"]
PAIRS = [("nn_gated", "nn_concat"), ("nn_gated", "nn_gated_noapoe"), ("nn_gated", "nn_gated_wb"), ("nn_gated", "nn_gated_roi")]
BASE = "ridge_direct"


def prep(tr, te):
    """fit all preprocessing on the training rows only."""
    def fitcols(cols, pca=False):
        Xtr = d.loc[tr, cols].astype(float); med = Xtr.median().fillna(0); Xtr = Xtr.fillna(med); Xte = d.loc[te, cols].astype(float).fillna(med)
        sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        if pca:
            p = PCA(min(a.pcs, len(Xtr) - 1), random_state=0).fit(Ztr); k = np.sqrt(p.explained_variance_[0])
            Ztr, Zte = p.transform(Ztr) / k, p.transform(Zte) / k
            if Ztr.shape[1] < a.pcs:
                Ztr = np.pad(Ztr, ((0, 0), (0, a.pcs - Ztr.shape[1]))); Zte = np.pad(Zte, ((0, 0), (0, a.pcs - Zte.shape[1])))
        return Ztr.astype(np.float32), Zte.astype(np.float32)
    B = [fitcols(b, True) for b in BLOCKS]; Dd = fitcols(DIRECT); C = fitcols(COND)
    Ytr = d.loc[tr, T].values.astype(float); mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    if a.classify: mu, sd = np.zeros(len(T)), np.ones(len(T))
    return dict(Btr=[b[0] for b in B], Bte=[b[1] for b in B], Dtr=Dd[0], Dte=Dd[1], Ctr=C[0], Cte=C[1],
                Ytr=((Ytr - mu) / sd).astype(np.float32), mu=mu, sd=sd)


class Net(nn.Module):
    def __init__(s, nb, pcs, nd, nc, nt, mode, dd, drop, hid=16):
        super().__init__(); s.mode = mode
        s.proj = nn.ModuleList([nn.Linear(pcs, dd) for _ in range(nb)]); s.drop = nn.Dropout(drop)
        if mode in ("gated", "resid"): s.gate = nn.Linear(nc, nb)
        din = nb * dd + nd if mode == "concat" else (dd if mode == "resid" else dd + nd)
        s.head = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, nt))
        if mode == "resid":
            s.anchor = nn.Linear(nd, nt)
            nn.init.zeros_(s.head[-1].weight); nn.init.zeros_(s.head[-1].bias)

    def weights(s, C): return torch.softmax(s.gate(C), 1)

    def forward(s, B, D, C):
        h = [s.drop(torch.tanh(p(b))) for p, b in zip(s.proj, B)]
        if s.mode == "concat": z = torch.cat(h + [D], 1)
        else:
            w = s.weights(C); fused = sum(w[:, i:i + 1] * h[i] for i in range(len(h)))
            z = fused if s.mode == "resid" else torch.cat([fused, D], 1)
        out = s.head(z)
        return out + s.anchor(D) if s.mode == "resid" else out


def train_predict(P, mode, seed, bsel=None, nocond=False):
    torch.manual_seed(seed); bsel = list(range(len(BLOCKS))) if bsel is None else bsel
    B = [torch.from_numpy(P["Btr"][i]) for i in bsel]; D = torch.from_numpy(P["Dtr"]); Y = torch.from_numpy(P["Ytr"])
    C = torch.from_numpy(np.zeros_like(P["Ctr"]) if nocond else P["Ctr"])
    net = Net(len(bsel), a.pcs, D.shape[1], C.shape[1], len(T), mode, a.d, a.drop)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.wd)
    for _ in range(a.epochs):
        net.train(); opt.zero_grad(); out = net(B, D, C); loss = nn.functional.binary_cross_entropy_with_logits(out, Y) if a.classify else ((out - Y) ** 2).mean(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Bt = [torch.from_numpy(P["Bte"][i]) for i in bsel]; Dt = torch.from_numpy(P["Dte"])
        Ct = torch.from_numpy(np.zeros_like(P["Cte"]) if nocond else P["Cte"])
        pred = net(Bt, Dt, Ct).numpy() * P["sd"] + P["mu"]
        gate = net.weights(Ct).numpy() if mode in ("gated", "resid") else None
    return pred, gate


def folds(rep):
    rng = np.random.RandomState(a.seed + rep); u = rng.permutation(d[a.group].unique()); fo = {g: i % a.folds for i, g in enumerate(u)}
    f = d[a.group].map(fo).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(a.folds)]



def ridge_pred(Xtr, Xte, P):
    m = RidgeCV(
        alphas=np.logspace(-2, 4, 13)
    ).fit(Xtr, P["Ytr"])

    pred = np.asarray(m.predict(Xte))

    # Keep single-target predictions as (n_samples, 1).
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    mu = np.asarray(P["mu"]).reshape(1, -1)
    sd = np.asarray(P["sd"]).reshape(1, -1)

    return pred * sd + mu
#def ridge_pred(Xtr, Xte, P):
#    m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr, P["Ytr"])
#    return m.predict(Xte) * P["sd"] + P["mu"]


preds = {k: np.zeros((len(d), len(T))) for k in ARMS}
for rep in range(a.reps):
    for tr, te in folds(rep):
        P = prep(tr, te)
        preds["ridge_direct"][te] += ridge_pred(P["Dtr"], P["Dte"], P) / a.reps
        preds["ridge_concat"][te] += ridge_pred(np.hstack(P["Btr"] + [P["Dtr"]]), np.hstack(P["Bte"] + [P["Dte"]]), P) / a.reps
        roi = list(range(1, len(BLOCKS))) if len(BLOCKS) < 6 else [i for i in range(len(BLOCKS)) if i % 3 != 0]
        wb = [0] if len(BLOCKS) < 6 else [i for i in range(len(BLOCKS)) if i % 3 == 0]
        CFG = [("nn_concat", "concat", None, False), ("nn_gated", "gated", None, False), ("nn_gated_noapoe", "gated", None, True),
               ("nn_gated_wb", "gated", wb, False), ("nn_gated_roi", "gated", roi, False), ("nn_resid_gated", "resid", None, False)]
        for arm, mode, bsel, nc in CFG:
            pr = np.mean([train_predict(P, mode, a.seed * 1000 + rep * 100 + s, bsel, nc)[0] for s in range(a.seeds)], axis=0)
            preds[arm][te] += pr / a.reps
    print(f"  finished repetition {rep + 1}/{a.reps}", flush=True)

from sklearn.metrics import roc_auc_score
score = (lambda y, p: roc_auc_score(y, p)) if a.classify else (lambda y, p: r2_score(y, p))
MET = "AUC" if a.classify else "R2"
idx = d.groupby(a.group).indices; keys = list(idx); rng = np.random.RandomState(a.seed); rows = []; prow = []
for j, t in enumerate(T):
    y = d[t].values; boots = [np.concatenate([idx[k] for k in rng.choice(keys, len(keys))]) for _ in range(a.boot)]
    boots = [s_ for s_ in boots if len(np.unique(y[s_])) > 1]
    for arm in ARMS:
        e = [score(y[s_], preds[arm][s_, j]) for s_ in boots]; dl = [score(y[s_], preds[arm][s_, j]) - score(y[s_], preds[BASE][s_, j]) for s_ in boots]
        rows.append(dict(target=t, arm=arm, metric=MET, est=score(y, preds[arm][:, j]), lo=np.percentile(e, 2.5), hi=np.percentile(e, 97.5),
                         delta_vs_ridge_direct=0.0 if arm == BASE else score(y, preds[arm][:, j]) - score(y, preds[BASE][:, j]),
                         d_lo=np.nan if arm == BASE else np.percentile(dl, 2.5), d_hi=np.nan if arm == BASE else np.percentile(dl, 97.5),
                         n=len(d), n_subj=len(keys)))
    for A, B_ in PAIRS:
        dl = [score(y[s_], preds[A][s_, j]) - score(y[s_], preds[B_][s_, j]) for s_ in boots]
        prow.append(dict(target=t, comparison=f"{A} - {B_}", delta=score(y, preds[A][:, j]) - score(y, preds[B_][:, j]),
                         lo=np.percentile(dl, 2.5), hi=np.percentile(dl, 97.5)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out_dir, "nn_results.csv"), index=False)
Pp = pd.DataFrame(prow); Pp.to_csv(os.path.join(a.out_dir, "paired_comparisons.csv"), index=False)
pd.set_option("display.width", 200)
print(R.round(3).drop(columns=["n", "n_subj"]).to_string(index=False))
print("\nPAIRED (is the gating / APOE conditioning / ROI blocks actually helping?)  interval excluding 0 = reliable")
print(Pp.round(3).to_string(index=False))

# interpretability: mean gate weight per block, split by APOE e4 (fit once on all rows, resid model)
if "e4" in d.columns and len(BLOCKS) > 1:
    allidx = np.arange(len(d)); P = prep(allidx, allidx); _, g = train_predict(P, "gated", a.seed)
    names = [f"block{i}" for i in range(len(BLOCKS))]
    print("\nmean gate weights (nn_gated, fit on all rows; descriptive only):")
    print(pd.DataFrame(g, columns=names).groupby((d.e4 > 0).map({False: "e4=0", True: "e4>=1"}).values).mean().round(3).to_string())
print(f"\nsaved {a.out_dir}/nn_results.csv   (d_lo/d_hi excluding 0 = reliable difference from the volume-ridge baseline)")