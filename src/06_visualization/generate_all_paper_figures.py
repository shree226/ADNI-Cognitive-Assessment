# Standalone script to reproduce all paper figures
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.makedirs('paper/figures', exist_ok=True)

# ------------------------------------------------------------------
# Figure 1: Static and Longitudinal Regression Panel Comparison
# Source: results/panel_ladder.csv
# ------------------------------------------------------------------
def plot_fig1():
    p = 'results/panel_ladder.csv'
    if not os.path.exists(p): return
    df = pd.read_csv(p)
    baseline = ['MMSCORE_0', 'CDRSB_0', 'TOTAL13_0']
    slopes = ['MMSCORE_slope', 'CDRSB_slope', 'TOTAL13_slope', 'hip_slope_pct', 'ent_slope_pct']
    labels = {'MMSCORE_0': 'MMSE (Baseline)', 'CDRSB_0': 'CDR-SB (Baseline)', 'TOTAL13_0': 'ADAS-Cog13 (Baseline)',
              'MMSCORE_slope': 'MMSE Slope', 'CDRSB_slope': 'CDR-SB Slope', 'TOTAL13_slope': 'ADAS-Cog13 Slope',
              'hip_slope_pct': 'Hippo Atrophy (%/yr)', 'ent_slope_pct': 'Entorhinal Atrophy (%/yr)'}
    
    targets = [t for t in baseline + slopes if t in df.target.values]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(targets))
    w = 0.25
    
    r2_demo = [df[(df.target == t) & (df.arm == 'demographics+APOE')].R2.values[0] for t in targets]
    r2_mtl = [df[(df.target == t) & (df.arm == '+2 MTL')].R2.values[0] for t in targets]
    r2_panel = [df[(df.target == t) & (df.arm == '+full FreeSurfer panel')].R2.values[0] for t in targets]
    
    ax.bar(x - w, r2_demo, width=w, label='Demographics + APOE', color='#a0aec0')
    ax.bar(x, r2_mtl, width=w, label='+ 2 MTL Volumes', color='#4299e1')
    ax.bar(x + w, r2_panel, width=w, label='+ Full FreeSurfer Panel (230 Parcels)', color='#2b6cb0')
    
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(t, t) for t in targets], rotation=25, ha='right')
    ax.set_ylabel('Out-of-Fold R^2')
    ax.set_title('Cross-Validated Regression: Whole-Brain Morphometry vs. Focal Volumes')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig('paper/figures/F1_panel_vs_two_volumes.png', dpi=300)
    plt.close(fig)
    print('Generated F1_panel_vs_two_volumes.png')

# ------------------------------------------------------------------
# Figure 2: Progression Ladders (3-Year Conversion AUC & Cox C-Index)
# Source: results/conversion_auc.csv & results/survival_cindex.csv
# ------------------------------------------------------------------
def plot_fig2():
    p_auc = 'results/conversion_auc.csv'
    p_surv = 'results/survival_cindex.csv'
    if not (os.path.exists(p_auc) and os.path.exists(p_surv)): return
    df_auc = pd.read_csv(p_auc)
    df_surv = pd.read_csv(p_surv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    # AUC
    arms = ['demographics', '+APOE e4', '+2 MTL volumes', '+full FreeSurfer panel']
    auc_vals = [df_auc[df_auc.arm == a].auc.values[0] for a in arms]
    auc_lo = [df_auc[df_auc.arm == a].lo.values[0] for a in arms]
    auc_hi = [df_auc[df_auc.arm == a].hi.values[0] for a in arms]
    y_err1 = [np.array(auc_vals) - np.array(auc_lo), np.array(auc_hi) - np.array(auc_vals)]
    
    ax1.errorbar(range(len(arms)), auc_vals, yerr=y_err1, fmt='o-', color='#3182ce', lw=2, capsize=4, markersize=7)
    ax1.set_xticks(range(len(arms)))
    ax1.set_xticklabels(arms, rotation=20, ha='right')
    ax1.set_ylabel('Out-of-Fold AUC')
    ax1.set_title('MCI -> AD 3-Year Binary Conversion (N=781)')
    ax1.set_ylim(0.55, 0.90)
    ax1.grid(linestyle='--', alpha=0.5)
    
    # C-index
    c_arms = ['demographics', '+APOE e4', '+2 MTL volumes', '+full panel']
    c_vals = [df_surv[df_surv.arm == a].c_index.values[0] for a in c_arms]
    c_lo = [df_surv[df_surv.arm == a].lo.values[0] for a in c_arms]
    c_hi = [df_surv[df_surv.arm == a].hi.values[0] for a in c_arms]
    y_err2 = [np.array(c_vals) - np.array(c_lo), np.array(c_hi) - np.array(c_vals)]
    
    ax2.errorbar(range(len(c_arms)), c_vals, yerr=y_err2, fmt='s-', color='#38a169', lw=2, capsize=4, markersize=7)
    ax2.set_xticks(range(len(c_arms)))
    ax2.set_xticklabels(arms, rotation=20, ha='right')
    ax2.set_ylabel('Harrell C-Index')
    ax2.set_title('Time-to-Conversion Survival Analysis (N=1,040)')
    ax2.set_ylim(0.55, 0.85)
    ax2.grid(linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig('paper/figures/F2_conversion_ladders.png', dpi=300)
    plt.close(fig)
    print('Generated F2_conversion_ladders.png')

plot_fig1()
plot_fig2()
