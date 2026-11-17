# Leakage-Aware Learning for Cognitive Assessment and Progression Prediction in Alzheimer's Disease

Official code repository for:

**"Leakage-Aware Learning for Cognitive Assessment and Progression Prediction in Alzheimer's Disease"**

*Devashree Parikh, Mehul S. Raval*
(Ahmedabad University)

Data obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**.

---

## Overview

This repository provides reproducible code for evaluating multimodal neuroimaging representations and validation protocols on a curated cohort of **394 longitudinally segmented 3D T1-weighted MRI scans across 204 participants**:

1. **Static Cognitive Severity Regression:** Benchmarks FastSurfer tabular morphometry (ExtraTrees), cortical graph attention networks (GATv3 with Hard-Concrete edge gating), and raw voxel models (ConvViT3D).

2. **Temporal Progression & Forecasting:** Distinguishes between autoregressive future-level prediction (R² = 0.788) and 1-year change-score prediction (ΔR² = 0.041 - 0.079).

3. **Diagnostic & Rapid-Decline Classification:** Evaluates multi-compartment neural fusion and tree ensembles for CN vs. AD (AUC = 0.979) and 12-month rapid decline (AUC = 0.785).

4. **The 6-Protocol Leakage Audit (P0-P5):** Quantifies artificial performance inflation caused by scan-level patient overlap (+0.094 real / +0.155 null) and transductive feature selection.

---

## Key Results

### 1. Static Cognitive Severity Regression

| Target | Model | R² (OOF) | 95% Bootstrap CI | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **ADAS-Cog13** | **GATv3 (APOE-conditioned)** | **0.680** | [0.571, 0.759] | 5.26 |
| ADAS-Cog13 | Tabular ExtraTrees | 0.672 | [0.598, 0.737] | 5.29 |
| ADAS-Cog13 | ConvViT3D (Voxel ViT) | 0.358 | [0.185, 0.501] | 7.19 |
| **CDR-SB** | **GATv3 (APOE-conditioned)** | **0.639** | [0.517, 0.749] | 0.87 |
| CDR-SB | Tabular ExtraTrees | 0.586 | [0.489, 0.675] | 1.05 |
| **MMSE** | **GATv3 (MRI control)** | **0.625** | [0.508, 0.709] | 1.69 |
| MMSE | Tabular ExtraTrees | 0.623 | [0.531, 0.700] | 1.70 |

### 2. Systematic Data Leakage Audit

N = 364 scans, N = 200 subjects.

| Protocol | Splitting & Preprocessing | Real Target R² | Inflation (ΔR²) | Permuted Null R² |
| :--- | :--- | :---: | :---: | :---: |
| **P0 (Honest Baseline)** | Subject-grouped K-Fold; fold-fitted scaling/PCA | **0.369** | Baseline | **-0.198** |
| **P1 (Scan-Level Split)** | Random K-Fold across scans (patient leak) | **0.463** | **+0.094** | **-0.043 (+0.155)** |
| **P3 (Transductive Sel)** | Global feature selection before subject split | **0.416** | **+0.047** | **-0.169** |
| **P5 (Compound Leaked)** | Scan split + global PCA + global selection | **0.473** | **+0.104** | **-0.043 (+0.155)** |

---

## Repository Structure

```text
├── src/
│   ├── 01_preprocessing/
│   ├── 02_static_regression/
│   ├── 03_temporal_forecasting/
│   ├── 04_classification/
│   ├── 05_leakage_audit/
│   └── 06_visualization/
├── data/
├── results/
├── paper/
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/shree226/ADNI-Cognitive-Assessment.git

cd ADNI-Cognitive-Assessment

pip install -r requirements.txt
```

### 2. Run the Static Regression Benchmark

```bash
python src/02_static_regression/s21_apoe_conditioned_anatomical.py
```

### 3. Run the Leakage Audit

```bash
python src/05_leakage_audit/s8_leakage_audit.py --scans data/scan_table.csv --reps 30 --null-reps 15
```

---

## Citation

If you use this codebase or benchmark results in your research, please cite:

```bibtex
@article{parikh2026leakage,
  title={Leakage-Aware Learning for Cognitive Assessment and Progression Prediction in Alzheimer's Disease},
  author={Parikh, Devashree and Raval, Mehul S.},
  journal={arXiv preprint},
  year={2026}
}
```
