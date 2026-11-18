# ADNI-Cognitive-Assessment

# Leakage-Aware Learning for Cognitive Assessment and Progression Prediction in Alzheimer's Disease

**Devashree Parikh, Mehul S. Raval**  
Ahmedabad University

Data obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**.

---

## Overview

This repository provides code for evaluating multimodal neuroimaging representations and leakage-aware validation protocols on a curated cohort of **394 longitudinally segmented 3D T1-weighted MRI scans across 204 participants**.

The research investigates four primary directions:

1. **Static Cognitive Severity Regression:** Benchmarks FastSurfer tabular morphometry (ExtraTrees), cortical graph attention networks (**GATv3** with Hard-Concrete edge gating), and raw voxel models (**ConvViT3D**).

2. **Temporal Progression & Forecasting:** Distinguishes between autoregressive future-level prediction (\(R^2 = 0.788\)) and 1-year change-score prediction (\(\Delta R^2 = 0.041 - 0.079\)).

3. **Diagnostic & Rapid-Decline Classification:** Evaluates multi-compartment neural fusion and tree ensembles for CN vs. AD (\(\text{AUC} = 0.979\)) and 12-month rapid decline (\(\text{AUC} = 0.785\)).

4. **Systematic Leakage Audit:** Evaluates six protocols (\(P_0\)–\(P_5\)) to quantify artificial performance inflation caused by scan-level patient overlap and transductive feature selection.

---

## Key Results

### 1. Static Cognitive Severity Regression

| Target | Model | \(R^2\) (OOF) | 95% Bootstrap CI | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **ADAS-Cog13** | **GATv3 (APOE-conditioned)** | **0.680** | [0.571, 0.759] | 5.26 |
| ADAS-Cog13 | Tabular ExtraTrees | 0.672 | [0.598, 0.737] | 5.29 |
| ADAS-Cog13 | ConvViT3D (Voxel ViT) | 0.358 | [0.185, 0.501] | 7.19 |
| **CDR-SB** | **GATv3 (APOE-conditioned)** | **0.639** | [0.517, 0.749] | 0.87 |
| CDR-SB | Tabular ExtraTrees | 0.586 | [0.489, 0.675] | 1.05 |
| **MMSE** | **GATv3 (MRI control)** | **0.625** | [0.508, 0.709] | 1.69 |
| MMSE | Tabular ExtraTrees | 0.623 | [0.531, 0.700] | 1.70 |

### 2. Systematic Data Leakage Audit

**Evaluation cohort:** 364 scans from 200 subjects.

| Protocol | Splitting & Preprocessing | Real Target \(R^2\) | Inflation (\(\Delta R^2\)) | Permuted Null \(R^2\) |
| :--- | :--- | :---: | :---: | :---: |
| **\(P_0\) (Honest Baseline)** | Subject-grouped K-Fold; fold-fitted scaling/PCA | **0.369** | Baseline | **-0.198** |
| **\(P_1\) (Scan-Level Split)** | Random K-Fold across scans (patient leakage) | **0.463** | **+0.094** | **-0.043** (+0.155) |
| **\(P_3\) (Transductive Selection)** | Global feature selection before subject split | **0.416** | **+0.047** | **-0.169** |
| **\(P_5\) (Compound Leakage)** | Scan split + global PCA + global selection | **0.473** | **+0.104** | **-0.043** (+0.155) |

---

## Repository Structure

```text
ADNI-Cognitive-Assessment/
├── src/
│   ├── 01_preprocessing/          # Scan anchoring and FastSurfer morphometry parsing
│   ├── 02_static_regression/      # GATv3, ExtraTrees, and 3D ConvViT models
│   ├── 03_temporal_forecasting/   # Autoregressive forecasting and longitudinal models
│   ├── 04_classification/         # CN vs. AD and rapid-decline classifiers
│   ├── 05_leakage_audit/          # P0-P5 leakage quantification
│   └── 06_visualization/          # Scripts for reproducing paper figures
├── data/                          # Derived feature matrices and scan metadata
├── results/                       # Experimental outputs and evaluation results
├── Leakage-Aware Learning for Cognitive Assessment and Progression Prediction in Alzheimer’s Disease                         # Final Paper
├── requirements.txt
└── README.md
```

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


## Project Background

This research follows an earlier collaborative CSE641 Computer Vision course project at Ahmedabad University, undertaken with **Dhriti Gandhi, Yash Shah, and Ura Modi**.

The initial project explored MRI-based approaches to Alzheimer's disease, including MedicalNet-derived features. The present work builds upon selected components of that earlier work while investigating additional MRI representations, cognitive prediction tasks, classification approaches, and leakage-aware evaluation.

The subsequent experiments and analyses presented in this repository were conducted independently by **Devashree Parikh**.

Original course-project repository:  
https://github.com/shree226/CV_2026_9_AccessDenied


}
```
