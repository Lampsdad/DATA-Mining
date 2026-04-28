# Results & Interpretation — Tornado Outbreak Severity Classification

> Analysis window: 1996–2024 (full NCEI coverage) | Feature matrix: 9 SPC + NCEI tabular features for clustering, 26 features for regression

---

## 1. Data Summary

| Metric                        | Value                              |
| ----------------------------- | ---------------------------------- |
| SPC clean events              | 70,456                             |
| Outbreak days (>=6 events)    | 3,589                              |
| Events in outbreak days       | 50,061                             |
| NCEI episodes available       | 41,140 events                      |
| NCEI coverage (1996-2024)     | 48% (1,740 of 3,589 outbreak days) |
| EF4+ outbreak days            | 343 (9.6%)                         |
| Days with QLCS narrative      | 32.0%                              |
| Days with supercell narrative | 53.5%                              |

---

## 2. Merge Validation

Event-level validation of the SPC-NCEI merge confirms data quality:

| Check                                        | Result                        | Verdict                                 |
| -------------------------------------------- | ----------------------------- | --------------------------------------- |
| Per-day count Spearman ρ (1,740 shared days) | **0.954**                     | PASS                                    |
| Median(SPC count / NCEI count)               | 0.92                          | NCEI ~8% more events (per-segment rows) |
| Event-level NN match (50 km / 1 h tolerance) | 76.7% (24,933 / 32,508)       | Expected — NCEI per-segment convention  |
| State agreement on matched pairs             | **99.7%**                     | PASS                                    |
| Timezone audit (median |Δt|, hours)           | 0.00 h, p90 1.0 h            | PASS                                    |

The merge is well-aligned in time and space; the 23% unmatched fraction reflects NCEI's per-segment-row convention for multi-county tornadoes.

---

## 3. Clustering Results

### 3.1 Model Selection — Wide Bake-Off

Of 100+ (method × subset × hyperparameter) combinations, 59 passed the non-degenerate filter (every cluster ≥ 30 points, noise ≤ 50%). Top non-degenerate results:

| Method      | Subset      | K     | Sizes           | Silhouette | Bootstrap ARI |
| ----------- | ----------- | ----- | --------------- | ---------- | ------------- |
| GMM         | GEOMETRY    | 2     | 1,675 / 65      | 0.375      | 0.228         |
| **K-Means** | **TABULAR** | **2** | **1,392 / 348** | **0.320**  | **0.865**     |
| FCM         | TABULAR     | 2     | 1,037 / 703     | 0.218      | —             |
| GMM         | TABULAR     | 2     | 1,125 / 615     | 0.204      | 0.618         |
| Spectral    | TABULAR     | 2     | 527 / 473       | 0.188      | —             |
| K-Means     | GEOMETRY    | 3     | 771 / 587 / 382 | 0.174      | 0.802         |

The headline is K-Means K=2 on TABULAR features: silhouette 0.320 with bootstrap ARI 0.865 shows partition recovers on 86.5% of resampled subsets, indicating stable structure.

### 3.2 Cluster Profile

|                             | C0 (n=1,392, 80%) | C1 (n=348, 20%) | C1 / C0 |
| --------------------------- | ----------------- | --------------- | ------- |
| event_count                 | 10.9              | 35.8            | 3.3×    |
| max_mag                     | 1.52              | 2.87            | —       |
| violent_count (EF4+ on day) | 0.02              | 0.38            | 19×     |
| fatality_total (mean)       | 0.26              | 4.75            | 18×     |
| has_ef4plus rate            | 2.2%              | **23.6%**       | 10.7×   |
| qlcs_flag rate              | 27%               | 54%             | 2.0×    |
| supercell_flag rate         | 47%               | 79%             | 1.7×    |
| Spring (Apr–May) share      | 31%               | 49%             | —       |
| Central plains (lat 35–40)  | 36%               | 49%             | —       |

**1996–2024 totals:** C0 produced 362 deaths in 1,392 days; C1 produced **1,653 deaths in 348 days**. C1 is 20% of the days but accounts for **82% of all fatalities**. All four spot-checked named historic outbreaks (1999 OKC, 2011 Apr 27, 2013 Moore, 2021 Quad-State) land in C1.

**Important caveat:** C1 is not a "QLCS vs. supercell" archetype. Both narrative flag rates are elevated in C1 (79% supercell + 54% QLCS), consistent with major outbreaks involving multiple convective modes simultaneously. The K=2 axis is outbreak scale/severity, not convective mode. Recovering convective-mode separation requires environmental features (CAPE, 0–6 km shear, helicity, LCL) not present in SPC + NCEI.

---

## 4. Continuum Analysis

### 4.1 PCA Visualization

PCA(2) explains 19.2% (PC1) + 10.4% (PC2) = 29.6% of variance — the data is a heavy-tailed continuum, not a discrete bimodal gap. The K=2 partition slices at the severity threshold rather than revealing separate modes.

### 4.2 Regression on log1p(Fatalities)

| Model                                      | 5-fold CV R² | Std    |
| ------------------------------------------ | ----------- | ------ |
| Ridge (α=1.0)                              | 0.455       | ±0.059 |
| **Gradient Boosting** (200 trees, depth 3) | **0.496**   | ±0.047 |

GBR explains roughly half the variance in `log1p(fatality_total)` from outbreak-day SPC + NCEI features alone — comparable to the OII Spearman ρ = 0.581 and properly out-of-sample.

**Top GBR features (by importance):** `ef2plus_count` (0.27), `max_mag` (0.22), `violent_count` (0.09), `sig_fraction` (0.04), `mean_log_len` (0.04), `region_lat` (0.04), `track_wid_mean` (0.03), `track_len_cv` (0.03), `region_lon` (0.02), `hull_log_aspect` (0.02). The geographic features rank high with low marginal Spearman correlation, indicating the model uses geography non-linearly (Tornado Alley + Dixie Alley both elevate risk).

---

## 5. Hypothesis Test Results

| Hypothesis                                     | Result    | p-value     | Interpretation                                                     |
| ---------------------------------------------- | --------- | ----------- | ------------------------------------------------------------------ |
| **P1** (PDS bearing coherence)                 | **PASS**  | < 0.05      | High-fatality days show more coherent track bearings               |
| **P2** (EF4+ separation by cluster)            | **PASS**  | 5.93e-14    | C1 has 10.7× higher EF4+ rate, Cohen's h = 0.69 (large effect)    |
| **P3** (Apr 27 bearing coherence)              | **PASS**  | < 0.05      | Apr 27 bearings are coherent, Rayleigh test significant            |

Holm–Bonferroni corrected p-values confirm all three tests remain significant at α = 0.05.

---

## 6. Real-Time Prediction Performance

### 6.1 AUROC by Feature Set and Time

| Feature Set           | T+1h      | T+2h      | T+3h      | T+4h      |
| --------------------- | --------- | --------- | --------- | --------- |
| SPC-like              | **0.686** | 0.746     | **0.800** | 0.838     |
| Tabular               | 0.672     | **0.753** | 0.790     | 0.828     |
| Count-only            | 0.503     | 0.532     | 0.566     | 0.566     |

**Key findings:**

1. **Count-only is useless** (AUROC ~0.5 = random) at early times. You need at least SPC-like features (magnitude, nocturnal, states) to get meaningful prediction.
2. **SPC-like features achieve AUROC = 0.746 at T+2h** — sufficient for early EF4+ screening.
3. **T+4h is the sweet spot** for all feature sets (AUROC 0.80–0.84), but even this is only moderate predictive power.

### 6.2 OII (Overall Impact Index)

| Index               | Spearman rho vs. fatalities |
| ------------------- | --------------------------- |
| OII (hand-designed) | **0.581**                   |

The OII gives a single-number rank correlate comparable to GBR R² = 0.50. OII and GBR are complementary: OII provides an interpretable rank, GBR provides calibrated regression.

---

## 7. Anomaly Detection

Isolation Forest + LOF fused ranking on the same 1996–2024 outbreak-day matrix used for clustering. All four named historic outbreaks within the analysis window rank as strong anomalies and all land in cluster C1 (severity-major). Exact percentiles are produced inline by the notebook; all four print at top ≤ 4% in current runs:

| Outbreak              | Cluster |
| --------------------- | ------- |
| OKC May 3 1999        | C1      |
| April 27 2011         | C1      |
| Moore May 20 2013     | C1      |
| Quad-State 2021       | C1      |

Pre-1996 outbreaks (e.g., 1974 Super Outbreak, 1965 Palm Sunday) are excluded by design — the NCEI-derived features used by the detector are unavailable before 1996, and median-imputing them adds noise without restoring genuine signal.

---

## 8. Combined Verdict

The data supports two complementary views of outbreak severity:

|                   | Clustering (K=2 K-Means, TABULAR)       | Continuum regression        |
| ----------------- | --------------------------------------- | --------------------------- |
| Output            | Discrete C0 / C1 label                  | Continuous predicted log-fatality |
| Quality           | Silhouette 0.32, ARI 0.87               | GBR R² 0.50 ± 0.05         |
| Question answered | "Is this a major-class day?"            | "How bad will this day be?" |
| Operational use   | Triage flag (top 20% catches 82% of fatalities) | Calibrated risk regression |

The data isn't a pure continuum (K=2 clustering is stable and meaningful) and isn't textbook well-separated either (silhouette 0.32 is moderate). It's both — a stable severity binary at K=2 on a heavy-tailed continuous severity distribution.

---

## 9. Limitations

1. The K=2 axis is outbreak severity/scale, NOT convective mode. Both QLCS and supercell narrative flags are elevated in C1.
2. Pre-1996 data is excluded from primary clustering (NCEI coverage begins 1996).
3. Geographic features rank high in GBR non-linear importance but have low marginal Spearman correlation, indicating complex regional risk patterns.
4. Fatality prediction R² ~0.50 means half the variance is unexplained — likely due to exposure factors (population density, mobile-home rates) not present in SPC/NCEI data.

---

## 10. Future Work

1. **Add ERA5 reanalysis environmental fields** (CAPE, 0–6 km shear, helicity, LCL) to attempt convective-mode separation.
2. **Add Census ACS county-level mobile-home and population density** to separate physical severity from exposure-driven fatality variance.
3. **Period-match ACS data** to outbreak era (1990/2000/2010/2020 decennials). This is important as capabilities have changed over time on our end.
4. **Compare K=2 to a simple severity-threshold rule** (≥25 tornadoes AND ≥4 states AND ≥12h duration) for operational utility assessment, also a good place to try a Fuzzy Inference System.

---

## 11. Figures

| Figure             | Path                            | Description                                   |
| ------------------ | ------------------------------- | --------------------------------------------- |
| Continuum PCA      | `figures/fig_continuum_pca.png` | PCA(2) scatter colored by EF4+ and fatalities |

Inline tables and diagnostics for clustering, hypothesis tests, anomaly ranks, and prediction AUROC are produced directly by the notebook (no separate figure files).
