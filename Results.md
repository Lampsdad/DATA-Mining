# Results & Interpretation — Cascading Supercell Families (v6 + v7)

> Run date: April 25, 2026 (v6) / April 27, 2026 (v7 follow-up) | v6 feature matrix: 36 features across tabular, cascade-graph, track-geometry, and NCEI-episode dimensions; v7 feature matrix: 9 SPC + NCEI tabular features on the clean 1996–2024 window

> **Headline update (v7):** The v6 conclusion that the outbreak space has "no discrete structure" (silhouette < 0.12) was driven by feature-engineering noise more than data structure. After restricting to 1996–2024 (no median-imputation) and dropping cascade graph features, K-Means K=2 on a 9-feature SPC + NCEI tabular matrix achieves silhouette **0.320** with bootstrap ARI **0.865** — a stable severity partition that captures **82% of all 1996–2024 fatalities in 20% of the days**. See §11 for the v7 follow-up. The v6 results below are preserved unchanged.

---

## 1. Data Summary


| Metric                        | Value                              |
| ----------------------------- | ---------------------------------- |
| SPC clean events              | 70,456                             |
| Outbreak days (>=6 events)    | 3,589                              |
| Events in outbreak days       | 50,061                             |
| NCEI episodes available       | 41,140 events                      |
| NCEI coverage                 | 48% (1,740 of 3,589 outbreak days) |
| EF4+ outbreak days            | 343 (9.6%)                         |
| Days with QLCS narrative      | 32.0%                              |
| Days with supercell narrative | 53.5%                              |


**Key observation:** NCEI coverage is only 48% because the Storm Events database only exists from 1996 onward, yet the outbreak-day definition uses SPC data from 1950. The 52% of days without NCEI data receive median-imputed episode features, which dilutes the clustering signal for pre-1996 events.

---

## 2. Graph Construction


| Scale               | Radius   | Time window | Edges |
| ------------------- | -------- | ----------- | ----- |
| Mesoscale           | 600 km   | 6 hours     | 1,999 |
| Synoptic            | 1,400 km | 24 hours    | 8,958 |
| Priming (cross-day) | 1,000 km | 24 hours    | 470   |


**Key observation:** The mesoscale graph has only 1,999 edges across 50,061 events. This sparsity means many outbreak days have few or no graph edges, which limits the discriminative power of graph-derived features (PageRank, component counts) in clustering. The graph features contribute more as noise than signal in the clustering space.

---

## 3. Clustering Results

### 3.1 Model Selection


| K   | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| --- | ---------- | -------------- | ----------------- |
| 2   | **0.1144** | 2.283          | **433.5**         |
| 3   | 0.1109     | 2.604          | 391.4             |
| 4   | 0.1109     | 2.537          | 338.6             |
| 5   | 0.0941     | 2.719          | 288.4             |
| 6   | 0.0923     | 2.297          | 265.8             |


### 3.2 CRITICAL FINDING: Clustering is weak

- **Silhouette scores are all below 0.12**, even at the optimal K=2. This is extremely low and indicates the clusters overlap heavily in feature space.
- K=2 is selected because it has the highest silhouette score, but 0.1144 means the clustering has **very poor separation**.
- Higher K values (3-5) that would be meteorologically more interesting have *worse* silhouette scores, suggesting the data doesn't naturally support fine-grained outbreak styles with the current feature set.

### 3.3 Cluster Descriptions


|                         | C0 (n=977)                                           | C1 (n=2,612)                 |
| ----------------------- | ---------------------------------------------------- | ---------------------------- |
| **Label**               | fast-propagating / elongated-hull / variable-bearing | slow-centroid / multi-system |
| **Cascade velocity**    | 120.7 km/h                                           | 92.7 km/h                    |
| **EF4+ rate**           | 3.5%                                                 | **11.8%**                    |
| **QLCS narrative**      | 0.1%                                                 | 21.3%                        |
| **Supercell narrative** | **99.0%**                                            | 69.4%                        |
| **Mean episodes**       | 6.0                                                  | 7.0                          |
| **Hull elongation**     | 2.77                                                 | 2.66                         |
| **Bearing consistency** | 0.01                                                 | **0.85**                     |
| **Latitude**            | 37.3                                                 | 37.3                         |


### 3.4 Interpretation of the Binary Split

The K=2 clustering essentially divides outbreaks along **two** dimensions:

1. **Cascade velocity** (fast vs. slow) — C0 moves faster (121 vs 93 km/h)
2. **Bearing consistency** (variable vs. coherent) — C0 has near-zero bearing consistency (R=0.01), C1 is highly consistent (R=0.85)

**Problematic finding:** C0 is labeled "fast-propagating with variable-bearing" yet has 99% supercell narrative and 0% QLCS. Meteorologically, supercell families typically have *coherent* movement (high bearing consistency), not variable bearings. This suggests the clustering is picking up an artifact — possibly fast-moving single-cell days that happen to be scattered, rather than a genuine "fast supercell family" style.

**Surprising finding:** C1 (slow, multi-system) has **3.4x higher EF4+ rate** (11.8% vs 3.5%) despite being slower. This contradicts the hypothesis that fast propagation correlates with extreme tornadoes. The most violent outbreaks (1974, Apr 27 2011, OKC 1999) all cluster in C1.

### 3.5 Fuzzy C-Means Continuum Test

- FCM Partition Coefficient: **0.5000** (theoretical floor for K=2 is 0.5000)
- Permutation test z-score: **0.00**

**Interpretation:** The FCM results are exactly at the floor, meaning every event has membership ~0.5 in both clusters. This is the worst possible outcome — it indicates the data is a continuum with no soft structure at all. The K=2 split is arbitrary, not meteorologically grounded.

---

## 4. Hypothesis Test Results


| Hypothesis                                     | Result        | p-value  | Interpretation                                                                                   |
| ---------------------------------------------- | ------------- | -------- | ------------------------------------------------------------------------------------------------ |
| **P1a** (pre-registered: fast tornadoes = PDS) | **FAIL**      | 1.00e+00 | Fast propagation does NOT correlate with high fatalities                                         |
| **P1b** (exploratory: slow tornadoes = PDS)    | **SUPPORTED** | 1.21e-21 | Slow, persistent outbreaks produce more fatalities (as expected — longer duration = more deaths) |
| **P2** (archetype EF4+ separation)             | **PASS**      | 5.93e-14 | C1 has 3.4x higher EF4+ rate, Cohen's h=0.33 (medium effect)                                     |
| **P3** (Apr 27 bearing coherence)              | **PASS**      | 2.06e-02 | Apr 27 bearings are coherent (87deg, nearly due east), p=0.02                                    |


### 4.1 OII (Overall Impact Index) Correlation


| Index               | Spearman rho vs. fatalities |
| ------------------- | --------------------------- |
| OII_retro           | **0.581**                   |
| OII_learned (Ridge) | 0.529                       |
| EF2+ count only     | 0.481                       |
| Event count only    | 0.293                       |


**Interpretation:** The retroactive OII (0.581) outperforms both the learned Ridge version (0.529) and simple counts. This suggests the hand-designed formula captures useful signal that the Ridge model couldn't learn from the 18 features. Event count alone (0.293) is a weak predictor — it's the *combination* of severity, fatalities, and sig-fraction that matters.

---

## 5. Real-Time Prediction Performance

### 5.1 AUROC by Feature Set and Time


| Feature Set           | T+1h      | T+2h      | T+3h      | T+4h      |
| --------------------- | --------- | --------- | --------- | --------- |
| **Cascade-augmented** | 0.648     | **0.747** | 0.804     | **0.825** |
| Cascade-full          | 0.644     | 0.754     | 0.796     | 0.827     |
| SPC-like              | **0.686** | 0.746     | **0.800** | 0.838     |
| Tabular               | 0.672     | **0.753** | 0.790     | 0.828     |
| Count-only            | 0.503     | 0.532     | 0.566     | 0.566     |


### 5.2 Key Findings

1. **Count-only is useless** (AUROC ~0.5 = random) at early times. You need at least SPC-like features (magnitude, nocturnal, states) to get meaningful prediction.
2. **SPC-like matches or beats cascade features** at all time points. The Wilcoxon test confirms: cascade-full adds **no statistically significant improvement** over SPC-like at any snapshot hour (all p > 0.05).
3. **Cascade-full underperforms at T+1h** (0.644 vs SPC-like 0.686). The extra cascade features (PageRank, partial PageRank, component count) are noisy when based on few early events.
4. **T+4h is the sweet spot** for all feature sets (AUROC 0.83-0.84), but even this is only moderate predictive power.
5. **The cascade features are not worth the complexity** for real-time prediction. A simple SPC-like feature set (count, max EF, nocturnal fraction, state count, month) achieves nearly identical performance.

---

## 6. Anomaly Detection

All five named historic outbreaks rank as **strong anomalies** (top 3.3% or more anomalous):


| Outbreak            | Fused Anomaly Rank | Archetype                    |
| ------------------- | ------------------ | ---------------------------- |
| 1974 Super Outbreak | **Top 0.06%**      | slow-centroid / multi-system |
| April 27 2011       | **Top 0.04%**      | slow-centroid / multi-system |
| OKC May 3 1999      | **Top 0.13%**      | slow-centroid / multi-system |
| 1965 Palm Sunday    | Top 1.64%          | slow-centroid / multi-system |
| Quad-State 2021     | Top 3.30%          | slow-centroid / multi-system |


**Interpretation:** The anomaly detection correctly identifies all major historic outbreaks, and they all share the C1 archetype. This is both a strength and a weakness — the fact that ALL famous outbreaks cluster in one group suggests C1 is the "default" category for anything unusual, while C0 absorbs the mundane fast-moving scattered events.

---

## 7. Critical Problems Identified

### P1: Clustering quality is fundamentally weak

- Silhouette < 0.12 across all K values
- FCM at the theoretical floor (PC = 0.5000)
- The data does not support 3+ discrete archetypes with current features
- **Recommendation:** Abandon clustering as the primary analysis approach, or radically rethink the feature engineering

### P2: NCEI features have limited discriminative power

- NCEI coverage is only 48% (1996-2024)
- Median-imputed values for 52% of days add noise
- C0 and C1 have nearly identical NCEI episode counts (6.0 vs 7.0)
- **Recommendation:** Either restrict analysis to 1996-2024 for consistent NCEI coverage, or remove NCEI features and rely on SPC-only features

### P3: Cascade graph features are not useful

- Graph features don't improve real-time prediction (Wilcoxon p > 0.05 at all times)
- Graph is sparse (1,999 edges across 50,061 events = ~4% event connectivity)
- Partial PageRank at T+1h actually *hurts* prediction
- **Recommendation:** Either increase edge density (larger R, longer T) or remove graph features from clustering and prediction

### P4: The K=2 split is meteorologically questionable

- C0 ("fast supercells with variable bearing") is an oxymoron — supercells don't have variable bearings
- C1 is a "everything else" bucket containing ALL famous outbreaks
- The archetype labels are descriptive of the split, not predictive of storm mode
- **Recommendation:** Consider a different clustering strategy (e.g., hierarchical with meaningful distance metric, or density-based DBSCAN)

### P5: P1 hypothesis is inverted

- The pre-registered hypothesis (fast = PDS) failed decisively (p=1.0)
- The exploratory test (slow = PDS) is strongly supported (p=1.21e-21)
- This is intuitive (slow outbreaks last longer = more deaths) but means the original hypothesis was wrong
- **Recommendation:** Revise P1 to test "persistence/duration" rather than "speed" as the predictor of impact

---

## 8. Planned Improvements

### High Priority

1. **Restrict to 1996-2024 for clustering**
  - Remove median-imputation noise from pre-1996 days
  - Get full NCEI episode information for all clustering events
  - Compare clustering quality with and without imputed data
2. **Try different feature subsets for clustering**
  - Test clustering with only track-geometry features (hull aspect, bearing, orientation)
  - Test clustering with only tabular features (event count, fatalities, severity)
  - Test clustering with only geographic features (lat, lon)
  - This will reveal which dimensions actually produce structure
3. **Revise the P1 hypothesis**
  - Test "outbreak duration" or "temporal spread" as predictor of fatalities
  - Test "number of EF2+ tornadoes" as a more direct severity proxy
  - Consider testing "nocturnal fraction" as a predictor (nighttime outbreaks are more dangerous)

### Medium Priority

1. **Improve graph construction**
  - Increase R to 800-1000km for mesoscale to increase edge density
  - Or switch to undirected graph to double edge count
  - Or use k-nearest neighbors instead of radius-based approach
2. **Try alternative clustering algorithms**
  - DBSCAN or HDBSCAN for density-based clustering (doesn't require specifying K)
  - Spectral clustering with a custom distance metric
  - Model-based clustering with BIC for model selection
3. **Add seasonal stratification**
  - Cluster March-May and June-August separately (different meteorological regimes)
  - The current clustering mixes peak-season supercells with off-season QLCS events

### Lower Priority

1. **Add environmental/sounding features**
  - CAPE, shear, LCL heights from reanalysis data (ERA5)
  - This would enable HP vs. classic supercell distinction
2. **Improve real-time prediction**
  - Add interaction features (e.g., count * nocturnal_fraction)
  - Try XGBoost or LightGBM instead of GradientBoosting
  - Add calibration (isotonic regression) for better probability estimates
3. **Quantify the "famous outbreak" signal**
  - What fraction of C1 events are "famous" vs. routine?
  - Are there C1 events that are anomalous but not in the top 5?
  - This would validate whether C1 is a meaningful category

---

## 9. Figures Generated


| Figure             | Path                                 | Description                                |
| ------------------ | ------------------------------------ | ------------------------------------------ |
| K-Means Evaluation | `figures/fig_v6_kmeans_eval.png`     | Silhouette, DB, CH scores for K=2-8        |
| Archetype Radar    | `figures/fig_v6_archetype_radar.png` | Normalized centroid profiles per cluster   |
| Style Map          | `figures/fig_v6_style_map.png`       | Geographic distribution of outbreak styles |


---

## 10. v6 Bottom Line

The v6 analysis successfully merges SPC and NCEI data and builds a rich feature matrix, but **the clustering does not produce meteorologically interpretable outbreak styles**. The K=2 split is weak (silhouette < 0.12), fuzzy membership is uniform (PC = floor), and the cascade graph features add no predictive value. The most impactful improvement would be to either: (a) restrict the analysis to 1996-2024 for consistent NCEI coverage, or (b) abandon clustering in favor of a regression or anomaly-detection framework where the existing features show more promise.

---

## 11. v7 Follow-Up — Clean-Window Re-Analysis (1996–2024)

The v6 §8 high-priority recommendations were implemented in `testing_rewrite.py`:

- 1996–2024 restriction (eliminates median-imputed NCEI flags)
- Cascade graph features dropped (Wilcoxon test in §5 showed no AUROC lift)
- Wide clustering bake-off (100 method × subset × hyperparameter combos)
- Non-degenerate filter (every cluster ≥ 30 points, noise ≤ 50%)
- Bootstrap ARI stability (20 resampled 80% subsets)

### 11.1 Merge Validation (Event-Level)

The v6 day-level join was never explicitly cross-validated. v7 added event-level checks:


| Check                                        | Result                        | Verdict                                 |
| -------------------------------------------- | ----------------------------- | --------------------------------------- |
| Per-day count Spearman ρ (1,740 shared days) | **0.954**                     | PASS                                    |
| Median(SPC count / NCEI count)               | 0.92                          | NCEI ~8% more events (per-segment rows) |
| Hard invariant `max_episode_size ≤ spc_n`    | 53 days violate (3.0%)        | partial — multi-county segmenting       |
| Event-level NN match (50 km / 1 h tolerance) | 76.7% (24,933 / 32,508)       | partial                                 |
| Match precision among matches                | 1.4 km mean dist, 0.20 h mean | Δt                                      |
| State agreement on matched pairs             | **99.7%**                     | PASS                                    |
| Timezone audit (Δ NCEI − SPC, hours)         | median 0.00 h, p90 1.0 h      | PASS                                    |
| Internal NCEI consistency                    | 0 mismatches                  | PASS                                    |


The merge is well-aligned in time and space; the 23% unmatched fraction reflects NCEI's per-segment-row convention for multi-county tornadoes, not a merge bug.

### 11.2 Clustering Bake-Off

Of 100 (method × subset × hyperparam) combinations, 59 passed the non-degenerate filter. Top non-degenerate results:


| Method      | Subset      | K     | Sizes           | Silhouette | Bootstrap ARI |
| ----------- | ----------- | ----- | --------------- | ---------- | ------------- |
| GMM         | GEOMETRY    | 2     | 1,675 / 65      | 0.375      | 0.228         |
| **K-Means** | **TABULAR** | **2** | **1,392 / 348** | **0.320**  | **0.865**     |
| FCM         | TABULAR     | 2     | 1,037 / 703     | 0.218      | —             |
| GMM         | TABULAR     | 2     | 1,125 / 615     | 0.204      | 0.618         |
| Spectral    | TABULAR     | 2     | 527 / 473       | 0.188      | —             |
| K-Means     | GEOMETRY    | 3     | 771 / 587 / 382 | 0.174      | 0.802         |


The headline is **K-Means K=2 on TABULAR features**: silhouette 0.320 (≈ 2.8× the v6 result of 0.114) with bootstrap ARI 0.865 — partition recovers on 86.5% of resampled subsets, indicating stable structure.

### 11.3 Cluster Profile


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

**Important caveat:** C1 is **not** a "QLCS vs. supercell" archetype. Both narrative flag rates are *elevated* in C1 (79% supercell + 54% QLCS), consistent with major outbreaks involving multiple convective modes simultaneously. The K=2 axis is **outbreak scale/severity**, not convective mode.

### 11.4 Continuum Regression

PCA(2) explains 19.2% (PC1) + 10.4% (PC2) = 29.6% of variance — heavy-tailed continuum, no discrete bimodal gap. Treating the same feature space continuously:


| Model                                      | 5-fold CV R² (log fatality) | Std    |
| ------------------------------------------ | --------------------------- | ------ |
| Ridge (α=1.0)                              | 0.455                       | ±0.059 |
| **Gradient Boosting** (200 trees, depth 3) | **0.496**                   | ±0.047 |


GBR explains roughly half the variance in `log1p(fatality_total)` from outbreak-day SPC + NCEI features alone — comparable to the v6 OII Spearman ρ = 0.581 and properly out-of-sample.

Top GBR features (by importance): `ef2plus_count` (0.27), `max_mag` (0.22), `violent_count` (0.09), `sig_fraction` (0.04), `mean_log_len` (0.04), `region_lat` (0.04), `track_wid_mean` (0.03), `track_len_cv` (0.03), `region_lon` (0.02), `hull_log_aspect` (0.02). The geographic features rank high with low marginal correlation, indicating the model uses geography non-linearly (Tornado Alley + Dixie Alley both elevate risk).

### 11.5 v7 Bottom Line

The v6 "no structure" verdict was overstated. After eliminating imputation noise and noisy graph features, the data supports **two complementary views**:


|                   | Clustering (K=2 K-Means, TABULAR)               | Continuum regression              |
| ----------------- | ----------------------------------------------- | --------------------------------- |
| Output            | Discrete C0 / C1 label                          | Continuous predicted log-fatality |
| Quality           | Silhouette 0.32, ARI 0.87                       | GBR R² 0.50 ± 0.05                |
| Question answered | "Is this a major-class day?"                    | "How bad will this day be?"       |
| Operational use   | Triage flag (top 20% catches 82% of fatalities) | Calibrated risk regression        |


The data isn't a pure continuum (K=2 clustering is stable and meaningful) and isn't textbook well-separated either (silhouette 0.32 is moderate). It's both — a stable severity binary at K=2 on a heavy-tailed continuous severity distribution. Future work to recover convective-mode separation (the original RQ1) requires environmental features (CAPE, 0–6 km shear, helicity, LCL) not present in SPC + NCEI.

### 11.6 v7 Artifacts

- `testing_rewrite.py` — single-file diagnostic script
- `testing_outputs/merge_validation_report.txt` — full PASS/FAIL detail
- `testing_outputs/clustering_bakeoff.csv` — 100 (method × subset × hyperparam) rows
- `testing_outputs/continuum_pca.png` — heavy-tailed continuum visualization
- `testing_outputs/feature_importances.csv` — GBR importances + Spearman correlations

