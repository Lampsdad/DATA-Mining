# Cascading Supercell Families: Tornado Outbreak Style Classification

**CSCE 676 — Graduate Data Mining, Texas A&M University (Spring 2026)**

This project analyzes **74 years of NOAA tornado data** (1950–2024) merged with **NCEI Storm Events records** (1996–2024) to classify tornado outbreaks into meteorologically interpretable styles. We build multi-scale directed cascade graphs, apply graph-theoretic features (PageRank, community structure), and combine them with track-geometry, temporal, and episode-count features — then cluster outbreaks using K-Means, HAC, GMM, and fuzzy c-means to discover outbreak archetypes and test hypotheses about what makes an outbreak devastating.

---

## 👉 Start Here

**The main deliverable is `[main_notebook.ipynb](main_notebook.ipynb)`.**
It walks through the full story: motivation, data, methods, results, and conclusions.

---

## 📖 Research Questions

1. **RQ1 — Outbreak Styles:** Can we identify distinct, meteorologically interpretable outbreak styles (e.g., fast-propagating supercell families vs. slow-moving QLCS lines) from historical tornado data alone?
2. **RQ2 — Extreme Tornado Separation:** Do certain outbreak archetypes produce significantly more EF4+ tornadoes than others? (P2)
3. **RQ3 — Propagation Coherence:** Do major outbreaks exhibit statistically significant directional coherence in their tornado cascade bearings? (P3 — tested on Apr 27, 2011)
4. **RQ4 — Real-Time Prediction:** Can we predict whether an evolving outbreak will produce an EF4+ tornado from early observations, and do graph-derived features add predictive value beyond basic tabular features?

---

## 🎥 Project Video

> **Add your video link here:** [▶ Watch the project video](https://youtu.be/Uxlq85QtMw4)

---

## 📊 Data

### Datasets

1. **NOAA SPC Tornado Database (1950–2024)** — 70,456 event-level tornado records with path geometry (start/end lat/lon, path length, width, magnitude).
  Source: `https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv`
2. **NOAA NCEI Storm Events Database (1996–2024)** — 41,140 storm episode records with EPISODE_ID grouping and narrative descriptions.
  Source: `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/`

Both datasets are downloaded on first run and cached as Apache Parquet files in `cache/`. The cache directory is gitignored; the first run will re-download and cache the data.

### Preprocessing

- SPC data: filtered to valid magnitudes (0–5), CONUS states, and valid coordinates. Time strings normalized to fractional hours.
- Outbreak days defined as days with ≥6 tornado events.
- NCEI data: filtered to Tornado event type, CONUS states, and outbreak days. Episode-level statistics computed (episode count, max size, narrative flags for QLCS vs. supercell mode).
- Track-geometry features computed from tornado start/end coordinates: hull area, hull aspect ratio (PCA), bearing consistency, geographic centroid.
- Multi-scale directed cascade graphs built using BallTree neighbor search: mesoscale (600 km / 6 h), synoptic (1,400 km / 24 h), and priming (cross-day).

---

## 🏃 How to Reproduce

### Requirements

- **Python 3.13** (Colab default)
- Install dependencies: `pip install -r requirements.txt`

### Steps

1. Clone or download this repository.
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
3. Open `main_notebook.ipynb` in Jupyter and run all cells in order.
  ```bash
   jupyter notebook main_notebook.ipynb
  ```
4. The notebook downloads SPC data from NOAA on first run and caches it as `cache/spc_cache.parquet`. NCEI data is similarly cached. Subsequent runs use the cache.

### Running in Google Colab

This project was built and tested in **Google Colab** (Python 3.13). The fastest way to reproduce the analysis:

1. Upload `requirements.txt` to a Colab notebook and run `!pip install -r requirements.txt`
2. Upload or clone this repository into the Colab environment
3. Open `main_notebook.ipynb` and run all cells in order with `Runtime → Run all`
4. Data is downloaded and cached automatically on first run

A one-click link can be added once the repo is public on GitHub (use [Colab's notebook viewer](https://colab.research.google.com/github/) with your repo URL).

---

## 🔑 Key Dependencies


| Package        | Version | Purpose                             |
| -------------- | ------- | ----------------------------------- |
| Python         | 3.13    | Runtime                             |
| pandas         | 3.0.2   | Data manipulation                   |
| numpy          | 2.4.4   | Numerical computing                 |
| scikit-learn   | 1.8.0   | Clustering, classification, metrics |
| scipy          | 1.17.1  | Statistical tests, spatial queries  |
| networkx       | 3.6.1   | Graph construction and PageRank     |
| matplotlib     | 3.10.8  | Visualization                       |
| seaborn        | 0.13.2  | Visualization                       |
| scikit-fuzzy   | 0.5.0   | Fuzzy c-means clustering            |
| python-igraph  | 1.0.0   | Leiden community detection          |
| leidenalg      | 0.11.0  | Leiden algorithm (Python binding)   |
| hdbscan        | 0.8.47  | Density-based clustering            |
| python-louvain | 0.16    | Louvain community detection         |


Full list in `[requirements.txt](requirements.txt)`.

---

## 📁 Repository Structure

```
DATA-Mining/
├── README.md                      # You are here
├── main_notebook.ipynb            # 🎯 Final deliverable — start here
├── requirements.txt               # Pinned Python dependencies
│
├── checkpoints/                   # Checkpoint notebooks
│   ├── checkpoint_1.ipynb         # Early exploratory analysis
│   └── checkpoint_2.ipynb         # Intermediate results and method selection
│
├── scripts/                       # Standalone processing script
│   └── final_project_tornado_outbreaks.py  # Jupytext-exported notebook
│
├── cache/                         # Cached datasets (gitignored)
│   ├── spc_cache.parquet          # NOAA SPC 1950-2024 (~1.8 MB)
│   ├── ncei_cache.parquet         # NOAA NCEI 1996-2024 (~14 MB)
│   └── sens_cache.parquet         # Sensitivity analysis cache
│
├── figures/                       # Figures produced by main_notebook.ipynb
│   ├── fig_v6_archetype_radar.png   # Radar chart of archetype profiles
│   ├── fig_v6_kmeans_eval.png       # Clustering evaluation metrics
│   └── fig_v6_style_map.png         # Geographic map of outbreak styles
│
├── scaffolding/                   # Early EDA scripts (not part of deliverable)
│   ├── checkpoint1_tornado.py
│   └── checkpoint2_tornado.py
│
└── .gitignore                     # Excludes cache/, figures/, archive/, etc.
```

---

## 🔬 Results Summary

**Headline (v7):** A clean-window re-analysis (1996–2024 only, no median imputation, cascade graph features dropped) achieves a stable severity partition. K-Means K=2 on a 9-feature SPC + NCEI tabular matrix produces silhouette **0.320** with bootstrap ARI **0.865**. The minority cluster C1 (n=348, 20% of days) captures all named historic outbreaks (1999 OKC, 2011 Apr 27, 2013 Moore, 2021 Quad-State), 23.6% of EF4+ days, and **82% of all 1996–2024 fatalities**. The v6 "no discrete structure" verdict was driven by imputation noise + sparse cascade-graph features rather than true data structure.

**Continuum regression on the same features:** 5-fold CV gradient boosting predicts `log1p(fatality_total)` with R² = **0.496 ± 0.047**. Top predictors: `ef2plus_count`, `max_mag`, `violent_count`, plus geography (`region_lat`, `region_lon`) entering non-linearly.

**Other results (v6, unchanged):**

- **P2 passed**: EF4+ rate 3.4× higher in the v6 archetype (11.8% vs 3.5%), p = 5.93×10⁻¹⁴
- **P3 passed**: Apr 27, 2011 bearing coherence (p = 0.021, mean bearing = 87°)
- **Real-time prediction**: SPC-like features achieve AUROC = 0.746 at T+2h for EF4+ outcomes; cascade graph features add no significant lift (Wilcoxon p > 0.05)
- **OII (Impact Index)**: Spearman ρ = 0.581 with fatalities
- **Anomaly detection**: All five named historic outbreaks rank in the top 3.3%

**Important caveat:** The K=2 axis is **outbreak scale/severity**, not convective mode (HP supercell vs. classic supercell vs. QLCS). C1 has *both* QLCS-narrative (54%) and supercell-narrative (79%) elevated — major outbreaks involve multiple convective modes simultaneously. Convective-mode separation requires environmental data (CAPE, 0–6 km shear, helicity) not in SPC or NCEI.

Full analysis in [Results.md](Results.md). v7 diagnostic script: [testing_rewrite.py](testing_rewrite.py). v7 outputs: [testing_outputs/](testing_outputs/).