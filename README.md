# Cascading Supercell Families: Tornado Outbreak Severity Classification

**CSCE 676 — Graduate Data Mining, Texas A&M University (Spring 2026)**

This project analyzes **74 years of NOAA tornado data** (1950–2024) merged with **NCEI Storm Events records** (1996–2024) to classify tornado outbreaks into severity classes using clustering, regression analysis, and hypothesis testing. We build event-level features from SPC path geometry and NCEI episode statistics — then cluster outbreaks using K-Means, HAC, GMM, HDBSCAN, and fuzzy c-means to discover outbreak severity patterns and test hypotheses about what makes an outbreak devastating.

---

## Start Here

**The main deliverable is [main_notebook.ipynb](main_notebook.ipynb).**
It walks through the full story: motivation, data, methods, results, and conclusions.

---

## Research Questions

1. **RQ1 Severity Classification:** Can we identify distinct, meteorologically interpretable outbreak severity classes from historical tornado data alone?
2. **RQ2 Extreme Tornado Separation:** Do certain outbreak archetypes produce significantly more EF4+ tornadoes than others?
3. **RQ3 Propagation Coherence:** Do major outbreaks exhibit statistically significant directional coherence in their tornado trajectories?
4. **RQ4 Real-Time Prediction:** Can we predict whether an evolving outbreak will produce an EF4+ tornado from early observations?

---

## Project Video

> [Watch the project video](https://youtu.be/Uxlq85QtMw4)

---

## Data

### Datasets

1. **NOAA SPC Tornado Database (1950–2024):** contains 70,456 event-level tornado records with path geometry (start/end lat/lon, path length, width, magnitude).
  Source: `https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv`
2. **NOAA NCEI Storm Events Database (1996–2024):** contains 41,140 storm episode records with EPISODE_ID grouping and narrative descriptions.
  Source: `https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/`

Both datasets are downloaded on first run and cached in `cache/`. For space saving, the cache directory is gitignored and the first run will re-download and cache the data.

### Preprocessing Steps

- SPC data: filtered to valid magnitudes (0–5), CONUS states, and valid coordinates. Time strings normalized to fractional hours.
- Outbreak days defined as days with ≥6 tornado events.
- NCEI data: filtered to Tornado event type, CONUS states, and outbreak days. Episode-level statistics computed (episode count, max size, narrative flags for QLCS vs. supercell mode).
- Track-geometry features computed from tornado start/end coordinates: hull area, hull aspect ratio (PCA), bearing consistency, geographic centroid.
- Merge validation: event-level nearest-neighbor matching (50 km / 1 h tolerance) confirms 76.7% match rate, 99.7% state agreement.

---

## How to Reproduce Results

### Requirements

- **Python 3.13** (Colab default)
- Install dependencies: `pip install -r requirements.txt`

### Local Steps

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

---

## Dependencies


| Package        | Version | Purpose                             |
| -------------- | ------- | ----------------------------------- |
| Python         | 3.13    | Runtime                             |
| pandas         | 3.0.2   | Data manipulation                   |
| numpy          | 2.4.4   | Numerical computing                 |
| scikit-learn   | 1.8.0   | Clustering, classification, metrics |
| scipy          | 1.17.1  | Statistical tests, spatial queries  |
| matplotlib     | 3.10.8  | Visualization                       |
| seaborn        | 0.13.2  | Visualization                       |
| scikit-fuzzy   | 0.5.0   | Fuzzy c-means clustering            |
| hdbscan        | 0.8.47  | Density-based clustering            |
| python-louvain | 0.16    | Louvain community detection          |
| python-igraph  | 1.0.0   | Graph analysis                       |
| leidenalg      | 0.11.0  | Leiden community detection           |
| networkx       | 3.6.1   | Graph analysis                       |

Full list in [requirements.txt](requirements.txt).

---

## Repository Structure

```
DATA-Mining/
├── README.md                      
├── main_notebook.ipynb            # Final deliverable, easiest starting location
├── Results.md                     # Detailed results and interpretation
├── requirements.txt               # Pinned Python dependencies
│
├── checkpoints/                   # Checkpoint notebooks
│   ├── checkpoint_1.ipynb         # Early exploratory analysis
│   └── checkpoint_2.ipynb         # Intermediate results and method selection
│
├── cache/                         # Cached datasets (gitignored)
│   ├── spc_cache.parquet          # NOAA SPC 1950-2024 (~1.8 MB)
│   └── ncei_cache.parquet         # NOAA NCEI 1996-2024 (~14 MB)
│
├── figures/                       # Figures produced by main_notebook.ipynb
│   └── fig_continuum_pca.png      # PCA continuum visualization (generated on run)
│
└── .gitignore                     # Excludes cache/, figures/, etc.
```

---

## Results Summary

**Headline finding:** A clean-window analysis (1996–2024 only, no median imputation) achieves a stable severity partition. K-Means K=2 on a 9-feature SPC + NCEI tabular matrix produces silhouette 0.320 with bootstrap ARI 0.865. The minority cluster C1 (n=348, 20% of days) captures all named historic outbreaks (1999 OKC, 2011 Apr 27, 2013 Moore, 2021 Quad-State), 23.6% of EF4+ days, and 82% of all 1996–2024 fatalities.

**Continuum regression on the same features:** 5-fold CV gradient boosting predicts `log1p(fatality_total)` with R² = **0.496 ± 0.047**. Top predictors: `ef2plus_count`, `max_mag`, `violent_count`, plus geography (`region_lat`, `region_lon`) entering non-linearly.

**Hypothesis tests:**
- **P2 passed:** EF4+ rate 23.6% in C1 vs 2.2% in C0, p = 5.93×10⁻¹⁴
- **P3 passed:** Apr 27, 2011 bearing coherence (Rayleigh test, p < 0.05)
- **Real-time prediction:** SPC-like features achieve AUROC = **0.746** at T+2h for EF4+ outcomes
- **OII (Impact Index):** Spearman ρ = **0.581** with fatalities

**Important caveat:** The K=2 axis is **outbreak scale/severity**, not convective mode (HP supercell vs. classic supercell vs. QLCS). C1 has *both* QLCS-narrative (54%) and supercell-narrative (79%) elevated — major outbreaks involve multiple convective modes simultaneously. Convective-mode separation requires environmental data (CAPE, 0–6 km shear, helicity) not in SPC or NCEI.

Full analysis in [Results.md](Results.md).
