# %% [markdown]
# # Cascading Supercell Families: Outbreak Style Classification via Merged SPC + NCEI Graph Mining
#
# **Course:** CSCE 676 - Data Mining (Spring 2026)
# **Datasets:**
#   1. NOAA SPC Tornado Database 1950-2024 (`spc_cache.parquet`)
#   2. NOAA NCEI Storm Events Database 1996-2024 (downloaded + cached as `ncei_cache.parquet`)
#
# **v6 additions over v5:**
# - **NCEI merge** — EPISODE_ID grouping counts distinct storm systems per day; narrative
#   text provides QLCS vs. supercell labels for external validation.
# - **Track-geometry features** — individual tornado bearings from elat/elon (98-100%
#   coverage post-2000), hull PCA aspect ratio, hull orientation, geographic centroid,
#   timing IQR, path-length CV.
# - **Multi-archetype clustering** — richer feature matrix expected to produce K=3-5
#   meteorologically interpretable outbreak styles.
# - **Style characterization** — radar chart + geographic map per archetype.
# - **NCEI-label external validation** — qlcs_flag / supercell_flag from narratives tested
#   against cluster assignments (chi-square); NOT used as clustering input.

# %% [markdown]
# ---
# ## Q1 -- Setup

# %% [markdown]
# # Cell 1 -- Setup
# **Before executing this notebook:**
# 1. Activate the project virtual environment
# 2. Run `pip install -r requirements.txt` once to install all dependencies
# 3. Then open this notebook with Jupyter and run all cells

# %%
# Cell 2 -- Imports
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
import time, os, re, gzip, io

import urllib.request
from scipy.spatial import ConvexHull
from scipy.stats import (mannwhitneyu, ttest_ind, spearmanr, chi2_contingency,
                         wilcoxon, rankdata, kendalltau)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score,
                             roc_auc_score, brier_score_loss,
                             average_precision_score)
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import LocalOutlierFactor, BallTree
from sklearn.calibration import CalibratedClassifierCV
import skfuzzy.cluster as fuzzy_cluster

try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
    print('Leiden: available')
except ImportError:
    LEIDEN_AVAILABLE = False
    print('Leiden: not available')

RNG = 42
np.random.seed(RNG)

# %% [markdown]
# ---
# ## Q2 -- SPC Data Load

# %%
# Cell 3 -- Load & clean SPC data (unchanged from v5)
URL   = 'https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv'
CACHE = 'cache/spc_cache.parquet'

if os.path.exists(CACHE):
    df_raw = pd.read_parquet(CACHE)
    print('Loaded SPC data from cache.')
else:
    df_raw = pd.read_csv(URL)
    df_raw.to_parquet(CACHE, index=False)
    print('Downloaded and cached SPC data.')

CONUS = [
    'AL','AR','AZ','CA','CO','CT','DE','FL','GA','IA','ID','IL','IN','KS','KY',
    'LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM',
    'NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA',
    'WI','WV','WY','DC'
]

df_raw['date']  = pd.to_datetime(df_raw['date'], errors='coerce')
df_raw['year']  = df_raw['date'].dt.year
df_raw['month'] = df_raw['date'].dt.month

df_clean = df_raw[
    (df_raw['mag'] >= 0) & (df_raw['mag'] <= 5) &
    df_raw['st'].isin(CONUS) &
    df_raw['date'].notna() &
    df_raw['slat'].between(-90, 90) & df_raw['slon'].between(-180, 180)
].copy()

time_str = df_clean['time'].astype(str).str.zfill(4)
hours = pd.to_numeric(time_str.str[:2], errors='coerce').fillna(0).clip(0, 23)
mins  = pd.to_numeric(time_str.str[2:4], errors='coerce').fillna(0).clip(0, 59)
df_clean['time_frac'] = hours + mins / 60.0
df_clean['datetime']  = df_clean['date'] + pd.to_timedelta(df_clean['time_frac'], unit='h')
df_clean['date_d']    = df_clean['date'].values.astype('datetime64[D]')
df_clean['log_len']   = np.log1p(df_clean['len'])
df_clean['log_wid']   = np.log1p(df_clean['wid'])
df_clean = df_clean.sort_values('datetime').reset_index(drop=True)

MIN_EVENTS = 6
daily_counts = df_clean.groupby('date_d').size()
outbreak_dates_set = set(daily_counts[daily_counts >= MIN_EVENTS].index)
df = df_clean[df_clean['date_d'].isin(outbreak_dates_set)].copy().reset_index(drop=True)

print(f'SPC: {len(df_clean):,} clean events -> {len(outbreak_dates_set):,} outbreak days '
      f'({len(df):,} events)')

# %% [markdown]
# ---
# ## Q3 -- NCEI Storm Events Download & Merge
#
# The SPC tornado CSV has no episode-grouping information. The NOAA NCEI Storm Events
# Database assigns an EPISODE_ID to each meteorological event, grouping tornadoes that
# share the same storm system. This lets us count *distinct storm systems* per day —
# the key signal for distinguishing "one squall line with 40 tornadoes" from
# "40 separate supercells with one tornado each."
#
# The narratives also contain convective-mode keywords (supercell, QLCS, squall line)
# used as external validation labels (NOT as clustering inputs — that would be circular).

# %%
# Cell 4 -- Download + cache NCEI Storm Events (1996-2024)
NCEI_CACHE = 'cache/ncei_cache.parquet'
NCEI_BASE  = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'

if os.path.exists(NCEI_CACHE):
    ncei_df = pd.read_parquet(NCEI_CACHE)
    print(f'Loaded NCEI cache: {len(ncei_df):,} tornado events.')
else:
    print('Fetching NCEI directory listing...')
    with urllib.request.urlopen(NCEI_BASE, timeout=30) as r:
        html = r.read().decode('utf-8', errors='replace')

    pattern = r'StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d+)\.csv\.gz'
    by_year = {}
    for yr, cv in re.findall(pattern, html):
        yr = int(yr); cv = int(cv)
        if yr not in by_year or cv > by_year[yr]:
            by_year[yr] = cv

    # Download 1996-2024
    years_to_dl = [y for y in by_year if 1996 <= y <= 2024]
    print(f'Downloading {len(years_to_dl)} years of NCEI Storm Events...')

    all_frames = []
    KEEP_COLS = ['BEGIN_DATE_TIME', 'EPISODE_ID', 'EVENT_TYPE', 'STATE',
                 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON',
                 'EPISODE_NARRATIVE', 'EVENT_NARRATIVE',
                 'DEATHS_DIRECT', 'INJURIES_DIRECT', 'DAMAGE_PROPERTY']

    for yr in sorted(years_to_dl):
        fname = f'StormEvents_details-ftp_v1.0_d{yr}_c{by_year[yr]}.csv.gz'
        url   = NCEI_BASE + fname
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
            with gzip.open(io.BytesIO(raw)) as f:
                df_yr = pd.read_csv(f, dtype=str, low_memory=False,
                                    encoding_errors='replace')
            # Filter to tornado events in CONUS
            df_yr = df_yr[df_yr['EVENT_TYPE'] == 'Tornado'].copy()
            df_yr = df_yr[df_yr['STATE'].str.upper().isin(
                [s.upper() for s in CONUS] + [  # state names in NCEI
                    'ALABAMA','ARKANSAS','ARIZONA','CALIFORNIA','COLORADO','CONNECTICUT',
                    'DELAWARE','FLORIDA','GEORGIA','IOWA','IDAHO','ILLINOIS','INDIANA',
                    'KANSAS','KENTUCKY','LOUISIANA','MASSACHUSETTS','MARYLAND','MAINE',
                    'MICHIGAN','MINNESOTA','MISSOURI','MISSISSIPPI','MONTANA',
                    'NORTH CAROLINA','NORTH DAKOTA','NEBRASKA','NEW HAMPSHIRE',
                    'NEW JERSEY','NEW MEXICO','NEVADA','NEW YORK','OHIO','OKLAHOMA',
                    'OREGON','PENNSYLVANIA','RHODE ISLAND','SOUTH CAROLINA',
                    'SOUTH DAKOTA','TENNESSEE','TEXAS','UTAH','VIRGINIA','VERMONT',
                    'WASHINGTON','WISCONSIN','WEST VIRGINIA','WYOMING',
                    'DISTRICT OF COLUMBIA'
                ]
            )].copy()
            keep = [c for c in KEEP_COLS if c in df_yr.columns]
            all_frames.append(df_yr[keep])
            print(f'  {yr}: {len(df_yr):,} tornado events', flush=True)
        except Exception as e:
            print(f'  {yr}: SKIP ({e})', flush=True)

    ncei_df = pd.concat(all_frames, ignore_index=True)
    # Parse date
    ncei_df['date_d'] = pd.to_datetime(
        ncei_df['BEGIN_DATE_TIME'], errors='coerce').dt.normalize().values.astype('datetime64[D]')
    ncei_df = ncei_df[ncei_df['date_d'].notna()].copy()
    ncei_df['EPISODE_ID'] = pd.to_numeric(ncei_df['EPISODE_ID'], errors='coerce')
    ncei_df.to_parquet(NCEI_CACHE, index=False)
    print(f'NCEI cache saved: {len(ncei_df):,} tornado events, '
          f'{ncei_df["date_d"].nunique():,} unique days.')

# %%
# Cell 5 -- Compute per-day NCEI episode statistics
print('Computing NCEI per-day episode statistics...')

QLCS_PATTERN   = re.compile(
    r'qlcs|squall.?line|quasi.?linear|bow.?echo|line.of.storms|linear.convect', re.I)
CELL_PATTERN   = re.compile(
    r'supercell|discrete.cell|isolated.cell|rotating.thunder|mesocyclone', re.I)

ncei_features = {}
ncei_df_ob = ncei_df[ncei_df['date_d'].isin(outbreak_dates_set)].copy()

for date_val, day_ncei in ncei_df_ob.groupby('date_d'):
    valid_ep = day_ncei['EPISODE_ID'].dropna()
    if len(valid_ep) == 0:
        continue
    episodes  = day_ncei.groupby('EPISODE_ID')
    n_ep      = len(episodes)
    ep_sizes  = episodes.size().values
    all_text  = ' '.join(
        day_ncei['EVENT_NARRATIVE'].fillna('').astype(str).tolist() +
        day_ncei['EPISODE_NARRATIVE'].fillna('').astype(str).tolist()
    )
    ncei_features[date_val] = {
        'n_ncei_episodes':        int(n_ep),
        'max_episode_tornadoes':  int(ep_sizes.max()),
        'mean_episode_tornadoes': float(ep_sizes.mean()),
        'episode_size_cv':        float(ep_sizes.std(ddof=0) / (ep_sizes.mean() + 1e-9)),
        'qlcs_flag':              int(bool(QLCS_PATTERN.search(all_text))),
        'supercell_flag':         int(bool(CELL_PATTERN.search(all_text))),
        'ncei_n_events':          int(len(day_ncei)),
    }

ncei_feat_df = pd.DataFrame.from_dict(ncei_features, orient='index').reset_index()
ncei_feat_df.rename(columns={'index': 'date'}, inplace=True)
ncei_feat_df['ncei_coverage_flag'] = 1

print(f'NCEI features computed for {len(ncei_feat_df):,} outbreak days '
      f'({len(ncei_feat_df)/len(outbreak_dates_set)*100:.0f}% coverage)')
print(f'Days with QLCS narrative flag: {ncei_feat_df["qlcs_flag"].sum():,} '
      f'({ncei_feat_df["qlcs_flag"].mean()*100:.1f}%)')
print(f'Days with supercell flag:      {ncei_feat_df["supercell_flag"].sum():,} '
      f'({ncei_feat_df["supercell_flag"].mean()*100:.1f}%)')
apr27 = np.datetime64('2011-04-27', 'D')
if apr27 in ncei_features:
    print(f'Apr 27 2011: {ncei_features[apr27]["n_ncei_episodes"]} episodes, '
          f'max_ep_size={ncei_features[apr27]["max_episode_tornadoes"]}')

# %% [markdown]
# ---
# ## Q4 -- Multi-Scale Directed Cascade Graph (unchanged from v5)

# %%
# Cell 6 -- Build directed cascade graph
def build_directed_edges(df_in, R_km, T_hours, cross_day=False):
    R_rad = np.radians(R_km / 6371.0)
    T_ns  = int(T_hours * 3600 * 1e9)
    slat_arr = df_in['slat'].values; slon_arr = df_in['slon'].values
    times_ns = df_in['datetime'].values.astype('int64')
    dates_d  = df_in['date_d'].values
    coords_rad = np.radians(np.column_stack([slat_arr, slon_arr]))
    tree = BallTree(coords_rad, metric='haversine')
    nbrs_list = tree.query_radius(coords_rad, r=R_rad)
    src_l, dst_l, w_l, dist_l, dt_l, bearing_l, vel_l = [], [], [], [], [], [], []
    for i, nbrs in enumerate(nbrs_list):
        ti = times_ns[i]; di = dates_d[i]
        lati = np.radians(slat_arr[i]); loni = np.radians(slon_arr[i])
        for j in nbrs:
            if j == i: continue
            if not cross_day and dates_d[j] != di: continue
            if cross_day:
                day_diff = (dates_d[j] - di).astype('timedelta64[D]').astype(int)
                if day_diff < 0 or day_diff > 1: continue
            dt_ns = int(times_ns[j]) - int(ti)
            if dt_ns <= 0 or dt_ns > T_ns: continue
            dt_h = dt_ns / 3.6e12
            latj = np.radians(slat_arr[j]); lonj = np.radians(slon_arr[j])
            dlat = latj - lati; dlon = lonj - loni
            a = np.sin(dlat/2)**2 + np.cos(lati)*np.cos(latj)*np.sin(dlon/2)**2
            dist = 6371.0 * 2.0 * np.arcsin(np.sqrt(min(1.0, max(0.0, a))))
            if dist < 1.0: continue
            w = np.exp(-dist / R_km) * np.exp(-dt_h / T_hours)
            bearing_rad = np.arctan2(
                np.sin(dlon)*np.cos(latj),
                np.cos(lati)*np.sin(latj) - np.sin(lati)*np.cos(latj)*np.cos(dlon))
            src_l.append(i); dst_l.append(j); w_l.append(w)
            dist_l.append(dist); dt_l.append(dt_h)
            bearing_l.append(np.degrees(bearing_rad)); vel_l.append(dist / dt_h)
    edges = pd.DataFrame({'src': src_l, 'dst': dst_l, 'weight': w_l,
                          'dist_km': dist_l, 'dt_hours': dt_l,
                          'bearing_deg': bearing_l, 'velocity_kmh': vel_l})
    if len(edges) > 0:
        edges['date_d'] = dates_d[edges['src'].values]
    else:
        edges['date_d'] = pd.Series([], dtype='datetime64[ns]')
    return edges

R_KM, T_HOURS = 600.0, 6.0
print(f'Building MESOSCALE edges R={R_KM}km T={T_HOURS}h...')
t0 = time.time()
edges_df = build_directed_edges(df, R_KM, T_HOURS, cross_day=False)
print(f'  {len(edges_df):,} edges  ({time.time()-t0:.1f}s)')

SYN_R, SYN_T = 1400.0, 24.0
print(f'Building SYNOPTIC edges R={SYN_R}km T={SYN_T}h...')
t0 = time.time()
edges_syn_df = build_directed_edges(df, SYN_R, SYN_T, cross_day=False)
print(f'  {len(edges_syn_df):,} edges  ({time.time()-t0:.1f}s)')

PRIME_R, PRIME_T = 1000.0, 24.0
print(f'Building PRIMING edges (cross-day)...')
t0 = time.time()
edges_prime_df = build_directed_edges(df, PRIME_R, PRIME_T, cross_day=True)
if len(edges_prime_df) > 0:
    dates_d_arr = df['date_d'].values
    cross_mask = dates_d_arr[edges_prime_df['src'].values] != dates_d_arr[edges_prime_df['dst'].values]
    edges_prime_df = edges_prime_df[cross_mask].reset_index(drop=True)
print(f'  {len(edges_prime_df):,} cross-day edges  ({time.time()-t0:.1f}s)')

DG = nx.DiGraph()
DG.add_nodes_from(range(len(df)))
for row in edges_df[['src','dst','weight']].itertuples(index=False):
    DG.add_edge(row.src, row.dst, weight=row.weight)

DG_syn = nx.DiGraph()
DG_syn.add_nodes_from(range(len(df)))
for row in edges_syn_df[['src','dst','weight']].itertuples(index=False):
    DG_syn.add_edge(row.src, row.dst, weight=row.weight)

print(f'DiGraph meso: {DG.number_of_edges():,} edges')

# %%
# Cell 7 -- PageRank
slat_arr = df['slat'].values; slon_arr = df['slon'].values
dates_d  = df['date_d'].values

print('Computing directed PageRank (meso + synoptic)...')
t0 = time.time()
dpr_scores     = nx.pagerank(DG,     alpha=0.85, weight='weight', max_iter=200, tol=1e-6)
dpr_syn_scores = nx.pagerank(DG_syn, alpha=0.85, weight='weight', max_iter=200, tol=1e-6)
df['dpr']     = pd.Series(dpr_scores)
df['dpr_syn'] = pd.Series(dpr_syn_scores)
print(f'  PageRank done ({time.time()-t0:.1f}s)')

# %%
# Cell 8 -- P3 bearing + Rayleigh test
DATE_APR27 = np.datetime64('2011-04-27', 'D')
apr27_edges = edges_df[edges_df['date_d'] == DATE_APR27]
if len(apr27_edges) > 0:
    bearings = apr27_edges['bearing_deg'].values
    bear_rad = np.radians(bearings)
    C_ = np.mean(np.cos(bear_rad)); S_ = np.mean(np.sin(bear_rad))
    R_bar = np.sqrt(C_**2 + S_**2)
    circ_mean_deg = (np.degrees(np.arctan2(S_, C_)) + 360) % 360
    n_b = len(bearings)
    z_ray = n_b * R_bar**2
    p_ray = float(max(0.0, np.exp(-z_ray)))
    print(f'P3: Apr 27 bearing = {circ_mean_deg:.1f}deg, Rayleigh p = {p_ray:.2e}')
    p_ray_val = p_ray
else:
    circ_mean_deg = np.nan; p_ray_val = np.nan

# %% [markdown]
# ---
# ## Q5 -- Feature Engineering: SPC Track Geometry + NCEI Episodes

# %%
# Cell 9 -- Helper functions

def convex_hull_area_km2(lats, lons):
    if len(lats) < 3: return 0.0
    lat_c = np.mean(lats)
    kpd_lon = 111.0 * np.cos(np.radians(lat_c))
    x = (np.array(lons) - np.mean(lons)) * kpd_lon
    y = (np.array(lats) - np.mean(lats)) * 111.0
    pts = np.unique(np.column_stack([x, y]), axis=0)
    if len(pts) < 3: return 0.0
    try: return ConvexHull(pts).volume
    except Exception: return 0.0

def convex_hull_aspect_pca(lats, lons):
    """Returns (log_aspect_ratio, orient_sin, orient_cos).
    log_aspect: 0=square, higher=elongated (log-scaled for robustness to extremes).
    orient: bearing of hull major axis (SW-NE = 45deg typical Plains outbreak).
    """
    if len(lats) < 4: return 0.0, 0.0, 1.0
    lat_c = np.mean(lats)
    x = (np.array(lons) - np.mean(lons)) * 111.0 * np.cos(np.radians(lat_c))
    y = (np.array(lats) - np.mean(lats)) * 111.0
    pts = np.unique(np.column_stack([x, y]), axis=0)
    if len(pts) < 4: return 0.0, 0.0, 1.0
    try:
        hull = ConvexHull(pts)
        pca  = PCA(n_components=2).fit(pts[hull.vertices])
        ev   = pca.explained_variance_ratio_
        log_ar = float(np.log1p(ev[0] / (ev[1] + 1e-9)))
        comp   = pca.components_[0]
        orient_deg = (np.degrees(np.arctan2(comp[0], comp[1])) + 360) % 360
        return (log_ar,
                float(np.sin(np.radians(orient_deg))),
                float(np.cos(np.radians(orient_deg))))
    except Exception:
        return 0.0, 0.0, 1.0

# %%
# Cell 10 -- Per-outbreak-day feature engineering (extended)
print('Engineering per-day features (SPC + NCEI track geometry)...')
t0 = time.time()

edges_by_date     = {d: g for d, g in edges_df.groupby('date_d')}
edges_syn_by_date = {d: g for d, g in edges_syn_df.groupby('date_d')}

dates_d_arr_full = df['date_d'].values
if len(edges_prime_df) > 0:
    prime_target_date = dates_d_arr_full[edges_prime_df['dst'].values]
    prime_count_by_date = pd.Series(prime_target_date).value_counts().to_dict()
else:
    prime_count_by_date = {}

wcc_meso_by_date = {}; wcc_syn_by_date = {}
for date_val, day_df_ in df.groupby('date_d'):
    idx = set(day_df_.index.tolist())
    wcc_meso_by_date[date_val] = len(list(nx.weakly_connected_components(DG.subgraph(idx))))
    wcc_syn_by_date[date_val]  = len(list(nx.weakly_connected_components(DG_syn.subgraph(idx))))

# First-pass: compute is_first_hour for DPR diagnostic
day_first_time = df.groupby('date_d')['time_frac'].transform('min')
df['is_first_hour'] = (df['time_frac'] - day_first_time) <= 1.0

outbreak_rows = []
for date_val, day_df_ in df.groupby('date_d'):
    n = len(day_df_)

    # --- Tabular ---
    ev_count     = n
    fat_total    = int(day_df_['fat'].sum())
    inj_total    = int(day_df_['inj'].sum())
    max_mag      = int(day_df_['mag'].max())
    sig_frac     = float((day_df_['mag'] >= 2).mean())
    ef2plus_ct   = int((day_df_['mag'] >= 2).sum())
    vio_count    = int((day_df_['mag'] >= 4).sum())
    mean_log_len = float(day_df_['log_len'].mean())
    noct_frac    = float(((day_df_['time_frac'] >= 18) | (day_df_['time_frac'] < 6)).mean())
    state_count  = int(day_df_['st'].nunique())
    hull_area    = convex_hull_area_km2(day_df_['slat'].values, day_df_['slon'].values)
    month_val    = int(day_df_['month'].iloc[0])
    year_val     = int(pd.Timestamp(str(date_val)[:10]).year)
    duration_h   = float(day_df_['time_frac'].max() - day_df_['time_frac'].min())

    # --- Cascade graph features ---
    day_edges     = edges_by_date.get(date_val, pd.DataFrame(columns=edges_df.columns))
    day_edges_syn = edges_syn_by_date.get(date_val, pd.DataFrame(columns=edges_syn_df.columns))
    n_edges       = len(day_edges); n_edges_syn = len(day_edges_syn)
    mean_edge_km  = float(day_edges['dist_km'].mean()) if n_edges > 0 else 0.0
    n_meso_comp   = int(wcc_meso_by_date.get(date_val, 0))
    n_syn_comp    = int(wcc_syn_by_date.get(date_val, 0))
    syn_connect   = float(n_meso_comp / max(1, n_syn_comp))
    n_priming     = int(prime_count_by_date.get(date_val, 0))

    # --- Centroid-translation velocity ---
    if n >= 4 and duration_h > 0.25:
        sorted_day = day_df_.sort_values('time_frac')
        half = max(1, n // 2)
        fh = sorted_day.head(half); lh = sorted_day.tail(half)
        t1 = float(fh['time_frac'].mean()); t2 = float(lh['time_frac'].mean()); dt = t2 - t1
        if dt > 0.1:
            phi1 = np.radians(float(fh['slat'].mean()))
            phi2 = np.radians(float(lh['slat'].mean()))
            dlam = np.radians(float(lh['slon'].mean()) - float(fh['slon'].mean()))
            dphi = phi2 - phi1
            a_h  = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
            dist_c = 2 * 6371.0 * np.arcsin(np.sqrt(a_h))
            vel_mean = float(dist_c / dt)
            y_b = np.sin(dlam)*np.cos(phi2)
            x_b = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam)
            bear = (np.degrees(np.arctan2(y_b, x_b)) + 360) % 360
            br_r = np.radians(bear)
            b_sin = float(np.sin(br_r)); b_cos = float(np.cos(br_r)); b_cons = 1.0
        else:
            vel_mean = b_sin = b_cos = b_cons = 0.0
    else:
        vel_mean = b_sin = b_cos = b_cons = 0.0

    vel_std = float(day_edges['velocity_kmh'].std(ddof=0)) if n_edges > 1 else 0.0

    if n >= 4:
        Xt = day_df_['time_frac'].values.reshape(-1, 1)
        ym = day_df_['mag'].values.astype(float)
        ef_slope = float(LinearRegression().fit(Xt, ym).coef_[0])
    else:
        ef_slope = 0.0

    first_t  = float(day_df_['time_frac'].min())
    ef3_evts = day_df_[day_df_['mag'] >= 3]
    t_ef3    = float(ef3_evts['time_frac'].min() - first_t) if len(ef3_evts) > 0 else 24.0
    wl_ratio = (day_df_['wid'] + 1.0) / (day_df_['len'] + 1.0)
    max_wl   = float(wl_ratio.max())

    dpr_day   = day_df_['dpr']
    mean_dpr  = float(dpr_day.mean()); max_dpr = float(dpr_day.max())
    anchor_dom   = float(max_dpr / (mean_dpr + 1e-12))
    dpr_syn_max  = float(day_df_['dpr_syn'].max())
    has_ef4      = int((day_df_['mag'] >= 4).any())

    # =========================================================
    # NEW §B: TRACK GEOMETRY FEATURES (from elat/elon)
    # =========================================================

    # Individual track bearings
    valid_end_mask = (day_df_['elat'] != 0) & (day_df_['elon'] != 0) & \
                     (day_df_['elat'].notna()) & (day_df_['elon'].notna())
    track_coverage = float(valid_end_mask.mean())

    if valid_end_mask.sum() >= 3:
        slats_v = day_df_.loc[valid_end_mask, 'slat'].values
        slons_v = day_df_.loc[valid_end_mask, 'slon'].values
        elats_v = day_df_.loc[valid_end_mask, 'elat'].values
        elons_v = day_df_.loc[valid_end_mask, 'elon'].values
        phi1_v  = np.radians(slats_v); phi2_v = np.radians(elats_v)
        dlam_v  = np.radians(elons_v - slons_v)
        y_tb    = np.sin(dlam_v) * np.cos(phi2_v)
        x_tb    = np.cos(phi1_v)*np.sin(phi2_v) - np.sin(phi1_v)*np.cos(phi2_v)*np.cos(dlam_v)
        track_bearings = (np.degrees(np.arctan2(y_tb, x_tb)) + 360) % 360
        C_tb = np.mean(np.cos(np.radians(track_bearings)))
        S_tb = np.mean(np.sin(np.radians(track_bearings)))
        R_tb = float(np.sqrt(C_tb**2 + S_tb**2))
        track_bearing_mean = float((np.degrees(np.arctan2(S_tb, C_tb)) + 360) % 360)
        track_bearing_R    = R_tb
        track_bearing_std  = float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(R_tb + 1e-9)))))
        tb_sin = float(np.sin(np.radians(track_bearing_mean)))
        tb_cos = float(np.cos(np.radians(track_bearing_mean)))
    else:
        # Low coverage: zero-impute bearing trig features; std to max; R to 0
        track_bearing_mean = 0.0; track_bearing_R = 0.0; track_bearing_std = 180.0
        tb_sin = 0.0; tb_cos = 0.0

    # Hull PCA aspect ratio + orientation
    hull_log_asp, hull_os, hull_oc = convex_hull_aspect_pca(
        day_df_['slat'].values, day_df_['slon'].values)

    # Geographic centroid
    region_lat = float(day_df_['slat'].mean())
    region_lon = float(day_df_['slon'].mean())

    # Temporal IQR
    tvals = day_df_['time_frac'].values
    timing_iqr = float(np.percentile(tvals, 75) - np.percentile(tvals, 25)) if n >= 4 else 0.0

    # Path-length spread
    lens_v = day_df_['len'].values
    track_len_cv   = float(lens_v.std(ddof=0) / (lens_v.mean() + 1e-9)) if n >= 4 else 0.0
    track_wid_mean = float(day_df_['wid'].mean())

    outbreak_rows.append({
        # Identifiers
        'date': date_val, 'year': year_val, 'month': month_val,
        'month_sin': np.sin(2*np.pi*month_val/12),
        'month_cos': np.cos(2*np.pi*month_val/12),
        # Tabular
        'event_count': ev_count, 'fatality_total': fat_total, 'injury_total': inj_total,
        'max_mag': max_mag, 'sig_fraction': sig_frac, 'ef2plus_count': ef2plus_ct,
        'violent_count': vio_count, 'mean_log_len': mean_log_len,
        'nocturnal_fraction': noct_frac, 'state_count': state_count,
        'hull_area_km2': hull_area, 'duration_hours': duration_h,
        # Cascade / graph
        'n_edges': n_edges, 'n_edges_syn': n_edges_syn,
        'cascade_velocity_kmh': vel_mean, 'cascade_velocity_std': vel_std,
        'cascade_bearing_sin': b_sin, 'cascade_bearing_cos': b_cos,
        'cascade_bearing_consistency': b_cons,
        'ef_slope_per_hour': ef_slope, 'time_to_first_ef3': t_ef3,
        'max_wid_len_ratio': max_wl, 'mean_edge_km': mean_edge_km,
        'anchor_dominance': anchor_dom, 'dpr_syn_max': dpr_syn_max,
        'n_meso_components': n_meso_comp, 'n_synoptic_components': n_syn_comp,
        'synoptic_connectivity': syn_connect, 'n_priming_edges': n_priming,
        # NEW: Track geometry
        'track_bearing_sin': tb_sin, 'track_bearing_cos': tb_cos,
        'track_bearing_R': track_bearing_R, 'track_bearing_std': track_bearing_std,
        'track_coverage': track_coverage,
        'hull_log_aspect': hull_log_asp,
        'hull_orient_sin': hull_os, 'hull_orient_cos': hull_oc,
        'region_lat': region_lat, 'region_lon': region_lon,
        'timing_iqr_hours': timing_iqr,
        'track_len_cv': track_len_cv, 'track_wid_mean': track_wid_mean,
        # Target
        'has_ef4plus': has_ef4,
    })

outbreaks = pd.DataFrame(outbreak_rows)

# Merge NCEI episode features (left join on date)
outbreaks = outbreaks.merge(ncei_feat_df, on='date', how='left')
outbreaks['ncei_coverage_flag'] = outbreaks['ncei_coverage_flag'].fillna(0).astype(int)
NCEI_FILL_COLS = ['n_ncei_episodes', 'max_episode_tornadoes', 'mean_episode_tornadoes',
                  'episode_size_cv', 'qlcs_flag', 'supercell_flag']
for col in NCEI_FILL_COLS:
    med = outbreaks[col].median()
    outbreaks[col] = outbreaks[col].fillna(med if not np.isnan(med) else 0.0)

print(f'Outbreak days engineered: {len(outbreaks):,}  ({time.time()-t0:.1f}s)')
print(f'EF4+ days: {outbreaks["has_ef4plus"].sum():,} ({outbreaks["has_ef4plus"].mean()*100:.1f}%)')
print(f'NCEI coverage: {outbreaks["ncei_coverage_flag"].mean()*100:.0f}% of outbreak days')
print(f'Track bearing coverage (mean): {outbreaks["track_coverage"].mean():.2f} '
      f'(post-2000: {outbreaks.loc[outbreaks["year"]>=2000,"track_coverage"].mean():.2f})')
print(f'Hull aspect ratio range: {outbreaks["hull_log_aspect"].min():.2f} - '
      f'{outbreaks["hull_log_aspect"].max():.2f}  (median={outbreaks["hull_log_aspect"].median():.2f})')
print(f'Apr 27 2011 n_ncei_episodes: '
      f'{outbreaks.loc[outbreaks["date"]==np.datetime64("2011-04-27","D"), "n_ncei_episodes"].values}')

# %% [markdown]
# ---
# ## Q6 -- Outbreak Style Clustering (K-Means + HAC + GMM)
#
# The v6 feature matrix adds track geometry, geographic centroid, timing IQR, and NCEI
# episode counts. NCEI narrative flags (qlcs_flag, supercell_flag) are kept OUT of the
# clustering input to serve as independent external validation labels.

# %%
# Cell 11 -- K-Means sweep with extended feature matrix
CASCADE_FEATS_V5 = [
    'event_count', 'hull_area_km2', 'state_count',
    'max_mag', 'sig_fraction', 'violent_count', 'mean_log_len',
    'nocturnal_fraction', 'cascade_velocity_kmh', 'cascade_bearing_sin',
    'cascade_bearing_cos', 'cascade_bearing_consistency', 'ef_slope_per_hour',
    'time_to_first_ef3', 'max_wid_len_ratio', 'mean_edge_km',
    'month_sin', 'month_cos',
    'n_meso_components', 'synoptic_connectivity', 'n_priming_edges',
]

CASCADE_FEATS = CASCADE_FEATS_V5 + [
    # Track geometry (NEW)
    'hull_log_aspect', 'hull_orient_sin', 'hull_orient_cos',
    'track_bearing_sin', 'track_bearing_cos', 'track_bearing_R',
    'track_bearing_std', 'track_coverage',
    'region_lat', 'region_lon',
    'timing_iqr_hours', 'track_len_cv', 'track_wid_mean',
    # NCEI episode features (NEW)
    'n_ncei_episodes', 'max_episode_tornadoes', 'episode_size_cv',
]
# qlcs_flag and supercell_flag are EXTERNAL VALIDATION labels -- excluded from clustering.

X_ob = outbreaks[CASCADE_FEATS].fillna(0).values
scaler_ob = StandardScaler()
X_ob_sc   = scaler_ob.fit_transform(X_ob)

sil_scores = {}; db_scores = {}; ch_scores = {}
for K_ in range(2, 9):
    km_ = KMeans(n_clusters=K_, n_init=20, random_state=RNG).fit(X_ob_sc)
    sil_scores[K_] = silhouette_score(X_ob_sc, km_.labels_)
    db_scores[K_]  = davies_bouldin_score(X_ob_sc, km_.labels_)
    ch_scores[K_]  = calinski_harabasz_score(X_ob_sc, km_.labels_)

print('K | Silhouette | Davies-Bouldin | Calinski-Harabasz')
for K_ in sorted(sil_scores):
    print(f'  K={K_}: sil={sil_scores[K_]:.4f}  DB={db_scores[K_]:.3f}  CH={ch_scores[K_]:.1f}')

K_BEST = max(sil_scores, key=sil_scores.get)
print(f'\nOptimal K={K_BEST} (silhouette)')

km  = KMeans(n_clusters=K_BEST, n_init=20, random_state=RNG).fit(X_ob_sc)
hac = AgglomerativeClustering(n_clusters=K_BEST, linkage='ward').fit(X_ob_sc)
gmm = GaussianMixture(n_components=K_BEST, random_state=RNG, n_init=5).fit(X_ob_sc)
outbreaks['kmeans_label'] = km.labels_
outbreaks['hac_label']    = hac.labels_
outbreaks['gmm_label']    = gmm.predict(X_ob_sc)

print(f'ARI K-Means vs HAC: {adjusted_rand_score(outbreaks["kmeans_label"], outbreaks["hac_label"]):.3f}')
print(f'ARI K-Means vs GMM: {adjusted_rand_score(outbreaks["kmeans_label"], outbreaks["gmm_label"]):.3f}')

# Plot evaluation curves
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
Ks = sorted(sil_scores.keys())
axes[0].plot(Ks, [sil_scores[k] for k in Ks], 'o-', color='#E63946', lw=2)
axes[0].set_title('Silhouette (higher=better)', fontweight='bold')
axes[0].axvline(K_BEST, ls='--', color='gray')
axes[1].plot(Ks, [db_scores[k] for k in Ks], 'o-', color='#457B9D', lw=2)
axes[1].set_title('Davies-Bouldin (lower=better)', fontweight='bold')
axes[1].axvline(K_BEST, ls='--', color='gray')
axes[2].plot(Ks, [ch_scores[k] for k in Ks], 'o-', color='#2D6A4F', lw=2)
axes[2].set_title('Calinski-Harabasz (higher=better)', fontweight='bold')
axes[2].axvline(K_BEST, ls='--', color='gray')
for ax in axes:
    ax.set_xlabel('K'); ax.set_ylabel('Score')
plt.tight_layout()
plt.savefig('figures/fig_v6_kmeans_eval.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %%
# Cell 12 -- Archetype centroid table + data-driven labels
CHAR_COLS = ['event_count', 'cascade_velocity_kmh', 'ef_slope_per_hour',
             'nocturnal_fraction', 'violent_count', 'fatality_total', 'has_ef4plus',
             'hull_log_aspect', 'track_bearing_R', 'track_bearing_std',
             'region_lat', 'region_lon', 'timing_iqr_hours',
             'n_ncei_episodes', 'max_episode_tornadoes']
cluster_means = outbreaks.groupby('kmeans_label')[CHAR_COLS].mean()

print('\n=== Archetype Centroids (K-Means, v6 — data-driven) ===')
print(cluster_means.round(2).to_string())

# Assign descriptive labels from centroids
archetype_labels = {}
for lbl in range(K_BEST):
    row = cluster_means.loc[lbl]
    parts = []
    if row['cascade_velocity_kmh'] < cluster_means['cascade_velocity_kmh'].median():
        parts.append('slow-centroid')
    else:
        parts.append('fast-propagating')
    if row['nocturnal_fraction'] > 0.45:
        parts.append('nocturnal')
    if row['hull_log_aspect'] > cluster_means['hull_log_aspect'].median():
        parts.append('elongated-hull')
    if row['region_lat'] < 35.0:
        parts.append('Dixie-Alley')
    elif row['region_lat'] > 40.0:
        parts.append('Northern-Plains')
    if row['n_ncei_episodes'] > cluster_means['n_ncei_episodes'].median():
        parts.append('multi-system')
    if row['track_bearing_R'] < 0.4:
        parts.append('variable-bearing')
    if row['violent_count'] > 0.5:
        parts.append('violent')
    archetype_labels[lbl] = ' / '.join(parts) if parts else f'Cluster {lbl}'
outbreaks['archetype'] = outbreaks['kmeans_label'].map(archetype_labels)

print('\nDerived archetype labels:')
for lbl in range(K_BEST):
    n_ = int((outbreaks['kmeans_label'] == lbl).sum())
    ef4_rate = float(outbreaks.loc[outbreaks['kmeans_label']==lbl, 'has_ef4plus'].mean())
    qlcs_rate = float(outbreaks.loc[outbreaks['kmeans_label']==lbl, 'qlcs_flag'].mean())
    cell_rate = float(outbreaks.loc[outbreaks['kmeans_label']==lbl, 'supercell_flag'].mean())
    print(f'  C{lbl} (n={n_:4d}): {archetype_labels[lbl][:55]:55s} '
          f'EF4+={ef4_rate*100:5.1f}%  QLCS-narrative={qlcs_rate*100:4.1f}%  '
          f'Cell-narrative={cell_rate*100:4.1f}%')

# %%
# Cell 13 -- External validation: NCEI narrative flags vs cluster (chi-square)
print('\n=== External validation: NCEI narrative labels vs. K-Means archetypes ===')
for flag, name in [('qlcs_flag', 'QLCS'), ('supercell_flag', 'Supercell')]:
    ct = pd.crosstab(outbreaks['kmeans_label'], outbreaks[flag].astype(int))
    if ct.shape[1] < 2: continue
    chi2, p_chi, _, _ = chi2_contingency(ct.values)
    print(f'  {name} flag vs. archetype: chi2={chi2:.1f}, p={p_chi:.2e}  '
          f'{"(significant association)" if p_chi < 0.05 else "(no significant association)"}')
    for lbl in range(K_BEST):
        rate = float(outbreaks.loc[outbreaks['kmeans_label']==lbl, flag].mean())
        print(f'    C{lbl}: {rate*100:.1f}%')

# %%
# Cell 14 -- Radar chart (spider plot) per archetype
RADAR_DIMS = ['event_count', 'cascade_velocity_kmh', 'hull_log_aspect',
              'track_bearing_R', 'nocturnal_fraction', 'n_ncei_episodes',
              'region_lat', 'timing_iqr_hours']
RADAR_LABELS = ['Event Count', 'Cascade Vel.', 'Hull Elongation',
                'Bearing Consistency', 'Nocturnal Frac.', 'NCEI Episodes',
                'Latitude', 'Timing IQR']

# Normalize each dimension to 0-1 across cluster means for radar
means_for_radar = cluster_means[RADAR_DIMS].copy()
means_norm = (means_for_radar - means_for_radar.min()) / \
             (means_for_radar.max() - means_for_radar.min() + 1e-9)

N = len(RADAR_DIMS)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

colors_arch = cm.Set1(np.linspace(0, 0.8, K_BEST))
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
for lbl in range(K_BEST):
    vals = means_norm.loc[lbl, RADAR_DIMS].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', lw=2, color=colors_arch[lbl],
            label=f'C{lbl}: {archetype_labels[lbl][:30]}')
    ax.fill(angles, vals, alpha=0.1, color=colors_arch[lbl])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_LABELS, size=9)
ax.set_ylim(0, 1)
ax.set_title('Outbreak Archetype Profiles (normalized centroid values)', fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig_v6_archetype_radar.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %%
# Cell 15 -- Geographic style map
fig, ax = plt.subplots(figsize=(13, 7))
colors_arch_map = {lbl: colors_arch[lbl] for lbl in range(K_BEST)}
for lbl in range(K_BEST):
    mask = outbreaks['kmeans_label'] == lbl
    ax.scatter(outbreaks.loc[mask, 'region_lon'],
               outbreaks.loc[mask, 'region_lat'],
               c=[colors_arch_map[lbl]], alpha=0.35, s=8,
               label=f'C{lbl}: {archetype_labels[lbl][:30]}')
# Mark named outbreaks
NAMED = {'1974-04-03': '1974', '2011-04-27': 'Apr27\n2011',
         '1999-05-03': 'OKC\n1999', '2021-12-10': 'Quad-St\n2021'}
for ds, name in NAMED.items():
    dt = np.datetime64(ds, 'D')
    row = outbreaks[outbreaks['date'] == dt]
    if len(row) > 0:
        r = row.iloc[0]
        ax.scatter(r['region_lon'], r['region_lat'], s=120, color='gold',
                   edgecolor='black', zorder=5)
        ax.annotate(name, (r['region_lon'], r['region_lat']),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel('Mean Longitude of Outbreak'); ax.set_ylabel('Mean Latitude of Outbreak')
ax.set_title('Geographic Distribution of Outbreak Styles (K-Means Archetypes)', fontweight='bold')
ax.set_xlim(-110, -65); ax.set_ylim(24, 52)
ax.legend(fontsize=7, loc='upper left')
plt.tight_layout()
plt.savefig('figures/fig_v6_style_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% [markdown]
# ---
# ## Q7 -- P1 / P2 / P3 Hypothesis Tests (carried forward from v5)

# %%
# Cell 16 -- OII + P1 two-stage test
eps = 1e-9
outbreaks['density'] = outbreaks['event_count'] / (outbreaks['hull_area_km2'] / 1000.0 + eps)
vel_max = outbreaks['cascade_velocity_kmh'].quantile(0.99) + eps
outbreaks['cascade_velocity_norm'] = outbreaks['cascade_velocity_kmh'] / vel_max

outbreaks['oii_retro'] = (
    np.log10(1 + 10*outbreaks['fatality_total'] + outbreaks['injury_total'])
    + np.sqrt(outbreaks['event_count'])
    + 0.5 * outbreaks['sig_fraction'] * outbreaks['max_mag']
    + 0.3 * np.log1p(outbreaks['density'])
)

# OII_learned: Ridge on extended feature matrix
RIDGE_FEATS = [
    'event_count', 'hull_area_km2', 'state_count', 'max_mag',
    'sig_fraction', 'violent_count', 'mean_log_len', 'nocturnal_fraction',
    'cascade_velocity_kmh', 'ef_slope_per_hour', 'n_meso_components',
    'synoptic_connectivity', 'hull_log_aspect', 'n_ncei_episodes',
    'max_episode_tornadoes', 'region_lat', 'timing_iqr_hours', 'track_bearing_R',
]
X_r = outbreaks[RIDGE_FEATS].fillna(0).values
X_r_sc = StandardScaler().fit_transform(X_r)
y_r = np.log1p(outbreaks['fatality_total'].values)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0, random_state=RNG).fit(X_r_sc, y_r)
outbreaks['oii_learned'] = ridge.predict(X_r_sc)
outbreaks['oii_rt'] = outbreaks['oii_retro'] + 0.2 * outbreaks['cascade_velocity_norm']

spr_retro, _ = spearmanr(outbreaks['oii_retro'],  outbreaks['fatality_total'])
spr_learn, _ = spearmanr(outbreaks['oii_learned'], outbreaks['fatality_total'])
spr_ef2,   _ = spearmanr(outbreaks['ef2plus_count'], outbreaks['fatality_total'])
spr_ct,    _ = spearmanr(outbreaks['event_count'],   outbreaks['fatality_total'])
print(f'OII Spearman: retro={spr_retro:.3f}  learned={spr_learn:.3f}  '
      f'EF2+count={spr_ef2:.3f}  event_count={spr_ct:.3f}')

# P1 two-stage
pds_threshold = outbreaks['oii_retro'].quantile(0.90)
outbreaks['is_pds'] = (outbreaks['oii_retro'] >= pds_threshold).astype(int)
pds_vel   = outbreaks.loc[outbreaks['is_pds']==1, 'cascade_velocity_kmh'].values
npds_vel  = outbreaks.loc[outbreaks['is_pds']==0, 'cascade_velocity_kmh'].values
_, p_p1a  = mannwhitneyu(pds_vel, npds_vel, alternative='greater')
_, p_p1b  = mannwhitneyu(pds_vel, npds_vel, alternative='less')
print(f'\nP1a pre-registered (PDS > non-PDS): p={p_p1a:.2e}  -> '
      f'{"FAIL (as expected)" if p_p1a >= 0.05 else "PASS"}')
print(f'P1b exploratory (PDS < non-PDS):    p={p_p1b:.2e}  -> '
      f'{"SUPPORTED" if p_p1b < 0.05 else "NOT SUPPORTED"}')

# P2
ef4_rate_by_cl = outbreaks.groupby('kmeans_label')['has_ef4plus'].mean()
hi_cl = int(ef4_rate_by_cl.idxmax()); lo_cl = int(ef4_rate_by_cl.idxmin())
hi_mask = outbreaks['kmeans_label']==hi_cl; lo_mask = outbreaks['kmeans_label']==lo_cl
n_hi = hi_mask.sum(); n_lo = lo_mask.sum()
k_hi = int(outbreaks.loc[hi_mask,'has_ef4plus'].sum())
k_lo = int(outbreaks.loc[lo_mask,'has_ef4plus'].sum())
rate_hi = k_hi/max(1,n_hi); rate_lo = k_lo/max(1,n_lo)
ct_p2 = np.array([[k_hi,n_hi-k_hi],[k_lo,n_lo-k_lo]])
chi2_p2, p_p2, _, _ = chi2_contingency(ct_p2)
rr = rate_hi / max(rate_lo, 1e-9)
cohen_h = 2*(np.arcsin(np.sqrt(rate_hi)) - np.arcsin(np.sqrt(max(rate_lo,1e-9))))
print(f'P2 chi-square: p={p_p2:.2e}, risk-ratio={rr:.1f}x, Cohen h={cohen_h:.2f}  -> '
      f'{"PASS" if p_p2<0.05 and rr>=3 else "FAIL"}')

# Holm-Bonferroni
raw_ps = {'P1b': p_p1b, 'P2': p_p2, 'P3': p_ray_val if not np.isnan(p_ray_val) else 1.0}
sorted_ps = sorted(raw_ps.items(), key=lambda kv: kv[1])
holm_ps = {}
for rank, (name, p) in enumerate(sorted_ps):
    holm_ps[name] = min(1.0, (len(sorted_ps)-rank)*p)
print(f'\nHolm-Bonferroni: P1b_adj={holm_ps["P1b"]:.2e}  P2_adj={holm_ps["P2"]:.2e}  '
      f'P3_adj={holm_ps["P3"]:.2e}')

# %% [markdown]
# ---
# ## Q8 -- Fuzzy C-Means + Permutation Test (continuum check with extended features)

# %%
# Cell 17 -- FCM + permutation test
FCM_K = K_BEST
X_fuzz = X_ob_sc.T
cntr, u, u0, d, jm, p_iter, fpc = fuzzy_cluster.cmeans(
    X_fuzz, c=FCM_K, m=2.0, error=1e-5, maxiter=1000, seed=RNG)
print(f'FCM K={FCM_K}: PC = {fpc:.4f}  (floor = {1/FCM_K:.4f})')

for k_ in range(FCM_K):
    outbreaks[f'fuzzy_u{k_}'] = u[k_]
outbreaks['fuzzy_max_u']   = u.max(axis=0)
outbreaks['fuzzy_entropy'] = -np.sum(u * np.log(u + 1e-12), axis=0)

print('Permutation test (50 shuffles)...')
null_pcs = []
rng_ = np.random.RandomState(RNG)
for _ in range(50):
    X_shuf = X_ob_sc.copy()
    for col in range(X_shuf.shape[1]):
        rng_.shuffle(X_shuf[:, col])
    try:
        _, _, _, _, _, _, fpc_null = fuzzy_cluster.cmeans(
            X_shuf.T, c=FCM_K, m=2.0, error=1e-4, maxiter=300, seed=RNG)
        null_pcs.append(fpc_null)
    except Exception:
        continue
null_mean = float(np.mean(null_pcs)); null_std = float(np.std(null_pcs))
z_pc = (fpc - null_mean) / max(null_std, 1e-9)
print(f'  Observed PC={fpc:.4f}  null={null_mean:.4f}+/-{null_std:.4f}  z={z_pc:.2f}')
if abs(z_pc) < 2.0:
    print('  -> Within 2 sigma of null: outbreak space remains a continuum with v6 features.')
else:
    print('  -> Significantly above null: richer features reveal discrete soft structure.')

# %% [markdown]
# ---
# ## Q9 -- Anomaly Detection (re-run on v6 feature matrix)

# %%
# Cell 18 -- Isolation Forest + LOF
from scipy.stats import rankdata as _rankdata
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=RNG)
outbreaks['iso_label'] = iso.fit_predict(X_ob_sc)
outbreaks['iso_score'] = -iso.score_samples(X_ob_sc)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
outbreaks['lof_label'] = lof.fit_predict(X_ob_sc)
outbreaks['lof_score'] = -lof.negative_outlier_factor_

outbreaks['fused_rank'] = (rankdata(outbreaks['iso_score']) +
                           rankdata(outbreaks['lof_score'])) / 2.0

NAMED_DATES = {
    '1974-04-03': '1974 Super Outbreak',
    '2011-04-27': 'April 27 2011',
    '1965-04-11': '1965 Palm Sunday',
    '2021-12-10': 'Quad-State 2021',
    '1999-05-03': 'OKC May 3 1999',
}
print('Named outbreak anomaly ranks (fused, top % = most anomalous):')
for ds, name in NAMED_DATES.items():
    dt = np.datetime64(ds, 'D')
    row = outbreaks[outbreaks['date'] == dt]
    if len(row) > 0:
        r = row.iloc[0]
        pct = 100 * (1 - r['fused_rank'] / len(outbreaks))
        arch = r['archetype']
        print(f'  {name:28s}: top {pct:.2f}%  archetype={arch[:30]}')

# %% [markdown]
# ---
# ## Q10 -- Real-Time Simulation (v5 logic preserved; NCEI features excluded as retrospective)
#
# NCEI episode IDs are assigned retrospectively by NCEI staff, NOT available in real time.
# Therefore, NCEI-derived features (n_ncei_episodes, qlcs_flag) are EXCLUDED from the
# real-time simulation. The 5 feature sets from v5 are preserved unchanged.

# %%
# Cell 19 -- Real-time snapshot loop
SNAPSHOT_HOURS = [1, 2, 3, 4]
df_by_date = {d: g.sort_values('time_frac') for d, g in df.groupby('date_d')}
edges_by_date_full = {d: g for d, g in edges_df.groupby('date_d')}

rt_snapshots = []
for _, ob_row in outbreaks.iterrows():
    date_val = ob_row['date']; target = int(ob_row['has_ef4plus'])
    year_ = int(ob_row['year']); month_ = int(ob_row['month'])
    day_df_ = df_by_date.get(date_val)
    if day_df_ is None or len(day_df_) == 0: continue
    first_t = float(day_df_['time_frac'].min())
    for t_h in SNAPSHOT_HOURS:
        cutoff = first_t + t_h
        sub = day_df_[day_df_['time_frac'] <= cutoff]
        n_so_far = len(sub)
        if n_so_far < 2:
            rt_snapshots.append({'date': date_val, 'year': year_, 't_h': t_h,
                'target': target, 'count_so_far': n_so_far, 'ef_max_so_far': 0,
                'hull_rt': 0.0, 'cascade_vel_rt': 0.0, 'ef_slope_rt': 0.0,
                'dpr_max_rt': 0.0, 'dpr_syn_max_rt': 0.0,
                'noct_frac_rt': 0.0, 'state_count_rt': 0,
                'partial_dpr_rt': 0.0, 'n_meso_comp_rt': 0,
                'fuzzy_entropy_rt': 0.0,
                'month_sin': np.sin(2*np.pi*month_/12),
                'month_cos': np.cos(2*np.pi*month_/12)})
            continue
        ef_max    = int(sub['mag'].max())
        hull_rt   = convex_hull_area_km2(sub['slat'].values, sub['slon'].values)
        noct_rt   = float(((sub['time_frac']>=18)|(sub['time_frac']<6)).mean())
        sc_rt     = int(sub['st'].nunique())
        sub_s = sub.sort_values('time_frac'); half_n = max(1, n_so_far//2)
        fh = sub_s.head(half_n); lh = sub_s.tail(half_n)
        t1 = float(fh['time_frac'].mean()); t2 = float(lh['time_frac'].mean()); dt_ = t2-t1
        if dt_ > 0.1:
            phi1r = np.radians(float(fh['slat'].mean())); phi2r = np.radians(float(lh['slat'].mean()))
            dphi_r = phi2r-phi1r; dlam_r = np.radians(float(lh['slon'].mean())-float(fh['slon'].mean()))
            a_ = np.sin(dphi_r/2)**2 + np.cos(phi1r)*np.cos(phi2r)*np.sin(dlam_r/2)**2
            cas_vel_rt = float(2*6371.0*np.arcsin(np.sqrt(a_))/dt_)
        else: cas_vel_rt = 0.0
        ef_slope_rt = 0.0
        if n_so_far >= 3:
            try:
                ef_slope_rt = float(LinearRegression().fit(
                    sub['time_frac'].values.reshape(-1,1), sub['mag'].values.astype(float)).coef_[0])
            except Exception: pass
        dpr_max_rt     = float(sub['dpr'].max())     if 'dpr' in sub.columns else 0.0
        dpr_syn_max_rt = float(sub['dpr_syn'].max()) if 'dpr_syn' in sub.columns else 0.0
        sub_idx = set(sub.index.tolist()); SG_rt = DG.subgraph(sub_idx)
        if SG_rt.number_of_edges() > 0:
            try:
                pr_rt = nx.pagerank(SG_rt, alpha=0.85, weight='weight', max_iter=100, tol=1e-5)
                partial_dpr_rt = float(max(pr_rt.values()))
                n_meso_comp_rt = len(list(nx.weakly_connected_components(SG_rt)))
            except Exception: partial_dpr_rt = 0.0; n_meso_comp_rt = 1
        else: partial_dpr_rt = 0.0; n_meso_comp_rt = n_so_far
        rt_snapshots.append({'date': date_val, 'year': year_, 't_h': t_h, 'target': target,
            'count_so_far': n_so_far, 'ef_max_so_far': ef_max, 'hull_rt': hull_rt,
            'cascade_vel_rt': cas_vel_rt, 'ef_slope_rt': ef_slope_rt,
            'dpr_max_rt': dpr_max_rt, 'dpr_syn_max_rt': dpr_syn_max_rt,
            'noct_frac_rt': noct_rt, 'state_count_rt': sc_rt,
            'partial_dpr_rt': partial_dpr_rt, 'n_meso_comp_rt': n_meso_comp_rt,
            'fuzzy_entropy_rt': float(ob_row['fuzzy_entropy']),
            'month_sin': np.sin(2*np.pi*month_/12),
            'month_cos': np.cos(2*np.pi*month_/12)})

rt_df = pd.DataFrame(rt_snapshots)
print(f'RT snapshots: {len(rt_df):,}')

# %%
# Cell 20 -- AUROC temporal holdout (GBM, 5 feature sets, unchanged from v5)
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob>=bins[i])&(y_prob<bins[i+1]) if i<n_bins-1 \
               else (y_prob>=bins[i])&(y_prob<=bins[i+1])
        if mask.sum()>0:
            ece += (mask.sum()/len(y_true))*abs(y_true[mask].mean()-y_prob[mask].mean())
    return ece

FEATURE_SETS = {
    'Count-only':        ['count_so_far'],
    'SPC-like':          ['count_so_far','ef_max_so_far','noct_frac_rt','state_count_rt',
                          'month_sin','month_cos'],
    'Tabular':           ['count_so_far','ef_max_so_far','hull_rt','noct_frac_rt',
                          'state_count_rt','month_sin','month_cos'],
    'Cascade-augmented': ['count_so_far','ef_max_so_far','hull_rt','noct_frac_rt',
                          'state_count_rt','month_sin','month_cos',
                          'cascade_vel_rt','ef_slope_rt','dpr_max_rt'],
    'Cascade-full':      ['count_so_far','ef_max_so_far','hull_rt','noct_frac_rt',
                          'state_count_rt','month_sin','month_cos',
                          'cascade_vel_rt','ef_slope_rt','dpr_max_rt',
                          'partial_dpr_rt','n_meso_comp_rt','dpr_syn_max_rt',
                          'fuzzy_entropy_rt'],
}
years_all = sorted(rt_df['year'].unique())
per_year_auroc = {}
pooled_results = []
for t_h in SNAPSHOT_HOURS:
    sub_t = rt_df[rt_df['t_h']==t_h].copy()
    for fs_name, fs_cols in FEATURE_SETS.items():
        aurocs, briers, pr_aucs, eces = [], [], [], []
        for yr in years_all[2:]:
            train = sub_t[sub_t['year']<yr]; test = sub_t[sub_t['year']==yr]
            if len(test)<5 or test['target'].nunique()<2 or train['target'].nunique()<2: continue
            Xtr = train[fs_cols].fillna(0).values; ytr = train['target'].values
            Xte = test[fs_cols].fillna(0).values;  yte = test['target'].values
            sc = StandardScaler().fit(Xtr)
            try:
                mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=RNG)
                mdl.fit(sc.transform(Xtr), ytr)
                proba = mdl.predict_proba(sc.transform(Xte))[:,1]
            except Exception: continue
            aurocs.append(roc_auc_score(yte, proba))
            briers.append(brier_score_loss(yte, proba))
            pr_aucs.append(average_precision_score(yte, proba))
            eces.append(expected_calibration_error(yte, proba))
            per_year_auroc.setdefault((fs_name, t_h), {})[yr] = roc_auc_score(yte, proba)
        if aurocs:
            rng_b = np.random.RandomState(RNG)
            boot = [np.array(aurocs)[rng_b.randint(0,len(aurocs),len(aurocs))].mean() for _ in range(1000)]
            pooled_results.append({
                't_h': t_h, 'feature_set': fs_name,
                'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)),
                'auroc_ci_lo': float(np.percentile(boot, 2.5)),
                'auroc_ci_hi': float(np.percentile(boot, 97.5)),
                'pr_auc_mean': float(np.mean(pr_aucs)),
                'brier_mean': float(np.mean(briers)), 'ece_mean': float(np.mean(eces)),
                'n_folds': len(aurocs),
            })

results_df = pd.DataFrame(pooled_results)
print('\n=== GBM AUROC (temporal holdout) ===')
pivot = results_df.pivot(index='feature_set', columns='t_h', values='auroc_mean')
print(pivot.round(3).to_string())

print('\n=== Paired Wilcoxon: Cascade-full vs SPC-like ===')
for t_h in SNAPSHOT_HOURS:
    a = per_year_auroc.get(('Cascade-full', t_h), {})
    b = per_year_auroc.get(('SPC-like',     t_h), {})
    common = sorted(set(a)&set(b))
    if len(common)<10: continue
    a_v = np.array([a[y] for y in common]); b_v = np.array([b[y] for y in common])
    try: _, p_w = wilcoxon(a_v, b_v, alternative='greater')
    except Exception: p_w = np.nan
    print(f'  T+{t_h}h: lift={a_v.mean()-b_v.mean():+.3f}  Wilcoxon p={p_w:.3f}  '
          f'{"NOT sig" if p_w>=0.05 else "sig"}')

# %% [markdown]
# ---
# ## Q11 -- One-Page Summary Table

# %%
# Cell 21 -- Summary
cas_t2 = results_df[(results_df['feature_set']=='Cascade-full')&(results_df['t_h']==2)]
spc_t2 = results_df[(results_df['feature_set']=='SPC-like')&(results_df['t_h']==2)]
full_auroc = float(cas_t2['auroc_mean'].iloc[0]) if len(cas_t2)>0 else np.nan
spc_auroc  = float(spc_t2['auroc_mean'].iloc[0]) if len(spc_t2)>0 else np.nan

print('\n' + '='*76)
print('FINAL SUMMARY -- Cascading Supercell Families (v6)')
print('='*76)
rows = [
    ('P1a pre-registered (fast=PDS)',    f'{p_p1a:.2e}', 'FAIL (as expected)'),
    ('P1b exploratory (slow=PDS)',       f'{p_p1b:.2e}',
     'SUPPORTED' if p_p1b < 0.05 else 'NOT supported'),
    ('P2 archetype EF4+ separation',     f'{p_p2:.2e}',
     f'risk-ratio {rr:.1f}x  Cohen h={cohen_h:.2f}'),
    ('P3 Apr 27 bearing + Rayleigh',     f'{p_ray_val:.2e}' if not np.isnan(p_ray_val) else 'N/A',
     f'bearing={circ_mean_deg:.0f}deg'),
    ('Holm P1b adj',                     f'{holm_ps["P1b"]:.2e}', ''),
    ('Holm P2  adj',                     f'{holm_ps["P2"]:.2e}', ''),
    ('FCM PC (floor)',                   f'{fpc:.3f} ({1/FCM_K:.3f})',
     f'z={z_pc:.2f} vs null'),
    ('Optimal K',                        f'{K_BEST}',
     f'archetypes: {", ".join(f"C{l}" for l in range(K_BEST))}'),
    ('OII_retro Spearman',               f'{spr_retro:.3f}', ''),
    ('OII_learned Spearman',             f'{spr_learn:.3f}', ''),
    ('Grazulis EF2+ Spearman',           f'{spr_ef2:.3f}', ''),
    ('Cascade-full T+2h AUROC',          f'{full_auroc:.3f}', ''),
    ('SPC-like T+2h AUROC',              f'{spc_auroc:.3f}', ''),
    ('Cascade vs SPC-like T+2h lift',    f'{full_auroc-spc_auroc:+.3f}', 'see Wilcoxon above'),
]
for name, val, note in rows:
    print(f'  {name:40s} {val:>14s}  {note}')

print('\n--- Outbreak Style Classification (v6 headline result) ---')
for lbl in range(K_BEST):
    n_ = int((outbreaks['kmeans_label']==lbl).sum())
    ef4  = float(outbreaks.loc[outbreaks['kmeans_label']==lbl,'has_ef4plus'].mean())
    qlcs = float(outbreaks.loc[outbreaks['kmeans_label']==lbl,'qlcs_flag'].mean())
    sc_  = float(outbreaks.loc[outbreaks['kmeans_label']==lbl,'supercell_flag'].mean())
    lat_ = float(outbreaks.loc[outbreaks['kmeans_label']==lbl,'region_lat'].mean())
    asp_ = float(outbreaks.loc[outbreaks['kmeans_label']==lbl,'hull_log_aspect'].mean())
    ep_  = float(outbreaks.loc[outbreaks['kmeans_label']==lbl,'n_ncei_episodes'].mean())
    print(f'  C{lbl} (n={n_:4d}) {archetype_labels[lbl][:45]:45s} | '
          f'EF4+={ef4*100:4.1f}% | QLCS-narr={qlcs*100:4.1f}% | '
          f'Cell-narr={sc_*100:4.1f}% | lat={lat_:.1f} | '
          f'hull_asp={asp_:.2f} | n_ep={ep_:.1f}')
print('='*76)

print('\nv6 run complete.')

# %% [markdown]
# ---
# ## Q12 -- Limitations
#
# ### What v6 can identify
#
# With track geometry + NCEI episode grouping, the clustering now has signal to
# distinguish outbreak styles along several meteorological dimensions:
#
# - **Hull elongation** separates squall-line-type events (very elongated footprint)
#   from compact isolated-supercell days.
# - **Track bearing consistency** separates coherent supercell families (all moving NE)
#   from scattered QLCS events (variable bearing).
# - **Geographic centroid (region_lat/lon)** separates Dixie Alley (SE, low lat) from
#   Tornado Alley (central, mid-lat) from Northern Plains (high lat).
# - **n_ncei_episodes** separates single-system events (1-3 episodes) from multi-system
#   synoptic outbreaks (>10 episodes).
# - **Nocturnal fraction** separates overnight MCS-driven events from afternoon supercell
#   peak-heating events.
#
# ### What v6 still cannot identify
#
# - **HP supercell vs. classic supercell** — track geometry is similar; radar reflectivity
#   structure required.
# - **Elevated convection** — requires proximity soundings (LCL/LFC heights, CAPE above
#   boundary layer).
# - **Tropical tornado outbreaks** — partially detectable from region_lat/lon + low EF
#   ratings, but not robustly.
#
# ### Coverage caveats
#
# - **NCEI coverage:** episode features are reliable only 1996-2024 (~59% of outbreak days).
#   Pre-1996 values are median-imputed; clustering findings for these years rely on
#   SPC-derived features only.
# - **Track bearing coverage:** elat/elon are 98-100% complete post-2000, 25-56% pre-1990.
#   `track_coverage` quality flag is included as a clustering feature to handle this.
#
# ### Datasets merged
#
# Two independent NOAA datasets are combined in this analysis:
# 1. **NOAA SPC Tornado Database (1950-2024)** — event-level tornado records with path
#    geometry (slat, slon, elat, elon, len, wid, mag). Primary dataset.
# 2. **NOAA NCEI Storm Events Database (1996-2024)** — storm episode records with
#    EPISODE_ID grouping and narrative descriptions. Provides episode-grouping signal
#    and convective-mode labels not present in the SPC CSV.
