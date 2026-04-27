"""testing_rewrite.py — diagnostic rewrite of the SPC + NCEI merge and clustering bake-off.

Restricts to 1996-2024 (clean NCEI overlap, no median-imputation), validates the
SPC <-> NCEI merge at the event level, then runs a wide clustering bake-off and
a continuum-regression fallback if clustering stays weak.

Run from DATA-Mining/:  python testing_rewrite.py
Reads:  cache/spc_cache.parquet, cache/ncei_cache.parquet
Writes: testing_outputs/{merge_validation_report.txt, clustering_bakeoff.csv,
                         continuum_pca.png, feature_importances.csv}

Plan: ~/.claude/plans/partitioned-knitting-yeti.md
"""
from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, HDBSCAN,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

import skfuzzy as fuzz

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    rng: int = 42
    min_events: int = 6
    year_lo: int = 1996
    year_hi: int = 2024
    radius_km: float = 50.0
    dt_hours: float = 1.0
    time_weight: float = 30.0  # km-per-hour penalty in matching cost
    bootstrap_iters: int = 20
    bootstrap_frac: float = 0.8
    silhouette_floor: float = 0.20  # below this, declare clustering weak
    spc_path: str = "cache/spc_cache.parquet"
    ncei_path: str = "cache/ncei_cache.parquet"
    out_dir: str = "testing_outputs"


CONUS = [
    "AL", "AR", "AZ", "CA", "CO", "CT", "DE", "FL", "GA", "IA", "ID", "IL",
    "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT",
    "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
    "DC",
]

STATE_NAME_TO_CODE = {
    "ALABAMA": "AL", "ARKANSAS": "AR", "ARIZONA": "AZ", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL",
    "GEORGIA": "GA", "IOWA": "IA", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MASSACHUSETTS": "MA", "MARYLAND": "MD", "MAINE": "ME", "MICHIGAN": "MI",
    "MINNESOTA": "MN", "MISSOURI": "MO", "MISSISSIPPI": "MS", "MONTANA": "MT",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "NEBRASKA": "NE",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEVADA": "NV", "NEW YORK": "NY", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN",
    "TEXAS": "TX", "UTAH": "UT", "VIRGINIA": "VA", "VERMONT": "VT",
    "WASHINGTON": "WA", "WISCONSIN": "WI", "WEST VIRGINIA": "WV",
    "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC",
}

QLCS_PATTERN = re.compile(
    r"qlcs|squall.?line|quasi.?linear|bow.?echo|line.of.storms|linear.convect", re.I
)
CELL_PATTERN = re.compile(
    r"supercell|discrete.cell|isolated.cell|rotating.thunder|mesocyclone", re.I
)
MAG_PATTERN = re.compile(r"\b(?:EF|F)([0-5])\b", re.I)


def _normalize_state(name: Any) -> str | None:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip().upper()
    if len(s) == 2 and s in CONUS:
        return s
    return STATE_NAME_TO_CODE.get(s)


def _parse_spc_time_to_frac(t: pd.Series) -> pd.Series:
    """Parse SPC `time` column (string, may be 'HH:MM:SS' or 'HHMM') to fractional hours."""
    s = t.astype(str).str.strip()
    has_colon = s.str.contains(":")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    # HH:MM:SS or HH:MM branch
    parsed = pd.to_datetime(s[has_colon], format="mixed", errors="coerce")
    out.loc[has_colon] = parsed.dt.hour + parsed.dt.minute / 60.0
    # HHMM zero-padded branch
    z = s[~has_colon].str.zfill(4)
    h = pd.to_numeric(z.str[:2], errors="coerce").clip(0, 23)
    m = pd.to_numeric(z.str[2:4], errors="coerce").clip(0, 59)
    out.loc[~has_colon] = h + m / 60.0
    return out.fillna(0.0).clip(0, 24)


def _parse_ncei_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%b-%y %H:%M:%S", errors="coerce")


def _haversine_km(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================================
# Phase A — Load and restrict
# ============================================================================

def phase_a_load(cfg: Config):
    print("\n" + "=" * 76)
    print("PHASE A — Load and restrict to {}-{}".format(cfg.year_lo, cfg.year_hi))
    print("=" * 76)

    spc_raw = pd.read_parquet(cfg.spc_path)
    spc_raw["date"] = pd.to_datetime(spc_raw["date"], errors="coerce")
    spc_raw["year"] = spc_raw["date"].dt.year
    spc_raw["month"] = spc_raw["date"].dt.month

    spc = spc_raw[
        (spc_raw["mag"] >= 0) & (spc_raw["mag"] <= 5)
        & spc_raw["st"].isin(CONUS)
        & spc_raw["date"].notna()
        & spc_raw["slat"].between(-90, 90)
        & spc_raw["slon"].between(-180, 180)
    ].copy()
    spc["time_frac"] = _parse_spc_time_to_frac(spc["time"])
    spc["datetime"] = spc["date"] + pd.to_timedelta(spc["time_frac"], unit="h")
    spc["date_d"] = spc["date"].values.astype("datetime64[D]")
    spc["log_len"] = np.log1p(spc["len"])
    spc["log_wid"] = np.log1p(spc["wid"])

    spc = spc[(spc["year"] >= cfg.year_lo) & (spc["year"] <= cfg.year_hi)].copy()
    spc = spc.sort_values("datetime").reset_index(drop=True)

    ncei = pd.read_parquet(cfg.ncei_path)
    ncei["date_d"] = pd.to_datetime(ncei["date_d"]).values.astype("datetime64[D]")
    ncei["begin_dt"] = _parse_ncei_dt(ncei["BEGIN_DATE_TIME"])
    for col in ["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]:
        ncei[col] = pd.to_numeric(ncei[col], errors="coerce")
    ncei["state_code"] = ncei["STATE"].map(_normalize_state)
    yr = ncei["begin_dt"].dt.year.fillna(pd.to_datetime(ncei["date_d"]).dt.year)
    ncei = ncei[(yr >= cfg.year_lo) & (yr <= cfg.year_hi)].copy()
    ncei = ncei.reset_index(drop=True)

    daily_counts = spc.groupby("date_d").size()
    outbreak_dates = set(daily_counts[daily_counts >= cfg.min_events].index)
    spc_ob = spc[spc["date_d"].isin(outbreak_dates)].copy().reset_index(drop=True)

    print(f"SPC clean events       : {len(spc):,}")
    print(f"Outbreak days (>={cfg.min_events})  : {len(outbreak_dates):,}")
    print(f"SPC events on outbreak : {len(spc_ob):,}")
    print(f"NCEI tornado events    : {len(ncei):,}")
    print(f"NCEI date range        : {ncei['date_d'].min()} -> {ncei['date_d'].max()}")
    print(f"NCEI state-code coverage: {ncei['state_code'].notna().mean()*100:.1f}%")
    print(f"NCEI begin_dt parsed   : {ncei['begin_dt'].notna().mean()*100:.1f}%")

    return spc_ob, ncei, outbreak_dates


# ============================================================================
# Phase B — Merge validation
# ============================================================================

def phase_b_validate(spc_ob: pd.DataFrame, ncei: pd.DataFrame,
                     outbreak_dates: set, cfg: Config) -> dict:
    print("\n" + "=" * 76)
    print("PHASE B — Merge validation (event-level)")
    print("=" * 76)

    rng = np.random.RandomState(cfg.rng)
    mv: dict[str, Any] = {}

    # ------- B.1 Per-day count agreement on intersection -------
    spc_counts = spc_ob.groupby("date_d").size().rename("spc_n")
    ncei_ob = ncei[ncei["date_d"].isin(outbreak_dates)].copy()
    ncei_counts = ncei_ob.groupby("date_d").size().rename("ncei_n")
    cnt = pd.concat([spc_counts, ncei_counts], axis=1, join="inner")
    rho, _ = spearmanr(cnt["spc_n"], cnt["ncei_n"])
    ratio_med = float((cnt["spc_n"] / cnt["ncei_n"].clip(lower=1)).median())
    mv["count_spearman"] = float(rho)
    mv["count_ratio_median"] = ratio_med
    mv["count_n_days"] = int(len(cnt))
    mv["count_pass"] = bool(rho >= 0.85)
    print(f"\n[B.1] Per-day count agreement on {len(cnt)} shared outbreak days:")
    print(f"      Spearman rho = {rho:.3f}        median(SPC/NCEI) = {ratio_med:.2f}")
    print(f"      {'PASS' if mv['count_pass'] else 'FAIL'} (threshold rho >= 0.85)")

    # ------- B.2 Hard invariants -------
    # max episode tornadoes <= SPC count per day
    ep_per_day = (
        ncei_ob.dropna(subset=["EPISODE_ID"]).groupby(["date_d", "EPISODE_ID"]).size()
        .groupby(level=0).max().rename("max_ep_size")
    )
    inv_check = pd.concat([spc_counts, ep_per_day], axis=1, join="inner")
    violations = int((inv_check["max_ep_size"] > inv_check["spc_n"]).sum())
    mv["invariant_violations"] = violations
    mv["invariant_n_days"] = int(len(inv_check))
    mv["invariant_pass"] = bool(violations <= max(1, int(0.01 * len(inv_check))))
    # NCEI dates subset of SPC clean dates check
    spc_all_dates = set(spc_ob["date_d"].unique())
    ncei_dates = set(ncei["date_d"].unique())
    extra_ncei = ncei_dates - spc_all_dates
    mv["ncei_dates_outside_spc_outbreak"] = len(extra_ncei)
    print(f"\n[B.2] Hard invariants:")
    print(f"      max_episode_size > spc_n : {violations} / {len(inv_check)} days "
          f"({'PASS' if mv['invariant_pass'] else 'FAIL'})")
    print(f"      NCEI dates not in SPC clean: {len(extra_ncei):,}  "
          f"(expected: many, since most NCEI tornadoes fall on non-outbreak days)")

    # ------- B.3 Event-level NN match -------
    print("\n[B.3] Event-level nearest-neighbor matching (radius=50km, |dt|<=1h)...")
    matched_pairs = []
    spc_used_global = set()
    shared_days = sorted(set(cnt.index))
    t0 = time.time()
    for d in shared_days:
        spc_d = spc_ob[spc_ob["date_d"] == d]
        ncei_d = ncei_ob[ncei_ob["date_d"] == d]
        ncei_d = ncei_d[ncei_d["BEGIN_LAT"].notna() & ncei_d["BEGIN_LON"].notna()
                        & ncei_d["begin_dt"].notna()]
        if len(spc_d) == 0 or len(ncei_d) == 0:
            continue
        spc_idx = spc_d.index.to_numpy()
        spc_coords = np.radians(spc_d[["slat", "slon"]].to_numpy())
        spc_dt = spc_d["datetime"].to_numpy()
        tree = BallTree(spc_coords, metric="haversine")
        ncei_coords = np.radians(ncei_d[["BEGIN_LAT", "BEGIN_LON"]].to_numpy())
        ncei_dt = ncei_d["begin_dt"].to_numpy()
        radius_rad = cfg.radius_km / 6371.0
        inds = tree.query_radius(ncei_coords, r=radius_rad)
        candidates = []
        for i_n, neighbor_local_idx in enumerate(inds):
            for j_local in neighbor_local_idx:
                spc_global_idx = spc_idx[j_local]
                d_km = _haversine_km(
                    np.degrees(ncei_coords[i_n, 0]), np.degrees(ncei_coords[i_n, 1]),
                    np.degrees(spc_coords[j_local, 0]), np.degrees(spc_coords[j_local, 1]),
                )
                dt_h = (ncei_dt[i_n] - spc_dt[j_local]) / np.timedelta64(1, "h")
                if abs(dt_h) > cfg.dt_hours:
                    continue
                cost = d_km + cfg.time_weight * abs(dt_h)
                candidates.append((cost, d_km, dt_h, i_n, spc_global_idx,
                                   ncei_d.index[i_n]))
        candidates.sort(key=lambda x: x[0])
        ncei_used_local = set()
        for cost, d_km, dt_h, i_n, spc_g, ncei_g in candidates:
            if spc_g in spc_used_global or i_n in ncei_used_local:
                continue
            spc_used_global.add(spc_g)
            ncei_used_local.add(i_n)
            matched_pairs.append({
                "date_d": d, "spc_idx": spc_g, "ncei_idx": ncei_g,
                "dist_km": float(d_km), "dt_hours": float(dt_h),
            })
    matched = pd.DataFrame(matched_pairs)
    n_ncei_eligible = int(
        ((ncei_ob["BEGIN_LAT"].notna()) & (ncei_ob["BEGIN_LON"].notna())
         & (ncei_ob["begin_dt"].notna())).sum()
    )
    match_rate = float(len(matched) / max(1, n_ncei_eligible))
    mv["match_rate"] = match_rate
    mv["match_n_pairs"] = int(len(matched))
    mv["match_n_ncei_eligible"] = n_ncei_eligible
    mv["match_dist_mean_km"] = float(matched["dist_km"].mean()) if len(matched) else None
    mv["match_dt_mean_h"] = float(matched["dt_hours"].abs().mean()) if len(matched) else None
    mv["match_pass"] = bool(match_rate >= 0.80)
    print(f"      Matched {len(matched):,} / {n_ncei_eligible:,} eligible NCEI events "
          f"({match_rate*100:.1f}%)  in {time.time()-t0:.1f}s")
    if len(matched):
        print(f"      Mean dist = {mv['match_dist_mean_km']:.1f} km   "
              f"mean |dt| = {mv['match_dt_mean_h']:.2f} h")
    print(f"      {'PASS' if mv['match_pass'] else 'FAIL'} (threshold >= 80%)")

    # ------- B.3b Magnitude agreement on matched pairs -------
    if len(matched):
        ncei_with_mag = ncei.loc[matched["ncei_idx"]].copy()
        narrative_concat = (
            ncei_with_mag["EVENT_NARRATIVE"].fillna("").astype(str)
            + " " + ncei_with_mag["EPISODE_NARRATIVE"].fillna("").astype(str)
        )
        mag_extracted = narrative_concat.str.extract(MAG_PATTERN)[0]
        mag_extracted = pd.to_numeric(mag_extracted, errors="coerce")
        spc_mag = spc_ob.loc[matched["spc_idx"], "mag"].to_numpy()
        coverage = float(mag_extracted.notna().mean())
        if coverage >= 0.10:
            paired = pd.DataFrame({"spc": spc_mag, "ncei": mag_extracted.values}).dropna()
            agree = float((paired["spc"] == paired["ncei"]).mean())
            close = float((paired["spc"] - paired["ncei"]).abs().le(1).mean())
            mv["mag_regex_coverage"] = coverage
            mv["mag_exact_agreement"] = agree
            mv["mag_within1_agreement"] = close
            print(f"      Magnitude regex coverage = {coverage*100:.1f}%   "
                  f"exact = {agree*100:.1f}%   within-1 = {close*100:.1f}%")
        else:
            mv["mag_regex_coverage"] = coverage
            print(f"      Magnitude regex coverage = {coverage*100:.1f}% < 10%, skipping")

    # ------- B.4 State cross-tab -------
    if len(matched):
        ncei_state = ncei.loc[matched["ncei_idx"], "state_code"].to_numpy()
        spc_state = spc_ob.loc[matched["spc_idx"], "st"].to_numpy()
        mask = pd.Series(ncei_state).notna().to_numpy() & pd.Series(spc_state).notna().to_numpy()
        agree_state = float((ncei_state[mask] == spc_state[mask]).mean()) if mask.any() else 0.0
        mv["state_agreement"] = agree_state
        mv["state_pass"] = bool(agree_state >= 0.90)
        print(f"\n[B.4] State agreement on matched pairs: {agree_state*100:.1f}%   "
              f"({'PASS' if mv['state_pass'] else 'FAIL'} threshold >= 90%)")

    # ------- B.5 Timezone audit -------
    if len(matched):
        dt_h = matched["dt_hours"].to_numpy()
        big = int((np.abs(dt_h) > 12).sum())
        # day-boundary crossers among matched pairs (using same day in matching is enforced,
        # so this is 0 by construction — instead check on UNMATCHED ncei: dates that differ
        # between BEGIN_DATE_TIME-derived date and the cached date_d)
        ncei_self_consistent = (
            ncei["begin_dt"].dt.normalize()
            == pd.to_datetime(ncei["date_d"])
        )
        cross_internal = int(
            (~ncei_self_consistent[ncei["begin_dt"].notna()]).sum()
        )
        mv["tz_dt_hist_p50"] = float(np.median(dt_h))
        mv["tz_dt_hist_p90"] = float(np.percentile(np.abs(dt_h), 90))
        mv["tz_big_offset_count"] = big
        mv["tz_internal_inconsistent_count"] = cross_internal
        print(f"\n[B.5] Timezone audit on matched pairs (NCEI - SPC, hours):")
        print(f"      median dt = {mv['tz_dt_hist_p50']:+.2f}h   "
              f"p90 |dt| = {mv['tz_dt_hist_p90']:.2f}h")
        print(f"      |dt| > 12h count = {big}")
        print(f"      NCEI internal date vs begin_dt mismatch: {cross_internal}")

    # ------- B.6 Narrative regex audit -------
    print("\n[B.6] Narrative regex audit (random samples):")
    ncei_concat_text = (
        ncei["EVENT_NARRATIVE"].fillna("").astype(str)
        + " " + ncei["EPISODE_NARRATIVE"].fillna("").astype(str)
    )
    qlcs_hit = ncei_concat_text.str.contains(QLCS_PATTERN)
    cell_hit = ncei_concat_text.str.contains(CELL_PATTERN)
    mv["narrative_qlcs_rate"] = float(qlcs_hit.mean())
    mv["narrative_supercell_rate"] = float(cell_hit.mean())
    mv["narrative_cooccurrence_rate"] = float((qlcs_hit & cell_hit).mean())
    print(f"      QLCS regex hit rate     : {qlcs_hit.mean()*100:.1f}%")
    print(f"      Supercell regex hit rate: {cell_hit.mean()*100:.1f}%")
    print(f"      Co-occurrence rate      : {(qlcs_hit & cell_hit).mean()*100:.1f}%")

    def _print_narrative_sample(label, mask, n=5):
        idx_pool = np.where(mask.to_numpy())[0]
        if len(idx_pool) == 0:
            print(f"      [{label}] no rows.")
            return
        sample = rng.choice(idx_pool, size=min(n, len(idx_pool)), replace=False)
        for i in sample:
            text = ncei_concat_text.iloc[i].replace("\n", " ").strip()
            print(f"      [{label}] {text[:180]}")

    print("    -- 5 random QLCS HITS --")
    _print_narrative_sample("QLCS-HIT ", qlcs_hit, n=5)
    print("    -- 5 random QLCS MISSES (with non-empty narrative) --")
    _print_narrative_sample("QLCS-MISS", (~qlcs_hit) & (ncei_concat_text.str.len() > 50), n=5)
    print("    -- 5 random SUPERCELL HITS --")
    _print_narrative_sample("CELL-HIT ", cell_hit, n=5)

    # ------- B-Summary -------
    mv["all_pass"] = bool(
        mv.get("count_pass", False)
        and mv.get("invariant_pass", False)
        and mv.get("match_pass", False)
        and mv.get("state_pass", False)
    )
    print("\n[B] OVERALL: " + ("PASS" if mv["all_pass"] else "PARTIAL — see details"))
    return mv, matched


# ============================================================================
# Phase C — Feature build (no median imputation)
# ============================================================================

def _convex_hull_area_km2(lats, lons):
    if len(lats) < 3:
        return 0.0
    lat_c = float(np.mean(lats))
    x = (np.asarray(lons) - np.mean(lons)) * 111.0 * np.cos(np.radians(lat_c))
    y = (np.asarray(lats) - np.mean(lats)) * 111.0
    pts = np.unique(np.column_stack([x, y]), axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def _convex_hull_aspect_pca(lats, lons):
    if len(lats) < 4:
        return 0.0, 0.0, 1.0
    lat_c = float(np.mean(lats))
    x = (np.asarray(lons) - np.mean(lons)) * 111.0 * np.cos(np.radians(lat_c))
    y = (np.asarray(lats) - np.mean(lats)) * 111.0
    pts = np.unique(np.column_stack([x, y]), axis=0)
    if len(pts) < 4:
        return 0.0, 0.0, 1.0
    try:
        hull = ConvexHull(pts)
        pca = PCA(n_components=2).fit(pts[hull.vertices])
        ev = pca.explained_variance_ratio_
        log_ar = float(np.log1p(ev[0] / (ev[1] + 1e-9)))
        comp = pca.components_[0]
        orient_deg = (np.degrees(np.arctan2(comp[0], comp[1])) + 360) % 360
        return (
            log_ar,
            float(np.sin(np.radians(orient_deg))),
            float(np.cos(np.radians(orient_deg))),
        )
    except Exception:
        return 0.0, 0.0, 1.0


def phase_c_features(spc_ob: pd.DataFrame, ncei: pd.DataFrame, cfg: Config):
    print("\n" + "=" * 76)
    print("PHASE C — Feature engineering (no imputation)")
    print("=" * 76)
    t0 = time.time()

    # NCEI per-day stats (no imputation; NaN for days without NCEI rows)
    ncei_concat = (
        ncei["EVENT_NARRATIVE"].fillna("").astype(str)
        + " " + ncei["EPISODE_NARRATIVE"].fillna("").astype(str)
    )
    ncei = ncei.assign(
        _qlcs=ncei_concat.str.contains(QLCS_PATTERN).astype(int),
        _cell=ncei_concat.str.contains(CELL_PATTERN).astype(int),
    )
    ncei_days = []
    for d, g in ncei.groupby("date_d"):
        ep = g.dropna(subset=["EPISODE_ID"]).groupby("EPISODE_ID").size()
        n_ep = int(len(ep))
        max_ep = int(ep.max()) if n_ep else 0
        cv = float(ep.std(ddof=0) / (ep.mean() + 1e-9)) if n_ep else 0.0
        ncei_days.append({
            "date": d, "n_ncei_episodes": n_ep,
            "max_episode_tornadoes": max_ep,
            "episode_size_cv": cv,
            "qlcs_flag": int(g["_qlcs"].max()),
            "supercell_flag": int(g["_cell"].max()),
        })
    ncei_day_df = pd.DataFrame(ncei_days)

    # Per-outbreak-day SPC features
    rows = []
    for date_val, day_df in spc_ob.groupby("date_d"):
        n = len(day_df)
        ev_count = n
        fat_total = int(day_df["fat"].sum())
        inj_total = int(day_df["inj"].sum())
        max_mag = int(day_df["mag"].max())
        sig_frac = float((day_df["mag"] >= 2).mean())
        ef2plus_ct = int((day_df["mag"] >= 2).sum())
        violent_ct = int((day_df["mag"] >= 4).sum())
        mean_log_len = float(day_df["log_len"].mean())
        noct_frac = float(((day_df["time_frac"] >= 18) | (day_df["time_frac"] < 6)).mean())
        state_count = int(day_df["st"].nunique())
        hull_area = _convex_hull_area_km2(day_df["slat"].values, day_df["slon"].values)
        month_val = int(day_df["month"].iloc[0])
        year_val = int(day_df["year"].iloc[0])
        duration_h = float(day_df["time_frac"].max() - day_df["time_frac"].min())
        has_ef4 = int((day_df["mag"] >= 4).any())

        # Track bearings (need elat/elon valid)
        valid_end = (
            (day_df["elat"] != 0) & (day_df["elon"] != 0)
            & day_df["elat"].notna() & day_df["elon"].notna()
        )
        track_coverage = float(valid_end.mean())
        if valid_end.sum() >= 3:
            slv = day_df.loc[valid_end, "slat"].values
            slv2 = day_df.loc[valid_end, "slon"].values
            elv = day_df.loc[valid_end, "elat"].values
            elv2 = day_df.loc[valid_end, "elon"].values
            phi1 = np.radians(slv); phi2 = np.radians(elv)
            dlam = np.radians(elv2 - slv2)
            y_b = np.sin(dlam) * np.cos(phi2)
            x_b = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
            bearings = (np.degrees(np.arctan2(y_b, x_b)) + 360) % 360
            C = float(np.mean(np.cos(np.radians(bearings))))
            S = float(np.mean(np.sin(np.radians(bearings))))
            R = float(np.sqrt(C ** 2 + S ** 2))
            tb_mean = float((np.degrees(np.arctan2(S, C)) + 360) % 360)
            tb_std = float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(R + 1e-9)))))
            tb_sin = float(np.sin(np.radians(tb_mean)))
            tb_cos = float(np.cos(np.radians(tb_mean)))
        else:
            R = 0.0; tb_std = 180.0; tb_sin = 0.0; tb_cos = 0.0

        hull_log_asp, hull_os, hull_oc = _convex_hull_aspect_pca(
            day_df["slat"].values, day_df["slon"].values
        )
        region_lat = float(day_df["slat"].mean())
        region_lon = float(day_df["slon"].mean())
        tvals = day_df["time_frac"].values
        timing_iqr = float(np.percentile(tvals, 75) - np.percentile(tvals, 25)) if n >= 4 else 0.0
        lens_v = day_df["len"].values
        track_len_cv = float(lens_v.std(ddof=0) / (lens_v.mean() + 1e-9)) if n >= 4 else 0.0
        track_wid_mean = float(day_df["wid"].mean())

        if n >= 4:
            try:
                ef_slope = float(LinearRegression().fit(
                    day_df["time_frac"].values.reshape(-1, 1),
                    day_df["mag"].values.astype(float),
                ).coef_[0])
            except Exception:
                ef_slope = 0.0
        else:
            ef_slope = 0.0

        rows.append({
            "date": date_val, "year": year_val, "month": month_val,
            "month_sin": float(np.sin(2 * np.pi * month_val / 12)),
            "month_cos": float(np.cos(2 * np.pi * month_val / 12)),
            "event_count": ev_count, "fatality_total": fat_total,
            "injury_total": inj_total, "max_mag": max_mag,
            "sig_fraction": sig_frac, "ef2plus_count": ef2plus_ct,
            "violent_count": violent_ct, "mean_log_len": mean_log_len,
            "nocturnal_fraction": noct_frac, "state_count": state_count,
            "hull_area_km2": hull_area, "duration_hours": duration_h,
            "track_bearing_sin": tb_sin, "track_bearing_cos": tb_cos,
            "track_bearing_R": R, "track_bearing_std": tb_std,
            "track_coverage": track_coverage,
            "hull_log_aspect": hull_log_asp,
            "hull_orient_sin": hull_os, "hull_orient_cos": hull_oc,
            "region_lat": region_lat, "region_lon": region_lon,
            "timing_iqr_hours": timing_iqr,
            "track_len_cv": track_len_cv, "track_wid_mean": track_wid_mean,
            "ef_slope_per_hour": ef_slope,
            "has_ef4plus": has_ef4,
        })

    outbreaks = pd.DataFrame(rows)
    outbreaks = outbreaks.merge(ncei_day_df, on="date", how="left")
    n_pre = len(outbreaks)
    outbreaks = outbreaks.dropna(subset=["n_ncei_episodes"]).reset_index(drop=True)
    n_post = len(outbreaks)
    print(f"Outbreak days engineered: {n_pre:,}")
    print(f"Dropped {n_pre - n_post:,} days with no NCEI coverage; kept {n_post:,}.")

    full_feats = [
        "event_count", "max_mag", "sig_fraction", "ef2plus_count", "violent_count",
        "mean_log_len", "nocturnal_fraction", "state_count", "hull_area_km2",
        "duration_hours", "track_bearing_sin", "track_bearing_cos", "track_bearing_R",
        "track_bearing_std", "track_coverage", "hull_log_aspect",
        "hull_orient_sin", "hull_orient_cos", "region_lat", "region_lon",
        "timing_iqr_hours", "track_len_cv", "track_wid_mean", "ef_slope_per_hour",
        "month_sin", "month_cos",
        "n_ncei_episodes", "max_episode_tornadoes", "episode_size_cv",
    ]
    geometry_feats = [
        "hull_log_aspect", "hull_orient_sin", "hull_orient_cos",
        "track_bearing_R", "track_bearing_std",
        "region_lat", "region_lon", "timing_iqr_hours", "track_coverage",
    ]
    tabular_feats = [
        "event_count", "max_mag", "hull_area_km2", "nocturnal_fraction",
        "state_count", "mean_log_len", "duration_hours",
        "n_ncei_episodes", "max_episode_tornadoes",
    ]

    subsets = {"FULL": full_feats, "GEOMETRY": geometry_feats, "TABULAR": tabular_feats}
    X_full = StandardScaler().fit_transform(outbreaks[full_feats].values)
    print(f"Feature matrix (FULL): {X_full.shape}    "
          f"NaN cells = {int(np.isnan(X_full).sum())}")
    print(f"Feature subsets: " + ", ".join(f"{k}={len(v)}" for k, v in subsets.items()))
    print(f"Phase C runtime: {time.time()-t0:.1f}s")
    return outbreaks, X_full, full_feats, subsets


# ============================================================================
# Phase D — Clustering bake-off
# ============================================================================

def _cluster_metrics(X, labels):
    valid = labels[labels >= 0]
    if len(valid) < 2:
        return None
    uniq = np.unique(valid)
    if len(uniq) < 2:
        return None
    mask = labels >= 0
    if mask.sum() < 5 or len(np.unique(labels[mask])) < 2:
        return None
    try:
        sil = float(silhouette_score(X[mask], labels[mask]))
        db = float(davies_bouldin_score(X[mask], labels[mask]))
        ch = float(calinski_harabasz_score(X[mask], labels[mask]))
    except Exception:
        return None
    sizes = pd.Series(labels[mask]).value_counts().sort_values(ascending=False).to_numpy()
    return {
        "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
        "min_cluster_n": int(sizes.min()),
        "max_cluster_n": int(sizes.max()),
        "size_balance": float(sizes.min() / sizes.max()),
        "sizes": ",".join(str(int(s)) for s in sizes),
    }


def _bootstrap_ari(fit_fn, X, n_iters: int, frac: float, rng_seed: int):
    rng = np.random.RandomState(rng_seed)
    n = len(X)
    sample_size = max(20, int(frac * n))
    base_labels = fit_fn(X)
    if base_labels is None:
        return float("nan")
    aris = []
    for _ in range(n_iters):
        idx = rng.choice(n, size=sample_size, replace=True)
        sub_labels = fit_fn(X[idx])
        if sub_labels is None:
            continue
        try:
            aris.append(adjusted_rand_score(base_labels[idx], sub_labels))
        except Exception:
            continue
    return float(np.mean(aris)) if aris else float("nan")


def phase_d_bakeoff(outbreaks: pd.DataFrame, X_full: np.ndarray,
                    full_feats: list, subsets: dict, cfg: Config):
    print("\n" + "=" * 76)
    print("PHASE D — Clustering bake-off")
    print("=" * 76)

    results = []

    def add(method, subset, params, labels, fit_fn=None, X_for_metrics=None):
        Xm = X_for_metrics if X_for_metrics is not None else X_for_subset
        m = _cluster_metrics(Xm, labels)
        if m is None:
            return
        n_clusters = int(len(np.unique(labels[labels >= 0])))
        noise_pct = float((labels == -1).mean() * 100)
        ari = (
            _bootstrap_ari(fit_fn, Xm, cfg.bootstrap_iters, cfg.bootstrap_frac, cfg.rng)
            if fit_fn is not None else float("nan")
        )
        results.append({
            "method": method, "subset": subset, "params": params,
            "n_clusters": n_clusters, "noise_pct": noise_pct,
            "min_cluster_n": m["min_cluster_n"],
            "max_cluster_n": m["max_cluster_n"],
            "size_balance": m["size_balance"],
            "silhouette": m["silhouette"],
            "davies_bouldin": m["davies_bouldin"],
            "calinski_harabasz": m["calinski_harabasz"],
            "bootstrap_ari": ari,
            "sizes": m["sizes"],
        })

    for subset_name, feats in subsets.items():
        print(f"\n--- subset = {subset_name} ({len(feats)} feats) ---")
        X_for_subset = StandardScaler().fit_transform(outbreaks[feats].values)
        n = len(X_for_subset)
        t_subset = time.time()

        # KMeans K=2..8
        for k in range(2, 9):
            km = KMeans(n_clusters=k, n_init=10, random_state=cfg.rng).fit(X_for_subset)
            add(
                "KMeans", subset_name, f"k={k}", km.labels_,
                fit_fn=lambda Xs, k=k: KMeans(n_clusters=k, n_init=5, random_state=cfg.rng).fit(Xs).labels_,
            )

        # HDBSCAN
        for mcs in (15, 30, 50, 80):
            try:
                clu = HDBSCAN(min_cluster_size=mcs).fit(X_for_subset)
                add(
                    "HDBSCAN", subset_name, f"mcs={mcs}", clu.labels_,
                    fit_fn=lambda Xs, mcs=mcs: HDBSCAN(min_cluster_size=mcs).fit(Xs).labels_,
                )
            except Exception as e:
                print(f"  HDBSCAN mcs={mcs} failed: {e}")

        # DBSCAN — eps from k-distance plot knee
        try:
            kk = max(4, 2 * len(feats))
            tree = BallTree(X_for_subset)
            dists, _ = tree.query(X_for_subset, k=kk + 1)
            kdist = np.sort(dists[:, kk])
            # Knee = max curvature point, use 90th percentile as a pragmatic eps
            eps = float(kdist[int(0.9 * len(kdist))])
            clu = DBSCAN(eps=eps, min_samples=kk).fit(X_for_subset)
            add(
                "DBSCAN", subset_name, f"eps={eps:.2f},ms={kk}", clu.labels_,
                fit_fn=lambda Xs, eps=eps, kk=kk: DBSCAN(eps=eps, min_samples=kk).fit(Xs).labels_,
            )
        except Exception as e:
            print(f"  DBSCAN failed: {e}")

        # Agglomerative
        for linkage in ("ward", "average", "complete"):
            for k in (3, 4, 5):
                try:
                    ag = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(X_for_subset)
                    add(
                        "Agglom", subset_name, f"link={linkage},k={k}", ag.labels_,
                        fit_fn=lambda Xs, link=linkage, k=k: AgglomerativeClustering(
                            n_clusters=k, linkage=link).fit(Xs).labels_,
                    )
                except Exception as e:
                    print(f"  Agglom {linkage} k={k} failed: {e}")

        # GMM with BIC
        for k in range(2, 9):
            try:
                gmm = GaussianMixture(n_components=k, random_state=cfg.rng, n_init=3).fit(X_for_subset)
                labels = gmm.predict(X_for_subset)
                m = _cluster_metrics(X_for_subset, labels)
                if m is None:
                    continue
                results.append({
                    "method": "GMM", "subset": subset_name,
                    "params": f"k={k},bic={gmm.bic(X_for_subset):.0f}",
                    "n_clusters": int(k), "noise_pct": 0.0,
                    "min_cluster_n": m["min_cluster_n"],
                    "max_cluster_n": m["max_cluster_n"],
                    "size_balance": m["size_balance"],
                    "silhouette": m["silhouette"],
                    "davies_bouldin": m["davies_bouldin"],
                    "calinski_harabasz": m["calinski_harabasz"],
                    "bootstrap_ari": _bootstrap_ari(
                        lambda Xs, k=k: GaussianMixture(n_components=k, random_state=cfg.rng).fit(Xs).predict(Xs),
                        X_for_subset, cfg.bootstrap_iters, cfg.bootstrap_frac, cfg.rng,
                    ),
                    "sizes": m["sizes"],
                })
            except Exception as e:
                print(f"  GMM k={k} failed: {e}")

        # Spectral (subsample if large)
        Xs_use = X_for_subset
        if len(Xs_use) > 1000:
            sub_idx = np.random.RandomState(cfg.rng).choice(len(Xs_use), 1000, replace=False)
            Xs_use = X_for_subset[sub_idx]
        for k in range(2, 7):
            try:
                sp = SpectralClustering(
                    n_clusters=k, affinity="nearest_neighbors", n_neighbors=10,
                    random_state=cfg.rng, assign_labels="kmeans",
                ).fit(Xs_use)
                m = _cluster_metrics(Xs_use, sp.labels_)
                if m is None:
                    continue
                results.append({
                    "method": "Spectral", "subset": subset_name, "params": f"k={k}",
                    "n_clusters": int(k), "noise_pct": 0.0,
                    "min_cluster_n": m["min_cluster_n"],
                    "max_cluster_n": m["max_cluster_n"],
                    "size_balance": m["size_balance"],
                    "silhouette": m["silhouette"],
                    "davies_bouldin": m["davies_bouldin"],
                    "calinski_harabasz": m["calinski_harabasz"],
                    "bootstrap_ari": float("nan"),
                    "sizes": m["sizes"],
                })
            except Exception as e:
                print(f"  Spectral k={k} failed: {e}")

        # Fuzzy C-Means
        for k in range(2, 6):
            try:
                _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                    X_for_subset.T, c=k, m=2.0, error=1e-4, maxiter=300, seed=cfg.rng,
                )
                labels = u.argmax(axis=0)
                m = _cluster_metrics(X_for_subset, labels)
                if m is None:
                    continue
                results.append({
                    "method": "FCM", "subset": subset_name,
                    "params": f"k={k},pc={fpc:.3f}",
                    "n_clusters": int(k), "noise_pct": 0.0,
                    "min_cluster_n": m["min_cluster_n"],
                    "max_cluster_n": m["max_cluster_n"],
                    "size_balance": m["size_balance"],
                    "silhouette": m["silhouette"],
                    "davies_bouldin": m["davies_bouldin"],
                    "calinski_harabasz": m["calinski_harabasz"],
                    "bootstrap_ari": float("nan"),
                    "sizes": m["sizes"],
                })
            except Exception as e:
                print(f"  FCM k={k} failed: {e}")

        # HDBSCAN on PCA(10)
        try:
            n_comp = min(10, len(feats))
            X_pca = PCA(n_components=n_comp, random_state=cfg.rng).fit_transform(X_for_subset)
            for mcs in (15, 30):
                clu = HDBSCAN(min_cluster_size=mcs).fit(X_pca)
                m = _cluster_metrics(X_pca, clu.labels_)
                if m is None:
                    continue
                results.append({
                    "method": "HDBSCAN+PCA", "subset": subset_name,
                    "params": f"mcs={mcs},pca={n_comp}",
                    "n_clusters": int(len(np.unique(clu.labels_[clu.labels_ >= 0]))),
                    "noise_pct": float((clu.labels_ == -1).mean() * 100),
                    "min_cluster_n": m["min_cluster_n"],
                    "max_cluster_n": m["max_cluster_n"],
                    "size_balance": m["size_balance"],
                    "silhouette": m["silhouette"],
                    "davies_bouldin": m["davies_bouldin"],
                    "calinski_harabasz": m["calinski_harabasz"],
                    "bootstrap_ari": float("nan"),
                    "sizes": m["sizes"],
                })
        except Exception as e:
            print(f"  HDBSCAN+PCA failed: {e}")

        print(f"  subset {subset_name} runtime: {time.time()-t_subset:.1f}s")

    bake = pd.DataFrame(results).sort_values("silhouette", ascending=False).reset_index(drop=True)
    # Mark as "non-degenerate" only if every cluster has >= 30 points AND
    # noise share is <= 50%. Tiny outlier clusters can inflate silhouette without
    # representing real structure.
    bake["non_degenerate"] = (bake["min_cluster_n"] >= 30) & (bake["noise_pct"] <= 50.0)
    print("\n--- Top 12 bake-off results by silhouette (all rows) ---")
    cols = ["method", "subset", "params", "n_clusters", "min_cluster_n",
            "noise_pct", "silhouette", "bootstrap_ari", "non_degenerate"]
    print(bake[cols].head(12).round(3).to_string(index=False))
    nd = bake[bake["non_degenerate"]].head(12)
    print(f"\n--- Top 12 NON-DEGENERATE results (min_cluster_n >= 30, noise <= 50%) ---")
    if len(nd) == 0:
        print("  (none — all clusterings are degenerate by these thresholds)")
    else:
        print(nd[cols + ["sizes"]].round(3).to_string(index=False))
    return bake


# ============================================================================
# Phase E — Verdict + continuum fallback
# ============================================================================

def phase_e_verdict_or_continuum(outbreaks: pd.DataFrame, X_full: np.ndarray,
                                 full_feats: list, bake: pd.DataFrame, cfg: Config):
    print("\n" + "=" * 76)
    print("PHASE E — Verdict")
    print("=" * 76)

    best_overall = float(bake["silhouette"].max())
    nd = bake[bake["non_degenerate"]]
    best_nd = float(nd["silhouette"].max()) if len(nd) else float("nan")
    print(f"Best silhouette overall          : {best_overall:.4f} ({len(bake)} combos)")
    print(f"Best silhouette non-degenerate    : {best_nd:.4f} ({len(nd)} combos passed filter)")
    print(f"Threshold for 'usable clustering' : silhouette >= {cfg.silhouette_floor}")

    verdict = {
        "best_silhouette_overall": best_overall,
        "best_silhouette_non_degenerate": best_nd,
        "threshold": cfg.silhouette_floor,
        "n_non_degenerate": int(len(nd)),
    }
    if len(nd) and best_nd >= cfg.silhouette_floor:
        print("\n=== CLUSTERING IS USABLE (non-degenerate split passes threshold) ===")
        verdict["recommendation"] = "keep_clustering"
        verdict["best_row"] = nd.iloc[0].to_dict()
        return verdict

    print("\n=== CLUSTERING IS NOT RECOMMENDED ===")
    print("Falling back to continuum + regression analysis.")
    verdict["recommendation"] = "drop_clustering_pivot_to_regression"

    # Continuum visualization
    pca2 = PCA(n_components=2, random_state=cfg.rng).fit(X_full)
    Z = pca2.transform(X_full)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc1 = axes[0].scatter(Z[:, 0], Z[:, 1], c=outbreaks["has_ef4plus"],
                          cmap="coolwarm", s=8, alpha=0.6)
    axes[0].set_title("Continuum PCA — colored by has_ef4plus")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(sc1, ax=axes[0])
    fat = np.log1p(outbreaks["fatality_total"].values)
    sc2 = axes[1].scatter(Z[:, 0], Z[:, 1], c=fat, cmap="viridis", s=8, alpha=0.6)
    axes[1].set_title("Continuum PCA — colored by log1p(fatality_total)")
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    plt.colorbar(sc2, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "continuum_pca.png"), dpi=120)
    plt.close(fig)
    pca_var = pca2.explained_variance_ratio_
    verdict["pca_var_explained"] = pca_var.tolist()
    print(f"Saved continuum_pca.png (PC1 var={pca_var[0]:.2%}, PC2 var={pca_var[1]:.2%}).")

    # Per-feature Spearman vs targets
    rows = []
    y_fat = outbreaks["fatality_total"].values
    y_ef4 = outbreaks["has_ef4plus"].values
    for f in full_feats:
        v = outbreaks[f].values
        rho_fat, _ = spearmanr(v, y_fat)
        rho_ef4, _ = spearmanr(v, y_ef4)
        rows.append({"feature": f, "spearman_vs_fatality": float(rho_fat),
                     "spearman_vs_ef4": float(rho_ef4)})
    rho_df = (
        pd.DataFrame(rows)
        .assign(absrho=lambda d: d["spearman_vs_fatality"].abs())
        .sort_values("absrho", ascending=False).drop(columns="absrho")
    )
    print("\nTop-10 features by |Spearman vs. fatality_total|:")
    print(rho_df.head(10).round(3).to_string(index=False))

    # Ridge + GBR with 5-fold CV predicting log1p(fatality)
    y = np.log1p(y_fat)
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.rng)
    ridge_r2, gbr_r2 = [], []
    gbr_importances = np.zeros(len(full_feats))
    for tr, te in kf.split(X_full):
        ridge = Ridge(alpha=1.0).fit(X_full[tr], y[tr])
        ridge_r2.append(float(ridge.score(X_full[te], y[te])))
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=cfg.rng
        ).fit(X_full[tr], y[tr])
        gbr_r2.append(float(gbr.score(X_full[te], y[te])))
        gbr_importances += gbr.feature_importances_
    gbr_importances /= 5.0
    print(f"\nRidge   CV R^2 (mean +/- std): {np.mean(ridge_r2):+.3f} +/- {np.std(ridge_r2):.3f}")
    print(f"GBR     CV R^2 (mean +/- std): {np.mean(gbr_r2):+.3f} +/- {np.std(gbr_r2):.3f}")
    imp_df = pd.DataFrame({
        "feature": full_feats, "gbr_importance": gbr_importances,
    }).sort_values("gbr_importance", ascending=False)
    imp_df = imp_df.merge(rho_df, on="feature", how="left")
    print("\nTop-10 GBR feature importances:")
    print(imp_df.head(10).round(3).to_string(index=False))

    imp_df.to_csv(os.path.join(cfg.out_dir, "feature_importances.csv"), index=False)

    verdict.update({
        "ridge_cv_r2_mean": float(np.mean(ridge_r2)),
        "ridge_cv_r2_std": float(np.std(ridge_r2)),
        "gbr_cv_r2_mean": float(np.mean(gbr_r2)),
        "gbr_cv_r2_std": float(np.std(gbr_r2)),
        "top_gbr_features": imp_df["feature"].head(10).tolist(),
    })
    return verdict


# ============================================================================
# Phase F — Save artifacts
# ============================================================================

def phase_f_save(mv: dict, bake: pd.DataFrame, verdict: dict, cfg: Config):
    print("\n" + "=" * 76)
    print("PHASE F — Save artifacts")
    print("=" * 76)
    os.makedirs(cfg.out_dir, exist_ok=True)

    bake_path = os.path.join(cfg.out_dir, "clustering_bakeoff.csv")
    bake.to_csv(bake_path, index=False)
    print(f"Wrote {bake_path}  ({len(bake)} rows)")

    mv_path = os.path.join(cfg.out_dir, "merge_validation_report.txt")
    lines = ["MERGE_VALIDATION REPORT", "=" * 60, ""]
    for k, v in mv.items():
        if isinstance(v, float):
            lines.append(f"{k:40s} = {v:.4f}")
        else:
            lines.append(f"{k:40s} = {v}")
    lines.append("")
    lines.append("VERDICT")
    lines.append("=" * 60)
    for k, v in verdict.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        else:
            lines.append(f"{k:40s} = {v}")
    with open(mv_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {mv_path}")


# ============================================================================
# main
# ============================================================================

def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    spc_ob, ncei, outbreak_dates = phase_a_load(cfg)
    mv, matched = phase_b_validate(spc_ob, ncei, outbreak_dates, cfg)
    outbreaks, X_full, full_feats, subsets = phase_c_features(spc_ob, ncei, cfg)

    # Hard guards (fail loudly only on truly broken assumptions)
    assert len(outbreaks) > 1500, f"Expected >1500 outbreak days, got {len(outbreaks)}"
    assert not pd.DataFrame(X_full, columns=full_feats).isna().any().any(), "NaNs in feature matrix"
    if not mv["invariant_pass"]:
        print(f"\n[!] WARNING: hard invariant has {mv['invariant_violations']} "
              f"/ {mv['invariant_n_days']} day violations — see merge_validation_report.txt")

    bake = phase_d_bakeoff(outbreaks, X_full, full_feats, subsets, cfg)
    verdict = phase_e_verdict_or_continuum(outbreaks, X_full, full_feats, bake, cfg)
    phase_f_save(mv, bake, verdict, cfg)

    print("\n" + "=" * 76)
    print("DONE")
    print("=" * 76)
    print(f"Recommendation: {verdict.get('recommendation')}")
    print(f"Best silhouette overall        : "
          f"{verdict.get('best_silhouette_overall', float('nan')):.4f}")
    print(f"Best silhouette non-degenerate  : "
          f"{verdict.get('best_silhouette_non_degenerate', float('nan')):.4f}  "
          f"(threshold {cfg.silhouette_floor})")


if __name__ == "__main__":
    main()
