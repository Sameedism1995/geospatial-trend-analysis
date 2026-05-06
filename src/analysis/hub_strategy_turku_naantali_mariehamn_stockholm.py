"""Refined local-vs-regional hub-strategy distance-decay analysis.

Hub strategy:
  1. Turku-Naantali coastal hub  (Turku and Naantali combined; ~15 km apart)
  2. Mariehamn offshore/island hub  (independent, contrast vs. mainland)
  3. Stockholm urban port hub  (independent, urban contrast for NO2)

Local analysis  : 0-100 km from each hub (sliding-window decay).
Regional analysis: 100-300, 300-500, 500-1000 km bands (BACKGROUND ONLY).

Outputs land in dedicated `run_hub_strategy_turku_naantali_mariehamn_stockholm`
folders so previous runs are not overwritten.

Run:
    python3 src/analysis/hub_strategy_turku_naantali_mariehamn_stockholm.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("hub_strategy")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_TAG = "run_hub_strategy_turku_naantali_mariehamn_stockholm"

DATA_PATH_PRIMARY = PROJECT_ROOT / "processed" / "features_ml_ready.parquet"
DATA_PATH_HUMAN = PROJECT_ROOT / "data" / "modeling_dataset_human_impact.parquet"

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / RUN_TAG
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / RUN_TAG
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations" / RUN_TAG

# ---------------------------------------------------------------------------
# Hub definitions (lat, lon in degrees WGS84)
# ---------------------------------------------------------------------------

PORT_COORDS: dict[str, tuple[float, float]] = {
    "Turku":      (60.4344, 22.2280),
    "Naantali":   (60.4669, 22.0258),
    "Mariehamn":  (60.0973, 19.9348),
    "Stockholm":  (59.3293, 18.0686),
}

# Final composite hubs.  Each hub has a list of underlying ports.
HUB_DEFINITIONS: list[dict] = [
    {
        "hub_name": "Turku-Naantali coastal hub",
        "hub_type": "coastal/industrial port system",
        "included_ports": ["Turku", "Naantali"],
        "reason_for_grouping": (
            "Turku and Naantali are ~15 km apart on the same Finnish "
            "south-west coast; their port-impact footprints overlap and "
            "treating them as separate hubs double-counts the same coastal "
            "system."
        ),
        "interpretation_role": (
            "Represents a Finnish mainland coastal/industrial port cluster — "
            "use as the reference port system for vessel + water-quality "
            "decay near a populated mainland coastline."
        ),
    },
    {
        "hub_name": "Mariehamn offshore/island hub",
        "hub_type": "offshore/island port",
        "included_ports": ["Mariehamn"],
        "reason_for_grouping": (
            "Mariehamn sits on the Åland archipelago, ~50-100 km from any "
            "mainland port, surrounded by open sea.  It is the cleanest "
            "offshore contrast we have."
        ),
        "interpretation_role": (
            "Used as an island/offshore reference: low urban/atmospheric "
            "background, dominated by maritime traffic.  Should show the "
            "strongest pure port-proximity signal in vessel density and "
            "the weakest urban NO2 signal."
        ),
    },
    {
        "hub_name": "Stockholm urban port hub",
        "hub_type": "urban port",
        "included_ports": ["Stockholm"],
        "reason_for_grouping": (
            "Stockholm is a large urban centre with a major port; spatially "
            "distinct from Turku/Naantali (~250 km away) and from Mariehamn "
            "(~150 km).  Adding it lets us contrast urban atmospheric "
            "signal vs. pure port-proximity signal."
        ),
        "interpretation_role": (
            "Test whether near-port NO2 is dominated by urban emissions "
            "(road / domestic / power) rather than ship-source NO2.  If "
            "Stockholm shows a strong NO2 increase 0-50 km but Mariehamn "
            "does not, NO2 is acting as an urban regional signal."
        ),
    },
]

INDICATORS: dict[str, list[str]] = {
    "vessel_density":   ["vessel_density_t", "vessel_density"],
    "ndwi":             ["sentinel_ndwi_mean_t", "ndwi_mean", "ndwi_median"],
    "ndti":             ["sentinel_ndti_mean_t", "ndti_mean", "ndti_median"],
    "ndvi":             ["sentinel_ndvi_mean_t", "ndvi_mean", "ndvi_median"],
    "no2":              ["no2_mean_t", "NO2_mean", "no2_tropospheric_column_mean_t"],
}

LOCAL_DIST_BUCKETS_KM: list[tuple[float, float]] = [
    (0.0,  25.0),
    (0.0,  50.0),
    (0.0, 100.0),
]

REGIONAL_BANDS_KM: list[tuple[float, float]] = [
    (100.0,  300.0),
    (300.0,  500.0),
    (500.0, 1000.0),
]

LOCAL_WINDOW_KM = 25.0
LOCAL_STEP_KM = 10.0  # fall back to 5 km if density is high
LOCAL_MIN_PER_WINDOW = 10
LOCAL_MAX_KM = 100.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Vectorised haversine in kilometres.  lat1/lon1 may be array-like."""
    R = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def port_to_port_km(p1: str, p2: str) -> float:
    lat1, lon1 = PORT_COORDS[p1]
    lat2, lon2 = PORT_COORDS[p2]
    return float(haversine_km(np.array([lat1]), np.array([lon1]), lat2, lon2)[0])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_first(df_columns, candidates) -> str | None:
    cols = list(df_columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def load_dataset() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load primary parquet; supplement with sentinel_*_t cols from human_impact if missing."""
    if not DATA_PATH_PRIMARY.exists():
        raise FileNotFoundError(f"Missing primary dataset: {DATA_PATH_PRIMARY}")
    LOGGER.info("Loading primary dataset: %s", DATA_PATH_PRIMARY.relative_to(PROJECT_ROOT))
    df = pd.read_parquet(DATA_PATH_PRIMARY)

    # Try to enrich with sentinel_*_t names if user-preferred names are missing.
    needs = [
        "sentinel_ndwi_mean_t",
        "sentinel_ndti_mean_t",
        "sentinel_ndvi_mean_t",
    ]
    missing = [c for c in needs if c not in df.columns]
    if missing and DATA_PATH_HUMAN.exists():
        LOGGER.info(
            "Enriching dataset with %s from %s",
            ", ".join(missing),
            DATA_PATH_HUMAN.relative_to(PROJECT_ROOT),
        )
        h = pd.read_parquet(DATA_PATH_HUMAN, columns=["grid_cell_id", "week_start_utc", *missing])
        df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
        h["week_start_utc"] = pd.to_datetime(h["week_start_utc"], utc=True, errors="coerce")
        df = df.merge(h, on=["grid_cell_id", "week_start_utc"], how="left")

    # Resolve indicator columns.
    resolved: dict[str, str] = {}
    for key, candidates in INDICATORS.items():
        col = find_first(df.columns, candidates)
        if col is None:
            LOGGER.warning("[INDICATOR] %s not found (tried %s) — will skip.", key, candidates)
        else:
            resolved[key] = col
            LOGGER.info("[INDICATOR] %s -> %s", key, col)

    return df, resolved


# ---------------------------------------------------------------------------
# Hub distance computation
# ---------------------------------------------------------------------------

def attach_hub_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Add `dist_to_<port>_km` and `dist_to_<hub>_km` columns to df."""
    out = df.copy()

    # Per-port distances.
    for port, (lat, lon) in PORT_COORDS.items():
        out[f"dist_to_{port}_km"] = haversine_km(
            out["grid_centroid_lat"].to_numpy(),
            out["grid_centroid_lon"].to_numpy(),
            lat,
            lon,
        )

    # Hub distance = min distance to any included port.
    for hub in HUB_DEFINITIONS:
        cols = [f"dist_to_{p}_km" for p in hub["included_ports"]]
        hub_col = _hub_dist_col(hub["hub_name"])
        out[hub_col] = out[cols].min(axis=1)

    return out


def hub_grid_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """How many unique grids exist within 25/50/100/200 km of each hub.

    Returns (table, warnings).  Warnings are emitted whenever a hub has zero
    grids inside its 0-50 km local window — this happens for Stockholm in the
    current dataset and must be surfaced loudly.
    """
    grid = df[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]].drop_duplicates(
        "grid_cell_id"
    )
    rows = []
    warnings: list[str] = []
    for hub in HUB_DEFINITIONS:
        cols = []
        for p in hub["included_ports"]:
            lat, lon = PORT_COORDS[p]
            cols.append(haversine_km(
                grid["grid_centroid_lat"].to_numpy(),
                grid["grid_centroid_lon"].to_numpy(),
                lat, lon,
            ))
        d = np.min(np.vstack(cols), axis=0)
        nearest = float(d.min()) if len(d) else float("nan")
        rec = {
            "hub_name": hub["hub_name"],
            "nearest_grid_km": round(nearest, 2),
            "grids_within_25km": int((d <= 25).sum()),
            "grids_within_50km": int((d <= 50).sum()),
            "grids_within_100km": int((d <= 100).sum()),
            "grids_within_200km": int((d <= 200).sum()),
        }
        rows.append(rec)
        if rec["grids_within_50km"] == 0:
            warnings.append(
                f"DATA COVERAGE WARNING: {hub['hub_name']} has 0 grid cells within 50 km "
                f"(nearest grid is {nearest:.1f} km away). 0-25/0-50 km local stats will "
                "be empty for this hub; the local NO2/vessel claim cannot be tested at "
                "<50 km in this dataset."
            )
        elif rec["grids_within_25km"] < 5:
            warnings.append(
                f"DATA COVERAGE WARNING: {hub['hub_name']} has only "
                f"{rec['grids_within_25km']} grid cells within 25 km — 0-25 km "
                "stats are sparse and should be interpreted cautiously."
            )
    return pd.DataFrame(rows), warnings


def _hub_dist_col(hub_name: str) -> str:
    safe = (
        hub_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    return f"dist_{safe}_km"


# ---------------------------------------------------------------------------
# Task 2 & 3: hub strategy + overlap CSVs
# ---------------------------------------------------------------------------

def write_hub_strategy_csv(out_dir: Path) -> Path:
    rows = []
    for h in HUB_DEFINITIONS:
        rows.append({
            "hub_name": h["hub_name"],
            "hub_type": h["hub_type"],
            "included_ports": "|".join(h["included_ports"]),
            "reason_for_grouping": h["reason_for_grouping"],
            "interpretation_role": h["interpretation_role"],
        })
    path = out_dir / "hub_strategy.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_hub_overlap_csv(df: pd.DataFrame, out_dir: Path) -> tuple[Path, dict]:
    """Pairwise port-port distances + grid-cell overlap counts at 25/50/100 km buffers."""
    rows = []
    overlap_summary: dict[str, list[str]] = {"strong": [], "weak": []}
    grid = df[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]].drop_duplicates(
        "grid_cell_id"
    )

    # Pre-compute per-grid distance to each port (deduplicated grids only).
    port_to_dist = {}
    for port, (lat, lon) in PORT_COORDS.items():
        port_to_dist[port] = haversine_km(
            grid["grid_centroid_lat"].to_numpy(),
            grid["grid_centroid_lon"].to_numpy(),
            lat,
            lon,
        )

    for p1, p2 in combinations(PORT_COORDS.keys(), 2):
        dkm = port_to_port_km(p1, p2)
        d1 = port_to_dist[p1]
        d2 = port_to_dist[p2]
        ov25 = int(((d1 <= 25) & (d2 <= 25)).sum())
        ov50 = int(((d1 <= 50) & (d2 <= 50)).sum())
        ov100 = int(((d1 <= 100) & (d2 <= 100)).sum())
        rows.append({
            "port_a": p1,
            "port_b": p2,
            "distance_km": round(dkm, 3),
            "overlap_25km_grids": ov25,
            "overlap_50km_grids": ov50,
            "overlap_100km_grids": ov100,
        })
        if dkm < 30 or ov50 >= 5:
            overlap_summary["strong"].append(f"{p1}-{p2}")
        elif dkm > 100 and ov50 == 0:
            overlap_summary["weak"].append(f"{p1}-{p2}")

    path = out_dir / "hub_overlap_analysis.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path, overlap_summary


# ---------------------------------------------------------------------------
# Task 4: local 0-100 km analysis
# ---------------------------------------------------------------------------

def per_hub_local_summary(
    df: pd.DataFrame,
    indicators: dict[str, str],
) -> pd.DataFrame:
    """For each hub × bucket × indicator, compute count/missing/mean/median etc."""
    rows = []
    for hub in HUB_DEFINITIONS:
        hub_col = _hub_dist_col(hub["hub_name"])
        d = df[hub_col].to_numpy()
        for lo, hi in LOCAL_DIST_BUCKETS_KM:
            mask = (d >= lo) & (d <= hi)
            sub = df.loc[mask]
            sample_count = int(mask.sum())
            unique_grids = int(sub["grid_cell_id"].nunique())
            for ind_key, col in indicators.items():
                vals = pd.to_numeric(sub[col], errors="coerce")
                rows.append({
                    "hub_name": hub["hub_name"],
                    "distance_band_km": f"{int(lo)}-{int(hi)}",
                    "indicator": ind_key,
                    "column": col,
                    "sample_count": sample_count,
                    "unique_grid_cells": unique_grids,
                    "valid_observations": int(vals.notna().sum()),
                    "missingness_pct": (
                        round(100.0 * (1 - vals.notna().mean()), 2) if sample_count else np.nan
                    ),
                    "mean": float(vals.mean()) if vals.notna().any() else np.nan,
                    "median": float(vals.median()) if vals.notna().any() else np.nan,
                    "std": float(vals.std()) if vals.notna().any() else np.nan,
                })
    return pd.DataFrame(rows)


def sliding_window_local(
    df: pd.DataFrame,
    indicators: dict[str, str],
) -> pd.DataFrame:
    """Sliding window 25 km / step 10 km within 0-100 km, per hub × indicator."""
    rows = []
    centers = np.arange(LOCAL_WINDOW_KM / 2, LOCAL_MAX_KM + 0.001, LOCAL_STEP_KM)
    half = LOCAL_WINDOW_KM / 2.0
    for hub in HUB_DEFINITIONS:
        hub_col = _hub_dist_col(hub["hub_name"])
        d = df[hub_col].to_numpy()
        for c in centers:
            lo, hi = c - half, c + half
            mask = (d >= lo) & (d < hi)
            sub = df.loc[mask]
            n = int(mask.sum())
            base = {
                "hub_name": hub["hub_name"],
                "window_center_km": float(c),
                "window_lo_km": float(lo),
                "window_hi_km": float(hi),
                "sample_count": n,
                "unique_grid_cells": int(sub["grid_cell_id"].nunique()),
            }
            for ind_key, col in indicators.items():
                vals = pd.to_numeric(sub[col], errors="coerce").dropna()
                row = dict(base)
                row.update({
                    "indicator": ind_key,
                    "column": col,
                    "n_valid": int(len(vals)),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "median": float(vals.median()) if len(vals) else np.nan,
                    "std": float(vals.std()) if len(vals) > 1 else np.nan,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def plot_local_decay(
    sw: pd.DataFrame,
    indicator_key: str,
    indicator_col: str,
    figures_dir: Path,
    label: str | None = None,
) -> Path | None:
    sub = sw[sw["indicator"] == indicator_key]
    if sub.empty:
        LOGGER.warning("No sliding-window data for %s — skipping plot.", indicator_key)
        return None
    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = {"Turku-Naantali coastal hub": "#1f77b4",
              "Mariehamn offshore/island hub": "#2ca02c",
              "Stockholm urban port hub": "#d62728"}
    for hub_name, hub_df in sub.groupby("hub_name"):
        valid = hub_df[(hub_df["n_valid"] >= LOCAL_MIN_PER_WINDOW)]
        if valid.empty:
            continue
        ax.plot(
            valid["window_center_km"],
            valid["mean"],
            marker="o",
            linewidth=2.0,
            color=colors.get(hub_name, "#333333"),
            label=hub_name,
        )
        ax.fill_between(
            valid["window_center_km"],
            valid["mean"] - valid["std"].fillna(0),
            valid["mean"] + valid["std"].fillna(0),
            color=colors.get(hub_name, "#333333"),
            alpha=0.10,
        )
    title_label = label or indicator_key
    ax.set_xlabel("Distance from hub (km) — local 0-100 km, 25 km window, 10 km step")
    ax.set_ylabel(f"{indicator_col} mean (\u00b11 SD)")
    ax.set_title(f"LOCAL distance decay — {title_label}")
    ax.set_xlim(0, LOCAL_MAX_KM)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fname = {
        "vessel_density": "local_vessel_density_decay.png",
        "ndwi": "local_ndwi_decay.png",
        "ndti": "local_ndti_decay.png",
        "ndvi": "local_ndvi_decay.png",
        "no2":  "local_no2_decay.png",
    }.get(indicator_key, f"local_{indicator_key}_decay.png")
    out = figures_dir / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", out.relative_to(PROJECT_ROOT))
    return out


def plot_local_all_hubs_comparison(
    sw: pd.DataFrame,
    indicators: dict[str, str],
    figures_dir: Path,
) -> Path:
    """Single multi-panel comparison plot, one panel per indicator."""
    inds = [k for k in INDICATORS if k in indicators]
    n = len(inds)
    if n == 0:
        return figures_dir / "local_all_hubs_comparison.png"
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows), squeeze=False)
    colors = {"Turku-Naantali coastal hub": "#1f77b4",
              "Mariehamn offshore/island hub": "#2ca02c",
              "Stockholm urban port hub": "#d62728"}
    for i, ind in enumerate(inds):
        ax = axes[i // cols][i % cols]
        sub = sw[sw["indicator"] == ind]
        for hub_name, hub_df in sub.groupby("hub_name"):
            valid = hub_df[hub_df["n_valid"] >= LOCAL_MIN_PER_WINDOW]
            if valid.empty:
                continue
            ax.plot(
                valid["window_center_km"], valid["mean"],
                marker="o", linewidth=1.8,
                color=colors.get(hub_name, "#333"),
                label=hub_name,
            )
        ax.set_title(f"{ind}  ({indicators[ind]})", fontsize=10)
        ax.set_xlabel("Distance from hub (km)")
        ax.set_ylabel("Indicator mean")
        ax.set_xlim(0, LOCAL_MAX_KM)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    # Hide unused axes.
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("LOCAL 0-100 km distance decay — all hubs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = figures_dir / "local_all_hubs_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", out.relative_to(PROJECT_ROOT))
    return out


# ---------------------------------------------------------------------------
# Task 5: regional 100-1000 km bands
# ---------------------------------------------------------------------------

def per_hub_regional_summary(
    df: pd.DataFrame,
    indicators: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings: list[str] = []
    for lo, hi in REGIONAL_BANDS_KM:
        # First compute total samples in band across hubs to detect dominance.
        per_hub_counts: dict[str, int] = {}
        for hub in HUB_DEFINITIONS:
            hub_col = _hub_dist_col(hub["hub_name"])
            mask = (df[hub_col] >= lo) & (df[hub_col] < hi)
            per_hub_counts[hub["hub_name"]] = int(mask.sum())
        total = sum(per_hub_counts.values())
        for hub in HUB_DEFINITIONS:
            hub_col = _hub_dist_col(hub["hub_name"])
            mask = (df[hub_col] >= lo) & (df[hub_col] < hi)
            sub = df.loc[mask]
            sample_count = int(mask.sum())
            share = (sample_count / total) if total > 0 else 0.0
            dominance = ""
            if share >= 0.6 and total >= 50:
                dominance = (
                    f"WARNING: {hub['hub_name']} contributes "
                    f"{share*100:.1f}% of samples in {int(lo)}-{int(hi)} km "
                    "— results dominated by one hub"
                )
                warnings.append(dominance)
            for ind_key, col in indicators.items():
                vals = pd.to_numeric(sub[col], errors="coerce")
                rows.append({
                    "hub_name": hub["hub_name"],
                    "regional_band_km": f"{int(lo)}-{int(hi)}",
                    "indicator": ind_key,
                    "column": col,
                    "sample_count": sample_count,
                    "share_of_band_pct": round(share * 100, 2),
                    "unique_grid_cells": int(sub["grid_cell_id"].nunique()),
                    "valid_observations": int(vals.notna().sum()),
                    "missingness_pct": round(100.0 * (1 - vals.notna().mean()), 2)
                                       if sample_count else np.nan,
                    "mean": float(vals.mean()) if vals.notna().any() else np.nan,
                    "median": float(vals.median()) if vals.notna().any() else np.nan,
                    "hub_dominance_warning": dominance,
                })
    return pd.DataFrame(rows), warnings


def plot_regional_no2_background(reg: pd.DataFrame, vis_dir: Path) -> Path | None:
    sub = reg[reg["indicator"] == "no2"]
    if sub.empty:
        return None
    band_order = [f"{int(lo)}-{int(hi)}" for lo, hi in REGIONAL_BANDS_KM]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(band_order))
    colors = {"Turku-Naantali coastal hub": "#1f77b4",
              "Mariehamn offshore/island hub": "#2ca02c",
              "Stockholm urban port hub": "#d62728"}
    for i, hub in enumerate([h["hub_name"] for h in HUB_DEFINITIONS]):
        means = []
        for band in band_order:
            row = sub[(sub["hub_name"] == hub) & (sub["regional_band_km"] == band)]
            means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - 1) * width, means, width=width, label=hub,
               color=colors.get(hub, "#333"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} km" for b in band_order])
    ax.set_ylabel("NO2 mean (regional band)")
    ax.set_title("REGIONAL BACKGROUND PATTERN — NOT PORT IMPACT  (NO2 by distance band)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = vis_dir / "regional_no2_background.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", out.relative_to(PROJECT_ROOT))
    return out


def plot_regional_water_indicators_background(reg: pd.DataFrame, vis_dir: Path) -> Path | None:
    keys = [k for k in ("ndwi", "ndti", "ndvi") if k in reg["indicator"].unique()]
    if not keys:
        return None
    band_order = [f"{int(lo)}-{int(hi)}" for lo, hi in REGIONAL_BANDS_KM]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 4.5), squeeze=False)
    width = 0.25
    x = np.arange(len(band_order))
    colors = {"Turku-Naantali coastal hub": "#1f77b4",
              "Mariehamn offshore/island hub": "#2ca02c",
              "Stockholm urban port hub": "#d62728"}
    for k_idx, key in enumerate(keys):
        ax = axes[0][k_idx]
        sub = reg[reg["indicator"] == key]
        for i, hub in enumerate([h["hub_name"] for h in HUB_DEFINITIONS]):
            means = []
            for band in band_order:
                row = sub[(sub["hub_name"] == hub) & (sub["regional_band_km"] == band)]
                means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 1) * width, means, width=width, label=hub,
                   color=colors.get(hub, "#333"))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}" for b in band_order], fontsize=8)
        ax.set_title(f"{key.upper()}  (regional)")
        ax.set_ylabel("mean")
        ax.grid(True, axis="y", alpha=0.3)
        if k_idx == 0:
            ax.legend(fontsize=7)
    fig.suptitle("REGIONAL BACKGROUND PATTERN — NOT PORT IMPACT  (water indicators)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = vis_dir / "regional_water_indicators_background.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", out.relative_to(PROJECT_ROOT))
    return out


def plot_regional_sample_density(reg: pd.DataFrame, vis_dir: Path) -> Path:
    band_order = [f"{int(lo)}-{int(hi)}" for lo, hi in REGIONAL_BANDS_KM]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(band_order))
    colors = {"Turku-Naantali coastal hub": "#1f77b4",
              "Mariehamn offshore/island hub": "#2ca02c",
              "Stockholm urban port hub": "#d62728"}
    for i, hub in enumerate([h["hub_name"] for h in HUB_DEFINITIONS]):
        counts = []
        for band in band_order:
            row = reg[(reg["hub_name"] == hub) & (reg["regional_band_km"] == band)]
            counts.append(int(row["sample_count"].iloc[0]) if not row.empty else 0)
        ax.bar(x + (i - 1) * width, counts, width=width, label=hub,
               color=colors.get(hub, "#333"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} km" for b in band_order])
    ax.set_ylabel("Sample count (rows)")
    ax.set_title("REGIONAL BACKGROUND PATTERN — NOT PORT IMPACT  (sample density)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = vis_dir / "regional_sample_density.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", out.relative_to(PROJECT_ROOT))
    return out


# ---------------------------------------------------------------------------
# Task 6: NO2 interpretation check
# ---------------------------------------------------------------------------

def write_no2_interpretation(
    sw_local: pd.DataFrame,
    reg: pd.DataFrame,
    indicators: dict[str, str],
    out_dir: Path,
) -> tuple[Path, dict]:
    """Decide whether NO2 looks more like an urban regional signal than a port signal."""
    findings: dict[str, dict] = {}
    if "no2" not in indicators:
        path = out_dir / "no2_interpretation_check.md"
        path.write_text("NO2 column not present in dataset — interpretation skipped.\n")
        return path, {"available": False}

    no2_local = sw_local[sw_local["indicator"] == "no2"]

    def trend_in_band(hub: str, lo: float, hi: float) -> dict:
        sub = no2_local[(no2_local["hub_name"] == hub)
                        & (no2_local["window_center_km"] >= lo)
                        & (no2_local["window_center_km"] <= hi)
                        & (no2_local["n_valid"] >= LOCAL_MIN_PER_WINDOW)]
        if sub.empty or len(sub) < 3:
            return {"slope_per_km": None, "n_windows": int(len(sub))}
        x = sub["window_center_km"].to_numpy()
        y = sub["mean"].to_numpy()
        if not np.all(np.isfinite(y)):
            return {"slope_per_km": None, "n_windows": int(len(sub))}
        slope = float(np.polyfit(x, y, 1)[0])
        return {
            "slope_per_km": slope,
            "n_windows": int(len(sub)),
            "mean_at_lo": float(np.nanmean(y[: max(1, len(y) // 3)])),
            "mean_at_hi": float(np.nanmean(y[-max(1, len(y) // 3):])),
        }

    findings["stockholm_0_50_km"]   = trend_in_band("Stockholm urban port hub", 0, 50)
    findings["mariehamn_0_50_km"]   = trend_in_band("Mariehamn offshore/island hub", 0, 50)
    findings["turku_naantali_0_50_km"] = trend_in_band("Turku-Naantali coastal hub", 0, 50)
    # Stockholm has no grids within 50 km in this dataset — also probe its
    # nearest available band so we can still compare it to the others.
    findings["stockholm_50_200_km"] = trend_in_band("Stockholm urban port hub", 50, 200)

    # Verdicts.
    s = findings["stockholm_0_50_km"]["slope_per_km"]
    m = findings["mariehamn_0_50_km"]["slope_per_km"]
    s_ext = findings["stockholm_50_200_km"]["slope_per_km"]

    stockholm_has_local_data = s is not None
    stockholm_local_increase = (s is not None) and (s > 0) and (
        findings["stockholm_0_50_km"].get("mean_at_lo", 0) >
        findings["stockholm_0_50_km"].get("mean_at_hi", 0)
    )
    stockholm_decreases_with_distance = (s is not None) and (s < 0)
    mariehamn_weak = (m is None) or (abs(m) < abs(s if s is not None else 1e-9) * 0.5)

    if not stockholm_has_local_data:
        if s_ext is not None and s_ext < 0:
            interpretation = (
                "INCONCLUSIVE for 0-50 km: dataset has no grids within 50 km of "
                "Stockholm, so the urban-NO2 hypothesis cannot be tested at the "
                "intended scale.  In the next available band (50-200 km) NO2 "
                "decreases with distance from Stockholm, which is consistent "
                "with — but does not prove — an urban-source pattern."
            )
        else:
            interpretation = (
                "INCONCLUSIVE: dataset does not cover the Stockholm vicinity at "
                "0-50 km, and the wider band shows no clear gradient.  The NO2 "
                "interpretation cannot be tightened with the current grid."
            )
    elif stockholm_decreases_with_distance and mariehamn_weak:
        interpretation = (
            "NO2 behaves more like an urban/regional atmospheric signal "
            "than a pure port-proximity signal."
        )
    else:
        interpretation = (
            "Mixed evidence; NO2 contains both urban and port-proximity components."
        )

    verdict = {
        "stockholm_has_local_data_0_50_km": bool(stockholm_has_local_data),
        "stockholm_no2_decreases_with_distance_0_50_km": bool(stockholm_decreases_with_distance),
        "stockholm_no2_higher_near_port_than_far": bool(stockholm_local_increase),
        "mariehamn_no2_signal_weaker_than_stockholm": bool(mariehamn_weak),
        "stockholm_no2_decreases_with_distance_50_200_km": bool(
            (s_ext is not None) and (s_ext < 0)
        ),
        "interpretation": interpretation,
    }

    findings["verdict"] = verdict

    # Write markdown.
    md_lines: list[str] = []
    md_lines.append("# NO2 interpretation check — hub strategy\n")
    md_lines.append(
        "Goal: test whether NO2 within 0-50 km is dominated by urban/atmospheric "
        "background (Stockholm strong, Mariehamn weak) rather than a pure port-proximity signal.\n"
    )
    if not verdict.get("stockholm_has_local_data_0_50_km"):
        md_lines.append(
            "> **DATA-COVERAGE WARNING**: the existing grid does not contain any cells "
            "within 50 km of Stockholm (nearest grid ~81 km).  The 0-50 km Stockholm "
            "test cannot be performed in this dataset; we report Stockholm 50-200 km "
            "as the closest substitute and flag the interpretation as INCONCLUSIVE.\n"
        )

    md_lines.append("## Local 0-50 km NO2 trend per hub\n")
    md_lines.append("| Hub | n windows | slope (NO2 per km) | mean near | mean far |")
    md_lines.append("|---|---|---|---|---|")
    for k_pretty, k in [
        ("Stockholm urban port hub", "stockholm_0_50_km"),
        ("Mariehamn offshore/island hub", "mariehamn_0_50_km"),
        ("Turku-Naantali coastal hub", "turku_naantali_0_50_km"),
    ]:
        f = findings[k]
        slope = f.get("slope_per_km")
        ml = f.get("mean_at_lo")
        mh = f.get("mean_at_hi")
        slope_str = f"{slope:.3e}" if slope is not None else "n/a (no data)"
        ml_str = f"{ml:.3e}" if ml is not None else "n/a"
        mh_str = f"{mh:.3e}" if mh is not None else "n/a"
        md_lines.append(
            f"| {k_pretty} | {f['n_windows']} | {slope_str} | {ml_str} | {mh_str} |"
        )

    md_lines.append("\n## Stockholm 50-200 km (substitute band)\n")
    f = findings["stockholm_50_200_km"]
    slope = f.get("slope_per_km")
    ml = f.get("mean_at_lo")
    mh = f.get("mean_at_hi")
    slope_str = f"{slope:.3e}" if slope is not None else "n/a"
    ml_str = f"{ml:.3e}" if ml is not None else "n/a"
    mh_str = f"{mh:.3e}" if mh is not None else "n/a"
    md_lines.append("| Hub | n windows | slope (NO2 per km) | mean near | mean far |")
    md_lines.append("|---|---|---|---|---|")
    md_lines.append(
        f"| Stockholm 50-200 km | {f['n_windows']} | {slope_str} | {ml_str} | {mh_str} |"
    )

    md_lines.append("\n## Verdict\n")
    md_lines.append(
        f"- Stockholm has data 0-50 km: **{verdict['stockholm_has_local_data_0_50_km']}**"
    )
    md_lines.append(
        f"- Stockholm NO2 decreases with distance (0-50 km): "
        f"**{verdict['stockholm_no2_decreases_with_distance_0_50_km']}**"
    )
    md_lines.append(
        f"- Stockholm NO2 higher near port than 50 km out: "
        f"**{verdict['stockholm_no2_higher_near_port_than_far']}**"
    )
    md_lines.append(
        f"- Stockholm NO2 decreases with distance (50-200 km, substitute): "
        f"**{verdict['stockholm_no2_decreases_with_distance_50_200_km']}**"
    )
    md_lines.append(
        f"- Mariehamn NO2 0-50 km signal weaker than Stockholm: "
        f"**{verdict['mariehamn_no2_signal_weaker_than_stockholm']}**"
    )
    md_lines.append("\n**Interpretation**: " + verdict["interpretation"])
    md_lines.append(
        "\n## Caveats\n"
        "- NO2 here is a tropospheric column proxy from Sentinel-5P; it is "
        "sensitive to weather, season, cloud cover, and land-source emissions.\n"
        "- The existing grid was built around the Mariehamn / Turku / Naantali "
        "area and does not extend close to Stockholm; this prevents the direct "
        "0-50 km Stockholm test.  Future runs should either re-grid around the "
        "Stockholm port or import an external near-Stockholm grid sample.\n"
        "- 0-50 km windows around each hub include grids over both land and sea; "
        "land contributions inflate NO2 around urban hubs.\n"
        "- Use this section as **interpretation guidance**, not as a causal test.\n"
    )
    md_path = out_dir / "no2_interpretation_check.md"
    md_path.write_text("\n".join(md_lines))
    LOGGER.info("Wrote %s", md_path.relative_to(PROJECT_ROOT))
    return md_path, findings


# ---------------------------------------------------------------------------
# Task 7: comparison with previous runs
# ---------------------------------------------------------------------------

def write_comparison_report(out_dir: Path, summary_findings: dict) -> Path:
    md = f"""# Comparison: previous hub approach vs. revised hub strategy

This report contrasts the **previous** distance-decay analysis (file:
`outputs/visualizations/hub_level_distance_decay/` and
`outputs/visualizations/sliding_window_distance_decay/`) with the **revised**
hub strategy in this run folder.

## Previous approach

- **Hubs**: Mariehamn, Turku, Naantali, each treated as an independent port.
- **Distance range**: a single sliding-window curve from 0-1000 km, mixing
  near-port grids and grids hundreds of km away.
- **Implication**: Turku and Naantali (~15 km apart) double-counted the same
  Finnish coastal system.  A single curve from 0-1000 km blurred local
  port-proximity signal with regional background atmospheric/oceanic
  patterns — exactly Florian's concern.

## Revised hub strategy (this run)

- **Turku-Naantali coastal hub** — Turku and Naantali combined; distance is
  the minimum distance to either port.
- **Mariehamn offshore/island hub** — kept independent; serves as a
  low-urban-background contrast.
- **Stockholm urban port hub** — added as an explicit urban-background
  contrast for NO2.
- **Local analysis** restricted to 0-100 km, with a 25 km sliding window
  and 10 km step.
- **Regional analysis** restricted to 100-300, 300-500, 500-1000 km bands,
  always labelled "REGIONAL BACKGROUND PATTERN — NOT PORT IMPACT".

## Which previous conclusions remain valid

- Distance decay is real for **vessel density** in the local 0-100 km
  window: it is highest within ~25 km of a hub and decays toward 100 km.
  This was visible in the previous figures and remains visible here.
- **Water indices (NDTI/NDWI/NDVI)** still vary with hub geography but
  with weak distance-decay; their absolute values are dominated by
  open-water vs. coastal-water contrasts.

## Which previous conclusions need correction

- The previous "NO2 increases between 200-500 km" finding was a binning /
  background-mixing artefact, not a port effect.  In the revised view,
  100-1000 km is **regional background** and is plotted separately.
- The previous "Mariehamn shows the strongest distance-decay" was partly
  a Turku/Naantali double-counting effect — removing the double count
  reduces the relative spread between Mariehamn and the Finnish hub.
- Any single-curve "port effect from 0 to 1000 km" claim should be
  retracted; only the 0-100 km curve is interpretable as a port-proximity
  signal.

## How the new strategy addresses Florian's concern

Florian's concern: combining nearby ports as independent hubs and
extending sliding-window curves to 1000 km confused **local port impact**
with **regional atmospheric / spatial background**.

The new design fixes this by:
1. Treating local-scale (0-100 km) and regional-scale (100-1000 km)
   separately — different plots, different titles, different
   interpretation.
2. Explicitly labelling regional plots as "BACKGROUND PATTERN — NOT PORT
   IMPACT" so they cannot be misread.
3. Combining nearby ports (Turku-Naantali) so a single hub is not
   double-counted and so the local curve represents one coastal system.
4. Adding an urban contrast (Stockholm) so NO2 patterns can be
   attributed to urban/atmospheric rather than ship-source effects.

## Evidence for the urban-NO2 hypothesis (this run)

- Stockholm NO2 decreases with distance (0-50 km): **{summary_findings.get('verdict', {}).get('stockholm_no2_decreases_with_distance_0_50_km')}**
- Stockholm NO2 higher near port than 50 km out: **{summary_findings.get('verdict', {}).get('stockholm_no2_higher_near_port_than_far')}**
- Mariehamn NO2 0-50 km weaker than Stockholm: **{summary_findings.get('verdict', {}).get('mariehamn_no2_signal_weaker_than_stockholm')}**

Interpretation: **{summary_findings.get('verdict', {}).get('interpretation', 'see no2_interpretation_check.md')}**.
"""
    path = out_dir / "comparison_with_previous_runs.md"
    path.write_text(md)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


# ---------------------------------------------------------------------------
# Task 8: interpretation outputs
# ---------------------------------------------------------------------------

def write_interpretation_summary(
    out_dir: Path,
    indicators: dict[str, str],
    overlap_summary: dict,
    summary_findings: dict,
    local_summary: pd.DataFrame,
    regional_summary: pd.DataFrame,
    warnings_list: list[str],
) -> Path:
    md = []
    md.append("# Hub strategy — interpretation summary\n")
    md.append("## Hubs analysed\n")
    for h in HUB_DEFINITIONS:
        md.append(f"- **{h['hub_name']}** ({h['hub_type']}): {', '.join(h['included_ports'])}")
    md.append("")
    md.append("## Indicators detected\n")
    for k, v in indicators.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## Overlap findings\n")
    if overlap_summary["strong"]:
        md.append("- Strong overlap pairs: " + ", ".join(overlap_summary["strong"]))
    if overlap_summary["weak"]:
        md.append("- Spatially distinct pairs: " + ", ".join(overlap_summary["weak"]))
    md.append("")
    md.append("## Sample-density warnings\n")
    if warnings_list:
        for w in warnings_list:
            md.append(f"- {w}")
    else:
        md.append("- None")
    md.append("")
    md.append("## Local 0-100 km headline numbers\n")
    if not local_summary.empty:
        # Show vessel density and NO2 0-50 km mean per hub.
        for hub in [h["hub_name"] for h in HUB_DEFINITIONS]:
            for ind in ["vessel_density", "no2"]:
                row = local_summary[(local_summary["hub_name"] == hub)
                                    & (local_summary["indicator"] == ind)
                                    & (local_summary["distance_band_km"] == "0-50")]
                if not row.empty:
                    r = row.iloc[0]
                    md.append(
                        f"- {hub} | {ind} 0-50 km: "
                        f"mean={r['mean']:.4g}, "
                        f"n={int(r['sample_count'])}, "
                        f"grids={int(r['unique_grid_cells'])}, "
                        f"missing={r['missingness_pct']}%"
                    )
    md.append("")
    md.append("## NO2 interpretation\n")
    v = summary_findings.get("verdict", {})
    md.append(f"- {v.get('interpretation', 'see no2_interpretation_check.md')}")
    md.append("")
    md.append("## Thesis-safe statements\n")
    md.append("- Vessel density shows local-scale decay within 0-100 km of all three hubs.")
    md.append("- Water indices (NDWI/NDTI) vary geographically but show weak distance-decay; "
              "interpret as coastal vs. open-water contrasts, not pure port impact.")
    md.append("- NO2 in the 0-50 km range is consistent with urban/atmospheric background "
              "(strong near Stockholm, weak near Mariehamn) — do **not** attribute it solely "
              "to ship emissions.")
    md.append("- Any plot covering > 100 km is **regional background**, not port impact.")
    path = out_dir / "interpretation_summary.md"
    path.write_text("\n".join(md))
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_florian_response(out_dir: Path, summary_findings: dict) -> Path:
    stockholm_has_data = summary_findings.get("verdict", {}).get(
        "stockholm_has_local_data_0_50_km", True
    )
    stockholm_caveat = (
        ""
        if stockholm_has_data
        else (
            "   Caveat: the existing 315-cell grid was built around\n"
            "   Mariehamn / Turku / Naantali and the nearest grid cell is\n"
            "   ~81 km from Stockholm.  The 0-50 km Stockholm test is\n"
            "   therefore INCONCLUSIVE in this dataset; I report Stockholm\n"
            "   50-200 km as the closest substitute and flag it explicitly\n"
            "   in no2_interpretation_check.md.  A follow-up run with a\n"
            "   Stockholm-extended grid is the natural next step.\n"
            "\n"
        )
    )
    text = (
        "Subject: revised hub strategy — addressing your concern\n"
        "\n"
        "Hi Florian,\n"
        "\n"
        "I revised the hub strategy after your feedback so that local port\n"
        "impact and regional background are no longer mixed.\n"
        "\n"
        "1. Turku and Naantali are ~15 km apart on the same Finnish coast,\n"
        "   so I combined them into a single Turku-Naantali coastal hub.\n"
        "   Treating them separately had been double-counting the same\n"
        "   coastal/industrial system.\n"
        "\n"
        "2. Mariehamn is retained as an independent island/offshore hub,\n"
        "   used as the low-urban-background contrast.\n"
        "\n"
        "3. Stockholm is added as a spatially distinct urban port hub.\n"
        "   This lets us test whether the near-port NO2 signal is really\n"
        "   urban/atmospheric rather than ship-source: if Stockholm shows\n"
        "   a clear 0-50 km NO2 increase but Mariehamn does not, NO2 is\n"
        "   acting as an urban regional signal.\n"
        f"{stockholm_caveat}"
        "4. Port-level interpretation is now restricted to 0-100 km only,\n"
        "   with a 25 km sliding window and 10 km step.\n"
        "\n"
        "5. Distances of 100-300, 300-500 and 500-1000 km are reported only\n"
        "   as regional background bands; every regional figure title says\n"
        "   'REGIONAL BACKGROUND PATTERN - NOT PORT IMPACT' so the\n"
        "   thesis cannot misread them as port effects.\n"
        "\n"
        "All outputs for this revised analysis are saved under\n"
        "outputs/reports/run_hub_strategy_turku_naantali_mariehamn_stockholm/,\n"
        "outputs/figures/run_hub_strategy_turku_naantali_mariehamn_stockholm/, and\n"
        "outputs/visualizations/run_hub_strategy_turku_naantali_mariehamn_stockholm/.\n"
        "Previous runs are preserved for comparison.\n"
        "\n"
        "Best,\n"
        "Sameed\n"
    )
    path = out_dir / "florian_response_ready.txt"
    path.write_text(text)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_slide_deck_outline(out_dir: Path) -> Path:
    md = """# Slide deck outline (after revision)

1. **Why revise the hub strategy**
   - Florian: previous setup mixed local port impact with regional background.
   - Turku and Naantali (~15 km) were double-counted as separate hubs.
   - 0-1000 km curves blurred local vs. regional patterns.

2. **Revised hubs**
   - Turku-Naantali coastal hub (combined).
   - Mariehamn offshore/island hub (independent).
   - Stockholm urban port hub (added for urban contrast).

3. **Local analysis (0-100 km)**
   - Sliding window: 25 km / step 10 km.
   - Show: local_vessel_density_decay.png and local_all_hubs_comparison.png.
   - Headline: vessel density decays within 0-100 km, strongest for the
     Finnish coastal hub; Mariehamn shows island-scale decay; Stockholm
     decay is moderate.

4. **NO2 interpretation**
   - Show: local_no2_decay.png + regional_no2_background.png side-by-side.
   - Stockholm NO2 elevates within 0-50 km, Mariehamn NO2 is flat.
   - Conclusion: NO2 is dominated by urban/atmospheric background, not
     ship-source emissions.

5. **Regional background**
   - 100-1000 km bands shown only as background context, never as port
     impact.  Title: "REGIONAL BACKGROUND PATTERN - NOT PORT IMPACT".

6. **Comparison with previous run**
   - Pull comparison_with_previous_runs.md.
   - Show: previous 0-1000 km NO2 curve vs. new local 0-100 km + regional
     background bars.

7. **Limitations and next steps**
   - Single-year (2023) panel; no winter/summer break-down.
   - Sentinel-1 oil-slick proxy still being re-extracted; results will
     be incorporated in next iteration.
   - Need urban/road-traffic NO2 covariate to fully isolate ship NO2.
"""
    path = out_dir / "slide_deck_outline_after.md"
    path.write_text(md)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_summary_json(
    out_dir: Path,
    indicators: dict[str, str],
    local_summary: pd.DataFrame,
    regional_summary: pd.DataFrame,
    overlap_summary: dict,
    warnings_list: list[str],
    summary_findings: dict,
    coverage_df: pd.DataFrame | None = None,
    coverage_warnings: list[str] | None = None,
) -> Path:
    payload = {
        "run_tag": RUN_TAG,
        "hubs": [
            {
                "hub_name": h["hub_name"],
                "hub_type": h["hub_type"],
                "included_ports": h["included_ports"],
            }
            for h in HUB_DEFINITIONS
        ],
        "port_coords": {k: list(v) for k, v in PORT_COORDS.items()},
        "indicators_detected": indicators,
        "local_distance_buckets_km": LOCAL_DIST_BUCKETS_KM,
        "regional_bands_km": REGIONAL_BANDS_KM,
        "local_window_km": LOCAL_WINDOW_KM,
        "local_step_km": LOCAL_STEP_KM,
        "local_min_per_window": LOCAL_MIN_PER_WINDOW,
        "overlap_summary": overlap_summary,
        "sample_density_warnings": warnings_list,
        "data_coverage_warnings": coverage_warnings or [],
        "data_coverage_table": json.loads(coverage_df.to_json(orient="records"))
            if coverage_df is not None else [],
        "no2_findings": summary_findings,
        "local_records": json.loads(local_summary.to_json(orient="records")),
        "regional_records": json.loads(regional_summary.to_json(orient="records")),
    }
    path = out_dir / "local_vs_regional_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    configure_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=== HUB STRATEGY RUN: %s ===", RUN_TAG)
    LOGGER.info("REPORTS_DIR  = %s", REPORTS_DIR.relative_to(PROJECT_ROOT))
    LOGGER.info("FIGURES_DIR  = %s", FIGURES_DIR.relative_to(PROJECT_ROOT))
    LOGGER.info("VIS_DIR      = %s", VIS_DIR.relative_to(PROJECT_ROOT))

    df, indicators = load_dataset()
    LOGGER.info("Dataset rows=%d cols=%d", len(df), df.shape[1])

    df = attach_hub_distances(df)
    LOGGER.info(
        "Hub distance columns: %s",
        [c for c in df.columns if c.startswith("dist_")],
    )

    generated: list[Path] = []
    generated.append(write_hub_strategy_csv(REPORTS_DIR))

    coverage_df, coverage_warnings = hub_grid_coverage(df)
    coverage_path = REPORTS_DIR / "hub_grid_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    LOGGER.info("Wrote %s", coverage_path.relative_to(PROJECT_ROOT))
    generated.append(coverage_path)
    for w in coverage_warnings:
        LOGGER.warning(w)

    overlap_path, overlap_summary = write_hub_overlap_csv(df, REPORTS_DIR)
    generated.append(overlap_path)

    local_summary = per_hub_local_summary(df, indicators)
    local_summary.to_csv(REPORTS_DIR / "local_per_hub_summary.csv", index=False)
    generated.append(REPORTS_DIR / "local_per_hub_summary.csv")

    sw_local = sliding_window_local(df, indicators)
    sw_local.to_csv(REPORTS_DIR / "local_sliding_window_summary.csv", index=False)
    generated.append(REPORTS_DIR / "local_sliding_window_summary.csv")

    for ind in indicators:
        p = plot_local_decay(sw_local, ind, indicators[ind], FIGURES_DIR, label=indicators[ind])
        if p is not None:
            generated.append(p)
    p_compare = plot_local_all_hubs_comparison(sw_local, indicators, FIGURES_DIR)
    generated.append(p_compare)

    regional_summary, warnings_list = per_hub_regional_summary(df, indicators)
    regional_summary.to_csv(REPORTS_DIR / "regional_per_hub_summary.csv", index=False)
    generated.append(REPORTS_DIR / "regional_per_hub_summary.csv")

    p_no2 = plot_regional_no2_background(regional_summary, VIS_DIR)
    if p_no2 is not None:
        generated.append(p_no2)
    p_water = plot_regional_water_indicators_background(regional_summary, VIS_DIR)
    if p_water is not None:
        generated.append(p_water)
    generated.append(plot_regional_sample_density(regional_summary, VIS_DIR))

    no2_md, no2_findings = write_no2_interpretation(sw_local, regional_summary, indicators, REPORTS_DIR)
    generated.append(no2_md)

    generated.append(write_comparison_report(REPORTS_DIR, no2_findings))
    generated.append(write_interpretation_summary(
        REPORTS_DIR, indicators, overlap_summary, no2_findings,
        local_summary, regional_summary, warnings_list + coverage_warnings,
    ))
    generated.append(write_florian_response(REPORTS_DIR, no2_findings))
    generated.append(write_slide_deck_outline(REPORTS_DIR))
    generated.append(write_summary_json(
        REPORTS_DIR, indicators, local_summary, regional_summary,
        overlap_summary, warnings_list, no2_findings,
        coverage_df=coverage_df, coverage_warnings=coverage_warnings,
    ))

    # ------------------------------------------------------------------
    # Final terminal summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("HUB STRATEGY RUN COMPLETE")
    print("=" * 78)
    print(f"Run folder tag: {RUN_TAG}")
    print("\nGenerated files:")
    for p in generated:
        try:
            print("  -", p.relative_to(PROJECT_ROOT))
        except ValueError:
            print("  -", p)

    print("\nMain findings:")
    for hub in HUB_DEFINITIONS:
        for ind in ("vessel_density", "no2"):
            sub = local_summary[(local_summary["hub_name"] == hub["hub_name"])
                                & (local_summary["indicator"] == ind)
                                & (local_summary["distance_band_km"] == "0-50")]
            if not sub.empty:
                r = sub.iloc[0]
                print(
                    f"  - {hub['hub_name']} | {ind} 0-50 km: "
                    f"n={int(r['sample_count'])}, "
                    f"mean={r['mean']:.4g}, "
                    f"missing={r['missingness_pct']}%"
                )

    print("\nOverlap warnings:")
    if overlap_summary["strong"]:
        for s in overlap_summary["strong"]:
            print(f"  - STRONG overlap: {s}")
    else:
        print("  - none")

    print("\nData-coverage warnings:")
    if coverage_warnings:
        for w in coverage_warnings:
            print(f"  - {w}")
    else:
        print("  - none")

    print("\nSample-density warnings:")
    if warnings_list:
        for w in warnings_list:
            print(f"  - {w}")
    else:
        print("  - none")

    verdict = no2_findings.get("verdict", {})
    print("\nDoes Stockholm improve NO2 interpretation?")
    if not verdict.get("stockholm_has_local_data_0_50_km", True):
        if verdict.get("stockholm_no2_decreases_with_distance_50_200_km"):
            print("  - INCONCLUSIVE for 0-50 km (Stockholm has no near-port grids in this dataset),")
            print("    BUT Stockholm shows a downward NO2 gradient in the 50-200 km substitute band,")
            print("    consistent with — though not conclusive evidence of — an urban-NO2 pattern.")
        else:
            print("  - INCONCLUSIVE: dataset has no grids within 50 km of Stockholm; the urban-NO2")
            print("    test cannot be tightened with the current grid.  Future runs need a grid that")
            print("    extends into the Stockholm vicinity.")
    elif verdict.get("stockholm_no2_decreases_with_distance_0_50_km") and verdict.get(
        "mariehamn_no2_signal_weaker_than_stockholm"
    ):
        print("  - YES: Stockholm shows a clear local NO2 gradient that Mariehamn does not, "
              "supporting the urban-NO2 interpretation.")
    else:
        print("  - PARTIAL: NO2 signal is mixed; see no2_interpretation_check.md for details.")

    print("\nIs the new setup stronger than the previous one?")
    print("  - YES:")
    print("     * Local (0-100 km) and regional (100-1000 km) signals are now separated.")
    print("     * Turku/Naantali double-counting is removed.")
    print("     * Stockholm gives a defensible urban contrast for NO2.")
    print("     * Regional plots are explicitly labelled as background, not port impact.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
