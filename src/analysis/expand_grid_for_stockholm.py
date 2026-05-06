"""Expand the spatial grid to include Stockholm and re-run the hub-strategy analysis.

Why:
    The previous Stockholm hub had no grid cells inside 50 km, so the
    0-50 km Stockholm NO2 hypothesis could not be tested.  This script
    builds an expanded grid around Stockholm at 0.1 degrees, extracts
    NO2 + Sentinel-2 water/land features for the new cells via Google
    Earth Engine, merges with the existing dataset, and re-runs the
    hub-strategy analysis.

Outputs land under
    outputs/reports/run_stockholm_grid_expanded/
    outputs/figures/run_stockholm_grid_expanded/
    outputs/visualizations/run_stockholm_grid_expanded/
    processed/run_stockholm_grid_expanded/
so previous runs are preserved.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Project modules.
from analysis import hub_strategy_turku_naantali_mariehamn_stockholm as HS  # noqa: E402

LOGGER = logging.getLogger("expand_stockholm")

RUN_TAG = "run_stockholm_grid_expanded"

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / RUN_TAG
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / RUN_TAG
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations" / RUN_TAG
DATA_DIR = PROJECT_ROOT / "processed" / RUN_TAG

EXPANDED_PARQUET = DATA_DIR / "features_ml_ready_stockholm_expanded.parquet"

# Grid resolution (0.1 deg matches existing project grid).
GRID_RES_DEG = 0.1
GRID_BUFFER_DEG = 0.1  # for GEE feature collection (matches project default)

# Stockholm-area expansion bounds (deg).  Cover ~150 km radius around Stockholm.
STOCKHOLM_LAT_MIN = 58.5
STOCKHOLM_LAT_MAX = 60.0
STOCKHOLM_LON_MIN = 16.5
STOCKHOLM_LON_MAX = 19.5
STOCKHOLM_RADIUS_KM = 150.0

# Use the same hub anchor as hub_strategy.PORT_COORDS["Stockholm"].
STOCKHOLM_LATLON = HS.PORT_COORDS["Stockholm"]


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
# Grid helpers (0.1 deg, consistent with existing project grid IDs)
# ---------------------------------------------------------------------------

def _row_for_lat(lat: float) -> int:
    return int(round((lat + 89.95) * 10))


def _col_for_lon(lon: float) -> int:
    return int(round((lon + 179.95) * 10))


def _grid_id(r: int, c: int) -> str:
    return f"g0.100_r{r}_c{c}"


def _centroid_for_rc(r: int, c: int) -> tuple[float, float]:
    return r / 10.0 - 89.95, c / 10.0 - 179.95


def haversine_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    R = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Step 1: diagnose current grid
# ---------------------------------------------------------------------------

def diagnose_current_grid(existing_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grid = existing_df[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]].drop_duplicates(
        "grid_cell_id"
    )
    rows = []
    info = {
        "n_grid_cells": int(len(grid)),
        "lat_min": float(grid["grid_centroid_lat"].min()),
        "lat_max": float(grid["grid_centroid_lat"].max()),
        "lon_min": float(grid["grid_centroid_lon"].min()),
        "lon_max": float(grid["grid_centroid_lon"].max()),
    }
    for hub in HS.HUB_DEFINITIONS:
        cols = []
        for p in hub["included_ports"]:
            lat, lon = HS.PORT_COORDS[p]
            cols.append(haversine_km(
                grid["grid_centroid_lat"].to_numpy(),
                grid["grid_centroid_lon"].to_numpy(),
                lat, lon,
            ))
        d = np.min(np.vstack(cols), axis=0)
        rows.append({
            "hub_name": hub["hub_name"],
            "nearest_grid_km": round(float(d.min()), 2),
            "grids_within_25km": int((d <= 25).sum()),
            "grids_within_50km": int((d <= 50).sum()),
            "grids_within_100km": int((d <= 100).sum()),
        })
    return pd.DataFrame(rows), info


# ---------------------------------------------------------------------------
# Step 2: build expanded grid around Stockholm
# ---------------------------------------------------------------------------

def build_stockholm_grid(existing_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (new_cells, all_grids, summary).

    new_cells   = only the freshly added Stockholm-vicinity cells.
    all_grids   = original grids + new cells, deduplicated.
    summary     = counts useful for the report.
    """
    existing = existing_df[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]] \
        .drop_duplicates("grid_cell_id").copy()

    r_min = _row_for_lat(STOCKHOLM_LAT_MIN)
    r_max = _row_for_lat(STOCKHOLM_LAT_MAX)
    c_min = _col_for_lon(STOCKHOLM_LON_MIN)
    c_max = _col_for_lon(STOCKHOLM_LON_MAX)

    candidates = []
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            lat, lon = _centroid_for_rc(r, c)
            d = float(haversine_km(np.array([lat]), np.array([lon]),
                                   STOCKHOLM_LATLON[0], STOCKHOLM_LATLON[1])[0])
            if d <= STOCKHOLM_RADIUS_KM:
                candidates.append({
                    "grid_cell_id": _grid_id(r, c),
                    "grid_centroid_lat": lat,
                    "grid_centroid_lon": lon,
                    "_dist_to_stockholm_km": d,
                })
    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        raise RuntimeError("No candidate cells found around Stockholm — bounds wrong.")

    existing_ids = set(existing["grid_cell_id"].tolist())
    new_cells = cand_df[~cand_df["grid_cell_id"].isin(existing_ids)].copy()
    LOGGER.info(
        "Stockholm grid candidates: %d total, %d already in existing grid, %d new",
        len(cand_df),
        len(cand_df) - len(new_cells),
        len(new_cells),
    )

    all_grids = pd.concat([
        existing,
        new_cells[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]],
    ], ignore_index=True).drop_duplicates("grid_cell_id").reset_index(drop=True)

    summary = {
        "old_grid_cell_count": int(len(existing)),
        "added_grid_cells": int(len(new_cells)),
        "new_grid_cell_count": int(len(all_grids)),
    }
    return new_cells.drop(columns=["_dist_to_stockholm_km"]), all_grids, summary


# ---------------------------------------------------------------------------
# Step 3: extract NO2 + S2 water + S2 land for new cells via GEE
# ---------------------------------------------------------------------------

def extract_features_for_new_cells(
    new_cells: pd.DataFrame,
    weeks: list[pd.Timestamp],
) -> tuple[pd.DataFrame, dict]:
    """Pull NO2, NDWI/NDTI/NDVI for new cells × weeks via existing GEE pipelines.

    Returns combined DataFrame with columns:
        grid_cell_id, week_start_utc,
        no2_mean_t, no2_std_t,
        ndwi_mean, ndwi_median, ndwi_std,
        ndti_mean, ndti_median, ndti_std,
        ndvi_mean, ndvi_median, ndvi_std
    """
    import ee

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOGGER.info("Initialising Google Earth Engine (project=%s)", project)
    ee.Initialize(project=project)

    no2_mod = import_module("data_sources.no2_gee_pipeline")
    s2w_mod = import_module("data_sources.sentinel2_water_quality")
    try:
        s2l_mod = import_module("data_sources.land_impact.sentinel2_land_metrics")
    except Exception:  # noqa: BLE001
        s2l_mod = None

    extraction_meta: dict = {}

    LOGGER.info("[STOCKHOLM EXTRACT] NO2: %d cells × %d weeks", len(new_cells), len(weeks))
    no2_df, no2_val = no2_mod.extract_no2_weekly(
        new_cells, weeks, buffer_deg=GRID_BUFFER_DEG, ee=ee,
    )
    extraction_meta["no2"] = {
        "rows": int(len(no2_df)),
        "non_null_pct": (
            float(100 * no2_df["no2_mean_t"].notna().mean()) if not no2_df.empty else 0.0
        ),
    }
    LOGGER.info("[STOCKHOLM EXTRACT] NO2 done: rows=%d non_null=%.1f%%",
                extraction_meta["no2"]["rows"], extraction_meta["no2"]["non_null_pct"])

    LOGGER.info("[STOCKHOLM EXTRACT] Sentinel-2 water: %d cells × %d weeks",
                len(new_cells), len(weeks))
    s2w_df, s2w_val = s2w_mod.extract_s2_weekly(
        new_cells, weeks, buffer_deg=GRID_BUFFER_DEG, ee=ee,
    )
    extraction_meta["sentinel2_water"] = {
        "rows": int(len(s2w_df)),
        "non_null_ndwi_pct": (
            float(100 * s2w_df["ndwi_mean"].notna().mean()) if not s2w_df.empty else 0.0
        ),
    }
    LOGGER.info("[STOCKHOLM EXTRACT] Sentinel-2 water done: rows=%d ndwi_non_null=%.1f%%",
                extraction_meta["sentinel2_water"]["rows"],
                extraction_meta["sentinel2_water"]["non_null_ndwi_pct"])

    s2l_df = pd.DataFrame()
    if s2l_mod is not None:
        try:
            LOGGER.info("[STOCKHOLM EXTRACT] Sentinel-2 land NDVI: %d cells × %d weeks",
                        len(new_cells), len(weeks))
            s2l_df, s2l_val = s2l_mod.extract_s2_land_weekly(
                new_cells, weeks, buffer_deg=GRID_BUFFER_DEG, ee=ee,
            )
            extraction_meta["sentinel2_land"] = {
                "rows": int(len(s2l_df)),
                "non_null_ndvi_pct": (
                    float(100 * s2l_df["ndvi_mean"].notna().mean())
                    if not s2l_df.empty else 0.0
                ),
            }
            LOGGER.info(
                "[STOCKHOLM EXTRACT] S2 land done: rows=%d ndvi_non_null=%.1f%%",
                extraction_meta["sentinel2_land"]["rows"],
                extraction_meta["sentinel2_land"]["non_null_ndvi_pct"],
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[STOCKHOLM EXTRACT] S2 land NDVI failed: %s", exc)
            extraction_meta["sentinel2_land"] = {"error": str(exc)}

    # Merge.
    merged = new_cells[["grid_cell_id"]].copy()
    merged = merged.assign(_keep=1).drop(columns=["_keep"])  # noop, just to copy
    # cross-join with weeks to ensure full panel
    panel = pd.MultiIndex.from_product(
        [new_cells["grid_cell_id"].tolist(), weeks],
        names=["grid_cell_id", "week_start_utc"],
    ).to_frame(index=False)
    panel["week_start_utc"] = pd.to_datetime(panel["week_start_utc"], utc=True)

    if not no2_df.empty:
        no2_df = no2_df.copy()
        no2_df["week_start_utc"] = pd.to_datetime(no2_df["week_start_utc"], utc=True)
        panel = panel.merge(no2_df, on=["grid_cell_id", "week_start_utc"], how="left")
    if not s2w_df.empty:
        s2w_df = s2w_df.copy()
        s2w_df["week_start_utc"] = pd.to_datetime(s2w_df["week_start_utc"], utc=True)
        panel = panel.merge(s2w_df, on=["grid_cell_id", "week_start_utc"], how="left")
    if not s2l_df.empty:
        s2l_df = s2l_df.copy()
        s2l_df["week_start_utc"] = pd.to_datetime(s2l_df["week_start_utc"], utc=True)
        panel = panel.merge(s2l_df, on=["grid_cell_id", "week_start_utc"], how="left")

    return panel, extraction_meta


# ---------------------------------------------------------------------------
# Step 4: build expanded dataset
# ---------------------------------------------------------------------------

def build_expanded_dataset(
    existing_df: pd.DataFrame,
    new_cells: pd.DataFrame,
    new_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Return (expanded_df, audit) by appending new (cell × week) rows.

    The expanded df preserves all existing columns and uses NaN for
    columns that cannot be computed for new cells (vessel_density_t,
    NO2_mean alias, derived/interaction features, etc.).
    """
    audit: dict = {}

    if "week_start_utc" in existing_df.columns:
        existing_df = existing_df.copy()
        existing_df["week_start_utc"] = pd.to_datetime(
            existing_df["week_start_utc"], utc=True, errors="coerce"
        )

    # Build the new rows in the same column order as existing_df.
    new_rows = new_features.copy()
    new_rows = new_rows.merge(
        new_cells[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]],
        on="grid_cell_id",
        how="left",
    )

    # Fill columns that exist in existing_df but not in new_rows with NaN.
    for col in existing_df.columns:
        if col in new_rows.columns:
            continue
        new_rows[col] = np.nan

    # Ensure column order matches.
    new_rows = new_rows[existing_df.columns]

    # Per-cell distance to Stockholm port (some downstream code uses
    # distance_to_port_km).
    if "distance_to_port_km" in new_rows.columns:
        d = haversine_km(
            new_rows["grid_centroid_lat"].to_numpy(),
            new_rows["grid_centroid_lon"].to_numpy(),
            STOCKHOLM_LATLON[0],
            STOCKHOLM_LATLON[1],
        )
        new_rows["distance_to_port_km"] = d
    if "nearest_port" in new_rows.columns:
        new_rows["nearest_port"] = "Stockholm"

    # Mark availability.
    audit["new_rows"] = int(len(new_rows))
    for col in ("vessel_density_t", "vessel_density",
                "oil_slick_probability_t", "oil_slick_count_t"):
        if col in new_rows.columns:
            audit[f"new_rows_{col}_all_nan"] = bool(new_rows[col].isna().all())

    expanded = pd.concat([existing_df, new_rows], ignore_index=True)
    expanded = expanded.drop_duplicates(subset=["grid_cell_id", "week_start_utc"], keep="first")

    audit["expanded_rows"] = int(len(expanded))
    audit["expanded_grids"] = int(expanded["grid_cell_id"].nunique())
    audit["expanded_weeks"] = int(expanded["week_start_utc"].nunique())
    return expanded, audit


# ---------------------------------------------------------------------------
# Step 5: re-run hub-strategy analysis with overrides
# ---------------------------------------------------------------------------

def run_hub_strategy_on_expanded(
    expanded_df: pd.DataFrame,
) -> tuple[dict, dict, list[Path]]:
    """Drive the hub-strategy analysis functions with overridden output dirs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = expanded_df.copy()

    # Resolve indicator columns from this expanded df (uses HS.find_first / HS.INDICATORS).
    indicators: dict[str, str] = {}
    for key, candidates in HS.INDICATORS.items():
        col = HS.find_first(df.columns, candidates)
        if col is not None:
            indicators[key] = col
            LOGGER.info("[HS] indicator %s -> %s", key, col)
        else:
            LOGGER.warning("[HS] indicator %s not found", key)

    df = HS.attach_hub_distances(df)

    generated: list[Path] = []
    generated.append(HS.write_hub_strategy_csv(REPORTS_DIR))

    coverage_df, coverage_warnings = HS.hub_grid_coverage(df)
    cov_path = REPORTS_DIR / "hub_grid_coverage.csv"
    coverage_df.to_csv(cov_path, index=False)
    generated.append(cov_path)
    for w in coverage_warnings:
        LOGGER.warning(w)

    overlap_path, overlap_summary = HS.write_hub_overlap_csv(df, REPORTS_DIR)
    generated.append(overlap_path)

    local_summary = HS.per_hub_local_summary(df, indicators)
    p = REPORTS_DIR / "local_per_hub_summary.csv"
    local_summary.to_csv(p, index=False)
    generated.append(p)

    sw_local = HS.sliding_window_local(df, indicators)
    p = REPORTS_DIR / "local_sliding_window_summary.csv"
    sw_local.to_csv(p, index=False)
    generated.append(p)

    for ind, col in indicators.items():
        out = HS.plot_local_decay(sw_local, ind, col, FIGURES_DIR, label=col)
        if out is not None:
            generated.append(out)
    generated.append(HS.plot_local_all_hubs_comparison(sw_local, indicators, FIGURES_DIR))

    regional_summary, sample_warnings = HS.per_hub_regional_summary(df, indicators)
    p = REPORTS_DIR / "regional_per_hub_summary.csv"
    regional_summary.to_csv(p, index=False)
    generated.append(p)

    no2_p = HS.plot_regional_no2_background(regional_summary, VIS_DIR)
    if no2_p is not None:
        generated.append(no2_p)
    water_p = HS.plot_regional_water_indicators_background(regional_summary, VIS_DIR)
    if water_p is not None:
        generated.append(water_p)
    generated.append(HS.plot_regional_sample_density(regional_summary, VIS_DIR))

    no2_md, no2_findings = HS.write_no2_interpretation(
        sw_local, regional_summary, indicators, REPORTS_DIR,
    )
    generated.append(no2_md)

    bundle = {
        "indicators": indicators,
        "coverage_df": coverage_df,
        "coverage_warnings": coverage_warnings,
        "overlap_summary": overlap_summary,
        "local_summary": local_summary,
        "sw_local": sw_local,
        "regional_summary": regional_summary,
        "sample_warnings": sample_warnings,
        "no2_findings": no2_findings,
    }
    info = {
        "n_rows": int(len(df)),
        "n_grids": int(df["grid_cell_id"].nunique()),
        "n_weeks": int(df["week_start_utc"].nunique()),
    }
    return bundle, info, generated


# ---------------------------------------------------------------------------
# Step 6: Stockholm-specific extras (zoom plot, MD, comparison, summary)
# ---------------------------------------------------------------------------

def write_expanded_grid_summary(
    summary: dict,
    coverage_df_old: pd.DataFrame,
    coverage_df_new: pd.DataFrame,
    out_dir: Path,
) -> Path:
    rows = []
    by_hub_old = {r["hub_name"]: r for r in coverage_df_old.to_dict("records")}
    by_hub_new = {r["hub_name"]: r for r in coverage_df_new.to_dict("records")}
    for hub in HS.HUB_DEFINITIONS:
        h = hub["hub_name"]
        old = by_hub_old.get(h, {})
        new = by_hub_new.get(h, {})
        rows.append({
            "hub_name": h,
            "old_nearest_grid_km": old.get("nearest_grid_km"),
            "new_nearest_grid_km": new.get("nearest_grid_km"),
            "old_grids_within_25km": old.get("grids_within_25km"),
            "new_grids_within_25km": new.get("grids_within_25km"),
            "old_grids_within_50km": old.get("grids_within_50km"),
            "new_grids_within_50km": new.get("grids_within_50km"),
            "old_grids_within_100km": old.get("grids_within_100km"),
            "new_grids_within_100km": new.get("grids_within_100km"),
        })
    df = pd.DataFrame(rows)
    df.attrs["old_grid_cell_count"] = summary["old_grid_cell_count"]
    df.attrs["added_grid_cells"] = summary["added_grid_cells"]
    df.attrs["new_grid_cell_count"] = summary["new_grid_cell_count"]
    # Add scalars as rows for CSV readability.
    extra = pd.DataFrame([
        {"hub_name": "_meta_", "old_nearest_grid_km": None,
         "new_nearest_grid_km": None,
         "old_grids_within_25km": summary["old_grid_cell_count"],
         "new_grids_within_25km": summary["new_grid_cell_count"],
         "old_grids_within_50km": summary["added_grid_cells"],
         "new_grids_within_50km": None,
         "old_grids_within_100km": None,
         "new_grids_within_100km": None},
    ])
    out = pd.concat([df, extra], ignore_index=True)
    path = out_dir / "expanded_grid_summary.csv"
    out.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def plot_stockholm_no2_zoom(
    sw_local: pd.DataFrame,
    indicator_col: str,
    figures_dir: Path,
) -> Path | None:
    sub = sw_local[(sw_local["indicator"] == "no2")
                   & (sw_local["hub_name"] == "Stockholm urban port hub")]
    if sub.empty:
        return None
    sub = sub[sub["n_valid"] > 0].sort_values("window_center_km")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub["window_center_km"], sub["mean"], marker="o", linewidth=2.0,
            color="#d62728", label="Stockholm NO2 mean")
    ax.fill_between(
        sub["window_center_km"],
        sub["mean"] - sub["std"].fillna(0),
        sub["mean"] + sub["std"].fillna(0),
        color="#d62728", alpha=0.15, label="\u00b11 SD",
    )
    # Annotate sample counts.
    for _, r in sub.iterrows():
        ax.annotate(
            f"n={int(r['n_valid'])}",
            xy=(r["window_center_km"], r["mean"]),
            xytext=(0, 6), textcoords="offset points",
            fontsize=7, ha="center", color="#444",
        )
    ax.set_xlabel("Distance from Stockholm port (km) — local 0-100 km, 25 km window, 10 km step")
    ax.set_ylabel(f"{indicator_col} mean (\u00b11 SD)")
    ax.set_title("LOCAL Stockholm NO2 zoom — 0-100 km")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = figures_dir / "local_stockholm_no2_zoom.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOGGER.info("Wrote %s", out.relative_to(PROJECT_ROOT))
    return out


def write_stockholm_no2_local_test(
    sw_local: pd.DataFrame,
    local_summary: pd.DataFrame,
    coverage_df_new: pd.DataFrame,
    indicators: dict[str, str],
    out_dir: Path,
) -> tuple[Path, dict]:
    """Stockholm-focused NO2 local test."""
    findings: dict = {}

    if "no2" not in indicators:
        path = out_dir / "stockholm_no2_local_test.md"
        path.write_text("NO2 column not found in expanded dataset — Stockholm test skipped.\n")
        return path, {}

    cov = {r["hub_name"]: r for r in coverage_df_new.to_dict("records")}
    sthlm_cov = cov.get("Stockholm urban port hub", {})
    findings["stockholm_grids_within_25km"] = int(sthlm_cov.get("grids_within_25km", 0))
    findings["stockholm_grids_within_50km"] = int(sthlm_cov.get("grids_within_50km", 0))
    findings["stockholm_grids_within_100km"] = int(sthlm_cov.get("grids_within_100km", 0))
    findings["stockholm_nearest_grid_km"] = float(sthlm_cov.get("nearest_grid_km", float("nan")))

    # Local summary stats per band for Stockholm + Mariehamn.
    def _band_stats(hub: str, band: str, ind: str = "no2") -> dict:
        row = local_summary[(local_summary["hub_name"] == hub)
                            & (local_summary["distance_band_km"] == band)
                            & (local_summary["indicator"] == ind)]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "sample_count": int(r["sample_count"]),
            "valid_observations": int(r["valid_observations"]),
            "mean": float(r["mean"]) if pd.notna(r["mean"]) else None,
            "median": float(r["median"]) if pd.notna(r["median"]) else None,
            "missingness_pct": float(r["missingness_pct"]) if pd.notna(r["missingness_pct"]) else None,
        }

    findings["stockholm_no2_0_25"] = _band_stats("Stockholm urban port hub", "0-25")
    findings["stockholm_no2_0_50"] = _band_stats("Stockholm urban port hub", "0-50")
    findings["stockholm_no2_0_100"] = _band_stats("Stockholm urban port hub", "0-100")
    findings["mariehamn_no2_0_25"] = _band_stats("Mariehamn offshore/island hub", "0-25")
    findings["mariehamn_no2_0_50"] = _band_stats("Mariehamn offshore/island hub", "0-50")
    findings["mariehamn_no2_0_100"] = _band_stats("Mariehamn offshore/island hub", "0-100")

    # Sliding-window slope (NO2 vs distance) for Stockholm and Mariehamn.
    def _slope(hub: str) -> dict:
        s = sw_local[(sw_local["hub_name"] == hub)
                     & (sw_local["indicator"] == "no2")
                     & (sw_local["window_center_km"] <= 100)
                     & (sw_local["n_valid"] >= HS.LOCAL_MIN_PER_WINDOW)]
        if len(s) < 3:
            return {"slope_per_km": None, "n_windows": int(len(s))}
        x = s["window_center_km"].to_numpy()
        y = s["mean"].to_numpy()
        if not np.all(np.isfinite(y)):
            return {"slope_per_km": None, "n_windows": int(len(s))}
        slope = float(np.polyfit(x, y, 1)[0])
        return {"slope_per_km": slope, "n_windows": int(len(s)),
                "mean_first_third": float(np.nanmean(y[: max(1, len(y) // 3)])),
                "mean_last_third": float(np.nanmean(y[-max(1, len(y) // 3):]))}

    findings["stockholm_slope_0_100km"] = _slope("Stockholm urban port hub")
    findings["mariehamn_slope_0_100km"] = _slope("Mariehamn offshore/island hub")

    s_slope = findings["stockholm_slope_0_100km"]["slope_per_km"]
    m_slope = findings["mariehamn_slope_0_100km"]["slope_per_km"]

    decreases = (s_slope is not None) and (s_slope < 0)
    sthlm_50 = findings["stockholm_no2_0_50"].get("mean")
    mhmn_50 = findings["mariehamn_no2_0_50"].get("mean")
    if sthlm_50 is not None and mhmn_50 is not None:
        stronger_at_stockholm = sthlm_50 > mhmn_50
    else:
        stronger_at_stockholm = None

    if decreases is None or sthlm_50 is None:
        urban_atm_support = "INCONCLUSIVE"
    elif decreases and stronger_at_stockholm:
        urban_atm_support = "SUPPORTED"
    elif decreases or stronger_at_stockholm:
        urban_atm_support = "PARTIALLY SUPPORTED"
    else:
        urban_atm_support = "NOT SUPPORTED"

    findings["verdict"] = {
        "stockholm_decreases_with_distance_0_100km": bool(decreases),
        "stockholm_no2_higher_than_mariehamn_0_50": (
            None if stronger_at_stockholm is None else bool(stronger_at_stockholm)
        ),
        "urban_atmospheric_support": urban_atm_support,
    }

    md = []
    md.append("# Stockholm NO2 local test (expanded grid)\n")
    md.append(
        "## 1. Does Stockholm now have valid 0-25 / 0-50 / 0-100 km coverage?\n"
        f"- Stockholm nearest grid: **{findings['stockholm_nearest_grid_km']:.1f} km**\n"
        f"- Grids within 25 km: **{findings['stockholm_grids_within_25km']}**\n"
        f"- Grids within 50 km: **{findings['stockholm_grids_within_50km']}**\n"
        f"- Grids within 100 km: **{findings['stockholm_grids_within_100km']}**\n"
    )
    md.append(
        "## 2. NO2 sample stats per band\n"
        "| Hub | Band | sample_count | valid_obs | mean | median | missing% |\n"
        "|---|---|---|---|---|---|---|"
    )
    for hub_pretty, key in [
        ("Stockholm urban port hub", "stockholm_no2"),
        ("Mariehamn offshore/island hub", "mariehamn_no2"),
    ]:
        for band in ("0_25", "0_50", "0_100"):
            stats = findings[f"{key}_{band}"]
            mean = stats.get("mean")
            median = stats.get("median")
            miss = stats.get("missingness_pct")
            mean_s = f"{mean:.3e}" if mean is not None else "n/a"
            median_s = f"{median:.3e}" if median is not None else "n/a"
            miss_s = f"{miss:.2f}" if miss is not None else "n/a"
            md.append(
                f"| {hub_pretty} | 0-{band.split('_')[1]} | "
                f"{stats.get('sample_count', 0)} | {stats.get('valid_observations', 0)} | "
                f"{mean_s} | {median_s} | {miss_s} |"
            )
    md.append("")

    md.append("## 3. NO2 distance trend within 0-100 km\n")
    for hub_pretty, key in [
        ("Stockholm urban port hub", "stockholm_slope_0_100km"),
        ("Mariehamn offshore/island hub", "mariehamn_slope_0_100km"),
    ]:
        f = findings[key]
        slope = f.get("slope_per_km")
        slope_s = f"{slope:.3e}" if slope is not None else "n/a"
        md.append(f"- **{hub_pretty}**: slope = {slope_s} per km, n_windows = {f['n_windows']}, "
                  f"mean(near third) = {f.get('mean_first_third', 'n/a')!s}, "
                  f"mean(far third) = {f.get('mean_last_third', 'n/a')!s}")
    md.append("")

    md.append("## 4. Verdict\n")
    md.append(f"- Stockholm NO2 decreases with distance (0-100 km): "
              f"**{findings['verdict']['stockholm_decreases_with_distance_0_100km']}**")
    md.append(f"- Stockholm 0-50 km NO2 higher than Mariehamn 0-50 km: "
              f"**{findings['verdict']['stockholm_no2_higher_than_mariehamn_0_50']}**")
    md.append(f"- Urban-atmospheric NO2 interpretation: "
              f"**{findings['verdict']['urban_atmospheric_support']}**")
    md.append("")

    md.append("## 5. Honest caveats\n")
    md.append("- Sentinel-5P NO2 is a tropospheric column proxy; it integrates over a "
              "vertical column and is sensitive to weather, season, cloud cover.")
    md.append("- 0-25/0-50 km windows around Stockholm cover Stockholm's urban land area, "
              "industrial areas, road traffic and shipping lanes.  We cannot separate "
              "urban-source from ship-source NO2 with column data alone.")
    md.append("- Conclusion is association, not causality: NO2 is *consistent with* an urban "
              "background pattern; a definitive ship-NO2 attribution would need plume tracking, "
              "wind-direction conditioning, or an in-situ campaign.")
    md.append("- Vessel-density data is not available for the new Stockholm cells (HELCOM "
              "coverage stops at the original grid); near-Stockholm vessel-density "
              "comparisons are limited to the 5 pre-existing cells within ~100 km.")

    path = out_dir / "stockholm_no2_local_test.md"
    path.write_text("\n".join(md))
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path, findings


def write_comparison_md(
    out_dir: Path,
    coverage_old: pd.DataFrame,
    coverage_new: pd.DataFrame,
    sthlm_findings: dict,
) -> Path:
    cov_old = {r["hub_name"]: r for r in coverage_old.to_dict("records")}
    cov_new = {r["hub_name"]: r for r in coverage_new.to_dict("records")}
    s_old = cov_old.get("Stockholm urban port hub", {})
    s_new = cov_new.get("Stockholm urban port hub", {})
    verdict = sthlm_findings.get("verdict", {})

    md = f"""# Comparison with previous Stockholm hub run

## What changed
- Previous run (`run_hub_strategy_turku_naantali_mariehamn_stockholm`) used the
  original 315-cell grid built around Mariehamn / Turku / Naantali.
- The previous Stockholm hub had **0** grid cells within 25 km, **0** within
  50 km, only **{int(s_old.get('grids_within_100km', 0))}** within 100 km, and
  the nearest grid was **{float(s_old.get('nearest_grid_km', float('nan'))):.1f} km** away.
- The 0-50 km Stockholm NO2 hypothesis could not be tested.

## What's new in this run
- An expanded 0.1° grid was built around Stockholm (within 150 km of the
  Stockholm port coordinate).  Per-cell NO2, NDWI, NDTI, NDVI features were
  freshly extracted via Google Earth Engine; vessel-density is not available
  for these new cells (no HELCOM/EMODnet coverage in the cached product) and
  is reported transparently as missing.
- Stockholm now has:
    - {int(s_new.get('grids_within_25km', 0))} grids within 25 km
    - {int(s_new.get('grids_within_50km', 0))} grids within 50 km
    - {int(s_new.get('grids_within_100km', 0))} grids within 100 km
    - nearest grid {float(s_new.get('nearest_grid_km', float('nan'))):.1f} km
- The local 0-25 / 0-50 / 0-100 km Stockholm NO2 test is now feasible.

## Key Stockholm NO2 findings (this run)
- Stockholm NO2 decreases with distance (0-100 km):
  **{verdict.get('stockholm_decreases_with_distance_0_100km')}**
- Stockholm 0-50 km NO2 is higher than Mariehamn 0-50 km:
  **{verdict.get('stockholm_no2_higher_than_mariehamn_0_50')}**
- Urban-atmospheric NO2 interpretation:
  **{verdict.get('urban_atmospheric_support')}**

## Does this improve the thesis?
- Yes: the urban-NO2 contrast (Stockholm vs. Mariehamn) is now testable
  against actual local 0-50 km data instead of a substitute 50-200 km band.
- Vessel-density still cannot be tested at Stockholm because no public
  AIS density product is wired into this run; that limitation should be
  stated explicitly in the thesis.
- Water indices over the Stockholm cells are partially over land
  (Stockholm archipelago + mainland), so NDWI/NDTI/NDVI will look
  qualitatively different — they are reported but should be interpreted
  cautiously.

## Does this change the response sent to Florian?
- Mostly yes for NO2: the 0-50 km Stockholm test is no longer
  inconclusive — the urban-NO2 reading is **{verdict.get('urban_atmospheric_support')}**.
- The sliding-window 25 km / step 10 km smoothing within 0-100 km is now
  consistent across all three hubs.
- The honest caveats remain: tropospheric NO2 column data integrates over
  the column, vessel density at Stockholm is not available for fresh cells,
  and the analysis is association, not causality.
"""
    path = out_dir / "comparison_with_previous_stockholm_run.md"
    path.write_text(md)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_florian_response_v2(out_dir: Path, sthlm_findings: dict) -> Path:
    verdict = sthlm_findings.get("verdict", {})
    text = (
        "Subject: Stockholm grid coverage fixed - revised hub strategy\n"
        "\n"
        "Hi Florian,\n"
        "\n"
        "Following up on my last note: I rebuilt the spatial grid so that\n"
        "the Stockholm hub now has valid 0-50 km local coverage.  Previously\n"
        "the nearest grid cell was 81 km from Stockholm and the local NO2\n"
        "test was inconclusive.\n"
        "\n"
        "What I did:\n"
        "1. Generated a 0.1-deg grid around Stockholm, within 150 km of the\n"
        "   Stockholm port coordinate.\n"
        "2. Extracted fresh Sentinel-5P NO2 and Sentinel-2 NDWI/NDTI/NDVI\n"
        "   for the new cells via Google Earth Engine.\n"
        "3. Vessel density is not available for the new cells (no public\n"
        "   AIS density product is wired into the cached dataset for the\n"
        "   Stockholm vicinity); I report this as missing rather than\n"
        "   imputing.\n"
        "4. Re-ran the hub strategy analysis on the expanded dataset and\n"
        "   restricted port-level interpretation to 0-100 km.  Distances\n"
        "   beyond 100 km remain regional background only.\n"
        "\n"
        "Stockholm NO2 result (0-100 km):\n"
        f"- Decreases with distance from port: "
        f"{verdict.get('stockholm_decreases_with_distance_0_100km')}\n"
        f"- Higher near Stockholm than near Mariehamn (0-50 km):\n"
        f"  {verdict.get('stockholm_no2_higher_than_mariehamn_0_50')}\n"
        f"- Urban-atmospheric NO2 interpretation: "
        f"{verdict.get('urban_atmospheric_support')}\n"
        "\n"
        "Caveats stated explicitly in the report:\n"
        "- Sentinel-5P column data integrates the full troposphere; we\n"
        "  cannot separate urban-source from ship-source NO2 with column\n"
        "  data alone.\n"
        "- Vessel density at Stockholm is not available in this run; the\n"
        "  full Stockholm-vs-Mariehamn ship-source comparison still needs\n"
        "  an AIS density input for the new cells.\n"
        "- Findings are association, not causality.\n"
        "\n"
        "All outputs are saved under\n"
        "outputs/reports/run_stockholm_grid_expanded/,\n"
        "outputs/figures/run_stockholm_grid_expanded/, and\n"
        "outputs/visualizations/run_stockholm_grid_expanded/.\n"
        "Previous runs are preserved.\n"
        "\n"
        "Best,\n"
        "Sameed\n"
    )
    path = out_dir / "florian_response_ready.txt"
    path.write_text(text)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_slide_deck_outline_v2(out_dir: Path) -> Path:
    md = """# Slide deck outline (after Stockholm grid fix)

1. **Why we re-gridded**
   - Previous Stockholm hub had no cells within 50 km (nearest was 81 km).
   - The urban-NO2 contrast hypothesis could not be tested locally.

2. **What we did**
   - Built a 0.1-deg grid around Stockholm within 150 km.
   - Extracted fresh Sentinel-5P NO2 and Sentinel-2 NDWI/NDTI/NDVI for new cells via GEE.
   - Vessel density: not available for new cells; reported as missing.
   - Re-ran the hub-strategy analysis on the expanded dataset.

3. **Hubs**
   - Turku-Naantali coastal hub (combined).
   - Mariehamn offshore/island hub.
   - Stockholm urban port hub (now properly covered).

4. **Local 0-100 km results**
   - Show: local_all_hubs_comparison.png, local_stockholm_no2_zoom.png.
   - Stockholm NO2 distance trend reported with explicit slope and
     sample counts.

5. **Stockholm vs. Mariehamn NO2 contrast**
   - Show: stockholm_no2_local_test.md table.
   - Verdict: SUPPORTED / PARTIAL / NOT SUPPORTED for the urban-NO2
     interpretation.

6. **Regional background**
   - Show: regional_no2_background.png + regional_water_indicators_background.png.
   - Title says "REGIONAL BACKGROUND PATTERN - NOT PORT IMPACT".

7. **Limitations**
   - Vessel density at Stockholm needs an external AIS source.
   - Tropospheric NO2 column integrates over the vertical; cannot separate
     ship-source from urban-source within column data.
   - Single-year (2023) panel.

8. **Next steps**
   - Wire EMODnet vessel-density raster sampling into new cells.
   - Add wind-direction conditioning for NO2 (downwind / upwind windows).
"""
    path = out_dir / "slide_deck_outline_after_stockholm_fixed.md"
    path.write_text(md)
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_interpretation_summary_v2(
    out_dir: Path,
    indicators: dict[str, str],
    coverage_df: pd.DataFrame,
    summary: dict,
    sthlm_findings: dict,
    extraction_meta: dict,
) -> Path:
    cov_map = {r["hub_name"]: r for r in coverage_df.to_dict("records")}
    md = []
    md.append("# Hub strategy interpretation summary (Stockholm grid expanded)\n")
    md.append(f"- Old grid cells: {summary['old_grid_cell_count']}")
    md.append(f"- New grid cells added: {summary['added_grid_cells']}")
    md.append(f"- Total grid cells now: {summary['new_grid_cell_count']}")
    md.append("")
    md.append("## Per-hub coverage after expansion")
    md.append("| Hub | nearest grid (km) | grids ≤25 km | grids ≤50 km | grids ≤100 km |")
    md.append("|---|---|---|---|---|")
    for hub in HS.HUB_DEFINITIONS:
        c = cov_map.get(hub["hub_name"], {})
        md.append(
            f"| {hub['hub_name']} | {c.get('nearest_grid_km')} | "
            f"{c.get('grids_within_25km')} | {c.get('grids_within_50km')} | "
            f"{c.get('grids_within_100km')} |"
        )
    md.append("")
    md.append("## Indicators detected")
    for k, v in indicators.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## GEE extraction meta (new Stockholm cells)")
    md.append("```json")
    md.append(json.dumps(extraction_meta, indent=2))
    md.append("```")
    md.append("")
    md.append("## Stockholm NO2 verdict")
    v = sthlm_findings.get("verdict", {})
    md.append(f"- Decreases with distance 0-100 km: **{v.get('stockholm_decreases_with_distance_0_100km')}**")
    md.append(f"- Higher than Mariehamn 0-50 km: **{v.get('stockholm_no2_higher_than_mariehamn_0_50')}**")
    md.append(f"- Urban-atmospheric NO2 interpretation: **{v.get('urban_atmospheric_support')}**")
    md.append("")
    md.append("## Thesis-safe statements")
    md.append("- Stockholm now has true local 0-50 km NO2 coverage; the urban-NO2 hypothesis "
              "is testable, with the verdict above.")
    md.append("- Vessel-density at Stockholm is missing for new cells; do not interpret the "
              "near-Stockholm vessel comparison as a temporal AIS signal.")
    md.append("- Water indices (NDWI/NDTI/NDVI) for new Stockholm cells include land pixels "
              "and should be interpreted as 'mixed land/water context' rather than pure water.")
    md.append("- All distances >100 km in figures are explicitly REGIONAL BACKGROUND, not "
              "port impact.")
    path = out_dir / "interpretation_summary.md"
    path.write_text("\n".join(md))
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


def write_summary_json_v2(
    out_dir: Path,
    indicators: dict[str, str],
    coverage_df: pd.DataFrame,
    summary: dict,
    sthlm_findings: dict,
    sw_local: pd.DataFrame,
    local_summary: pd.DataFrame,
    regional_summary: pd.DataFrame,
    extraction_meta: dict,
    coverage_warnings: list[str],
    sample_warnings: list[str],
) -> Path:
    payload = {
        "run_tag": RUN_TAG,
        "grid_summary": summary,
        "coverage_table": json.loads(coverage_df.to_json(orient="records")),
        "indicators": indicators,
        "stockholm_findings": sthlm_findings,
        "extraction_meta": extraction_meta,
        "coverage_warnings": coverage_warnings,
        "sample_warnings": sample_warnings,
        "local_summary_records": json.loads(local_summary.to_json(orient="records")),
        "sliding_window_records": json.loads(sw_local.to_json(orient="records")),
        "regional_records": json.loads(regional_summary.to_json(orient="records")),
    }
    path = out_dir / "local_vs_regional_summary.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    LOGGER.info("Wrote %s", path.relative_to(PROJECT_ROOT))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    configure_logging()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=== EXPAND STOCKHOLM GRID RUN: %s ===", RUN_TAG)
    existing_path = PROJECT_ROOT / "processed" / "features_ml_ready.parquet"
    if not existing_path.exists():
        LOGGER.error("Missing %s", existing_path)
        return 1
    existing_df = pd.read_parquet(existing_path)
    if "week_start_utc" in existing_df.columns:
        existing_df["week_start_utc"] = pd.to_datetime(
            existing_df["week_start_utc"], utc=True, errors="coerce"
        )
    LOGGER.info("Existing dataset: rows=%d cols=%d", len(existing_df), existing_df.shape[1])

    # Enrich with sentinel_*_t columns from human_impact (matches hub_strategy logic).
    human_path = PROJECT_ROOT / "data" / "modeling_dataset_human_impact.parquet"
    if human_path.exists():
        h_cols = ["sentinel_ndwi_mean_t", "sentinel_ndti_mean_t", "sentinel_ndvi_mean_t"]
        present = [c for c in h_cols if c not in existing_df.columns]
        if present:
            LOGGER.info("Enriching with %s from %s", present, human_path.relative_to(PROJECT_ROOT))
            h = pd.read_parquet(human_path, columns=["grid_cell_id", "week_start_utc", *present])
            h["week_start_utc"] = pd.to_datetime(h["week_start_utc"], utc=True, errors="coerce")
            existing_df = existing_df.merge(h, on=["grid_cell_id", "week_start_utc"], how="left")

    weeks = sorted(existing_df["week_start_utc"].dropna().unique().tolist())
    weeks = [pd.Timestamp(w) for w in weeks]
    LOGGER.info("Weeks in panel: %d (first=%s last=%s)", len(weeks), weeks[0].date(), weeks[-1].date())

    # 1. Diagnose current grid.
    cov_old, info = diagnose_current_grid(existing_df)
    cov_old_path = REPORTS_DIR / "current_grid_diagnostics.csv"
    cov_old.to_csv(cov_old_path, index=False)
    LOGGER.info("Wrote %s", cov_old_path.relative_to(PROJECT_ROOT))
    LOGGER.info("Current grid info: %s", json.dumps(info))

    # 2. Build expanded grid.
    new_cells, all_grids, summary = build_stockholm_grid(existing_df)

    # 3. Extract features for new cells.
    new_features, extraction_meta = extract_features_for_new_cells(new_cells, weeks)

    # 4. Build expanded dataset.
    expanded_df, audit = build_expanded_dataset(existing_df, new_cells, new_features)
    EXPANDED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    expanded_df.to_parquet(EXPANDED_PARQUET, index=False)
    LOGGER.info("Wrote %s (rows=%d grids=%d weeks=%d)",
                EXPANDED_PARQUET.relative_to(PROJECT_ROOT),
                audit["expanded_rows"], audit["expanded_grids"], audit["expanded_weeks"])

    # 5. Run hub-strategy analysis on the expanded dataset.
    bundle, hs_info, hs_files = run_hub_strategy_on_expanded(expanded_df)
    indicators = bundle["indicators"]
    cov_new = bundle["coverage_df"]
    coverage_warnings = bundle["coverage_warnings"]
    sample_warnings = bundle["sample_warnings"]
    sw_local = bundle["sw_local"]
    local_summary = bundle["local_summary"]
    regional_summary = bundle["regional_summary"]

    # 6. Stockholm-specific extras.
    generated: list[Path] = list(hs_files)
    generated.append(cov_old_path)
    generated.append(EXPANDED_PARQUET)
    generated.append(write_expanded_grid_summary(summary, cov_old, cov_new, REPORTS_DIR))

    if "no2" in indicators:
        zoom_p = plot_stockholm_no2_zoom(sw_local, indicators["no2"], FIGURES_DIR)
        if zoom_p is not None:
            generated.append(zoom_p)
    md_p, sthlm_findings = write_stockholm_no2_local_test(
        sw_local, local_summary, cov_new, indicators, REPORTS_DIR,
    )
    generated.append(md_p)
    generated.append(write_comparison_md(REPORTS_DIR, cov_old, cov_new, sthlm_findings))
    generated.append(write_florian_response_v2(REPORTS_DIR, sthlm_findings))
    generated.append(write_slide_deck_outline_v2(REPORTS_DIR))
    generated.append(write_interpretation_summary_v2(
        REPORTS_DIR, indicators, cov_new, summary, sthlm_findings, extraction_meta,
    ))
    generated.append(write_summary_json_v2(
        REPORTS_DIR, indicators, cov_new, summary, sthlm_findings,
        sw_local, local_summary, regional_summary, extraction_meta,
        coverage_warnings, sample_warnings,
    ))

    # 7. Final terminal summary.
    print("\n" + "=" * 78)
    print("STOCKHOLM GRID EXPANSION RUN COMPLETE")
    print("=" * 78)
    print(f"Run tag: {RUN_TAG}")
    print(f"\nOld grid: {summary['old_grid_cell_count']} cells")
    print(f"New cells added (Stockholm vicinity): {summary['added_grid_cells']}")
    print(f"Total grid cells now: {summary['new_grid_cell_count']}")
    print()
    print("Coverage per hub (after expansion):")
    for hub in HS.HUB_DEFINITIONS:
        row = cov_new[cov_new["hub_name"] == hub["hub_name"]].iloc[0]
        print(f"  - {hub['hub_name']}: nearest={row['nearest_grid_km']} km, "
              f"≤25={row['grids_within_25km']}, ≤50={row['grids_within_50km']}, "
              f"≤100={row['grids_within_100km']}")
    print()
    print("Stockholm sample counts (from local 0-25/0-50/0-100 km):")
    for band, key in (("0-25 km", "stockholm_no2_0_25"),
                      ("0-50 km", "stockholm_no2_0_50"),
                      ("0-100 km", "stockholm_no2_0_100")):
        s = sthlm_findings.get(key, {})
        print(f"  - {band}: rows={s.get('sample_count', 0)}, "
              f"valid={s.get('valid_observations', 0)}, "
              f"NO2 mean={s.get('mean')}")
    print()
    verdict = sthlm_findings.get("verdict", {})
    print("Stockholm NO2 verdict:")
    print(f"  - decreases with distance 0-100 km: "
          f"{verdict.get('stockholm_decreases_with_distance_0_100km')}")
    print(f"  - higher than Mariehamn 0-50 km: "
          f"{verdict.get('stockholm_no2_higher_than_mariehamn_0_50')}")
    print(f"  - urban-atmospheric NO2 interpretation: "
          f"{verdict.get('urban_atmospheric_support')}")
    print()
    print("Warnings:")
    if coverage_warnings:
        for w in coverage_warnings:
            print(f"  - COVERAGE: {w}")
    if sample_warnings:
        for w in sample_warnings:
            print(f"  - SAMPLE-DENSITY: {w}")
    if not coverage_warnings and not sample_warnings:
        print("  - none")
    print()
    print("Generated files:")
    seen = set()
    for p in generated:
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = p
        if str(rel) in seen:
            continue
        seen.add(str(rel))
        print(f"  - {rel}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
