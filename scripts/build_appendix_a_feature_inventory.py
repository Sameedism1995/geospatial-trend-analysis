#!/usr/bin/env python3
"""Build Appendix A: complete ML feature inventory (markdown + CSV)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "outputs/reports/appendix_a_feature_inventory.md"
OUT_CSV = ROOT / "outputs/reports/appendix_a_feature_inventory.csv"

ML_READY = ROOT / "processed/features_ml_ready.parquet"
MERGED = ROOT / "processed/merged_dataset.parquet"
MODELING = ROOT / "data/modeling_dataset.parquet"

# User-facing categories (thesis Appendix A)
CATEGORIES = (
    "Environmental indicator",
    "Atmospheric variable",
    "Maritime variable",
    "Oil slick proxy",
    "Spatial/proximity feature",
    "Wind/meteorological feature",
    "Engineered exposure index",
    "Interaction feature",
    "Temporal feature",
    "Port/geographic attribution feature",
    "Panel metadata / quality",  # internal for keys not in list — remap below
)

FEATURE_META: dict[str, dict[str, str]] = {
    "grid_cell_id": {
        "category": "Port/geographic attribution feature",
        "unit": "categorical string (e.g. g0.100_r…_c…)",
        "source": "Pipeline grid lattice (`src/extract_sentinel_weekly_features.py`)",
        "purpose": "Unique spatial unit identifier linking all weekly observations to a lattice cell.",
        "created_in": "src/extract_sentinel_weekly_features.py",
    },
    "week_start_utc": {
        "category": "Temporal feature",
        "unit": "UTC timestamp (ISO week anchor)",
        "source": "Panel index (weekly aggregation)",
        "purpose": "Defines the weekly timestep for the balanced spatiotemporal panel.",
        "created_in": "src/pipeline/run_full_pipeline.py (panel construction)",
    },
    "vessel_density_t": {
        "category": "Maritime variable",
        "unit": "normalised vessel-density units (EMODnet layer; ~0–1 in panel)",
        "source": "EMODnet AIS maritime activity layers",
        "purpose": "Time-aligned maritime traffic intensity at each grid-week.",
        "created_in": "src/data_sources/emodnet_vessel_density.py (ingestion)",
    },
    "grid_centroid_lat": {
        "category": "Spatial/proximity feature",
        "unit": "degrees north (WGS84)",
        "source": "Grid lattice definition",
        "purpose": "Cell centroid latitude for mapping, distance metrics, and spatial ML predictors.",
        "created_in": "src/extract_sentinel_weekly_features.py",
    },
    "grid_centroid_lon": {
        "category": "Spatial/proximity feature",
        "unit": "degrees east (WGS84)",
        "source": "Grid lattice definition",
        "purpose": "Cell centroid longitude for mapping and spatial stratification.",
        "created_in": "src/extract_sentinel_weekly_features.py",
    },
    "no2_mean_t": {
        "category": "Atmospheric variable",
        "unit": "mol m⁻² (Sentinel-5P tropospheric NO₂ column mean, weekly)",
        "source": "Sentinel-5P NO₂ (GEE pipeline)",
        "purpose": "Weekly mean tropospheric NO₂ at the grid cell.",
        "created_in": "src/data_sources/no2_gee_pipeline.py",
    },
    "no2_std_t": {
        "category": "Atmospheric variable",
        "unit": "mol m⁻² (within-week std)",
        "source": "Sentinel-5P NO₂",
        "purpose": "Sub-weekly NO₂ variability (episodic pollution vs stable background).",
        "created_in": "src/data_sources/no2_gee_pipeline.py",
    },
    "oil_slick_probability_t": {
        "category": "Oil slick proxy",
        "unit": "probability / fraction (0–1 dark-water anomaly proxy)",
        "source": "Sentinel-1 SAR dark-feature detection",
        "purpose": "Weekly mean probability of oil-slick-like dark features (not confirmed spill attribution).",
        "created_in": "src/data_sources/sentinel1_oil_pipeline.py",
    },
    "oil_slick_count_t": {
        "category": "Oil slick proxy",
        "unit": "count (detections per cell-week)",
        "source": "Sentinel-1 SAR",
        "purpose": "Count of dark-feature detections supporting the oil proxy.",
        "created_in": "src/data_sources/sentinel1_oil_pipeline.py",
    },
    "ndwi_mean": {
        "category": "Environmental indicator",
        "unit": "dimensionless index (−1 to 1 typical)",
        "source": "Sentinel-2 multispectral (NDWI)",
        "purpose": "Mean Normalised Difference Water Index — open water vs turbidity sensitivity.",
        "created_in": "src/data_sources/sentinel2_water_quality.py",
    },
    "ndwi_median": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDWI", "purpose": "Robust weekly NDWI aggregate.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "ndwi_std": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDWI", "purpose": "Within-week NDWI dispersion.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "ndti_mean": {
        "category": "Environmental indicator",
        "unit": "dimensionless turbidity index",
        "source": "Sentinel-2 NDTI",
        "purpose": "Mean Normalised Difference Turbidity Index — suspended matter / turbidity proxy.",
        "created_in": "src/data_sources/sentinel2_water_quality.py",
    },
    "ndti_median": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDTI", "purpose": "Robust NDTI aggregate.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "ndti_std": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDTI", "purpose": "NDTI variability within the week.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "ndci_mean": {
        "category": "Environmental indicator",
        "unit": "dimensionless chlorophyll proxy",
        "source": "Sentinel-2 NDCI",
        "purpose": "Mean Normalised Difference Chlorophyll Index — productivity / eutrophication-related signal.",
        "created_in": "src/data_sources/sentinel2_water_quality.py",
    },
    "ndci_median": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDCI", "purpose": "Robust NDCI aggregate.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "ndci_std": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDCI", "purpose": "NDCI dispersion.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "fai_mean": {
        "category": "Environmental indicator",
        "unit": "dimensionless floating-algae index",
        "source": "Sentinel-2 FAI",
        "purpose": "Floating algae / detritus / surface anomaly indicator.",
        "created_in": "src/data_sources/sentinel2_water_quality.py",
    },
    "fai_median": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 FAI", "purpose": "Robust FAI aggregate.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "fai_std": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 FAI", "purpose": "FAI dispersion.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "b11_mean": {
        "category": "Environmental indicator",
        "unit": "surface reflectance (Band 11 SWIR)",
        "source": "Sentinel-2 Band 11",
        "purpose": "SWIR reflectance context for water / anomaly interpretation.",
        "created_in": "src/data_sources/sentinel2_water_quality.py",
    },
    "b11_median": {"category": "Environmental indicator", "unit": "reflectance", "source": "Sentinel-2 Band 11", "purpose": "Robust Band-11 aggregate.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "b11_std": {"category": "Environmental indicator", "unit": "reflectance", "source": "Sentinel-2 Band 11", "purpose": "Band-11 dispersion.", "created_in": "src/data_sources/sentinel2_water_quality.py"},
    "ndvi_mean": {
        "category": "Environmental indicator",
        "unit": "dimensionless NDVI (−1 to 1)",
        "source": "Sentinel-2 NDVI (nearest-land linkage where computed)",
        "purpose": "Vegetation / land-response proxy for coastal–terrestrial coupling (sparse coverage).",
        "created_in": "src/analysis/run_nearest_land_ndvi_linkage.py",
    },
    "ndvi_median": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDVI", "purpose": "Robust NDVI aggregate.", "created_in": "src/analysis/run_nearest_land_ndvi_linkage.py"},
    "ndvi_std": {"category": "Environmental indicator", "unit": "dimensionless", "source": "Sentinel-2 NDVI", "purpose": "NDVI dispersion.", "created_in": "src/analysis/run_nearest_land_ndvi_linkage.py"},
    "NO2_mean": {
        "category": "Atmospheric variable",
        "unit": "mol m⁻² (alias of weekly NO₂)",
        "source": "Derived from `no2_mean_t` in pipeline harmonisation",
        "purpose": "Thesis-facing NO₂ column for correlations and land–sea indices.",
        "created_in": "src/pipeline/run_full_pipeline.py (`_harmonize_feature_aliases`)",
    },
    "NO2_trend": {
        "category": "Atmospheric variable",
        "unit": "mol m⁻² week⁻¹ (first difference within cell)",
        "source": "Engineered from `NO2_mean`",
        "purpose": "Week-to-week NO₂ change per grid cell.",
        "created_in": "src/pipeline/run_full_pipeline.py (grouped diff on NO2_mean)",
    },
    "vessel_density": {
        "category": "Maritime variable",
        "unit": "normalised density (panel alias)",
        "source": "EMODnet AIS (alias of `vessel_density_t`)",
        "purpose": "Maritime intensity column used by interaction and min–max indices.",
        "created_in": "src/pipeline/run_full_pipeline.py",
    },
    "detection_score": {
        "category": "Oil slick proxy",
        "unit": "probability / score (0–1 typical)",
        "source": "Alias of primary oil-slick probability column",
        "purpose": "Harmonised detection score for ML and stratified NO₂–oil analysis.",
        "created_in": "src/pipeline/run_full_pipeline.py (alias of oil_slick_probability_t)",
    },
    "nan_ratio_row": {
        "category": "Environmental indicator",
        "unit": "fraction in [0, 1]",
        "source": "Engineered QC",
        "purpose": "[Data-quality diagnostic] Row-wise fraction of missing values across input columns (not an ecological indicator).",
        "created_in": "src/pipeline/run_full_pipeline.py",
    },
    "nearest_port": {
        "category": "Port/geographic attribution feature",
        "unit": "categorical (port name)",
        "source": "Port/coastline GIS (`data/aux/baltic_ports.csv`)",
        "purpose": "Nearest focal port label for Turku / Mariehamn / Naantali stratification.",
        "created_in": "src/features/port_proximity.py",
    },
    "distance_to_port_km": {
        "category": "Spatial/proximity feature",
        "unit": "kilometres (geodesic)",
        "source": "Port GIS + grid centroids",
        "purpose": "Distance from cell centroid to nearest catalogued port — port-distance decay axis.",
        "created_in": "src/features/port_proximity.py",
    },
    "port_exposure_score": {
        "category": "Engineered exposure index",
        "unit": "density / km (vessel_density / (1 + distance_km))",
        "source": "Engineered from EMODnet + port distance",
        "purpose": "Port-proximity-weighted maritime intensity.",
        "created_in": "src/features/port_exposure.py",
    },
    "distance_to_nearest_high_vessel_density_cell": {
        "category": "Spatial/proximity feature",
        "unit": "kilometres (haversine to weekly P90 vessel-density seeds)",
        "source": "EMODnet + lattice geometry",
        "purpose": "Proximity to high-traffic cells — shipping-lane / corridor structure.",
        "created_in": "src/features/land_sea_buffering.py",
    },
    "coastal_exposure_band": {
        "category": "Spatial/proximity feature",
        "unit": "categorical band (0–10 km, 10–50 km, 50+ km)",
        "source": "Engineered from distance to high-density seeds",
        "purpose": "Discrete coastal exposure stratification.",
        "created_in": "src/features/land_sea_buffering.py",
    },
    "coastal_exposure_score": {
        "category": "Engineered exposure index",
        "unit": "unitless score in [0, 1]",
        "source": "Engineered distance-decay exposure function",
        "purpose": "Piecewise coastal exposure to maritime activity (CES in thesis figures).",
        "created_in": "src/features/land_sea_buffering.py",
    },
    "maritime_pressure_index": {
        "category": "Engineered exposure index",
        "unit": "min–max normalised [0, 1] over panel",
        "source": "Engineered from vessel density",
        "purpose": "Normalised maritime pressure composite (MEI input in Fig. 5.8-style ESI).",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "atmospheric_transfer_index": {
        "category": "Engineered exposure index",
        "unit": "min–max normalised [0, 1]",
        "source": "Engineered from NO2_mean",
        "purpose": "Normalised atmospheric transfer / NO₂ connectivity index.",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "land_response_index": {
        "category": "Engineered exposure index",
        "unit": "min–max normalised [0, 1]",
        "source": "Engineered from ndvi_mean",
        "purpose": "Normalised land-vegetation response where NDVI is available.",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "vessel_x_no2": {
        "category": "Interaction feature",
        "unit": "product of density × mol m⁻²",
        "source": "Engineered interaction",
        "purpose": "Maritime–atmospheric co-stress interaction term.",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "no2_x_ndvi": {
        "category": "Interaction feature",
        "unit": "product of NO₂ × NDVI",
        "source": "Engineered interaction",
        "purpose": "Atmosphere–land coupling interaction.",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "vessel_x_ndvi_lag1": {
        "category": "Interaction feature",
        "unit": "lagged product (1 week)",
        "source": "Engineered lag interaction",
        "purpose": "Vessel density at t−1 multiplied by NDVI at t (per grid).",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "vessel_x_ndvi_lag2": {
        "category": "Interaction feature",
        "unit": "lagged product (2 weeks)",
        "source": "Engineered lag interaction",
        "purpose": "Two-week delayed vessel–vegetation interaction.",
        "created_in": "src/features/land_sea_interactions.py",
    },
    "vessel_x_ndvi_lag3": {
        "category": "Interaction feature",
        "unit": "lagged product (3 weeks)",
        "source": "Engineered lag interaction",
        "purpose": "Three-week delayed vessel–vegetation interaction.",
        "created_in": "src/features/land_sea_interactions.py",
    },
}

# Supplement: data/modeling_dataset.parquet (ΔNDTI ML task)
MODELING_META: dict[str, dict[str, str]] = {
    "grid_res_deg": {
        "category": "Spatial/proximity feature",
        "unit": "degrees",
        "source": "Encoded in grid_cell_id prefix",
        "purpose": "Nominal lattice resolution for ML spatial predictors.",
        "created_in": "data/modeling_dataset build (pipeline)",
    },
    "week_of_year": {
        "category": "Temporal feature",
        "unit": "integer 1–53",
        "source": "Derived from week_start_utc",
        "purpose": "Seasonal cycle encoding for ML.",
        "created_in": "data/modeling_dataset build",
    },
    "week_sin": {
        "category": "Temporal feature",
        "unit": "dimensionless [−1, 1]",
        "source": "Fourier seasonality",
        "purpose": "Sine component of annual seasonality.",
        "created_in": "data/modeling_dataset build",
    },
    "week_cos": {
        "category": "Temporal feature",
        "unit": "dimensionless [−1, 1]",
        "source": "Fourier seasonality",
        "purpose": "Cosine component of annual seasonality.",
        "created_in": "data/modeling_dataset build",
    },
    "vessel_density_t_minus_1": {
        "category": "Temporal feature",
        "unit": "lagged maritime density",
        "source": "EMODnet (1-week lag per grid)",
        "purpose": "Autoregressive maritime predictor for ΔNDTI models.",
        "created_in": "data/modeling_dataset build",
    },
    "vessel_density_t_minus_2": {
        "category": "Temporal feature",
        "unit": "lagged maritime density",
        "source": "EMODnet (2-week lag)",
        "purpose": "Second maritime lag feature.",
        "created_in": "data/modeling_dataset build",
    },
    "sentinel_ndvi_mean_t": {
        "category": "Environmental indicator",
        "unit": "dimensionless NDVI",
        "source": "Sentinel-2",
        "purpose": "Current-week NDVI for ML panel.",
        "created_in": "data/modeling_dataset build",
    },
    "sentinel_ndvi_mean_t_minus_1": {"category": "Temporal feature", "unit": "NDVI lag 1", "source": "Sentinel-2", "purpose": "Spectral lag predictor.", "created_in": "data/modeling_dataset build"},
    "sentinel_ndvi_mean_t_minus_2": {"category": "Temporal feature", "unit": "NDVI lag 2", "source": "Sentinel-2", "purpose": "Spectral lag predictor.", "created_in": "data/modeling_dataset build"},
    "sentinel_ndwi_mean_t": {"category": "Environmental indicator", "unit": "NDWI", "source": "Sentinel-2", "purpose": "Current-week NDWI for ML.", "created_in": "data/modeling_dataset build"},
    "sentinel_ndwi_mean_t_minus_1": {"category": "Temporal feature", "unit": "NDWI lag 1", "source": "Sentinel-2", "purpose": "Spectral lag.", "created_in": "data/modeling_dataset build"},
    "sentinel_ndwi_mean_t_minus_2": {"category": "Temporal feature", "unit": "NDWI lag 2", "source": "Sentinel-2", "purpose": "Spectral lag.", "created_in": "data/modeling_dataset build"},
    "sentinel_evi_mean_t": {"category": "Environmental indicator", "unit": "EVI", "source": "Sentinel-2", "purpose": "Enhanced vegetation index at t.", "created_in": "data/modeling_dataset build"},
    "sentinel_evi_mean_t_minus_1": {"category": "Temporal feature", "unit": "EVI lag 1", "source": "Sentinel-2", "purpose": "Spectral lag.", "created_in": "data/modeling_dataset build"},
    "sentinel_evi_mean_t_minus_2": {"category": "Temporal feature", "unit": "EVI lag 2", "source": "Sentinel-2", "purpose": "Spectral lag.", "created_in": "data/modeling_dataset build"},
    "sentinel_ndti_mean_t": {
        "category": "Environmental indicator",
        "unit": "NDTI",
        "source": "Sentinel-2",
        "purpose": "Current-week turbidity for ΔNDTI target construction.",
        "created_in": "data/modeling_dataset build",
    },
    "sentinel_ndti_mean_t_minus_1": {"category": "Temporal feature", "unit": "NDTI lag 1", "source": "Sentinel-2", "purpose": "Spectral lag.", "created_in": "data/modeling_dataset build"},
    "sentinel_ndti_mean_t_minus_2": {"category": "Temporal feature", "unit": "NDTI lag 2", "source": "Sentinel-2", "purpose": "Spectral lag.", "created_in": "data/modeling_dataset build"},
    "sentinel_observation_count_t": {
        "category": "Environmental indicator",
        "unit": "count",
        "source": "Sentinel-2",
        "purpose": "Number of valid S2 observations contributing to the weekly aggregate.",
        "created_in": "data/modeling_dataset build",
    },
    "delta_ndti": {
        "category": "Environmental indicator",
        "unit": "NDTI change (t+1 − t)",
        "source": "Engineered target",
        "purpose": "Primary ML target: week-ahead turbidity change.",
        "created_in": "src/run_delta_ndti_models.py / modeling_dataset build",
    },
    "has_valid_delta_ndti": {
        "category": "Temporal feature",
        "unit": "boolean",
        "source": "Engineered mask",
        "purpose": "Flags rows usable for ΔNDTI training.",
        "created_in": "data/modeling_dataset build",
    },
    "has_sentinel": {"category": "Temporal feature", "unit": "boolean", "source": "Coverage flag", "purpose": "[ML mask] Sentinel-2 data present at t.", "created_in": "data/modeling_dataset build"},
    "has_emodnet": {"category": "Temporal feature", "unit": "boolean", "source": "Coverage flag", "purpose": "[ML mask] EMODnet vessel layer present.", "created_in": "data/modeling_dataset build"},
    "has_helcom": {"category": "Temporal feature", "unit": "boolean", "source": "Coverage flag", "purpose": "[ML mask] HELCOM/auxiliary data flag.", "created_in": "data/modeling_dataset build"},
}

# Analysis-time features (not in features_ml_ready.parquet; merged at runtime)
RUNTIME_FEATURES: list[dict[str, str]] = [
    {
        "Feature": "maritime_exposure_index (MEI)",
        "Category": "Engineered exposure index",
        "Type": "float64",
        "Unit / Scale": "rank percentile [0, 1]",
        "Source": "Engineered: vessel × wind-alignment × inverse coast distance",
        "Purpose": "Maritime exposure composite (RQ2, coastal analysis).",
        "Dataset": "runtime merge via build_indices",
        "Created_in": "src/analysis/run_coastal_exposure_analysis.py",
    },
    {
        "Feature": "atmospheric_coastal_exposure_index (ACEI)",
        "Category": "Engineered exposure index",
        "Type": "float64",
        "Unit / Scale": "rank percentile [0, 1]",
        "Source": "Engineered: local NO₂ excess × coastal wind × transport alignment",
        "Purpose": "Atmospheric coastal exposure composite.",
        "Dataset": "runtime merge via build_indices",
        "Created_in": "src/analysis/run_coastal_exposure_analysis.py",
    },
    {
        "Feature": "environmental_stress_index (ESI)",
        "Category": "Engineered exposure index",
        "Type": "float64",
        "Unit / Scale": "rank percentile [0, 1]",
        "Source": "Engineered: weekly z-mean of NO₂, vessel, NDTI, oil, wind terms",
        "Purpose": "Experimental multivariate stress composite (Fig. 5.8, sensitivity analysis).",
        "Dataset": "runtime merge via build_indices",
        "Created_in": "src/analysis/run_coastal_exposure_analysis.py",
    },
    {
        "Feature": "coastal_wind_alignment_score",
        "Category": "Wind/meteorological feature",
        "Type": "float64",
        "Unit / Scale": "cos(angle) in [−1, 1]",
        "Source": "Open-Meteo ERA5 wind + coastline bearing",
        "Purpose": "Alignment of wind toward nearest coast bearing.",
        "Dataset": "coastal_wind_alignment_features.csv",
        "Created_in": "src/analysis/run_coastal_wind_transport.py",
    },
    {
        "Feature": "coastal_wind_shoreward_45deg",
        "Category": "Wind/meteorological feature",
        "Type": "float64",
        "Unit / Scale": "binary {0, 1}",
        "Source": "ERA5-derived; threshold cos(45°)",
        "Purpose": "Shoreward vs non-shoreward wind regime (§5.4, §5.7).",
        "Dataset": "coastal_wind_alignment_features.csv",
        "Created_in": "src/analysis/run_coastal_wind_transport.py",
    },
    {
        "Feature": "wind_u_mean / wind_v_mean",
        "Category": "Wind/meteorological feature",
        "Type": "float64",
        "Unit / Scale": "m s⁻¹",
        "Source": "Open-Meteo ERA5 archive",
        "Purpose": "Mean zonal/meridional wind components for the grid-week.",
        "Dataset": "coastal_wind_alignment_features.csv",
        "Created_in": "src/analysis/run_coastal_wind_transport.py",
    },
    {
        "Feature": "local_no2_excess",
        "Category": "Atmospheric variable",
        "Type": "float64",
        "Unit / Scale": "mol m⁻² anomaly",
        "Source": "Engineered from no2_mean_t vs 15–30 km band baseline",
        "Purpose": "Weekly NO₂ excess relative to offshore reference band.",
        "Dataset": "runtime via prepare_panel",
        "Created_in": "src/analysis/run_coastal_exposure_analysis.py",
    },
]


def _dtype_str(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime64[UTC]"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return str(s.dtype)
    return "string / object"


def _example_value(s: pd.Series) -> str:
    non = s.dropna()
    if non.empty:
        return "—"
    v = non.iloc[0]
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v))
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e4:
            return f"{v:.4e}"
        return f"{v:.4f}"
    return str(v)[:80]


def profile_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    n = len(s)
    miss_pct = 100.0 * float(s.isna().mean()) if n else 0.0
    meta = FEATURE_META.get(col, MODELING_META.get(col, {}))
    return {
        "Feature": col,
        "Category": meta.get("category", "Environmental indicator"),
        "Type": _dtype_str(s),
        "Unit / Scale": meta.get("unit", "see pipeline documentation"),
        "Source": meta.get("source", "see src/ and scripts/"),
        "Purpose": meta.get("purpose", "Thesis panel variable."),
        "Missing_pct": round(miss_pct, 2),
        "Example": _example_value(s),
        "Created_in": meta.get("created_in", ""),
        "N_rows": n,
    }


def build_inventory_table(df: pd.DataFrame, meta_dict: dict) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        row = profile_column(df, col)
        if col not in meta_dict and col not in FEATURE_META:
            row["Category"] = _infer_category(col)
        rows.append(row)
    return pd.DataFrame(rows)


def _infer_category(name: str) -> str:
    n = name.lower()
    if name in ("grid_cell_id", "nearest_port"):
        return "Port/geographic attribution feature"
    if "week" in n or "lag" in n or "minus_" in n:
        return "Temporal feature"
    if "wind" in n or "shoreward" in n:
        return "Wind/meteorological feature"
    if "oil" in n or "slick" in n or "detection" in n:
        return "Oil slick proxy"
    if "vessel" in n or "maritime" in n:
        return "Maritime variable"
    if "no2" in n or name.startswith("NO2"):
        return "Atmospheric variable"
    if "distance" in n or "centroid" in n or "grid_res" in n:
        return "Spatial/proximity feature"
    if "exposure" in n or "pressure" in n or "transfer" in n or "response_index" in n:
        return "Engineered exposure index"
    if "_x_" in n or "interaction" in n:
        return "Interaction feature"
    if any(x in n for x in ("ndvi", "ndwi", "ndti", "ndci", "fai", "b11", "evi", "sentinel", "delta")):
        return "Environmental indicator"
    if n.startswith("has_") or "nan_ratio" in n:
        return "Panel metadata / quality"
    return "Environmental indicator"


def _md_table(inv: pd.DataFrame, cols: list[str]) -> str:
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [hdr, sep]
    for _, r in inv.iterrows():
        cells = [str(r[c]).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    if not ML_READY.is_file():
        print(f"Missing {ML_READY}", file=sys.stderr)
        return 1

    ml = pd.read_parquet(ML_READY)
    merged_note = ""
    if MERGED.is_file():
        mdf = pd.read_parquet(MERGED)
        merged_note = (
            f"`processed/merged_dataset.parquet` contains **{len(mdf.columns)}** columns "
            f"(observational merge prior to land–sea engineering); all are a subset of or precursor to "
            f"the **{len(ml.columns)}** columns in `features_ml_ready.parquet`."
        )

    inv_ml = build_inventory_table(ml, FEATURE_META)

    modeling_inv = pd.DataFrame()
    modeling_only = []
    if MODELING.is_file():
        mod = pd.read_parquet(MODELING)
        inv_mod = build_inventory_table(mod, MODELING_META)
        ml_cols = set(ml.columns)
        modeling_only = [c for c in mod.columns if c not in ml_cols]
        modeling_inv = inv_mod[inv_mod["Feature"].isin(modeling_only)].copy()
        modeling_inv["Dataset"] = "data/modeling_dataset.parquet"

    inv_ml["Dataset"] = "processed/features_ml_ready.parquet"

  # Combined CSV export
    export_cols = [
        "Feature",
        "Category",
        "Type",
        "Unit / Scale",
        "Source",
        "Purpose",
        "Missing_pct",
        "Example",
        "Dataset",
        "Created_in",
    ]
    full_csv = pd.concat([inv_ml, modeling_inv], ignore_index=True)
    if RUNTIME_FEATURES:
        rt = pd.DataFrame(RUNTIME_FEATURES)
        for c in export_cols:
            if c not in rt.columns:
                rt[c] = ""
        full_csv = pd.concat([full_csv, rt[export_cols]], ignore_index=True)
    full_csv.to_csv(OUT_CSV, index=False)

    # Category counts
    cat_counts = inv_ml["Category"].value_counts().sort_index()

    md_cols = ["Feature", "Category", "Type", "Unit / Scale", "Source", "Purpose"]

    lines = [
        "Appendix A. Complete Machine Learning Feature Inventory",
        "",
        "This appendix summarises the environmental, atmospheric, maritime, spatial, temporal, and engineered "
        "variables used in the machine-learning-ready dataset. The features were derived from Sentinel-2, Sentinel-5P, "
        "Sentinel-1 SAR, EMODnet vessel-density layers, Open-Meteo ERA5 wind data, port/coastline GIS layers, and "
        "engineered exposure-analysis procedures.",
        "",
        f"**Primary panel:** `processed/features_ml_ready.parquet` — **{len(ml):,}** grid-week rows × **{len(ml.columns)}** columns "
        f"(315 unique `grid_cell_id`, 40 UTC week anchors in this build).",
        "",
    ]
    if merged_note:
        lines.append(merged_note + "\n")

    lines.extend(
        [
            "### Coverage summary (`features_ml_ready.parquet`)",
            "",
            _md_table(
                inv_ml.groupby("Category", as_index=False)
                .agg(Features=("Feature", "count"), Mean_missing_pct=("Missing_pct", "mean"))
                .round({"Mean_missing_pct": 1}),
                ["Category", "Features", "Mean_missing_pct"],
            ),
            "",
            "### A.1 Core machine-learning-ready features",
            "",
            _md_table(inv_ml[md_cols], md_cols),
            "",
            "#### Example values and missingness (core panel)",
            "",
            _md_table(
                inv_ml[["Feature", "Missing_pct", "Example", "Created_in"]],
                ["Feature", "Missing_pct", "Example", "Created_in"],
            ),
            "",
        ],
    )

    if len(modeling_inv):
        lines.extend(
            [
                "### A.2 Additional features in `data/modeling_dataset.parquet` (ΔNDTI ML task)",
                "",
                "Used by Ridge / HistGradientBoosting models in `src/run_delta_ndti_models.py` (time-aware week split, target `delta_ndti`). "
                "Columns below are **not** stored in `features_ml_ready.parquet`.",
                "",
                _md_table(modeling_inv[md_cols], md_cols),
                "",
                _md_table(
                    modeling_inv[["Feature", "Missing_pct", "Example", "Created_in"]],
                    ["Feature", "Missing_pct", "Example", "Created_in"],
                ),
                "",
            ],
        )

    lines.extend(
        [
            "### A.3 Runtime-derived features (coastal exposure & wind merge)",
            "",
            "These variables are computed during coastal-exposure and wind-transport analysis and merged at runtime "
            "(e.g. `scripts/fix_rq_evidence_pipeline.py`, `src/analysis/run_coastal_exposure_analysis.py`); they are "
            "**not** persisted as columns in `features_ml_ready.parquet`.",
            "",
            _md_table(pd.DataFrame(RUNTIME_FEATURES)[md_cols], md_cols),
            "",
            "### A.4 Engineering pipeline map",
            "",
            "| Stage | Module | Outputs |",
            "| --- | --- | --- |",
            "| Ingestion | `src/data_sources/sentinel2_water_quality.py` | NDWI, NDTI, NDCI, FAI, B11 weekly stats |",
            "| Ingestion | `src/data_sources/no2_gee_pipeline.py` | `no2_mean_t`, `no2_std_t` |",
            "| Ingestion | `src/data_sources/sentinel1_oil_pipeline.py` | `oil_slick_probability_t`, `oil_slick_count_t` |",
            "| Ingestion | EMODnet vessel layers | `vessel_density_t` |",
            "| Merge | `processed/merged_dataset.parquet` | Observational columns only (27 fields) |",
            "| Port GIS | `src/features/port_proximity.py` | `nearest_port`, `distance_to_port_km` |",
            "| Port exposure | `src/features/port_exposure.py` | `port_exposure_score` |",
            "| Land–sea buffer | `src/features/land_sea_buffering.py` | `distance_to_nearest_high_vessel_density_cell`, `coastal_exposure_band`, `coastal_exposure_score` |",
            "| Land–sea interactions | `src/features/land_sea_interactions.py` | MEI/ATI/LRI aliases + interaction terms |",
            "| Harmonisation | `src/pipeline/run_full_pipeline.py` | `NO2_mean`, `NO2_trend`, `vessel_density`, `detection_score`, `nan_ratio_row` |",
            "| Final ML table | `processed/features_ml_ready.parquet` | 46-column thesis panel |",
            "| Wind transport | `src/analysis/run_coastal_wind_transport.py` | Wind alignment CSV → MEI/ACEI/ESI via `build_indices` |",
            "| ML modelling | `data/modeling_dataset.parquet` | Lags, seasonality, `delta_ndti` target |",
            "",
            "### A.5 Notes for interpretation",
            "",
            "- **Duplicate maritime / NO₂ columns:** `vessel_density_t` and `vessel_density`, `no2_mean_t` and `NO2_mean` are harmonised aliases for the same signals.",
            "- **MEI in figures:** `maritime_pressure_index` in the parquet corresponds to the min–max maritime pressure index; thesis Figure 5.8 ESI uses a related but distinct six-variable z-mean (see `scripts/generate_thesis_sections_5_5_to_5_10.py`).",
            "- **Sparse optical coverage:** `ndvi_*` and land-linked interactions exceed **97%** missing in the coastal panel; interpret as optional land-side comparators.",
            "- **Oil variables:** Treat as **dark-feature proxies**, not confirmed oil-spill detections.",
            "",
            f"*Generated by `{Path(__file__).relative_to(ROOT)}`.*",
            "",
        ],
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD} ({len(inv_ml)} core features)")
    print(f"Wrote {OUT_CSV} ({len(full_csv)} total rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
