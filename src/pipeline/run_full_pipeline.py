from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import pkgutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class SourceConfig:
    name: str
    module_name: str
    output_path: Path
    validation_path: Path
    critical: bool = False


FEATURE_CATALOG: dict[str, list[dict[str, Any]]] = {
    "water_quality": [
        {"name": "NDCI", "description": "Normalized Difference Chlorophyll Index", "candidates": ["ndci_mean", "ndci_median", "ndci"], "source_module": "data_sources.sentinel2_water_quality"},
        {"name": "NDTI", "description": "Normalized Difference Turbidity Index", "candidates": ["ndti_mean", "ndti_median", "ndti"], "source_module": "data_sources.sentinel2_water_quality"},
        {"name": "NDWI", "description": "Normalized Difference Water Index", "candidates": ["ndwi_mean", "ndwi_median", "ndwi"], "source_module": "data_sources.sentinel2_water_quality"},
        {"name": "B4/B3", "description": "Red/Green reflectance ratio proxy", "candidates": ["b4_b3", "red_green_ratio"], "source_module": "data_sources.sentinel2_water_quality"},
        {"name": "FAI", "description": "Floating Algae Index", "candidates": ["fai_mean", "fai_median", "fai"], "source_module": "data_sources.sentinel2_water_quality"},
        {"name": "SWIR B11", "description": "Sentinel-2 SWIR band 11", "candidates": ["b11_mean", "b11_median", "b11"], "source_module": "data_sources.sentinel2_water_quality"},
        {"name": "SST", "description": "Sea Surface Temperature", "candidates": ["sst"], "source_module": "data_sources.sentinel2_water_quality"},
    ],
    "maritime_activity": [
        {"name": "vessel_density", "description": "AIS/EMODnet vessel density proxy", "candidates": ["vessel_density", "vessel_density_t"], "source_module": "vessels"},
        {"name": "total_density", "description": "Total traffic density aggregate", "candidates": ["total_density"], "source_module": "vessels"},
        {"name": "density_total_log", "description": "Log-transformed vessel density", "candidates": ["density_total_log"], "source_module": "vessels"},
        {"name": "seasonal_metrics", "description": "Seasonal activity indicators", "candidates": ["season", "seasonal_metric"], "source_module": "vessels"},
    ],
    "atmospheric": [
        {"name": "NO2 mean", "description": "Weekly tropospheric NO2 mean", "candidates": ["no2_mean_t", "NO2_mean"], "source_module": "data_sources.no2_gee_pipeline"},
        {"name": "NO2 variability", "description": "Weekly NO2 variability (std)", "candidates": ["no2_std_t"], "source_module": "data_sources.no2_gee_pipeline"},
        {"name": "Sentinel-1 disturbance proxy", "description": "Dark-water anomaly probability/count", "candidates": ["oil_slick_probability_t", "oil_slick_count_t", "detection_score"], "source_module": "data_sources.sentinel1_oil_pipeline"},
    ],
    "exposure": [
        {"name": "distance_to_port", "description": "Distance from grid cell to nearest port", "candidates": ["distance_to_port", "distance_to_port_km"], "source_module": "analysis.exposure"},
        {"name": "distance_to_lane", "description": "Distance to major shipping corridor", "candidates": ["distance_to_lane", "distance_to_shipping_km"], "source_module": "analysis.exposure"},
    ],
    "validation": [
        {"name": "chlorophyll-a in-situ", "description": "In-situ chlorophyll validation reference", "candidates": ["chlorophyll_a_insitu", "chl_a"], "source_module": "validation"},
        {"name": "EQRS score", "description": "Ecological quality ratio score", "candidates": ["eqrs_score", "eqrs"], "source_module": "validation"},
        {"name": "water quality class", "description": "Categorical water quality label", "candidates": ["water_quality_class", "quality_class"], "source_module": "validation"},
    ],
    "land": [
        {"name": "NDVI mean", "description": "Land NDVI (water-masked) mean per grid-week", "candidates": ["ndvi_mean"], "source_module": "data_sources.land_impact.sentinel2_land_metrics"},
        {"name": "NDVI std", "description": "Land NDVI variability per grid-week", "candidates": ["ndvi_std"], "source_module": "data_sources.land_impact.sentinel2_land_metrics"},
        {"name": "NDVI median", "description": "Land NDVI median per grid-week", "candidates": ["ndvi_median"], "source_module": "data_sources.land_impact.sentinel2_land_metrics"},
        {"name": "coastal_exposure_score", "description": "Buffered coastal exposure to maritime activity (0-1)", "candidates": ["coastal_exposure_score"], "source_module": "features.land_sea_buffering"},
        {"name": "distance_to_high_vessel_cell", "description": "Distance (km) to nearest high-activity vessel cell", "candidates": ["distance_to_nearest_high_vessel_density_cell"], "source_module": "features.land_sea_buffering"},
        {"name": "land_response_index", "description": "Normalised NDVI-based land response index", "candidates": ["land_response_index"], "source_module": "features.land_sea_interactions"},
    ],
}


FEATURE_REGISTRY: dict[str, list[dict[str, Any]]] = {
    "water_quality": [],
    "maritime_activity": [],
    "atmospheric": [],
    "exposure": [],
    "validation": [],
    "land": [],
}


SOURCE_CATEGORY_MAP: dict[str, list[str]] = {
    "vessels": ["maritime_activity"],
    "no2": ["atmospheric"],
    "sentinel1": ["atmospheric"],
    "sentinel2_water_quality": ["water_quality"],
    "land_impact_ndvi": ["land"],
    "features": ["water_quality", "maritime_activity", "atmospheric", "exposure", "validation", "land"],
}


def _resolve_feature_stats(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, str]:
    for candidate in candidates:
        if candidate in df.columns:
            series = pd.to_numeric(df[candidate], errors="coerce")
            coverage = float(series.notna().mean() * 100.0) if len(series) else 0.0
            if series.notna().any():
                mean = float(series.mean())
                std = float(series.std()) if series.notna().sum() > 1 else 0.0
                return candidate, f"mean={mean:.4f}, std={std:.4f}, coverage={coverage:.1f}%"
            return candidate, f"mean=NA, std=NA, coverage={coverage:.1f}%"
    return None, "missing"


def print_feature_table(category: str, features: list[dict[str, Any]], logger: logging.Logger) -> None:
    if not features:
        logger.info("[FEATURE TABLE: %s] no matching features found", category.replace("_", " ").title())
        return
    header = f"[FEATURE TABLE: {category.replace('_', ' ').title()}]"
    logger.info(header)
    col1, col2 = 28, 48
    logger.info("%-28s %-48s %s", "Feature", "Description", "Stats")
    logger.info("%s", "-" * 108)
    for feature in features:
        logger.info(
            "%-28s %-48s %s",
            str(feature["name"])[:col1],
            str(feature["description"])[:col2],
            str(feature["stats"]),
        )


def register_and_print_feature_tables(
    df: pd.DataFrame,
    source_name: str,
    source_module: str,
    logger: logging.Logger,
) -> None:
    categories = SOURCE_CATEGORY_MAP.get(source_name, [])
    for category in categories:
        rows: list[dict[str, Any]] = []
        for spec in FEATURE_CATALOG.get(category, []):
            resolved_col, stats = _resolve_feature_stats(df, spec["candidates"])
            entry = {
                "name": spec["name"],
                "description": spec["description"],
                "stats": stats,
                "source_module": source_module or spec["source_module"],
                "resolved_column": resolved_col,
            }
            FEATURE_REGISTRY[category].append(entry)
            rows.append(entry)
        print_feature_table(category, rows, logger)


def print_feature_registry_summary(logger: logging.Logger) -> None:
    logger.info("[GLOBAL FEATURE REGISTRY SUMMARY]")
    total = 0
    labels = {
        "water_quality": "Water Quality",
        "maritime_activity": "Maritime",
        "atmospheric": "Atmospheric",
        "exposure": "Exposure",
        "validation": "Validation",
        "land": "Land (LAND IMPACT EXTENSION)",
    }
    for key in ["water_quality", "maritime_activity", "atmospheric", "exposure", "validation", "land"]:
        count = len(FEATURE_REGISTRY.get(key, []))
        total += count
        logger.info("- %s: %d features", labels[key], count)
    logger.info("- Total features: %d", total)


def write_feature_registry_summary_json(path: Path) -> None:
    counts = {k: int(len(v)) for k, v in FEATURE_REGISTRY.items()}
    payload = {
        "counts": counts,
        "total_features": int(sum(counts.values())),
        "registry": FEATURE_REGISTRY,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("full_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def discover_sources(include_land_impact: bool = False) -> list[SourceConfig]:
    import data_sources

    out: list[SourceConfig] = []
    known_outputs = {
        "no2_gee_pipeline": ("no2", ROOT / "data" / "aux" / "no2_grid_week.parquet", ROOT / "data" / "aux" / "no2_gee_validation.json"),
        "sentinel1_oil_pipeline": (
            "sentinel1",
            ROOT / "data" / "aux" / "sentinel1_oil_slicks.parquet",
            ROOT / "data" / "aux" / "sentinel1_oil_validation.json",
        ),
        "sentinel2_water_quality": (
            "sentinel2_water_quality",
            ROOT / "data" / "aux" / "sentinel2_water_quality.parquet",
            ROOT / "data" / "aux" / "sentinel2_water_quality_validation.json",
        ),
    }
    for mod in pkgutil.iter_modules(data_sources.__path__):
        if mod.name.startswith("_") or mod.name in {"gee_grid_utils", "run_eo_pipeline", "land_impact"}:
            continue
        if mod.name in known_outputs:
            name, out_p, val_p = known_outputs[mod.name]
            out.append(SourceConfig(name=name, module_name=f"data_sources.{mod.name}", output_path=out_p, validation_path=val_p))

    # LAND IMPACT EXTENSION LAYER — additive, opt-in source.
    if include_land_impact:
        out.append(
            SourceConfig(
                name="land_impact_ndvi",
                module_name="data_sources.land_impact.sentinel2_land_metrics",
                output_path=ROOT / "data" / "aux" / "sentinel2_land_metrics.parquet",
                validation_path=ROOT / "data" / "aux" / "sentinel2_land_metrics_validation.json",
            )
        )
    return out


def run_source_extraction(
    source: SourceConfig,
    input_dataset: Path,
    quick_test: bool,
    sentinel_safe_mode: bool,
    force_refresh: bool,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage = f"[SOURCE: {source.name}]"
    logger.info("%s Starting extraction", stage)
    module = importlib.import_module(source.module_name)
    validation: dict[str, Any] = {}

    # Quick mode favors reproducibility/speed by reusing existing artifacts when available.
    # land_impact_ndvi reuses its cached parquet by default; pass --force-refresh to clear it.
    reuse_allowed = (
        not (source.name == "sentinel1" and sentinel_safe_mode)
        and not force_refresh
    )
    if quick_test and source.output_path.exists() and reuse_allowed:
        logger.info("%s quick-test reusing existing artifact: %s", stage, source.output_path)
        df = pd.read_parquet(source.output_path)
        if not df.empty:
            df = df.sample(min(len(df), 1500), random_state=42).copy()
        return df, validation

    # Full-mode reuse: if the aux parquet already exists and we're not forcing a refresh,
    # reuse it. Expensive GEE extractions don't change between runs on the same study window.
    if not quick_test and source.output_path.exists() and reuse_allowed:
        logger.info("%s reusing existing aux artifact: %s (use --force-refresh to re-extract)", stage, source.output_path)
        df = pd.read_parquet(source.output_path)
        return df, validation

    if hasattr(module, "run_extraction"):
        # Preferred standard interface requested by user.
        cfg = {
            "input": input_dataset,
            "output": source.output_path,
            "validation_json": source.validation_path,
            "quick_test": quick_test,
            "safe_mode": bool(sentinel_safe_mode and source.name == "sentinel1"),
            # Honour force-refresh for land_impact_ndvi too; otherwise reuse cache.
            "overwrite": bool(source.name == "land_impact_ndvi" and force_refresh),
        }
        df = module.run_extraction(cfg)
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"{source.module_name}.run_extraction did not return DataFrame")
        return df, validation

    if hasattr(module, "run_pipeline"):
        kwargs: dict[str, Any] = {}
        sig = inspect.signature(module.run_pipeline)
        if "resume" in sig.parameters:
            kwargs["resume"] = not quick_test
        if "debug_log_json" in sig.parameters:
            kwargs["debug_log_json"] = ROOT / "data" / "validation" / f"{source.name}_debug_log.json"
        if "safe_mode" in sig.parameters and source.name == "sentinel1":
            kwargs["safe_mode"] = bool(sentinel_safe_mode)
        if "quick_test" in sig.parameters:
            kwargs["quick_test"] = bool(quick_test)

        buffer_value = getattr(module, "DEFAULT_BUFFER_DEG_NO2", getattr(module, "DEFAULT_BUFFER_DEG_S1", 0.1))
        validation = module.run_pipeline(
            input_dataset,
            source.output_path,
            source.validation_path,
            buffer_value,
            **kwargs,
        )
        df = pd.read_parquet(source.output_path) if source.output_path.exists() else pd.DataFrame()
        if quick_test and not df.empty:
            df = df.sample(min(len(df), 1500), random_state=42).copy()
        return df, validation if isinstance(validation, dict) else {}

    raise RuntimeError(f"No compatible extraction function found in {source.module_name}")


def quality_checks(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "empty_dataset": df.empty,
        "rows": int(len(df)),
        "nan_percentage_per_column": {c: float(v) for c, v in (df.isna().mean() * 100.0).to_dict().items()},
        "constant_columns": [],
        "temporal_variation": {},
        "spatial_variation": {},
    }
    if df.empty:
        return out

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= 1:
            out["constant_columns"].append(c)

    time_col = next((c for c in ["week_start_utc", "week_start", "week", "timestamp", "date", "time"] if c in df.columns), None)
    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        out["temporal_variation"] = {
            "time_col": time_col,
            "n_unique_windows": int(t.dt.tz_localize(None).dt.to_period("W").nunique()),
        }

    grid_col = next((c for c in ["grid_cell_id", "grid_id", "cell_id"] if c in df.columns), None)
    if grid_col:
        out["spatial_variation"] = {
            "grid_col": grid_col,
            "n_unique_grids": int(df[grid_col].nunique(dropna=True)),
        }
    return out


def _save_hist(series: pd.Series, path: Path, title: str) -> None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=50, edgecolor="black", linewidth=0.4)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def generate_source_previews(source_name: str, df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    for c in numeric_cols:
        _save_hist(df[c], out_dir / f"hist_{c}.png", f"{source_name}: {c} distribution")

    time_col = next((c for c in ["week_start_utc", "week_start", "week", "timestamp", "date", "time"] if c in df.columns), None)
    if time_col and numeric_cols:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        grp = pd.DataFrame({"time": t, "value": pd.to_numeric(df[numeric_cols[0]], errors="coerce")}).dropna()
        if not grp.empty:
            weekly = grp.groupby(grp["time"].dt.tz_localize(None).dt.to_period("W")).agg(mean_value=("value", "mean")).reset_index()
            weekly["time"] = weekly["time"].astype(str)
            plt.figure(figsize=(10, 5))
            plt.plot(weekly["time"], weekly["mean_value"], marker="o", markersize=2, linewidth=1.1)
            plt.xticks(rotation=90)
            plt.title(f"{source_name}: weekly {numeric_cols[0]} mean")
            plt.tight_layout()
            plt.savefig(out_dir / "timeseries_weekly.png", dpi=220)
            plt.close()

    lat = next((c for c in ["grid_centroid_lat", "centroid_lat", "latitude", "lat"] if c in df.columns), None)
    lon = next((c for c in ["grid_centroid_lon", "centroid_lon", "longitude", "lon", "lng"] if c in df.columns), None)
    if lat and lon and numeric_cols:
        tmp = df[[lat, lon, numeric_cols[0]]].copy()
        tmp[lat] = pd.to_numeric(tmp[lat], errors="coerce")
        tmp[lon] = pd.to_numeric(tmp[lon], errors="coerce")
        tmp[numeric_cols[0]] = pd.to_numeric(tmp[numeric_cols[0]], errors="coerce")
        tmp = tmp.dropna()
        if not tmp.empty:
            plt.figure(figsize=(8, 5))
            s = plt.scatter(tmp[lon], tmp[lat], c=tmp[numeric_cols[0]], cmap="viridis", s=12, alpha=0.8)
            plt.colorbar(s, label=numeric_cols[0])
            plt.title(f"{source_name}: spatial proxy")
            plt.tight_layout()
            plt.savefig(out_dir / "spatial_proxy.png", dpi=220)
            plt.close()


def load_base_vessel_source(input_dataset: Path, quick_test: bool) -> pd.DataFrame:
    if not input_dataset.exists():
        return pd.DataFrame()
    base = pd.read_parquet(input_dataset)
    cols = [c for c in ["grid_cell_id", "week_start_utc", "vessel_density_t", "grid_centroid_lat", "grid_centroid_lon"] if c in base.columns]
    if not cols:
        return pd.DataFrame()
    out = base[cols].copy()
    if quick_test and not out.empty:
        out = out.sample(min(1500, len(out)), random_state=42).copy()
    return out


def merge_sources(source_frames: dict[str, pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    logger.info("[MERGE] Starting source alignment and merge")
    keys = ["grid_cell_id", "week_start_utc"]
    merged: pd.DataFrame | None = None
    for name, df in source_frames.items():
        if df.empty:
            continue
        local = df.copy()
        rename_map = {}
        if "grid_id" in local.columns and "grid_cell_id" not in local.columns:
            rename_map["grid_id"] = "grid_cell_id"
        if "week_start" in local.columns and "week_start_utc" not in local.columns:
            rename_map["week_start"] = "week_start_utc"
        if rename_map:
            local = local.rename(columns=rename_map)
        if not all(k in local.columns for k in keys):
            logger.warning("[MERGE] Skipping %s: missing merge keys", name)
            continue
        local["week_start_utc"] = pd.to_datetime(local["week_start_utc"], errors="coerce", utc=True)
        local = local.dropna(subset=keys).drop_duplicates(subset=keys, keep="last")
        merged = local if merged is None else merged.merge(local, on=keys, how="outer")
        logger.info("[MERGE] Included source=%s rows=%d", name, len(local))
    return merged if merged is not None else pd.DataFrame()


def feature_engineering(merged: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("[FEATURES] Building global ML-ready features")
    if merged.empty:
        return merged
    out = merged.copy()
    vv_candidates = [c for c in out.columns if "vv" in c.lower() and "std" not in c.lower()]
    vh_candidates = [c for c in out.columns if "vh" in c.lower() and "std" not in c.lower()]
    vv_std_candidates = [c for c in out.columns if "vv" in c.lower() and "std" in c.lower()]
    vh_std_candidates = [c for c in out.columns if "vh" in c.lower() and "std" in c.lower()]
    no2_candidates = [c for c in out.columns if c.lower().startswith("no2")]
    vessel_candidates = [c for c in out.columns if "vessel" in c.lower() and "density" in c.lower()]
    detect_candidates = [c for c in out.columns if "oil_slick_probability" in c.lower() or "detection_score" in c.lower()]

    if vv_candidates and "VV_mean" not in out.columns:
        out["VV_mean"] = pd.to_numeric(out[vv_candidates[0]], errors="coerce")
    if vh_candidates and "VH_mean" not in out.columns:
        out["VH_mean"] = pd.to_numeric(out[vh_candidates[0]], errors="coerce")
    if vv_std_candidates and "VV_std" not in out.columns:
        out["VV_std"] = pd.to_numeric(out[vv_std_candidates[0]], errors="coerce")
    if vh_std_candidates and "VH_std" not in out.columns:
        out["VH_std"] = pd.to_numeric(out[vh_std_candidates[0]], errors="coerce")
    if "VV_mean" in out.columns and "VH_mean" in out.columns and "VV_VH_ratio" not in out.columns:
        out["VV_VH_ratio"] = pd.to_numeric(out["VV_mean"], errors="coerce") / pd.to_numeric(out["VH_mean"], errors="coerce").replace(0, np.nan)

    if no2_candidates and "NO2_mean" not in out.columns:
        out["NO2_mean"] = pd.to_numeric(out[no2_candidates[0]], errors="coerce")
        if "week_start_utc" in out.columns and "grid_cell_id" in out.columns:
            out = out.sort_values(["grid_cell_id", "week_start_utc"])
            out["NO2_trend"] = out.groupby("grid_cell_id")["NO2_mean"].diff()
    if vessel_candidates and "vessel_density" not in out.columns:
        out["vessel_density"] = pd.to_numeric(out[vessel_candidates[0]], errors="coerce")
    if detect_candidates and "detection_score" not in out.columns:
        out["detection_score"] = pd.to_numeric(out[detect_candidates[0]], errors="coerce")

    out["nan_ratio_row"] = out.isna().mean(axis=1)
    return out


def global_validation(merged: pd.DataFrame, features: pd.DataFrame, skip_validation: bool, logger: logging.Logger) -> dict[str, Any]:
    stage = "[VALIDATION]"
    report: dict[str, Any] = {"cross_source": {}, "aux_reports": {}}
    if merged.empty:
        report["cross_source"]["status"] = "failed"
        report["cross_source"]["reason"] = "merged dataset is empty"
        return report

    key_cols = [c for c in ["grid_cell_id", "week_start_utc"] if c in merged.columns]
    missing_joins = float(merged[key_cols].isna().any(axis=1).mean() * 100.0) if key_cols else None
    report["cross_source"] = {
        "temporal_alignment_unique_windows": int(
            pd.to_datetime(merged["week_start_utc"], errors="coerce", utc=True).dt.tz_localize(None).dt.to_period("W").nunique()
        )
        if "week_start_utc" in merged.columns
        else None,
        "missing_join_keys_percent": missing_joins,
        "feature_completeness_percent": float((1.0 - features.isna().mean().mean()) * 100.0) if not features.empty else 0.0,
    }

    correlation = {}
    cols = [c for c in ["detection_score", "NO2_mean", "vessel_density"] if c in features.columns]
    if len(cols) >= 2:
        correlation = features[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").round(4).to_dict()
    report["cross_source"]["cross_signal_correlation_spearman"] = correlation

    # Basic flags
    report["cross_source"]["no_temporal_variation"] = report["cross_source"].get("temporal_alignment_unique_windows", 0) <= 1
    if "grid_cell_id" in features.columns:
        by_grid = features.groupby("grid_cell_id").size()
        report["cross_source"]["spatial_flatness"] = bool(by_grid.nunique(dropna=True) <= 1)
    else:
        report["cross_source"]["spatial_flatness"] = None

    if skip_validation:
        logger.info("%s Skipping validate_aux_layers.py per flag", stage)
        return report

    try:
        from validation.validate_aux_layers import integrated_validation, resolve_oil_parquet, validate_no2, validate_oil

        root = ROOT
        no2_p = root / "data" / "aux" / "no2_grid_week.parquet"
        oil_p = resolve_oil_parquet(root, None)
        no2_meta = root / "data" / "aux" / "no2_gee_validation.json"
        oil_meta = root / "data" / "aux" / "sentinel1_oil_validation.json"
        report["aux_reports"]["no2_validation_report"] = validate_no2(root, no2_p, no2_meta)
        report["aux_reports"]["sentinel1_oil_validation_report"] = validate_oil(root, oil_p, oil_meta)
        report["aux_reports"]["human_impact_integrity_report"] = integrated_validation(root, no2_p, oil_p)
        logger.info("%s Completed validate_aux_layers checks", stage)
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s validate_aux_layers call failed: %s", stage, exc)
        report["aux_reports"]["error"] = str(exc)
    return report


def run_eda(features_path: Path, skip_eda: bool, logger: logging.Logger) -> None:
    if skip_eda:
        logger.info("[EDA] Skipped by flag")
        return
    if not features_path.exists():
        logger.warning("[EDA] Features parquet missing: %s", features_path)
        return
    try:
        from analysis.eda_report import run as run_eda_report

        run_eda_report(
            input_path=features_path,
            output_dir=ROOT / "outputs" / "eda",
            summary_path=ROOT / "outputs" / "eda_summary.md",
            weekly_breakdown=True,
        )
        logger.info("[EDA] EDA report completed")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[EDA] Failed to run EDA: %s", exc)


def run_correlation_analysis(
    features_path: Path,
    skip_correlation: bool,
    show_correlation: bool,
    logger: logging.Logger,
) -> None:
    if skip_correlation:
        logger.info("[CORRELATION] Skipped by flag")
        return
    if not features_path.exists():
        logger.warning("[CORRELATION] Features parquet missing: %s", features_path)
        return
    try:
        from analysis.correlation_analysis import run as run_correlation

        run_correlation(
            input_path=features_path,
            reports_dir=ROOT / "outputs" / "reports",
            plots_dir=ROOT / "outputs" / "plots" / "correlations",
            show_plots=show_correlation,
        )
        logger.info("[CORRELATION] Correlation analysis completed")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[CORRELATION] Failed to run correlation analysis: %s", exc)


def run_feature_interaction_map_stage(
    features: pd.DataFrame,
    enabled: bool,
    logger: logging.Logger,
) -> None:
    if not enabled:
        logger.info("[FEATURE_INTERACTION] Skipped by flag")
        return
    if features.empty:
        logger.warning("[FEATURE_INTERACTION] Skipped: feature dataframe is empty")
        return
    try:
        from analysis.feature_interaction_map import run_feature_interaction_map

        run_feature_interaction_map(features, FEATURE_REGISTRY, logger)
        logger.info("[FEATURE_INTERACTION] Completed")
    except Exception as exc:  # noqa: BLE001
        # Non-blocking analytical layer: log and continue pipeline completion.
        logger.exception("[FEATURE_INTERACTION] Failed but continuing pipeline: %s", exc)


def run_scientific_validation_stage(
    features: pd.DataFrame,
    enabled: bool,
    logger: logging.Logger,
) -> None:
    if not enabled:
        logger.info("[SCIENTIFIC VALIDATION] Skipped by flag")
        return
    if features.empty:
        logger.warning("[SCIENTIFIC VALIDATION] Skipped: feature dataframe is empty")
        return
    try:
        from analysis.scientific_validation import run_scientific_validation

        run_scientific_validation(features, FEATURE_REGISTRY, logger)
        logger.info("[SCIENTIFIC VALIDATION] Completed")
    except Exception as exc:  # noqa: BLE001
        # Non-blocking analytical layer: never crash pipeline.
        logger.exception("[SCIENTIFIC VALIDATION] Failed but continuing pipeline: %s", exc)


def run_anomaly_detection_stage(
    features: pd.DataFrame,
    enabled: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not enabled:
        logger.info("[ANOMALY] Skipped by flag")
        return pd.DataFrame()
    if features.empty:
        logger.warning("[ANOMALY] Skipped: feature dataframe is empty")
        return pd.DataFrame()
    try:
        from analysis.anomaly_detection import run_anomaly_detection

        out = run_anomaly_detection(features, FEATURE_REGISTRY, logger)
        logger.info("[ANOMALY] Completed")
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("[ANOMALY] Failed but continuing pipeline: %s", exc)
        return pd.DataFrame()


def run_coastal_impact_score_stage(
    features: pd.DataFrame,
    enabled: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not enabled:
        logger.info("[COASTAL IMPACT] Skipped by flag")
        return pd.DataFrame()
    if features.empty:
        logger.warning("[COASTAL IMPACT] Skipped: feature dataframe is empty")
        return pd.DataFrame()
    try:
        from analysis.coastal_impact_score import run_coastal_impact_score

        out = run_coastal_impact_score(features, FEATURE_REGISTRY, logger)
        logger.info("[COASTAL IMPACT] Completed")
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("[COASTAL IMPACT] Failed but continuing pipeline: %s", exc)
        return pd.DataFrame()


def run_final_visualization_stage(
    features: pd.DataFrame,
    enabled: bool,
    logger: logging.Logger,
) -> None:
    if not enabled:
        logger.info("[FINAL VIS] Skipped by flag")
        return
    if features.empty:
        logger.warning("[FINAL VIS] Skipped: feature dataframe is empty")
        return
    try:
        from visualization.impact_heatmap import run_final_visualization

        run_final_visualization(features, logger)
        logger.info("[FINAL VIS] Completed")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[FINAL VIS] Failed but continuing pipeline: %s", exc)


# ---------------------------------------------------------------------------
# LAND IMPACT EXTENSION LAYER — additive analytical stages
# ---------------------------------------------------------------------------

def run_land_impact_extension(
    features: pd.DataFrame,
    features_path: Path,
    *,
    enabled: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run the Land Impact extension stack on the ML-ready feature frame.

    Order:
      1. land-sea buffering (coastal exposure features)
      2. land-sea interaction indices + lagged product terms
      3. lagged land-sea correlation analysis
      4. optional RF with permutation importance
      5. final land-impact report CSV

    Persists the enriched features back to `features_path` so downstream
    stages (correlation, anomaly, coastal impact, finalization) observe the
    new NDVI/buffering/interaction columns. Returns the enriched frame.
    """
    if not enabled:
        logger.info("[LAND IMPACT] Skipped by flag")
        return features
    if features.empty:
        logger.warning("[LAND IMPACT] Skipped: feature dataframe is empty")
        return features

    # Coverage safeguard (non-blocking): NDVI is only defined over coastal/inland
    # cells, so low coverage is EXPECTED for maritime-dominant study areas. We
    # log a warning if coverage drops below 5% but NEVER abort — downstream
    # stages (buffering, interactions, correlation, ML, report) continue to run
    # on whatever NDVI samples are available.
    ndvi_coverage_percent: float | None = None
    if "ndvi_mean" in features.columns:
        ndvi_values = pd.to_numeric(features["ndvi_mean"], errors="coerce")
        valid = int(ndvi_values.notna().sum())
        ndvi_coverage_percent = round(100.0 * valid / max(1, len(features)), 4)
        if ndvi_coverage_percent < 5.0:
            logger.warning(
                "[LAND IMPACT] NDVI coverage %.2f%% below 5%% threshold — continuing pipeline (non-blocking). "
                "Typical for maritime-dominant study areas where most grids are offshore.",
                ndvi_coverage_percent,
            )
        else:
            logger.info("[LAND IMPACT] NDVI coverage %.2f%%", ndvi_coverage_percent)
    else:
        logger.warning(
            "[LAND IMPACT] ndvi_mean column missing from features — downstream NDVI-dependent outputs will be empty"
        )

    enriched = features

    try:
        from features.land_sea_buffering import run_land_sea_buffering

        enriched = run_land_sea_buffering(enriched, logger=logger)
        logger.info("[LAND IMPACT] Coastal-exposure features appended")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[LAND IMPACT] Buffering failed but continuing: %s", exc)

    try:
        from features.land_sea_interactions import add_land_sea_interactions

        enriched = add_land_sea_interactions(enriched, logger=logger)
        logger.info("[LAND IMPACT] Interaction features appended")
    except Exception as exc:  # noqa: BLE001
        logger.exception("[LAND IMPACT] Interaction feature build failed but continuing: %s", exc)

    try:
        enriched.to_parquet(features_path, index=False)
        logger.info(
            "[LAND IMPACT] Persisted enriched features to %s (shape=%d×%d)",
            features_path,
            len(enriched),
            len(enriched.columns),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[LAND IMPACT] Failed to persist enriched features: %s", exc)

    reports_dir = ROOT / "outputs" / "reports"
    lag_summary_path = reports_dir / "land_sea_lag_summary.csv"
    try:
        from analysis.land_sea_correlation import run_land_sea_correlation

        run_land_sea_correlation(enriched, reports_dir=reports_dir, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[LAND IMPACT] Lagged correlation stage failed but continuing: %s", exc)

    try:
        from analysis.land_impact_ml import run_land_impact_ml

        run_land_impact_ml(enriched, reports_dir=reports_dir, logger=logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[LAND IMPACT] ML stage failed but continuing: %s", exc)

    try:
        from analysis.land_impact_report import run_land_impact_report

        run_land_impact_report(
            enriched,
            reports_dir=reports_dir,
            lag_summary_path=lag_summary_path,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[LAND IMPACT] Final report stage failed but continuing: %s", exc)

    # Refresh land-category feature registry entries against the enriched frame.
    try:
        register_and_print_feature_tables(enriched, "land_impact_ndvi", "land_impact_extension", logger)
    except Exception:  # noqa: BLE001
        pass

    return enriched


def print_final_thesis_summary(logger: logging.Logger) -> None:
    reports = ROOT / "outputs" / "reports"
    logger.info("[THESIS SUMMARY] Top 10 coastal impact zones (best week per distinct cell)")
    try:
        cis = pd.read_csv(reports / "coastal_impact_score.csv")
        cols = [c for c in ["grid_cell_id", "week_start_utc", "coastal_impact_score"] if c in cis.columns]
        if cols and "grid_cell_id" in cis.columns:
            best_per_cell = (
                cis.sort_values("coastal_impact_score", ascending=False)
                .drop_duplicates("grid_cell_id", keep="first")
                .head(10)[cols]
            )
            for _, row in best_per_cell.iterrows():
                logger.info(
                    "- %s | %s | score=%.4f",
                    row.get("grid_cell_id"),
                    row.get("week_start_utc"),
                    float(row.get("coastal_impact_score", np.nan)),
                )
    except Exception:  # noqa: BLE001
        logger.info("- unavailable")

    logger.info("[THESIS SUMMARY] Top 10 spatial-outlier grid-week events")
    try:
        adf = pd.read_csv(reports / "anomaly_scores.csv")
        cols = [c for c in ["grid_cell_id", "week_start_utc", "anomaly_score"] if c in adf.columns]
        if cols:
            for _, row in adf.sort_values("anomaly_score", ascending=False).head(10)[cols].iterrows():
                logger.info(
                    "- %s | %s | anomaly=%.4f",
                    row.get("grid_cell_id"),
                    row.get("week_start_utc"),
                    float(row.get("anomaly_score", np.nan)),
                )
    except Exception:  # noqa: BLE001
        logger.info("- unavailable")

    logger.info("[THESIS SUMMARY] Top 10 within-cell temporal anomalies")
    try:
        tdf_path = reports / "anomaly_scores_temporal.csv"
        if tdf_path.exists():
            tdf = pd.read_csv(tdf_path)
            anom = tdf[tdf.get("temporal_anomaly_label") == "anomalous"].copy()
            anom["abs_z"] = anom["temporal_z_score"].abs()
            cols = [
                c
                for c in ["grid_cell_id", "week_start_utc", "anomaly_score", "temporal_z_score"]
                if c in anom.columns
            ]
            for _, row in anom.sort_values("abs_z", ascending=False).head(10)[cols].iterrows():
                logger.info(
                    "- %s | %s | z=%.2f | anomaly=%.4f",
                    row.get("grid_cell_id"),
                    row.get("week_start_utc"),
                    float(row.get("temporal_z_score", np.nan)),
                    float(row.get("anomaly_score", np.nan)),
                )
        else:
            logger.info("- temporal anomaly report not produced")
    except Exception:  # noqa: BLE001
        logger.info("- unavailable")

    logger.info("[THESIS SUMMARY] Strongest cross-domain correlations")
    try:
        from analysis.results_aggregator import summarize_correlations

        corr = summarize_correlations(min_abs=0.1)
        for row in corr.get("top10", [])[:10]:
            logger.info(
                "- %s vs %s (%s): %.4f",
                row.get("feature_x"),
                row.get("feature_y"),
                row.get("method"),
                float(row.get("value", np.nan)),
            )
    except Exception:  # noqa: BLE001
        logger.info("- unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full multi-source extraction → validation → preview → merge → features → EDA")
    parser.add_argument("--quick-test", action="store_true", help="Run lightweight sample mode")
    parser.add_argument("--skip-eda", action="store_true", help="Skip final EDA stage")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation stage")
    parser.add_argument("--skip-correlation", action="store_true", help="Skip correlation analysis stage")
    parser.add_argument("--show-correlation", action="store_true", help="Show correlation plots interactively during run")
    parser.add_argument("--feature-interaction-map", action="store_true", help="Run feature interaction map analysis stage")
    parser.add_argument("--scientific-validation", action="store_true", help="Run temporal/lag/ML scientific validation stage")
    parser.add_argument("--anomaly-detection", action="store_true", help="Run anomaly detection on combined feature space")
    parser.add_argument("--coastal-impact-score", action="store_true", help="Compute composite coastal impact score")
    parser.add_argument("--final-visualization", action="store_true", help="Render final environmental pressure visualization")
    parser.add_argument(
        "--sentinel-safe-mode",
        action="store_true",
        help="Enable Sentinel-1 extraction safety mode (limits grids and increases runtime logging).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-extraction of remote-sensing sources (skip aux artifact reuse).",
    )
    parser.add_argument(
        "--land-impact",
        action="store_true",
        help=(
            "Enable the LAND IMPACT EXTENSION LAYER: NDVI-on-land extraction, "
            "coastal buffering features, interaction indices, lagged land-sea "
            "correlation, optional RF + permutation importance, and a final "
            "land_impact_analysis.csv report."
        ),
    )
    args = parser.parse_args()

    logger = setup_logger(ROOT / "logs" / "pipeline_run.log")
    logger.info("=== Full Pipeline Start ===")
    for key in FEATURE_REGISTRY:
        FEATURE_REGISTRY[key].clear()

    # LAND IMPACT bootstrap cache invalidation.
    # `--land-impact` alone reuses the existing NDVI parquet. Pair it with
    # `--force-refresh` to clear the cache and force a fresh GEE re-extraction.
    if args.land_impact:
        ndvi_cache_files = [
            ROOT / "data" / "aux" / "sentinel2_land_metrics.parquet",
            ROOT / "data" / "aux" / "sentinel2_land_metrics_validation.json",
        ]
        if args.force_refresh:
            removed = [p for p in ndvi_cache_files if p.exists()]
            for p in removed:
                try:
                    p.unlink()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[LAND IMPACT] Failed to remove %s: %s", p, exc)
            if removed:
                logger.info(
                    "NDVI aux cache cleared via --force-refresh (removed %d file(s))",
                    len(removed),
                )
                for p in removed:
                    logger.info("  removed: %s", p)
            else:
                logger.info("NDVI aux cache check: nothing to clear; fresh extraction will run")
        else:
            existing = [p for p in ndvi_cache_files if p.exists()]
            if existing:
                logger.info(
                    "[LAND IMPACT] reusing cached NDVI parquet (%d artifact(s)); pass --force-refresh to re-extract",
                    len(existing),
                )
            else:
                logger.info(
                    "[LAND IMPACT] no NDVI cache present — extraction will run on first execution"
                )

    input_dataset = ROOT / "data" / "modeling_dataset.parquet"
    interm_dir = ROOT / "data" / "intermediate"
    preview_root = ROOT / "outputs" / "previews"
    processed_dir = ROOT / "processed"
    validation_dir = ROOT / "data" / "validation"
    for p in [interm_dir, preview_root, processed_dir, validation_dir]:
        p.mkdir(parents=True, exist_ok=True)

    source_frames: dict[str, pd.DataFrame] = {}
    source_reports: dict[str, Any] = {}

    # Include vessel/AIS baseline from master modeling parquet.
    vessel_df = load_base_vessel_source(input_dataset, quick_test=args.quick_test)
    if not vessel_df.empty:
        vessel_path = interm_dir / "vessels.parquet"
        vessel_df.to_parquet(vessel_path, index=False)
        source_frames["vessels"] = vessel_df
        rep = quality_checks(vessel_df)
        source_reports["vessels"] = rep
        logger.info("[SOURCE: vessels] rows=%d non_null_pct=%.2f", len(vessel_df), (1.0 - vessel_df.isna().mean().mean()) * 100.0)
        generate_source_previews("vessels", vessel_df, preview_root / "vessels")
        register_and_print_feature_tables(vessel_df, "vessels", "vessels", logger)

    for source in discover_sources(include_land_impact=args.land_impact):
        try:
            df, extraction_meta = run_source_extraction(
                source,
                input_dataset,
                args.quick_test,
                args.sentinel_safe_mode,
                args.force_refresh,
                logger,
            )
            if args.quick_test and not df.empty:
                df = df.sample(min(len(df), 1500), random_state=42).copy()
            path = interm_dir / f"{source.name}.parquet"
            df.to_parquet(path, index=False)
            source_frames[source.name] = df
            qc = quality_checks(df)
            source_reports[source.name] = {"quality": qc, "extraction_meta": extraction_meta}
            logger.info(
                "[SOURCE: %s] rows=%d non_null_pct=%.2f constant_cols=%d",
                source.name,
                len(df),
                (1.0 - df.isna().mean().mean()) * 100.0 if not df.empty else 0.0,
                len(qc.get("constant_columns", [])),
            )
            generate_source_previews(source.name, df, preview_root / source.name)
            register_and_print_feature_tables(df, source.name, source.module_name, logger)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SOURCE: %s] extraction failed: %s", source.name, exc)
            source_reports[source.name] = {"error": str(exc)}
            if source.critical:
                raise
            continue

    merged = merge_sources(source_frames, logger)
    merged_path = processed_dir / "merged_dataset.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.info("[MERGE] Wrote %s with %d rows", merged_path, len(merged))

    features = feature_engineering(merged, logger)
    features_path = processed_dir / "features_ml_ready.parquet"
    features.to_parquet(features_path, index=False)
    logger.info("[FEATURES] Wrote %s with %d rows and %d cols", features_path, len(features), len(features.columns))

    # LAND IMPACT EXTENSION LAYER — additive; non-blocking; runs before downstream
    # analytics so correlation/anomaly/coastal-impact see the new land features.
    features = run_land_impact_extension(
        features,
        features_path,
        enabled=args.land_impact,
        logger=logger,
    )

    validation_report = global_validation(merged, features, args.skip_validation, logger)
    val_out = validation_dir / "full_pipeline_validation_report.json"
    val_out.write_text(json.dumps(validation_report, indent=2, default=str), encoding="utf-8")
    logger.info("[VALIDATION] Wrote %s", val_out)

    run_eda(features_path, skip_eda=args.skip_eda, logger=logger)
    run_correlation_analysis(
        features_path,
        skip_correlation=args.skip_correlation,
        show_correlation=args.show_correlation,
        logger=logger,
    )
    run_feature_interaction_map_stage(features, enabled=args.feature_interaction_map, logger=logger)
    run_scientific_validation_stage(features, enabled=args.scientific_validation, logger=logger)
    run_anomaly_detection_stage(features, enabled=args.anomaly_detection, logger=logger)
    run_coastal_impact_score_stage(features, enabled=args.coastal_impact_score, logger=logger)
    run_final_visualization_stage(features, enabled=args.final_visualization, logger=logger)
    # Register post-merge feature groups that may be added downstream.
    register_and_print_feature_tables(features, "features", "pipeline.features", logger)

    full_summary = {
        "sources": source_reports,
        "merged_rows": int(len(merged)),
        "features_shape": [int(features.shape[0]), int(features.shape[1])],
        "paths": {
            "merged_dataset": str(merged_path),
            "features_ml_ready": str(features_path),
            "validation_report": str(val_out),
            "log_file": str(ROOT / "logs" / "pipeline_run.log"),
        },
    }
    summary_path = validation_dir / "full_pipeline_run_summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2, default=str), encoding="utf-8")
    logger.info("=== Full Pipeline Complete ===")
    logger.info("Summary written: %s", summary_path)
    print_feature_registry_summary(logger)
    feature_registry_path = validation_dir / "feature_registry_summary.json"
    write_feature_registry_summary_json(feature_registry_path)
    logger.info("Feature registry summary written: %s", feature_registry_path)
    print_final_thesis_summary(logger)


if __name__ == "__main__":
    main()
