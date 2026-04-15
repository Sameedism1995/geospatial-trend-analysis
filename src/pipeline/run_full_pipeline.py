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


def discover_sources() -> list[SourceConfig]:
    import data_sources

    out: list[SourceConfig] = []
    known_outputs = {
        "no2_gee_pipeline": ("no2", ROOT / "data" / "aux" / "no2_grid_week.parquet", ROOT / "data" / "aux" / "no2_gee_validation.json"),
        "sentinel1_oil_pipeline": (
            "sentinel1",
            ROOT / "data" / "aux" / "sentinel1_oil_slicks.parquet",
            ROOT / "data" / "aux" / "sentinel1_oil_validation.json",
        ),
    }
    for mod in pkgutil.iter_modules(data_sources.__path__):
        if mod.name.startswith("_") or mod.name in {"gee_grid_utils", "run_eo_pipeline"}:
            continue
        if mod.name in known_outputs:
            name, out_p, val_p = known_outputs[mod.name]
            out.append(SourceConfig(name=name, module_name=f"data_sources.{mod.name}", output_path=out_p, validation_path=val_p))
    return out


def run_source_extraction(source: SourceConfig, input_dataset: Path, quick_test: bool, logger: logging.Logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage = f"[SOURCE: {source.name}]"
    logger.info("%s Starting extraction", stage)
    module = importlib.import_module(source.module_name)
    validation: dict[str, Any] = {}

    # Quick mode favors reproducibility/speed by reusing existing artifacts when available.
    if quick_test and source.output_path.exists():
        logger.info("%s quick-test reusing existing artifact: %s", stage, source.output_path)
        df = pd.read_parquet(source.output_path)
        if not df.empty:
            df = df.sample(min(len(df), 1500), random_state=42).copy()
        return df, validation

    if hasattr(module, "run_extraction"):
        # Preferred standard interface requested by user.
        cfg = {"input": input_dataset, "output": source.output_path, "validation_json": source.validation_path, "quick_test": quick_test}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Full multi-source extraction → validation → preview → merge → features → EDA")
    parser.add_argument("--quick-test", action="store_true", help="Run lightweight sample mode")
    parser.add_argument("--skip-eda", action="store_true", help="Skip final EDA stage")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation stage")
    args = parser.parse_args()

    logger = setup_logger(ROOT / "logs" / "pipeline_run.log")
    logger.info("=== Full Pipeline Start ===")

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

    for source in discover_sources():
        try:
            df, extraction_meta = run_source_extraction(source, input_dataset, args.quick_test, logger)
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

    validation_report = global_validation(merged, features, args.skip_validation, logger)
    val_out = validation_dir / "full_pipeline_validation_report.json"
    val_out.write_text(json.dumps(validation_report, indent=2, default=str), encoding="utf-8")
    logger.info("[VALIDATION] Wrote %s", val_out)

    run_eda(features_path, skip_eda=args.skip_eda, logger=logger)

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


if __name__ == "__main__":
    main()
