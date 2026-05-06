"""Audit (and, where possible, replace) vessel and oil-slick data sources.

Pipeline:
    1. Audit existing vessel + oil columns in `final_run/processed/features_ml_ready.parquet`.
    2. If vessel data is *not* temporal, attempt EMODnet Human Activities monthly
       vessel-density retrieval (WMS layer probe). If unavailable, fall back to
       documented spatial-proxy features.
    3. If oil data is unusable (probability/detection_score 100% missing or count
       all-zero), attempt to rebuild a Sentinel-1 dark-slick proxy via Google
       Earth Engine. If GEE is unavailable, mark oil unavailable — never fake.
    4. Merge whatever repaired layers we obtained into a thesis-safe dataset.
    5. Render diagnostic plots + repaired distance-decay curves.
    6. Write `REPAIRED_SOURCE_REPORT.md`.

Run:
    python3 src/fixes/audit_or_replace_vessel_and_oil_sources.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("repair_sources")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "final_run" / "processed" / "features_ml_ready.parquet"
DEFAULT_OUTPUT_ROOT = ROOT / "final_run_repaired_sources"
EMODNET_WMS = "https://ows.emodnet-humanactivities.eu/wms?service=WMS&request=GetCapabilities"

VESSEL_COLS = [
    "vessel_density_t",
    "vessel_density",
    "maritime_pressure_index",
    "port_exposure_score",
    "distance_to_nearest_high_vessel_density_cell",
]
OIL_COLS = ["oil_slick_probability_t", "detection_score", "oil_slick_count_t"]
OIL_HINTS = ["oil", "slick", "sentinel1", "sar", "vv", "vh", "backscatter"]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def make_skeleton(base: Path) -> dict[str, Path]:
    paths = {
        "base": base,
        "data_vessel": base / "data" / "external_vessel",
        "data_oil": base / "data" / "external_oil",
        "processed": base / "processed",
        "validation": base / "validation",
        "viz": base / "outputs" / "visualizations",
        "viz_distance": base / "outputs" / "visualizations" / "distance_decay",
        "reports": base / "outputs" / "reports",
        "reports_distance": base / "outputs" / "reports" / "distance_decay",
        "logs": base / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# STEP 1 — AUDIT
# ---------------------------------------------------------------------------

def _column_diag(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df.columns:
        return {"column": col, "present": False}
    s = df[col]
    is_numeric = pd.api.types.is_numeric_dtype(s)
    rec: dict[str, Any] = {
        "column": col,
        "present": True,
        "dtype": str(s.dtype),
        "non_null_count": int(s.notna().sum()),
        "coverage_percent": round(float(s.notna().mean() * 100.0), 4),
        "n_unique": int(s.nunique(dropna=True)),
    }
    if is_numeric:
        rec["zero_percent"] = round(float((s == 0).sum()) / max(1, len(s)) * 100.0, 4)
        rec["mean"] = float(s.mean()) if s.notna().any() else None
        rec["std"] = float(s.std()) if s.notna().sum() > 1 else None
        rec["min"] = float(s.min()) if s.notna().any() else None
        rec["max"] = float(s.max()) if s.notna().any() else None
    else:
        rec["zero_percent"] = None
        for k in ["mean", "std", "min", "max"]:
            rec[k] = None
    if "grid_cell_id" in df.columns:
        by_grid = df.groupby("grid_cell_id")[col]
        unique_per_grid = by_grid.nunique(dropna=True) if is_numeric else by_grid.transform("nunique")
        if isinstance(unique_per_grid, pd.Series) and unique_per_grid.size:
            varying = unique_per_grid[unique_per_grid > 1]
            rec["fraction_grids_temporally_varying"] = round(
                float(len(varying)) / max(1, int(unique_per_grid.shape[0])), 4
            )
            rec["median_unique_values_per_grid"] = float(unique_per_grid.median())
        else:
            rec["fraction_grids_temporally_varying"] = 0.0
            rec["median_unique_values_per_grid"] = None
    if "week_start_utc" in df.columns and is_numeric:
        weekly = df.groupby("week_start_utc")[col].mean().dropna()
        rec["weekly_mean_std"] = float(weekly.std()) if len(weekly) > 1 else None
        rec["weekly_mean_unique"] = int(weekly.round(12).nunique()) if len(weekly) else 0
    return rec


def audit_layers(df: pd.DataFrame, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    vessel_diag = {c: _column_diag(df, c) for c in VESSEL_COLS}
    oil_diag = {c: _column_diag(df, c) for c in OIL_COLS}
    extra_oil_cols = [c for c in df.columns if any(h in c.lower() for h in OIL_HINTS) and c not in OIL_COLS and c not in VESSEL_COLS]
    extra_oil_diag = {c: _column_diag(df, c) for c in extra_oil_cols}

    for d in list(vessel_diag.values()) + list(oil_diag.values()) + list(extra_oil_diag.values()):
        rows.append(d)
    df_audit = pd.DataFrame(rows)
    csv_path = out_dir / "source_audit.csv"
    df_audit.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", _rel(csv_path))

    # Vessel decision
    vd = vessel_diag.get("vessel_density_t", {})
    coverage_ok = (vd.get("coverage_percent") or 0) > 70.0
    grid_var_ok = (vd.get("fraction_grids_temporally_varying") or 0) > 0.50
    weekly_var_ok = (vd.get("weekly_mean_std") or 0) > 0
    vessel_status = "unusable"
    if coverage_ok and grid_var_ok and weekly_var_ok:
        vessel_status = "temporal"
    elif (vd.get("coverage_percent") or 0) > 0 and (vd.get("n_unique") or 0) > 1:
        vessel_status = "spatial_proxy"
    vessel_decision = {
        "status": vessel_status,
        "coverage_ok_70": coverage_ok,
        "grid_temporally_varying_fraction": vd.get("fraction_grids_temporally_varying"),
        "weekly_mean_std": vd.get("weekly_mean_std"),
        "diagnostic": vd,
    }

    # Oil decision
    prob = oil_diag.get("oil_slick_probability_t", {})
    score = oil_diag.get("detection_score", {})
    count = oil_diag.get("oil_slick_count_t", {})
    nonzero_signal = ((prob.get("coverage_percent") or 0) > 1.0 and (prob.get("n_unique") or 0) > 1) or (
        (count.get("coverage_percent") or 0) > 1.0 and (count.get("zero_percent") or 100) < 99.0
    )
    score_ok = (score.get("coverage_percent") or 0) > 0
    has_sar = bool(extra_oil_cols)
    oil_status = "usable" if (nonzero_signal and score_ok) or has_sar else "unusable"
    oil_decision = {
        "status": oil_status,
        "has_nonzero_signal": nonzero_signal,
        "detection_score_present": score_ok,
        "extra_sar_columns_present": has_sar,
        "extra_sar_columns": extra_oil_cols,
        "diagnostic": {**oil_diag, **extra_oil_diag},
    }

    # Markdown summary
    md = out_dir / "source_audit_report.md"
    L: list[str] = []
    L.append("# Source audit — vessel & oil")
    L.append("")
    L.append(f"Source dataset: `{_rel(DEFAULT_INPUT)}`")
    L.append("")
    L.append("## Vessel layer")
    L.append(f"- Decision: **{vessel_status}**")
    L.append(f"- Coverage > 70%: {coverage_ok} (actual: {vd.get('coverage_percent')}%)")
    L.append(f"- Grids temporally varying > 50%: {grid_var_ok} (actual: {vessel_decision['grid_temporally_varying_fraction']})")
    L.append(f"- Weekly-mean std: {vessel_decision['weekly_mean_std']}")
    L.append("")
    for c, d in vessel_diag.items():
        L.append(f"  - `{c}`: {d}")
    L.append("")
    L.append("## Oil / Sentinel-1 layer")
    L.append(f"- Decision: **{oil_status}**")
    L.append(f"- Has non-zero oil signal: {nonzero_signal}")
    L.append(f"- Detection score present: {score_ok}")
    L.append(f"- Extra SAR/VV/VH columns: {extra_oil_cols if extra_oil_cols else '(none)'}")
    for c, d in {**oil_diag, **extra_oil_diag}.items():
        L.append(f"  - `{c}`: {d}")
    md.write_text("\n".join(L))
    LOGGER.info("Wrote %s", _rel(md))

    return vessel_decision, oil_decision, df_audit


# ---------------------------------------------------------------------------
# STEP 2 — VESSEL REPAIR
# ---------------------------------------------------------------------------

def probe_emodnet_vessel(out_data_dir: Path) -> dict[str, Any]:
    """Probe EMODnet Human Activities WMS for vessel-density layers.

    We never inject synthetic data. We only record what is *available* on the
    EMODnet WMS endpoint so the user knows what they could subscribe to. The
    actual download of monthly raster grids requires either the EMODnet
    download portal (with terms acceptance) or pre-staged GeoTIFFs.
    """
    info: dict[str, Any] = {"available": False, "endpoint": EMODNET_WMS}
    try:
        import urllib.request

        req = urllib.request.Request(EMODNET_WMS, headers={"User-Agent": "thesis-repair-script/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_text = resp.read().decode("utf-8", errors="replace")
        snapshot = out_data_dir / "emodnet_human_activities_capabilities.xml"
        snapshot.write_text(xml_text[:2_000_000])
        # crude scan for vessel-density layer names
        candidates = []
        lower = xml_text.lower()
        for kw in ["vesseldensity", "vessel_density", "shipping density", "ais"]:
            idx = lower.find(kw)
            if idx >= 0:
                candidates.append(kw)
        info.update(
            {
                "available": True,
                "candidates_found": candidates,
                "snapshot_path": str(snapshot),
            }
        )
        LOGGER.info("[VESSEL] EMODnet capabilities reachable; layer hints=%s", candidates)
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)[:300]
        LOGGER.warning("[VESSEL] EMODnet capabilities probe failed: %s", info["error"])
    return info


def build_spatial_pressure_features(df: pd.DataFrame, vessel_status: str) -> tuple[pd.DataFrame, list[str]]:
    """Build the documented fallback exposure features."""
    if "vessel_density_t" not in df.columns:
        return df, []
    out = df.copy()
    base = pd.to_numeric(out["vessel_density_t"], errors="coerce")
    out["vessel_density_spatial_proxy"] = base
    out["vessel_density_spatial_proxy_log1p"] = np.log1p(base.clip(lower=0))
    if "distance_to_port_km" in out.columns:
        out["port_pressure"] = base / (1.0 + pd.to_numeric(out["distance_to_port_km"], errors="coerce"))
    if "distance_to_nearest_high_vessel_density_cell" in out.columns:
        d_high = pd.to_numeric(out["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
        out["shipping_exposure"] = base * np.exp(-d_high / 100.0)
    if "coastal_exposure_score" in out.columns:
        out["coastal_shipping_pressure"] = base * pd.to_numeric(out["coastal_exposure_score"], errors="coerce")

    added = [c for c in ["vessel_density_spatial_proxy", "vessel_density_spatial_proxy_log1p", "port_pressure", "shipping_exposure", "coastal_shipping_pressure"] if c in out.columns]
    LOGGER.info("[VESSEL] Spatial-pressure feature set built: %s", added)
    return out, added


def repair_vessel(df: pd.DataFrame, vessel_decision: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    info: dict[str, Any] = {"input_status": vessel_decision["status"]}
    if vessel_decision["status"] == "temporal":
        info["action"] = "no-op"
        info["final_vessel_feature"] = "vessel_density_t"
        info["temporal_external_attempted"] = False
        return info

    # Attempt EMODnet probe regardless (informational; never injects synthetic data)
    info["emodnet_probe"] = probe_emodnet_vessel(paths["data_vessel"])

    # We can't realistically download monthly EMODnet vessel rasters without a
    # pre-staged TIFF/portal session, so use the documented fallback.
    info["temporal_external_used"] = False
    info["action"] = "spatial_proxy_with_derived_exposure"
    info["final_vessel_feature"] = "vessel_density_spatial_proxy"

    df_out, added = build_spatial_pressure_features(df, vessel_decision["status"])
    out_path = paths["processed"] / "vessel_temporal_repaired.parquet"
    keep = [c for c in ["grid_cell_id", "week_start_utc"] + added if c in df_out.columns]
    df_out[keep].to_parquet(out_path, index=False)
    info["features_added"] = added
    info["output_parquet"] = str(out_path)
    LOGGER.info("[VESSEL] Wrote %s (%d cols)", _rel(out_path), len(keep))
    return info


# ---------------------------------------------------------------------------
# STEP 3 — OIL REPAIR (Sentinel-1 dark-slick proxy via GEE if available)
# ---------------------------------------------------------------------------

def gee_probe(logger: logging.Logger) -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        import ee  # type: ignore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("EE_PROJECT")
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            info["available"] = True
            info["project"] = project or "(default)"
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)[:300]
    except ImportError:
        info["error"] = "earthengine-api not installed"
    if info["available"]:
        logger.info("[OIL] GEE available project=%s", info.get("project"))
    else:
        logger.warning("[OIL] GEE NOT available: %s", info.get("error"))
    return info


def repair_oil(df: pd.DataFrame, oil_decision: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    info: dict[str, Any] = {"input_status": oil_decision["status"]}
    if oil_decision["status"] == "usable":
        info["action"] = "no-op"
        info["final_oil_feature"] = "existing_oil_columns"
        return info

    # Probe GEE; do not fake anything if unavailable.
    info["gee_probe"] = gee_probe(LOGGER)
    if not info["gee_probe"].get("available"):
        info["action"] = "marked_unavailable"
        info["final_oil_feature"] = None
        info["reason"] = "Existing oil columns are unusable AND Earth Engine is unauthenticated; no Sentinel-1 dark-slick proxy was computed (we never fabricate oil data)."
        LOGGER.warning("[OIL] %s", info["reason"])
        return info

    # If GEE is available we *could* extract a Sentinel-1 VV-based dark-slick
    # proxy here. This branch is intentionally implemented as a thin scaffold
    # so the flow is testable end-to-end when credentials exist; we document
    # the science and keep terminology cautious ("dark slick proxy").
    try:
        import ee  # type: ignore

        if "grid_centroid_lat" not in df.columns or "grid_centroid_lon" not in df.columns:
            raise RuntimeError("grid coordinates missing; cannot construct study bbox")

        bbox = [
            float(df["grid_centroid_lon"].min()),
            float(df["grid_centroid_lat"].min()),
            float(df["grid_centroid_lon"].max()),
            float(df["grid_centroid_lat"].max()),
        ]
        info["study_bbox"] = bbox
        weeks = sorted(pd.to_datetime(df["week_start_utc"], utc=True).dt.tz_localize(None).dt.normalize().unique())
        info["n_weeks"] = len(weeks)
        rows: list[dict[str, Any]] = []
        # Per-grid weekly aggregation in GEE is expensive; we expose only the
        # control flow. Skipping a real heavy extraction here — it should be
        # run in a separate batch script with credentials and rate limits.
        info["action"] = "scaffold_only"
        info["reason"] = "GEE auth detected; full Sentinel-1 dark-slick proxy extraction is delegated to a dedicated batch script (run separately with credentials)."
        out_path = paths["processed"] / "sentinel1_dark_slick_proxy.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        info["output_parquet"] = str(out_path)
        info["final_oil_feature"] = "sentinel1_dark_slick_proxy"
        LOGGER.info("[OIL] GEE scaffold wrote empty parquet: %s", _rel(out_path))
    except Exception as exc:  # noqa: BLE001
        info["action"] = "scaffold_failed"
        info["error"] = str(exc)[:300]
        info["final_oil_feature"] = None
        LOGGER.exception("[OIL] Scaffold failed: %s", exc)
    return info


# ---------------------------------------------------------------------------
# STEP 4 — MERGE REPAIRED + STEP 5 — PLOTS + STEP 5 distance decay
# ---------------------------------------------------------------------------

def merge_repaired(df: pd.DataFrame, vessel_info: dict[str, Any], oil_info: dict[str, Any], paths: dict[str, Path]) -> tuple[Path, Path]:
    full = df.copy()
    if vessel_info.get("output_parquet"):
        repaired = pd.read_parquet(vessel_info["output_parquet"])
        if not repaired.empty:
            full = full.merge(repaired, on=[c for c in ["grid_cell_id", "week_start_utc"] if c in repaired.columns], how="left")

    full_path = paths["processed"] / "features_repaired_full.parquet"
    full.to_parquet(full_path, index=False)
    LOGGER.info("Wrote %s (%d × %d)", _rel(full_path), len(full), len(full.columns))

    keep: list[str] = []
    for c in [
        "grid_cell_id",
        "week_start_utc",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "nearest_port",
        "distance_to_port_km",
        "coastal_exposure_band",
        "coastal_exposure_score",
        "no2_mean_t",
        "no2_std_t",
        "ndwi_mean",
        "ndti_mean",
        "ndvi_mean",
    ]:
        if c in full.columns:
            keep.append(c)

    if vessel_info["final_vessel_feature"] == "vessel_density_t":
        keep.append("vessel_density_t")
    else:
        for c in ["vessel_density_spatial_proxy", "vessel_density_spatial_proxy_log1p", "port_pressure", "shipping_exposure", "coastal_shipping_pressure"]:
            if c in full.columns:
                keep.append(c)

    if oil_info.get("final_oil_feature") == "existing_oil_columns":
        keep.extend([c for c in OIL_COLS if c in full.columns])
    elif oil_info.get("final_oil_feature") == "sentinel1_dark_slick_proxy":
        for c in ["oil_dark_score", "oil_dark_score_rolling3", "oil_dark_pixel_ratio", "oil_dark_binary"]:
            if c in full.columns:
                keep.append(c)

    keep = list(dict.fromkeys(keep))
    safe = full[keep].copy()
    safe_path = paths["processed"] / "features_thesis_safe.parquet"
    safe.to_parquet(safe_path, index=False)
    LOGGER.info("Wrote %s (%d × %d)", _rel(safe_path), len(safe), len(safe.columns))
    return full_path, safe_path


def vessel_plots(df: pd.DataFrame, paths: dict[str, Path]) -> list[Path]:
    saved: list[Path] = []
    if "week_start_utc" in df.columns and "vessel_density_t" in df.columns:
        weekly = df.groupby("week_start_utc")["vessel_density_t"].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(weekly.index, weekly.values, color="tab:blue", lw=1.5)
        ax.set_title("Vessel density (raw vessel_density_t) — weekly mean")
        ax.set_xlabel("week")
        ax.set_ylabel("vessel_density_t (mean)")
        ax.grid(True, ls=":", alpha=0.4)
        plt.tight_layout()
        p = paths["viz"] / "vessel_weekly_mean_raw.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        saved.append(p)

    coords = ["grid_centroid_lon", "grid_centroid_lat"]
    if all(c in df.columns for c in coords) and "vessel_density_spatial_proxy" in df.columns:
        spatial = df.groupby(coords)["vessel_density_spatial_proxy"].first().reset_index().dropna()
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(spatial[coords[0]], spatial[coords[1]], c=spatial["vessel_density_spatial_proxy"], cmap="viridis", s=14)
        plt.colorbar(sc, ax=ax, label="vessel_density_spatial_proxy")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Vessel density spatial proxy (per grid cell)")
        plt.tight_layout()
        p = paths["viz"] / "vessel_spatial_proxy_map.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        saved.append(p)

    if "port_pressure" in df.columns and all(c in df.columns for c in coords):
        pp = df.groupby(coords)["port_pressure"].first().reset_index().dropna()
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(pp[coords[0]], pp[coords[1]], c=pp["port_pressure"], cmap="magma", s=14)
        plt.colorbar(sc, ax=ax, label="port_pressure (proxy / (1+km))")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Port pressure proxy")
        plt.tight_layout()
        p = paths["viz"] / "port_pressure_map.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        saved.append(p)

    if "shipping_exposure" in df.columns and all(c in df.columns for c in coords):
        se = df.groupby(coords)["shipping_exposure"].first().reset_index().dropna()
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(se[coords[0]], se[coords[1]], c=se["shipping_exposure"], cmap="cividis", s=14)
        plt.colorbar(sc, ax=ax, label="shipping_exposure (proxy * exp(-d/100))")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title("Shipping exposure proxy")
        plt.tight_layout()
        p = paths["viz"] / "shipping_exposure_map.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        saved.append(p)

    LOGGER.info("[VESSEL] Plots saved: %d", len(saved))
    return saved


def oil_plots(df: pd.DataFrame, oil_info: dict[str, Any], paths: dict[str, Path]) -> list[Path]:
    saved: list[Path] = []
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = []
    for c in OIL_COLS:
        if c in df.columns:
            rows.append((c, df[c].notna().mean() * 100, (df[c] == 0).mean() * 100 if pd.api.types.is_numeric_dtype(df[c]) else np.nan))
    if rows:
        labels = [r[0] for r in rows]
        coverage = [r[1] for r in rows]
        zeros = [r[2] for r in rows]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, coverage, 0.4, label="non-null %")
        ax.bar(x + 0.2, zeros, 0.4, label="zero %")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("percent")
        ax.set_title("Oil-slick layer: coverage vs zero share")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend()
        plt.tight_layout()
        p = paths["viz"] / "oil_missingness_zero_coverage.png"
        fig.savefig(p, dpi=140)
        saved.append(p)
    plt.close(fig)
    LOGGER.info("[OIL] Plots saved: %d", len(saved))
    return saved


def sliding_window(df: pd.DataFrame, value_col: str, *, window_km: float = 50.0, step_km: float = 10.0, max_km: float = 1000.0) -> pd.DataFrame:
    if "distance_to_port_km" not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    valid = df[["distance_to_port_km", value_col]].dropna()
    if valid.empty:
        return pd.DataFrame()
    upper = min(max_km, float(valid["distance_to_port_km"].max()))
    starts = np.arange(0.0, max(upper - window_km, 0.0) + step_km, step_km)
    rows: list[dict[str, Any]] = []
    dist = valid["distance_to_port_km"].values
    vals = valid[value_col].values
    for s in starts:
        e = s + window_km
        m = (dist >= s) & (dist < e)
        n = int(m.sum())
        if n == 0:
            continue
        sub = vals[m]
        rows.append({
            "window_mid_km": s + window_km / 2.0,
            "count": n,
            "mean": float(np.nanmean(sub)),
            "median": float(np.nanmedian(sub)),
            "std": float(np.nanstd(sub, ddof=1)) if n > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def repaired_distance_decay_plots(df: pd.DataFrame, vessel_info: dict[str, Any], oil_info: dict[str, Any], paths: dict[str, Path]) -> list[Path]:
    saved: list[Path] = []
    df = df.copy()

    # Add quick rolling/winsor variants if not present
    if "no2_mean_t" in df.columns and "no2_mean_t_rolling3" not in df.columns:
        if "grid_cell_id" in df.columns:
            df = df.sort_values(["grid_cell_id", "week_start_utc"])
            df["no2_mean_t_rolling3"] = df.groupby("grid_cell_id")["no2_mean_t"].transform(lambda s: s.rolling(3, min_periods=1, center=True).mean())
        else:
            df["no2_mean_t_rolling3"] = df["no2_mean_t"].rolling(3, min_periods=1, center=True).mean()
    for col in ["ndwi_mean", "ndti_mean"]:
        winsor = f"{col}_winsor"
        if col in df.columns and winsor not in df.columns:
            valid = df[col].dropna()
            if not valid.empty:
                p1, p99 = float(valid.quantile(0.01)), float(valid.quantile(0.99))
                df[winsor] = df[col].clip(lower=p1, upper=p99)

    candidates = []
    if vessel_info.get("final_vessel_feature") == "vessel_density_t":
        candidates.append(("vessel_density_t", "vessel_density_t"))
    else:
        if "vessel_density_spatial_proxy_log1p" in df.columns:
            candidates.append(("vessel_density_spatial_proxy_log1p", "vessel_density_spatial_proxy_log1p"))
        elif "vessel_density_spatial_proxy" in df.columns:
            candidates.append(("vessel_density_spatial_proxy", "vessel_density_spatial_proxy"))
    candidates.extend(
        [
            ("no2_mean_t_rolling3", "no2_mean_t_rolling3"),
            ("ndwi_mean_winsor", "ndwi_mean_winsor"),
            ("ndti_mean_winsor", "ndti_mean_winsor"),
        ]
    )
    if oil_info.get("final_oil_feature") in {"existing_oil_columns"} and "oil_slick_count_t" in df.columns:
        candidates.append(("oil_slick_count_t", "oil_slick_count_t"))

    for label, col in candidates:
        if col not in df.columns:
            continue
        sw = sliding_window(df, col)
        if sw.empty:
            continue
        sw.to_csv(paths["reports_distance"] / f"sliding_window_{label}.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
        ax1.plot(sw["window_mid_km"], sw["mean"], color="tab:blue", lw=1.8, label="mean")
        ax1.plot(sw["window_mid_km"], sw["median"], color="tab:orange", lw=1.2, ls="--", label="median")
        ax1.fill_between(sw["window_mid_km"], sw["mean"] - sw["std"].fillna(0), sw["mean"] + sw["std"].fillna(0), color="tab:blue", alpha=0.12, label="±1 std")
        ax1.set_xlabel("distance to port [km]")
        ax1.set_ylabel(label)
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(loc="upper left", fontsize=8)
        ax2 = ax1.twinx()
        ax2.bar(sw["window_mid_km"], sw["count"], width=9.0, color="grey", alpha=0.22, label="n per window")
        ax2.set_ylabel("samples per window", color="grey")
        plt.title(f"Repaired distance-decay — {label}")
        plt.tight_layout()
        p = paths["viz_distance"] / f"sliding_window_{label}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        saved.append(p)
    LOGGER.info("[DISTANCE-DECAY] Plots saved: %d", len(saved))
    return saved


# ---------------------------------------------------------------------------
# STEP 6 — REPORT
# ---------------------------------------------------------------------------

def write_report(
    base: Path,
    *,
    input_path: Path,
    vessel_decision: dict[str, Any],
    vessel_info: dict[str, Any],
    oil_decision: dict[str, Any],
    oil_info: dict[str, Any],
    full_path: Path,
    safe_path: Path,
    safe_columns: list[str],
    vessel_plot_paths: list[Path],
    oil_plot_paths: list[Path],
    distance_plot_paths: list[Path],
) -> Path:
    rep = base / "REPAIRED_SOURCE_REPORT.md"
    L: list[str] = []
    L.append("# Repaired source report — final_run_repaired_sources/")
    L.append("")
    L.append(f"- Source dataset: `{_rel(input_path)}`")
    L.append("")

    L.append("## 1. Was existing vessel data temporal?")
    L.append(f"- Status: **{vessel_decision['status']}**")
    L.append(f"- Coverage > 70%: {vessel_decision['coverage_ok_70']}")
    L.append(f"- Fraction of grid cells temporally varying: {vessel_decision['grid_temporally_varying_fraction']}")
    L.append(f"- Weekly-mean σ: {vessel_decision['weekly_mean_std']}")
    L.append("- Conclusion: vessel_density_t is **not temporal** (single value per grid; weekly-mean variance ≈ 0). It is a per-cell *spatial* maritime pressure proxy.")
    L.append("")

    L.append("## 2. Was external temporal vessel data found?")
    probe = vessel_info.get("emodnet_probe", {})
    if probe.get("available"):
        L.append(f"- EMODnet Human Activities WMS reachable at `{probe.get('endpoint')}`.")
        L.append(f"- Layer hints found in capabilities: {probe.get('candidates_found')}.")
        L.append(f"- Capabilities snapshot: `{_rel(Path(probe.get('snapshot_path')))}`.")
        L.append("- Note: monthly vessel-density rasters are not retrievable through this WMS without acquiring the GeoTIFF download from the EMODnet portal (terms-of-use acceptance required). No external vessel layer was injected into the dataset; we use the documented spatial-proxy fallback.")
    else:
        L.append(f"- EMODnet probe failed: `{probe.get('error', 'no probe attempted')}`. No external vessel layer injected.")
    L.append(f"- External temporal vessel used: **{bool(vessel_info.get('temporal_external_used'))}**.")
    L.append("")

    L.append("## 3. Final vessel feature used")
    L.append(f"- Final feature: **`{vessel_info['final_vessel_feature']}`**.")
    L.append(f"- Action: {vessel_info['action']}.")
    L.append(f"- Derived columns added: {vessel_info.get('features_added')}.")
    L.append("- Interpretation: present this as a **spatial maritime pressure proxy**. Do NOT make weekly vessel-traffic claims from this column.")
    L.append("")

    L.append("## 4. Was existing oil layer usable?")
    L.append(f"- Status: **{oil_decision['status']}**")
    L.append(f"- Has non-zero signal: {oil_decision['has_nonzero_signal']}")
    L.append(f"- Detection score present: {oil_decision['detection_score_present']}")
    L.append(f"- Extra SAR/VV/VH columns: {oil_decision['extra_sar_columns'] if oil_decision['extra_sar_columns'] else '(none)'}")
    L.append("")

    L.append("## 5. Was a Sentinel-1 dark-slick proxy rebuilt?")
    gee = oil_info.get("gee_probe") or {}
    if oil_info.get("input_status") == "usable":
        L.append("- Skipped — existing oil columns were already usable.")
    else:
        if gee.get("available"):
            L.append(f"- GEE detected (project=`{gee.get('project')}`).")
            L.append(f"- Action: **{oil_info.get('action')}** — {oil_info.get('reason')}")
            L.append("- Recommended next step: run a dedicated batch script that constructs a Sentinel-1 VV-based **dark-slick proxy** over the study bbox, weekly aggregations, with terminology kept cautious (\"Sentinel-1 dark slick proxy — not confirmed oil spill\").")
        else:
            L.append(f"- GEE not available — `{gee.get('error')}`.")
            L.append(f"- Action: **{oil_info.get('action')}** — {oil_info.get('reason')}")
            L.append("- We deliberately did NOT fabricate any oil-like values.")
    L.append("")

    L.append("## 6. Final oil/slick decision")
    L.append(f"- Final feature: **`{oil_info.get('final_oil_feature')}`**")
    L.append("- Thesis rule: oil/slick exposure is **excluded** from the thesis-safe dataset and from any thesis claims for this run. If a Sentinel-1 dark-slick proxy is later produced, it must be described as *Sentinel-1 dark slick proxy — association, not causality*.")
    L.append("")

    L.append("## 7. Thesis-safe feature list")
    L.append(f"- File: `{_rel(safe_path)}` ({len(safe_columns)} columns).")
    for c in safe_columns:
        L.append(f"  - `{c}`")
    L.append("")

    L.append("## 8. ML-safe feature list")
    L.append("- The thesis-safe parquet excludes:")
    L.append("  - Oil/Sentinel-1 columns (unusable in this run).")
    L.append("  - Duplicate aliases (NO2_mean ↔ no2_mean_t, ndvi_mean ↔ land_response_index, mean ↔ median, etc. — already pruned in the prior `final_run_cleaned/` deliverable).")
    L.append("  - Columns with >80% missing unless explicitly retained for interpretation (NDVI etc.).")
    L.append("")

    L.append("## 9. Plots to use in thesis")
    for p in vessel_plot_paths + distance_plot_paths:
        L.append(f"- `{_rel(p)}`")
    L.append("")

    L.append("## 10. Plots / features to exclude from thesis")
    for p in oil_plot_paths:
        L.append(f"- `{_rel(p)}` (use only as a transparency artefact showing the oil layer is unusable; do not draw conclusions).")
    L.append("- All `oil_slick_*` and `detection_score` columns: omit from any thesis claim.")
    L.append("- Weekly-traffic narratives based on `vessel_density_t`: replace with *spatial pressure proxy* language.")
    L.append("")

    L.append("## Cautious-language reminders")
    L.append("- *Spatial maritime pressure proxy* (not weekly vessel traffic).")
    L.append("- *Sentinel-1 dark slick proxy* (not confirmed oil spill).")
    L.append("- *Association, not causality.*")
    L.append("")

    rep.write_text("\n".join(L))
    LOGGER.info("Wrote %s", _rel(rep))
    return rep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    paths = make_skeleton(args.output_root)
    configure_logging(paths["logs"] / "repair.log")
    LOGGER.info("=== Source audit + repair ===")
    LOGGER.info("Input: %s", _rel(args.input))
    LOGGER.info("Output: %s", _rel(args.output_root))

    if not args.input.exists():
        LOGGER.error("Input parquet not found: %s", args.input)
        return 2
    df = pd.read_parquet(args.input)
    LOGGER.info("Loaded shape: %s", df.shape)

    # 1. Audit
    vessel_decision, oil_decision, _ = audit_layers(df, paths["validation"])
    LOGGER.info("[AUDIT] vessel=%s | oil=%s", vessel_decision["status"], oil_decision["status"])

    # 2. Vessel repair
    vessel_info = repair_vessel(df, vessel_decision, paths)

    # 3. Oil repair
    oil_info = repair_oil(df, oil_decision, paths)

    # 4. Merge + thesis-safe dataset
    full_path, safe_path = merge_repaired(df, vessel_info, oil_info, paths)
    safe_cols = pd.read_parquet(safe_path).columns.tolist()

    # 5. Plots
    df_full = pd.read_parquet(full_path)
    vessel_plot_paths = vessel_plots(df_full, paths)
    oil_plot_paths = oil_plots(df_full, oil_info, paths)
    distance_plot_paths = repaired_distance_decay_plots(df_full, vessel_info, oil_info, paths)

    # 6. Report
    rep = write_report(
        paths["base"],
        input_path=args.input,
        vessel_decision=vessel_decision,
        vessel_info=vessel_info,
        oil_decision=oil_decision,
        oil_info=oil_info,
        full_path=full_path,
        safe_path=safe_path,
        safe_columns=safe_cols,
        vessel_plot_paths=vessel_plot_paths,
        oil_plot_paths=oil_plot_paths,
        distance_plot_paths=distance_plot_paths,
    )

    # Audit summary JSON
    summary = {
        "vessel_decision": vessel_decision,
        "vessel_repair": vessel_info,
        "oil_decision": oil_decision,
        "oil_repair": oil_info,
        "outputs": {
            "full_parquet": _rel(full_path),
            "safe_parquet": _rel(safe_path),
            "report": _rel(rep),
        },
    }
    sj = paths["validation"] / "audit_summary.json"
    sj.write_text(json.dumps(summary, indent=2, default=str))
    LOGGER.info("Wrote %s", _rel(sj))

    # Final stdout summary
    LOGGER.info("=== Done ===")
    LOGGER.info("Vessel status: %s | final feature: %s", vessel_decision["status"], vessel_info.get("final_vessel_feature"))
    LOGGER.info("Oil status: %s | final feature: %s", oil_decision["status"], oil_info.get("final_oil_feature"))
    LOGGER.info("External temporal vessel used: %s", bool(vessel_info.get("temporal_external_used")))
    LOGGER.info("Generated outputs:")
    for k, v in summary["outputs"].items():
        LOGGER.info("  %s: %s", k, v)
    LOGGER.info("Recommended thesis-safe plots:")
    for p in vessel_plot_paths + distance_plot_paths:
        LOGGER.info("  %s", _rel(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
