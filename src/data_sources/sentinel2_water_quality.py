"""
Sentinel-2 Surface Reflectance (GEE: COPERNICUS/S2_SR_HARMONIZED) → grid × week parquet.

Observational only: weekly composites via GEE; spatial statistics via reduceRegions
(mean / stdDev / median) over a square buffer around each grid centroid.

Computed bands per weekly composite (all vectorized in GEE, no Python-side loops):
    NDWI = (B3 - B8) / (B3 + B8)      (McFeeters normalized difference water index)
    NDTI = (B4 - B3) / (B4 + B3)      (normalized difference turbidity index)
    NDCI = (B5 - B4) / (B5 + B4)      (normalized difference chlorophyll index)
    FAI  = B8 - (B4 + (B11 - B4) * ((865 - 665) / (1610 - 665)))  (floating algae index)
    B11  = raw SWIR reflectance (scaled to [0, 1] by dividing S2 SR by 10000)

Cloud masking: Scene Classification Layer (SCL) — pixels in classes
    {3 cloud_shadow, 8 cloud_medium, 9 cloud_high, 10 thin_cirrus}
are dropped. CLOUDY_PIXEL_PERCENTAGE is also used as a scene-level pre-filter.

Output columns (strict naming, one row per grid_cell_id × week_start_utc):
    ndwi_mean, ndwi_median, ndwi_std
    ndti_mean, ndti_median, ndti_std
    ndci_mean, ndci_median, ndci_std
    fai_mean,  fai_median,  fai_std
    b11_mean,  b11_median,  b11_std

Requires: `earthengine authenticate` (+ GOOGLE_CLOUD_PROJECT / EE_PROJECT if needed).
Output: data/aux/sentinel2_water_quality.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data_sources.gee_grid_utils import (  # noqa: E402
    DEFAULT_BUFFER_DEG_NO2,
    build_grid_feature_collection,
    initialize_earth_engine,
    iter_week_utc_bounds,
    load_modeling_grids_and_weeks,
    study_bbox,
    week_filter_dates_ee,
)
from utils.ee_timeout import safe_ee_call  # noqa: E402

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
REDUCE_SCALE_M = 60
EE_TIMEOUT_SECONDS = 180
EE_MAX_RETRIES = 2
CLOUDY_PIXEL_PERCENTAGE_MAX = 60
IMAGE_COLLECTION_LIMIT = 500
SCL_BAD_CLASSES = (3, 8, 9, 10)
S2_REFLECTANCE_SCALE = 10000.0

# Band keys used in the output schema.
BAND_KEYS: tuple[str, ...] = ("ndwi", "ndti", "ndci", "fai", "b11")
STAT_KEYS: tuple[str, ...] = ("mean", "median", "std")

LOGGER = logging.getLogger("sentinel2_water_quality")


def _import_ee():
    try:
        import ee  # noqa: WPS433
    except ImportError as e:  # pragma: no cover - exercised at runtime only
        raise SystemExit(
            "Install earthengine-api (pip install earthengine-api) and run: earthengine authenticate"
        ) from e
    return ee


def _empty_row(grid_id: Any, week_ts: pd.Timestamp) -> dict[str, Any]:
    row: dict[str, Any] = {
        "grid_cell_id": grid_id,
        "week_start_utc": week_ts,
    }
    for band in BAND_KEYS:
        for stat in STAT_KEYS:
            row[f"{band}_{stat}"] = float("nan")
    return row


def _mask_and_scale(ee, image):
    """Apply SCL cloud mask and scale reflectance to [0, 1]. Returns image with 5 analytic bands."""
    scl = image.select("SCL")
    mask = scl.neq(SCL_BAD_CLASSES[0])
    for cls in SCL_BAD_CLASSES[1:]:
        mask = mask.And(scl.neq(cls))

    scaled = image.select(["B3", "B4", "B5", "B8", "B11"]).divide(S2_REFLECTANCE_SCALE).updateMask(mask)

    b3 = scaled.select("B3")
    b4 = scaled.select("B4")
    b5 = scaled.select("B5")
    b8 = scaled.select("B8")
    b11 = scaled.select("B11")

    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename("ndwi")
    ndti = b4.subtract(b3).divide(b4.add(b3)).rename("ndti")
    ndci = b5.subtract(b4).divide(b5.add(b4)).rename("ndci")
    fai_factor = (865.0 - 665.0) / (1610.0 - 665.0)
    fai = b8.subtract(b4.add(b11.subtract(b4).multiply(fai_factor))).rename("fai")
    b11_out = b11.rename("b11")

    composite = ndwi.addBands(ndti).addBands(ndci).addBands(fai).addBands(b11_out)
    return composite.copyProperties(image, ["system:time_start"])


def _build_weekly_collection(ee, bbox, start_ee, end_ee):
    """Bounded, cloud-filtered S2 collection for a given week window."""
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(bbox)
        .filterDate(start_ee, end_ee)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE_MAX))
        .limit(IMAGE_COLLECTION_LIMIT)
    )


def _extract_stats(props: dict) -> dict[str, float]:
    """Map the flat reduceRegions property keys to our strict output naming."""
    out: dict[str, float] = {}
    for band in BAND_KEYS:
        mean_v = props.get(f"{band}_mean")
        # EE reduceRegions keys:  {band}_mean, {band}_stdDev, {band}_median
        std_v = props.get(f"{band}_stdDev")
        if std_v is None:
            std_v = props.get(f"{band}_stddev")
        median_v = props.get(f"{band}_median")
        out[f"{band}_mean"] = float(mean_v) if mean_v is not None else float("nan")
        out[f"{band}_std"] = float(std_v) if std_v is not None else float("nan")
        out[f"{band}_median"] = float(median_v) if median_v is not None else float("nan")
    return out


def extract_s2_weekly(
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
    *,
    buffer_deg: float,
    ee,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the Sentinel-2 weekly × grid extraction and return (dataframe, validation_meta)."""
    logger = logger or LOGGER
    logger.info("[S2] STEP 1: sentinel-2 water-quality module entered")

    bbox = study_bbox(grids, ee)
    fc = build_grid_feature_collection(grids, buffer_deg, ee)

    reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
    )

    logger.info("[S2] STEP 2: building EE ImageCollection (S2_SR_HARMONIZED)")
    logger.info(
        "[S2] STEP 3: reducer = mean+stdDev+median, cloud_max=%d, scale=%d m, limit=%d",
        CLOUDY_PIXEL_PERCENTAGE_MAX,
        REDUCE_SCALE_M,
        IMAGE_COLLECTION_LIMIT,
    )
    logger.info("[S2] STEP 4: %d grids × %d weeks, entering loop", len(grids), len(weeks))

    rows: list[dict[str, Any]] = []
    total_weeks = len(weeks)

    for week_idx, (wk, start_dt, end_dt) in enumerate(iter_week_utc_bounds(weeks), start=1):
        start_ee, end_ee = week_filter_dates_ee(ee, start_dt, end_dt)
        ic_raw = _build_weekly_collection(ee, bbox, start_ee, end_ee)

        n_img = 0
        try:
            n_img_raw = safe_ee_call(
                lambda ic=ic_raw: ic.size().getInfo(),
                timeout_seconds=EE_TIMEOUT_SECONDS,
                max_retries=EE_MAX_RETRIES,
                logger=logger,
                context=f"s2.scene_count week={pd.Timestamp(wk).date()}",
            )
            n_img = int(n_img_raw) if n_img_raw is not None else 0
        except Exception:  # noqa: BLE001
            n_img = 0

        if week_idx % 5 == 0 or week_idx == 1:
            logger.info(
                "[S2] HEARTBEAT: week %d/%d (%s) scenes=%d",
                week_idx,
                total_weeks,
                pd.Timestamp(wk).date(),
                n_img,
            )

        if n_img == 0:
            for _, r in grids.iterrows():
                rows.append(_empty_row(r["grid_cell_id"], pd.Timestamp(wk)))
            continue

        ic_analytic = ic_raw.map(lambda img, _ee=ee: _mask_and_scale(_ee, img))
        composite = ic_analytic.mean()

        reduced = composite.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=REDUCE_SCALE_M,
            tileScale=4,
        )
        reduced_info = safe_ee_call(
            lambda rr=reduced: rr.getInfo(),
            timeout_seconds=EE_TIMEOUT_SECONDS,
            max_retries=EE_MAX_RETRIES,
            logger=logger,
            context=f"s2.reduceRegions week={pd.Timestamp(wk).date()}",
        )

        if reduced_info is None:
            logger.warning("[S2][SKIP] week=%s reason=EE_TIMEOUT_OR_ERROR", pd.Timestamp(wk).date())
            for _, r in grids.iterrows():
                rows.append(_empty_row(r["grid_cell_id"], pd.Timestamp(wk)))
            continue

        feats = reduced_info.get("features", []) or []
        by_id: dict[str, dict[str, float]] = {}
        for f in feats:
            props = f.get("properties", {}) or {}
            gid = props.get("grid_cell_id")
            if gid is None:
                continue
            by_id[str(gid)] = _extract_stats(props)

        for _, r in grids.iterrows():
            gid = str(r["grid_cell_id"])
            stats = by_id.get(gid)
            if stats is None:
                rows.append(_empty_row(r["grid_cell_id"], pd.Timestamp(wk)))
            else:
                row = {"grid_cell_id": r["grid_cell_id"], "week_start_utc": pd.Timestamp(wk)}
                row.update(stats)
                rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True)
        # Enforce strict column order.
        schema = ["grid_cell_id", "week_start_utc"] + [f"{b}_{s}" for b in BAND_KEYS for s in STAT_KEYS]
        for col in schema:
            if col not in out.columns:
                out[col] = float("nan")
        out = out[schema]

    validation = compute_validation(out, grids, weeks)
    logger.info("[S2] Extraction complete: rows=%d coverage=%.2f%%", len(out), validation.get("coverage_percent", 0.0))
    return out, validation


def compute_validation(
    extracted: pd.DataFrame,
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
) -> dict[str, Any]:
    n_grids = int(len(grids))
    n_weeks = int(len([w for w in weeks if not pd.isna(w)]))
    expected = n_grids * n_weeks
    n_obs = int(len(extracted))

    coverage_percent = 0.0
    per_band_valid: dict[str, int] = {}
    if not extracted.empty and expected:
        for band in BAND_KEYS:
            col = f"{band}_mean"
            if col in extracted.columns:
                per_band_valid[band] = int(extracted[col].notna().sum())
        # Primary coverage is driven by NDWI since it relies only on B3/B8 (10m, always present).
        primary = per_band_valid.get("ndwi", 0)
        coverage_percent = round(100.0 * primary / expected, 4)

    return {
        "number_of_grid_week_observations": n_obs,
        "expected_grid_week_pairs": int(expected),
        "coverage_percent": coverage_percent,
        "per_band_valid_counts": per_band_valid,
        "n_grids": n_grids,
        "n_weeks": n_weeks,
        "cloud_cover_threshold": CLOUDY_PIXEL_PERCENTAGE_MAX,
        "reduce_scale_m": REDUCE_SCALE_M,
        "collection_limit_per_week": IMAGE_COLLECTION_LIMIT,
        "data_policy": "GEE reducers only; no interpolation or synthetic water-quality filling",
    }


def run_pipeline(
    input_parquet: Path,
    output_parquet: Path,
    validation_json: Path | None,
    buffer_deg: float,
    *,
    quick_test: bool = False,
    debug_log_json: Path | None = None,
) -> dict[str, Any]:
    ee = _import_ee()
    try:
        initialize_earth_engine(ee)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Earth Engine init failed. Run: earthengine authenticate\n"
            "Set GOOGLE_CLOUD_PROJECT or EE_PROJECT if required.\n"
            f"Detail: {e}"
        ) from e

    grids, weeks = load_modeling_grids_and_weeks(input_parquet)
    if quick_test and not grids.empty:
        grids = grids.head(20).copy()
        weeks = list(weeks)[:6]
        LOGGER.info("[S2][QUICK] Limiting to grids=%d weeks=%d", len(grids), len(weeks))

    df, validation = extract_s2_weekly(grids, weeks, buffer_deg=buffer_deg, ee=ee, logger=LOGGER)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    if validation_json:
        validation_json.parent.mkdir(parents=True, exist_ok=True)
        with validation_json.open("w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2, default=str)

    if debug_log_json:
        debug_log_json.parent.mkdir(parents=True, exist_ok=True)
        with debug_log_json.open("w", encoding="utf-8") as f:
            json.dump(
                {"rows": int(len(df)), "columns": list(df.columns), "validation": validation},
                f,
                indent=2,
                default=str,
            )

    print(json.dumps({"sentinel2_water_quality_validation": validation}, indent=2, default=str))
    print(f"Wrote {output_parquet.resolve()}")
    return validation


def run_extraction(config) -> pd.DataFrame:
    """Standardized extraction wrapper for orchestrators."""
    root = Path(__file__).resolve().parents[2]
    if isinstance(config, dict):
        input_parquet = Path(config.get("input", root / "data" / "modeling_dataset.parquet"))
        output_parquet = Path(
            config.get("output", root / "data" / "aux" / "sentinel2_water_quality.parquet")
        )
        validation_json = Path(
            config.get(
                "validation_json",
                root / "data" / "aux" / "sentinel2_water_quality_validation.json",
            )
        )
        buffer_deg = float(config.get("buffer_deg", DEFAULT_BUFFER_DEG_NO2))
        quick_test = bool(config.get("quick_test", False))
    else:
        input_parquet = root / "data" / "modeling_dataset.parquet"
        output_parquet = root / "data" / "aux" / "sentinel2_water_quality.parquet"
        validation_json = root / "data" / "aux" / "sentinel2_water_quality_validation.json"
        buffer_deg = DEFAULT_BUFFER_DEG_NO2
        quick_test = False

    run_pipeline(
        input_parquet,
        output_parquet,
        validation_json,
        buffer_deg,
        quick_test=quick_test,
    )
    return pd.read_parquet(output_parquet)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Sentinel-2 water-quality indices (NDWI, NDTI, NDCI, FAI, B11) — grid-week extraction via Google Earth Engine"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "modeling_dataset.parquet",
        help="Modeling parquet (grid_cell_id, week_start_utc, centroids)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "aux" / "sentinel2_water_quality.parquet",
    )
    p.add_argument(
        "--validation-json",
        type=Path,
        default=root / "data" / "aux" / "sentinel2_water_quality_validation.json",
    )
    p.add_argument(
        "--buffer-deg",
        type=float,
        default=DEFAULT_BUFFER_DEG_NO2,
        help="Half-width (deg) of square buffer around centroid (default 0.1°)",
    )
    p.add_argument("--quick-test", action="store_true", help="Run on a small grids×weeks subset")
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else root / args.input
    outp = args.output if args.output.is_absolute() else root / args.output
    vj = args.validation_json if args.validation_json.is_absolute() else root / args.validation_json
    run_pipeline(inp, outp, vj, args.buffer_deg, quick_test=args.quick_test)


if __name__ == "__main__":
    main()
