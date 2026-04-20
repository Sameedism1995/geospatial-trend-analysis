"""
Sentinel-2 Surface Reflectance (GEE: COPERNICUS/S2_SR_HARMONIZED) → grid × week parquet of
LAND-ONLY NDVI statistics.

LAND IMPACT EXTENSION LAYER — additive module (does not modify or replace the existing
`data_sources.sentinel2_water_quality` module).

Observational only: weekly composites via GEE; spatial statistics via reduceRegions
(mean / stdDev / median) over a square buffer around each grid centroid.

Computed per weekly composite (all vectorized in GEE):
    NDVI = (B8 - B4) / (B8 + B4)      (normalized difference vegetation index)

Masking applied BEFORE NDVI is reduced:
    * Cloud mask via SCL classes {3 cloud_shadow, 8 cloud_medium, 9 cloud_high, 10 thin_cirrus}
      plus CLOUDY_PIXEL_PERCENTAGE scene-level pre-filter.
    * Water mask via NDWI = (B3 - B8) / (B3 + B8).
      Pixels with NDWI > NDWI_WATER_THRESHOLD are dropped — standard McFeeters water detection.
      This guarantees NDVI is computed on LAND pixels only (Step-7 validation rule).

Output columns (strict naming, one row per grid_cell_id × week_start_utc):
    ndvi_mean, ndvi_median, ndvi_std

Requires: `earthengine authenticate` (+ GOOGLE_CLOUD_PROJECT / EE_PROJECT if needed).
Output: data/aux/sentinel2_land_metrics.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[2]
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
# Land NDVI is aggregated to grid-level; 200 m is ample and keeps EE compute
# budgets realistic. (Water-quality module stays at 60 m — left untouched.)
REDUCE_SCALE_M = 200
# Larger EE budget per call; NDVI-over-land has many more valid pixels than the
# water-quality reducers and needs extra headroom.
EE_TIMEOUT_SECONDS = 300
EE_MAX_RETRIES = 2
CLOUDY_PIXEL_PERCENTAGE_MAX = 60
# Weekly composites rarely benefit from >150 scenes; smaller cap cuts compute.
IMAGE_COLLECTION_LIMIT = 150
# Halve internal EE tile size vs default (4); reduces memory / per-tile failures.
REDUCE_TILE_SCALE = 8
SCL_BAD_CLASSES = (3, 8, 9, 10)
S2_REFLECTANCE_SCALE = 10000.0

# McFeeters NDWI threshold above which pixels are considered open water and
# therefore DROPPED from the land-NDVI reducer. 0.0 is the standard value; we
# keep it slightly inclusive of shallow/bright-sediment coasts. This is the
# canonical water-exclusion rule enforcing the "no water in NDVI" validation
# requirement (Step 7).
NDWI_WATER_THRESHOLD = 0.0

STAT_KEYS: tuple[str, ...] = ("mean", "median", "std")

# Make this module's logger a child of the pipeline logger so [S2-LAND] logs
# are visible when run inside run_full_pipeline.py. Standalone usage still works
# (logging just falls back to the root logger).
LOGGER = logging.getLogger("full_pipeline.s2_land")


def _import_ee():
    try:
        import ee  # noqa: WPS433
    except ImportError as e:  # pragma: no cover - runtime-only
        raise SystemExit(
            "Install earthengine-api (pip install earthengine-api) and run: earthengine authenticate"
        ) from e
    return ee


def _empty_row(grid_id: Any, week_ts: pd.Timestamp) -> dict[str, Any]:
    row: dict[str, Any] = {"grid_cell_id": grid_id, "week_start_utc": week_ts}
    for stat in STAT_KEYS:
        row[f"ndvi_{stat}"] = float("nan")
    return row


def _mask_and_compute_ndvi(ee, image):
    """Per-image: apply cloud+water mask, then compute NDVI. Returns 1-band image ('ndvi')."""
    scl = image.select("SCL")
    cloud_mask = scl.neq(SCL_BAD_CLASSES[0])
    for cls in SCL_BAD_CLASSES[1:]:
        cloud_mask = cloud_mask.And(scl.neq(cls))

    scaled = image.select(["B3", "B4", "B8"]).divide(S2_REFLECTANCE_SCALE)
    b3 = scaled.select("B3")
    b4 = scaled.select("B4")
    b8 = scaled.select("B8")

    ndwi = b3.subtract(b8).divide(b3.add(b8))
    land_mask = ndwi.lte(NDWI_WATER_THRESHOLD)

    combined_mask = cloud_mask.And(land_mask)
    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("ndvi").updateMask(combined_mask)
    return ndvi.copyProperties(image, ["system:time_start"])


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
    # EE reduceRegions on a SINGLE-band image returns properties named by the
    # reducer only ('mean', 'stdDev', 'median'). On multi-band images (like the
    # water-quality module) it prefixes with the band name. Accept both forms so
    # this module is robust to future EE changes.
    mean_v = props.get("mean", props.get("ndvi_mean"))
    std_v = props.get("stdDev", props.get("stddev", props.get("ndvi_stdDev", props.get("ndvi_stddev"))))
    median_v = props.get("median", props.get("ndvi_median"))
    return {
        "ndvi_mean": float(mean_v) if mean_v is not None else float("nan"),
        "ndvi_std": float(std_v) if std_v is not None else float("nan"),
        "ndvi_median": float(median_v) if median_v is not None else float("nan"),
    }


def extract_s2_land_weekly(
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
    *,
    buffer_deg: float,
    ee,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the Sentinel-2 weekly × grid LAND-NDVI extraction."""
    logger = logger or LOGGER
    logger.info("[S2-LAND] STEP 1: sentinel-2 land-metrics module entered")

    bbox = study_bbox(grids, ee)
    fc = build_grid_feature_collection(grids, buffer_deg, ee)

    reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
    )

    logger.info("[S2-LAND] STEP 2: building EE ImageCollection (S2_SR_HARMONIZED, NDVI on land)")
    logger.info(
        "[S2-LAND] STEP 3: reducer=mean+std+median; cloud_max=%d; scale=%dm; NDWI<=%.2f = land",
        CLOUDY_PIXEL_PERCENTAGE_MAX,
        REDUCE_SCALE_M,
        NDWI_WATER_THRESHOLD,
    )
    logger.info("[S2-LAND] STEP 4: %d grids × %d weeks, entering loop", len(grids), len(weeks))

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
                context=f"s2land.scene_count week={pd.Timestamp(wk).date()}",
            )
            n_img = int(n_img_raw) if n_img_raw is not None else 0
        except Exception:  # noqa: BLE001
            n_img = 0

        if week_idx % 5 == 0 or week_idx == 1:
            logger.info(
                "[S2-LAND] HEARTBEAT: week %d/%d (%s) scenes=%d",
                week_idx,
                total_weeks,
                pd.Timestamp(wk).date(),
                n_img,
            )

        if n_img == 0:
            for _, r in grids.iterrows():
                rows.append(_empty_row(r["grid_cell_id"], pd.Timestamp(wk)))
            continue

        ic_ndvi = ic_raw.map(lambda img, _ee=ee: _mask_and_compute_ndvi(_ee, img))
        composite = ic_ndvi.mean()

        reduced = composite.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=REDUCE_SCALE_M,
            tileScale=REDUCE_TILE_SCALE,
        )
        reduced_info = safe_ee_call(
            lambda rr=reduced: rr.getInfo(),
            timeout_seconds=EE_TIMEOUT_SECONDS,
            max_retries=EE_MAX_RETRIES,
            logger=logger,
            context=f"s2land.reduceRegions week={pd.Timestamp(wk).date()}",
        )

        if reduced_info is None:
            logger.warning("[S2-LAND][SKIP] week=%s reason=EE_TIMEOUT_OR_ERROR", pd.Timestamp(wk).date())
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
        schema = ["grid_cell_id", "week_start_utc", "ndvi_mean", "ndvi_median", "ndvi_std"]
        for col in schema:
            if col not in out.columns:
                out[col] = float("nan")
        out = out[schema]

    validation = compute_validation(out, grids, weeks)
    logger.info(
        "[S2-LAND] Extraction complete: rows=%d coverage=%.2f%%",
        len(out),
        validation.get("coverage_percent", 0.0),
    )
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
    valid_ndvi = 0
    if not extracted.empty and expected and "ndvi_mean" in extracted.columns:
        valid_ndvi = int(extracted["ndvi_mean"].notna().sum())
        coverage_percent = round(100.0 * valid_ndvi / expected, 4)

    return {
        "number_of_grid_week_observations": n_obs,
        "expected_grid_week_pairs": int(expected),
        "coverage_percent": coverage_percent,
        "valid_ndvi_count": valid_ndvi,
        "n_grids": n_grids,
        "n_weeks": n_weeks,
        "cloud_cover_threshold": CLOUDY_PIXEL_PERCENTAGE_MAX,
        "reduce_scale_m": REDUCE_SCALE_M,
        "reduce_tile_scale": REDUCE_TILE_SCALE,
        "ee_timeout_seconds": EE_TIMEOUT_SECONDS,
        "collection_limit_per_week": IMAGE_COLLECTION_LIMIT,
        "ndwi_water_threshold": NDWI_WATER_THRESHOLD,
        "water_exclusion_rule": "pixels with NDWI > threshold excluded prior to NDVI reducer",
        "data_policy": "GEE reducers only; no interpolation or land/water gap-filling",
    }


def run_pipeline(
    input_parquet: Path,
    output_parquet: Path,
    validation_json: Path | None,
    buffer_deg: float,
    *,
    quick_test: bool = False,
    debug_log_json: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the S2 NDVI-on-land extraction.

    When ``overwrite=True`` any existing ``output_parquet`` or ``validation_json``
    is removed up-front so the run is guaranteed to be fresh. This is the
    thesis-reproducibility path and is invoked automatically by the pipeline
    bootstrap whenever ``--land-impact`` is passed.
    """
    if overwrite:
        removed = 0
        for target in (output_parquet, validation_json):
            if target is None:
                continue
            try:
                if target.exists():
                    target.unlink()
                    removed += 1
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[S2-LAND] overwrite could not remove %s: %s", target, exc)
        if removed:
            LOGGER.info("NDVI aux cache cleared — forcing fresh extraction (module-level, %d file(s))", removed)
        else:
            LOGGER.info("NDVI aux overwrite requested — no existing artifact found; extracting fresh")

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
        LOGGER.info("[S2-LAND][QUICK] Limiting to grids=%d weeks=%d", len(grids), len(weeks))

    df, validation = extract_s2_land_weekly(grids, weeks, buffer_deg=buffer_deg, ee=ee, logger=LOGGER)

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

    print(json.dumps({"sentinel2_land_metrics_validation": validation}, indent=2, default=str))
    print(f"Wrote {output_parquet.resolve()}")
    return validation


def run_extraction(config) -> pd.DataFrame:
    """Standardized extraction wrapper for orchestrators.

    Accepts an ``overwrite`` flag in ``config`` (default True for land NDVI —
    thesis reproducibility: this module never consumes a stale cache).
    """
    root = Path(__file__).resolve().parents[3]
    if isinstance(config, dict):
        input_parquet = Path(config.get("input", root / "data" / "modeling_dataset.parquet"))
        output_parquet = Path(
            config.get("output", root / "data" / "aux" / "sentinel2_land_metrics.parquet")
        )
        validation_json = Path(
            config.get(
                "validation_json",
                root / "data" / "aux" / "sentinel2_land_metrics_validation.json",
            )
        )
        buffer_deg = float(config.get("buffer_deg", DEFAULT_BUFFER_DEG_NO2))
        quick_test = bool(config.get("quick_test", False))
        overwrite = bool(config.get("overwrite", True))
    else:
        input_parquet = root / "data" / "modeling_dataset.parquet"
        output_parquet = root / "data" / "aux" / "sentinel2_land_metrics.parquet"
        validation_json = root / "data" / "aux" / "sentinel2_land_metrics_validation.json"
        buffer_deg = DEFAULT_BUFFER_DEG_NO2
        quick_test = False
        overwrite = True

    run_pipeline(
        input_parquet,
        output_parquet,
        validation_json,
        buffer_deg,
        quick_test=quick_test,
        overwrite=overwrite,
    )
    return pd.read_parquet(output_parquet)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(
        description="Sentinel-2 LAND NDVI — grid-week extraction via Google Earth Engine (Land Impact Extension Layer)"
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
        default=root / "data" / "aux" / "sentinel2_land_metrics.parquet",
    )
    p.add_argument(
        "--validation-json",
        type=Path,
        default=root / "data" / "aux" / "sentinel2_land_metrics_validation.json",
    )
    p.add_argument(
        "--buffer-deg",
        type=float,
        default=DEFAULT_BUFFER_DEG_NO2,
        help="Half-width (deg) of square buffer around centroid (default 0.1°)",
    )
    p.add_argument("--quick-test", action="store_true", help="Run on a small grids×weeks subset")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output parquet / validation json before extraction (thesis reproducibility)",
    )
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else root / args.input
    outp = args.output if args.output.is_absolute() else root / args.output
    vj = args.validation_json if args.validation_json.is_absolute() else root / args.validation_json
    run_pipeline(inp, outp, vj, args.buffer_deg, quick_test=args.quick_test, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
