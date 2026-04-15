"""
Sentinel-5P TROPOMI NO₂ (GEE: COPERNICUS/S5P/NRTI/L3_NO2) → grid × week parquet.

Observational only:
- Weekly temporal composite: mean of daily L3 images in [week_start_utc, week_start_utc + 7d) via GEE only.
- Spatial statistics: reduceRegions(mean + stdDev) at 3500 m — no Python-side interpolation or filling.

Requires: `earthengine authenticate` [+ GOOGLE_CLOUD_PROJECT or EE_PROJECT if needed].

Output: data/aux/no2_grid_week.parquet
"""

from __future__ import annotations

import argparse
import json
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

S5P_COLLECTION = "COPERNICUS/S5P/NRTI/L3_NO2"
NO2_BAND = "tropospheric_NO2_column_number_density"
REDUCE_SCALE_M = 3500


def _import_ee():
    try:
        import ee  # noqa: WPS433
    except ImportError as e:
        raise SystemExit(
            "Install earthengine-api (pip install earthengine-api) and run: earthengine authenticate"
        ) from e
    return ee


def _mean_std_from_props(props: dict) -> tuple[float | None, float | None]:
    mean_v, std_v = None, None
    for k, v in props.items():
        if k == "grid_cell_id":
            continue
        kl = k.lower()
        if v is None or (isinstance(v, float) and (v != v)):
            continue
        if "mean" in kl and NO2_BAND in k:
            mean_v = float(v)
        elif "stddev" in kl and NO2_BAND in k:
            std_v = float(v)
    if mean_v is None:
        mean_v = props.get(f"{NO2_BAND}_mean")
        if mean_v is not None:
            mean_v = float(mean_v)
    if std_v is None:
        std_v = props.get(f"{NO2_BAND}_stdDev") or props.get(f"{NO2_BAND}_stddev")
        if std_v is not None:
            std_v = float(std_v)
    # Some reduceRegions payloads use generic keys ("mean", "stdDev") for single-band images.
    if mean_v is None and props.get("mean") is not None:
        mean_v = float(props["mean"])
    if std_v is None:
        raw_std = props.get("stdDev", props.get("stddev"))
        if raw_std is not None:
            std_v = float(raw_std)
    return mean_v, std_v


def extract_no2_weekly(
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
    *,
    buffer_deg: float,
    ee,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bbox = study_bbox(grids, ee)
    fc = build_grid_feature_collection(grids, buffer_deg, ee)
    ic_base = ee.ImageCollection(S5P_COLLECTION).select(NO2_BAND).filterBounds(bbox)

    rows: list[dict[str, Any]] = []
    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)

    for wk, start_dt, end_dt in iter_week_utc_bounds(weeks):
        start_ee, end_ee = week_filter_dates_ee(ee, start_dt, end_dt)
        ic = ic_base.filterDate(start_ee, end_ee)
        n_img = ic.size().getInfo()
        if n_img == 0:
            for _, r in grids.iterrows():
                rows.append(
                    {
                        "grid_cell_id": r["grid_cell_id"],
                        "week_start_utc": pd.Timestamp(wk),
                        "no2_mean_t": float("nan"),
                        "no2_std_t": float("nan"),
                    }
                )
            continue

        img = ic.mean()
        reduced = img.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=REDUCE_SCALE_M,
            tileScale=4,
        )
        feats = reduced.getInfo().get("features", [])
        by_id: dict[str, tuple[float | None, float | None]] = {}
        for f in feats:
            props = f.get("properties", {})
            gid = props.get("grid_cell_id")
            if gid is None:
                continue
            mv, sv = _mean_std_from_props(props)
            by_id[str(gid)] = (mv, sv)

        for _, r in grids.iterrows():
            gid = str(r["grid_cell_id"])
            pair = by_id.get(gid)
            if pair is None:
                mean_v, std_v = float("nan"), float("nan")
            else:
                mean_v, std_v = pair[0], pair[1]
                if mean_v is None:
                    mean_v = float("nan")
                if std_v is None:
                    std_v = float("nan")
            rows.append(
                {
                    "grid_cell_id": r["grid_cell_id"],
                    "week_start_utc": pd.Timestamp(wk),
                    "no2_mean_t": mean_v,
                    "no2_std_t": std_v,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True)

    validation = compute_validation(out, grids, weeks)
    return out, validation


def compute_validation(
    extracted: pd.DataFrame,
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
) -> dict[str, Any]:
    n_grids = len(grids)
    n_weeks = len([w for w in weeks if not pd.isna(w)])
    expected = n_grids * n_weeks
    n_obs = int(len(extracted))
    valid = extracted["no2_mean_t"].notna() if not extracted.empty else pd.Series(dtype=bool)
    n_valid = int(valid.sum()) if len(valid) else 0

    week_ts = [pd.Timestamp(x) for x in weeks if not pd.isna(x)]
    week_norm = pd.to_datetime(week_ts, utc=True)
    expected_keys = pd.MultiIndex.from_product(
        [grids["grid_cell_id"].values, week_norm],
        names=["grid_cell_id", "week_start_utc"],
    )
    if extracted.empty:
        missing_weeks_per_grid = {str(g): int(n_weeks) for g in grids["grid_cell_id"].values}
        coverage_pct = 0.0
    else:
        ex = extracted.copy()
        ex["week_start_utc"] = pd.to_datetime(ex["week_start_utc"], utc=True)
        merged = (
            pd.DataFrame(index=expected_keys)
            .reset_index()
            .merge(
                ex[["grid_cell_id", "week_start_utc", "no2_mean_t"]],
                on=["grid_cell_id", "week_start_utc"],
                how="left",
            )
        )
        missing_weeks_per_grid = (
            merged.groupby("grid_cell_id", sort=False)["no2_mean_t"]
            .apply(lambda s: int(s.isna().sum()))
            .astype(int)
            .to_dict()
        )
        coverage_pct = round(100.0 * n_valid / expected, 4) if expected else 0.0

    return {
        "number_of_grid_week_observations": n_obs,
        "number_of_valid_no2_values": n_valid,
        "expected_grid_week_pairs": int(expected),
        "coverage_percent": coverage_pct,
        "missing_weeks_per_grid": {str(k): int(v) for k, v in missing_weeks_per_grid.items()},
        "n_grids": int(n_grids),
        "n_weeks": int(n_weeks),
        "data_policy": "GEE reducers only; no interpolation or synthetic NO₂ filling",
    }


def run_pipeline(
    input_parquet: Path,
    output_parquet: Path,
    validation_json: Path | None,
    buffer_deg: float,
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
    df, validation = extract_no2_weekly(grids, weeks, buffer_deg=buffer_deg, ee=ee)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    if validation_json:
        validation_json.parent.mkdir(parents=True, exist_ok=True)
        with validation_json.open("w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2, default=str)

    print(json.dumps({"no2_gee_validation": validation}, indent=2, default=str))
    print(f"Wrote {output_parquet.resolve()}")
    return validation


def run_extraction(config) -> pd.DataFrame:
    """Standardized extraction wrapper for orchestrators."""
    if isinstance(config, dict):
        input_parquet = Path(config.get("input", Path(__file__).resolve().parents[2] / "data" / "modeling_dataset.parquet"))
        output_parquet = Path(config.get("output", Path(__file__).resolve().parents[2] / "data" / "aux" / "no2_grid_week.parquet"))
        validation_json = Path(config.get("validation_json", Path(__file__).resolve().parents[2] / "data" / "aux" / "no2_gee_validation.json"))
        buffer_deg = float(config.get("buffer_deg", DEFAULT_BUFFER_DEG_NO2))
    else:
        input_parquet = Path(__file__).resolve().parents[2] / "data" / "modeling_dataset.parquet"
        output_parquet = Path(__file__).resolve().parents[2] / "data" / "aux" / "no2_grid_week.parquet"
        validation_json = Path(__file__).resolve().parents[2] / "data" / "aux" / "no2_gee_validation.json"
        buffer_deg = DEFAULT_BUFFER_DEG_NO2
    run_pipeline(input_parquet, output_parquet, validation_json, buffer_deg)
    return pd.read_parquet(output_parquet)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="S5P NO₂ grid-week extraction via Google Earth Engine")
    p.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "modeling_dataset.parquet",
        help="Modeling parquet (grid_cell_id, week_start_utc, centroids)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "aux" / "no2_grid_week.parquet",
    )
    p.add_argument(
        "--validation-json",
        type=Path,
        default=root / "data" / "aux" / "no2_gee_validation.json",
        help="Write coverage / missing-week stats",
    )
    p.add_argument(
        "--buffer-deg",
        type=float,
        default=DEFAULT_BUFFER_DEG_NO2,
        help="Half-width (deg) of square buffer around centroid (default 0.1°)",
    )
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else root / args.input
    outp = args.output if args.output.is_absolute() else root / args.output
    vj = args.validation_json if args.validation_json.is_absolute() else root / args.validation_json
    run_pipeline(inp, outp, vj, args.buffer_deg)


if __name__ == "__main__":
    main()
