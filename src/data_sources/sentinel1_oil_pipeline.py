"""
Sentinel-1 GRD SAR (GEE: COPERNICUS/S1_GRD) → grid-week oil *proxy* (dark-water fraction).

Method (observational only, no synthetic oil):
1. Weekly IW/VH composite: pixel-wise median of VH in [week_start_utc, week_start_utc + 7d).
2. Regional p20(VH) over study bbox → threshold for that week.
3. Per grid buffer: mean(dark mask) = oil_slick_probability_t; sum(dark mask) = oil_slick_count_t
   (sum = count of VH-resolution pixels classified dark at reducer scale).

Missing SAR week: NaN probability, count 0 — no imputation.

Requires: `earthengine authenticate` [+ GOOGLE_CLOUD_PROJECT or EE_PROJECT if needed].

Output: data/aux/sentinel1_oil_slicks.parquet (contract path)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from data_sources.gee_grid_utils import (  # noqa: E402
    DEFAULT_BUFFER_DEG_S1,
    build_grid_feature_collection,
    initialize_earth_engine,
    iter_week_utc_bounds,
    load_modeling_grids_and_weeks,
    study_bbox,
    week_filter_dates_ee,
)
from utils.ee_safe import safe_ee_call  # noqa: E402

S1_COLLECTION = "COPERNICUS/S1_GRD"
# Use coarser reducer scale for tractable full-study weekly extraction.
# This preserves observational dark-pixel statistics while avoiding stalled runs.
S1_SCALE_M = 30
GRID_CHUNK_SIZE = 48
STALL_WARNING_SECONDS = 300
STALL_ABORT_SECONDS = 600
EE_TIMEOUT_SECONDS = 120
EE_MAX_RETRIES = 2
SAFE_MODE_MAX_GRIDS = 20

LOGGER = logging.getLogger("sentinel1_oil_pipeline")


def _emit(message: str, *args) -> None:
    LOGGER.info(message, *args)
    try:
        print(message % args if args else message)
    except Exception:  # noqa: BLE001
        print(message)


def _import_ee():
    try:
        import ee  # noqa: WPS433
    except ImportError as e:
        raise SystemExit(
            "Install earthengine-api (pip install earthengine-api) and run: earthengine authenticate"
        ) from e
    return ee


def _p20_threshold(img: Any, bbox: Any, ee) -> float | None:
    """Single p20(VH) over study bbox for the weekly median composite (regional reference)."""
    try:
        stat = img.select("VH").reduceRegion(
            reducer=ee.Reducer.percentile([20]),
            geometry=bbox,
            scale=200,
            maxPixels=1e9,
            bestEffort=True,
        )
        info = safe_ee_call(
            lambda: stat.getInfo(),
            timeout=EE_TIMEOUT_SECONDS,
            max_retries=EE_MAX_RETRIES,
            logger=LOGGER,
            context="p20_threshold.getInfo",
        )
    except Exception:  # noqa: BLE001
        return None
    if not info:
        return None
    for k, v in info.items():
        if v is None or (isinstance(v, float) and v != v):
            continue
        if "VH" in k.upper():
            return float(v)
    for k, v in info.items():
        if "p20" in k.lower() and isinstance(v, (int, float)):
            return float(v)
    return None


def _percentile_threshold(img: Any, band: str, percentile: float, bbox: Any, ee) -> float | None:
    """Percentile helper with robust property-name handling."""
    try:
        stat = img.select(band).reduceRegion(
            reducer=ee.Reducer.percentile([percentile]),
            geometry=bbox,
            scale=200,
            maxPixels=1e9,
            bestEffort=True,
        )
        info = safe_ee_call(
            lambda: stat.getInfo(),
            timeout=EE_TIMEOUT_SECONDS,
            max_retries=EE_MAX_RETRIES,
            logger=LOGGER,
            context=f"percentile_threshold.getInfo band={band} p={percentile}",
        )
    except Exception:  # noqa: BLE001
        return None
    if not info:
        return None
    for k, v in info.items():
        if v is None or (isinstance(v, float) and v != v):
            continue
        ku = k.upper()
        if band.upper() in ku and "P" in ku:
            return float(v)
    for _, v in info.items():
        if isinstance(v, (float, int)):
            return float(v)
    return None


def _band_distribution_stats(img: Any, band: str, bbox: Any, ee) -> dict[str, float | None]:
    """Quick distribution stats for debug logging."""
    out = {
        "mean": None,
        "stddev": None,
        "p05": None,
        "p20": None,
        "p50": None,
        "p80": None,
        "p95": None,
    }
    try:
        reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True).combine(
            reducer2=ee.Reducer.percentile([5, 20, 50, 80, 95]),
            sharedInputs=True,
        )
        stat = img.select(band).reduceRegion(
            reducer=reducer,
            geometry=bbox,
            scale=200,
            maxPixels=1e9,
            bestEffort=True,
        )
        info = (
            safe_ee_call(
                lambda: stat.getInfo(),
                timeout=EE_TIMEOUT_SECONDS,
                max_retries=EE_MAX_RETRIES,
                logger=LOGGER,
                context=f"band_distribution_stats.getInfo band={band}",
            )
            or {}
        )
    except Exception:  # noqa: BLE001
        return out

    for key in list(out.keys()):
        for k, v in info.items():
            if v is None:
                continue
            kl = k.lower()
            if band.lower() not in kl:
                continue
            if key == "mean" and "mean" in kl:
                out[key] = float(v)
            elif key == "stddev" and "std" in kl:
                out[key] = float(v)
            elif key.startswith("p") and key in kl.replace("_", ""):
                out[key] = float(v)
    return out


def _parse_sum(props: dict) -> float | None:
    for k in ("sum", "dark"):
        if k in props and props[k] is not None:
            try:
                return float(props[k])
            except (TypeError, ValueError):
                continue
    for k, v in props.items():
        if k == "grid_cell_id" or v is None:
            continue
        if "sum" in k.lower():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _iter_grid_chunks(grids: pd.DataFrame, chunk_size: int):
    n = len(grids)
    for i in range(0, n, chunk_size):
        yield grids.iloc[i : i + chunk_size].copy()


class ProgressWatchdog:
    def __init__(self, logger: logging.Logger, warning_after_s: int = STALL_WARNING_SECONDS) -> None:
        self.logger = logger
        self.warning_after_s = warning_after_s
        self._last_progress = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.timed_out = False

    def start(self) -> None:
        self._thread.start()

    def touch(self) -> None:
        self._last_progress = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        warned = False
        while not self._stop.wait(1.0):
            idle = time.monotonic() - self._last_progress
            if idle >= self.warning_after_s:
                self.logger.warning("[S1][WARNING] No progress detected for 5 minutes - pipeline still alive")
                warned = True
                if idle >= STALL_ABORT_SECONDS:
                    self.logger.warning("[S1] TIMEOUT: skipping Sentinel-1 stage safely")
                    self.timed_out = True
                    self._stop.set()
                    return
            elif warned and idle < 10:
                warned = False


def _empty_sentinel_output(grids: pd.DataFrame, weeks: list[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for wk in weeks:
        wk_ts = pd.Timestamp(wk)
        for _, r in grids.iterrows():
            rows.append(
                {
                    "grid_cell_id": r["grid_cell_id"],
                    "week_start_utc": wk_ts,
                    "oil_slick_probability_t": float("nan"),
                    "oil_slick_count_t": 0.0,
                }
            )
    out = pd.DataFrame(rows, columns=["grid_cell_id", "week_start_utc", "oil_slick_probability_t", "oil_slick_count_t"])
    if not out.empty:
        out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    return out


def extract_oil_weekly(
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
    *,
    buffer_deg: float,
    ee,
    existing: pd.DataFrame | None = None,
    on_week_complete: Callable[[pd.DataFrame], None] | None = None,
    safe_mode: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _emit("[S1] STEP 1: sentinel module entered")
    if safe_mode and len(grids) > SAFE_MODE_MAX_GRIDS:
        LOGGER.info("[S1][SAFE MODE] Limiting grids from %d to %d", len(grids), SAFE_MODE_MAX_GRIDS)
        grids = grids.head(SAFE_MODE_MAX_GRIDS).copy()

    _emit("[S1] STEP 2: building EE ImageCollection")
    bbox = study_bbox(grids, ee)
    bounded_start = pd.Timestamp(min(weeks)).to_pydatetime()
    bounded_end = (pd.Timestamp(max(weeks)) + pd.Timedelta(days=7)).to_pydatetime()
    start_ee, end_ee = week_filter_dates_ee(ee, bounded_start, bounded_end)
    _emit("[S1] STEP 3: applying filters (date/bounds)")
    # Build ingestion collections explicitly for VV+VH (preferred) and VH-only (fallback).
    s1_base_iw = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(bbox)
        .filterDate(start_ee, end_ee)
        .limit(500)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )
    s1_dual = s1_base_iw.filter(
        ee.Filter.And(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
            ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
        )
    ).select(["VV", "VH"])
    s1_vh = s1_base_iw.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")).select("VH")
    _emit("[S1] STEP 4: collection ready, entering loop")
    _emit("[S1] collection size check skipped to avoid blocking")

    rows: list[dict[str, Any]] = []
    debug: dict[str, Any] = {
        "collection": S1_COLLECTION,
        "band_policy": "Prefer VV+VH (dual-pol), fallback to VH-only",
        "grid_chunk_size": GRID_CHUNK_SIZE,
        "reducer_scale_m": S1_SCALE_M,
        "week_stats": [],
    }
    n_grids = len(grids)
    n_weeks_total = len(weeks)
    grid_position = {str(gid): idx for idx, gid in enumerate(grids["grid_cell_id"].astype(str).tolist(), start=1)}
    completed_weeks: set[pd.Timestamp] = set()
    if existing is not None and not existing.empty:
        ex = existing.copy()
        ex["week_start_utc"] = pd.to_datetime(ex["week_start_utc"], utc=True, errors="coerce")
        ex = ex.dropna(subset=["grid_cell_id", "week_start_utc"])
        rows.extend(ex.to_dict("records"))
        wk_stats = ex.groupby("week_start_utc", sort=False).agg(
            n_grids=("grid_cell_id", "nunique"),
            n_prob=("oil_slick_probability_t", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            n_count_pos=(
                "oil_slick_count_t",
                lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum()),
            ),
        )
        # Resume only from weeks that are both complete and contain any usable oil signal.
        # This prevents persisting all-NaN/all-zero failed checkpoints indefinitely.
        reusable = wk_stats[
            (wk_stats["n_grids"] >= n_grids) & ((wk_stats["n_prob"] > 0) | (wk_stats["n_count_pos"] > 0))
        ]
        completed_weeks = set(reusable.index)

    watchdog = ProgressWatchdog(LOGGER)
    watchdog.start()
    heartbeat_counter = 0
    week_index = 0

    try:
        _emit("[S1] ENTERING GRID LOOP")
        for wk, start_dt, end_dt in iter_week_utc_bounds(weeks):
            if watchdog.timed_out:
                out = _empty_sentinel_output(grids, weeks)
                validation = compute_oil_validation(out, grids, weeks)
                debug["summary"] = {"total_rows": int(len(out)), "rows_with_prob": 0, "rows_with_positive_count": 0}
                validation["debug"] = debug
                return out, validation
            week_index += 1
            week_ts = pd.Timestamp(wk)
            if week_ts.tzinfo is None:
                week_ts = week_ts.tz_localize("UTC")
            else:
                week_ts = week_ts.tz_convert("UTC")
            if week_ts in completed_weeks:
                continue

            LOGGER.info("[S1] Grid extraction loop entering week %d/%d", week_index, n_weeks_total)
            watchdog.touch()
            start_ee, end_ee = week_filter_dates_ee(ee, start_dt, end_dt)
            ic_dual = s1_dual.filterDate(start_ee, end_ee)
            ic_vh = s1_vh.filterDate(start_ee, end_ee)
            try:
                n_dual_raw = safe_ee_call(
                    lambda: ic_dual.size().getInfo(),
                    timeout=EE_TIMEOUT_SECONDS,
                    max_retries=EE_MAX_RETRIES,
                    logger=LOGGER,
                    context=f"scene_count_dual week={week_ts.date()}",
                )
                n_dual = int(n_dual_raw) if n_dual_raw is not None else 0
            except Exception:  # noqa: BLE001
                LOGGER.warning("[S1][SKIP] grid=ALL week=%s reason=EE_TIMEOUT", week_ts.date())
                for _, r in grids.iterrows():
                    rows.append(
                        {
                            "grid_cell_id": r["grid_cell_id"],
                            "week_start_utc": week_ts,
                            "oil_slick_probability_t": float("nan"),
                            "oil_slick_count_t": 0.0,
                        }
                    )
                continue
            try:
                n_vh_raw = safe_ee_call(
                    lambda: ic_vh.size().getInfo(),
                    timeout=EE_TIMEOUT_SECONDS,
                    max_retries=EE_MAX_RETRIES,
                    logger=LOGGER,
                    context=f"scene_count_vh week={week_ts.date()}",
                )
                n_vh = int(n_vh_raw) if n_vh_raw is not None else 0
            except Exception:  # noqa: BLE001
                LOGGER.warning("[S1][SKIP] grid=ALL week=%s reason=EE_TIMEOUT", week_ts.date())
                for _, r in grids.iterrows():
                    rows.append(
                        {
                            "grid_cell_id": r["grid_cell_id"],
                            "week_start_utc": week_ts,
                            "oil_slick_probability_t": float("nan"),
                            "oil_slick_count_t": 0.0,
                        }
                    )
                continue
            LOGGER.info("[S1] Scenes found: dual=%d vh_only=%d", n_dual, n_vh)
            watchdog.touch()
            week_log: dict[str, Any] = {
            "week_start_utc": week_ts.isoformat(),
            "scene_count_dual_pol_vv_vh": n_dual,
            "scene_count_vh_only": n_vh,
            "selected_stream": "none",
            "chunk_count": 0,
            "chunks_with_threshold": 0,
            "chunks_with_features": 0,
            "rows_appended": 0,
            "rows_with_prob": 0,
            "rows_with_count_positive": 0,
            "threshold_vh_global_p20": None,
            "threshold_vh_global_p15_fallback": None,
            "threshold_ratio_global_p80": None,
            "vv_distribution": {},
            "vh_distribution": {},
            "ratio_distribution": {},
        }

            if n_dual > 0:
                ic = ic_dual
                week_log["selected_stream"] = "dual_pol"
            elif n_vh > 0:
                ic = ic_vh
                week_log["selected_stream"] = "vh_only"
            else:
                for _, r in grids.iterrows():
                    rows.append(
                        {
                            "grid_cell_id": r["grid_cell_id"],
                            "week_start_utc": week_ts,
                            "oil_slick_probability_t": float("nan"),
                            "oil_slick_count_t": 0.0,
                        }
                    )
                week_log["rows_appended"] = int(len(grids))
                debug["week_stats"].append(week_log)
                continue

            img = ic.median()
            has_vv = week_log["selected_stream"] == "dual_pol"
            vh_global_p20 = _p20_threshold(img, bbox, ee)
            vh_global_p15 = _percentile_threshold(img, "VH", 15, bbox, ee)
            ratio_global_p80 = None
            ratio_img = None
            if has_vv:
                ratio_img = img.select("VV").subtract(img.select("VH")).rename("VV_minus_VH")
                ratio_global_p80 = _percentile_threshold(ratio_img, "VV_minus_VH", 80, bbox, ee)

            week_log["threshold_vh_global_p20"] = vh_global_p20
            week_log["threshold_vh_global_p15_fallback"] = vh_global_p15
            week_log["threshold_ratio_global_p80"] = ratio_global_p80
            if not safe_mode:
                week_log["vh_distribution"] = _band_distribution_stats(img, "VH", bbox, ee)
                if has_vv:
                    week_log["vv_distribution"] = _band_distribution_stats(img, "VV", bbox, ee)
                    week_log["ratio_distribution"] = _band_distribution_stats(ratio_img, "VV_minus_VH", bbox, ee)

            combined_red = ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True)
            for gchunk in _iter_grid_chunks(grids, GRID_CHUNK_SIZE):
                week_log["chunk_count"] = int(week_log["chunk_count"]) + 1
                chunk_fc = build_grid_feature_collection(gchunk, buffer_deg, ee)
                chunk_bbox = study_bbox(gchunk, ee)
                p20 = _p20_threshold(img, chunk_bbox, ee)
                if p20 is None:
                    p20 = vh_global_p20 if vh_global_p20 is not None else vh_global_p15
                ratio_p80 = None
                if has_vv and ratio_img is not None:
                    ratio_p80 = _percentile_threshold(ratio_img, "VV_minus_VH", 80, chunk_bbox, ee)
                    if ratio_p80 is None:
                        ratio_p80 = ratio_global_p80

                if p20 is None:
                    for _, r in gchunk.iterrows():
                        LOGGER.warning("[S1][SKIP] grid=%s, week=%s, reason=EE_TIMEOUT", r["grid_cell_id"], week_ts.date())
                        rows.append(
                            {
                                "grid_cell_id": r["grid_cell_id"],
                                "week_start_utc": week_ts,
                                "oil_slick_probability_t": float("nan"),
                                "oil_slick_count_t": 0.0,
                            }
                        )
                    continue

                week_log["chunks_with_threshold"] = int(week_log["chunks_with_threshold"]) + 1
                vh = img.select("VH")
                dark = vh.lt(ee.Number(p20))
                # Dual-pol refinement: prioritize dark VH with elevated VV-VH contrast.
                if has_vv and ratio_p80 is not None and ratio_img is not None:
                    dark = dark.And(ratio_img.gte(ee.Number(ratio_p80)))
                dark = dark.rename("dark")
                reduced = dark.float().reduceRegions(
                    collection=chunk_fc,
                    reducer=combined_red,
                    scale=S1_SCALE_M,
                    tileScale=2,
                )

                reduced_info = safe_ee_call(
                    lambda: reduced.getInfo(),
                    timeout=EE_TIMEOUT_SECONDS,
                    max_retries=EE_MAX_RETRIES,
                    logger=LOGGER,
                    context=f"reduceRegions.getInfo week={week_ts.date()}",
                )
                feats = (reduced_info or {}).get("features", [])
                if reduced_info is None:
                    for _, r in gchunk.iterrows():
                        LOGGER.warning("[S1][SKIP] grid=%s, week=%s, reason=EE_TIMEOUT", r["grid_cell_id"], week_ts.date())
                        rows.append(
                            {
                                "grid_cell_id": r["grid_cell_id"],
                                "week_start_utc": week_ts,
                                "oil_slick_probability_t": float("nan"),
                                "oil_slick_count_t": 0.0,
                            }
                        )
                    continue

                if feats:
                    week_log["chunks_with_features"] = int(week_log["chunks_with_features"]) + 1
                LOGGER.info("[S1] Scenes found: %d", len(feats))
                by_prob: dict[str, float] = {}
                by_sum: dict[str, float] = {}
                for f in feats:
                    p = f.get("properties", {})
                    gid = p.get("grid_cell_id")
                    if gid is None:
                        continue
                    pval = p.get("mean", p.get("dark"))
                    if pval is not None:
                        by_prob[str(gid)] = float(pval)
                    val = _parse_sum(p)
                    if val is None:
                        for kk, vv in p.items():
                            if kk != "grid_cell_id" and isinstance(vv, (int, float)):
                                val = float(vv)
                                break
                    if val is not None:
                        by_sum[str(gid)] = float(val)

                for _, r in gchunk.iterrows():
                    gid = str(r["grid_cell_id"])
                    _emit("[S1] Processing grid %d/%d - %s", grid_position.get(gid, 0), n_grids, gid)
                    _emit("[S1] Grid %s Week %d/%d", gid, week_index, n_weeks_total)
                    watchdog.touch()
                    pr = by_prob.get(gid)
                    sm = by_sum.get(gid)
                    week_log["rows_appended"] = int(week_log["rows_appended"]) + 1
                    if pr is not None:
                        week_log["rows_with_prob"] = int(week_log["rows_with_prob"]) + 1
                    if sm is not None and sm > 0:
                        week_log["rows_with_count_positive"] = int(week_log["rows_with_count_positive"]) + 1
                    rows.append(
                        {
                            "grid_cell_id": r["grid_cell_id"],
                            "week_start_utc": week_ts,
                            "oil_slick_probability_t": float(pr) if pr is not None else float("nan"),
                            "oil_slick_count_t": float(sm) if sm is not None else 0.0,
                        }
                    )
                    vv_mean = week_log.get("vv_distribution", {}).get("mean")
                    vh_mean = week_log.get("vh_distribution", {}).get("mean")
                    LOGGER.info("[S1] Features computed: VV_mean=%s, VH_mean=%s", vv_mean, vh_mean)
                    heartbeat_counter += 1
                    if heartbeat_counter % 5 == 0:
                        _emit("[S1] HEARTBEAT: still processing grids")
            completed_weeks.add(week_ts)
            debug["week_stats"].append(week_log)
            if on_week_complete is not None:
                on_week_complete(pd.DataFrame(rows))
    finally:
        watchdog.stop()

    out = pd.DataFrame(rows)
    if not out.empty:
        out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True)

    validation = compute_oil_validation(out, grids, weeks)
    debug["summary"] = {
        "total_rows": int(len(out)),
        "rows_with_prob": int(out["oil_slick_probability_t"].notna().sum()) if "oil_slick_probability_t" in out.columns else 0,
        "rows_with_positive_count": (
            int((pd.to_numeric(out["oil_slick_count_t"], errors="coerce").fillna(0) > 0).sum())
            if "oil_slick_count_t" in out.columns
            else 0
        ),
    }
    validation["debug"] = debug
    return out, validation


def compute_oil_validation(
    extracted: pd.DataFrame,
    grids: pd.DataFrame,
    weeks: list[pd.Timestamp],
) -> dict[str, Any]:
    n_grids = len(grids)
    n_weeks = len([w for w in weeks if not pd.isna(w)])
    expected = n_grids * n_weeks
    n_obs = int(len(extracted))
    if extracted.empty:
        oil_obs = pd.Series(dtype=bool)
    else:
        oil_obs = extracted["oil_slick_probability_t"].notna() | (
            pd.to_numeric(extracted["oil_slick_count_t"], errors="coerce").fillna(0) > 0
        )
    n_oil = int(oil_obs.sum()) if len(oil_obs) else 0
    coverage_pct = round(100.0 * n_oil / expected, 4) if expected else 0.0

    week_ts = [pd.Timestamp(x) for x in weeks if not pd.isna(x)]
    week_norm = pd.to_datetime(week_ts, utc=True)
    expected_keys = pd.MultiIndex.from_product(
        [grids["grid_cell_id"].values, week_norm],
        names=["grid_cell_id", "week_start_utc"],
    )
    if extracted.empty:
        missing_weeks_per_grid = {str(g): int(n_weeks) for g in grids["grid_cell_id"].values}
    else:
        ex = extracted.copy()
        ex["week_start_utc"] = pd.to_datetime(ex["week_start_utc"], utc=True)
        merged = (
            pd.DataFrame(index=expected_keys)
            .reset_index()
            .merge(
                ex[
                    [
                        "grid_cell_id",
                        "week_start_utc",
                        "oil_slick_probability_t",
                        "oil_slick_count_t",
                    ]
                ],
                on=["grid_cell_id", "week_start_utc"],
                how="left",
            )
        )
        has_data = merged["oil_slick_probability_t"].notna() | (
            pd.to_numeric(merged["oil_slick_count_t"], errors="coerce").fillna(0) > 0
        )
        merged["_has_oil"] = has_data.fillna(False)
        missing_weeks_per_grid = (
            merged.groupby("grid_cell_id", sort=False)["_has_oil"]
            .apply(lambda s: int((~s).sum()))
            .to_dict()
        )

    return {
        "number_of_grid_week_observations": n_obs,
        "rows_with_oil_data": n_oil,
        "expected_grid_week_pairs": int(expected),
        "oil_coverage_percent": coverage_pct,
        "missing_weeks_per_grid": {str(k): int(v) for k, v in missing_weeks_per_grid.items()},
        "n_grids": int(n_grids),
        "n_weeks": int(n_weeks),
        "method": "median(VH) weekly; dark = VH < p20(bbox); mean(mask)=prob, count(dark)=proxy count",
        "data_policy": "No synthetic oil; missing week → NaN + 0 count",
    }


def run_pipeline(
    input_parquet: Path,
    output_parquet: Path,
    validation_json: Path | None,
    buffer_deg: float,
    *,
    resume: bool = True,
    debug_log_json: Path | None = None,
    safe_mode: bool = False,
) -> dict[str, Any]:
    if safe_mode:
        LOGGER.setLevel(logging.INFO)
    LOGGER.info("[S1] Sentinel-1 module STARTED")
    print("[S1] Sentinel-1 module STARTED")
    ee = _import_ee()
    try:
        initialize_earth_engine(ee)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Earth Engine init failed. Run: earthengine authenticate\n"
            "Set GOOGLE_CLOUD_PROJECT or EE_PROJECT if required.\n"
            f"Detail: {e}"
        ) from e

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    grids, weeks = load_modeling_grids_and_weeks(input_parquet)
    existing: pd.DataFrame | None = None
    if resume and output_parquet.exists():
        try:
            existing = pd.read_parquet(output_parquet)
        except Exception:  # noqa: BLE001
            existing = None

    def _checkpoint(df_partial: pd.DataFrame) -> None:
        if df_partial.empty:
            return
        dfx = df_partial.copy()
        dfx["week_start_utc"] = pd.to_datetime(dfx["week_start_utc"], utc=True, errors="coerce")
        dfx = (
            dfx.sort_values(["week_start_utc", "grid_cell_id"])
            .drop_duplicates(subset=["grid_cell_id", "week_start_utc"], keep="last")
            .reset_index(drop=True)
        )
        dfx.to_parquet(output_parquet, index=False)

    df, validation = extract_oil_weekly(
        grids,
        weeks,
        buffer_deg=buffer_deg,
        ee=ee,
        existing=existing,
        on_week_complete=_checkpoint,
        safe_mode=safe_mode,
    )

    df = (
        df.sort_values(["week_start_utc", "grid_cell_id"])
        .drop_duplicates(subset=["grid_cell_id", "week_start_utc"], keep="last")
        .reset_index(drop=True)
    )
    df.to_parquet(output_parquet, index=False)

    if validation_json:
        validation_json.parent.mkdir(parents=True, exist_ok=True)
        with validation_json.open("w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2, default=str)
    if debug_log_json:
        debug_log_json.parent.mkdir(parents=True, exist_ok=True)
        with debug_log_json.open("w", encoding="utf-8") as f:
            json.dump(validation.get("debug", {}), f, indent=2, default=str)

    print(json.dumps({"sentinel1_oil_validation": validation}, indent=2, default=str))
    print(f"Wrote {output_parquet.resolve()}")
    if debug_log_json:
        print(f"Wrote {debug_log_json.resolve()}")
    return validation


def run_extraction(config) -> pd.DataFrame:
    """Standardized extraction wrapper for orchestrators."""
    root = Path(__file__).resolve().parents[2]
    if isinstance(config, dict):
        input_parquet = Path(config.get("input", root / "data" / "modeling_dataset.parquet"))
        output_parquet = Path(config.get("output", root / "data" / "aux" / "sentinel1_oil_slicks.parquet"))
        validation_json = Path(config.get("validation_json", root / "data" / "aux" / "sentinel1_oil_validation.json"))
        buffer_deg = float(config.get("buffer_deg", DEFAULT_BUFFER_DEG_S1))
        resume = bool(config.get("resume", True))
        debug_log_json = Path(config.get("debug_log_json", root / "data" / "validation" / "sentinel1_oil_debug_log.json"))
        safe_mode = bool(config.get("safe_mode", False))
    else:
        input_parquet = root / "data" / "modeling_dataset.parquet"
        output_parquet = root / "data" / "aux" / "sentinel1_oil_slicks.parquet"
        validation_json = root / "data" / "aux" / "sentinel1_oil_validation.json"
        buffer_deg = DEFAULT_BUFFER_DEG_S1
        resume = True
        debug_log_json = root / "data" / "validation" / "sentinel1_oil_debug_log.json"
        safe_mode = False
    run_pipeline(
        input_parquet,
        output_parquet,
        validation_json,
        buffer_deg,
        resume=resume,
        debug_log_json=debug_log_json,
        safe_mode=safe_mode,
    )
    return pd.read_parquet(output_parquet)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Sentinel-1 SAR oil proxy → grid-week parquet (GEE)")
    p.add_argument("--input", type=Path, default=root / "data" / "modeling_dataset.parquet")
    p.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "aux" / "sentinel1_oil_slicks.parquet",
    )
    p.add_argument(
        "--validation-json",
        type=Path,
        default=root / "data" / "aux" / "sentinel1_oil_validation.json",
    )
    p.add_argument(
        "--buffer-deg",
        type=float,
        default=DEFAULT_BUFFER_DEG_S1,
        help="Half-width (deg) of square buffer around centroid (default 0.1°)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute all weeks from scratch (ignore existing output checkpoint)",
    )
    p.add_argument(
        "--debug-log-json",
        type=Path,
        default=root / "data" / "validation" / "sentinel1_oil_debug_log.json",
        help="Write step-by-step extraction diagnostics (scene counts, thresholds, retention stats).",
    )
    p.add_argument("--safe-mode", action="store_true", help="Safety mode: limit grids and force verbose runtime logging.")
    args = p.parse_args()
    inp = args.input if args.input.is_absolute() else root / args.input
    outp = args.output if args.output.is_absolute() else root / args.output
    vj = args.validation_json if args.validation_json.is_absolute() else root / args.validation_json
    dbg = args.debug_log_json if args.debug_log_json.is_absolute() else root / args.debug_log_json
    run_pipeline(inp, outp, vj, args.buffer_deg, resume=not args.no_resume, debug_log_json=dbg, safe_mode=args.safe_mode)


if __name__ == "__main__":
    main()
