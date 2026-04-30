from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCES = ("sentinel", "emodnet", "helcom")
GRID_RES_DEG = 0.1


def parse_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def derive_week_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    # Week starts on Monday in UTC.
    return ts.dt.floor("D") - pd.to_timedelta(ts.dt.weekday, unit="D")


def fixed_grid_id(lat: float | None, lon: float | None, res_deg: float = GRID_RES_DEG) -> str:
    if lat is None or lon is None or np.isnan(lat) or np.isnan(lon):
        return "unmapped"
    row = int(np.floor((lat + 90.0) / res_deg))
    col = int(np.floor((lon + 180.0) / res_deg))
    return f"g{res_deg:.3f}_r{row}_c{col}"


def grid_centroid_from_id(grid_id: str, res_deg: float = GRID_RES_DEG) -> tuple[float | None, float | None]:
    if grid_id == "unmapped":
        return (None, None)
    try:
        parts = grid_id.split("_")
        row = int(parts[1][1:])
        col = int(parts[2][1:])
        lat = (row + 0.5) * res_deg - 90.0
        lon = (col + 0.5) * res_deg - 180.0
        return (lat, lon)
    except Exception:  # noqa: BLE001
        return (None, None)


def load_source_records(project_root: Path, source: str) -> pd.DataFrame:
    src_dir = project_root / "data" / "processed" / source
    files = sorted(src_dir.rglob("*.parquet")) if src_dir.exists() else []
    if not files:
        return pd.DataFrame(
            columns=[
                "source",
                "dataset",
                "record_timestamp_utc",
                "ingested_at_utc",
                "latitude",
                "longitude",
                "geometry_wkt",
                "crs",
                "grid_id",
                "raw_record_ref",
                "payload",
            ]
        )
    parts = [pd.read_parquet(p) for p in files]
    out = pd.concat(parts, ignore_index=True)
    out["source"] = source
    out["event_ts"] = pd.to_datetime(out["record_timestamp_utc"], errors="coerce", utc=True)
    fallback = pd.to_datetime(out["ingested_at_utc"], errors="coerce", utc=True)
    out["event_ts"] = out["event_ts"].fillna(fallback)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude", "event_ts"])
    out["grid_cell_id"] = [
        fixed_grid_id(lat, lon, GRID_RES_DEG) for lat, lon in zip(out["latitude"], out["longitude"])
    ]
    out = out[out["grid_cell_id"] != "unmapped"].copy()
    return out


def _extract_numeric_properties(payload_obj: dict[str, Any]) -> dict[str, float]:
    props = payload_obj.get("properties", {}) if isinstance(payload_obj.get("properties"), dict) else {}
    out: dict[str, float] = {}
    for key, value in props.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            safe = key.replace(":", "_").replace("-", "_").replace(".", "_")
            out[safe] = float(value)
    return out


def _extract_feature_origin(payload_obj: dict[str, Any]) -> str | None:
    return payload_obj.get("id") or payload_obj.get("record_id") or payload_obj.get("name")


def _extract_obs_count_from_stats_payload(payload_str: Any) -> int | None:
    obj = parse_payload(payload_str)
    try:
        ndvi_stats = obj.get("outputs", {}).get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {})
        sample = ndvi_stats.get("sampleCount")
        nodata = ndvi_stats.get("noDataCount", 0)
        if sample is None:
            return None
        return int(sample - nodata)
    except Exception:  # noqa: BLE001
        return None


def sentinel_weekly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "week_start_utc",
                "grid_cell_id",
                "sentinel_observation_count",
                "sentinel_raw_refs",
                "sentinel_feature_ids",
                "sentinel_first_record_ts",
                "sentinel_last_record_ts",
            ]
        )

    x = df.copy()
    x["week_start_utc"] = derive_week_start(x["event_ts"])
    payload_objs = x["payload"].apply(parse_payload)
    numeric_dicts = payload_objs.apply(_extract_numeric_properties)
    numeric_df = pd.DataFrame(list(numeric_dicts)).fillna(np.nan)
    x = pd.concat([x.reset_index(drop=True), numeric_df.reset_index(drop=True)], axis=1)
    x["feature_origin"] = payload_objs.apply(_extract_feature_origin)

    group_cols = ["week_start_utc", "grid_cell_id"]
    base = (
        x.groupby(group_cols, dropna=False)
        .agg(
            sentinel_observation_count=("raw_record_ref", "count"),
            sentinel_raw_refs=("raw_record_ref", lambda s: sorted(set(s.dropna().astype(str)))),
            sentinel_feature_ids=("feature_origin", lambda s: sorted(set(v for v in s if isinstance(v, str)))),
            sentinel_first_record_ts=("event_ts", "min"),
            sentinel_last_record_ts=("event_ts", "max"),
        )
        .reset_index()
    )

    # Weekly mean/median over numeric sentinel properties (e.g., cloud cover).
    num_cols = [
        c
        for c in x.columns
        if c
        not in {
            "source",
            "dataset",
            "record_timestamp_utc",
            "ingested_at_utc",
            "latitude",
            "longitude",
            "geometry_wkt",
            "crs",
            "grid_id",
            "raw_record_ref",
            "payload",
            "event_ts",
            "grid_cell_id",
            "week_start_utc",
            "feature_origin",
        }
        and pd.api.types.is_numeric_dtype(x[c])
    ]
    if num_cols:
        agg_dict: dict[str, list[str]] = {c: ["mean", "median"] for c in num_cols}
        num_agg = x.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        num_agg.columns = [
            "week_start_utc"
            if c == "week_start_utc"
            else "grid_cell_id"
            if c == "grid_cell_id"
            else f"sentinel_{c}_{stat}"
            for c, stat in [
                (col if not isinstance(col, tuple) else col[0], "" if not isinstance(col, tuple) else col[1])
                for col in num_agg.columns
            ]
        ]
        # clean duplicate suffix for non-multiindex keys
        num_agg = num_agg.rename(
            columns={
                "sentinel_week_start_utc_": "week_start_utc",
                "sentinel_grid_cell_id_": "grid_cell_id",
            }
        )
        base = base.merge(num_agg, on=["week_start_utc", "grid_cell_id"], how="left")
    return base


def emodnet_static_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "grid_cell_id",
                "emodnet_record_count_static",
                "emodnet_raw_refs_static",
                "emodnet_feature_ids_static",
                "emodnet_first_record_ts_static",
                "emodnet_last_record_ts_static",
            ]
        )

    x = df.copy()
    payload_objs = x["payload"].apply(parse_payload)
    x["feature_origin"] = payload_objs.apply(_extract_feature_origin)

    agg = (
        x.groupby("grid_cell_id", dropna=False)
        .agg(
            emodnet_record_count_static=("raw_record_ref", "count"),
            emodnet_raw_refs_static=("raw_record_ref", lambda s: sorted(set(s.dropna().astype(str)))),
            emodnet_feature_ids_static=(
                "feature_origin",
                lambda s: sorted(set(v for v in s if isinstance(v, str))),
            ),
            emodnet_first_record_ts_static=("event_ts", "min"),
            emodnet_last_record_ts_static=("event_ts", "max"),
        )
        .reset_index()
    )
    return agg


def _nearest_week(ts: pd.Timestamp, canonical_weeks: list[pd.Timestamp]) -> pd.Timestamp:
    if not canonical_weeks:
        return ts
    return min(canonical_weeks, key=lambda w: abs((ts - w).total_seconds()))


def _extract_helcom_bbox(payload_obj: dict[str, Any]) -> list[float] | None:
    bbox = payload_obj.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        try:
            return [float(v) for v in bbox[:4]]
        except Exception:  # noqa: BLE001
            return None
    return None


def _extract_helcom_titles(payload_obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    title = payload_obj.get("title") or payload_obj.get("record_title")
    if isinstance(title, str) and title.strip():
        out.append(title.strip())
    return out


def helcom_spatial_panel_aggregate(
    df: pd.DataFrame,
    canonical_weeks: list[pd.Timestamp],
    canonical_grid_ids: list[str],
) -> pd.DataFrame:
    """Intersect every HELCOM record's bounding box with the weekly grid panel.

    Prior versions joined HELCOM records only on the single grid cell
    containing the bbox-centroid. HELCOM records describe pan-Baltic
    datasets (fishing intensity, biodiversity, radionuclides, …) whose
    bounding boxes span the full study AOI, so the correct join is a
    spatial within-bbox test. The temporal axis is broadcast across every
    canonical week because HELCOM catalog records describe static reference
    layers (or, when dates exist, multi-year aggregates) that apply to the
    whole study period.
    """
    base_cols = [
        "week_start_utc",
        "grid_cell_id",
        "helcom_record_count",
        "helcom_raw_refs",
        "helcom_feature_ids",
        "helcom_titles",
        "helcom_first_record_ts",
        "helcom_last_record_ts",
    ]
    if df.empty or not canonical_weeks or not canonical_grid_ids:
        return pd.DataFrame(columns=base_cols)

    payload_objs = df["payload"].apply(parse_payload)
    df = df.copy()
    df["_bbox"] = payload_objs.apply(_extract_helcom_bbox)
    df["_titles"] = payload_objs.apply(_extract_helcom_titles)
    df["_feature_origin"] = payload_objs.apply(_extract_feature_origin)

    # Precompute grid centroids once.
    centroids = {
        gid: grid_centroid_from_id(gid, GRID_RES_DEG) for gid in canonical_grid_ids
    }
    centroids = {gid: v for gid, v in centroids.items() if v[0] is not None}
    if not centroids:
        return pd.DataFrame(columns=base_cols)

    centroid_lat = np.array([v[0] for v in centroids.values()])
    centroid_lon = np.array([v[1] for v in centroids.values()])
    centroid_ids = np.array(list(centroids.keys()))

    per_record_grids: list[tuple[str, str, str, list[str], pd.Timestamp]] = []
    for _, row in df.iterrows():
        bbox = row["_bbox"]
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        minx, miny, maxx, maxy = bbox[:4]
        mask = (
            (centroid_lon >= minx)
            & (centroid_lon <= maxx)
            & (centroid_lat >= miny)
            & (centroid_lat <= maxy)
        )
        if not mask.any():
            continue
        rec_id = str(row.get("raw_record_ref", ""))
        feat_id = row.get("_feature_origin") or row.get("grid_id") or rec_id
        titles = row["_titles"] if isinstance(row["_titles"], list) else []
        ts = row.get("event_ts", pd.NaT)
        for gid in centroid_ids[mask].tolist():
            per_record_grids.append((gid, rec_id, str(feat_id), titles, ts))

    if not per_record_grids:
        return pd.DataFrame(columns=base_cols)

    grid_to_records: dict[str, dict[str, Any]] = {}
    for gid, rec_id, feat_id, titles, ts in per_record_grids:
        slot = grid_to_records.setdefault(
            gid,
            {"raw_refs": set(), "feature_ids": set(), "titles": set(), "ts_list": []},
        )
        if rec_id:
            slot["raw_refs"].add(rec_id)
        if feat_id:
            slot["feature_ids"].add(feat_id)
        for t in titles:
            slot["titles"].add(t)
        if pd.notna(ts):
            slot["ts_list"].append(ts)

    grid_records: list[dict[str, Any]] = []
    for gid, slot in grid_to_records.items():
        ts_values = slot["ts_list"]
        grid_records.append(
            {
                "grid_cell_id": gid,
                "helcom_record_count": len(slot["raw_refs"]),
                "helcom_raw_refs": sorted(slot["raw_refs"]),
                "helcom_feature_ids": sorted(slot["feature_ids"]),
                "helcom_titles": sorted(slot["titles"]),
                "helcom_first_record_ts": min(ts_values) if ts_values else pd.NaT,
                "helcom_last_record_ts": max(ts_values) if ts_values else pd.NaT,
            }
        )
    grid_df = pd.DataFrame(grid_records)
    if grid_df.empty:
        return pd.DataFrame(columns=base_cols)

    # Broadcast over every canonical week. HELCOM catalog records describe
    # reference layers valid for the whole study window.
    weeks_df = pd.DataFrame({"week_start_utc": canonical_weeks})
    grid_df["_key"] = 1
    weeks_df["_key"] = 1
    out = weeks_df.merge(grid_df, on="_key").drop(columns="_key")
    return out[base_cols]


# Back-compat alias retained for anything that still imports the old name.
helcom_nearest_week_aggregate = helcom_spatial_panel_aggregate


def build_master_dataset(project_root: Path) -> pd.DataFrame:
    sentinel_raw = load_source_records(project_root, "sentinel")
    emodnet_raw = load_source_records(project_root, "emodnet")
    helcom_raw = load_source_records(project_root, "helcom")
    spectral_files = sorted((project_root / "data" / "processed" / "sentinel" / "spectral_weekly").rglob("*.parquet"))
    sentinel_spectral = pd.concat([pd.read_parquet(p) for p in spectral_files], ignore_index=True) if spectral_files else pd.DataFrame()

    if not sentinel_spectral.empty:
        sentinel_spectral["week_start_utc"] = pd.to_datetime(
            sentinel_spectral["week_start_utc"], errors="coerce", utc=True
        )
        sentinel_spectral["sentinel_observation_count"] = pd.to_numeric(
            sentinel_spectral.get("sentinel_observation_count"), errors="coerce"
        )
        if "sentinel_raw_stats_payload" in sentinel_spectral.columns:
            derived_counts = sentinel_spectral["sentinel_raw_stats_payload"].apply(
                _extract_obs_count_from_stats_payload
            )
            sentinel_spectral["sentinel_observation_count"] = sentinel_spectral[
                "sentinel_observation_count"
            ].fillna(derived_counts)
        sentinel_weeks = sorted(sentinel_spectral["week_start_utc"].dropna().unique().tolist())
    else:
        sentinel_raw["week_start_utc"] = derive_week_start(sentinel_raw["event_ts"])
        sentinel_weeks = sorted(sentinel_raw["week_start_utc"].dropna().unique().tolist())
    if not sentinel_weeks:
        # Fallback to all temporal sources if Sentinel absent.
        tmp = pd.concat([helcom_raw["event_ts"]], ignore_index=True) if not helcom_raw.empty else pd.Series(dtype="datetime64[ns, UTC]")
        all_weeks = sorted(derive_week_start(tmp).dropna().unique().tolist()) if len(tmp) else []
    else:
        all_weeks = sentinel_weeks

    if not all_weeks:
        # Last-resort: week of current UTC time.
        now = pd.Timestamp.utcnow().floor("D")
        all_weeks = [now - pd.to_timedelta(now.weekday(), unit="D")]
    else:
        start = min(all_weeks)
        end = max(all_weeks)
        all_weeks = list(pd.date_range(start=start, end=end, freq="7D", tz="UTC"))

    if not sentinel_spectral.empty:
        sentinel_spectral["grid_cell_id"] = sentinel_spectral["grid_cell_id"].astype(str)
        grid_ids = sorted(
            set(sentinel_spectral["grid_cell_id"].tolist())
            | set(emodnet_raw["grid_cell_id"].tolist())
            | set(helcom_raw["grid_cell_id"].tolist())
        )
    else:
        grid_ids = sorted(
            set(sentinel_raw["grid_cell_id"].tolist())
            | set(emodnet_raw["grid_cell_id"].tolist())
            | set(helcom_raw["grid_cell_id"].tolist())
        )
    panel_index = pd.MultiIndex.from_product(
        [all_weeks, grid_ids], names=["week_start_utc", "grid_cell_id"]
    )
    master = panel_index.to_frame(index=False)

    if not sentinel_spectral.empty:
        sentinel_weekly = (
            sentinel_spectral.groupby(["week_start_utc", "grid_cell_id"], dropna=False)
            .agg(
                sentinel_observation_count=("sentinel_observation_count", "sum"),
                sentinel_ndvi_mean=("sentinel_ndvi_mean", "mean"),
                sentinel_ndvi_median=("sentinel_ndvi_median", "median"),
                sentinel_ndwi_mean=("sentinel_ndwi_mean", "mean"),
                sentinel_ndwi_median=("sentinel_ndwi_median", "median"),
                sentinel_evi_mean=("sentinel_evi_mean", "mean"),
                sentinel_evi_median=("sentinel_evi_median", "median"),
                sentinel_ndti_mean=("sentinel_ndti_mean", "mean"),
                sentinel_ndti_median=("sentinel_ndti_median", "median"),
                sentinel_raw_refs=(
                    "raw_record_ref",
                    lambda s: sorted(set(v for v in s.dropna().astype(str))),
                ),
            )
            .reset_index()
        )
        sentinel_weekly["sentinel_feature_ids"] = [[] for _ in range(len(sentinel_weekly))]
        sentinel_weekly["sentinel_first_record_ts"] = sentinel_weekly["week_start_utc"]
        sentinel_weekly["sentinel_last_record_ts"] = sentinel_weekly["week_start_utc"]
    else:
        sentinel_weekly = sentinel_weekly_aggregate(sentinel_raw)
    helcom_weekly = helcom_spatial_panel_aggregate(helcom_raw, all_weeks, grid_ids)
    emodnet_static = emodnet_static_aggregate(emodnet_raw)

    master = master.merge(sentinel_weekly, on=["week_start_utc", "grid_cell_id"], how="left")
    master = master.merge(helcom_weekly, on=["week_start_utc", "grid_cell_id"], how="left")
    master = master.merge(emodnet_static, on=["grid_cell_id"], how="left")

    # EMODnet is static and replicated across all weeks by merge on grid_cell_id only.
    master["sentinel_observation_count"] = master["sentinel_observation_count"].fillna(0).astype(int)
    master["helcom_record_count"] = master["helcom_record_count"].fillna(0).astype(int)
    master["emodnet_record_count_static"] = (
        master["emodnet_record_count_static"].fillna(0).astype(int)
    )
    master["emodnet_record_count"] = master["emodnet_record_count_static"]
    master["sentinel_record_count"] = master["sentinel_observation_count"]

    master["grid_res_deg"] = GRID_RES_DEG
    master["grid_centroid_lat"] = master["grid_cell_id"].apply(lambda g: grid_centroid_from_id(g)[0])
    master["grid_centroid_lon"] = master["grid_cell_id"].apply(lambda g: grid_centroid_from_id(g)[1])
    master["has_sentinel"] = master["sentinel_record_count"].gt(0)
    master["has_emodnet"] = master["emodnet_record_count"].gt(0)
    master["has_helcom"] = master["helcom_record_count"].gt(0)

    def _count_value(v: Any) -> int:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0
        return int(v)

    def _origin_json(row: pd.Series) -> str:
        data = {
            "sentinel": {
                "record_count": _count_value(row.get("sentinel_record_count")),
                "raw_refs": row.get("sentinel_raw_refs")
                if isinstance(row.get("sentinel_raw_refs"), list)
                else [],
                "feature_ids": row.get("sentinel_feature_ids")
                if isinstance(row.get("sentinel_feature_ids"), list)
                else [],
                "week_window": [str(row.get("sentinel_first_record_ts")), str(row.get("sentinel_last_record_ts"))],
            },
            "emodnet": {
                "record_count_static": _count_value(row.get("emodnet_record_count")),
                "raw_refs": row.get("emodnet_raw_refs_static")
                if isinstance(row.get("emodnet_raw_refs_static"), list)
                else [],
                "feature_ids": row.get("emodnet_feature_ids_static")
                if isinstance(row.get("emodnet_feature_ids_static"), list)
                else [],
            },
            "helcom": {
                "record_count": _count_value(row.get("helcom_record_count")),
                "raw_refs": row.get("helcom_raw_refs") if isinstance(row.get("helcom_raw_refs"), list) else [],
                "feature_ids": row.get("helcom_feature_ids")
                if isinstance(row.get("helcom_feature_ids"), list)
                else [],
                "assigned_week": str(row.get("week_start_utc")),
            },
        }
        return json.dumps(data, default=str)

    master["provenance_json"] = master.apply(_origin_json, axis=1)
    master = master.sort_values(["week_start_utc", "grid_cell_id"]).reset_index(drop=True)
    return master


def write_schema_doc(path: Path) -> None:
    text = """# Master Dataset Schema

`data/master_dataset.parquet` is a weekly panel table with one row per (`grid_cell_id`, `week_start_utc`).

## Grain
- One row per `week_start_utc` x `grid_cell_id` (balanced panel)
- Weekly bucket uses Monday start (UTC)
- Fixed geographic grid resolution: `0.1` degrees
- Same set of grid cells repeats across all canonical weeks (consistent row count per grid cell over time).

## Core Keys
- `week_start_utc` (timestamp, UTC)
- `grid_cell_id` (string, fixed-grid ID)
- `grid_res_deg` (float)
- `grid_centroid_lat` (float, nullable)
- `grid_centroid_lon` (float, nullable)

## Source Presence Flags
- `has_sentinel` (bool)
- `has_emodnet` (bool)
- `has_helcom` (bool)

## Sentinel Weekly Aggregates (Temporal)
- `sentinel_observation_count` (int, source observation count in that week+grid)
- `sentinel_record_count` (alias for panel compatibility)
- `sentinel_datasets` (list[str])
- `sentinel_raw_refs` (list[str])
- `sentinel_feature_ids` (list[str])
- `sentinel_first_record_ts` (timestamp, UTC)
- `sentinel_last_record_ts` (timestamp, UTC)
- `sentinel_<numeric_property>_mean` (float, per week+grid)
- `sentinel_<numeric_property>_median` (float, per week+grid)
  - Example currently available from Sentinel catalog properties: `eo_cloud_cover`.
  - Band-level means/medians are included automatically if band numeric properties are present in ingested payloads.

## EMODnet Static Features (Replicated Across Weeks)
- `emodnet_record_count_static` (int per grid from full EMODnet source)
- `emodnet_raw_refs_static` (list[str])
- `emodnet_feature_ids_static` (list[str])
- `emodnet_first_record_ts_static` (timestamp, UTC)
- `emodnet_last_record_ts_static` (timestamp, UTC)
- `emodnet_record_count` (replicated static count on each week row)

## HELCOM Weekly Aggregates (Spatial-Intersect Panel)
- `helcom_record_count` (int, number of HELCOM reference datasets whose bounding
  polygon contains this grid cell)
- `helcom_raw_refs` (list[str])
- `helcom_feature_ids` (list[str])
- `helcom_titles` (list[str], ISO-19139 dataset titles covering this cell)
- `helcom_first_record_ts` (timestamp, UTC)
- `helcom_last_record_ts` (timestamp, UTC)
- HELCOM catalog records describe static pan-Baltic reference layers. Each
  record's bounding box is intersected with the study grid; every cell whose
  centroid lies inside the bbox is flagged `has_helcom=True` and the fields
  above are attached. The record set is broadcast across every canonical week
  because the catalog layers are not week-specific.

## Provenance
- `provenance_json` (json string)
  - Per-source record counts
  - Raw request signatures (`raw_refs`)
  - Feature identifiers captured from source payloads

## Temporal Consistency Rules
- No mixing of granularities:
  - Sentinel -> weekly aggregated
  - HELCOM -> static bbox-polygon intersection replicated across weeks
  - EMODnet -> static replicated across weeks
- Canonical weeks are derived from Sentinel observation weeks (or available temporal sources fallback).

## Notes
- No feature engineering or model targets are created here.
- Records without valid spatial coordinates are excluded from panel construction (no inferred spatial mapping).
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    master = build_master_dataset(project_root)
    out = project_root / "data" / "master_dataset.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out, index=False)

    schema_doc = project_root / "docs" / "master_dataset_schema.md"
    write_schema_doc(schema_doc)
    print(f"[OK] Master dataset written: {out}")
    print(f"[OK] Schema documentation written: {schema_doc}")
    print(f"[INFO] Rows: {len(master)}")


if __name__ == "__main__":
    main()

