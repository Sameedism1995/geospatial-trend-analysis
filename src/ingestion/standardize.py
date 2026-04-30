from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .base import ApiCallResult, utc_now_iso


TIME_KEYS = [
    "datetime",
    "timestamp",
    "time",
    "acquisition_time",
    "start_time",
    "end_time",
]


def _normalize_time(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return ts.astimezone(UTC).isoformat()
    except Exception:  # noqa: BLE001
        return None


def _extract_first(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _point_from_bbox(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    bbox = payload.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        minx, miny, maxx, maxy = bbox[:4]
        return (float(miny + maxy) / 2.0, float(minx + maxx) / 2.0)
    return (None, None)


def standardize_result(result: ApiCallResult) -> list[dict[str, Any]]:
    payload = result.response_payload
    ingested_at = utc_now_iso()
    rows: list[dict[str, Any]] = []

    if not result.success:
        return rows

    if isinstance(payload, dict) and isinstance(payload.get("layers"), list):
        iterable = payload.get("layers", [])
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        iterable = payload.get("records", [])
    elif isinstance(payload, dict) and isinstance(payload.get("features"), list):
        iterable = payload.get("features", [])
    elif isinstance(payload, dict) and isinstance(payload.get("metadata"), list):
        iterable = payload.get("metadata", [])
    elif isinstance(payload, list):
        iterable = payload
    else:
        iterable = [payload]

    for item in iterable:
        if not isinstance(item, dict):
            continue
        if (
            "bbox" in item
            and isinstance(item["bbox"], list)
            and len(item["bbox"]) >= 4
            and not isinstance(item.get("properties"), dict)
        ):
            try:
                minx, miny, maxx, maxy = item["bbox"][:4]
                lat = float(miny + maxy) / 2.0
                lon = float(minx + maxx) / 2.0
            except Exception:  # noqa: BLE001
                lat = lon = None
            # Only keep spatially mappable records.
            if lat is None or lon is None:
                continue
            # HELCOM-enriched payload may carry its own geometry_wkt (polygon
            # from bbox) and record_timestamp_utc (from ISO citation dates).
            # Fall through to None when the client did not enrich the record.
            geometry_wkt = item.get("geometry_wkt")
            record_ts = _normalize_time(item.get("record_timestamp_utc"))
            rows.append(
                {
                    "source": result.source,
                    "dataset": result.dataset,
                    "record_timestamp_utc": record_ts,
                    "ingested_at_utc": ingested_at,
                    "latitude": lat,
                    "longitude": lon,
                    "geometry_wkt": geometry_wkt,
                    "crs": item.get("crs") or "EPSG:4326",
                    "grid_id": item.get("name") or item.get("record_id") or item.get("title"),
                    "raw_record_ref": result.request_signature,
                    "payload": json.dumps(item, default=str),
                }
            )
            continue

        props = item.get("properties", {}) if isinstance(item.get("properties"), dict) else {}
        combined = {**item, **props}
        lat = _extract_first(combined, ["latitude", "lat", "y"])
        lon = _extract_first(combined, ["longitude", "lon", "lng", "x"])
        if lat is None or lon is None:
            bb_lat, bb_lon = _point_from_bbox(item)
            lat = lat if lat is not None else bb_lat
            lon = lon if lon is not None else bb_lon
        ts = None
        for key in TIME_KEYS:
            ts = _normalize_time(combined.get(key))
            if ts:
                break
        crs = None
        if isinstance(item.get("proj:epsg"), (str, int)):
            crs = f"EPSG:{item['proj:epsg']}"
        elif "crs" in item:
            crs = str(item["crs"])

        grid_id = _extract_first(combined, ["tile_id", "grid_id", "id"])
        geometry_wkt = None
        if isinstance(item.get("geometry"), dict):
            geometry_wkt = json.dumps(item["geometry"], default=str)

        rows.append(
            {
                "source": result.source,
                "dataset": result.dataset,
                "record_timestamp_utc": ts,
                "ingested_at_utc": ingested_at,
                "latitude": lat,
                "longitude": lon,
                "geometry_wkt": geometry_wkt,
                "crs": crs,
                "grid_id": grid_id,
                "raw_record_ref": result.request_signature,
                "payload": json.dumps(item, default=str),
            }
        )
    return rows

