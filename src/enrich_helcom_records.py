"""Offline enrichment of existing HELCOM standardized parquets.

The ISO-19139 metadata files fetched from HELCOM's GeoNetwork were parsed by
an older version of ``helcom_client.py`` that only captured ``title`` and
``bbox``. The raw XML was discarded before being written to
``data/raw/helcom/``, so we cannot recover per-record dateStamps offline.
However, every record already carries a valid bounding box and this script:

  1. builds a real WGS84 POLYGON WKT from the bbox and writes it into the
     ``geometry_wkt`` column (previously all NULL);
  2. expands the JSON ``payload`` into first-class columns (``title``,
     ``topic_category``, ``abstract``, ``keywords``, ``temporal_extent_*``)
     so the master-dataset builder can spatially intersect without parsing
     JSON per row;
  3. preserves the original payload under ``payload`` (unchanged) and the
     original schema-level columns (``source``, ``dataset``,
     ``ingested_at_utc``, ``raw_record_ref``);
  4. keeps the previous ``latitude``/``longitude`` centroid (used by the
     legacy grid-id join) and emits a polygon-aware version alongside.

Usage:

    python src/enrich_helcom_records.py

Writes in-place to every ``data/processed/helcom/**/*.parquet``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _polygon_wkt_from_bbox(bbox: list[float | None] | None) -> str | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        minx, miny, maxx, maxy = (float(v) for v in bbox[:4])
    except Exception:  # noqa: BLE001
        return None
    coords = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]
    coord_str = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in coords)
    return f"POLYGON (({coord_str}))"


def _parse_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    payloads = out.get("payload").apply(_parse_payload) if "payload" in out.columns else pd.Series([{}] * len(out))

    # Preserve bbox as a list column for downstream polygon intersection.
    out["bbox"] = payloads.apply(lambda p: p.get("bbox") if isinstance(p.get("bbox"), list) else None)

    geometry = payloads.apply(
        lambda p: p.get("geometry_wkt") or _polygon_wkt_from_bbox(p.get("bbox"))
    )
    if "geometry_wkt" in out.columns:
        out["geometry_wkt"] = out["geometry_wkt"].where(out["geometry_wkt"].notna(), geometry)
    else:
        out["geometry_wkt"] = geometry

    # First-class surface fields. When the client did not populate them (legacy
    # records), keep the columns but leave them NULL.
    for src_key, out_col in [
        ("title", "record_title"),
        ("topic_category", "topic_category"),
        ("organisation", "organisation"),
        ("abstract", "abstract"),
        ("purpose", "purpose"),
        ("lineage", "lineage"),
    ]:
        out[out_col] = payloads.apply(lambda p, k=src_key: p.get(k) if isinstance(p.get(k), str) else None)

    out["keywords_csv"] = payloads.apply(
        lambda p: ";".join(p.get("keywords", [])) if isinstance(p.get("keywords"), list) else None
    )

    out["record_timestamp_utc"] = out.get("record_timestamp_utc")
    # Fill record_timestamp_utc from payload if present.
    ts_from_payload = payloads.apply(lambda p: p.get("record_timestamp_utc"))
    out["record_timestamp_utc"] = out["record_timestamp_utc"].where(
        out["record_timestamp_utc"].notna(), ts_from_payload
    )

    temporal = payloads.apply(lambda p: p.get("temporal_extent") if isinstance(p.get("temporal_extent"), dict) else {})
    out["temporal_extent_begin_utc"] = temporal.apply(lambda t: t.get("begin") if isinstance(t, dict) else None)
    out["temporal_extent_end_utc"] = temporal.apply(lambda t: t.get("end") if isinstance(t, dict) else None)

    return out


def process_path(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    before_cols = set(df.columns)
    enriched = enrich_dataframe(df)
    enriched.to_parquet(path, index=False)
    new_cols = sorted(set(enriched.columns) - before_cols)
    return {
        "file": str(path),
        "rows": int(len(enriched)),
        "added_columns": new_cols,
        "geometry_wkt_non_null": int(enriched["geometry_wkt"].notna().sum()),
        "bbox_non_null": int(enriched["bbox"].notna().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/processed/helcom"),
        help="Root folder of standardized HELCOM parquets.",
    )
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else Path.cwd() / args.root
    files = sorted(root.rglob("*.parquet"))
    if not files:
        print(f"[HELCOM-ENRICH] No parquet files under {root}")
        return

    summary: list[dict[str, Any]] = []
    for path in files:
        try:
            summary.append(process_path(path))
        except Exception as exc:  # noqa: BLE001
            summary.append({"file": str(path), "error": str(exc)})

    total_rows = sum(s.get("rows", 0) for s in summary if "error" not in s)
    geom_rows = sum(s.get("geometry_wkt_non_null", 0) for s in summary if "error" not in s)
    print(f"[HELCOM-ENRICH] Processed {len(files)} file(s); rows={total_rows}; geometry_wkt populated={geom_rows}")
    for s in summary:
        if "error" in s:
            print(f"  [ERROR] {s['file']}: {s['error']}")
        else:
            print(f"  [OK] {Path(s['file']).name} rows={s['rows']} geom={s['geometry_wkt_non_null']} added={s['added_columns']}")


if __name__ == "__main__":
    main()
