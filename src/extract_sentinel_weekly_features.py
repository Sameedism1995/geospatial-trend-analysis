from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests


TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
STATS_URL = "https://services.sentinel-hub.com/api/v1/statistics"


def get_token() -> str:
    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing SENTINELHUB_CLIENT_ID / SENTINELHUB_CLIENT_SECRET for Sentinel feature extraction."
        )
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("OAuth token missing in Sentinel Hub response.")
    return token


def generate_grid_cells(min_lon: float, min_lat: float, max_lon: float, max_lat: float, res: float) -> list[dict]:
    cells = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            cell_min_lon = lon
            cell_min_lat = lat
            cell_max_lon = min(lon + res, max_lon)
            cell_max_lat = min(lat + res, max_lat)
            row = int((cell_min_lat + 90.0) // res)
            col = int((cell_min_lon + 180.0) // res)
            grid_id = f"g{res:.3f}_r{row}_c{col}"
            cells.append(
                {
                    "grid_cell_id": grid_id,
                    "bbox": [cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat],
                    "centroid_lat": (cell_min_lat + cell_max_lat) / 2.0,
                    "centroid_lon": (cell_min_lon + cell_max_lon) / 2.0,
                }
            )
            lon += res
        lat += res
    return cells


def grid_id_to_bbox(grid_id: str, res: float) -> list[float] | None:
    try:
        parts = grid_id.split("_")
        row = int(parts[1][1:])
        col = int(parts[2][1:])
        min_lat = row * res - 90.0
        min_lon = col * res - 180.0
        max_lat = min_lat + res
        max_lon = min_lon + res
        return [min_lon, min_lat, max_lon, max_lat]
    except Exception:  # noqa: BLE001
        return None


def load_existing_panel_grids(project_root: Path) -> tuple[list[dict], float] | tuple[None, None]:
    panel = project_root / "data" / "master_dataset.parquet"
    if not panel.exists():
        return (None, None)
    try:
        df = pd.read_parquet(panel, columns=["grid_cell_id", "grid_res_deg"]).dropna()
        if df.empty:
            return (None, None)
        res = float(df["grid_res_deg"].iloc[0])
        grids = []
        for gid in sorted(df["grid_cell_id"].astype(str).unique()):
            bbox = grid_id_to_bbox(gid, res)
            if bbox is None:
                continue
            grids.append(
                {
                    "grid_cell_id": gid,
                    "bbox": bbox,
                    "centroid_lat": (bbox[1] + bbox[3]) / 2.0,
                    "centroid_lon": (bbox[0] + bbox[2]) / 2.0,
                }
            )
        return (grids, res)
    except Exception:  # noqa: BLE001
        return (None, None)


def statistics_request(
    token: str,
    bbox: list[float],
    time_from: str,
    time_to: str,
    cloud_pct: int,
) -> dict:
    evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "dataMask"]
    }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
      { id: "evi", bands: 1, sampleType: "FLOAT32" },
      { id: "ndti", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(s) {
  let ndvi = index(s.B08, s.B04);
  let ndwi = index(s.B03, s.B08);
  let ndti = index(s.B04, s.B03);
  let evi = 2.5 * ((s.B08 - s.B04) / ((s.B08 + 6.0 * s.B04 - 7.5 * s.B02) + 1.0));
  return {
    ndvi: [ndvi],
    ndwi: [ndwi],
    evi: [evi],
    ndti: [ndti],
    dataMask: [s.dataMask]
  };
}
"""
    body = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": time_from, "to": time_to},
                        "maxCloudCoverage": cloud_pct,
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {"from": time_from, "to": time_to},
            "aggregationInterval": {"of": "P1W"},
            "width": 64,
            "height": 64,
            "evalscript": evalscript,
        },
        "calculations": {
            "ndvi": {"statistics": {"default": {}, "percentiles": {"k": [50]}}},
            "ndwi": {"statistics": {"default": {}, "percentiles": {"k": [50]}}},
            "evi": {"statistics": {"default": {}, "percentiles": {"k": [50]}}},
            "ndti": {"statistics": {"default": {}, "percentiles": {"k": [50]}}},
            "dataMask": {"statistics": {"default": {}}},
        },
    }
    r = requests.post(
        STATS_URL,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def parse_weekly_stats(grid_cell_id: str, stats_payload: dict) -> list[dict]:
    out: list[dict] = []
    for block in stats_payload.get("data", []):
        interval = block.get("interval", {})
        outputs = block.get("outputs", {})
        ndvi = outputs.get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {})
        ndwi = outputs.get("ndwi", {}).get("bands", {}).get("B0", {}).get("stats", {})
        evi = outputs.get("evi", {}).get("bands", {}).get("B0", {}).get("stats", {})
        ndti = outputs.get("ndti", {}).get("bands", {}).get("B0", {}).get("stats", {})
        mask = outputs.get("dataMask", {}).get("bands", {}).get("B0", {}).get("stats", {})

        sample_count = mask.get("sampleCount")
        no_data_count = mask.get("noDataCount")
        valid_obs = None
        if sample_count is not None and no_data_count is not None:
            valid_obs = int(sample_count - no_data_count)

        out.append(
            {
                "grid_cell_id": grid_cell_id,
                "week_start_utc": interval.get("from"),
                "week_end_utc": interval.get("to"),
                "sentinel_observation_count": valid_obs,
                "sentinel_ndvi_mean": ndvi.get("mean"),
                "sentinel_ndvi_median": ndvi.get("percentiles", {}).get("50.0"),
                "sentinel_ndwi_mean": ndwi.get("mean"),
                "sentinel_ndwi_median": ndwi.get("percentiles", {}).get("50.0"),
                "sentinel_evi_mean": evi.get("mean"),
                "sentinel_evi_median": evi.get("percentiles", {}).get("50.0"),
                "sentinel_ndti_mean": ndti.get("mean"),
                "sentinel_ndti_median": ndti.get("percentiles", {}).get("50.0"),
                "sentinel_raw_stats_payload": json.dumps(block, default=str),
            }
        )
    return out


def main() -> None:
    # 12+ month window
    aoi_bbox = [19.5, 59.7, 22.5, 60.7]
    time_from = "2023-01-01T00:00:00Z"
    time_to = "2023-12-31T23:59:59Z"
    grid_res = 0.1
    cloud_pct = 30

    project_root = Path(__file__).resolve().parent.parent
    token = get_token()
    cells = generate_grid_cells(*aoi_bbox, grid_res)
    print(f"[RUN] Sentinel weekly stats for generated AOI grid cells: {len(cells)}", flush=True)

    rows: list[dict] = []
    failures = 0
    failure_examples: list[str] = []
    for i, cell in enumerate(cells, start=1):
        try:
            payload = statistics_request(token, cell["bbox"], time_from, time_to, cloud_pct)
            stats_rows = parse_weekly_stats(cell["grid_cell_id"], payload)
            for r in stats_rows:
                r["grid_centroid_lat"] = cell["centroid_lat"]
                r["grid_centroid_lon"] = cell["centroid_lon"]
                r["source"] = "sentinel"
                r["dataset"] = "spectral_weekly"
                r["record_timestamp_utc"] = r["week_start_utc"]
                r["ingested_at_utc"] = datetime.now(UTC).isoformat()
                r["latitude"] = cell["centroid_lat"]
                r["longitude"] = cell["centroid_lon"]
                r["geometry_wkt"] = None
                r["crs"] = "EPSG:4326"
                r["grid_id"] = cell["grid_cell_id"]
                r["raw_record_ref"] = f"sentinel_stats_{cell['grid_cell_id']}_{time_from}_{time_to}_{cloud_pct}"
            rows.extend(stats_rows)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            if len(failure_examples) < 5:
                failure_examples.append(f"{cell['grid_cell_id']}: {exc}")
        if i % 25 == 0:
            print(f"[INFO] Processed {i}/{len(cells)} cells; failures={failures}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        if failure_examples:
            print("[ERROR] Sample failures:", flush=True)
            for e in failure_examples:
                print(f"  - {e}", flush=True)
        raise RuntimeError("No sentinel weekly spectral rows generated.")
    numeric_cols = [
        "sentinel_observation_count",
        "sentinel_ndvi_mean",
        "sentinel_ndvi_median",
        "sentinel_ndwi_mean",
        "sentinel_ndwi_median",
        "sentinel_evi_mean",
        "sentinel_evi_median",
        "sentinel_ndti_mean",
        "sentinel_ndti_median",
        "grid_centroid_lat",
        "grid_centroid_lon",
        "latitude",
        "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["week_start_utc", "grid_cell_id"]).reset_index(drop=True)

    out_dir = (
        project_root
        / "data"
        / "processed"
        / "sentinel"
        / "spectral_weekly"
        / datetime.now(UTC).date().isoformat()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "spectral_weekly.parquet"
    df.to_parquet(out_file, index=False)
    print(f"[DONE] Wrote sentinel weekly spectral features: {out_file}", flush=True)
    print(f"[INFO] Rows={len(df)} failures={failures}", flush=True)


if __name__ == "__main__":
    main()

