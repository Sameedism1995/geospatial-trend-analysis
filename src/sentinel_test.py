from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import requests


TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
CATALOG_URL = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"


def utc_now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def get_access_token() -> str:
    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Sentinel Hub credentials. Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET."
        )

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("OAuth succeeded but access_token missing in response.")
    return token


def run_catalog_query(token: str, bbox: list[float], time_from: str, time_to: str) -> dict:
    body = {
        "bbox": bbox,
        "datetime": f"{time_from}/{time_to}",
        "collections": ["sentinel-2-l2a"],
        "limit": 5,
    }
    resp = requests.post(
        CATALOG_URL,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def download_small_raster(token: str, bbox: list[float], time_from: str, time_to: str) -> bytes:
    evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5];
}
"""
    body = {
        "input": {
            "bounds": {"bbox": bbox},
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"timeRange": {"from": time_from, "to": time_to}},
                }
            ],
        },
        "output": {"width": 64, "height": 64, "responses": [{"identifier": "default", "format": {"type": "image/png"}}]},
        "evalscript": evalscript,
    }
    resp = requests.post(
        PROCESS_URL,
        json=body,
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    resp.raise_for_status()
    if "image/png" not in resp.headers.get("content-type", ""):
        raise RuntimeError(f"Unexpected process response type: {resp.headers.get('content-type')}")
    return resp.content


def main() -> None:
    # Baltic-area minimal query window
    bbox = [20.8, 60.0, 21.1, 60.2]
    time_from = "2023-06-01T00:00:00Z"
    time_to = "2023-06-10T23:59:59Z"

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "sentinel_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = utc_now_tag()

    print("[RUN] Sentinel OAuth authentication")
    token = get_access_token()
    print("[OK] OAuth authentication successful")

    print("[RUN] Sentinel-2 catalog query")
    catalog = run_catalog_query(token, bbox, time_from, time_to)
    features = catalog.get("features", [])
    print(f"[OK] Catalog query success. Features returned: {len(features)}")
    if not features:
        raise RuntimeError("Catalog query returned zero features for test window.")

    feature_sample = features[0]
    metadata = {
        "queried_at_utc": datetime.now(UTC).isoformat(),
        "bbox": bbox,
        "time_from": time_from,
        "time_to": time_to,
        "feature_count": len(features),
        "first_feature_id": feature_sample.get("id"),
        "first_feature_datetime": feature_sample.get("properties", {}).get("datetime"),
    }

    print("[RUN] Sentinel process API small raster request")
    png_bytes = download_small_raster(token, bbox, time_from, time_to)
    print(f"[OK] Raster download success. Bytes: {len(png_bytes)}")

    meta_path = out_dir / f"sentinel_test_metadata_{tag}.json"
    catalog_path = out_dir / f"sentinel_test_catalog_{tag}.json"
    raster_path = out_dir / f"sentinel_test_preview_{tag}.png"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    raster_path.write_bytes(png_bytes)

    print("[DONE] Sentinel test completed")
    print(f"metadata_file: {meta_path}")
    print(f"catalog_file: {catalog_path}")
    print(f"raster_file: {raster_path}")


if __name__ == "__main__":
    main()

