from __future__ import annotations

import json
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_raw = root / "data" / "raw"
    data_downloads = root / "data" / "downloads"
    data_raw.mkdir(parents=True, exist_ok=True)
    data_downloads.mkdir(parents=True, exist_ok=True)
    return {"raw": data_raw, "downloads": data_downloads}


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sanitize_filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name or fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name


def download_file(url: str, output_path: Path, max_mb: int = 500) -> bool:
    try:
        head = requests.head(url, allow_redirects=True, timeout=60)
        if head.ok:
            size = head.headers.get("content-length")
            if size and int(size) > max_mb * 1024 * 1024:
                print(f"[SKIP] Too large (> {max_mb} MB): {url}")
                return False
    except Exception:
        # Some servers do not support HEAD correctly; continue with GET.
        pass

    try:
        with requests.get(url, stream=True, timeout=180, allow_redirects=True) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] Downloaded: {output_path.name}")
        return True
    except Exception as exc:
        print(f"[WARN] Download failed for {url}: {exc}")
        return False


def fetch_sentinel2_stac(data_dir: Path) -> None:
    """
    Pull Sentinel-2 STAC metadata subset from Planetary Computer (public STAC).
    """
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [10.0, 53.0, 30.0, 66.0],  # Baltic-focused bbox
        "datetime": "2023-01-01/2023-12-31",
        "limit": 200,
    }
    out_json = data_dir / "sentinel2_stac_items.json"
    out_csv = data_dir / "sentinel2_stac_items.csv"

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    save_json(out_json, data)

    features = data.get("features", [])
    rows: List[Dict[str, Any]] = []
    for f in features:
        props = f.get("properties", {})
        bbox = f.get("bbox", [None, None, None, None])
        rows.append(
            {
                "id": f.get("id"),
                "datetime": props.get("datetime"),
                "eo_cloud_cover": props.get("eo:cloud_cover"),
                "platform": props.get("platform"),
                "proj_epsg": props.get("proj:epsg"),
                "bbox_min_lon": bbox[0],
                "bbox_min_lat": bbox[1],
                "bbox_max_lon": bbox[2],
                "bbox_max_lat": bbox[3],
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Sentinel-2 STAC metadata saved: {out_csv}")


def fetch_emodnet_metadata(data_dir: Path) -> None:
    """
    Pull EMODnet vessel density metadata record.
    """
    record_id = "0f2f3ff1-30ef-49e1-96e7-8ca78d58a07c"
    url = f"https://ows.emodnet-humanactivities.eu/geonetwork/srv/api/records/{record_id}"
    out_json = data_dir / "emodnet_vessel_density_record.json"
    out_csv = data_dir / "emodnet_vessel_density_links.csv"

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    raw_text = resp.text
    (data_dir / "emodnet_vessel_density_record_raw.txt").write_text(
        raw_text, encoding="utf-8"
    )

    links: List[Dict[str, Any]] = []
    if "application/json" in content_type:
        data = resp.json()
        save_json(out_json, data)
        for link in data.get("link", []):
            links.append(
                {
                    "name": link.get("name"),
                    "description": link.get("description"),
                    "url": link.get("url"),
                    "protocol": link.get("protocol"),
                }
            )
    else:
        # Fallback for HTML/XML payloads
        urls = sorted(set(re.findall(r"https?://[^\s\"'<>]+", raw_text)))
        for u in urls:
            links.append({"name": None, "description": None, "url": u, "protocol": None})

    pd.DataFrame(links).to_csv(out_csv, index=False)
    print(f"[OK] EMODnet metadata/raw saved to {data_dir}")


def fetch_helcom_metadata(data_dir: Path) -> None:
    """
    Pull HELCOM metadata catalogue API listing (first page).
    """
    url = "https://metadata.helcom.fi/geonetwork/srv/api/records"
    params = {"from": 1, "to": 50}
    out_json = data_dir / "helcom_records_page1.json"
    out_csv = data_dir / "helcom_records_page1.csv"

    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    raw_text = resp.text
    (data_dir / "helcom_records_page1_raw.txt").write_text(raw_text, encoding="utf-8")

    rows = []
    if "application/json" in content_type:
        data = resp.json()
        save_json(out_json, data)
        records = data.get("metadata", []) if isinstance(data, dict) else []
        for rec in records:
            rows.append(
                {
                    "uuid": rec.get("geonet:info", {}).get("uuid")
                    if isinstance(rec.get("geonet:info"), dict)
                    else None,
                    "title": rec.get("title") if isinstance(rec, dict) else None,
                }
            )
    else:
        urls = sorted(set(re.findall(r"https?://[^\s\"'<>]+", raw_text)))
        for u in urls:
            rows.append({"uuid": None, "title": u})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] HELCOM metadata/raw saved to {data_dir}")


def download_emodnet_assets(raw_dir: Path, downloads_dir: Path) -> None:
    links_csv = raw_dir / "emodnet_vessel_density_links.csv"
    if not links_csv.exists():
        print("[INFO] EMODnet links CSV not found; skipping asset downloads.")
        return

    df = pd.read_csv(links_csv)
    if "url" not in df.columns:
        print("[INFO] EMODnet links CSV has no URL column; skipping.")
        return

    # Keep this bounded and reproducible: download avg monthly ZIPs first.
    urls = (
        df["url"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.contains(r"attachments/.+Avg\.zip$", regex=True)]
        .head(6)
        .tolist()
    )

    if not urls:
        print("[INFO] No EMODnet asset URLs matched filters.")
        return

    out_dir = downloads_dir / "emodnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for i, url in enumerate(urls, start=1):
        filename = sanitize_filename_from_url(url, f"emodnet_{i}.zip")
        ok = download_file(url, out_dir / filename, max_mb=700)
        if ok:
            downloaded += 1
    print(f"[DONE] EMODnet assets downloaded: {downloaded}/{len(urls)}")


def download_helcom_assets(raw_dir: Path, downloads_dir: Path) -> None:
    links_csv = raw_dir / "helcom_records_page1.csv"
    if not links_csv.exists():
        print("[INFO] HELCOM links CSV not found; skipping asset downloads.")
        return

    df = pd.read_csv(links_csv)
    if "title" not in df.columns:
        print("[INFO] HELCOM links CSV has no title column; skipping.")
        return

    urls = (
        df["title"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace("&amp;", "&", regex=False)
        .loc[lambda s: s.str.startswith("https://maps.helcom.fi/website/MADS/download/?id=")]
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    if not urls:
        print("[INFO] No HELCOM direct download links matched filters.")
        return

    out_dir = downloads_dir / "helcom"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for i, url in enumerate(urls, start=1):
        filename = f"helcom_dataset_{i}.bin"
        ok = download_file(url, out_dir / filename, max_mb=700)
        if ok:
            downloaded += 1
    print(f"[DONE] HELCOM assets downloaded: {downloaded}/{len(urls)}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    dirs = ensure_dirs(root)
    raw_dir = dirs["raw"]
    downloads_dir = dirs["downloads"]

    jobs = [
        ("Sentinel-2 STAC", fetch_sentinel2_stac),
        ("EMODnet vessel density", fetch_emodnet_metadata),
        ("HELCOM metadata", fetch_helcom_metadata),
    ]

    for name, fn in jobs:
        try:
            print(f"[RUN] {name}")
            fn(raw_dir)
        except Exception as exc:
            print(f"[WARN] {name} fetch failed: {exc}")

    try:
        print("[RUN] EMODnet asset downloads")
        download_emodnet_assets(raw_dir, downloads_dir)
    except Exception as exc:
        print(f"[WARN] EMODnet asset download step failed: {exc}")

    try:
        print("[RUN] HELCOM asset downloads")
        download_helcom_assets(raw_dir, downloads_dir)
    except Exception as exc:
        print(f"[WARN] HELCOM asset download step failed: {exc}")

    print("[DONE] Online API fetch completed.")


if __name__ == "__main__":
    main()
