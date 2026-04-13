from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


FILENAME_PATTERN = re.compile(r"vesseldensity_(\d{2})_(\d{4})\.tif$", re.IGNORECASE)


def parse_tif_metadata(path: Path) -> tuple[str, int]:
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        return ("unknown", -1)
    vessel_code = match.group(1)
    year = int(match.group(2))
    return (vessel_code, year)


def summarize_tif(path: Path) -> dict:
    vessel_code, year = parse_tif_metadata(path)
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        valid = arr.compressed()
        if valid.size == 0:
            return {
                "file": str(path),
                "vessel_code": vessel_code,
                "year": year,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "pixel_count_total": src.width * src.height,
                "pixel_count_valid": 0,
                "min_density": np.nan,
                "q25_density": np.nan,
                "median_density": np.nan,
                "mean_density": np.nan,
                "q75_density": np.nan,
                "max_density": np.nan,
                "std_density": np.nan,
            }

        return {
            "file": str(path),
            "vessel_code": vessel_code,
            "year": year,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "pixel_count_total": src.width * src.height,
            "pixel_count_valid": int(valid.size),
            "min_density": float(np.nanmin(valid)),
            "q25_density": float(np.nanpercentile(valid, 25)),
            "median_density": float(np.nanmedian(valid)),
            "mean_density": float(np.nanmean(valid)),
            "q75_density": float(np.nanpercentile(valid, 75)),
            "max_density": float(np.nanmax(valid)),
            "std_density": float(np.nanstd(valid)),
        }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    extracted_dir = root / "data" / "downloads" / "emodnet_extracted"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "outputs" / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(extracted_dir.rglob("*.tif"))
    if not tif_files:
        print(f"[STOP] No TIFF files found in {extracted_dir}")
        return

    print(f"[RUN] Summarizing {len(tif_files)} EMODnet TIFF files")
    rows = [summarize_tif(p) for p in tif_files]
    df = pd.DataFrame(rows).sort_values(["vessel_code", "year", "file"])

    # Thesis-aligned exposure variables
    df["mean_density_log"] = np.log1p(df["mean_density"])
    q90 = df["mean_density_log"].quantile(0.90)
    q75 = df["mean_density_log"].quantile(0.75)
    q25 = df["mean_density_log"].quantile(0.25)
    df["sea_lane_flag"] = (df["mean_density_log"] >= q90).astype(int)
    df["traffic_group_25_75"] = np.where(
        df["mean_density_log"] >= q75,
        "high",
        np.where(df["mean_density_log"] <= q25, "low", "mid"),
    )
    df["traffic_category_q3"] = pd.qcut(
        df["mean_density_log"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )

    # Save detailed and compact outputs
    detailed_path = processed_dir / "emodnet_tiff_summary.csv"
    compact_path = reports_dir / "emodnet_vessel_density_overview.csv"
    df.to_csv(detailed_path, index=False)

    compact_cols = [
        "vessel_code",
        "year",
        "pixel_count_valid",
        "mean_density",
        "mean_density_log",
        "median_density",
        "q75_density",
        "sea_lane_flag",
        "traffic_group_25_75",
        "traffic_category_q3",
    ]
    df[compact_cols].to_csv(compact_path, index=False)

    print(f"[OK] Saved detailed summary: {detailed_path}")
    print(f"[OK] Saved compact report: {compact_path}")
    print(f"[INFO] sea-lane threshold (q90, log density): {q90:.6f}")
    print("[DONE] EMODnet processing completed.")


if __name__ == "__main__":
    main()
