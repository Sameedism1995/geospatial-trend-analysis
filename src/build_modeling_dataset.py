"""
Build leakage-aware modeling dataset: predict delta NDTI (t+1 − t) from lags ≤ t.

Does not train models, impute missing values, or scale features.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

FILENAME_PATTERN = re.compile(r"vesseldensity_(\d{2})_(\d{4})\.tif$", re.IGNORECASE)

SPECTRAL = ("ndvi", "ndwi", "evi", "ndti")
COL = {s: f"sentinel_{s}_mean" for s in SPECTRAL}
OBS_COL = "sentinel_observation_count"


def grid_centroid_from_id(grid_id: str, res_deg: float = 0.1) -> tuple[float | None, float | None]:
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


def _finite_sample_from_tif(path: Path, lon: float, lat: float) -> float | None:
    """Sample one band at WGS84 lon/lat; return None if nodata or outside."""
    with rasterio.open(path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [float(lon)], [float(lat)])
        x, y = float(xs[0]), float(ys[0])
        val = float(next(src.sample([(x, y)]))[0])
        nd = src.nodata
        if nd is not None:
            if np.isnan(val) or (not np.isfinite(val)):
                return None
            if val == nd or abs(val - nd) < 1e30:
                return None
        elif not np.isfinite(val):
            return None
        return val


def vessel_density_sum_grid_year(
    grid_ids: list[str],
    centroids: dict[str, tuple[float, float]],
    year: int,
    tif_root: Path,
) -> dict[str, float]:
    """Sum sampled density across all vesseldensity_*_{year}.tif layers (raster-derived)."""
    paths = sorted(tif_root.rglob(f"vesseldensity_*_{year}.tif"))
    paths = [p for p in paths if FILENAME_PATTERN.search(p.name)]
    totals: dict[str, float] = {g: 0.0 for g in grid_ids}
    any_valid = {g: False for g in grid_ids}
    for path in paths:
        for gid in grid_ids:
            lat, lon = centroids[gid]
            v = _finite_sample_from_tif(path, lon, lat)
            if v is None:
                continue
            totals[gid] += v
            any_valid[gid] = True
    return {g: (totals[g] if any_valid[g] else np.nan) for g in grid_ids}


def build_vessel_lookup(
    grid_ids: list[str],
    years: set[int],
    tif_root: Path,
) -> dict[tuple[str, int], float]:
    centroids: dict[str, tuple[float, float]] = {}
    for gid in grid_ids:
        lat, lon = grid_centroid_from_id(gid)
        if lat is None or lon is None:
            centroids[gid] = (np.nan, np.nan)
        else:
            centroids[gid] = (lat, lon)
    out: dict[tuple[str, int], float] = {}
    for y in sorted(years):
        dens = vessel_density_sum_grid_year(grid_ids, centroids, y, tif_root)
        for gid in grid_ids:
            out[(gid, y)] = dens[gid]
    return out


def add_lagged_spectral(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["grid_cell_id", "week_start_utc"]).copy()
    out = df
    for _name, col in COL.items():
        if col not in out.columns:
            out[col] = np.nan
        g = out.groupby("grid_cell_id", sort=False)[col]
        out[f"{col}_t"] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_t_minus_1"] = g.shift(1)
        out[f"{col}_t_minus_2"] = g.shift(2)
    if OBS_COL in out.columns:
        out[f"{OBS_COL}_t"] = pd.to_numeric(out[OBS_COL], errors="coerce").fillna(0).astype(int)
    else:
        out[f"{OBS_COL}_t"] = np.nan
    drop_raw = [c for c in list(COL.values()) + [OBS_COL] if c in out.columns]
    out = out.drop(columns=drop_raw)
    return out


def add_delta_ndti(df: pd.DataFrame) -> pd.DataFrame:
    """Target uses t and t+1 NDTI means only (no other t+1 features)."""
    df = df.sort_values(["grid_cell_id", "week_start_utc"]).copy()
    col_t = f"{COL['ndti']}_t"
    if col_t not in df.columns:
        df[col_t] = np.nan
    ndti = pd.to_numeric(df[col_t], errors="coerce")
    ndti_tp1 = df.groupby("grid_cell_id", sort=False)[col_t].shift(-1)
    both = ndti.notna() & ndti_tp1.notna()
    df["delta_ndti"] = (ndti_tp1 - ndti).where(both)
    df["has_valid_delta_ndti"] = both
    return df


def attach_vessel_density_lags(
    df: pd.DataFrame,
    vessel_lookup: dict[tuple[str, int], float],
) -> pd.DataFrame:
    w = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    w1 = w - pd.Timedelta(days=7)
    w2 = w - pd.Timedelta(days=14)
    gid = df["grid_cell_id"].astype(str)

    def col_series(years: pd.Series) -> pd.Series:
        out_vals: list[float] = []
        for g, yr in zip(gid.tolist(), years.tolist(), strict=True):
            if pd.isna(yr):
                out_vals.append(np.nan)
            else:
                out_vals.append(vessel_lookup.get((str(g), int(yr)), np.nan))
        return pd.Series(out_vals, index=df.index, dtype=float)

    out = df.copy()
    out["vessel_density_t"] = col_series(w.dt.year)
    out["vessel_density_t_minus_1"] = col_series(w1.dt.year)
    out["vessel_density_t_minus_2"] = col_series(w2.dt.year)
    return out


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    w = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    # ISO week 1–53 (thesis text often uses 1–52; document in schema)
    iso = w.dt.isocalendar()
    df = df.copy()
    df["week_of_year"] = iso.week.astype("Int64")
    woy = df["week_of_year"].astype(float)
    df["week_sin"] = np.sin(2 * np.pi * (woy - 1) / 53.0)
    df["week_cos"] = np.cos(2 * np.pi * (woy - 1) / 53.0)
    return df


def build_modeling_frame(
    master: pd.DataFrame,
    *,
    tif_root: Path | None,
) -> pd.DataFrame:
    master = master.copy()
    master["week_start_utc"] = pd.to_datetime(master["week_start_utc"], utc=True, errors="coerce")

    master = add_lagged_spectral(master)
    master = add_delta_ndti(master)

    # Vessel density from annual EMODnet rasters (calendar year of each week / lag week)
    if tif_root is not None and tif_root.is_dir():
        w = master["week_start_utc"]
        w1 = w - pd.Timedelta(days=7)
        w2 = w - pd.Timedelta(days=14)
        years = set(
            int(x)
            for s in (w.dt.year, w1.dt.year, w2.dt.year)
            for x in s.dropna().unique().tolist()
        )
        grids = sorted(master["grid_cell_id"].astype(str).unique())
        vessel_lookup = build_vessel_lookup(grids, years, tif_root)
        master = attach_vessel_density_lags(master, vessel_lookup)
    else:
        master["vessel_density_t"] = np.nan
        master["vessel_density_t_minus_1"] = np.nan
        master["vessel_density_t_minus_2"] = np.nan

    master = add_temporal_features(master)

    # Static geography (no leakage; replicated per row)
    for c in ("grid_res_deg", "grid_centroid_lat", "grid_centroid_lon"):
        if c not in master.columns:
            master[c] = np.nan

    wanted = modeling_columns()
    present = [c for c in wanted if c in master.columns]
    return master[present].copy()


def modeling_columns() -> list[str]:
    meta = [
        "grid_cell_id",
        "week_start_utc",
        "grid_res_deg",
        "grid_centroid_lat",
        "grid_centroid_lon",
    ]
    temporal = ["week_of_year", "week_sin", "week_cos"]
    vessel = ["vessel_density_t", "vessel_density_t_minus_1", "vessel_density_t_minus_2"]
    spectral: list[str] = []
    for s in SPECTRAL:
        base = COL[s]
        spectral.extend([f"{base}_t", f"{base}_t_minus_1", f"{base}_t_minus_2"])
    obs = [f"{OBS_COL}_t"]
    target = ["delta_ndti", "has_valid_delta_ndti"]
    flags = ["has_sentinel", "has_emodnet", "has_helcom"]
    return meta + temporal + vessel + spectral + obs + target + flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Build modeling dataset (delta NDTI target).")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/master_dataset.parquet"),
        help="Master panel parquet path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/modeling_dataset.parquet"),
        help="Output modeling parquet path",
    )
    parser.add_argument(
        "--tif-root",
        type=Path,
        default=Path("data/downloads/emodnet_extracted"),
        help="Root folder containing EMODnet vesseldensity GeoTIFFs (annual layers)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    tif_root = args.tif_root if args.tif_root.is_absolute() else root / args.tif_root

    if not input_path.exists():
        raise SystemExit(f"Missing input: {input_path}")

    tif_ok = tif_root.is_dir() and any(tif_root.rglob("vesseldensity_*.tif"))
    master = pd.read_parquet(input_path)
    out = build_modeling_frame(master, tif_root=tif_root if tif_ok else None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows, {len(out.columns)} columns)")
    if not tif_ok:
        print(
            f"[WARN] No EMODnet vesseldensity TIFFs under {tif_root}; "
            "vessel_density_* columns are NaN. Place extracted rasters to populate."
        )


if __name__ == "__main__":
    main()
