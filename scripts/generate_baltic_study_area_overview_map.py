#!/usr/bin/env python3
"""
Minimal Baltic **study-area** overview: Natural Earth coastline + modelling lattice footprints.

Contrasts with ``generate_chapter_5_1_maritime_map.py`` (continuous vessel-density colouring).
Here cells are neutrally filled to show **spatial support** only.

Outputs (default):
  outputs/final_figures/fig_baltic_study_area_lattice_extent.{png,pdf}
"""

from __future__ import annotations

import re
import ssl
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon, box

ROOT = Path(__file__).resolve().parents[1]
# Prefer ML panel parquet (always present); ``features_ml_ready.parquet`` is also valid if larger.
MODELING = ROOT / "data" / "modeling_dataset.parquet"
FEATURES_FALLBACK = ROOT / "processed" / "features_ml_ready.parquet"
OUT_DIR = ROOT / "outputs" / "final_figures"
BASE = OUT_DIR / "fig_baltic_study_area_lattice_extent"

CACHE_DIR = ROOT / "processed" / "basemap_cache"
CACHE_ZIP = CACHE_DIR / "ne_110m_land.zip"
NE_LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"

BALTIC_LAT_RANGE = (53.8, 67.6)
BALTIC_LON_RANGE = (8.4, 31.2)

PORT_COORDS: dict[str, tuple[float, float]] = {
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
    "Naantali": (60.4669, 22.0258),
}

_CELL_RES_RE = re.compile(r"^g(?P<res>[\d.]+)_")


def _download_ne_land() -> bytes:
    for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(NE_LAND_URL, headers={"User-Agent": "geospatial-thesis/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                return r.read()
        except Exception:
            continue
    raise RuntimeError("Could not download Natural Earth 110m land (network).")


def load_land_gdf() -> gpd.GeoDataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_ZIP.is_file():
        CACHE_ZIP.write_bytes(_download_ne_land())
    return gpd.read_file(f"zip://{CACHE_ZIP}!ne_110m_land.shp")


def parse_cell_resolution_deg(grid_cell_id: str, default_deg: float = 0.1) -> float:
    m = _CELL_RES_RE.match(str(grid_cell_id))
    if not m:
        return default_deg
    try:
        return float(m.group("res"))
    except ValueError:
        return default_deg


def square_polygon(lon: float, lat: float, res_deg: float) -> Polygon:
    h = float(res_deg) / 2.0
    return box(float(lon) - h, float(lat) - h, float(lon) + h, float(lat) + h)


def main() -> None:
    src = MODELING if MODELING.is_file() else FEATURES_FALLBACK
    if not src.is_file():
        raise FileNotFoundError(f"No panel parquet at {MODELING} or {FEATURES_FALLBACK}")

    df = pd.read_parquet(src)
    lon_c, lat_c = "grid_centroid_lon", "grid_centroid_lat"
    if lon_c not in df.columns or lat_c not in df.columns:
        raise KeyError(f"Expected {lon_c}, {lat_c} in {src}")

    agg = df.groupby("grid_cell_id", sort=False)[[lon_c, lat_c]].first().reset_index()
    agg = agg.dropna(subset=[lon_c, lat_c])
    lo1, lo2 = BALTIC_LON_RANGE
    la1, la2 = BALTIC_LAT_RANGE
    m = agg[lon_c].between(lo1, lo2) & agg[lat_c].between(la1, la2)
    agg = agg.loc[m]

    geoms = [
        square_polygon(float(r[lon_c]), float(r[lat_c]), parse_cell_resolution_deg(str(r["grid_cell_id"])))
        for _, r in agg.iterrows()
    ]
    cells = gpd.GeoDataFrame({"grid_cell_id": agg["grid_cell_id"].values}, geometry=geoms, crs="EPSG:4326")

    study_box = box(lo1, la1, lo2, la2)
    land = load_land_gdf()
    clip = land.clip(gpd.GeoSeries([study_box], crs="EPSG:4326"))

    fig, ax = plt.subplots(figsize=(9.2, 9.8), dpi=160)
    ax.set_facecolor("#cfe8fb")
    clip.plot(ax=ax, facecolor="#f4f6f8", edgecolor="#4b5567", linewidth=0.35, alpha=1.0, zorder=2)
    cells.plot(
        ax=ax,
        facecolor="#dfe7f8",
        edgecolor="#253045",
        linewidth=0.28,
        alpha=0.88,
        zorder=4,
        label=f"Lattice cells (n={len(cells):,})",
    )

    gpd.GeoSeries([study_box], crs="EPSG:4326").boundary.plot(ax=ax, color="#991b1b", linestyle="--", linewidth=1.1, zorder=8)

    for name, (lat, lon) in PORT_COORDS.items():
        pt = Point(float(lon), float(lat))
        gpd.GeoSeries([pt], crs="EPSG:4326").plot(ax=ax, color="#111827", markersize=28, marker="*", zorder=12, edgecolor="white", linewidth=0.35)
        ax.annotate(name, xy=(pt.x + 0.18, pt.y + 0.12), fontsize=8.25, fontweight="bold", color="#111827")

    pad_lon = max((lo2 - lo1) * 0.025, 0.15)
    pad_lat = max((la2 - la1) * 0.02, 0.12)
    ax.set_xlim(lo1 - pad_lon, lo2 + pad_lon)
    ax.set_ylim(la1 - pad_lat, la2 + pad_lat)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Baltic Sea study extent and modelling lattice (neutral footprint)", fontsize=12.9, pad=14)
    ax.grid(True, linestyle=":", alpha=0.28, linewidth=0.6)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    caption_md = OUT_DIR / "fig_baltic_study_area_lattice_extent_caption.md"
    caption_md.write_text(
        "# Baltic study-area map (neutral lattice footprint)\n\n"
        f"## Files\n\n- `{BASE.with_suffix('.png').relative_to(ROOT)}`\n"
        f"- `{BASE.with_suffix('.pdf').relative_to(ROOT)}`\n\n"
        "## Caption (draft)\n\n"
        "**Figure.** Baltic Sea analytic window (see dashed extent) showing **study lattice cell footprints** "
        f"(n = **{len(cells):,}** unique `grid_cell_id` within bounds) reconstructed from centroid coordinates and "
        "nominal resolution encoded in cell IDs (`g<deg>_…`). Light grey polygons are coastal land (**Natural Earth 110 m**); "
        "star markers denote focal harbours (**Stockholm**, **Turku**, **Mariehamn**, **Naantali**). "
        f"Derived from **`{src.relative_to(ROOT)}`**.\n\n"
        f"Regenerate: `python3 scripts/{Path(__file__).name}`\n",
        encoding="utf-8",
    )
    print(f"Wrote {BASE.with_suffix('.png')}")
    print(f"Wrote {BASE.with_suffix('.pdf')}")
    print(f"Wrote {caption_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
