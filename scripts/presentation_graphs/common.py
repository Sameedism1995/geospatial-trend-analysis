"""Shared paths, matplotlib style, and dual PNG/PDF export for thesis presentation figures."""

from __future__ import annotations

import json
import re
import ssl
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent.parent
OUT_ROOT = ROOT / "outputs" / "presentation_graphs"

SUBDIRS = (
    "dataset_overview",
    "environmental_indicators",
    "exposure_analysis",
    "wind_regime",
    "machine_learning",
    "temporal_lag_persistence",
    "validation",
    "anomaly_detection",
    "comparison_analysis",
    "summary_maps",
)

DPI = 300
BALTIC_LAT = (53.8, 67.6)
BALTIC_LON = (8.4, 31.2)
PORT_COORDS = {
    "Stockholm": (59.3293, 18.0686),
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
}

NE_CACHE = ROOT / "processed" / "basemap_cache" / "ne_110m_land.zip"
NE_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"

_CELL_RE = re.compile(r"^g(?P<res>[\d.]+)_")

LOG: list[dict] = []


def log_result(
    *,
    category: str,
    stem: str,
    ok: bool,
    rel_png: str | None = None,
    error: str | None = None,
) -> None:
    LOG.append(
        {
            "category": category,
            "stem": stem,
            "ok": ok,
            "png": rel_png,
            "error": error,
        },
    )


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for s in SUBDIRS:
        (OUT_ROOT / s).mkdir(parents=True, exist_ok=True)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "axes.facecolor": "#fafbfc",
            "figure.facecolor": "white",
        }
    )


def save_dual(fig: mpl.figure.Figure, category: str, stem: str) -> str:
    """Write PNG+PDF under category/; return relative path to PNG."""
    dest = OUT_ROOT / category
    dest.mkdir(parents=True, exist_ok=True)
    png = dest / f"{stem}.png"
    pdf = dest / f"{stem}.pdf"
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    rel = str(png.relative_to(ROOT))
    log_result(category=category, stem=stem, ok=True, rel_png=rel)
    return rel


def safe_plot(category: str, stem: str, fn: str) -> None:
    """Run ``fn`` — name of callable in caller is not used; wrap with try/except."""
    raise NotImplementedError


def parse_res_deg(grid_cell_id: str, default: float = 0.1) -> float:
    m = _CELL_RE.match(str(grid_cell_id))
    if not m:
        return default
    try:
        return float(m.group("res"))
    except ValueError:
        return default


def load_ne_land():
    import geopandas as gpd  # noqa: WPS433

    NE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not NE_CACHE.is_file():
        for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
            try:
                req = urllib.request.Request(NE_URL, headers={"User-Agent": "thesis-presentation/1.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                    NE_CACHE.write_bytes(r.read())
                    break
            except Exception:
                continue
        if not NE_CACHE.is_file():
            return None
    return gpd.read_file(f"zip://{NE_CACHE}!ne_110m_land.shp")


def modeling_path() -> Path:
    return ROOT / "data" / "modeling_dataset.parquet"


def features_path() -> Path:
    return ROOT / "processed" / "features_ml_ready.parquet"


def load_modeling() -> pd.DataFrame:
    return pd.read_parquet(modeling_path())


def load_features() -> pd.DataFrame:
    return pd.read_parquet(features_path())


def agg_cell_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("grid_cell_id", sort=False).agg(
        lat=("grid_centroid_lat", "first"),
        lon=("grid_centroid_lon", "first"),
        v=(col, "mean"),
    )
    return g.dropna(subset=["lat", "lon", "v"])


def ocean_cmap(name: str = "tempo"):
    """Prefer cmocean when installed; fallback to matplotlib sequential."""
    try:
        import cmocean

        return getattr(cmocean.cm, name)
    except Exception:
        return plt.get_cmap("Blues")


def trim_scale(s: pd.Series, lo: float = 0.02, hi: float = 0.98) -> tuple[float, float]:
    """Tight percentile limits for cleaner colour scales."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return 0.0, 1.0
    return float(x.quantile(lo)), float(x.quantile(hi))


def load_model_results() -> dict:
    p = ROOT / "data" / "model_results.json"
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def load_predictions() -> pd.DataFrame | None:
    p = ROOT / "data" / "predictions.parquet"
    if not p.is_file():
        return None
    return pd.read_parquet(p)
