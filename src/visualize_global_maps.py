"""
Interactive global / regional maps from data/modeling_dataset.parquet (Plotly HTML).

Outputs:
  - data/visualizations/global_map.html
  - data/visualizations/ocean_global_map.html
  - data/visualizations/global_temporal_animation.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAP_STYLE = "carto-darkmatter"
COLOR_SCALE = "Viridis"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    df["week_str"] = df["week_start_utc"].dt.strftime("%Y-%m-%d")
    for c in (
        "sentinel_ndti_mean_t",
        "vessel_density_t",
        "sentinel_ndvi_mean_t",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _mapbox_center_zoom(df: pd.DataFrame) -> dict[str, float | int]:
    lat = df["grid_centroid_lat"].astype(float)
    lon = df["grid_centroid_lon"].astype(float)
    m = dict(
        lat=float(lat.median()),
        lon=float(lon.median()),
        zoom=4,
    )
    lat_pad = max(2.0, (lat.max() - lat.min()) * 1.4)
    lon_pad = max(2.0, (lon.max() - lon.min()) * 1.4)
    return m | {"lat_pad": lat_pad, "lon_pad": lon_pad}


def try_load_land_union(cache_dir: Path):
    """Natural Earth land polygons for ocean mask; None if unavailable."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union
    except ImportError:
        return None

    path = cache_dir / "ne_110m_land.geojson"
    url = (
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
        "master/geojson/ne_110m_land.geojson"
    )
    if not path.exists():
        try:
            import urllib.request

            cache_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        except Exception:  # noqa: BLE001
            return None
    try:
        land = gpd.read_file(path)
        if hasattr(land.geometry, "union_all"):
            geom = land.geometry.union_all()
        else:
            geom = unary_union(land.geometry)
        return geom
    except Exception:  # noqa: BLE001
        return None


def add_marine_flag(df: pd.DataFrame, land_union) -> pd.DataFrame:
    from shapely.geometry import Point

    out = df.copy()
    pts = [Point(float(r["grid_centroid_lon"]), float(r["grid_centroid_lat"])) for _, r in out.iterrows()]
    # Slight buffer (~15 km) so coastal sea cells stay
    try:
        land_buf = land_union.buffer(0.02)
        out["_on_land"] = [p.within(land_buf) for p in pts]
        out["marine"] = ~out["_on_land"]
        out = out.drop(columns=["_on_land"])
    except Exception:  # noqa: BLE001
        out["marine"] = True
    return out


def build_global_map(df: pd.DataFrame, out_html: Path) -> None:
    """Per-grid medians; buttons toggle which scalar is mapped to color."""
    g = (
        df.groupby("grid_cell_id", sort=False)
        .agg(
            grid_centroid_lat=("grid_centroid_lat", "mean"),
            grid_centroid_lon=("grid_centroid_lon", "mean"),
            sentinel_ndti_mean_t=("sentinel_ndti_mean_t", "median"),
            vessel_density_t=("vessel_density_t", "median"),
            sentinel_ndvi_mean_t=("sentinel_ndvi_mean_t", "median"),
        )
        .reset_index()
    )
    g = g.dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])
    cz = _mapbox_center_zoom(g)

    metrics: list[tuple[str, str]] = [
        ("NDTI (t)", "sentinel_ndti_mean_t"),
        ("Vessel density (t)", "vessel_density_t"),
        ("NDVI (t)", "sentinel_ndvi_mean_t"),
    ]
    traces: list[go.Scattermapbox] = []
    for i, (_label, col) in enumerate(metrics):
        if col not in g.columns:
            continue
        z = g[col].astype(float)
        traces.append(
            go.Scattermapbox(
                lat=g["grid_centroid_lat"],
                lon=g["grid_centroid_lon"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=z,
                    colorscale=COLOR_SCALE,
                    showscale=(i == 0),
                    colorbar=dict(title=_label.split()[0], x=1.02) if i == 0 else None,
                    opacity=0.78,
                ),
                text=g["grid_cell_id"],
                customdata=np.column_stack(
                    [
                        g["sentinel_ndti_mean_t"],
                        g["vessel_density_t"],
                        g["sentinel_ndvi_mean_t"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "NDTI: %{customdata[0]:.4f}<br>"
                    "Vessel: %{customdata[1]:.6f}<br>"
                    "NDVI: %{customdata[2]:.4f}<extra></extra>"
                ),
                name=_label,
                visible=i == 0,
            )
        )

    fig = go.Figure(data=traces)
    n_vis = len(traces)
    buttons = []
    for i, (lab, _col) in enumerate(metrics):
        if i >= n_vis:
            break
        vis = [j == i for j in range(n_vis)]
        buttons.append(
            dict(
                label=lab,
                method="update",
                args=[
                    {"visible": vis},
                    {"title": f"Global map — color: {lab} (per-grid median)"},
                ],
            )
        )

    fig.update_layout(
        mapbox=dict(style=MAP_STYLE, center=dict(lat=cz["lat"], lon=cz["lon"]), zoom=4),
        title="Global map — use dropdown to change color field",
        height=720,
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.02,
                y=1.12,
                xanchor="left",
            )
        ],
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def build_ocean_map(
    df: pd.DataFrame,
    out_html: Path,
    *,
    land_union,
) -> None:
    """Marine-only when land mask loads; else study-extent zoom. Highlights top 10% vessel density."""
    g = (
        df.groupby("grid_cell_id", sort=False)
        .agg(
            grid_centroid_lat=("grid_centroid_lat", "mean"),
            grid_centroid_lon=("grid_centroid_lon", "mean"),
            sentinel_ndti_mean_t=("sentinel_ndti_mean_t", "median"),
            vessel_density_t=("vessel_density_t", "median"),
        )
        .reset_index()
    )
    g = g.dropna(subset=["grid_centroid_lat", "grid_centroid_lon"])

    if land_union is not None:
        g = add_marine_flag(g, land_union)
        g = g.loc[g["marine"] == True].copy()  # noqa: E712
        title_suffix = " — ocean cells (land mask)"
    else:
        g["marine"] = True
        title_suffix = " — study extent (land mask unavailable; zoomed marine region)"

    if g.empty:
        g = (
            df.groupby("grid_cell_id", sort=False)
            .agg(
                grid_centroid_lat=("grid_centroid_lat", "mean"),
                grid_centroid_lon=("grid_centroid_lon", "mean"),
                sentinel_ndti_mean_t=("sentinel_ndti_mean_t", "median"),
                vessel_density_t=("vessel_density_t", "median"),
            )
            .reset_index()
        )
        title_suffix += " [fallback: all grids]"

    v = g["vessel_density_t"].astype(float)
    thr = v.quantile(0.90) if v.notna().any() else np.nan
    g["top_traffic"] = v >= thr if np.isfinite(thr) else False

    # Opacity from normalized vessel density
    vn = v.copy()
    if vn.max() > vn.min():
        vn = (vn - vn.min()) / (vn.max() - vn.min())
    else:
        vn = pd.Series(0.5, index=g.index)
    g["opacity"] = (0.25 + 0.55 * vn.fillna(0.3)).clip(0.2, 0.95)

    base = g.loc[~g["top_traffic"]]
    hi = g.loc[g["top_traffic"]]

    fig = go.Figure()
    if len(base):
        fig.add_trace(
            go.Scattermapbox(
                lat=base["grid_centroid_lat"],
                lon=base["grid_centroid_lon"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=base["sentinel_ndti_mean_t"],
                    colorscale=COLOR_SCALE,
                    showscale=True,
                    colorbar=dict(title="median NDTI (t)"),
                    opacity=base["opacity"],
                ),
                text=base["grid_cell_id"],
                customdata=np.column_stack([base["sentinel_ndti_mean_t"], base["vessel_density_t"]]),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "NDTI: %{customdata[0]:.4f}<br>"
                    "Vessel: %{customdata[1]:.6f}<extra></extra>"
                ),
                name="Grids",
            )
        )
    if len(hi):
        fig.add_trace(
            go.Scattermapbox(
                lat=hi["grid_centroid_lat"],
                lon=hi["grid_centroid_lon"],
                mode="markers",
                marker=dict(size=14, color="#ffea00", opacity=0.95, symbol="circle"),
                text=hi["grid_cell_id"],
                hovertemplate="<b>Top 10% traffic</b><br>%{text}<extra></extra>",
                name="Top 10% vessel density",
            )
        )

    cz = _mapbox_center_zoom(g)
    fig.update_layout(
        mapbox=dict(
            style=MAP_STYLE,
            center=dict(lat=cz["lat"], lon=cz["lon"]),
            zoom=5,
        ),
        title=f"Sea-focused map{title_suffix}",
        height=720,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def build_animation(df: pd.DataFrame, out_html: Path) -> None:
    """Weekly frames: NDTI (t) per grid-week."""
    d = df.dropna(subset=["grid_centroid_lat", "grid_centroid_lon", "week_str"]).copy()
    d = d.sort_values("week_start_utc")
    # Stable color range
    lo = float(d["sentinel_ndti_mean_t"].quantile(0.02))
    hi = float(d["sentinel_ndti_mean_t"].quantile(0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = -1.0, 1.0

    fig = px.scatter_mapbox(
        d,
        lat="grid_centroid_lat",
        lon="grid_centroid_lon",
        color="sentinel_ndti_mean_t",
        animation_frame="week_str",
        hover_data={
            "grid_cell_id": True,
            "sentinel_ndti_mean_t": ":.4f",
            "vessel_density_t": ":.6f",
            "week_str": True,
        },
        color_continuous_scale=COLOR_SCALE,
        range_color=(lo, hi),
        mapbox_style=MAP_STYLE,
        height=720,
        title="Weekly NDTI (t) — use slider to change week",
        opacity=0.65,
    )
    cz = _mapbox_center_zoom(d)
    fig.update_layout(
        mapbox=dict(
            style=MAP_STYLE,
            center=dict(lat=cz["lat"], lon=cz["lon"]),
            zoom=4,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Plotly maps for modeling dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/modeling_dataset.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/visualizations"))
    parser.add_argument(
        "--only",
        choices=("all", "global", "ocean", "animation"),
        default="all",
        help="Which HTML to build",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    inp = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    cache_dir = root / "data" / "downloads" / "natural_earth"

    if not inp.exists():
        raise SystemExit(f"Missing {inp}")

    df = load_data(inp)
    land_union = try_load_land_union(cache_dir)

    if args.only in ("all", "global"):
        build_global_map(df, out_dir / "global_map.html")
        print(f"Wrote {out_dir / 'global_map.html'}")
    if args.only in ("all", "ocean"):
        build_ocean_map(df, out_dir / "ocean_global_map.html", land_union=land_union)
        print(f"Wrote {out_dir / 'ocean_global_map.html'}")
    if args.only in ("all", "animation"):
        build_animation(df, out_dir / "global_temporal_animation.html")
        print(f"Wrote {out_dir / 'global_temporal_animation.html'}")


if __name__ == "__main__":
    main()
