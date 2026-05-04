"""Interactive map: grid port_exposure_score with port markers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("port_exposure_map")


def write_port_exposure_map(
    df: pd.DataFrame,
    project_root: Path,
    *,
    ports_path: Path | None = None,
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path | None:
    """
    Plotly HTML: grid cells colored by port_exposure_score, ports overlaid.

    Saves outputs/visualizations/port_city_exposure_map.html
    """
    log = logger or LOGGER
    out_p = output_path or (project_root / "outputs" / "visualizations" / "port_city_exposure_map.html")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    csv_ports = ports_path or (project_root / "data" / "aux" / "baltic_ports.csv")

    if df.empty:
        log.warning("[PORT MAP] Empty dataframe; skip map.")
        return None

    lat_col = next(
        (c for c in ("grid_centroid_lat", "centroid_lat", "latitude") if c in df.columns),
        None,
    )
    lon_col = next(
        (c for c in ("grid_centroid_lon", "centroid_lon", "longitude") if c in df.columns),
        None,
    )
    if lat_col is None or lon_col is None or "port_exposure_score" not in df.columns:
        log.warning("[PORT MAP] Missing lat/lon or port_exposure_score; skip map.")
        return None

    try:
        import plotly.graph_objects as go
    except ImportError:
        log.warning("[PORT MAP] plotly not installed; skip map.")
        return None

    plot_df = df[[lat_col, lon_col, "port_exposure_score"]].copy()
    plot_df[lat_col] = pd.to_numeric(plot_df[lat_col], errors="coerce")
    plot_df[lon_col] = pd.to_numeric(plot_df[lon_col], errors="coerce")
    plot_df["port_exposure_score"] = pd.to_numeric(plot_df["port_exposure_score"], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        log.warning("[PORT MAP] No valid rows after cleaning; skip map.")
        return None

    max_pts = 12_000
    if len(plot_df) > max_pts:
        plot_df = plot_df.sample(max_pts, random_state=42)

    z_vals = plot_df["port_exposure_score"].to_numpy(dtype=float)
    z_label = np.log1p(np.clip(z_vals, 0, None))

    grid_trace = go.Scattermapbox(
        lat=plot_df[lat_col],
        lon=plot_df[lon_col],
        mode="markers",
        marker=dict(
            size=7,
            color=z_label,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="log1p(port<br>exposure)"),
        ),
        text=[f"score={v:.6g}" for v in z_vals],
        hoverinfo="lat+lon+text",
        name="Grid exposure",
    )

    fig_data: list = [grid_trace]

    if csv_ports.exists():
        ports = pd.read_csv(csv_ports)
        name_col = "port_name" if "port_name" in ports.columns else ports.columns[0]
        pla = pd.to_numeric(ports["latitude"], errors="coerce")
        plo = pd.to_numeric(ports["longitude"], errors="coerce")
        ok = pla.notna() & plo.notna()
        if ok.any():
            fig_data.append(
                go.Scattermapbox(
                    lat=pla[ok],
                    lon=plo[ok],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color="white",
                        symbol="circle",
                    ),
                    text=ports.loc[ok, name_col].astype(str),
                    textposition="top center",
                    textfont=dict(size=11, color="white"),
                    name="Ports",
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    mean_lat = float(plot_df[lat_col].median())
    mean_lon = float(plot_df[lon_col].median())
    pad_lat = max(1.5, float((plot_df[lat_col].max() - plot_df[lat_col].min()) * 0.65))
    pad_lon = max(1.5, float((plot_df[lon_col].max() - plot_df[lon_col].min()) * 0.65))

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        title="Port exposure score (grid) with named ports",
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=mean_lat, lon=mean_lon),
            zoom=5.8,
            bounds=dict(
                west=float(plot_df[lon_col].min() - pad_lon),
                east=float(plot_df[lon_col].max() + pad_lon),
                south=float(plot_df[lat_col].min() - pad_lat),
                north=float(plot_df[lat_col].max() + pad_lat),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    fig.write_html(out_p, include_plotlyjs="cdn")
    log.info("[PORT MAP] Wrote %s", out_p)
    return out_p
