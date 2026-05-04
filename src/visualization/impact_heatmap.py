from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PLOTS = ROOT / "outputs" / "plots"
OUTPUT_VIZ = ROOT / "outputs" / "visualizations"


def _pick(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def _scaled(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def _write_density_histogram(
    work: pd.DataFrame,
    *,
    plots_dir: Path,
    logger,
) -> None:
    plot_df = work[["shipping_intensity", "environmental_sensitivity"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        logger.warning("[FINAL VIS] Skipped histogram: no plottable pressure points.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    h = ax.hist2d(
        plot_df["shipping_intensity"],
        plot_df["environmental_sensitivity"],
        bins=35,
        cmap="YlOrRd",
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Grid-week density")

    if "anomaly_score" in work.columns:
        hotspots = work.copy()
        hotspots["anomaly_score"] = pd.to_numeric(hotspots["anomaly_score"], errors="coerce")
        hotspots = hotspots.dropna(subset=["shipping_intensity", "environmental_sensitivity", "anomaly_score"])
        if not hotspots.empty:
            cutoff = hotspots["anomaly_score"].quantile(0.95)
            hs = hotspots[hotspots["anomaly_score"] >= cutoff]
            if not hs.empty:
                ax.scatter(
                    hs["shipping_intensity"],
                    hs["environmental_sensitivity"],
                    c="deepskyblue",
                    s=20,
                    alpha=0.6,
                    label="Anomaly hotspots",
                )

    if "coastal_impact_score" in work.columns:
        top = work.copy()
        top["coastal_impact_score"] = pd.to_numeric(top["coastal_impact_score"], errors="coerce")
        top = top.dropna(subset=["shipping_intensity", "environmental_sensitivity", "coastal_impact_score"])
        if not top.empty:
            top = top.nlargest(15, "coastal_impact_score")
            ax.scatter(
                top["shipping_intensity"],
                top["environmental_sensitivity"],
                c="black",
                s=24,
                alpha=0.8,
                marker="x",
                label="Top impact score",
            )

    ax.set_title("Environmental pressure: shipping intensity vs sensitivity (2D density)")
    ax.set_xlabel("Shipping intensity")
    ax.set_ylabel("Environmental sensitivity (NO2 + water-quality proxy)")
    ax.legend(loc="best")
    fig.tight_layout()
    out = plots_dir / "final_environmental_pressure_map.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    logger.info("[FINAL VIS] Wrote %s", out)


def _write_geographic_pressure_map(
    work: pd.DataFrame,
    *,
    plots_dir: Path,
    logger,
) -> None:
    lat_col = _pick(work, ["grid_centroid_lat", "centroid_lat", "latitude", "lat"])
    lon_col = _pick(work, ["grid_centroid_lon", "centroid_lon", "longitude", "lon"])
    if lat_col is None or lon_col is None:
        logger.warning("[FINAL VIS] Skipped geographic map: no lat/lon columns.")
        return

    w = work.copy()
    w["pressure_index"] = (
        0.5 * pd.to_numeric(w["shipping_intensity"], errors="coerce")
        + 0.5 * pd.to_numeric(w["environmental_sensitivity"], errors="coerce")
    )
    g = w[
        [
            lat_col,
            lon_col,
            "pressure_index",
        ]
        + ([c for c in ["grid_cell_id", "week_start_utc", "coastal_impact_score"] if c in w.columns])
    ].copy()
    g = g.rename(columns={lat_col: "lat", lon_col: "lon"})
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g["pressure_index"] = pd.to_numeric(g["pressure_index"], errors="coerce")
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "pressure_index"])
    if g.empty:
        logger.warning("[FINAL VIS] Skipped geographic map: no valid coordinates.")
        return

    fig, ax = plt.subplots(figsize=(11, 8))
    hb = ax.hexbin(
        g["lon"],
        g["lat"],
        C=g["pressure_index"],
        reduce_C_function=np.mean,
        gridsize=56,
        cmap="inferno",
        mincnt=1,
        vmin=g["pressure_index"].quantile(0.02),
        vmax=g["pressure_index"].quantile(0.98),
        linewidths=0,
    )
    cb = fig.colorbar(hb, ax=ax, label="Pressure index (mean per hex)")
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Environmental pressure — geographic composite (shipping + sensitivity)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    geo_out = plots_dir / "environmental_pressure_geographic.png"
    fig.savefig(geo_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("[FINAL VIS] Wrote %s", geo_out)


def _write_interactive_pressure_map(work: pd.DataFrame, *, logger) -> None:
    try:
        import plotly.express as px
    except ImportError:
        logger.warning("[FINAL VIS] Skipped interactive map: plotly not installed.")
        return

    lat_col = _pick(work, ["grid_centroid_lat", "centroid_lat", "latitude", "lat"])
    lon_col = _pick(work, ["grid_centroid_lon", "centroid_lon", "longitude", "lon"])
    if lat_col is None or lon_col is None:
        return

    w = work.copy()
    w["pressure_index"] = (
        0.5 * pd.to_numeric(w["shipping_intensity"], errors="coerce")
        + 0.5 * pd.to_numeric(w["environmental_sensitivity"], errors="coerce")
    )
    cols = [
        lat_col,
        lon_col,
        "pressure_index",
        "shipping_intensity",
        "environmental_sensitivity",
    ]
    for c in ("grid_cell_id", "week_start_utc", "coastal_impact_score", "anomaly_score"):
        if c in w.columns:
            cols.append(c)
    sub = w[[c for c in cols if c in w.columns]].dropna(subset=[lat_col, lon_col, "pressure_index"])
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[lat_col, lon_col])
    if sub.empty or len(sub) < 10:
        logger.warning("[FINAL VIS] Skipped interactive map: insufficient rows.")
        return

    OUTPUT_VIZ.mkdir(parents=True, exist_ok=True)
    hover_cols = [
        "pressure_index",
        "shipping_intensity",
        "environmental_sensitivity",
    ]
    if "grid_cell_id" in sub.columns:
        hover_cols.append("grid_cell_id")
    if "week_start_utc" in sub.columns:
        hover_cols.append("week_start_utc")
    if "coastal_impact_score" in sub.columns:
        hover_cols.append("coastal_impact_score")

    if len(sub) > 5500:
        sub = sub.sample(n=5500, random_state=42)

    lo_c = float(pd.to_numeric(sub["pressure_index"], errors="coerce").quantile(0.03))
    hi_c = float(pd.to_numeric(sub["pressure_index"], errors="coerce").quantile(0.97))
    hover_map = {c: True for c in hover_cols}
    lat_m = float(pd.to_numeric(sub[lat_col], errors="coerce").median())
    lon_m = float(pd.to_numeric(sub[lon_col], errors="coerce").median())

    base_kw: dict[str, object] = dict(
        lat=lat_col,
        lon=lon_col,
        color="pressure_index",
        color_continuous_scale="Turbo",
        range_color=(lo_c, hi_c),
        hover_data=hover_map,
        zoom=4,
        height=700,
        title="Environmental pressure (composite) — interactive map",
        opacity=0.45,
        labels={"pressure_index": "Pressure index"},
        center=dict(lat=lat_m, lon=lon_m),
    )
    try:
        fig = px.scatter_map(sub, **base_kw, map_style="carto-darkmatter")
    except Exception:  # noqa: BLE001
        try:
            fig = px.scatter_mapbox(sub, **base_kw, mapbox_style="carto-darkmatter")
        except Exception as exc2:  # noqa: BLE001
            logger.warning("[FINAL VIS] Interactive map failed: %s", exc2)
            return

    out_html = OUTPUT_VIZ / "environmental_pressure_interactive.html"
    fig.write_html(out_html)
    logger.info("[FINAL VIS] Wrote %s", out_html)


def run_final_visualization(df: pd.DataFrame, logger) -> None:
    plots_dir = OUTPUT_PLOTS
    reports_dir = ROOT / "outputs" / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)

    vessel_col = _pick(df, ["vessel_density", "vessel_density_t"])
    no2_col = _pick(df, ["NO2_mean", "no2_mean_t"])
    water_cols_mean = ["ndci_mean", "ndti_mean", "ndwi_mean", "fai_mean", "b11_mean", "sst"]
    water_cols_short = ["ndci", "ndti", "ndwi", "fai", "sst"]
    water_candidates = water_cols_mean + water_cols_short
    water_cols = [c for c in water_candidates if c in df.columns]
    if vessel_col is None or no2_col is None:
        logger.warning("[FINAL VIS] Skipped: vessel or NO2 columns missing.")
        return

    work = df.copy()
    work["shipping_intensity"] = _scaled(work[vessel_col])
    water_proxy = (
        work[water_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if water_cols else pd.Series(np.nan, index=work.index)
    )
    work["environmental_sensitivity"] = 0.6 * _scaled(work[no2_col]).fillna(0) + 0.4 * _scaled(water_proxy).fillna(0)

    anomaly_path = reports_dir / "anomaly_scores.csv"
    impact_path = reports_dir / "coastal_impact_score.csv"
    if anomaly_path.exists() and {"grid_cell_id", "week_start_utc"}.issubset(set(work.columns)):
        adf = pd.read_csv(anomaly_path)
        adf["week_start_utc"] = pd.to_datetime(adf.get("week_start_utc"), utc=True, errors="coerce")
        work["week_start_utc"] = pd.to_datetime(work["week_start_utc"], utc=True, errors="coerce")
        work = work.merge(
            adf[[c for c in ["grid_cell_id", "week_start_utc", "anomaly_score", "anomaly_label"] if c in adf.columns]],
            on=["grid_cell_id", "week_start_utc"],
            how="left",
        )
    if impact_path.exists() and {"grid_cell_id", "week_start_utc"}.issubset(set(work.columns)):
        cdf = pd.read_csv(impact_path)
        cdf["week_start_utc"] = pd.to_datetime(cdf.get("week_start_utc"), utc=True, errors="coerce")
        work["week_start_utc"] = pd.to_datetime(work["week_start_utc"], utc=True, errors="coerce")
        work = work.merge(
            cdf[[c for c in ["grid_cell_id", "week_start_utc", "coastal_impact_score"] if c in cdf.columns]],
            on=["grid_cell_id", "week_start_utc"],
            how="left",
        )

    _write_density_histogram(work, plots_dir=plots_dir, logger=logger)
    _write_geographic_pressure_map(work, plots_dir=plots_dir, logger=logger)
    _write_interactive_pressure_map(work, logger=logger)
