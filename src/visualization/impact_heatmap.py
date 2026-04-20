from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def run_final_visualization(df: pd.DataFrame, logger) -> None:
    plots_dir = Path("outputs/plots")
    reports_dir = Path("outputs/reports")
    plots_dir.mkdir(parents=True, exist_ok=True)

    vessel_col = _pick(df, ["vessel_density", "vessel_density_t"])
    no2_col = _pick(df, ["NO2_mean", "no2_mean_t"])
    water_cols = [c for c in ["ndci", "ndti", "ndwi", "fai", "sst"] if c in df.columns]
    if vessel_col is None or no2_col is None:
        logger.warning("[FINAL VIS] Skipped: vessel or NO2 columns missing.")
        return

    work = df.copy()
    work["shipping_intensity"] = _scaled(work[vessel_col])
    water_proxy = work[water_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if water_cols else pd.Series(np.nan, index=work.index)
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

    plot_df = work[["shipping_intensity", "environmental_sensitivity"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        logger.warning("[FINAL VIS] Skipped: no plottable pressure points.")
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

    ax.set_title("Final Environmental Pressure Map")
    ax.set_xlabel("Shipping intensity")
    ax.set_ylabel("Environmental sensitivity (NO2 + water-quality proxy)")
    ax.legend(loc="best")
    fig.tight_layout()
    out = plots_dir / "final_environmental_pressure_map.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    logger.info("[FINAL VIS] Wrote %s", out)
