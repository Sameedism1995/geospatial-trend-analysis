#!/usr/bin/env python3
"""Coastal exposure indices, distance structuring, visuals, conservative stats (association only).

  python3 src/analysis/run_coastal_exposure_analysis.py \\
    --input outputs/processed/features_ml_ready_coastal_wind.parquet

Outputs: outputs/reports/exposure_indices_summary.csv,
  coastal_exposure_statistics.csv, coastal_exposure_analysis_summary.md,
  outputs/figures/coastal_exposure/*.png"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.run_land_pollution_drivers_wind import assign_shipping_band_tight, ensure_coast_distance  # noqa: E402
from human_impact_distance_analysis import load_coastline_points, load_land_boundary_points  # noqa: E402

REPORTS = _ROOT / "outputs" / "reports"
FIGURES = _ROOT / "outputs" / "figures" / "coastal_exposure"
DEFAULT_INPUT = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"
COAST_KM = 30.0
BANDS = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]
COS45 = float(np.cos(np.radians(45.0)))


def _rank_pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rank(pct=True, method="average")


def _relu_align(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0)


def _inv_km(d: pd.Series) -> pd.Series:
    x = pd.to_numeric(d, errors="coerce").clip(lower=0.0)
    return 1.0 / (1.0 + x)


def prepare_panel(df: pd.DataFrame, ne_cache: Path) -> pd.DataFrame:
    out = df.copy()
    out["grid_cell_id"] = out["grid_cell_id"].astype(str)
    out["week_start_utc"] = pd.to_datetime(out["week_start_utc"], utc=True, errors="coerce")
    out["_wk"] = out["week_start_utc"].dt.normalize()
    out = ensure_coast_distance(out, ne_cache)
    ds = pd.to_numeric(out["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    out["shipping_distance_band_tight"] = assign_shipping_band_tight(ds)
    no2 = pd.to_numeric(out["no2_mean_t"], errors="coerce")
    out["weekly_no2_anomaly"] = no2 - out.groupby("_wk")["no2_mean_t"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").mean(),
    )
    ref = out["shipping_distance_band_tight"] == "15-30 km"
    bm = (
        out.loc[ref]
        .groupby(out.loc[ref, "_wk"])["no2_mean_t"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    )
    out["local_no2_excess"] = no2 - out["_wk"].map(bm.get)
    ncol = "ndti_mean_t" if "ndti_mean_t" in out.columns else ("ndti_mean" if "ndti_mean" in out.columns else None)
    if ncol:
        nv = pd.to_numeric(out[ncol], errors="coerce")
        out["ndti_weekly_anomaly"] = nv - out.groupby("_wk")[ncol].transform(
            lambda s: pd.to_numeric(s, errors="coerce").mean(),
        )
    else:
        out["ndti_weekly_anomaly"] = np.nan
    ca = pd.to_numeric(out["coastal_wind_alignment_score"], errors="coerce")
    if "coastal_wind_shoreward_45deg" in out.columns:
        out["shoreward_binary"] = pd.to_numeric(out["coastal_wind_shoreward_45deg"], errors="coerce").fillna(
            (ca >= COS45).astype(float),
        )
    else:
        out["shoreward_binary"] = (ca >= COS45).astype(float)
    dc = pd.to_numeric(out["distance_to_coast_km"], errors="coerce")
    out["coastal_panel"] = dc.notna() & (dc <= COAST_KM) & ds.notna() & (ds >= 0) & (ds < 30)
    return out


def build_indices(out: pd.DataFrame) -> pd.DataFrame:
    wk = out["_wk"]
    rp_v = out.groupby(wk)["vessel_density_t"].transform(lambda s: _rank_pct(s))
    rp_w = _rank_pct(_relu_align(out["coastal_wind_alignment_score"]))
    rp_inv = _rank_pct(_inv_km(out["distance_to_coast_km"]))
    out["maritime_exposure_index_raw"] = rp_v * rp_w * rp_inv
    out["maritime_exposure_index"] = _rank_pct(out["maritime_exposure_index_raw"])

    rp_no2 = out.groupby(wk)["local_no2_excess"].transform(lambda s: _rank_pct(s))
    ca = pd.to_numeric(out["coastal_wind_alignment_score"], errors="coerce").clip(-1, 1)
    coast_01 = (ca + 1.0) / 2.0
    rp_ch = _rank_pct(coast_01)
    tr = pd.to_numeric(out["pollution_transport_wind_alignment_score"], errors="coerce").fillna(0).clip(-1, 1)
    rp_tr = _rank_pct((tr + 1.0) / 2.0)
    out["atmospheric_coastal_exposure_index_raw"] = rp_no2 * rp_ch * rp_tr * rp_inv
    out["atmospheric_coastal_exposure_index"] = _rank_pct(out["atmospheric_coastal_exposure_index_raw"])

    def z_wk(col: str) -> pd.Series:
        v = pd.to_numeric(out[col], errors="coerce")

        def _z(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            std = s.std(ddof=0)
            if std is None or std < 1e-12:
                return s * 0.0
            return (s - s.mean()) / std

        return v.groupby(wk).transform(_z)

    parts = [z_wk("local_no2_excess"), z_wk("vessel_density_t")]
    if out["ndti_weekly_anomaly"].notna().any():
        parts.append(
            out.groupby(wk)["ndti_weekly_anomaly"].transform(
                lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 1e-12 else s * 0,
            ),
        )
    if "oil_slick_probability_t" in out.columns:
        parts.append(z_wk("oil_slick_probability_t"))
    parts.append(_rank_pct(coast_01) - 0.5)
    zdf = pd.concat(parts, axis=1)
    out["environmental_stress_index"] = _rank_pct(zdf.mean(axis=1, skipna=True))
    return out


def index_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for title, col in [
        ("Maritime Exposure Index", "maritime_exposure_index"),
        ("Atmospheric Coastal Exposure Index", "atmospheric_coastal_exposure_index"),
        ("Environmental Stress Index (experimental)", "environmental_stress_index"),
    ]:
        x = pd.to_numeric(df.loc[df["coastal_panel"], col], errors="coerce")
        rows.append(
            {
                "index_name": title,
                "column": col,
                "n_non_null": int(x.notna().sum()),
                "mean": float(x.mean(skipna=True)),
                "std": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
                "p05": float(x.quantile(0.05)),
                "p50": float(x.quantile(0.5)),
                "p95": float(x.quantile(0.95)),
            },
        )
    return pd.DataFrame(rows)


def boot_mean_ci(vals: np.ndarray, n_boot: int = 2500, seed: int = 42) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float(np.nanmean(vals)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def band_stats_table(df: pd.DataFrame, cols: list[str], shore: str = "shoreward_binary") -> pd.DataFrame:
    coastal = df.loc[df["coastal_panel"]]
    rows: list[dict[str, Any]] = []
    for band in BANDS:
        slab = coastal[coastal["shipping_distance_band_tight"] == band]
        for split_name, pred in (
            ("shoreward_wind_ge_cos45", lambda d: pd.to_numeric(d[shore], errors="coerce").eq(1)),
            ("non_shoreward", lambda d: pd.to_numeric(d[shore], errors="coerce").eq(0)),
        ):
            sub = slab.loc[pred(slab)]
            for c in cols:
                v = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy(float)
                m, lo, hi = boot_mean_ci(v)
                rows.append(
                    {
                        "distance_band": band,
                        "wind_split": split_name,
                        "variable": c,
                        "n": len(v),
                        "mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    },
                )
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    rng = np.random.default_rng(42)
    ma, mb = min(len(a), 350), min(len(b), 350)
    if len(a) > ma:
        a = rng.choice(a, ma, replace=False)
    if len(b) > mb:
        b = rng.choice(b, mb, replace=False)
    cmp = (a[:, None] > b[None, :]).mean() - (a[:, None] < b[None, :]).mean()
    return float(cmp)


def mannwhitney_table(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    coastal = df.loc[df["coastal_panel"]]
    rows = []
    for band in BANDS:
        slab = coastal[coastal["shipping_distance_band_tight"] == band]
        sh = pd.to_numeric(slab["shoreward_binary"], errors="coerce").eq(1)
        for oc in outcomes:
            a = pd.to_numeric(slab.loc[sh, oc], errors="coerce").dropna().to_numpy(float)
            b = pd.to_numeric(slab.loc[~sh, oc], errors="coerce").dropna().to_numpy(float)
            r: dict[str, Any] = {"distance_band": band, "outcome": oc, "n_shoreward": len(a), "n_other": len(b)}
            if len(a) >= 4 and len(b) >= 4:
                r["mannwhitney_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                r["cliffs_delta"] = cliffs_delta(a, b)
                ma, mb = float(np.mean(a)), float(np.mean(b))
                r["mean_shoreward"] = ma
                r["mean_other"] = mb
                r["mean_diff"] = ma - mb
            else:
                r["mannwhitney_p"] = np.nan
                r["cliffs_delta"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def spearman_block(df: pd.DataFrame) -> pd.DataFrame:
    coastal = df.loc[df["coastal_panel"]].copy()
    pairs = [
        ("local_no2_excess", "coastal_wind_alignment_score"),
        ("local_no2_excess", "maritime_exposure_index"),
        ("local_no2_excess", "atmospheric_coastal_exposure_index"),
        ("oil_slick_probability_t", "coastal_wind_alignment_score"),
    ]
    rows = []
    for a, b in pairs:
        if a not in coastal.columns or b not in coastal.columns:
            continue
        x = pd.to_numeric(coastal[a], errors="coerce")
        y = pd.to_numeric(coastal[b], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < 10:
            continue
        rho, p = stats.spearmanr(x.loc[m], y.loc[m])
        rows.append({"x": a, "y": b, "spearman_rho": float(rho), "p_value": float(p), "n": int(m.sum())})
    return pd.DataFrame(rows)


def plot_decay(df: pd.DataFrame) -> None:
    st = band_stats_table(
        df,
        ["maritime_exposure_index", "local_no2_excess", "coastal_wind_alignment_score", "environmental_stress_index", "vessel_density_t"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    xi = np.arange(len(BANDS))
    for ax, var, ttl in zip(
        axes,
        ["maritime_exposure_index", "local_no2_excess"],
        ["Maritime exposure index (rank scale)", "Local NO2 excess (native units)"],
    ):
        for split, color, mk in (
            ("shoreward_wind_ge_cos45", "darkgreen", "o"),
            ("non_shoreward", "slategray", "s"),
        ):
            sub = st[(st["variable"] == var) & (st["wind_split"] == split)]
            sub = sub.set_index("distance_band").reindex(BANDS)
            ax.errorbar(
                xi,
                sub["mean"],
                yerr=[sub["mean"] - sub["ci95_low"], sub["ci95_high"] - sub["mean"]],
                fmt=mk + "-",
                color=color,
                capsize=3,
                label=split.replace("_", " "),
            )
        ax.set_xticks(xi)
        ax.set_xticklabels(BANDS, rotation=15)
        ax.set_xlabel("Shipping-distance band (from high-density cells)")
        ax.set_ylabel(ttl)
        ax.set_title("Coastal exposure decay structure (bootstrap 95% CI on mean)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Association framing: shoreline wind split vs inland distance bands", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig1_coastal_exposure_decay.png", dpi=180)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame) -> None:
    coastal = df.loc[df["coastal_panel"]].copy()
    coastal["_vb"] = pd.qcut(
        pd.to_numeric(coastal["vessel_density_t"], errors="coerce"),
        q=5,
        labels=False,
        duplicates="drop",
    )
    coastal["_ab"] = pd.qcut(
        pd.to_numeric(coastal["coastal_wind_alignment_score"], errors="coerce"),
        q=5,
        labels=False,
        duplicates="drop",
    )
    piv = coastal.pivot_table(
        values="local_no2_excess",
        index="_vb",
        columns="_ab",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    sns.heatmap(piv, cmap="RdYlBu_r", center=0, ax=ax, cbar_kws={"label": "Mean local NO2 excess"})
    ax.set_xlabel("Coastal wind alignment score quintile (low to high)")
    ax.set_ylabel("Vessel density quintile (low to high)")
    ax.set_title("Fig2. Environmental behaviour: NO2 excess by shipping × shoreward wind (exploratory)")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig2_shipping_wind_no2_heatmap.png", dpi=180)
    plt.close(fig)

    piv2 = coastal.pivot_table(
        values="environmental_stress_index",
        index="_vb",
        columns="_ab",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    sns.heatmap(piv2, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean environmental stress rank"})
    ax.set_title("Fig2b. Stress index rank by shipping × wind alignment")
    ax.set_xlabel("Wind alignment quintile")
    ax.set_ylabel("Vessel quintile")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig2b_shipping_wind_stress_heatmap.png", dpi=180)
    plt.close(fig)


def plot_hex(df: pd.DataFrame) -> None:
    coastal = df.loc[df["coastal_panel"]]
    x = pd.to_numeric(coastal["coastal_wind_alignment_score"], errors="coerce")
    y = pd.to_numeric(coastal["local_no2_excess"], errors="coerce")
    m = x.notna() & y.notna()
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    hb = ax.hexbin(x.loc[m], y.loc[m], gridsize=32, cmap="viridis", mincnt=2)
    plt.colorbar(hb, ax=ax, label="count")
    if int(m.sum()) > 15:
        r, p = stats.spearmanr(x.loc[m], y.loc[m])
        ax.text(0.02, 0.98, f"Spearman rho={r:.3f}\np={p:.3g}\nn={int(m.sum())}", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("Coastal wind alignment score (directional coupling proxy)")
    ax.set_ylabel("Local NO2 excess")
    ax.set_title("Fig3. Directional association structure (association, not transport proof)")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig3_wind_alignment_no2_hexbin.png", dpi=180)
    plt.close(fig)


def plot_risk_map(df: pd.DataFrame, coast_lat: np.ndarray, coast_lon: np.ndarray) -> None:
    coastal = df.loc[df["coastal_panel"]]
    spec = {
        "mei": ("maritime_exposure_index", "mean"),
        "lon": ("grid_centroid_lon", "first"),
        "lat": ("grid_centroid_lat", "first"),
        "no2m": ("local_no2_excess", "mean"),
        "oilm": ("oil_slick_probability_t", "mean"),
        "vd": ("vessel_density_t", "mean"),
    }
    if "wind_u_mean" in coastal.columns:
        spec["wu"] = ("wind_u_mean", "mean")
        spec["wv"] = ("wind_v_mean", "mean")
    agg = coastal.groupby("grid_cell_id").agg(**spec).reset_index()

    hi_n = agg["no2m"].quantile(0.9)
    hi_o = agg["oilm"].quantile(0.9)
    step = max(1, len(coast_lat) // 350)

    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.plot(coast_lon[::step], coast_lat[::step], color="#063", lw=1.2, alpha=0.65, label="Coast sampling")
    sc = ax.scatter(
        agg["lon"],
        agg["lat"],
        c=agg["mei"],
        cmap="inferno",
        s=140,
        edgecolors="k",
        linewidths=0.3,
        zorder=4,
        label="Exposure intensity rank",
    )
    plt.colorbar(sc, ax=ax, shrink=0.62, label="Mean maritime exposure index")

    has_wind = "wu" in agg.columns and pd.to_numeric(agg["wu"], errors="coerce").notna().any()
    if has_wind:
        scl = 0.12
        ax.quiver(
            agg["lon"],
            agg["lat"],
            pd.to_numeric(agg["wu"], errors="coerce") * scl,
            pd.to_numeric(agg["wv"], errors="coerce") * scl,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color="navy",
            alpha=0.55,
            zorder=5,
            label="Mean wind u,v",
        )
    hn = agg["no2m"] >= hi_n
    ax.scatter(
        agg.loc[hn, "lon"],
        agg.loc[hn, "lat"],
        s=260,
        facecolors="none",
        edgecolors="darkorange",
        linewidths=1.8,
        zorder=6,
        label="NO2 excess hotspot",
    )
    ho = agg["oilm"] >= hi_o
    ax.scatter(
        agg.loc[ho, "lon"],
        agg.loc[ho, "lat"],
        s=120,
        marker="x",
        c="crimson",
        linewidths=1.6,
        zorder=7,
        label="Oil-proxy hotspot",
    )
    hi_v = agg["vd"] >= agg["vd"].quantile(0.9)
    ax.scatter(
        agg.loc[hi_v, "lon"],
        agg.loc[hi_v, "lat"],
        s=44,
        c="cyan",
        edgecolors="navy",
        linewidths=0.6,
        zorder=3,
        alpha=0.85,
        label="High vessel density",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Fig4. Coastal exposure behaviour map — integrated overlay (spatial association context)",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig4_coastal_exposure_risk_map.png", dpi=180)
    plt.close(fig)


def plot_boxbands(df: pd.DataFrame) -> None:
    coastal = df.loc[df["coastal_panel"]].copy()
    coastal["_split"] = np.where(
        pd.to_numeric(coastal["shoreward_binary"], errors="coerce").eq(1),
        "shoreward",
        "non-shoreward",
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=False)
    for ax, metric, ylab in zip(
        axes,
        ["local_no2_excess", "maritime_exposure_index"],
        ["Local NO2 excess", "Maritime exposure index (rank)"],
    ):
        sns.boxplot(data=coastal, x="shipping_distance_band_tight", y=metric, hue="_split", ax=ax, palette="Pastel2")
        ax.set_xlabel("")
        ax.set_ylabel(ylab)
        ax.set_title("Fig5. Distribution structure by distance band & wind regime")
        ax.tick_params(axis="x", rotation=15)
        ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    axes[-1].set_xlabel("Shipping-distance band")
    fig.tight_layout()
    fig.savefig(FIGURES / "Fig5_distance_band_boxplots.png", dpi=180)
    plt.close(fig)


def plot_dashboard(df: pd.DataFrame, coast_lat: np.ndarray, coast_lon: np.ndarray) -> None:
    coastal = df.loc[df["coastal_panel"]]
    band_means = []
    for b in BANDS:
        slab = coastal[coastal["shipping_distance_band_tight"] == b]
        band_means.append(
            pd.to_numeric(slab["maritime_exposure_index"], errors="coerce").mean(),
        )

    plt.rcParams["axes.titlesize"] = 9
    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0])

    ax0 = fig.add_subplot(gs[0, 0:2])
    step = max(1, len(coast_lat) // 380)
    ax0.plot(coast_lon[::step], coast_lat[::step], color="#063", lw=1.1, alpha=0.7)
    sca = coastal.groupby("grid_cell_id")[["maritime_exposure_index", "grid_centroid_lon", "grid_centroid_lat"]].mean().reset_index()
    hb0 = ax0.scatter(
        sca["grid_centroid_lon"],
        sca["grid_centroid_lat"],
        c=sca["maritime_exposure_index"],
        cmap="inferno",
        s=52,
        edgecolors="k",
        linewidths=0.25,
        zorder=3,
    )
    plt.colorbar(hb0, ax=ax0, shrink=0.55, fraction=0.046, pad=0.02, label="Maritime exposure")
    ax0.set_title("Integrated view: shoreline + exposure rank (coastal panel)")
    ax0.set_aspect("equal", adjustable="datalim")

    ax1 = fig.add_subplot(gs[0, 2])
    xi = np.arange(len(BANDS))
    ax1.bar(xi - 0.18, band_means, width=0.35, label="mean MEI", color="steelblue")
    ax1.set_xticks(xi)
    ax1.set_xticklabels(BANDS, rotation=35, fontsize=8)
    ax1.set_ylabel("Mean maritime exposure rank")
    ax1.set_title("Distance structuring (pooled)")
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[1, :])
    xi = np.arange(len(BANDS))
    sh_m: list[float] = []
    o_m: list[float] = []
    for b in BANDS:
        slab = coastal[coastal["shipping_distance_band_tight"] == b]
        sh = pd.to_numeric(
            slab.loc[pd.to_numeric(slab["shoreward_binary"], errors="coerce").eq(1), "maritime_exposure_index"],
            errors="coerce",
        )
        ot = pd.to_numeric(
            slab.loc[pd.to_numeric(slab["shoreward_binary"], errors="coerce").eq(0), "maritime_exposure_index"],
            errors="coerce",
        )
        sh_m.append(float(sh.mean(skipna=True)) if len(sh) else np.nan)
        o_m.append(float(ot.mean(skipna=True)) if len(ot) else np.nan)
    ax2.plot(xi, sh_m, "o-", color="darkgreen", label="Shoreward wind (cos>=cos45)")
    ax2.plot(xi, o_m, "s--", color="dimgray", label="Non-shoreward")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(BANDS, rotation=15)
    ax2.set_ylabel("Mean maritime exposure index")
    ax2.set_title("Wind-regime contrast along shipping-distance gradient")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Fig6. Thesis dashboard — coastal coupling, structuring, regimes (association language only)",
        fontsize=11,
    )
    gs.tight_layout(fig, pad=2.5)
    fig.savefig(FIGURES / "Fig6_environmental_dashboard.png", dpi=180)
    plt.close(fig)


def merge_wind_vectors(df: pd.DataFrame, wind_csv: Path | None) -> pd.DataFrame:
    """Left-merge Open-Meteo / transport wind fields from CSV; CSV values replace existing columns on key match."""
    if wind_csv is None or not Path(wind_csv).is_file():
        return df
    wf = pd.read_csv(wind_csv)
    need_cols = {"grid_cell_id", "week_start_utc", "wind_u_mean", "wind_v_mean"}
    if not need_cols.issubset(set(wf.columns)):
        print(f"[WARN] wind CSV missing columns; skipping wind merge: {wind_csv}")
        return df
    overlay = [
        "coastal_wind_alignment_score",
        "coastal_wind_angle_diff_deg",
        "coastal_wind_shoreward_45deg",
        "pollution_transport_wind_alignment_score",
        "pollution_transport_angle_diff_deg",
        "wind_speed_mean",
        "wind_direction_to_degrees",
    ]
    pick = list(need_cols)
    for c in overlay:
        if c in wf.columns and c not in pick:
            pick.append(c)
    wf = wf[pick].copy()
    wf["grid_cell_id"] = wf["grid_cell_id"].astype(str)
    wf["_wk_merge"] = pd.to_datetime(wf["week_start_utc"], utc=True).dt.normalize()
    out = df.copy()
    for c in pick:
        if c in ("grid_cell_id", "week_start_utc"):
            continue
        if c in out.columns:
            out = out.drop(columns=[c])
    out["_wk_merge"] = pd.to_datetime(out["week_start_utc"], utc=True).dt.normalize()
    merged = out.merge(
        wf.drop(columns=["week_start_utc"]),
        on=["grid_cell_id", "_wk_merge"],
        how="left",
    ).drop(columns=["_wk_merge"])
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--ne-cache", type=Path, default=_ROOT / "data" / "aux" / "natural_earth_coast_cache")
    ap.add_argument(
        "--wind-features-csv",
        type=Path,
        default=_ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv",
        help="Merge wind_u_mean / wind_v_mean for maps (optional)",
    )
    args = ap.parse_args()

    if not Path(args.input).is_file():
        print(f"[FATAL] missing {args.input}")
        return 1

    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df = merge_wind_vectors(df, args.wind_features_csv)

    df = prepare_panel(df, Path(args.ne_cache))

    df = build_indices(df)

    expo_tbl = index_summary(df)
    expo_tbl.to_csv(REPORTS / "exposure_indices_summary.csv", index=False)

    vals = [
        "local_no2_excess",
        "maritime_exposure_index",
        "atmospheric_coastal_exposure_index",
        "environmental_stress_index",
        "vessel_density_t",
        "coastal_wind_alignment_score",
    ]
    band_df = band_stats_table(df, vals)
    mw_df = mannwhitney_table(
        df,
        ["local_no2_excess", "maritime_exposure_index", "atmospheric_coastal_exposure_index", "environmental_stress_index"],
    )
    sp_df = spearman_block(df)
    summary_all = pd.concat(
        [
            band_df.assign(block="distance_band_boot_mean"),
            mw_df.assign(block="mannwhitney_cliffs_delta"),
            sp_df.assign(block="spearman"),
        ],
        ignore_index=True,
    )
    summary_all.to_csv(REPORTS / "coastal_exposure_statistics.csv", index=False)

    cp = load_coastline_points(Path(args.ne_cache)) or load_land_boundary_points(Path(args.ne_cache))
    if cp is None:
        coast_lat = np.array([])
        coast_lon = np.array([])
    else:
        coast_lat, coast_lon = cp

    sns.set_theme(style="whitegrid", font_scale=0.95)
    plot_decay(df)
    plot_heatmap(df)
    plot_hex(df)
    plot_boxbands(df)
    if len(coast_lat):
        plot_risk_map(df, coast_lat.astype(float), coast_lon.astype(float))
        plot_dashboard(df, coast_lat.astype(float), coast_lon.astype(float))

    captions = {
        "Fig1": "Exposure decay vs shipping-distance bands; split shoreward/non-shoreward with bootstrap CI on pooled means.",
        "Fig2": "Mean NO2 excess across vessel vs wind alignment quintiles (spatial environmental behaviour lens).",
        "Fig2b": "Mean Environmental Stress Index rank in the same quintile cross-classification.",
        "Fig3": "Hexbin of coastal wind alignment vs local NO2 excess with Spearman caption.",
        "Fig4": "Integrated map — exposure rank, mean wind vectors, NO2/oil hotspots, vessel density.",
        "Fig5": "Boxplots quantify distributional structuring by band × wind regime.",
        "Fig6": "Dashboard: spatial exposure, pooled band means, shoreline wind contrast along gradient.",
    }
    md_body = expo_tbl.copy()
    try:
        import tabulate as _tb  # noqa: F401
    except ImportError:
        pass
    summary_all_trunc = summary_all.copy()

    lines = [
        "# Coastal exposure analysis (thesis adjunct)",
        "",
        "**Framing:** exposure association / directional coupling / coastal influence structuring — not causal maritime transport.",
        "",
        "## Index definitions",
        "- **Maritime Exposure Index**: rank(vessel)×rank(ReLU(wind_alignment))×rank(1/(1+d_coast)); final rank to [0,1].",
        "- **Atmospheric Coastal Exposure**: rank(NO2 excess weekly)×rank(aligned coastal wind)×rank(transport alignment)×rank(coast proximity); final rank.",
        "- **Environmental Stress (experimental):** mean weekly z(NO2,vessel,oil, ndti anomaly) plus centered wind-rank proxy; percentile rank.",
        "",
        "### `exposure_indices_summary.csv`",
        "",
        md_body.to_string(index=False),
        "",
        "### `coastal_exposure_statistics.csv` (first 55 rows excerpt)",
        "",
        summary_all_trunc.head(55).to_string(index=False),
        "",
        "### Figure captions",
    ]
    for k, v in captions.items():
        lines.append(f"- **{k}**: {v}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "coastal_exposure_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {REPORTS / 'exposure_indices_summary.csv'}")
    print(f"[OK] {REPORTS / 'coastal_exposure_statistics.csv'}")
    print(f"[OK] {REPORTS / 'coastal_exposure_analysis_summary.md'}")
    print(f"[OK] figures -> {FIGURES}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
