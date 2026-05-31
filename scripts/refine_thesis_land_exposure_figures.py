#!/usr/bin/env python3
"""Publication refinements for thesis land-exposure figures (re-plot from masked panel; no pipeline recompute)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Patch, Polygon as MplPolygon
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_thesis_land_exposure_figures as gle  # noqa: E402

OUT = ROOT / "outputs/thesis_land_exposure_refined"

N_MIN_SUMMARY = 10
N_MIN_VIOLIN = 20

DPI = 360

CAPTION_MD = OUT / "thesis_caption_notes.md"
INTERP_MD = OUT / "thesis_interpretation_notes.md"


def _style_pub() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
            "font.size": 10.55,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.25,
            "legend.fontsize": 8.95,
            "xtick.labelsize": 9.15,
            "ytick.labelsize": 9.15,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
            "grid.linewidth": 0.65,
        },
    )


def savefig(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / stem
    fig.savefig(p.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(p.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight", pad_inches=0.065, facecolor="white")
    plt.close(fig)


def add_north_arrow(ax: plt.Axes, *, x: float = 0.93, y: float = 0.12, len_ax: float = 0.065) -> None:
    ax.annotate(
        "",
        xy=(x, y + len_ax),
        xytext=(x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.35, mutation_scale=14),
        zorder=500,
    )
    ax.text(x, y + len_ax + 0.02, "N", ha="center", va="bottom", fontsize=10.5, fontweight="bold", transform=ax.transAxes, color="#111827")


def fig_a_gradient_refined(df: pd.DataFrame, audit: list[dict[str, Any]]) -> None:
    """Mean ± SE per band × port; omit points with n<N_MIN_SUMMARY; connect consecutive bands only."""

    metrics = [
        ("NO2_mean", "Mean NO₂"),
        ("maritime_pressure_index", "Mean MEI"),
        ("coastal_exposure_score", "Mean CES"),
        ("environmental_stress_index", "Mean ESI"),
    ]

    active_metrics: list[tuple[str, str]] = []
    for col, title in metrics:
        ok = False
        for pt in gle.PORTS_ORDER:
            for bl in gle.LAND_BAND_LABELS_ORDER:
                ser = df.loc[(df["_land_coast_band"].astype(str).eq(bl)) & df["nearest_port"].astype(str).eq(pt), col]
                _, _, n = gle.mean_se(ser)
                if n >= N_MIN_SUMMARY:
                    ok = True
                    break
            if ok:
                break
        if ok:
            active_metrics.append((col, title))

    if not active_metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.text(0.5, 0.5, f"No panels meet n ≥ {N_MIN_SUMMARY}", ha="center", va="center")
        ax.axis("off")
        savefig(fig, "fig_landward_gradient_refined")
        return

    n_p = len(active_metrics)
    ncols = 2
    nrows = int(math.ceil(n_p / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.4, 4.15 * nrows + 0.6))
    axes = np.atleast_2d(axes).ravel()

    cmap = sns.color_palette("colorblind", n_colors=len(gle.PORTS_ORDER))
    band_ix = {b: i for i, b in enumerate(gle.LAND_BAND_LABELS_ORDER)}

    for ax_i, (col, ylab) in enumerate(active_metrics):
        ax = axes[ax_i]
        valid_bands = [
            bl
            for bl in gle.LAND_BAND_LABELS_ORDER
            if any(
                gle.mean_se(
                    df.loc[(df["_land_coast_band"].astype(str).eq(bl)) & df["nearest_port"].astype(str).eq(pt), col],
                )[2]
                >= N_MIN_SUMMARY
                for pt in gle.PORTS_ORDER
            )
        ]
        if not valid_bands:
            ax.set_visible(False)
            continue

        xpos = np.arange(len(valid_bands), dtype=float)

        for pj, pt in enumerate(gle.PORTS_ORDER):
            ys: list[float] = []
            es: list[float] = []
            ns: list[int] = []
            xplot: list[float] = []

            for xi, bl in enumerate(valid_bands):
                ser = df.loc[(df["_land_coast_band"].astype(str).eq(bl)) & df["nearest_port"].astype(str).eq(pt), col]
                mu, se_, n = gle.mean_se(ser)
                grids = int(df.loc[(df["_land_coast_band"].astype(str).eq(bl)) & df["nearest_port"].astype(str).eq(pt), "grid_cell_id"].nunique())
                wk = int(df.loc[(df["_land_coast_band"].astype(str).eq(bl)) & df["nearest_port"].astype(str).eq(pt), "week_start_utc"].nunique())
                audit.append(dict(figure="A_refined", metric=col, port=pt, band=bl, n_rows=n, n_grids=grids, n_weeks=wk))
                if n < N_MIN_SUMMARY:
                    continue
                ys.append(mu)
                es.append(se_ if se_ == se_ else 0.0)
                ns.append(n)
                xplot.append(float(xi))

            if not ys:
                continue

            yerr_u = [
                float(s) if (s == s and math.isfinite(float(s))) else 0.0 for s in es
            ]

            ax.errorbar(
                xplot,
                ys,
                yerr=yerr_u,
                fmt="o",
                capsize=4.0,
                color=cmap[pj],
                linewidth=1.82,
                label=pt,
                markeredgecolor="#111827",
                markeredgewidth=0.5,
                ecolor="#4b5563",
                alpha=0.98,
                zorder=25,
            )

            for xv, ym, ym_se, nv in zip(xplot, ys, es, ns, strict=True):
                if ym == ym and math.isfinite(ym):
                    ax.annotate(f"n={nv}", (xv, ym + ym_se + 1e-9), fontsize=7.85, ha="center", va="bottom", color="#475569")

            for j in range(len(xplot) - 1):
                b0 = valid_bands[int(xplot[j])]
                b1 = valid_bands[int(xplot[j + 1])]
                if band_ix[b1] - band_ix[b0] != 1:
                    continue
                ax.plot([xplot[j], xplot[j + 1]], [ys[j], ys[j + 1]], "-", color=cmap[pj], linewidth=1.35, alpha=0.78, zorder=20)

        ax.set_xticks(xpos)
        ax.set_xticklabels(valid_bands)
        ax.set_title(ylab, pad=11)
        ax.set_xlabel("Shoreline distance (km)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    for ax_j in range(len(active_metrics), len(axes)):
        axes[ax_j].set_visible(False)

    leg_h, leg_l = [], []
    for ax in axes[: len(active_metrics)]:
        if not ax.get_visible():
            continue
        h_, l_ = ax.get_legend_handles_labels()
        if h_:
            leg_h, leg_l = h_, l_
            break
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            title="nearest_port",
            loc="lower center",
            bbox_to_anchor=(0.48, -0.02),
            ncol=3,
            frameon=True,
            fontsize=9.0,
            title_fontsize=9.35,
        )

    fig.suptitle(
        "Landward exposure gradients across shoreline-distance bands",
        fontsize=13.05,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.985,
        "Mean ± SE; points plotted only where n≥10 weekly rows; segments connect consecutive distance bins when both endpoints qualify.",
        ha="center",
        fontsize=9.05,
        style="italic",
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14, hspace=0.45, wspace=0.32)
    savefig(fig, "fig_landward_gradient_refined")


def fig_b_wind_refined(df: pd.DataFrame, audit: list[dict[str, Any]]) -> None:
    metrics = [
        ("NO2_mean", "Mean NO₂"),
        ("maritime_pressure_index", "Mean MEI"),
        ("coastal_exposure_score", "Mean CES"),
        ("environmental_stress_index", "Mean ESI"),
    ]

    cmap_p = sns.color_palette("muted", n_colors=len(gle.PORTS_ORDER))
    seq = [(pt, wd) for pt in gle.PORTS_ORDER for wd in ("shoreward", "nonshoreward")]

    fig, axes = plt.subplots(2, 2, figsize=(12.45, 9.65))
    axes = axes.ravel()

    for ax_i, (col, title) in enumerate(metrics):
        ax = axes[ax_i]
        active_bands = [
            bl
            for bl in gle.LAND_BAND_LABELS_ORDER
            if any(
                gle.mean_se(
                    df.loc[
                        (df["_land_coast_band"].astype(str).eq(bl))
                        & (df["nearest_port"].astype(str).eq(pt))
                        & (df["_wind_regime"].astype(str).eq(wd)),
                        col,
                    ],
                )[2]
                >= N_MIN_SUMMARY
                for pt, wd in seq
            )
        ]

        if not active_bands:
            ax.set_visible(False)
            continue

        centers = np.arange(len(active_bands), dtype=float)
        n_grp = len(seq)
        group_span = 0.74
        bw = group_span / n_grp * 1.02
        ymax_l = -math.inf

        for bi, band_lab in enumerate(active_bands):
            for k, (pt, wd) in enumerate(seq):
                ser = df.loc[
                    (df["_land_coast_band"].astype(str).eq(band_lab))
                    & (df["nearest_port"].astype(str).eq(pt))
                    & (df["_wind_regime"].astype(str).eq(wd)),
                    col,
                ]
                mu, se_, n = gle.mean_se(ser)
                g_ = int(df.loc[(df["_land_coast_band"].astype(str).eq(band_lab)) & (df["nearest_port"].astype(str).eq(pt)) & (df["_wind_regime"].astype(str).eq(wd)), "grid_cell_id"].nunique())
                audit.append(dict(figure="B_refined", metric=col, port=pt, wind=wd, band=band_lab, n_rows=n, n_grids=g_))

                if n < N_MIN_SUMMARY:
                    continue

                xpos = bi - group_span / 2 + bw / 2 + k * bw
                ax.bar(
                    xpos,
                    mu if mu == mu else 0.0,
                    width=bw * 0.96,
                    color=cmap_p[gle.PORTS_ORDER.index(pt)],
                    edgecolor="#1f2937",
                    linewidth=0.5,
                    hatch="//" if wd == "shoreward" else "",
                    alpha=0.86 if wd == "shoreward" else 0.72,
                    yerr=(se_ if se_ == se_ and math.isfinite(float(se_)) else 0.0),
                    capsize=2.9,
                    error_kw=dict(ecolor="#475569", elinewidth=0.98),
                )
                if mu == mu:
                    ymax_l = max(ymax_l, mu + float(se_) if se_ == se_ else mu)
                    ax.text(xpos, (mu + (se_ if se_ == se_ else 0.0)), f"n={n}", fontsize=7.55, ha="center", va="bottom", color="#475569", rotation=90)

        ax.set_xticks(centers)
        ax.set_xticklabels(active_bands, rotation=22, ha="right")
        ax.set_title(title, pad=10)
        ax.set_xlabel("Shoreline distance")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _t: f"{v:.3g}"))
        if ymax_l > -math.inf and ymax_l > 0:
            ax.set_ylim(0.0, ymax_l * 1.16)

    h_w = (
        Patch(facecolor="0.85", edgecolor="#1f2937", hatch="//", label="Shoreward"),
        Patch(facecolor="0.85", edgecolor="#1f2937", label="Non-shoreward"),
    )
    leg1 = fig.legend(
        handles=list(h_w),
        loc="upper left",
        bbox_to_anchor=(1.01, 0.98),
        borderaxespad=0.0,
        frameon=True,
        title="Wind regime",
        fontsize=9.0,
    )
    fig.add_artist(leg1)
    sq = tuple(
        mlines.Line2D([0], [0], linestyle="none", marker="s", markersize=8, color=cmap_p[i], label=gle.PORTS_ORDER[i], markeredgecolor="#111827")
        for i in range(len(gle.PORTS_ORDER))
    )
    fig.legend(handles=list(sq), loc="upper left", bbox_to_anchor=(1.01, 0.58), borderaxespad=0.0, frameon=False, title="Port")

    fig.suptitle("Shoreward vs non-shoreward by shoreline band", fontsize=12.55, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.075, right=0.78, top=0.91, bottom=0.11, hspace=0.34, wspace=0.24)
    savefig(fig, "fig_wind_land_exposure_refined")


def fig_violin_combined_refined(df: pd.DataFrame, audit: list[dict[str, Any]], appendix_flag: list[bool]) -> None:
    metrics = [
        ("NO2_mean", "NO₂"),
        ("maritime_pressure_index", "MEI"),
        ("coastal_exposure_score", "CES"),
        ("environmental_stress_index", "ESI"),
    ]

    panels: list[tuple[str, str, pd.DataFrame]] = []
    for col, slab in metrics:
        sub = df.copy()
        sub["_band"] = pd.Categorical(sub["_land_coast_band"], categories=gle.LAND_BAND_LABELS_ORDER, ordered=True)
        if len(sub) < N_MIN_VIOLIN:
            continue
        active_bands = []
        for bl in gle.LAND_BAND_LABELS_ORDER:
            m = sub["_land_coast_band"].astype(str).eq(bl) & sub["_wind_regime"].isin(["shoreward", "nonshoreward"])
            if int(m.sum()) >= N_MIN_VIOLIN:
                active_bands.append(bl)
        if not active_bands:
            continue
        sub2 = sub[sub["_land_coast_band"].astype(str).isin(active_bands) & sub["_wind_regime"].isin(["shoreward", "nonshoreward"])].copy()
        if len(sub2) < N_MIN_VIOLIN:
            continue
        panels.append((col, slab, sub2))
        audit.append(dict(figure="C_violin_refined", metric=col, n_rows=int(len(sub2)), bands=",".join(active_bands)))

    if not panels:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.text(0.5, 0.5, f"No violin panel meets n ≥ {N_MIN_VIOLIN}.", ha="center", va="center")
        ax.axis("off")
        appendix_flag.append(True)
        savefig(fig, "fig_wind_land_violin_refined")
        return

    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.35, 3.35 * nrows + 1.1))
    axes = np.atleast_1d(axes)

    for ax, (col, slab, sub2) in zip(axes, panels, strict=True):
        vv = pd.to_numeric(sub2[col], errors="coerce")
        lo, hi = float(vv.quantile(0.01)), float(vv.quantile(0.99))
        sub2["_yc"] = vv.clip(lo, hi)
        kws = dict(
            data=sub2,
            x="_land_coast_band",
            y="_yc",
            hue="_wind_regime",
            ax=ax,
            order=[b for b in gle.LAND_BAND_LABELS_ORDER if b in sub2["_land_coast_band"].astype(str).unique()],
            hue_order=["shoreward", "nonshoreward"],
            cut=0,
            inner="quartile",
            density_norm="width",
            palette={"shoreward": "#2f5597", "nonshoreward": "#c4a6d8"},
            linewidth=1.05,
            saturation=0.94,
            legend=False,
        )
        try:
            sns.violinplot(**kws, gap=0.12)
        except TypeError:
            sns.violinplot(**kws)

        ax.set_title(f"{slab} · shore vs nonshore (combined ports, 1–99% trim)")
        ax.set_xlabel("Shoreline distance")
        ax.set_ylabel(slab)

        ttl_n = len(sub2)
        g_n = sub2["grid_cell_id"].nunique()
        w_n = sub2["week_start_utc"].nunique()
        ax.text(
            0.02,
            0.98,
            f"n={ttl_n:,} weekly rows · {g_n:,} grids · {w_n:,} weeks",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.95,
            color="#475569",
        )

    if nrows <= 2:
        appendix_flag.append(True)
    else:
        appendix_flag.append(False)

    legend_handles = (
        Patch(facecolor="#2f5597", edgecolor="#111827", label="Shoreward"),
        Patch(facecolor="#c4a6d8", edgecolor="#111827", label="Non-shoreward"),
    )
    fig.legend(handles=list(legend_handles), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=10.05, frameon=False)

    fig.suptitle("Wind-stratified distributions by shoreline distance (merged ports)", fontsize=12.95, fontweight="bold")
    if appendix_flag[-1]:
        fig.text(
            0.5,
            0.006,
            "Appendix recommended: limited panels survived n≥20 screening.",
            fontsize=9.05,
            ha="center",
            color="#92400e",
            style="italic",
        )

    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.08, top=0.94, hspace=0.45)
    savefig(fig, "fig_wind_land_violin_refined")


def fig_e_crossport_refined(df: pd.DataFrame, audit: list[dict[str, Any]]) -> None:

    def slope_row(pt: str) -> tuple[float, float, int]:
        slope_, wmse, _r2, _wsum, n_bnds = gle._mei_decay_slope(df, pt)
        se_est = math.sqrt(max(float(wmse), 0.0)) if wmse == wmse else float("nan")
        sup_rows = int(
            pd.to_numeric(
                df.loc[df["_land_coast_band"].notna() & df["nearest_port"].astype(str).eq(pt), "maritime_pressure_index"],
                errors="coerce",
            ).notna().sum(),
        )
        return slope_, float(se_est) if se_est == se_est else 0.0, sup_rows

    titles = [
        ("mean_ces", "Mean CES", lambda p: gle.mean_se(df.loc[df["nearest_port"].astype(str).eq(p), "coastal_exposure_score"])),
        ("mean_esi", "Mean ESI", lambda p: gle.mean_se(df.loc[df["nearest_port"].astype(str).eq(p), "environmental_stress_index"])),
        ("no2_rat", "NO₂ ratio (far / near shore)", lambda p: _no2_ratio(df, p)),
        ("mei_slope", "MEI vs shoreline distance slope", slope_row),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.95, 8.95))
    axes = axes.ravel()
    x = np.arange(len(gle.PORTS_ORDER), dtype=float)

    cmap = sns.color_palette("colorblind", n_colors=len(gle.PORTS_ORDER))

    for ax, (_key, ttl, getter) in zip(axes, titles, strict=True):
        vals: list[float] = []
        errs: list[float] = []
        sups: list[int] = []
        for pt in gle.PORTS_ORDER:
            mu, se_, sup = getter(pt)
            sup = int(sup)
            vals.append(float(mu) if mu == mu else float("nan"))
            errs.append(float(se_) if se_ == se_ else 0.0)
            sups.append(sup)
            audit.append(dict(figure="E_refined", panel=ttl, port=pt, value=mu, se=se_, support=sup))

        finite_heights = [
            (vals[i] + errs[i])
            for i in range(len(vals))
            if sups[i] >= N_MIN_SUMMARY and vals[i] == vals[i] and math.isfinite(vals[i])
        ]
        ymax = max(finite_heights) if finite_heights else 1.0
        ymax *= 1.12
        ymin_candidates = [0.0] + [
            (vals[i] - errs[i])
            for i in range(len(vals))
            if sups[i] >= N_MIN_SUMMARY and vals[i] == vals[i] and math.isfinite(vals[i])
        ]
        ymin = min(ymin_candidates) if ymin_candidates else 0.0

        has_any = False
        rng = ymax - ymin if ymax > ymin else 1.0
        for xi, (mu, se_, sup, cl) in enumerate(zip(vals, errs, sups, cmap, strict=False)):
            if sup >= N_MIN_SUMMARY and mu == mu and math.isfinite(mu):
                has_any = True
                ax.bar(
                    float(xi),
                    mu,
                    width=0.46,
                    color=cl,
                    edgecolor="#111827",
                    linewidth=0.55,
                    yerr=float(se_) if se_ == se_ and math.isfinite(se_) else 0.0,
                    capsize=3.05,
                    error_kw=dict(ecolor="#475569"),
                )
                ax.text(float(xi), ymin - 0.068 * rng, f"n={sup}", ha="center", va="top", fontsize=8.2, color="#1f2937")
            elif sup > 0 and mu == mu and math.isfinite(mu):
                ax.bar(
                    float(xi),
                    mu,
                    width=0.46,
                    color="#e5e7eb",
                    edgecolor="#9ca3af",
                    linewidth=0.72,
                    hatch="///",
                    alpha=0.65,
                )
                ax.text(float(xi), ymin - 0.068 * rng, f"n={sup} (<thr)", ha="center", va="top", fontsize=8.0, color="#b45309")
            else:
                ax.text(float(xi), ymin - 0.05 * rng, "omitted", ha="center", va="top", fontsize=8.0, color="#9ca3af")

        ax.set_xticks(x)
        ax.set_xticklabels(gle.PORTS_ORDER)
        ax.axhline(0, color="#9ca3af", linewidth=0.72, linestyle=":", zorder=0)
        ax.set_title(f"{ttl}\n(descriptive comparison)", fontsize=10.65, pad=11)
        ax.set_xlabel("")
        ax.set_ylabel("Value")

        ax.set_ylim(ymin - 0.09 * rng, ymax + 0.08 * rng)

        ax.text(
            0.02,
            1.02,
            f"Thr. n≥{N_MIN_SUMMARY} for full-colour bars.",
            transform=ax.transAxes,
            fontsize=8.0,
            ha="left",
            color="#6b7280",
            style="italic",
        )

        if not has_any:
            ax.text(0.5, 0.5, "No port meets support threshold", ha="center", va="center", transform=ax.transAxes, color="#b45309")

    fig.suptitle("Cross-port coastal exposure comparison", fontsize=13.0, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Descriptive contrasts; hatched tiles use sub-threshold support.",
        ha="center",
        fontsize=9.1,
        color="#4b5563",
        style="italic",
    )
    fig.subplots_adjust(left=0.10, bottom=0.092, top=0.90, right=0.96, hspace=0.40, wspace=0.33)
    savefig(fig, "fig_cross_port_land_exposure_refined")


def _no2_ratio(df: pd.DataFrame, pt: str) -> tuple[float, float, int]:
    blk = df[df["_land_coast_band"].notna() & df["nearest_port"].astype(str).eq(pt)].copy()
    inner = blk[blk["_land_coast_band"].astype(str).eq(gle.LAND_BAND_LABELS_ORDER[0])]
    outer = blk[blk["_land_coast_band"].astype(str).eq(gle.LAND_BAND_LABELS_ORDER[3])]
    mn, _, n_in = gle.mean_se(inner["NO2_mean"])
    mf, _, n_out = gle.mean_se(outer["NO2_mean"])
    if not math.isfinite(mn) or abs(mn) < 1e-9:
        return float("nan"), 0.0, int(n_in + n_out)
    return float(mf / mn), 0.0, int(n_in + n_out)


def fig_hotspots_refined(analysis: pd.DataFrame) -> None:
    ves_g = analysis.groupby("grid_cell_id", observed=False)["vessel_density_t"].median().astype(float)
    thr_cor = float(np.percentile(ves_g.dropna().values, gle.CORRIDOR_PCT_GLOBAL)) if ves_g.dropna().size else float("nan")
    corridor_cells = set(ves_g.index[ves_g >= thr_cor].astype(str))

    rows = gle._agg_land_cells(analysis)

    for pt in gle.PORTS_ORDER:
        m = rows["nearest_port"].eq(pt)
        if not bool(m.any()):
            continue
        th_c = float(rows.loc[m, "ces_med"].quantile(0.90)) if rows.loc[m, "ces_med"].notna().any() else math.nan
        th_e = float(rows.loc[m, "esi_med"].quantile(0.90)) if rows.loc[m, "esi_med"].notna().any() else math.nan
        rows.loc[m, "_thr90_ces"] = th_c
        rows.loc[m, "_thr90_esi"] = th_e

    rows["hot_ces_port_scope"] = rows["ces_med"] >= rows["_thr90_ces"]
    rows["hot_esi_port_scope"] = rows["esi_med"] >= rows["_thr90_esi"]

    zooms = dict(
        Turku=dict(lon_min=21.07, lon_max=23.28, lat_min=59.74, lat_max=61.92),
        Mariehamn=dict(lon_min=18.74, lon_max=20.32, lat_min=59.84, lat_max=60.44),
    )
    lw_coast = {k: zooms[k]["lon_max"] - zooms[k]["lon_min"] for k in zooms}
    lw_lat = {k: zooms[k]["lat_max"] - zooms[k]["lat_min"] for k in zooms}
    targ = max(lw_coast.values())
    land_union = gle.load_land_union()
    fig = plt.figure(figsize=(14.5, 8.82))

    for ip, pname in enumerate(gle.PORTS_ORDER):
        bb = zooms[pname]
        if pname not in rows["nearest_port"].astype(str).values and rows[rows["nearest_port"].astype(str).eq(pname)].empty:
            continue
        cur_w = bb["lon_max"] - bb["lon_min"]
        expand = max(0.0, (targ - cur_w) / 2)
        bx = box(bb["lon_min"] - expand - 1e-3, bb["lat_min"] - 1e-3, bb["lon_max"] + expand + 1e-3, bb["lat_max"] + 1e-3)

        clip = gpd.clip(gpd.GeoDataFrame(geometry=gpd.GeoSeries([land_union], crs=gle.CRS_WGS84), crs=gle.CRS_WGS84), bx.buffer(8e-3))
        subset = rows[rows["nearest_port"].eq(pname)]
        has_data = len(subset) > 0

        for iy, (flag, clr, ttl_short) in enumerate(
            (
                ("hot_ces_port_scope", "#b45309", "CES hot"),
                ("hot_esi_port_scope", "#6b21a8", "ESI hot"),
            ),
        ):
            ax = plt.subplot2grid((2, 3), (iy, ip))
            if not has_data:
                ax.set_visible(False)
                continue

            if not clip.empty:
                clip.boundary.plot(ax=ax, lw=0.92, edgecolor="#1c1917", alpha=1.0, zorder=3)

            for _, r in subset.iterrows():
                lon_ = float(r["grid_centroid_lon"]) if pd.notna(r["grid_centroid_lon"]) else math.nan
                lat_ = float(r["grid_centroid_lat"]) if pd.notna(r["grid_centroid_lat"]) else math.nan
                if not math.isfinite(lon_) or not math.isfinite(lat_):
                    continue
                p = Point(lon_, lat_)
                if not bx.intersects(p):
                    continue

                gid = str(r["grid_cell_id"])
                rd = gle.parse_cell_resolution_deg(gid) * 1.12

                geom = gle.square_polygon(lon_, lat_, rd)
                hotspot = bool(r[flag])
                in_corridor = gid in corridor_cells

                face = clr if hotspot else "#e8edf4"
                edc = "#1f2937" if hotspot else "#94a3b8"
                lw_poly = 1.72 if hotspot else 0.32 + (0.95 if in_corridor else 0)

                gsp = gpd.GeoSeries([geom], crs=gle.CRS_WGS84)
                gx = gsp.clip(box(bb["lon_min"] - expand - 0.02, bb["lat_min"] - 0.02, bb["lon_max"] + expand + 0.02, bb["lat_max"] + 0.02))
                gx.plot(ax=ax, facecolor=face, edgecolor=edc, linewidth=lw_poly, alpha=1.0, zorder=6)

                if in_corridor:
                    gx.plot(ax=ax, facecolor="none", edgecolor="#0f172a", linewidth=1.05, linestyle=(0, (3.0, 1.85)), alpha=1.0, zorder=10)

            plat, plon = gle.PORT_COORDS[pname]
            ax.scatter([plon], [plat], s=154, marker="*", c="#facc15", edgecolors="#111827", linewidths=0.55, zorder=120)

            lon_span = bb["lon_max"] - bb["lon_min"] + 2 * expand
            mid_lon = (bb["lon_min"] + bb["lon_max"]) / 2
            lon_min_a = mid_lon - lon_span / 2
            lon_max_a = mid_lon + lon_span / 2
            lat_mid = (bb["lat_min"] + bb["lat_max"]) / 2
            lat_half = targ * (lw_lat[pname] / lw_coast[pname]) / 2
            ax.set_xlim(lon_min_a, lon_max_a)
            ax.set_ylim(lat_mid - lat_half, lat_mid + lat_half)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{pname} · {ttl_short}")
            ax.tick_params(axis="x", labelsize=8.95)
            if iy == 1:
                ax.set_xlabel("Longitude (°E)")
            if ip == 0:
                ax.set_ylabel("Latitude (°N)")

    corr_l = mlines.Line2D([], [], linestyle=(0, (3.0, 1.85)), linewidth=2.1, color="#0f172a", label=f"Corridor rims (median vessel ≥P{gle.CORRIDOR_PCT_GLOBAL})")
    fig.legend(handles=[corr_l], ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.003), fontsize=9.05, frameon=True)
    fig.suptitle("Port-adjacent exposure hotspots · discrete lattice", fontsize=13.0, fontweight="bold")
    fig.subplots_adjust(left=0.055, bottom=0.098, top=0.93, right=0.986, hspace=0.30, wspace=0.22)
    savefig(fig, "fig_land_hotspots_refined")


def fig_transport_refined(analysis: pd.DataFrame) -> None:
    uw = (
        analysis.groupby("grid_cell_id", observed=False)
        .agg(
            u_med=("wind_u_mean", lambda s: pd.to_numeric(s, errors="coerce").median()),
            v_med=("wind_v_mean", lambda s: pd.to_numeric(s, errors="coerce").median()),
            mei_m=("maritime_pressure_index", lambda s: pd.to_numeric(s, errors="coerce").median()),
            no2a_m=("_no2_weekly_anomaly", lambda s: pd.to_numeric(s, errors="coerce").median()),
        )
        .reset_index()
    )
    uniq = analysis.drop_duplicates("grid_cell_id")[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]].copy()
    uniq["grid_centroid_lat"] = pd.to_numeric(uniq["grid_centroid_lat"], errors="coerce")
    uniq["grid_centroid_lon"] = pd.to_numeric(uniq["grid_centroid_lon"], errors="coerce")
    uw = uw.merge(uniq.rename(columns=dict(grid_centroid_lat="lat", grid_centroid_lon="lon")), on="grid_cell_id")

    panel_cfgs = [
        dict(
            box_key="Turku",
            fld="no2a_m",
            cmap_name="PuOr_r",
            label="Median NO₂ weekly anomaly (panel residuals)",
            highlight_ports=("Turku",),
        ),
        dict(
            box_key="Aaland",
            fld="mei_m",
            cmap_name="viridis",
            label="Median MEI (`maritime_pressure_index`)",
            highlight_ports=("Mariehamn",),
        ),
    ]

    land_union = gle.load_land_union()
    fig, axes = plt.subplots(1, 2, figsize=(15.92, 7.85))

    for ax, cfg in zip(axes, panel_cfgs, strict=True):
        spec = gle._TRANSPORT_BOX[cfg["box_key"]]
        lon0, lon1, la0, la1 = float(spec["lon0"]), float(spec["lon1"]), float(spec["lat0"]), float(spec["lat1"])
        bbox = box(lon0 - 1e-3, la0 - 1e-3, lon1 + 1e-3, la1 + 1e-3)

        lc = gpd.clip(gpd.GeoDataFrame(geometry=gpd.GeoSeries([land_union], crs=gle.CRS_WGS84), crs=gle.CRS_WGS84), bbox.buffer(8e-3))
        if lc is not None and len(lc.index):
            lc.boundary.plot(ax=ax, lw=1.38, edgecolor="#030712", alpha=1.0, zorder=1)

        chunk = uw[(uw["lat"] >= la0 - 2e-2) & (uw["lat"] <= la1 + 2e-2) & (uw["lon"] >= lon0 - 2e-2) & (uw["lon"] <= lon1 + 2e-2)].copy()

        cmap = plt.get_cmap(cfg["cmap_name"])
        vals_all = pd.to_numeric(chunk[cfg["fld"]], errors="coerce")
        vmin = float(np.nanquantile(vals_all, 0.05))
        vmax = float(np.nanquantile(vals_all, 0.95))
        if not math.isfinite(vmin):
            vmin, vmax = 0.0, 1.0
        elif abs(vmax - vmin) < 1e-9:
            vmin, vmax = vmin - 0.05, vmin + 0.05
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        ax.set_aspect("equal")
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(la0, la1)
        ax.set_autoscale_on(False)

        quiver_rows = []
        for _, r in chunk.iterrows():
            lo = float(pd.to_numeric(r["lon"], errors="coerce"))
            la = float(pd.to_numeric(r["lat"], errors="coerce"))
            if not math.isfinite(lo) or not math.isfinite(la):
                continue

            gid = str(r["grid_cell_id"])
            geom = gle.square_polygon(float(lo), float(la), gle.parse_cell_resolution_deg(gid))

            centroid = geom.centroid
            pt = Point(centroid.x, centroid.y)
            if not bbox.intersects(pt):
                continue

            val = pd.to_numeric(r[cfg["fld"]], errors="coerce")
            if val != val or not math.isfinite(float(val)):
                fc = "#f3f4f6"
            else:
                fv = float(np.clip(norm(float(val)), 0.0, 1.0))
                fc = cmap(fv)

            coords = [(float(px), float(py)) for px, py in geom.exterior.coords if math.isfinite(float(px)) and math.isfinite(float(py))]
            poly = MplPolygon(coords, closed=True, facecolor=fc, edgecolor="#64748b", linewidth=0.3, alpha=1.0, zorder=3)
            poly.set_clip_on(True)
            ax.add_patch(poly)

            uu = pd.to_numeric(r["u_med"], errors="coerce")
            vv = pd.to_numeric(r["v_med"], errors="coerce")
            if math.isfinite(float(uu)) and math.isfinite(float(vv)):
                quiver_rows.append((float(lo), float(la), float(uu), float(vv)))

        if quiver_rows:
            q_arr = np.asarray(quiver_rows, dtype=float)[np.isfinite(np.asarray(quiver_rows)).all(axis=1)]
            stride = max(1, len(q_arr) // 52)
            q_sub = q_arr[::stride]
            spd = np.hypot(q_sub[:, 2], q_sub[:, 3])
            ref = float(np.nanquantile(spd, 0.85))
            scale = float(np.clip(max(360.0, ref * 24.0), 200.0, 2900.0))

            ax.quiver(
                q_sub[:, 0],
                q_sub[:, 1],
                q_sub[:, 2],
                q_sub[:, 3],
                angles="uv",
                scale=max(scale * 0.92, 50.0),
                width=8.2e-3,
                headwidth=4.55,
                headlength=7.05,
                color="#111827",
                alpha=0.82,
                zorder=45,
                label="Prevailing-flow proxy (median u,v)",
                clip_on=True,
            )

        for pname in cfg["highlight_ports"]:
            plat, plon = gle.PORT_COORDS[pname]
            if bbox.intersects(Point(plon, plat)):
                ax.scatter(plon, plat, s=142, marker="*", c="#facc15", edgecolors="#111827", linewidths=0.55, zorder=160)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm._A = []
        fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.048, pad=0.138, shrink=0.88).set_label(cfg["label"])

        ax.set_title(spec["title"], pad=13)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        add_north_arrow(ax, x=0.90, y=0.09, len_ax=0.07)

        q_art = mlines.Line2D(
            [0],
            [0],
            linestyle="none",
            marker=r"$\rightarrow$",
            color="#111827",
            markersize=11,
            label="Arrows = prevailing wind–alignment vectors (median u,v)",
        )
        leg = ax.legend(handles=[q_art], loc="upper right", fontsize=8.55, frameon=True, bbox_to_anchor=(0.989, -0.12))
        ax.add_artist(leg)

    fig.suptitle("Lattice wind vectors over composited exposure fields", fontsize=13.15, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Median NO₂ anomalies and MEI gradients",
        ha="center",
        fontsize=11.05,
        style="italic",
        color="#1f2937",
    )
    fig.subplots_adjust(bottom=0.18, left=0.055, top=0.865, right=0.98, wspace=0.16)
    savefig(fig, "fig_wind_transport_landward_refined")


def _write_docs(analysis: pd.DataFrame, audit: pd.DataFrame, appendix_note: bool) -> None:

    uniq = analysis[analysis["_land_coast_band"].notna()].groupby("nearest_port").agg(
        n_weekly=("week_start_utc", "count"),
        n_grids=("grid_cell_id", lambda s: s.nunique()),
    )

    blk = "".join([f"| {idx} | {int(r.n_weekly)} | {int(r.n_grids)} |\n" for idx, r in uniq.iterrows()])

    captions = rf"""## Figure A — fig_landward_gradient_refined

**Caption:** Landward exposure *gradients* (mean ± SE) across shoreline-distance bins for archival port assignments inside the masked coastal lattice; markers appear only where weekly row counts reach **n≥{N_MIN_SUMMARY}**; line segments bridge **consecutive bins** satisfying the same criterion (non-monotonic inland structure is intentional).

---

## Figure B — fig_wind_land_exposure_refined

**Caption:** Shoreward (hatched) versus non-shoreward bars by shoreline band; bars omitted when subgroup **n<{N_MIN_SUMMARY}**. Legends sit outside plotting margins to minimise occlusion.

---

## Figure C — fig_wind_land_violin_refined

**Caption:** Combined-port violins stratified by wind regime retain **median–quartile** enclosures and **1–99 % trims** only when pooled supports reach **≥{N_MIN_VIOLIN}**. Empty arrangements are suppressed.

{'**Placement note:** Because post-threshold visuals remain sparse, keep this figure near appendices/discussion auxiliary material.' if appendix_note else ''}

---

## Figure D — fig_cross_port_land_exposure_refined

**Caption:** Cross-port descriptive comparison annotated with pooled supports **`n`** under the shared receptor mask; hatched tiles mark **sub-threshold** support (**n<{N_MIN_SUMMARY}**) where summaries are nonetheless shown cautiously.

---

## Figure E — fig_land_hotspots_refined

**Caption:** Discrete coastal-port lattice emphasising CES/ESI hotspots (temporal-medians exceeding within-port **P90**); corridor rims denote Baltic-wide vessel-density percentile; polygons enlarged modestly (**+12 %** edge length vs nominal resolution).

---

## Figure F — fig_wind_transport_landward_refined

**Caption:** Lattice-scale median weekly NO₂ residuals (Turku) and MEI (Åland maritime window) coloured by percentile stretch; vectors encode **median u/v**. North arrow communicates orientation; captions stress non-causal, descriptive intent.

"""

    captions += "\n### Panel-level pooled supports (nearest_port slices in masked analysis)\n\n| Port | Weekly rows | Grids |\n|------|------------|-------|\n" + blk

    captions += "\n\nFull row-level thresholds logged in **`refined_support_audit.csv`**.\n"

    OUT.mkdir(parents=True, exist_ok=True)
    CAPTION_MD.write_text(captions.strip() + "\n", encoding="utf-8")

    interp = f"""## Interpretations (concise thesis paragraphs)

### A — Landward gradient figure
Bands summarize how NO₂/MEI/CES/ESI redistribute along shoreline chords while enforcing **n≥{N_MIN_SUMMARY}**. Read as *structure*, not monotone decay toward inland grids.

### B — Shoreward regimes
Contrast explores whether shoreward-aligned weeks shift distributional mass; sparse bins are withheld so differences are visually honest.

### C — Combined violins (appendix leaning)
Merged ports preserve statistical power yet sacrifice port-specific vignettes—appropriate for appendix style discussion when coastal slices are sparse.

### D — Cross-port bars
Panels benchmark Turku versus Mariehamn versus Stockholm descriptors (plus ratio/slope summaries) beneath explicit support tags; hatched tiles flag sub-threshold support.

### E — Hotspots
Hotspots localize where composite CES/ESI medians crest relative to contextual shipping corridors—interpret as **spatial coincidence**, not attributable burden.

### F — Wind transport schematic
Median wind vectors juxtapose anomalies/MEI contrasts on discrete cells useful for aligning atmospheric transport hypotheses with shoreline geometry **without asserting causality.**

### Methodological caveat
Upstream masking—including Stockholm’s archival extension on high-density-distance—is inherited unchanged from `generate_thesis_land_exposure_figures.py`; this refinement pass only adjusts **visual eligibility** thresholds (**{N_MIN_SUMMARY}/{N_MIN_VIOLIN}**).
"""
    INTERP_MD.write_text(interp.strip() + "\n", encoding="utf-8")


def main() -> None:
    _style_pub()
    OUT.mkdir(parents=True, exist_ok=True)

    analysis = gle.build_base_panel()
    df_ann = analysis[analysis["_land_coast_band"].notna()].copy()

    appendix: list[bool] = []
    audit_records: list[dict[str, Any]] = []

    fig_a_gradient_refined(df_ann, audit_records)
    fig_b_wind_refined(df_ann[df_ann["_wind_regime"].isin(["shoreward", "nonshoreward"])].copy(), audit_records)
    fig_violin_combined_refined(df_ann[df_ann["_wind_regime"].isin(["shoreward", "nonshoreward"])].copy(), audit_records, appendix)
    fig_e_crossport_refined(df_ann, audit_records)
    fig_hotspots_refined(analysis)
    fig_transport_refined(analysis)

    pd.DataFrame(audit_records).to_csv(OUT / "refined_support_audit.csv", index=False)
    _write_docs(df_ann, pd.DataFrame(audit_records), appendix_note=appendix[-1] if appendix else False)

    print(f"Refined artefacts → {OUT}")


if __name__ == "__main__":
    main()
