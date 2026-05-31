#!/usr/bin/env python3
"""Thesis Figures 5.5-5.10 and CSV artefacts.

Primary: outputs/5_2_to_5_10/thesis_figures/ , outputs/5_2_to_5_10/reports/
Mirror: outputs/thesis_figures/ , outputs/reports/ (same filenames)

MEI = maritime_pressure_index in ``features_ml_ready.parquet``.

Usage: python3 scripts/generate_thesis_sections_5_5_to_5_10.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
import matplotlib.dates as mdates


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from visualization.thesis_plot_ports import EXCLUDED_PORTS, PORT_COORDS, exclude_ports  # noqa: E402

PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
WIND_CSV = ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"

BASE = ROOT / "outputs" / "5_2_to_5_10"
FIGB = BASE / "thesis_figures"
REPB = BASE / "reports"
FIGM = ROOT / "outputs" / "thesis_figures"
REPM = ROOT / "outputs" / "reports"
OUT_FINAL = ROOT / "outputs" / "final_figures"

PORTS_LL = {k: v for k, v in PORT_COORDS.items() if k in ("Turku", "Mariehamn")}
PORTS_ORDER = list(PORTS_LL.keys())

WIND_DISTANCE_BANDS_ORDER = ["0–3 km", "3–7 km", "7–15 km", "15–30 km"]
MAP_BB = dict(lon0=14.05, lon1=25.2, lat0=55.05, lat1=62.25)
GID_RE = re.compile(r"^g(?P<r>[\d.]+)_")


def cell_deg(gid: str) -> float:
    m = GID_RE.match(str(gid))
    return float(m.group("r")) if m else 0.1


def dirs() -> None:
    FIGB.mkdir(parents=True, exist_ok=True)
    REPB.mkdir(parents=True, exist_ok=True)
    FIGM.mkdir(parents=True, exist_ok=True)
    REPM.mkdir(parents=True, exist_ok=True)
    OUT_FINAL.mkdir(parents=True, exist_ok=True)


def save_png(fig: plt.Figure, fname: str) -> None:
    # Avoid bbox_inches="tight" on large subplot grids: mpl can return a pathological
    # bbox (very tall) when x-ticklabels are rotated + tight_layout friction.
    kw = dict(dpi=340, bbox_inches=None, pad_inches=0.02, facecolor="white")
    fig.savefig(FIGB / fname, **kw)
    fig.savefig(FIGM / fname, **kw)
    plt.close(fig)


def save_png_pdf(fig: plt.Figure, stem: str) -> None:
    """Publication-quality PNG + PDF for thesis finals (dual output dirs)."""
    png_kw = dict(dpi=400, bbox_inches=None, pad_inches=0.04, facecolor="white")
    pdf_kw = dict(dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    for base in (FIGB, FIGM):
        fig.savefig(base / f"{stem}.png", **png_kw)
        fig.savefig(base / f"{stem}.pdf", **pdf_kw)
    plt.close(fig)


def write_dual_caption(stem: str, body_md: str) -> None:
    txt = body_md.strip() + "\n"
    for base in (FIGB, FIGM):
        (base / f"{stem}_caption.md").write_text(txt, encoding="utf-8")


def delete_old_outputs(names: tuple[str, ...]) -> None:
    for fname in names:
        for base in (FIGB, FIGM):
            p = base / fname
            if p.is_file():
                p.unlink()


def delete_final_outputs(stems_no_ext: tuple[str, ...]) -> None:
    OUT_FINAL.mkdir(parents=True, exist_ok=True)
    for stem in stems_no_ext:
        for suf in (f"{stem}.png", f"{stem}.pdf", f"{stem}_caption.md"):
            p = OUT_FINAL / suf
            if p.is_file():
                p.unlink()


def save_wrapped_final(fig: plt.Figure, stem: str) -> None:
    png_kw = dict(dpi=400, bbox_inches="tight", pad_inches=0.038, facecolor="white")
    pdf_kw = dict(dpi=300, bbox_inches="tight", pad_inches=0.036, facecolor="white")
    OUT_FINAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FINAL / f"{stem}.png", **png_kw)
    fig.savefig(OUT_FINAL / f"{stem}.pdf", **pdf_kw)
    plt.close(fig)


def write_final_caption(stem: str, body_md: str) -> None:
    OUT_FINAL.mkdir(parents=True, exist_ok=True)
    (OUT_FINAL / f"{stem}_caption.md").write_text(body_md.strip() + "\n", encoding="utf-8")


def wind_distance_band(km_series: pd.Series) -> pd.Series:
    km = pd.to_numeric(km_series, errors="coerce")
    labs = WIND_DISTANCE_BANDS_ORDER
    m0 = (km >= 0) & (km < 3)
    m1 = (km >= 3) & (km < 7)
    m2 = (km >= 7) & (km < 15)
    m3 = (km >= 15) & (km <= 30)
    out = pd.Series(pd.NA, index=km_series.index, dtype=object)
    out.loc[m0] = labs[0]
    out.loc[m1] = labs[1]
    out.loc[m2] = labs[2]
    out.loc[m3] = labs[3]
    return out


def aggregated_mean_se(series: pd.Series) -> tuple[float, float, int]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    if n == 0:
        return math.nan, math.nan, 0
    mu = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mu, se, n


def thesis_pub_style(*, thesis_final: bool = False) -> None:
    """Publication defaults: whitespace, subdued grid, typography, editable PDF fonts."""
    fz = 10.9 if thesis_final else 10.25
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#344054",
            "axes.labelcolor": "#1b2430",
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.92,
            "axes.labelpad": 5,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.32,
            "grid.linewidth": 0.72,
            "grid.linestyle": ":",
            "xtick.major.width": 0.82,
            "ytick.major.width": 0.82,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#293241",
            "ytick.color": "#293241",
            "lines.linewidth": 1.32,
            "lines.markeredgewidth": 0.72,
            "lines.markersize": 6.0,
            "patch.linewidth": 0.76,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "Liberation Sans", "sans-serif"],
            "font.size": fz,
            "axes.labelsize": fz,
            "axes.titlesize": fz + (1.3 if thesis_final else 1.05),
            "axes.titlepad": 9,
            "xtick.labelsize": fz - 1.05,
            "ytick.labelsize": fz - 1.05,
            "legend.fontsize": fz - (1.2 if thesis_final else 1.15),
            "legend.edgecolor": "#d5dae3",
            "legend.framealpha": 0.98,
            "legend.fancybox": False,
            "figure.dpi": 112,
            "savefig.dpi": 420,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_csv(df: pd.DataFrame, fname: str) -> None:
    df.to_csv(REPB / fname, index=False)
    df.to_csv(REPM / fname, index=False)


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True)
    df["mei"] = pd.to_numeric(df["maritime_pressure_index"], errors="coerce")

    if WIND_CSV.exists():
        w = pd.read_csv(WIND_CSV)
        w["week_start_utc"] = pd.to_datetime(w["week_start_utc"], utc=True)
        use = ["grid_cell_id", "week_start_utc", "wind_u_mean", "wind_v_mean"]
        if "coastal_wind_shoreward_45deg" in w.columns:
            use.append("coastal_wind_shoreward_45deg")
        if "coastal_wind_alignment_score" in w.columns:
            use.append("coastal_wind_alignment_score")
        w = w[[c for c in use if c in w.columns]]
        df = df.merge(w, on=["grid_cell_id", "week_start_utc"], how="left")

    sh = pd.to_numeric(df.get("coastal_wind_shoreward_45deg"), errors="coerce")
    df["wind_regime"] = np.where(sh >= 1, "shoreward", np.where(sh.notna(), "nonshoreward", pd.NA))
    df["wind_regime"] = df["wind_regime"].astype("object")
    df["nearest_port"] = df["nearest_port"].astype(str)
    return exclude_ports(df)


def bands_km(series: pd.Series) -> pd.Series:
    km = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(np.isnan(km), pd.NA, np.where(km < 3, "0-3 km", np.where(km < 10, "3-10 km", np.where(km < 50, "10-50 km", ">50 km")))), index=series.index)


def mean_se(s: pd.Series) -> tuple[float, float, int]:
    arr = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    nn = arr.size

    if nn == 0:
        return math.nan, math.nan, 0

    mn = float(np.mean(arr))

    se = float(np.std(arr, ddof=1) / math.sqrt(nn)) if nn > 1 else 0.0
    return mn, se, int(nn)


def fig55(df: pd.DataFrame) -> None:

    specs = [
        ("ndwi_mean", "NDWI"),
        ("ndti_mean", "NDTI"),
        ("ndci_mean", "NDCI"),
        ("fai_mean", "FAI"),
        ("NO2_mean", "NO2 mean"),

        ("vessel_density_t", "Vessel density"),

        ("mei", "Maritime Exposure Index"),
    ]

    rows = []

    data = []

    for col, label in specs:

        raw = pd.to_numeric(df[col], errors="coerce")

        cap = raw.quantile(0.99)

        clip = pd.to_numeric(raw.clip(upper=cap), errors="coerce").dropna()

        rows.append(
            dict(
                variable_label=label,
                column_name=col,
                missing_pct=round(raw.isna().mean() * 100, 4),
                mean=float(clip.mean()),
                median=float(clip.median()),
                std=float(clip.std(ddof=1)) if len(clip) > 1 else 0.0,
                min=float(clip.min()),
                max=float(clip.max()),
                skewness=float(scipy_stats.skew(clip.to_numpy())),
                kurtosis=float(scipy_stats.kurtosis(clip.to_numpy(), fisher=True)),
            )
        )


        data.append(clip.to_numpy())

    save_csv(pd.DataFrame(rows), "statistical_summary_table.csv")



    nrow = math.ceil(len(specs) / 2)



    fg, axes = plt.subplots(nrow, 2, figsize=(10.0, nrow * 2.9))





    axes = axes.flatten()


    for ix, axi in enumerate(axes):


        if ix >= len(specs):


            axi.axis("off")

            continue
        axi.boxplot(
            data[ix],
            widths=0.56,
            patch_artist=True,
            medianprops=dict(color="#17304d", linewidth=1.25),
            boxprops=dict(facecolor="#eaf1fb", edgecolor="#65789b", linewidth=0.92),
            whiskerprops=dict(color="#65789b", linewidth=0.85),
            capprops=dict(color="#65789b", linewidth=0.85),
            flierprops=dict(marker="o", markerfacecolor="#9aaabf", markersize=2.9, linestyle="none", alpha=0.72),
        )
        axi.set_xticks([1])


        axi.set_xticklabels([specs[ix][1]])
        sns.despine(ax=axi)

    fg.suptitle(
        "Fig. 5.5 · Distributional summaries (values capped at pooled 99th percentile)",
        fontsize=12.1,
        fontweight="semibold",
        color="#172133",
        y=0.992,
    )
    save_png(fg, "figure_5_5_statistical_boxplots.png")



def fig56(df: pd.DataFrame) -> None:
    cols = dict(
        mei="Maritime Exposure Index",
        NO2_mean="NO2 mean",
        vessel_density_t="Vessel density",
        coastal_exposure_score="Coastal exposure score",

    )


    zp = df.copy()

    zp["band"] = bands_km(zp["distance_to_port_km"])

    zp = zp[zp[list(cols)].notna().all(axis=1)]

    zon = ["0-3 km", "3-10 km", "10-50 km", ">50 km"]


    rec = []

    for pr in PORTS_LL:



        sx = zp[zp["nearest_port"].eq(pr)]
        for zb in zon:





            bx = sx[sx["band"].astype(str).eq(zb)]


            if bx.empty:


                continue


            for cn, lbl in cols.items():



                mu, sse, nc = mean_se(bx[cn])

                rec.append(dict(port=pr, distance_band=zb, metric=lbl, n=nc, mean=mu, stderr=sse))



    agg = pd.DataFrame(rec).sort_values(["metric", "port", "distance_band"])
    save_csv(agg, "distance_decay_statistics.csv")

    xi = np.arange(len(zon))





    pal = sns.color_palette("deep", len(PORTS_LL))



    if agg.empty:






        fg, ax = plt.subplots(figsize=(5.5, 2.4))



        ax.text(0.5, 0.5, "No decay strata")


        plt.axis("off")

        save_png(fg, "figure_5_6_distance_decay.png")

        save_png(fg, "figure_5_6_portwise_decay.png")

        return

    met_list = list(agg["metric"].unique())


    fg, axe = plt.subplots(max(1, len(met_list)), 1, figsize=(9.1, len(met_list) * 2.75))






    axe = np.atleast_1d(axe).flatten()


    ofs = np.linspace(-0.1, 0.1, len(PORTS_LL))


    for kk, mt in enumerate(met_list):


        ax = axe[kk]


        gg = agg[agg["metric"].eq(mt)]


        for j, pt in enumerate(PORTS_LL):


            gx = gg[gg["port"].eq(pt)].set_index("distance_band").reindex(zon)

            ym = gx["mean"].astype(float)



            ee = gx["stderr"].astype(float).fillna(0.0)



            ax.errorbar(xi + ofs[j], ym.to_numpy(), yerr=ee.to_numpy(), lw=1, marker="s", capsize=3, label=pt, color=pal[j])

            for ia, xv in enumerate(xi + ofs[j]):


                ni = gx["n"].iloc[ia]


                yi = ym.iloc[ia]


                if pd.notna(ni) and int(ni) > 0 and pd.notna(yi):

                    ax.annotate(str(int(ni)), (xv, float(yi)), fontsize=7, color=pal[j], ha="center", va="bottom")

        ax.set_xticks(xi)


        ax.set_xticklabels(zon)


        ax.set_title(mt)

        sns.despine(ax=ax)











    axe[0].legend(fontsize=8, ncol=3)


    fg.suptitle("Fig. 5.6 Combined decay (mean +- SE)")



















    fg.tight_layout(rect=[0, 0.01, 1, 0.97])


    save_png(fg, "figure_5_6_distance_decay.png")



















    fg2, axe2 = plt.subplots(len(PORTS_LL), len(met_list), figsize=(len(met_list) * 3.05, len(PORTS_LL) * 3.0))


    for ii, pt in enumerate(PORTS_LL):


        for jj, mt in enumerate(met_list):

            ax = axe2[ii, jj]


            gx = agg[(agg["port"].eq(pt)) & (agg["metric"].eq(mt))].set_index("distance_band").reindex(zon)


            ym = gx["mean"].astype(float)



            ee = gx["stderr"].astype(float).fillna(0.0)



            ax.errorbar(xi, ym, yerr=ee, marker="s", capsize=3, lw=1.12, color=pal[ii])


            for k, xv in enumerate(xi):

                nk = gx["n"].iloc[k]


                yk = ym.iloc[k]


                if pd.notna(nk) and nk and pd.notna(yk):







                    ax.annotate(str(int(nk)), (xv, float(yk)), fontsize=7, xytext=(0, 8), va="bottom", ha="center", color=pal[ii])



















            ax.set_xticks(xi)


            ax.set_xticklabels(zon, fontsize=8, rotation=18, ha="right")


            if ii == 0:


                ax.set_title(mt)



            if jj == 0:


                ax.set_ylabel(f"{pt}\\nMean +- SE")

            sns.despine(ax=ax)



















    fg2.suptitle("Fig. 5.6 Port-wise facets")



















    fg2.tight_layout(rect=[0, 0.01, 1, 0.96])


    save_png(fg2, "figure_5_6_portwise_decay.png")




def fig57(df: pd.DataFrame) -> None:


    stems = ("fig_5_7a_wind_regime_distributions", "fig_5_7b_wind_regime_mean_comparison")
    delete_old_outputs(
        (
            "figure_5_7_wind_regime_analysis.png",
            "figure_5_7_wind_regime_analysis.pdf",
        ),
    )
    delete_final_outputs(stems)

    required_src = dict(
        maritime_pressure_index="Maritime Exposure Index",
        vessel_density_t="Vessel density",
        atmospheric_transfer_index="Atmospheric transfer index",
        NO2_mean="NO₂ mean",
    )
    if not all(col in df.columns for col in required_src):
        miss = ",".join(c for c in required_src if c not in df.columns)
        stub = f"Wind-regime finals skipped (missing columns: {miss})"
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        save_csv(pd.DataFrame([dict(note=f"missing_panel_columns::{miss}")]), "wind_regime_statistics.csv")
        return

    base = df.copy()
    wk_med = pd.to_numeric(
        base.groupby("week_start_utc", sort=False)["NO2_mean"].transform("median"),
        errors="coerce",
    )
    no2_lin = pd.to_numeric(base["NO2_mean"], errors="coerce")
    base["_no2_excess"] = no2_lin - wk_med
    base["_band"] = wind_distance_band(base["distance_to_port_km"])
    base = base.dropna(subset=["_band"])
    ww = base[base["wind_regime"].isin(["shoreward", "nonshoreward"])]
    ww = ww[ww["nearest_port"].isin(list(PORTS_ORDER))]

    if ww.empty:
        stub = "No shoreward/non-shoreward rows after wind merge."
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        save_csv(pd.DataFrame([dict(note="empty_wind_regime_slice")]), "wind_regime_statistics.csv")
        return

    violin_cols = {
        "maritime_pressure_index": "Maritime Exposure Index",
        "vessel_density_t": "Vessel density",
        "atmospheric_transfer_index": "Atmospheric transfer index",
        "_no2_excess": "NO₂ excess (weekly anomaly)",
    }
    vn_welch_labels = dict(
        vessel_density_t="Vessel density",
        maritime_pressure_index="Maritime Exposure Index",
        atmospheric_transfer_index="Atmospheric transfer index",
        NO2_mean="NO₂ mean",
    )

    lw = []
    for col, slab in violin_cols.items():
        r = ww[col].astype(float).rank(pct=True, method="average")
        chunk = ww.assign(
            indicator=slab,
            participation_rank_pct=r.astype(float),
        )[["nearest_port", "wind_regime", "_band", "indicator", "participation_rank_pct"]].dropna(subset=["participation_rank_pct"])
        lw.append(chunk)
    violin_long = pd.concat(lw, ignore_index=True)
    violin_long["_band_cat"] = pd.Categorical(
        violin_long["_band"].astype(object),
        categories=WIND_DISTANCE_BANDS_ORDER,
        ordered=True,
    )
    violin_long = violin_long.sort_values("_band_cat")

    hues = sns.color_palette("colorblind", n_colors=len(violin_cols))
    fig_a, axes_a = plt.subplots(1, 2, figsize=(13.85, 5.62), sharey=True)
    for ax_a, wd in zip(axes_a, ("shoreward", "nonshoreward")):
        sub = violin_long.loc[violin_long["wind_regime"].astype(str).eq(wd)]
        sns.violinplot(
            data=sub,
            x="_band_cat",
            y="participation_rank_pct",
            hue="indicator",
            order=WIND_DISTANCE_BANDS_ORDER,
            hue_order=list(violin_cols.values()),
            ax=ax_a,
            dodge=True,
            gap=0.16,
            cut=0,
            linewidth=1.03,
            inner="quartile",
            density_norm="width",
            palette=list(hues),
            saturation=0.96,
            legend=False,
        )
        ax_a.set_title("Shoreward wind" if wd == "shoreward" else "Non-shoreward wind", fontsize=12.75, pad=11)
        ax_a.set_xlabel("Distance to nearest labelled port coastline")
        plt.setp(ax_a.get_xticklabels(), rotation=34, fontsize=11, ha="right", rotation_mode="anchor")
        ax_a.set_ylabel("Rank percentile (pooled)")
    axes_a[0].set_ylim(-0.02, 1.02)
    ln_custom = tuple(
        plt.Rectangle((0, 0), 1.0, 1.08, lw=2.05, ec="#284165", fc=hues[idx], clip_on=False, alpha=0.9)
        for idx in range(len(violin_cols))
    )
    lg_a = fig_a.legend(
        ln_custom,
        list(violin_cols.values()),
        title="Environmental indicator",
        loc="upper center",
        ncol=len(violin_cols),
        frameon=False,
        bbox_to_anchor=(0.53, -0.10),
        fontsize=10.2,
        title_fontsize=11.2,
        handlelength=1.25,
        handletextpad=0.6,
        columnspacing=1.62,
        labelspacing=0.75,
        borderaxespad=0.92,
        labelcolor="#1e2b3f",
    )
    lg_a.get_title().set_color("#293241")
    axes_a[-1].set_ylabel("")
    fig_a.subplots_adjust(bottom=0.22, left=0.068, right=0.992, top=0.80, wspace=0.12)
    fig_a.suptitle(
        "Figure 5.7a — Participation by coastal distance stratified on wind regime",
        fontsize=13.85,
        fontweight="bold",
        y=0.96,
    )
    fig_a.text(
        0.5,
        0.90,
        "Rank-normalized participation distributions across coastal distance bands",
        fontsize=11.35,
        style="italic",
        color="#393939",
        ha="center",
        transform=fig_a.transFigure,
    )
    save_wrapped_final(fig_a, stems[0])
    write_final_caption(
        stems[0],
        "## Figure 5.7a — Wind-regime participation by distance bands\n\n"
        "Indicators are summarized with **within-variable rank percentiles** computed on the plotted observation slice "
        "(MEI via `maritime_pressure_index`, week-aligned vessel density, atmospheric-transfer index, and **NO₂ excess** "
        "defined as observation-level NO₂ minus the weekly pooled median). Distance annuli tighten to coastal rings "
        "0–30 km referenced to `nearest_port`. Violins use width-normalised densities with quartile enclosures for medians.",
    )

    bar_specs = dict(
        vessel_density_t="Vessel density",
        maritime_pressure_index="Maritime Exposure Index",
        NO2_mean="NO₂ mean",
    )
    x_order = ["shoreward", "nonshoreward"]
    w = 0.24
    port_offsets = {pt: ((-w), 0.0, (+w))[i] for i, pt in enumerate(PORTS_ORDER)}
    hues_port = sns.color_palette("muted", len(PORTS_ORDER))

    fig_b, axes_b = plt.subplots(1, 3, figsize=(13.95, 4.55))
    descriptive_rows = []
    x = np.arange(len(x_order), dtype=float)

    def _lbl_fmt(mu: float) -> str:
        if not math.isfinite(mu):
            return ""
        aa = abs(mu)
        if aa >= 250 or aa < 0.01:
            return f"{mu:.5g}"
        return f"{mu:.4g}"

    for axi, (col_raw, ylab) in enumerate(bar_specs.items()):
        ax_b = axes_b[axi]
        ymax = -math.inf
        for pj, pt in enumerate(PORTS_ORDER):
            xv, hv, ev = [], [], []
            ns = []
            for xi, wd in enumerate(x_order):
                ser = ww.loc[
                    ww["nearest_port"].astype(str).eq(pt) & ww["wind_regime"].astype(str).eq(wd),
                    col_raw,
                ].astype(float)
                mu_, se_, n_ = aggregated_mean_se(ser)
                descriptive_rows.append(
                    dict(
                        record_type="mean_se_bar_detail",
                        metric=ylab,
                        port=pt,
                        wind_regime=wd,
                        mean=float(mu_) if mu_ == mu_ else math.nan,
                        se=float(se_) if se_ == se_ else math.nan,
                        n=int(n_),
                    )
                )
                pos = xi + port_offsets[pt]
                xv.append(pos)
                hv.append(mu_ if mu_ == mu_ else 0.0)
                ev.append(se_ if se_ == se_ else 0.0)
                ns.append(n_)
                if mu_ == mu_ and math.isfinite(mu_):
                    yhi = mu_ + (ev[-1] if math.isfinite(ev[-1]) else 0)
                    ymax = max(ymax, yhi + max(1e-6, 0.04 * abs(mu_)))
                    ax_b.annotate(_lbl_fmt(mu_), xy=(pos, yhi), fontsize=9.05, ha="center", va="bottom", color="#172133")
                    ax_b.annotate(
                        f"n={n_}",
                        xy=(pos, 0),
                        xytext=(0, -13),
                        textcoords="offset points",
                        fontsize=8.05,
                        ha="center",
                        va="top",
                        color="#555560",
                        clip_on=False,
                    )
            yerr_sym = [float(s) if s == s and math.isfinite(float(s)) else 0.0 for s in ev]
            ax_b.bar(
                xv,
                hv,
                width=w * 0.92,
                yerr=yerr_sym,
                capsize=4.05,
                color=hues_port[pj],
                edgecolor="#1c1f2c",
                linewidth=0.66,
                error_kw=dict(elinewidth=1.35, ecolor="#393939"),
            )
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(["Shoreward", "Non-shoreward"], fontsize=11.5)
        ax_b.set_title(ylab, fontsize=12.3, pad=9)
        sns.despine(ax=ax_b)
        ym = ymax if ymax > -math.inf else float("nan")
        if math.isfinite(ym) and ym > 0:
            ax_b.set_ylim(0.0, ym * 1.15)

    fig_b.subplots_adjust(bottom=0.19, left=0.069, right=0.986, top=0.84, wspace=0.32)
    ln_p = tuple(
        plt.Rectangle((0, 0), 1, 1, fc=hues_port[i], ec="#223", lw=0.52) for i in range(len(PORTS_ORDER))
    )
    lg = fig_b.legend(
        ln_p,
        PORTS_ORDER,
        title="Adjacent port attribution",
        loc="upper center",
        ncol=len(PORTS_ORDER),
        bbox_to_anchor=(0.513, -0.05),
        frameon=False,
        fontsize=11.1,
        title_fontsize=11.35,
        handlelength=1.25,
        handletextpad=0.74,
        columnspacing=3.85,
        labelcolor="#2b3343",
    )
    lg.get_title().set_color("#343d4d")
    fig_b.suptitle(
        "Figure 5.7b — Mean ± SE exposure contrast by wind regime and port linkage",
        fontsize=13.9,
        fontweight="bold",
        y=0.985,
    )
    fig_b.text(
        0.01,
        0.068,
        "Bars show pooled means ± standard error; labelled values summarise means.",
        fontsize=11.0,
        style="italic",
        color="#3a434b",
        transform=fig_b.transFigure,
    )
    axes_b[-1].annotate(
        "Compare regimes for asymmetric maritime amplification.",
        xy=(1.02, -0.22),
        xycoords="axes fraction",
        fontsize=10.45,
        color="#444c57",
        ha="left",
    )
    save_wrapped_final(fig_b, stems[1])
    write_final_caption(
        stems[1],
        "## Figure 5.7b — Directional regime comparison\n\n"
        "Three outcome panels summarise **means ± SE** for vessel density (`vessel_density_t`), Maritime Exposure Index "
        "(`maritime_pressure_index`), and weekly NO₂. Within each subplot, clustered bars contrast shoreward versus "
        "non-shoreward classifications at each archival `nearest_port` assignment (Turku, Mariehamn); numerals atop "
        "bars give means, while **n=** captions record analytic sample sizes.",
    )

    stats_welch = []
    for pt in PORTS_ORDER:
        for ckey, slab in vn_welch_labels.items():
            a = ww.loc[(ww["wind_regime"].eq("shoreward")) & (ww["nearest_port"].eq(pt)), ckey].astype(float).dropna()
            b = ww.loc[(ww["wind_regime"].eq("nonshoreward")) & (ww["nearest_port"].eq(pt)), ckey].astype(float).dropna()
            wl = scipy_stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else math.nan
            stats_welch.append(
                dict(
                    record_type="welch_between_wind_regimes",
                    nearest_port=pt,
                    indicator=slab,
                    n_shoreward=int(len(a)),
                    n_nonshoreward=int(len(b)),
                    welch_p_value=float(wl) if wl == wl and math.isfinite(wl) else math.nan,
                    mean_shoreward=float(np.mean(a)) if len(a) else math.nan,
                    mean_nonshoreward=float(np.mean(b)) if len(b) else math.nan,
                )
            )

    w2 = pd.DataFrame(descriptive_rows)
    w3 = pd.DataFrame(stats_welch)
    save_csv(pd.concat([w2, w3], ignore_index=True).sort_values("record_type"), "wind_regime_statistics.csv")



def fig58(df: pd.DataFrame) -> None:
    """Figure 5.8 — Environmental Stress Index: distribution histogram + spatial hotspot map."""


    stems = ("fig_5_8a_esi_distribution", "fig_5_8b_esi_spatial_hotspots")
    delete_final_outputs(stems)
    delete_old_outputs(("figure_5_8_environmental_stress.png",))

    comps = ["mei", "NO2_mean", "fai_mean", "ndti_mean", "vessel_density_t", "coastal_exposure_score"]
    if not all(c in df.columns for c in comps):
        miss = ",".join(c for c in comps if c not in df.columns)
        stub = f"ESI figures skipped (missing columns: {miss})"
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        return

    T = df[comps].apply(pd.to_numeric, errors="coerce")

    zstack = []
    for c in comps:
        zstack.append((T[c] - T[c].mean()) / (T[c].std(ddof=0) + 1e-11))

    esi = pd.concat(zstack, axis=1).mean(axis=1)

    cdf = pd.DataFrame(
        dict(
            esi=esi,
            gid=df["grid_cell_id"],
            lat=pd.to_numeric(df["grid_centroid_lat"], errors="coerce"),
            lon=pd.to_numeric(df["grid_centroid_lon"], errors="coerce"),
            no2=pd.to_numeric(df["NO2_mean"], errors="coerce"),
            port=df["nearest_port"].astype(str),
        )
    )

    cen = (
        cdf.groupby("gid", as_index=False).agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            esi=("esi", "mean"),
            no2=("no2", "mean"),
            port=("port", "first"),
        )
        .dropna(subset=["lat", "lon"])
    )

    cen = cen[np.isfinite(cen["esi"])]
    if cen.empty:
        stub = "No finite ESI after cell aggregation."
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        return

    hq = cen["esi"].quantile(0.92)


    cen["hot"] = cen["esi"].values >= hq


    nq = cen["no2"].quantile(0.90)



    cen["no2_peak"] = cen["no2"].values >= nq

















    rp_rows = [
        dict(
            summary="overall",
            note=(
                "ESI = mean of column-standardised z scores "
                "(MEI,NO2,FAI,NDTI,vessel_density,coastal_exposure)"
            ),
            pct_hot_pct92=float(cen["hot"].mean() * 100),
            no2_above_p90_cell_pct=float(cen["no2_peak"].mean() * 100),
            n_cells=len(cen),
        ),
    ]
    for pn, gx in cen.groupby("port"):
        if pn is None or str(pn).lower() == "nan":
            continue
        rp_rows.append(
            dict(
                summary="by_port_zone",
                port=str(pn),
                mean_ESI=float(np.nanmean(gx["esi"])),
                median_ESI=float(np.nanmedian(gx["esi"])),
                pct_cells_hot_pct92=float(gx["hot"].mean() * 100),
                n_cells=int(len(gx)),
            )
        )
    save_csv(pd.DataFrame(rp_rows), "environmental_stress_summary.csv")

    esi_arr = np.asarray(cen["esi"], dtype=float)
    vmin = float(np.nanquantile(esi_arr, 0.06))
    vmax = float(np.nanquantile(esi_arr, 0.98))
    n_cells = len(cen)
    n_hot = int(cen["hot"].sum())

    # —— Figure 5.8a: ESI distribution (histogram + KDE) ——
    fig_a, ax_a = plt.subplots(figsize=(8.15, 5.15))
    sns.histplot(
        esi_arr,
        kde=False,
        ax=ax_a,
        stat="density",
        bins=46,
        color="#1f4e79",
        edgecolor="#f8fafc",
        linewidth=0.4,
        alpha=0.9,
    )
    sns.kdeplot(esi_arr, ax=ax_a, color="#b91c1c", linewidth=2.0, cut=0, zorder=4)
    ax_a.set_xlabel("Environmental Stress Index (cell mean, z-composite)", fontsize=12.25)
    ax_a.set_ylabel("Density", fontsize=12.25)
    ax_a.tick_params(axis="both", labelsize=10.9)
    sns.despine(ax=ax_a)
    fig_a.suptitle(
        "Figure 5.8a — Environmental Stress Index composite",
        fontsize=14.05,
        fontweight="bold",
        y=0.965,
    )
    fig_a.text(
        0.5,
        0.905,
        "Distribution of cell-mean scores (standardized MEI, NO₂, FAI, NDTI, vessel density, coastal exposure)",
        fontsize=10.35,
        style="italic",
        color="#3a3a3a",
        ha="center",
        transform=fig_a.transFigure,
    )
    fig_a.text(
        0.02,
        0.025,
        f"n = {n_cells} grid cells",
        fontsize=10.6,
        color="#2d2d2d",
        transform=fig_a.transFigure,
    )
    fig_a.subplots_adjust(top=0.80, bottom=0.11, left=0.11, right=0.97)
    save_wrapped_final(fig_a, "fig_5_8a_esi_distribution")
    write_final_caption(
        "fig_5_8a_esi_distribution",
        "## Figure 5.8a — Environmental Stress Index (distribution)\n\n"
        "Per-grid **cell mean** of the composite built as the **average of column-wise z-scores** over MEI "
        "(`maritime_pressure_index`), NO₂, FAI, NDTI, vessel density, and coastal exposure score on the ML-ready panel, "
        "then aggregated by `grid_cell_id`. The **red curve** is a kernel density estimate.",
    )

    # —— Figure 5.8b: spatial hotspots ——
    mid_lat = 0.5 * (MAP_BB["lat0"] + MAP_BB["lat1"])
    asp = 1.0 / np.cos(np.deg2rad(mid_lat))

    fig_b, ax_b = plt.subplots(figsize=(9.0, 9.35))
    ax_b.set_aspect(asp, adjustable="box")

    sc = ax_b.scatter(
        cen["lon"],
        cen["lat"],
        c=cen["esi"],
        cmap="Spectral_r",
        s=36,
        alpha=0.86,
        edgecolors="#14182a",
        linewidths=0.11,
        vmin=vmin,
        vmax=vmax,
        zorder=4,
        rasterized=True,
    )
    cb = fig_b.colorbar(sc, ax=ax_b, fraction=0.0385, pad=0.018)
    cb.set_label("Mean ESI (z-composite)", fontsize=11.3)
    cb.ax.tick_params(labelsize=9.8)

    if cen["hot"].any():
        ax_b.scatter(
            cen.loc[cen["hot"], "lon"],
            cen.loc[cen["hot"], "lat"],
            facecolors="none",
            edgecolors="#b91c1c",
            s=132,
            linewidths=1.08,
            zorder=8,
            label=f"Stress hotspot (≥ pooled p92, n={n_hot})",
        )

    plpc = sns.color_palette("muted", len(PORTS_LL))
    for ix, (pna, (la_, ln_)) in enumerate(PORTS_LL.items()):
        ax_b.scatter(
            ln_,
            la_,
            marker="*",
            s=260,
            fc=plpc[ix],
            ec="#171717",
            lw=0.52,
            label=pna,
            zorder=12,
        )

    ax_b.set_xlim(MAP_BB["lon0"], MAP_BB["lon1"])
    ax_b.set_ylim(MAP_BB["lat0"], MAP_BB["lat1"])
    ax_b.set_xlabel("Longitude (°E)", fontsize=12.05)
    ax_b.set_ylabel("Latitude (°N)", fontsize=12.05)
    ax_b.tick_params(axis="both", labelsize=10.75)
    ax_b.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#d4d4d8",
        fontsize=9.65,
        ncol=2,
        handlelength=1.35,
    )
    sns.despine(ax=ax_b)
    fig_b.suptitle(
        "Figure 5.8b — Environmental Stress Index spatial hotspots",
        fontsize=14.1,
        fontweight="bold",
        y=0.98,
    )
    fig_b.text(
        0.5,
        0.938,
        "Grid-cell means coloured by ESI; hollow rings mark the upper tail (≥ pooled 92nd percentile)",
        fontsize=10.2,
        style="italic",
        color="#3d3d45",
        ha="center",
        transform=fig_b.transFigure,
    )
    fig_b.text(
        0.02,
        0.06,
        f"n = {n_cells} cells",
        fontsize=10.4,
        color="#2d2d2d",
        transform=fig_b.transFigure,
    )
    fig_b.subplots_adjust(left=0.075, right=0.98, top=0.87, bottom=0.07)
    save_wrapped_final(fig_b, "fig_5_8b_esi_spatial_hotspots")
    write_final_caption(
        "fig_5_8b_esi_spatial_hotspots",
        "## Figure 5.8b — Environmental Stress Index (spatial hotspots)\n\n"
        "**Spectral_r** colours encode **cell-mean ESI** on the study bounding box. **Red open circles** emphasise "
        "the pooled **upper 8%** of cells (≥ 92nd percentile among distinct `grid_cell_id`). Stars mark reference ports "
        "(Turku, Mariehamn). Rasterised points keep vector PDF exports lightweight.",
    )


def fig59(df: pd.DataFrame) -> None:
    """Figure 5.9 — Weekly median dynamics + lag Spearman autocorrelation (thesis finals)."""


    stems = ("fig_5_9a_temporal_weekly_medians", "fig_5_9b_lag_autocorrelation_heatmap")
    delete_final_outputs(stems)
    delete_old_outputs(
        (
            "figure_5_9_temporal_lag_analysis.png",
            "figure_5_9_temporal_lag_analysis.pdf",
            "figure_5_9_temporal_lag_final.png",
            "figure_5_9_temporal_lag_final.pdf",
            "figure_5_10_anomaly_analysis.png",
            "figure_5_10_anomaly_analysis.pdf",
        ),
    )

    lab = dict(
        vessel_density_t="Vessel dens.",
        NO2_mean="NO2",
        ndti_mean="NDTI",
        ndwi_mean="NDWI",
        mei="MEI",
    )
    cols_map = lab

    if not all(c in df.columns for c in cols_map):
        miss = ",".join(c for c in cols_map if c not in df.columns)
        stub = f"Figure 5.9 skipped (missing columns: {miss})"
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        return

    week_ix = pd.to_datetime(df["week_start_utc"], utc=True)
    pooled_m = (
        pd.DataFrame({vn: pd.to_numeric(df[vn], errors="coerce").values for vn in cols_map})
        .assign(_w=week_ix)
        .groupby("_w")
        .median(numeric_only=True)
        .sort_index()
    )
    pooled_lab = pooled_m.rename(columns={k: v for k, v in cols_map.items()})

    if pooled_m.shape[0] < 5 or pooled_m.shape[1] < 1:
        stub = "Insufficient weekly coverage for temporal-lag figures."
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        return

    lag_rows = []
    for vn in pooled_m.columns:
        s = pooled_m[vn].astype(float)
        for L in (1, 2):
            a = s.shift(L).to_numpy(dtype=float)
            b = s.to_numpy(dtype=float)
            msk = np.isfinite(a) & np.isfinite(b)
            nv = int(np.sum(msk))
            if nv > 9:
                rho, _ = scipy_stats.spearmanr(a[msk], b[msk])
                pr = float(np.asarray(rho).ravel()[0])
                if not math.isfinite(pr):
                    pr = math.nan
            else:
                pr = math.nan
            lag_rows.append(
                dict(metric=vn, friendly=cols_map[vn], lag_weeks=L, rho_spearman=pr, n_windows=nv),
            )

    lag_tbl = pd.DataFrame(lag_rows)
    corr_long = pooled_m.astype(float).corr(method="spearman").stack().reset_index()
    corr_long.columns = ["metric_pair_a", "metric_pair_b", "rho_weekly_medians"]

    lag_save = lag_tbl.rename(columns=dict(rho_spearman="rho")).assign(
        segment="lag_autocorr_spearman", metric_pair_a=np.nan, metric_pair_b=np.nan
    )
    corr_save = corr_long.rename(columns=dict(rho_weekly_medians="rho")).assign(
        segment="weekly_median_corr", lag_weeks=math.nan, metric=np.nan, friendly=np.nan, n_windows=int(len(pooled_m))
    )
    save_csv(pd.concat([lag_save, corr_save], ignore_index=True), "temporal_lag_statistics.csv")

    lag_wide_full = lag_tbl.pivot(index="friendly", columns="lag_weeks", values="rho_spearman").sort_index()
    lag_wide = lag_wide_full.copy()
    noisy = lag_wide.abs().max(axis=1).fillna(0) < 0.035
    lag_wide = lag_wide.loc[~noisy]
    if lag_wide.shape[0] < 3:
        lag_wide = lag_wide_full

    z = pooled_lab.astype(float).apply(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-9))
    z_smooth = z.rolling(5, center=True, min_periods=2).mean()
    pals = sns.color_palette("colorblind", n_colors=min(8, len(z_smooth.columns)))

    # —— Figure 5.9a: weekly median trends ——
    fig_a, ax0 = plt.subplots(figsize=(11.85, 5.55))
    for ci, cn in enumerate(z_smooth.columns):
        ax0.plot(z_smooth.index, z_smooth[cn], lw=2.05, color=pals[ci % len(pals)], label=cn, alpha=0.93)
    ax0.axhline(0, ls=":", color="#778899", lw=0.88)
    ax0.grid(True, axis="y", alpha=0.28, color="#b8c0cc", linewidth=0.62)
    ax0.set_axisbelow(True)

    mscore = z_smooth.abs().max(axis=1)
    if len(mscore):
        thr_ann = float(mscore.quantile(0.985))
        top_w = (mscore[mscore >= thr_ann]).sort_values(ascending=False).head(8).sort_index()
        for wt in top_w.index:
            ax0.annotate(
                wt.strftime("%Y-%m"),
                xy=(wt, float(z_smooth.loc[wt].abs().max())),
                fontsize=7.9,
                rotation=68,
                ha="left",
                va="bottom",
                color="#2a3352",
                alpha=0.85,
            )

    ax0.set_ylabel("Rolling-mean z-score", fontsize=12.35)
    ax0.set_xlabel("Week (UTC start)", fontsize=12.2)
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax0.tick_params(axis="x", rotation=32, labelsize=10.4)
    ax0.tick_params(axis="y", labelsize=10.4)
    ax0.legend(
        loc="lower left",
        ncol=3,
        frameon=True,
        fontsize=9.5,
        framealpha=0.95,
        edgecolor="#c5cdd9",
        handlelength=2.1,
    )
    sns.despine(ax=ax0)
    fig_a.suptitle(
        "Figure 5.9a — Temporal lag structure: weekly median dynamics",
        fontsize=14.05,
        fontweight="bold",
        y=0.975,
    )
    fig_a.text(
        0.5,
        0.04,
        "Pooled cross-cell medians per week, z-scored over the series, then smoothed with a centered 5-week rolling mean.",
        fontsize=10.25,
        style="italic",
        color="#3f4a58",
        ha="center",
        transform=fig_a.transFigure,
    )
    fig_a.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.18)
    save_wrapped_final(fig_a, stems[0])
    write_final_caption(
        stems[0],
        "## Figure 5.9a — Weekly median trend dynamics\n\n"
        "Lines show **standardised weekly medians** (each indicator z-scored across weeks on the pooled series) "
        "followed by a **centered 5-week rolling mean** to emphasise multi-week structure. Year–month labels mark "
        "weeks in the extreme upper tail of pooled absolute smoothed scores (~top 1–2%).",
    )

    # —— Figure 5.9b: lag autocorrelation heatmap ——
    n_row = int(lag_wide.shape[0])
    fig_h = max(4.6, 3.35 + 0.42 * n_row)
    fig_b, ax1 = plt.subplots(figsize=(7.15, fig_h))
    sns.heatmap(
        lag_wide,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax1,
        linewidths=0.7,
        linecolor="#f4f4f8",
        cbar_kws=dict(shrink=0.85, label="Spearman ρ"),
        annot_kws=dict(size=11.8, weight="semibold"),
    )
    ax1.set_title("Lag autocorrelation of weekly medians", fontsize=13.15, pad=12)
    ax1.set_xlabel("Lag (weeks)", fontsize=12.15)
    ax1.set_ylabel("Indicator", fontsize=12.15)
    ax1.tick_params(axis="both", labelsize=10.9)
    ax1.set_xticklabels([str(int(round(float(t.get_text())))) for t in ax1.get_xticklabels()])
    fig_b.suptitle(
        "Figure 5.9b — Temporal lag structure: autocorrelation heatmap",
        fontsize=14.05,
        fontweight="bold",
        y=0.97,
    )
    fig_b.text(
        0.5,
        0.02,
        "Spearman ρ between the weekly median series and itself shifted by 1–2 weeks. Near-zero rows may be omitted.",
        fontsize=10.1,
        style="italic",
        color="#3f4a58",
        ha="center",
        transform=fig_b.transFigure,
    )
    fig_b.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.12)
    save_wrapped_final(fig_b, stems[1])
    write_final_caption(
        stems[1],
        "## Figure 5.9b — Lag autocorrelation heatmap\n\n"
        "Cells report **Spearman rank correlation** between pooled weekly median series at **1- and 2-week** lags. "
        "Indicators with both |ρ| < 0.035 across lags are dropped from display (fallback keeps the full matrix if "
        "fewer than three series would remain).",
    )



def fig510(df: pd.DataFrame) -> None:
    """Figure 5.10 — Rolling-baseline anomalies: co-occurrence matrix + simultaneous timeline."""


    stems = ("fig_5_10a_anomaly_cooccurrence_heatmap", "fig_5_10b_anomaly_simultaneous_timeline")
    delete_final_outputs(stems)
    delete_old_outputs(("figure_5_10_anomaly_final.png", "figure_5_10_anomaly_final.pdf"))

    col_to_label = dict(
        NO2_mean="NO2",
        mei="MEI",
        vessel_density_t="VDens",
        ndti_mean="NDTI",
        fai_mean="FAI",
    )

    if not all(c in df.columns for c in col_to_label):
        miss = ",".join(c for c in col_to_label if c not in df.columns)
        stub = f"Figure 5.10 skipped (missing columns: {miss})"
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        return

    week_ix = pd.to_datetime(df["week_start_utc"], utc=True)

    raw = pd.DataFrame({vn: pd.to_numeric(df[vn], errors="coerce").values for vn in col_to_label})
    wk_named = raw.assign(_w=week_ix).groupby("_w").median(numeric_only=True).sort_index()
    wk_lab = wk_named.rename(columns=col_to_label)

    baseline = wk_lab.rolling(9, center=True, min_periods=4).median()
    resid = wk_lab - baseline
    q = resid.quantile(0.933, interpolation="nearest")
    flags = resid.ge(q, axis=1).astype(int)
    timeline_cols = ["NO2", "MEI", "NDTI", "FAI"]

    names = flags.columns.to_list()
    overlap = pd.DataFrame(
        np.dot(flags.values.T.astype(np.float64), flags.values.astype(np.float64)).astype(np.int64),
        index=names,
        columns=names,
    )
    # restrict to headline indicators present
    avail = [c for c in timeline_cols if c in overlap.index and c in overlap.columns]
    if len(avail) < 2:
        stub = "Too few indicators for co-occurrence figure."
        for stem in stems:
            fg, ax = plt.subplots(figsize=(6.0, 2.4))
            ax.text(0.5, 0.5, stub, ha="center", va="center")
            plt.axis("off")
            save_wrapped_final(fg, stem)
            write_final_caption(stem, f"# {stem}\n\n*{stub}*.")
        return

    hm_labels = overlap.loc[avail, avail]

    tally = pd.DataFrame(
        dict(metric_name=list(wk_lab.columns), anomaly_week_counts=flags.sum(axis=0).astype(int), segment="per_metric_totals")
    )
    pair_rows = [
        dict(var_row=r, var_col=c, co_anomaly_week_counts=int(hm_labels.loc[r, c])) for r in hm_labels.index for c in hm_labels.columns
    ]
    pair_df = pd.DataFrame(pair_rows).assign(segment="pairwise_overlap_weeks")
    save_csv(pd.concat([tally, pair_df], ignore_index=True), "anomaly_detection_summary.csv")

    co_raw = flags.sum(axis=1).astype(float)
    co_smooth = co_raw.rolling(3, center=True, min_periods=1).median()

    evt_rows = []
    for vn in timeline_cols:
        if vn not in resid.columns:
            continue
        for ts in flags.index:
            if int(flags.loc[ts, vn]) != 1:
                continue
            co_all = int(co_raw.loc[ts])
            strength = float(resid.loc[ts, vn])
            tsu = pd.Timestamp(ts)
            if tsu.tzinfo is None:
                tsu = tsu.tz_localize("UTC")
            sub = df[df["week_start_utc"].eq(tsu)]
            modal_port = ""
            if len(sub):
                vc = sub["nearest_port"].value_counts(dropna=False)
                if len(vc):
                    cand = str(vc.index[0])
                    modal_port = "" if cand.lower() == "nan" else cand
            evt_rows.append(
                dict(
                    metric=str(vn),
                    anomaly_week=pd.Timestamp(ts).strftime("%Y-%m-%d"),
                    anomaly_strength=strength,
                    co_occurrence_count=co_all,
                    nearest_port=modal_port,
                )
            )
    save_csv(pd.DataFrame(evt_rows).sort_values(["metric", "anomaly_week"]), "anomaly_event_summary.csv")

    n_m = hm_labels.shape[0]
    fig_hm = max(5.4, 3.8 + 0.55 * n_m)

    # —— 5.10a: pairwise weekly co-occurrence (count matrix) ——
    fig_a, ax_h = plt.subplots(figsize=(7.0, fig_hm))
    sns.heatmap(
        hm_labels.astype(int),
        annot=True,
        fmt="d",
        cmap="GnBu_r",
        ax=ax_h,
        vmin=0,
        vmax=max(int(hm_labels.to_numpy().max()), 1),
        linewidths=1.05,
        linecolor="#f0f7fc",
        cbar_kws=dict(shrink=0.78, label="Weeks"),
        square=True,
        annot_kws=dict(size=12.8, weight="semibold"),
    )
    ax_h.set_xlabel("", fontsize=11.5)
    ax_h.set_ylabel("", fontsize=11.5)
    ax_h.tick_params(axis="both", labelsize=11.2)
    ax_h.set_title("Pairwise overlap (rolling-baseline anomalies)", fontsize=13.05, pad=14)
    fig_a.suptitle(
        "Figure 5.10a — Rolling-baseline anomalies: co-occurrence heatmap",
        fontsize=14.0,
        fontweight="bold",
        y=0.98,
    )
    fig_a.text(
        0.5,
        0.02,
        "Cell (i,j): number of weeks where both indicators exceed the pooled 93rd percentile residual (9-week rolling baseline).",
        fontsize=9.95,
        style="italic",
        color="#3d4654",
        ha="center",
        transform=fig_a.transFigure,
    )
    fig_a.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.14)
    save_wrapped_final(fig_a, stems[0])
    write_final_caption(
        stems[0],
        "## Figure 5.10a — Weekly co-occurrence (pairwise heatmap)\n\n"
        "Off-diagonal and symmetric entries count **how many week-starts** both indicators simultaneously exceed the "
        "**rolling nine-week baseline** residual pooled at the **93rd percentile** (see `anomaly_detection_summary.csv`). "
        "Diagonal: total anomaly weeks for that indicator.",
    )

    # —— 5.10b: simultaneous anomaly-count timeline ——
    fig_b, ax_t = plt.subplots(figsize=(11.85, 5.0))
    ax_t.plot(co_smooth.index, co_smooth.values, lw=2.05, color="#1a3f78", alpha=0.96, label="3-week rolling median (count)")
    ax_t.fill_between(
        co_smooth.index.to_numpy(dtype="datetime64[ns]"),
        np.full(len(co_smooth), 0.0),
        co_smooth.to_numpy(dtype=float),
        color="#3d6aa834",
        lw=0,
    )
    y_hi = float(max(float(co_smooth.max()), 4.08))
    ax_t.set_ylim(0, y_hi * 1.08)
    for ht, _vv in co_raw[co_raw >= 3].sort_values(ascending=False).head(14).items():
        t_ = pd.Timestamp(ht)
        ax_t.axvline(t_, ymin=0, ymax=0.96, color="#9b2d2d66", lw=1.05)
        ax_t.text(t_, y_hi * 1.035, pd.Timestamp(ht).strftime("%y-%m"), fontsize=8.2, ha="center", va="bottom", color="#4a252c")

    ax_t.set_ylabel("Number of simultaneous anomalies", fontsize=12.35)
    ax_t.set_xlabel("Week (UTC start)", fontsize=12.1)
    ax_t.set_title("Weekly co-fire load (raw count, smoothed)", fontsize=12.35, pad=10)
    ax_t.legend(loc="upper left", fontsize=10.1, frameon=True, framealpha=0.95, edgecolor="#cfd6e0")
    ax_t.grid(False)
    ax_t.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_t.tick_params(axis="x", rotation=34, labelsize=10.2)
    ax_t.tick_params(axis="y", labelsize=10.2)
    sns.despine(ax=ax_t)
    fig_b.suptitle(
        "Figure 5.10b — Rolling-baseline anomalies: simultaneous count timeline",
        fontsize=14.05,
        fontweight="bold",
        y=0.97,
    )
    fig_b.text(
        0.5,
        0.045,
        "Each week: count of headline indicators (NO₂, MEI, NDTI, FAI) flagging residual above the pooled threshold; line is a 3-week rolling median.",
        fontsize=9.95,
        style="italic",
        color="#3d4654",
        ha="center",
        transform=fig_b.transFigure,
    )
    fig_b.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.2)
    save_wrapped_final(fig_b, stems[1])
    write_final_caption(
        stems[1],
        "## Figure 5.10b — Simultaneous anomaly-count timeline\n\n"
        "For each week, **how many** of NO₂, MEI, NDTI, and FAI simultaneously register as anomalies (binary flags). The **blue line** "
        "is a centred **three-week rolling median** of that count; vertical markers highlight weeks with **three or more** concurrent anomalies.",
    )



def main() -> None:
    dirs()
    sns.set_theme(style="ticks", palette="deep")
    thesis_pub_style(thesis_final=True)
    df_panel = load_panel()
    fig55(df_panel)
    fig56(df_panel)
    fig57(df_panel)
    fig58(df_panel)
    fig59(df_panel)
    fig510(df_panel)


if __name__ == "__main__":
    main()
