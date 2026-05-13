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
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
WIND_CSV = ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"

BASE = ROOT / "outputs" / "5_2_to_5_10"
FIGB = BASE / "thesis_figures"
REPB = BASE / "reports"
FIGM = ROOT / "outputs" / "thesis_figures"
REPM = ROOT / "outputs" / "reports"

PORTS_LL = dict(Stockholm=(59.3293, 18.0686), Turku=(60.435, 22.225), Mariehamn=(60.0973, 19.9348))
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


def save_png(fig: plt.Figure, fname: str) -> None:
    # Avoid bbox_inches="tight" on large subplot grids: mpl can return a pathological
    # bbox (very tall) when x-ticklabels are rotated + tight_layout friction.
    kw = dict(dpi=340, bbox_inches=None, pad_inches=0.02, facecolor="white")
    fig.savefig(FIGB / fname, **kw)
    fig.savefig(FIGM / fname, **kw)
    plt.close(fig)


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
    return df


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
        axi.boxplot(data[ix], widths=0.58, patch_artist=True, boxprops=dict(facecolor="#dbebfb"))
        axi.set_xticks([1])


        axi.set_xticklabels([specs[ix][1]])
        sns.despine(ax=axi)

    fg.suptitle("Fig. 5.5 Boxplots capped at pooled 99th percentile")
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
    vn = dict(
        NO2_mean="NO2 mean",
        mei="Maritime Exposure Index",
        vessel_density_t="Vessel density",
        atmospheric_transfer_index="Atmospheric transfer index",
    )
    vn = {key: lbl for key, lbl in vn.items() if key in df.columns}

    if not vn:

        fg, ax = plt.subplots(figsize=(5.9, 2.6))

        ax.text(0.5, 0.5, "No wind-aligned rows (indicator columns absent)")

        plt.axis("off")

        save_png(fg, "figure_5_7_wind_regime_analysis.png")

        save_csv(pd.DataFrame([dict(note="missing_wind_indicators_columns")]), "wind_regime_statistics.csv")

        return


    ww = df[df["wind_regime"].isin(["shoreward", "nonshoreward"])]
    ww = ww[ww["nearest_port"].isin(list(PORTS_LL))]


    if ww.empty:


        fg, ax = plt.subplots(figsize=(5.9, 2.6))



        ax.text(0.5, 0.5, "No wind-aligned rows")


        plt.axis("off")


        save_png(fg, "figure_5_7_wind_regime_analysis.png")


        save_csv(pd.DataFrame([dict(note="missing_wind_merge")]), "wind_regime_statistics.csv")


        return

    long_df = ww.melt(id_vars=["nearest_port", "wind_regime"], value_vars=list(vn.keys()), var_name="ind", value_name="val")


    stat = []


    for pt in PORTS_LL:


        for c, lbl in vn.items():



            a = ww.loc[(ww["wind_regime"].eq("shoreward")) & (ww["nearest_port"].eq(pt)), c].astype(float).dropna()



            b = ww.loc[(ww["wind_regime"].eq("nonshoreward")) & (ww["nearest_port"].eq(pt)), c].astype(float).dropna()



            wl = scipy_stats.ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else math.nan

















            stat.append(
                dict(
                    port=pt,
                    indicator=lbl,

                    n_shoreward=len(a),


                    n_nonshoreward=len(b),





                    mean_shore=float(a.mean()) if len(a) else math.nan,





                    mean_non=float(b.mean()) if len(b) else math.nan,

                    median_sh=float(a.median()) if len(a) else math.nan,

                    median_ns=float(b.median()) if len(b) else math.nan,




                    welch_p_value=float(wl) if wl == wl else math.nan,

                )


            )


    save_csv(pd.DataFrame(stat), "wind_regime_statistics.csv")



    fg = plt.figure(figsize=(14.0, 16.05))


    gs = gridspec.GridSpec(5, 1, height_ratios=[1.04, 0.96, 1.04, 0.92, 1.03])



    a0 = fg.add_subplot(gs[0])



    sns.boxplot(long_df, x="ind", y="val", hue="wind_regime", ax=a0, palette=["#2f5f90", "#9c3e40"], showfliers=False)



    a0.set_xticklabels([vn[t.get_text()] for t in a0.get_xticklabels()])
    a0.set_title("Panel A")


    a0.legend(title="Wind")


    a1 = fg.add_subplot(gs[1])



    sns.violinplot(long_df, x="ind", y="val", hue="wind_regime", split=True, inner="quart", palette=["#2f5f90", "#9c3e40"], ax=a1)



    a1.set_xticklabels([vn[t.get_text()] for t in a1.get_xticklabels()])



    a1.legend().remove()



    a2 = fg.add_subplot(gs[2])



    gmean = ww.groupby(["nearest_port", "wind_regime"])[list(vn.keys())].mean()



















    xlab = ["{}".format(ix[0] + chr(10) + ix[1]) for ix in gmean.index]













    sns.heatmap(gmean.T, ax=a2, cmap="mako_r", linewidths=0.52, xticklabels=xlab)
















    ylab = list(gmean.columns)


    if ylab:


        a2.set_yticklabels([vn[h] for h in ylab])


    a2.set_title("Panel C Mean heatmap")



    a3 = fg.add_subplot(gs[3])



















    sns.barplot(long_df, x="ind", y="val", hue="wind_regime", estimator=np.nanmean, errorbar=("ci", 95), ax=a3, palette=["#2f5f90", "#9c3e40"])



    a3.set_xticklabels([vn[t.get_text()] for t in a3.get_xticklabels()])
    a3.set_title("Panel D")



    a3.legend(ncol=4, fontsize=7)
















    a4 = fg.add_subplot(gs[4])
















    qb = ww[pd.to_numeric(ww["grid_centroid_lat"], errors="coerce").between(MAP_BB["lat0"], MAP_BB["lat1"])
              & pd.to_numeric(ww["grid_centroid_lon"], errors="coerce").between(MAP_BB["lon0"], MAP_BB["lon1"])]


    gv = qb.assign(lat=qb["grid_centroid_lat"].round(2), lon=qb["grid_centroid_lon"].round(2)).groupby(["lat", "lon"], as_index=False).agg(mu=("wind_u_mean", "median"), mv=("wind_v_mean", "median")).dropna()



















    a4.add_patch(
        mpatches.Rectangle(
            (MAP_BB["lon0"], MAP_BB["lat0"]), MAP_BB["lon1"] - MAP_BB["lon0"], MAP_BB["lat1"] - MAP_BB["lat0"], facecolor="#e9f2fb", lw=0
        )


    )


    if len(gv):


        a4.quiver(
            gv["lon"].to_numpy(),
            gv["lat"].to_numpy(),
            gv["mu"].to_numpy() / 40.0,

            gv["mv"].to_numpy() / 40.0,

            angles="uv",
            scale_units="xy",
            scale=46,
            width=0.007,
            color="#123961",
            zorder=2,


        )
















    plp = sns.color_palette("muted", len(PORTS_LL))


    for ix, pname in enumerate(PORTS_LL):


        la, ln = PORTS_LL[pname]


        a4.scatter(ln, la, marker="*", s=220, fc=plp[ix], ec="#1a1a1a", lw=0.45, label=pname, zorder=5)



















    a4.set_xlim(MAP_BB["lon0"] - 0.55, MAP_BB["lon1"] + 0.62)


    a4.set_ylim(MAP_BB["lat0"] - 0.4, MAP_BB["lat1"] + 0.35)


    a4.set_title("Panel E Lattice-median vectors")


    a4.legend(loc="lower right")


    sns.despine(ax=a4)



























    fg.tight_layout(rect=[0.015, 0.01, 0.986, 0.98])


    save_png(fg, "figure_5_7_wind_regime_analysis.png")
















def fig58(df: pd.DataFrame) -> None:



    comps = ["mei", "NO2_mean", "fai_mean", "ndti_mean", "vessel_density_t", "coastal_exposure_score"]




    vn = dict(
        mei="MEI",

        NO2_mean="NO2",

        fai_mean="FAI",

        ndti_mean="NDTI",
        vessel_density_t="Vdensity",




        coastal_exposure_score="CoastExp",

    )





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

    vmin = float(np.nanquantile(np.asarray(cen["esi"], dtype=float), 0.06))
    vmax = float(np.nanquantile(np.asarray(cen["esi"], dtype=float), 0.98))
    fg, axp = plt.subplots(2, 2, figsize=(13.35, 9.42))
    a0, a1, a2, a3 = axp.ravel()
    sc = a0.scatter(
        cen["lon"],
        cen["lat"],
        c=cen["esi"],
        cmap="Spectral_r",
        s=60,
        alpha=0.86,
        edgecolors="#1b1d2044",
        linewidths=0.15,
        zorder=5,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(sc, ax=a0, label="ESI cell mean")
    if cen["hot"].any():
        a0.scatter(
            cen.loc[cen["hot"], "lon"],
            cen.loc[cen["hot"], "lat"],
            facecolors="none",
            edgecolors="#c83232",
            s=125,
            lw=1.12,
            zorder=7,
            label="Stress hotspot (>= p92 pooled)",
        )
    plpc = sns.color_palette("muted", len(PORTS_LL))
    for ix, (pna, (la_, ln_)) in enumerate(PORTS_LL.items()):
        a0.scatter(
            ln_,
            la_,
            marker="*",
            s=200,
            fc=plpc[ix],
            ec="#171717",
            lw=0.52,
            label=pna,
            zorder=12,
        )
    a0.legend(loc="lower center", ncol=4, fontsize=7.4)
    a0.set_xlabel("Longitude")
    a0.set_ylabel("Latitude")
    a0.set_title("Environmental stress hotspots")
    top_p = (
        cen.groupby("port", dropna=True)["esi"]
        .mean()
        .sort_values(ascending=False)
        .head(16)
    )
    a1.barh(top_p.index.astype(str)[::-1], top_p.values[::-1], color="#446b8b")
    a1.axvline(0, color="#333", lw=1)
    a1.set_xlabel("Mean ESI")
    a1.set_title("Port-aligned stress contrasts")
    dsc = pd.DataFrame(
        dict(
            ln_no2=np.log1p(pd.to_numeric(df["NO2_mean"], errors="coerce")),
            ndti=pd.to_numeric(df["ndti_mean"], errors="coerce"),
            esi=np.asarray(esi, dtype=float),
        )
    ).replace([np.inf, -np.inf], np.nan)
    dsc = dsc.dropna(how="any")
    hmn = float(dsc["esi"].quantile(0.04))
    hmx = float(dsc["esi"].quantile(0.97))
    sc2 = a2.scatter(
        dsc["ln_no2"],
        dsc["ndti"],
        c=dsc["esi"],
        cmap="rocket",
        vmin=hmn,
        vmax=hmx,
        s=36,
        lw=0,
        alpha=0.87,
    )
    plt.colorbar(sc2, ax=a2, label="ESI")
    a2.set_xlabel("ln(1 + NO2)")
    a2.set_ylabel("NDTI")
    a2.set_title("Aquatic continuum vs NO2 residuals")
    sns.histplot(cen["esi"].astype(float), kde=True, ax=a3, color="#2f5f90")
    a3.set_title("Cell-wise ESI distribution")
    sns.despine(ax=a0)
    sns.despine(ax=a1)
    sns.despine(ax=a2)
    sns.despine(ax=a3)
    fg.suptitle("Fig. 5.8 Environmental stress composite index")
    fg.tight_layout(rect=[0, 0, 1, 0.974])
    save_png(fg, "figure_5_8_environmental_stress.png")


def fig59(df: pd.DataFrame) -> None:
    friendly = dict(
        vessel_density_t="Vdensity",
        NO2_mean="NO2",
        ndti_mean="NDTI",
        ndwi_mean="NDWI",
        mei="MEI",
    )
    week_ix = pd.to_datetime(df["week_start_utc"], utc=True)
    pooled = (
        pd.DataFrame({vn: pd.to_numeric(df[vn], errors="coerce").values for vn in friendly})
        .assign(_w=week_ix)
        .groupby("_w")
        .median(numeric_only=True)
        .sort_index()
    )
    rows = []
    for vn in friendly:
        s = pooled[vn].astype(float)
        for L in (1, 2):
            a = s.shift(L).to_numpy(dtype=float)
            b = s.to_numpy(dtype=float)
            msk = np.isfinite(a) & np.isfinite(b)
            nv = int(np.sum(msk))
            if nv > 9:
                rho, _ = scipy_stats.spearmanr(a[msk], b[msk])
                pr = float(rho) if rho == rho else math.nan
            else:
                pr = math.nan
            rows.append(dict(metric=vn, friendly=friendly[vn], lag_weeks=L, rho_spearman=pr, n_windows=nv))
    lag_tbl = pd.DataFrame(rows)
    corr = pooled.astype(float).rename(columns=friendly).corr(method="spearman").stack().reset_index()
    corr.columns = ["metric_pair_a", "metric_pair_b", "rho_weekly_medians"]

    lag_save = lag_tbl.rename(columns=dict(rho_spearman="rho")).assign(segment="lag_autocorr_spearman", metric_pair_a=np.nan, metric_pair_b=np.nan)
    corr_save = (
        corr.rename(columns=dict(rho_weekly_medians="rho"))
        .assign(segment="weekly_median_corr", lag_weeks=math.nan, metric=np.nan, friendly=np.nan, n_windows=int(len(pooled)))
    )
    save_csv(pd.concat([lag_save, corr_save], ignore_index=True), "temporal_lag_statistics.csv")

    lag_wide = lag_tbl.set_index(["friendly", "lag_weeks"])["rho_spearman"].unstack(level=1)
    fig, axes = plt.subplots(2, 2, figsize=(13.06, 9.65))
    z = pooled.astype(float).rename(columns=friendly).apply(
        lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-11)
    )
    for col in z.columns:
        axes[0, 0].plot(z.index, z[col], lw=1.12, alpha=0.82, label=col)
    axes[0, 0].axhline(0, color="#999", lw=0.92)
    axes[0, 0].set_title("Weekly medians (z-scored)")
    axes[0, 0].legend(ncol=2, fontsize=7.8, loc="upper left")
    sns.heatmap(
        lag_wide,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=axes[0, 1],
        linewidths=0.45,
    )
    axes[0, 1].set_title("Lag-1 & lag-2 autocorrelation (Spearman)")
    piv_corr = corr.pivot(index="metric_pair_a", columns="metric_pair_b", values="rho_weekly_medians")
    sns.heatmap(piv_corr, annot=True, fmt=".2f", cmap="PuOr", vmin=-1, vmax=1, ax=axes[1, 0], linewidths=0.45)
    axes[1, 0].set_title("Contemporaneous Spearman (weekly medians)")
    roll = pooled[["mei", "NO2_mean"]].rolling(5, min_periods=2, center=True).mean()
    axb, b2 = axes[1, 1], axes[1, 1].twinx()
    axb.plot(roll.index, roll["NO2_mean"], lw=1.55, color="#223e8c", label="NO2 smooth")
    b2.plot(roll.index, roll["mei"], lw=1.25, linestyle="--", color="#aa2222", label="MEI smooth")
    axb.set_title("Rolling 5-week median persistence")
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.8)
    fig.suptitle("Fig. 5.9 Temporal lag structure")
    fig.tight_layout(rect=[0.02, 0.015, 0.985, 0.97])
    save_png(fig, "figure_5_9_temporal_lag_analysis.png")


def fig510(df: pd.DataFrame) -> None:
    friendly = dict(NO2_mean="NO2", mei="MEI", vessel_density_t="Vdensity", ndti_mean="NDTI", fai_mean="FAI")
    wk = (
        pd.DataFrame({vn: pd.to_numeric(df[vn], errors="coerce").values for vn in friendly})
        .assign(_w=pd.to_datetime(df["week_start_utc"], utc=True))
        .groupby("_w")
        .median(numeric_only=True)
        .sort_index()
        .rename(columns=friendly)
        .astype(float)
    )
    baseline = wk.rolling(9, center=True, min_periods=4).median()
    resid = wk - baseline
    q = resid.quantile(0.933, interpolation="nearest")
    flags = resid.ge(q, axis=1).astype(int)
    names = wk.columns.to_list()
    overlap = pd.DataFrame(
        np.dot(flags.values.T.astype(float), flags.values.astype(float)).astype(int),
        index=names,
        columns=names,
    )
    pair_rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            pair_rows.append(dict(var_row=a, var_col=b, co_anomaly_week_counts=int(overlap.iloc[i, j])))
    tally = pd.DataFrame(dict(metric_name=names, anomaly_week_counts=flags.sum(axis=0).astype(int), segment="per_metric_totals"))
    pair_df = pd.DataFrame(pair_rows).assign(segment="pairwise_overlap_weeks")

    save_csv(pd.concat([tally, pair_df], ignore_index=True), "anomaly_detection_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.92, 9.18))
    sns.heatmap(flags.T.astype(float), ax=axes[0, 0], cmap="YlOrRd", vmin=0, vmax=1, cbar=False, linewidths=0)
    axes[0, 0].set_title("Anomaly spikes (baseline-residual >= ~93rd pct)")
    axes[0, 0].set_ylabel("")
    sns.heatmap(
        overlap,
        annot=True,
        fmt="d",
        cmap="Purples",
        ax=axes[0, 1],
        linewidths=0.45,
        square=False,
    )
    axes[0, 1].set_title("Weekly co-occurrence matrix")
    hot = overlap.values.copy()
    np.fill_diagonal(hot, np.nanmax(hot) + 0.001)
    sort_idx = np.argsort(np.nansum(hot, axis=1))[::-1]
    sns.heatmap(
        overlap.iloc[sort_idx].iloc[:, sort_idx],
        ax=axes[1, 0],
        cmap="Blues",
        linewidths=0.45,
    )
    axes[1, 0].set_title("Stress overlap ordering (heatmap)")
    axl = axes[1, 1]
    stacked = flags.copy()
    stacked["co_cnt"] = flags.sum(axis=1)
    axl.fill_between(stacked.index, 0.0, stacked["co_cnt"], color="#5577aa44", lw=0)
    axl.plot(stacked.index, stacked["co_cnt"], color="#334477", lw=1.05, marker=".", ms=5)
    axl.axhline(math.ceil(stacked["co_cnt"].mean()), color="#aa6644", linestyle="--", lw=1)
    axl.set_title("Weekly count of simultaneous stress anomalies")
    fig.suptitle("Fig. 5.10 Rolling-baseline anomalies")
    fig.tight_layout(rect=[0.02, 0.015, 0.986, 0.97])
    save_png(fig, "figure_5_10_anomaly_analysis.png")


def main() -> None:
    dirs()
    df_panel = load_panel()
    fig55(df_panel)
    fig56(df_panel)
    fig57(df_panel)
    fig58(df_panel)
    fig59(df_panel)
    fig510(df_panel)


if __name__ == "__main__":
    main()
