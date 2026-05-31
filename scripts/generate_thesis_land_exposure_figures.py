#!/usr/bin/env python3
"""Thesis-ready land/coastal exposure figures from processed parquet + wind artefacts."""

from __future__ import annotations

import math
import re
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon as MplPolygon
from scipy import stats as scipy_stats
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import human_impact_distance_analysis as hid  # noqa: E402


from visualization.thesis_plot_ports import EXCLUDED_PORTS, PORT_COORDS, exclude_ports  # noqa: E402

PARQUET = ROOT / "processed" / "features_ml_ready.parquet"
WIND_CSV = ROOT / "outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv"
OUT = ROOT / "outputs/thesis_land_exposure"
MID = OUT / "intermediate"
NOTES_MD = OUT / "thesis_integration_notes.md"

CACHE_DIR = ROOT / "processed/basemap_cache"
CACHE_ZIP = CACHE_DIR / "ne_110m_land.zip"
NE_LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"

CRS_WGS84 = "EPSG:4326"

PORTS_ORDER = ["Turku", "Mariehamn"]
PORT_COORDS: dict[str, tuple[float, float]] = {
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
}

COAST_DISTANCE_MAX_KM = 50.0
SHIPPING_DISTANCE_MAX_KM_ARCHIVAL = 30.0
SHIPPING_EXTENDED_INLAND_PORTS = frozenset()
SHIPPING_EXTENDED_CAP_KM = 7200.0


def shipping_ok(distance_ship_km: pd.Series, nearest_port_lab: pd.Series) -> pd.Series:
    ds_ = pd.to_numeric(distance_ship_km, errors="coerce")
    archival = ds_.notna() & (ds_ >= 0) & (ds_ < SHIPPING_DISTANCE_MAX_KM_ARCHIVAL)
    inland = nearest_port_lab.astype(str).isin(SHIPPING_EXTENDED_INLAND_PORTS)
    extended = inland & ds_.notna() & (ds_ >= 0) & (ds_ < SHIPPING_EXTENDED_CAP_KM)
    return archival | extended

LAND_BAND_LABELS_ORDER = ["0–3 km", "3–10 km", "10–30 km", "30–50 km"]

BALTIC_LAT_RANGE = (53.8, 67.6)
BALTIC_LON_RANGE = (8.4, 31.2)
CORRIDOR_PCT_GLOBAL = 88

DPI_SAVE = 360
_CELL_RES_RE = re.compile(r"^g(?P<res>[\d.]+)_")

# Corridor panels for directional transport schematic
_TRANSPORT_BOX = dict(
    Turku=dict(title="Turku littoral corridor", lon0=21.07, lon1=23.32, lat0=59.82, lat1=61.92),
    Aaland=dict(title="Åland maritime corridor", lon0=18.95, lon1=21.15, lat0=59.90, lat1=60.46),
)


def thesis_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.92,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
            "font.size": 10.4,
            "axes.titlesize": 11.4,
            "axes.labelsize": 10.3,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "figure.dpi": 110,
            "savefig.dpi": DPI_SAVE,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "lines.linewidth": 1.6,
            "lines.markersize": 7.2,
            "axes.grid": True,
            "grid.alpha": 0.26,
            "grid.linestyle": ":",
            "grid.linewidth": 0.65,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / stem
    fig.savefig(p.with_suffix(".png"), dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(p.with_suffix(".pdf"), dpi=DPI_SAVE, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def parse_cell_resolution_deg(grid_cell_id: str, default_deg: float = 0.1) -> float:
    m = _CELL_RES_RE.match(str(grid_cell_id))
    if not m:
        return default_deg
    try:
        return float(m.group("res"))
    except ValueError:
        return default_deg


def square_polygon(lon: float, lat: float, res_deg: float):
    half = float(res_deg) / 2.0
    return box(float(lon) - half, float(lat) - half, float(lon) + half, float(lat) + half)


def load_land_union() -> Any:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_ZIP.is_file():
        for ctx in (ssl.create_default_context(), ssl._create_unverified_context()):
            try:
                req = urllib.request.Request(NE_LAND_URL, headers={"User-Agent": "geospatial-thesis/land-exposure"})
                with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
                    CACHE_ZIP.write_bytes(r.read())
                    break
            except Exception:
                continue
        if not CACHE_ZIP.is_file():
            raise RuntimeError("Could not download Natural Earth land zip.")
    land = gpd.read_file(f"zip://{CACHE_ZIP}!ne_110m_land.shp").to_crs(CRS_WGS84)
    if hasattr(land.geometry, "union_all"):
        return land.geometry.union_all()
    from shapely.ops import unary_union

    return unary_union(land.geometry)


def point_on_land(lat: np.ndarray, lon: np.ndarray, land_union: Any) -> np.ndarray:
    pts = [Point(float(lo), float(la)) for la, lo in zip(lat, lon, strict=False)]
    return np.asarray([land_union.contains(p) or land_union.touches(p) for p in pts], dtype=bool)


def esi_series(df: pd.DataFrame) -> pd.Series:
    cols = [
        "maritime_pressure_index",
        "NO2_mean",
        "fai_mean",
        "ndti_mean",
        "vessel_density_t",
        "coastal_exposure_score",
    ]
    if any(c not in df.columns for c in cols):
        return pd.Series(np.nan, index=df.index)
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    zparts = [(x[c] - x[c].mean()) / (float(x[c].std(ddof=0) or 1.0)) for c in cols]
    return pd.concat(zparts, axis=1).mean(axis=1)


def assign_land_coast_band(dc_km: pd.Series) -> pd.Series:
    d = pd.to_numeric(dc_km, errors="coerce")
    out = pd.Series(pd.NA, index=dc_km.index, dtype=object)
    out.loc[d.notna() & (d >= 0) & (d < 3)] = LAND_BAND_LABELS_ORDER[0]
    out.loc[d.notna() & (d >= 3) & (d < 10)] = LAND_BAND_LABELS_ORDER[1]
    out.loc[d.notna() & (d >= 10) & (d < 30)] = LAND_BAND_LABELS_ORDER[2]
    out.loc[d.notna() & (d >= 30) & (d <= COAST_DISTANCE_MAX_KM)] = LAND_BAND_LABELS_ORDER[3]
    return out


def mean_se(series: pd.Series) -> tuple[float, float, int]:
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return mu, se, n


def band_centers_numeric() -> dict[str, float]:
    return dict(zip(LAND_BAND_LABELS_ORDER, [1.5, 6.5, 20.0, 40.0], strict=True))


def build_base_panel() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True)
    df["nearest_port"] = df["nearest_port"].astype(str)
    df = exclude_ports(df)

    if Path(WIND_CSV).is_file():
        w = pd.read_csv(WIND_CSV)
        w["week_start_utc"] = pd.to_datetime(w["week_start_utc"], utc=True)
        extra = [c for c in ("wind_u_mean", "wind_v_mean", "coastal_wind_shoreward_45deg") if c in w.columns]
        keep = ["grid_cell_id", "week_start_utc"] + extra
        df = df.merge(w[keep], on=["grid_cell_id", "week_start_utc"], how="left")

    sh = pd.to_numeric(df.get("coastal_wind_shoreward_45deg"), errors="coerce")
    wr = np.full(len(df), "", dtype=object)
    wr[np.asarray(sh >= 1)] = "shoreward"
    wr[np.asarray(sh.notna() & (sh < 1))] = "nonshoreward"
    df["_wind_regime"] = wr
    df["environmental_stress_index"] = esi_series(df)

    uniq = df.drop_duplicates("grid_cell_id").copy()
    lat_u = pd.to_numeric(uniq["grid_centroid_lat"], errors="coerce").to_numpy(dtype=float)
    lon_u = pd.to_numeric(uniq["grid_centroid_lon"], errors="coerce").to_numpy(dtype=float)
    land_union = load_land_union()
    uniq["_on_land_mask"] = point_on_land(lat_u, lon_u, land_union)

    coast_pts = hid.load_coastline_points(Path(CACHE_DIR))
    if coast_pts is None:
        coast_pts = hid.load_land_boundary_points(Path(CACHE_DIR))
        print("[warn] Using land-boundary fallback for shoreline distance.")

    clat, clon = coast_pts
    uniq["distance_to_coast_km"] = hid.distance_to_coast_km_for_grids(lat_u, lon_u, clat, clon)

    g_dc = uniq.set_index("grid_cell_id")["distance_to_coast_km"]
    g_ld = uniq.set_index("grid_cell_id")["_on_land_mask"]

    df["distance_to_coast_km"] = df["grid_cell_id"].map(g_dc).astype(float)
    df["_on_land_mask"] = df["grid_cell_id"].map(g_ld).fillna(False)

    latp = pd.to_numeric(df["grid_centroid_lat"], errors="coerce")
    lonp = pd.to_numeric(df["grid_centroid_lon"], errors="coerce")
    df["_baltic_window"] = latp.between(*BALTIC_LAT_RANGE) & lonp.between(*BALTIC_LON_RANGE)

    ds = pd.to_numeric(df["distance_to_nearest_high_vessel_density_cell"], errors="coerce")
    dc = pd.to_numeric(df["distance_to_coast_km"], errors="coerce")

    ship_ok = shipping_ok(ds, df["nearest_port"])

    df["_receiver_coastal_annulus"] = (
        dc.notna()
        & (dc <= COAST_DISTANCE_MAX_KM)
        & ship_ok
        & latp.between(*BALTIC_LAT_RANGE)
        & lonp.between(*BALTIC_LON_RANGE)
    )

    df["_strict_land_annulus"] = df["_receiver_coastal_annulus"] & df["_on_land_mask"]

    df["_coastal_land_shipping_annulus"] = df["_receiver_coastal_annulus"]

    ba = pd.to_numeric(df["NO2_mean"], errors="coerce")
    wkn = df["week_start_utc"].dt.normalize()
    df["_no2_weekly_anomaly"] = ba - df.groupby(wkn, sort=False)["NO2_mean"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").mean(),
    )

    df["_land_coast_band"] = assign_land_coast_band(df["distance_to_coast_km"])
    df.loc[~df["_coastal_land_shipping_annulus"], "_land_coast_band"] = pd.NA

    return df


def print_slice_stats(name: str, sub: pd.DataFrame, *, extras: list[str] | None = None) -> None:
    print(f"\n=== Validation · {name} ===")
    print(f"total_rows={len(sub):,}")
    if sub.empty:
        return
    if "distance_to_coast_km" in sub.columns:
        print(f"non_null_distance_to_coast_km={pd.to_numeric(sub['distance_to_coast_km'], errors='coerce').notna().sum():,}")
    print(f"unique_grid_cells={sub['grid_cell_id'].nunique():,}")
    need = "_land_coast_band"
    if need in sub.columns:
        vc_sub = pd.Series(sub[need])
        vc = vc_sub.astype(str).value_counts(dropna=False).to_dict()
        print("distance_band_counts:")
        for lbl in LAND_BAND_LABELS_ORDER:
            ct = vc.get(str(lbl), vc.get(str(lbl).replace("-", "-"), 0))
            print(f"  {lbl}: {int(ct)}")
        unk = vc_sub.isna().sum()
        if unk:
            print(f"  <unassigned_nan>: {int(unk)}")
    if "nearest_port" in sub.columns:
        for pt in PORTS_ORDER:
            m = sub["nearest_port"].astype(str).eq(pt)
            print(f"port[{pt}] rows={m.sum():,} grids={sub.loc[m,'grid_cell_id'].nunique()}")

    nm = pd.to_numeric(sub.get("_no2_weekly_anomaly"), errors="coerce").notna() if "_no2_weekly_anomaly" in sub.columns else None
    if nm is not None:
        print(f"_no2_weekly_anomaly_non_null_rows={nm.sum():,}")
    nw = pd.to_numeric(sub.get("wind_u_mean"), errors="coerce").notna() if "wind_u_mean" in sub.columns else None
    if nw is not None:
        print(f"wind_uv_non_null_rows={nw.sum():,}")

    for key in metrics_non_null_summary(sub):
        print(key)
    if extras:
        for e in extras:
            print(e)


def metrics_non_null_summary(sub: pd.DataFrame) -> list[str]:
    out = []
    for c in ("NO2_mean", "maritime_pressure_index", "coastal_exposure_score", "environmental_stress_index"):
        if c in sub.columns:
            nn = pd.to_numeric(sub[c], errors="coerce").notna().sum()
            out.append(f"non_null[{c}]={int(nn):,}")
    return out


def figure_a(analysis: pd.DataFrame) -> None:
    df = analysis[analysis["_land_coast_band"].notna()].copy()
    print_slice_stats("Figure A landward decay", df)

    metrics = [
        ("NO2_mean", "Mean NO₂ (panel-native units)", "NO₂"),
        ("maritime_pressure_index", "Maritime Exposure Index", "MEI"),
        ("coastal_exposure_score", "Mean coastal exposure score", "CES"),
        ("environmental_stress_index", "Environmental Stress Index (ESI)", "ESI"),
    ]
    xs = np.arange(len(LAND_BAND_LABELS_ORDER), dtype=float)
    cmap = sns.color_palette("colorblind", n_colors=len(PORTS_ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(10.95, 8.92))
    axes = axes.ravel()
    csv_rows = []
    for ax, (col, ylab, _) in zip(axes, metrics, strict=True):
        ymax = -math.inf
        for pj, pt in enumerate(PORTS_ORDER):
            mus: list[float] = []
            ses: list[float] = []
            for bl in LAND_BAND_LABELS_ORDER:
                ser = df.loc[
                    (df["_land_coast_band"].astype(str).eq(bl)) & df["nearest_port"].astype(str).eq(pt),
                    col,
                ]
                mu, se_, n = mean_se(ser)
                mus.append(mu)
                ses.append(se_)
                csv_rows.append(
                    dict(
                        figure="A",
                        metric=col,
                        port=pt,
                        shoreline_band_km_label=bl,
                        mean=float(mu) if mu == mu else math.nan,
                        se=float(se_) if se_ == se_ else math.nan,
                        n_rows=int(n),
                        non_null=int(pd.to_numeric(ser, errors="coerce").notna().sum()),
                    ),
                )
            xe = xs + (pj - 1) * 0.22
            yerr = [float(s) if s == s and math.isfinite(float(s)) else 0.0 for s in ses]
            ax.errorbar(
                xe,
                mus,
                yerr=yerr,
                fmt="-o",
                capsize=3.8,
                color=cmap[pj],
                linewidth=1.92,
                label=pt if ax is axes[0] else "_nolegend_",
                markeredgecolor="#111827",
                markeredgewidth=0.5,
                ecolor="#475569",
            )
            for mu, sv in zip(mus, ses):
                if mu == mu:
                    ymax = max(ymax, mu + ((sv if sv == sv else 0.0)))
        ax.set_xticks(xs)
        ax.set_xticklabels(LAND_BAND_LABELS_ORDER)
        ax.set_title(ylab, fontsize=11.05, fontweight="semibold")
        ax.set_xlabel("Distance to shoreline (km bands; chords to NE 110m coast)")
        if ymax > -math.inf and ymax > 0:
            ax.set_ylim(0.0, ymax * 1.13)

    axes[0].legend(title="Archived nearest_port assignment", fontsize=9, title_fontsize=9.35, framealpha=0.95)
    fig.suptitle(
        "Shore-distance decay of exposure stressors (nearshore/coastal lattice, shipping corridor annulus)",
        fontsize=12.95,
        fontweight="bold",
        y=1.01,
    )
    fig.subplots_adjust(bottom=0.089, left=0.069, top=0.932, right=0.986, hspace=0.33, wspace=0.32)
    save_fig(fig, "fig_landward_decay")

    pd.DataFrame(csv_rows).sort_values(["metric", "port", "shoreline_band_km_label"]).to_csv(
        OUT / "landward_decay_summary.csv",
        index=False,
    )


def figure_b(analysis: pd.DataFrame) -> None:
    df = analysis[
        analysis["_land_coast_band"].notna()
        & analysis["_wind_regime"].isin(["shoreward", "nonshoreward"])
        & analysis["nearest_port"].isin(PORTS_ORDER)
    ].copy()

    extras = ["Coastal ±45° shoreward flag from archival wind merges (structured missingness unchanged)."]
    print_slice_stats("Figure B shoreward vs non (land coastal bands)", df, extras=extras)

    metrics = [
        ("NO2_mean", "Mean NO₂", "NO₂"),
        ("maritime_pressure_index", "Mean MEI", "MEI"),
        ("coastal_exposure_score", "Mean coastal exposure", "CES"),
        ("environmental_stress_index", "Mean ESI", "ESI"),
    ]

    fig_bar, axes = plt.subplots(2, 2, figsize=(12.92, 9.92))
    axes = axes.ravel()
    cmap_p = sns.color_palette("muted", n_colors=len(PORTS_ORDER))

    csv_rows_bar: list[dict[str, Any]] = []

    seq_ports_wind = [(pt, wd) for pt in PORTS_ORDER for wd in ("shoreward", "nonshoreward")]
    n_grp = len(seq_ports_wind)
    group_span = 0.78
    bw = group_span / n_grp * 1.06

    for ax_i, (col, ylab, _) in enumerate(metrics):
        ax = axes[ax_i]
        centers = np.arange(len(LAND_BAND_LABELS_ORDER), dtype=float)
        ymax_l = -math.inf

        for bi, band_lab in enumerate(LAND_BAND_LABELS_ORDER):
            for k, (pt, wd) in enumerate(seq_ports_wind):
                ser = df.loc[
                    df["_land_coast_band"].astype(str).eq(band_lab)
                    & df["nearest_port"].astype(str).eq(pt)
                    & df["_wind_regime"].astype(str).eq(wd),
                    col,
                ]
                mu, se_, n = mean_se(ser)
                csv_rows_bar.append(
                    dict(
                        figure="B_bar",
                        metric=col,
                        port=pt,
                        wind_regime=wd,
                        shoreline_band_km_label=band_lab,
                        mean=float(mu) if mu == mu else math.nan,
                        se=float(se_) if se_ == se_ else math.nan,
                        n_rows=int(n),
                    ),
                )

                xpos = bi - group_span / 2 + bw / 2 + k * bw
                fc = cmap_p[PORTS_ORDER.index(pt)]
                ax.bar(
                    xpos,
                    mu if mu == mu else 0.0,
                    width=bw * 0.97,
                    color=fc,
                    edgecolor="#1f2937",
                    linewidth=0.55,
                    hatch="//" if wd == "shoreward" else "",
                    alpha=0.86 if wd == "shoreward" else 0.72,
                    yerr=(se_ if se_ == se_ and math.isfinite(float(se_)) else 0.0),
                    capsize=3.0,
                    error_kw=dict(ecolor="#475569", elinewidth=1.0),
                    label="_nl",
                )
                if mu == mu:
                    ymax_l = max(ymax_l, mu + float(se_) if se_ == se_ else mu)

        ax.set_xticks(centers)
        ax.set_xticklabels(LAND_BAND_LABELS_ORDER, rotation=0)
        ax.set_title(ylab, fontsize=11.05, fontweight="semibold")
        ax.set_xlabel("Land–shore distance bins · shipping annulus (same mask as Fig. A)")
        if ymax_l > -math.inf and ymax_l > 0:
            ax.set_ylim(0.0, ymax_l * 1.17)

        if ax_i == 3:
            hatches = (
                Patch(facecolor="gray", edgecolor="#1f2937", hatch="//", linewidth=0.6, alpha=0.86, label="Shorewind (±45°)"),
                Patch(facecolor="gray", edgecolor="#1f2937", alpha=0.72, linewidth=0.6, label="Non-shoreward"),
            )
            leg_a = ax.legend(handles=hatches, loc="upper left", fontsize=8.85, frameon=True, title="Wind regime")
            ax.add_artist(leg_a)
            hh = tuple(
                mlines.Line2D([0], [0], linestyle="none", marker="s", markersize=8, color=cmap_p[i], label=PORTS_ORDER[i], markeredgecolor="#111827")
                for i in range(len(PORTS_ORDER))
            )
            ax.legend(handles=list(hh), loc="upper right", fontsize=8.85, ncol=3, frameon=False, title="Colours · port linkage")

    fig_bar.suptitle("Shoreward vs nonshoreward exposures by shoreline-distance bin (mean ± SE)", fontsize=12.95, fontweight="bold", y=0.987)
    fig_bar.subplots_adjust(left=0.067, bottom=0.069, top=0.925, right=0.986, hspace=0.33, wspace=0.30)
    save_fig(fig_bar, "fig_wind_land_exposure")

    pd.DataFrame(csv_rows_bar).to_csv(OUT / "wind_land_exposure_summary.csv", index=False)

    _figure_b_violins(df)


def _figure_b_violins(df: pd.DataFrame) -> None:
    metrics = [
        ("NO2_mean", "NO₂"),
        ("maritime_pressure_index", "MEI"),
        ("coastal_exposure_score", "CES"),
        ("environmental_stress_index", "ESI"),
    ]
    fig_v, axes_v = plt.subplots(4, 3, figsize=(13.85, 16.82))
    for mi, (col, ylab) in enumerate(metrics):
        for pi, pt in enumerate(PORTS_ORDER):
            ax = axes_v[mi][pi]
            sub = df[df["nearest_port"].astype(str).eq(pt)].copy()
            sub["_band_cat"] = pd.Categorical(sub["_land_coast_band"], categories=LAND_BAND_LABELS_ORDER, ordered=True)

            vv = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vv):
                ql, qh = float(vv.quantile(0.01)), float(vv.quantile(0.99))
                sub["_yv"] = pd.to_numeric(sub[col], errors="coerce").clip(ql, qh)
            else:
                sub["_yv"] = np.nan

            kws = dict(
                data=sub,
                x="_band_cat",
                y="_yv",
                hue="_wind_regime",
                ax=ax,
                order=LAND_BAND_LABELS_ORDER,
                hue_order=["shoreward", "nonshoreward"],
                cut=0,
                inner="quartile",
                density_norm="width",
                palette={"shoreward": "#2f5597", "nonshoreward": "#b18bd6"},
                linewidth=1.02,
                saturation=0.95,
                legend=False,
            )
            try:
                sns.violinplot(**kws, gap=0.1)
            except TypeError:
                sns.violinplot(**kws)

            if mi == 0:
                ax.set_title(pt, fontsize=10.96, pad=7)
            if pi != 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(ylab)

            if mi < 3:
                ax.set_xlabel("")
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
                ax.set_xlabel("Shore-distance bin")

            ax.text(
                0.02,
                0.98,
                ylab,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.95,
                color="#475569",
            )

    fig_v.suptitle(
        "Distributional comparison (violins · 1–99% trims; width = kernel mass)",
        fontsize=12.95,
        fontweight="bold",
        y=0.994,
    )
    hh = plt.legend(handles=[Patch(color="#2f5597"), Patch(color="#b18bd6")], labels=["Shorewind", "Non-shoreward"], loc="lower center")
    hh.remove()

    ln = [
        Patch(facecolor="#2f5597", edgecolor="#1f2937", linewidth=0.55, label="Shorewind"),
        Patch(facecolor="#b18bd6", edgecolor="#1f2937", linewidth=0.55, label="Non-shoreward"),
    ]
    fig_v.legend(handles=ln, loc="lower center", ncol=4, fontsize=10.05, bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig_v.subplots_adjust(left=0.066, bottom=0.068, top=0.959, right=0.986, hspace=0.33, wspace=0.25)
    save_fig(fig_v, "fig_wind_land_violin")


def _agg_land_cells(analysis: pd.DataFrame) -> pd.DataFrame:
    """Spatial rows (nearshore/coastal lattice, optional strict land tagging)."""

    uniq = analysis.drop_duplicates("grid_cell_id").set_index("grid_cell_id")
    ves_med = analysis.groupby("grid_cell_id", observed=False)["vessel_density_t"].median()

    mask = uniq["_coastal_land_shipping_annulus"].astype(bool) & uniq["nearest_port"].astype(str).isin(PORTS_ORDER)
    base = uniq.loc[mask]

    agg_core = analysis[analysis["_land_coast_band"].notna()].groupby("grid_cell_id").agg(
        ces_med=("coastal_exposure_score", lambda s: pd.to_numeric(s, errors="coerce").median()),
        esi_med=("environmental_stress_index", lambda s: pd.to_numeric(s, errors="coerce").median()),
        mei_med=("maritime_pressure_index", lambda s: pd.to_numeric(s, errors="coerce").median()),
        no2_med=("NO2_mean", lambda s: pd.to_numeric(s, errors="coerce").median()),
    ).reindex(base.index)

    return pd.DataFrame(
        {
            "grid_cell_id": base.index.astype(str),
            "grid_centroid_lat": pd.to_numeric(base["grid_centroid_lat"], errors="coerce"),
            "grid_centroid_lon": pd.to_numeric(base["grid_centroid_lon"], errors="coerce"),
            "nearest_port": base["nearest_port"].astype(str),
            "distance_to_coast_km": pd.to_numeric(base["distance_to_coast_km"], errors="coerce"),
            "land_coast_band": assign_land_coast_band(pd.to_numeric(base["distance_to_coast_km"], errors="coerce")),
            "strict_land_centroid_ne110": base["_on_land_mask"].astype(bool),
            "ces_med": agg_core["ces_med"].astype(float),
            "esi_med": agg_core["esi_med"].astype(float),
            "mei_med": agg_core["mei_med"].astype(float),
            "no2_med": agg_core["no2_med"].astype(float),
            "vessel_med": ves_med.reindex(base.index).astype(float),
        },
    ).reset_index(drop=True)


def figure_c(analysis: pd.DataFrame) -> None:
    MID.mkdir(parents=True, exist_ok=True)

    ves_g = analysis.groupby("grid_cell_id", observed=False)["vessel_density_t"].median().astype(float)
    thr_cor = float(np.percentile(ves_g.dropna().values, CORRIDOR_PCT_GLOBAL)) if ves_g.dropna().size else float("nan")
    corridor_cells = set(ves_g.index[ves_g >= thr_cor].astype(str))

    rows = _agg_land_cells(analysis)
    print_slice_stats(
        "Figure C hotspots · unique land cells exported",
        pd.DataFrame({"grid_cell_id": rows["grid_cell_id"], "nearest_port": rows["nearest_port"]}),  # type: ignore[list-item]
    )

    for pt in PORTS_ORDER:
        m = rows["nearest_port"].eq(pt)
        if not bool(m.any()):
            continue
        th_c = float(rows.loc[m, "ces_med"].quantile(0.90)) if rows.loc[m, "ces_med"].notna().any() else math.nan
        th_e = float(rows.loc[m, "esi_med"].quantile(0.90)) if rows.loc[m, "esi_med"].notna().any() else math.nan
        rows.loc[m, "_thr90_ces"] = th_c
        rows.loc[m, "_thr90_esi"] = th_e

    rows["hot_ces_port_scope"] = rows["ces_med"] >= rows["_thr90_ces"]
    rows["hot_esi_port_scope"] = rows["esi_med"] >= rows["_thr90_esi"]

    rows.to_csv(OUT / "land_hotspot_cells.csv", index=False)
    zooms = dict(
        Turku=dict(lon_min=21.07, lon_max=23.28, lat_min=59.74, lat_max=61.92),
        Mariehamn=dict(lon_min=18.74, lon_max=20.32, lat_min=59.84, lat_max=60.44),
    )

    land_union = load_land_union()
    fig = plt.figure(figsize=(14.92, 8.94))

    for ip, pname in enumerate(PORTS_ORDER):
        bb = zooms[pname]
        bx = box(bb["lon_min"], bb["lat_min"], bb["lon_max"], bb["lat_max"])

        clip = gpd.clip(gpd.GeoDataFrame(geometry=gpd.GeoSeries([land_union], crs=CRS_WGS84), crs=CRS_WGS84), bx)
        subset = rows[rows["nearest_port"].eq(pname)]

        for iy, (flag, clr, ttl) in enumerate(
            (
                ("hot_ces_port_scope", "#b45309", "P90 CES (median)"),
                ("hot_esi_port_scope", "#6b21a8", "P90 ESI composite (median)"),
            ),
        ):
            ax = plt.subplot2grid((2, 3), (iy, ip))
            if not clip.empty:
                clip.boundary.plot(ax=ax, color="#59402b", lw=0.55, alpha=1.0, zorder=3)

            for _, r in subset.iterrows():
                lon_ = float(r["grid_centroid_lon"]) if pd.notna(r["grid_centroid_lon"]) else math.nan
                lat_ = float(r["grid_centroid_lat"]) if pd.notna(r["grid_centroid_lat"]) else math.nan
                if not math.isfinite(lon_) or not math.isfinite(lat_):
                    continue
                p = Point(lon_, lat_)
                if not bx.intersects(p):
                    continue

                gid = str(r["grid_cell_id"])
                rd = parse_cell_resolution_deg(gid)
                geom = square_polygon(lon_, lat_, rd)

                hotspot = bool(r[flag])
                in_corridor = gid in corridor_cells

                if hotspot:
                    face = clr
                    edc = "#1f2937"
                    lw_poly = 1.52
                    z = 14
                else:
                    face = "#e9eef5"
                    edc = "#94a3b8"
                    lw_poly = 0.28 + (1.08 if in_corridor else 0)
                    z = 3

                gsp = gpd.GeoSeries([geom], crs=CRS_WGS84)
                gx = gsp.clip(box(bb["lon_min"] - 0.01, bb["lat_min"] - 0.01, bb["lon_max"] + 0.01, bb["lat_max"] + 0.01))
                gx.plot(ax=ax, facecolor=face, edgecolor=edc, linewidth=lw_poly, alpha=1.0, zorder=z)

                if in_corridor:
                    gx.plot(ax=ax, facecolor="none", edgecolor="#0f172a", linewidth=0.95, linestyle=(0, (3.0, 1.85)), alpha=1.0, zorder=z + 8)

            plat, plon = PORT_COORDS[pname]
            ax.scatter([plon], [plat], s=146, marker="*", c="#facc15", edgecolors="#1f2937", linewidths=0.55, zorder=120)
            ax.set_xlim(bb["lon_min"], bb["lon_max"])
            ax.set_ylim(bb["lat_min"], bb["lat_max"])
            ax.set_aspect("equal", adjustable="box")

            ttl_line = ttl if iy == 0 else ttl
            ax.set_title(f"{pname} · hotspot · {ttl_line}", fontsize=10.95)

            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8.92)
            if iy == 1:
                ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel(("Latitude (°N)" if ip == 0 else "") + (""))

    corr_l = mlines.Line2D(
        [], [],
        linestyle=(0, (3.0, 1.85)),
        linewidth=1.92,
        color="#0f172a",
        label=f"Median vessel-density ≥ Baltic-wide P{CORRIDOR_PCT_GLOBAL} (shipping corridor rims)",
    )
    star_h = mlines.Line2D(
        [], [],
        linestyle="none",
        marker="*",
        markersize=12,
        color="#facc15",
        markeredgecolor="#111827",
        label="Archived port centroid (star)",
    )
    lg = fig.legend(handles=[corr_l, star_h], ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.002), fontsize=9.05, frameon=True)
    lg.get_frame().set_alpha(0.96)
    fig.subplots_adjust(bottom=0.092, left=0.049, top=0.942, right=0.986, hspace=0.28, wspace=0.2)
    fig.suptitle("Coastal-city land lattice · CES / ESI hotspots (per-port × P90 thresholds over medians)", fontsize=13.1, fontweight="bold")

    extras = [
        "Hotspot tiers use within-port P90 thresholds on cell-wise temporal medians.",
        "Corridor rims: Baltic-wide P88 on temporal median vessel density (same rule as maritime corridor figures).",
        f"exported_unique_land_cells={len(rows):,}",
    ]
    print_slice_stats("Figure C hotspots · export summary", rows, extras=extras)
    save_fig(fig, "fig_land_hotspots")


def figure_d(analysis: pd.DataFrame) -> None:
    MID.mkdir(parents=True, exist_ok=True)

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
    uniq = analysis.drop_duplicates("grid_cell_id")[["grid_cell_id", "grid_centroid_lat", "grid_centroid_lon"]]
    uniq = uniq.copy()
    uniq["grid_centroid_lat"] = pd.to_numeric(uniq["grid_centroid_lat"], errors="coerce")
    uniq["grid_centroid_lon"] = pd.to_numeric(uniq["grid_centroid_lon"], errors="coerce")
    uw = uw.merge(uniq.rename(columns={"grid_centroid_lat": "lat", "grid_centroid_lon": "lon"}), on="grid_cell_id")

    subset = analysis.loc[analysis["grid_cell_id"].isin(uw["grid_cell_id"])]
    print_slice_stats(
        "Figure D · cells with centroid + wind merges",
        subset,
        extras=[
            "Colours = temporal median at each centroid (no interpolated ocean surface between cells).",
            "Arrows denote median u/v (east / north wind components); scale is shared within each subplot.",
            "Åland corridor panel highlights Mariehamn's maritime fringe and adjacent shipping corridors.",
        ],
    )

    land_union = load_land_union()
    fig, axes = plt.subplots(1, 2, figsize=(15.92, 6.94))

    panel_cfgs = [
        dict(
            box_key="Turku",
            fld="no2a_m",
            cmap_name="PuOr_r",
            label="Median NO₂ weekly anomaly (observation − weekly mean; panel-native units)",
            highlight_ports=("Turku",),
        ),
        dict(
            box_key="Aaland",
            fld="mei_m",
            cmap_name="viridis",
            label="Median MEI (`maritime_pressure_index`, dimensionless)",
            highlight_ports=("Mariehamn",),
        ),
    ]

    for ax, cfg in zip(axes, panel_cfgs, strict=True):
        spec = _TRANSPORT_BOX[cfg["box_key"]]
        lon0, lon1, la0, la1 = float(spec["lon0"]), float(spec["lon1"]), float(spec["lat0"]), float(spec["lat1"])
        bbox = box(lon0 - 1e-3, la0 - 1e-3, lon1 + 1e-3, la1 + 1e-3)

        lc = gpd.clip(gpd.GeoDataFrame(geometry=gpd.GeoSeries([land_union], crs=CRS_WGS84), crs=CRS_WGS84), bbox.buffer(6e-3))
        if lc is not None and len(lc.index):
            lc.boundary.plot(ax=ax, lw=0.55, edgecolor="#4b3621", alpha=1.0, zorder=1)

        chunk = uw[(uw["lat"] >= la0 - 2e-2) & (uw["lat"] <= la1 + 2e-2) & (uw["lon"] >= lon0 - 2e-2) & (uw["lon"] <= lon1 + 2e-2)].copy()
        cmap = plt.get_cmap(cfg["cmap_name"])
        vals_all = pd.to_numeric(chunk[cfg["fld"]], errors="coerce")

        vmin, vmax = float(np.nanquantile(vals_all, 0.05)), float(np.nanquantile(vals_all, 0.95))
        if not math.isfinite(vmin) or not math.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        elif abs(vmax - vmin) < 1e-9:
            vmin, vmax = vmin - 0.05, vmin + 0.05

        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lon0, lon1)
        ax.set_ylim(la0, la1)
        ax.set_autoscale_on(False)

        quiver_rows = []
        for _, r in chunk.iterrows():
            lo, la = pd.to_numeric(r["lon"], errors="coerce"), pd.to_numeric(r["lat"], errors="coerce")
            if (
                lo != lo
                or la != la
                or not math.isfinite(float(lo))
                or not math.isfinite(float(la))
            ):
                continue
            gid = str(r["grid_cell_id"])
            geom = square_polygon(float(lo), float(la), parse_cell_resolution_deg(gid))
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
            poly = MplPolygon(coords, closed=True, facecolor=fc, edgecolor="#64748b", linewidth=0.28, alpha=1.0, zorder=3)
            poly.set_clip_on(True)
            ax.add_patch(poly)

            uu = pd.to_numeric(r["u_med"], errors="coerce")
            vv = pd.to_numeric(r["v_med"], errors="coerce")
            if math.isfinite(float(uu)) and math.isfinite(float(vv)):
                quiver_rows.append((float(lo), float(la), float(uu), float(vv)))

        if quiver_rows:
            q_arr = np.asarray(quiver_rows, dtype=float)
            q_arr = q_arr[np.isfinite(q_arr).all(axis=1)]
            if q_arr.size:
                stride = max(1, q_arr.shape[0] // 55)
                q_sub = q_arr[::stride, :]
                lon_q = q_sub[:, 0]
                lat_q = q_sub[:, 1]
                u_q = q_sub[:, 2]
                v_q = q_sub[:, 3]
                spd = np.hypot(u_q, v_q)
                ref = float(np.nanquantile(spd, 0.85))
                scale = max(420.0, ref * 28.0) if math.isfinite(ref) and ref > 1e-6 else 840.0
                ax.quiver(
                    lon_q,
                    lat_q,
                    u_q,
                    v_q,
                    angles="uv",
                    scale=float(np.clip(scale, 220.0, 3200.0)),
                    width=5.2e-3,
                    headwidth=3.95,
                    headlength=6.2,
                    color="#111827",
                    alpha=0.72,
                    zorder=42,
                    clip_on=True,
                )

        for pname in cfg["highlight_ports"]:
            plat, plon = PORT_COORDS[pname]
            if bbox.intersects(Point(plon, plat)):
                ax.scatter(plon, plat, s=138, marker="*", c="#fcd34d", edgecolors="#111827", linewidths=0.52, zorder=60)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm._A = []

        cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.048, pad=0.12, shrink=0.85)
        cb.set_label(cfg["label"])

        ax.set_title(spec["title"], fontsize=11.45, pad=11)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        hl = ",".join(cfg["highlight_ports"])
        ax.scatter([], [], marker="*", c="#fcd34d", edgecolors="#111827", linewidths=0.52, s=138, label=f"Stars: {hl}")
        ax.legend(loc="upper right", fontsize=8.92, frameon=True)
    fig.suptitle(
        "Wind-aligned transport schematic on discrete lattice (NO₂ anomalies vs MEI gradients)",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.89, bottom=0.098, left=0.055, right=0.985, wspace=0.15)
    save_fig(fig, "fig_wind_transport_landward")


def _mei_decay_slope(
    analysis: pd.DataFrame,
    pt: str,
) -> tuple[float, float, float, float, int]:
    """Weighted slope of MEI vs shoreline-band centres; returns slope, WMSE, r², sum_sqrt_n, n_bands_used."""

    centers = band_centers_numeric()
    df = analysis[
        analysis["_land_coast_band"].notna() & analysis["nearest_port"].astype(str).eq(pt)
    ].copy()

    xv: list[float] = []
    yv: list[float] = []
    ww: list[float] = []

    for bl in LAND_BAND_LABELS_ORDER:
        mu, _, nn = mean_se(pd.to_numeric(df.loc[df["_land_coast_band"].astype(str).eq(bl), "maritime_pressure_index"], errors="coerce"))
        if not (mu == mu and math.isfinite(mu) and nn > 0):
            continue
        xv.append(float(centers[bl]))
        yv.append(float(mu))
        ww.append(math.sqrt(float(nn)))

    if len(xv) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    coef = np.polyfit(xv, yv, 1, w=ww)
    slope = float(coef[0])
    pred = np.poly1d(coef)(np.asarray(xv))
    resid = np.asarray(yv) - pred
    warr = np.asarray(ww, dtype=float)
    wmse = float(np.sum(warr * resid**2) / max(np.sum(warr), 1e-9))
    r, _p = scipy_stats.pearsonr(xv, yv)
    rsq = float(r**2)
    return slope, wmse, rsq, float(np.sum(warr)), int(len(xv))


def figure_e(analysis: pd.DataFrame) -> None:
    df = analysis[analysis["_land_coast_band"].notna() & analysis["nearest_port"].isin(PORTS_ORDER)].copy()
    print_slice_stats("Figure E cross-port comparison slice", df)

    summary_rows = []
    bar_payloads: dict[str, list[tuple[float, float, float]]] = dict(
        mean_ces=[],
        mean_esi=[],
        no2_ratio=[],
        mei_slope=[],
    )

    x = np.arange(len(PORTS_ORDER), dtype=float)
    width = 0.46

    for pi, pt in enumerate(PORTS_ORDER):
        blk = df[df["nearest_port"].astype(str).eq(pt)]
        ces_m, ces_se, ces_n = mean_se(blk["coastal_exposure_score"])
        esi_m, esi_se, esi_n = mean_se(blk["environmental_stress_index"])

        inner = blk[blk["_land_coast_band"].astype(str).eq(LAND_BAND_LABELS_ORDER[0])]
        outer = blk[blk["_land_coast_band"].astype(str).eq(LAND_BAND_LABELS_ORDER[3])]
        no2_near_m, _, n_near = mean_se(inner["NO2_mean"])
        no2_far_m, _, n_far = mean_se(outer["NO2_mean"])
        persistence = np.nan if (not math.isfinite(no2_far_m)) or abs(no2_near_m) < 1e-6 else float(no2_far_m / no2_near_m)

        slope_decay, wmse, rsq_, wsum_sqrt, n_bands = _mei_decay_slope(analysis, pt)

        summary_rows.extend(
            [
                dict(port=pt, metric="mean_coastal_exposure_score", value=ces_m, se=ces_se, n_rows=int(ces_n)),
                dict(port=pt, metric="mean_environmental_stress_index", value=esi_m, se=esi_se, n_rows=int(esi_n)),
                dict(
                    port=pt,
                    metric="NO2_mean_0_to_3_km",
                    value=no2_near_m,
                    n_rows_band=int(n_near),
                ),
                dict(
                    port=pt,
                    metric="NO2_mean_30_to_50_km",
                    value=no2_far_m,
                    n_rows_band=int(n_far),
                ),
                dict(
                    port=pt,
                    metric="NO2_inland_persistence_ratio",
                    value=persistence,
                    definition="mean_NO2_30_50 / mean_NO2_0_3",
                ),
                dict(
                    port=pt,
                    metric="MEI_decay_slope_per_km_weighted",
                    value=slope_decay,
                    weighted_mse=wmse,
                    pearson_r2=rsq_,
                    sum_sqrt_n_points=wsum_sqrt,
                    n_distance_bands_used=int(n_bands),
                ),
            ],
        )

        bar_payloads["mean_ces"].append((ces_m, ces_se, float(ces_n)))
        bar_payloads["mean_esi"].append((esi_m, esi_se, float(esi_n)))
        bar_payloads["no2_ratio"].append((persistence, 0.0, float(n_near + n_far)))
        bar_payloads["mei_slope"].append((slope_decay, 0.0, float(n_bands)))

    pd.DataFrame(summary_rows).to_csv(OUT / "cross_port_land_exposure_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.45, 8.35))
    axes = axes.ravel()

    titles = [
        ("mean_ces", "Mean coastal exposure score\n(land annulus, port slice)"),
        ("mean_esi", "Mean ESI (z-mean composite)"),
        ("no2_ratio", "NO₂ persistence\n(mean 30–50 km ÷ mean 0–3 km)"),
        ("mei_slope", "MEI landward slope\n(weighted OLS vs band centre, km⁻¹)"),
    ]

    cmap_p = sns.color_palette("colorblind", n_colors=len(PORTS_ORDER))

    for ax, (key, ttl) in zip(axes, titles, strict=True):
        means = [v[0] if v[0] == v[0] else 0.0 for v in bar_payloads[key]]
        errs = [v[1] if v[1] == v[1] else 0.0 for v in bar_payloads[key]]
        ax.bar(
            x,
            means,
            width=width,
            yerr=errs,
            capsize=3.6,
            color=cmap_p,
            edgecolor="#111827",
            linewidth=0.55,
            error_kw=dict(ecolor="#475569", elinewidth=1.02),
        )
        for xi, (mu, se_, nn) in zip(x, bar_payloads[key], strict=False):
            if mu == mu:
                label = f"n={int(nn)}" if key in ("mean_ces", "mean_esi") else f"support={int(nn)}"
                ax.text(
                    float(xi),
                    mu + (se_ if se_ == se_ else 0.0) + 0.02 * max(abs(ax.get_ylim()[1]), 1e-6),
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    color="#475569",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(PORTS_ORDER, rotation=0)
        ax.set_title(ttl, fontsize=10.55, fontweight="semibold")
        ax.set_ylabel("Value (see metric label)")

    fig.suptitle("Cross-port urban/coastal structure (shared annulus mask)", fontsize=12.85, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.11, bottom=0.09, top=0.90, right=0.97, hspace=0.36, wspace=0.30)
    save_fig(fig, "fig_cross_port_land_exposure")


def write_integration_notes() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = """# Land / coastal exposure figures — thesis integration

This note accompanies artefacts in `outputs/thesis_land_exposure/` generated from the balanced weekly ML panel (`processed/features_ml_ready.parquet`) plus archived wind merges.

## Analytic mask (all quantitative panels)

- Coastal-receiver lattice inside the Baltic study window with **nearest NE110 shoreline distance ≤ 50 km** and a **shipping-proximity screen on `distance_to_nearest_high_vessel_density_cell`**.
- Turku/Mariehamn obey the archival **< 30 km** cutoff to offshore high-density hubs. Stockholm’s terrestrial lattice cells routinely sit **>** 30 km from those Baltic-wide seeds despite active coastal commerce, so the script applies a **bounded Stockholm-only extension (< 7200 km)** on the **same archival column**, documented here so supervisors can adjudicate sensitivity vs excluding the inland capital entirely.
- Littoral Baltic centroids flagged as seawater polygons are retained deliberately (Åland/Helsinki archipelagos rarely register as terrestrial in NE110); CSV exports annotate `strict_land_centroid_ne110` for terrestrial-only overlays.
- Stratification (post-mask): **0–3, 3–10, 10–30, 30–50 km** shoreline-distance bins computed as chord distances to sampled coastlines.

## Figure A — `fig_landward_decay.*` + `landward_decay_summary.csv`

**Suggested placement:** Results chapter — subsection on *landward decay of coastal stressors* after introducing MEI / CES / ESI.

**Caption (draft):** Mean NO₂, MEI, coastal exposure score, and ESI plotted against shoreline-distance bins for archival Turku/Mariehamn/Stockholm slices inside the masked coastal lattice. Connecting lines join port-specific means whose vertical bars encode standard errors of weekly observations pooled within bin × port.

**Key finding:** Summarises shoreline-distance structuring of maritime and atmospheric composites with port-specific footprints; contrasts Turku littoral densities against Mariehamn’s narrow annulus versus Stockholm’s extended shipping-distance caveat (integration note above).

## Figure B — `fig_wind_land_exposure.*`, `fig_wind_land_violin.*`, `wind_land_exposure_summary.csv`

**Placement:** Same chapter subsection on *directional atmospheric coupling* (pairs with existing wind-alignment discussion).

**Caption (bars):** Shoreward (hatched) versus non-shoreward weekly regimes within each shoreline bin; bars are means ± SE for each port assignment in the annulus.

**Caption (violins):** Width-normalised densities of weekly observations with 1–99% trimming to limit extreme tails while preserving structural missingness.

**Key finding:** Contrasts distributional mass under onshore transport windows versus other directions, emphasising land-receiver framing.

## Figure C — `fig_land_hotspots.*` + `land_hotspot_cells.csv`

**Placement:** Spatial results / case-study maps for each focal city.

**Caption:** Discrete lattice choropleths (no continuum smoothing between tiles) depicting cells exceeding **within-port P90 temporal medians** for coastal exposure score (top row) and ESI (bottom row); dashed rims flag the Baltic-wide **P88 vessel-density corridor** on the coincident temporal medians.

**Key finding:** Localises compounded exposure pressure adjoining each coastal metropolis while signalling where shipping-heavy lattice edges overlap elevated CES/ESI medians (`strict_land_centroid_ne110` documents terrestrial vs water centroids in the CSV export).

## Figure D — `fig_wind_transport_landward.*`

**Placement:** Methods–Results bridge for *directional transport interpretation* (Turku vs Åland corridors).

**Caption:** Left: Turku corridor with cell-filling based on median NO₂ weekly anomaly; right: Åland maritime corridor with MEI medians. Arrows show median u/v components (no vector interpolation between unsampled locations).

**Key finding:** Couples spatial gradients of stress with mean flow orientation on the actual ML lattice.

## Figure E — `fig_cross_port_land_exposure.*` + `cross_port_land_exposure_summary.csv`

**Placement:** Comparative synthesis paragraph before discussion.

**Caption:** Port-level means (± SE where applicable) for coastal exposure and ESI, a simple NO₂ “persistence” ratio (30–50 km mean divided by 0–3 km mean), and a weighted linear slope of MEI versus band centre as a compact decay descriptor.

**Key finding:** Summarises which archived port assignment exhibits the strongest land/coastal exposure structure under a shared mask.

---

**Radar / spider chart:** Omitted — mixed units (index vs ratio vs slope) would require arbitrary rescaling; the facetted bar layout keeps comparability transparent.
"""
    NOTES_MD.parent.mkdir(parents=True, exist_ok=True)
    NOTES_MD.write_text(body, encoding="utf-8")


def main() -> None:
    thesis_style()
    OUT.mkdir(parents=True, exist_ok=True)
    MID.mkdir(parents=True, exist_ok=True)

    analysis = build_base_panel()

    print_slice_stats("Full balanced panel (post-build)", analysis, extras=[f"coastal_land_shipping_annulus rows={int(analysis['_coastal_land_shipping_annulus'].sum()):,}"])

    figure_a(analysis)
    figure_b(analysis)
    figure_c(analysis)
    figure_d(analysis)
    figure_e(analysis)
    write_integration_notes()
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()

