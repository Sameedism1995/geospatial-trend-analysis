"""
Exploratory visualizations for data/modeling_dataset.parquet (pre-modeling).

Does not globally drop rows; NaNs are masked or omitted only per plot where needed.

Saves PNGs and opens them in the OS default viewer (e.g. Preview on macOS).
Optional: --matplotlib for Tk windows (often broken in IDE terminals).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Interactive backend before pyplot (required for plot viewer windows)
import matplotlib

def _configure_interactive_backend() -> str:
    if os.environ.get("MPLBACKEND"):
        return str(matplotlib.get_backend())
    for name in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(name, force=True)
            return name
        except Exception:  # noqa: BLE001
            continue
    return matplotlib.get_backend()


_BACKEND = _configure_interactive_backend()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Consistent style
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
    }
)
sns.set_theme(style="whitegrid", context="notebook")


def _should_show_interactive(no_show: bool) -> bool:
    if no_show:
        return False
    if os.environ.get("CI"):
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    return df


def plot_vessel_density_map(df: pd.DataFrame, out: Path) -> Figure:
    """One marker per grid: mean vessel_density_t (stable spatial pattern)."""
    g = (
        df.groupby("grid_cell_id", sort=False)
        .agg(
            grid_centroid_lat=("grid_centroid_lat", "mean"),
            grid_centroid_lon=("grid_centroid_lon", "mean"),
            vessel_density_t=("vessel_density_t", "mean"),
        )
        .reset_index()
    )
    lon = g["grid_centroid_lon"].to_numpy()
    lat = g["grid_centroid_lat"].to_numpy()
    v = g["vessel_density_t"].to_numpy()
    mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(v)
    lon, lat, v = lon[mask], lat[mask], v[mask]

    fig, ax = plt.subplots(figsize=(8, 7), num="Vessel density map")
    v_pos = v[np.isfinite(v) & (v > 0)]
    if v_pos.size and v_pos.max() / max(v_pos.min(), 1e-30) > 50:
        norm = mcolors.LogNorm(vmin=max(v_pos.min(), 1e-12), vmax=v_pos.max())
    else:
        norm = mcolors.Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v))
    sc = ax.scatter(
        lon,
        lat,
        c=v,
        cmap="viridis",
        s=22,
        alpha=0.85,
        edgecolors="none",
        norm=norm,
    )
    plt.colorbar(sc, ax=ax, label="Mean vessel_density_t (annual raster sum at centroid)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Spatial pattern of vessel traffic (mean vessel_density_t per grid)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    _set_window_title(fig, "Vessel density map")
    return fig


def _grid_rank_by_vessel(df: pd.DataFrame) -> pd.Series:
    """Median vessel_density_t per grid (robust to week / missingness)."""
    return df.groupby("grid_cell_id")["vessel_density_t"].median()


def plot_ndti_timeseries(df: pd.DataFrame, out: Path) -> Figure:
    rank = _grid_rank_by_vessel(df).sort_values(ascending=False)
    valid = rank.dropna()
    if len(valid) < 10:
        top5 = valid.head(min(5, len(valid))).index.tolist()
        bot5 = valid.tail(min(5, len(valid))).index.tolist()
    else:
        top5 = valid.head(5).index.tolist()
        bot5 = valid.tail(5).index.tolist()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, num="NDTI time series")
    ndti_col = "sentinel_ndti_mean_t"

    for ax, grids, title in zip(
        axes,
        [top5, bot5],
        ["Top 5 grids by median vessel_density_t", "Bottom 5 grids by median vessel_density_t"],
        strict=True,
    ):
        for gid in grids:
            sub = df.loc[df["grid_cell_id"] == gid].sort_values("week_start_utc")
            t = sub["week_start_utc"]
            y = pd.to_numeric(sub[ndti_col], errors="coerce")
            ax.plot(t, y, marker="o", ms=2, lw=1, alpha=0.85, label=str(gid)[:18])

        ax.set_ylabel("sentinel_ndti_mean_t")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7, ncol=1)

        if grids:
            sub0 = df.loc[df["grid_cell_id"] == grids[0]].sort_values("week_start_utc")
            ax2 = ax.twinx()
            ax2.plot(
                sub0["week_start_utc"],
                pd.to_numeric(sub0["vessel_density_t"], errors="coerce"),
                color="gray",
                ls="--",
                lw=1,
                alpha=0.6,
                label="vessel_density_t (one grid, dashed)",
            )
            ax2.set_ylabel("vessel_density_t (ref.)", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

    axes[-1].set_xlabel("week_start_utc")
    fig.suptitle("NDTI time series: high- vs low-traffic grids", y=1.02)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    _set_window_title(fig, "NDTI time series")
    return fig


def plot_ndti_distribution(df: pd.DataFrame, out: Path) -> Figure:
    v = pd.to_numeric(df["vessel_density_t"], errors="coerce")
    ndti = pd.to_numeric(df["sentinel_ndti_mean_t"], errors="coerce")
    mask = v.notna() & ndti.notna()
    v, ndti = v[mask], ndti[mask]

    p90 = v.quantile(0.90)
    p50 = v.quantile(0.50)
    sea = (v >= p90) & np.isfinite(ndti)
    back = (v <= p50) & np.isfinite(ndti)

    fig, ax = plt.subplots(figsize=(8, 5), num="NDTI distribution")
    if sea.sum() >= 2:
        sns.kdeplot(ndti[sea], ax=ax, fill=True, alpha=0.35, label=f"Sea lane (top 10%, n={int(sea.sum())})", color="darkred", warn_singular=False)
    else:
        ax.hist(ndti[sea], bins=20, density=True, alpha=0.4, color="darkred", label=f"Sea lane (n={int(sea.sum())})")
    if back.sum() >= 2:
        sns.kdeplot(ndti[back], ax=ax, fill=True, alpha=0.35, label=f"Background (bottom 50%, n={int(back.sum())})", color="steelblue", warn_singular=False)
    else:
        ax.hist(ndti[back], bins=20, density=True, alpha=0.4, color="steelblue", label=f"Background (n={int(back.sum())})")
    ax.set_xlabel("sentinel_ndti_mean_t")
    ax.set_ylabel("Density")
    ax.set_title("NDTI distribution: high vessel density vs background (row-level percentiles)")
    ax.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    _set_window_title(fig, "NDTI distribution")
    return fig


def plot_ndti_heatmap(df: pd.DataFrame, out: Path) -> Figure:
    """Rows = grids sorted by median vessel density; columns = weeks; NaN masked."""
    med_v = _grid_rank_by_vessel(df).sort_values(ascending=False)
    row_order = med_v.index.tolist()
    weeks = sorted(df["week_start_utc"].dropna().unique())

    ndti_col = "sentinel_ndti_mean_t"
    pivot = df.pivot_table(
        index="grid_cell_id",
        columns="week_start_utc",
        values=ndti_col,
        aggfunc="first",
    )
    pivot = pivot.reindex(index=[g for g in row_order if g in pivot.index])
    pivot = pivot.reindex(columns=weeks)

    arr = pivot.to_numpy(dtype=float)
    fig_h = max(6, min(24, 0.03 * arr.shape[0]))
    fig_w = max(10, min(22, 0.12 * arr.shape[1]))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), num="NDTI heatmap")
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(
        masked,
        aspect="auto",
        cmap="RdYlBu_r",
        interpolation="nearest",
        rasterized=True,
    )
    plt.colorbar(im, ax=ax, label="sentinel_ndti_mean_t", shrink=0.6)
    n_x = min(20, len(weeks))
    tick_idx = np.linspace(0, len(weeks) - 1, n_x, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(weeks[i])[:10] for i in tick_idx], rotation=45, ha="right", fontsize=7)
    n_y = min(25, arr.shape[0])
    ytick_idx = np.linspace(0, arr.shape[0] - 1, n_y, dtype=int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([pivot.index[i][:16] for i in ytick_idx], fontsize=6)
    ax.set_xlabel("week_start_utc")
    ax.set_ylabel("grid_cell_id (sorted by median vessel_density_t, high → low)")
    ax.set_title("NDTI heatmap (missing values shown as gap in colormap)")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    _set_window_title(fig, "NDTI heatmap")
    return fig


def _set_window_title(fig: Figure, title: str) -> None:
    try:
        mgr = fig.canvas.manager
        if mgr is not None and hasattr(mgr, "set_window_title"):
            mgr.set_window_title(title)
    except Exception:  # noqa: BLE001
        pass


def open_pngs_in_system_viewer(out_dir: Path) -> bool:
    """
    Open saved PNGs with the OS default app (Preview on macOS).
    Reliable when embedded terminals (e.g. Cursor) cannot show Tk windows.
    """
    pngs = sorted(out_dir.glob("*.png"))
    if not pngs:
        return False
    paths = [str(p.resolve()) for p in pngs]
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", *paths], check=False)
            return True
        if sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", paths[0]], check=False)
            return True
        if sys.platform == "win32":
            os.startfile(paths[0])  # noqa: S606
            return True
    except OSError:
        return False
    return False


def open_output_folder(out_dir: Path) -> None:
    """Reveal output directory in Finder / Explorer / file manager."""
    p = out_dir.resolve()
    if not p.is_dir():
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(p)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(p))  # noqa: S606
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Modeling dataset EDA plots.")
    parser.add_argument("--input", type=Path, default=Path("data/modeling_dataset.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/visualizations"))
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip matplotlib plt.show() only; OS still opens PNGs unless --no-open.",
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        help="Try native matplotlib/Tk windows (often fails in IDE terminals; PNGs open by default instead).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open PNGs or folder with the system (for CI).",
    )
    parser.add_argument(
        "--open-folder-only",
        action="store_true",
        help="Only open the output folder in Finder/Explorer, not each PNG.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    inp = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir

    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    df = load_dataset(inp)

    plot_vessel_density_map(df, out_dir / "vessel_density_map.png")
    plot_ndti_timeseries(df, out_dir / "ndti_timeseries.png")
    plot_ndti_distribution(df, out_dir / "ndti_distribution_comparison.png")
    plot_ndti_heatmap(df, out_dir / "ndti_heatmap.png")

    print(f"Saved plots under {out_dir}")
    print(f"Matplotlib backend: {_BACKEND}")

    # Open PNGs via OS (reliable from Cursor; Tk popups usually are not)
    if not args.no_open and not os.environ.get("CI"):
        if args.open_folder_only:
            open_output_folder(out_dir)
            print(f"Opened folder: {out_dir.resolve()}")
        elif open_pngs_in_system_viewer(out_dir):
            print("Opened PNGs in the default viewer (e.g. Preview on macOS).")
        else:
            open_output_folder(out_dir)
            print(f"Opened folder: {out_dir.resolve()}")
    elif args.no_open:
        print("Skipped opening files (--no-open).")

    if args.matplotlib and _should_show_interactive(args.no_show):
        print("Close matplotlib windows to continue…")
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()
