from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kurtosis, mannwhitneyu, shapiro, skew, ttest_ind
from shapely.geometry import Point


LAT_CANDIDATES = ["latitude", "lat", "y", "y_coord"]
LON_CANDIDATES = ["longitude", "lon", "lng", "x", "x_coord"]
DATE_HINTS = ["date", "time", "timestamp", "datetime"]
WATER_VARS = ["ndci", "ndti", "ndwi"]


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def infer_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    cols = [c.lower() for c in df.columns]
    lat_col = next((c for c in LAT_CANDIDATES if c in cols), None)
    lon_col = next((c for c in LON_CANDIDATES if c in cols), None)
    return lat_col, lon_col


def to_geodataframe(df: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
    if isinstance(df, gpd.GeoDataFrame):
        gdf = df.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf

    lat_col, lon_col = infer_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        return None

    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col])
    if tmp.empty:
        return None

    geom = [Point(xy) for xy in zip(tmp[lon_col], tmp[lat_col])]
    return gpd.GeoDataFrame(tmp, geometry=geom, crs="EPSG:4326")


def dataset_key_from_name(name: str) -> str:
    lower = name.lower()
    if "sentinel" in lower:
        return "sentinel"
    if "vessel" in lower or "emodnet" in lower or "density" in lower:
        return "vessel"
    if "port" in lower or "harbor" in lower or "harbour" in lower:
        return "ports"
    if "helcom" in lower:
        return "helcom"
    return lower


def print_df_info(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())


def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    expected = [
        data_dir / "sentinel2.csv",
        data_dir / "emodnet_vessel_density.csv",
        data_dir / "helcom.csv",
    ]

    for path in expected:
        if not path.exists():
            print(f"[INFO] Expected file missing (skipped): {path.name}")

    candidate_paths = []
    candidate_paths.extend(data_dir.glob("*.csv"))
    candidate_paths.extend(data_dir.glob("*.geojson"))
    candidate_paths = sorted(candidate_paths)

    if not candidate_paths:
        print(f"[WARN] No CSV/GeoJSON files found in {data_dir}")
        return datasets

    for path in candidate_paths:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = gpd.read_file(path)
            key = dataset_key_from_name(path.stem)
            datasets[key] = df
            print(f"[OK] Loaded {path.name} as '{key}'")
            print_df_info(key, df)
        except Exception as exc:
            print(f"[WARN] Failed to load {path.name}: {exc}")

    return datasets


def clean_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cleaned = standardize_column_names(df)

    missing_pct = cleaned.isna().mean().mul(100).sort_values(ascending=False)
    print(f"\nMissing values % for '{name}':")
    print(missing_pct.to_string())

    for col in cleaned.columns:
        lower = col.lower()
        if any(hint in lower for hint in DATE_HINTS):
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")
        if lower in LAT_CANDIDATES + LON_CANDIDATES:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    after = len(cleaned)
    if before != after:
        print(f"[INFO] Removed {before - after} duplicate rows from '{name}'")

    return cleaned


def spatial_nearest_merge(
    left: pd.DataFrame, right: pd.DataFrame, right_suffix: str
) -> pd.DataFrame:
    left_gdf = to_geodataframe(left)
    right_gdf = to_geodataframe(right)
    if left_gdf is None or right_gdf is None:
        print("[WARN] Spatial merge skipped: missing usable coordinates/geometry.")
        return left.copy()

    left_m = left_gdf.to_crs("EPSG:3857")
    right_m = right_gdf.to_crs("EPSG:3857")
    merged = gpd.sjoin_nearest(
        left_m,
        right_m,
        how="left",
        distance_col=f"nearest_distance_{right_suffix}_m",
        lsuffix="left",
        rsuffix=right_suffix,
    )
    merged = merged.to_crs("EPSG:4326")
    merged = merged.drop(columns=["index_right"], errors="ignore")
    return merged


def add_distance_to_port(df: pd.DataFrame, ports_df: pd.DataFrame) -> pd.DataFrame:
    points = to_geodataframe(df)
    ports = to_geodataframe(ports_df)
    if points is None or ports is None:
        print("[INFO] Port distance not added (ports/coordinates unavailable).")
        return df

    points_m = points.to_crs("EPSG:3857")
    ports_m = ports.to_crs("EPSG:3857")
    joined = gpd.sjoin_nearest(
        points_m,
        ports_m,
        how="left",
        distance_col="distance_to_port_m",
        lsuffix="obs",
        rsuffix="port",
    )
    joined["distance_to_port"] = joined["distance_to_port_m"] / 1000.0
    joined = joined.to_crs("EPSG:4326").drop(columns=["index_right"], errors="ignore")
    return joined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vessel_col = "vessel_density" if "vessel_density" in out.columns else None
    if vessel_col is None:
        # Fallback for renamed columns after joins
        for col in out.columns:
            if "vessel" in col and "density" in col:
                vessel_col = col
                break

    if vessel_col is not None:
        out["traffic_category"] = pd.qcut(
            out[vessel_col], q=3, labels=["low", "medium", "high"], duplicates="drop"
        )
        p90 = out[vessel_col].quantile(0.90)
        out["sea_lane_flag"] = (out[vessel_col] >= p90).astype(int)
    else:
        print("[INFO] traffic_category and sea_lane_flag skipped (no vessel density).")

    return out


def run_distribution_analysis(df: pd.DataFrame, plot_dir: Path) -> pd.DataFrame:
    records = []
    for col in WATER_VARS + ["vessel_density"]:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        records.append(
            {
                "variable": col,
                "skewness": float(skew(series, nan_policy="omit")),
                "kurtosis": float(kurtosis(series, nan_policy="omit")),
            }
        )

        plt.figure(figsize=(8, 5))
        sns.histplot(series, kde=True)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"dist_{col}.png", dpi=150)
        plt.close()

    return pd.DataFrame(records)


def run_normality_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in WATER_VARS + ["vessel_density"]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) < 3:
            continue
        sample = x.sample(n=min(len(x), 5000), random_state=42)
        stat, pval = shapiro(sample)
        rows.append({"variable": col, "shapiro_stat": stat, "shapiro_p": pval})
    return pd.DataFrame(rows)


def run_correlation_analysis(df: pd.DataFrame, plot_dir: Path) -> dict[str, pd.DataFrame]:
    cols = [c for c in WATER_VARS + ["vessel_density", "distance_to_port"] if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    for name, corr in [("pearson", pearson), ("spearman", spearman)]:
        if corr.empty:
            continue
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title(f"{name.title()} Correlation Matrix")
        plt.tight_layout()
        plt.savefig(plot_dir / f"corr_{name}.png", dpi=150)
        plt.close()

    return {"pearson": pearson, "spearman": spearman}


def run_group_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in WATER_VARS:
        if var not in df.columns:
            continue
        x = pd.to_numeric(df[var], errors="coerce")

        if "traffic_category" in df.columns:
            high = x[df["traffic_category"] == "high"].dropna()
            low = x[df["traffic_category"] == "low"].dropna()
            if len(high) >= 3 and len(low) >= 3:
                # Use normality check to choose test
                h_p = shapiro(high.sample(min(len(high), 5000), random_state=42))[1]
                l_p = shapiro(low.sample(min(len(low), 5000), random_state=42))[1]
                if h_p > 0.05 and l_p > 0.05:
                    stat, pval = ttest_ind(high, low, equal_var=False, nan_policy="omit")
                    test_name = "t-test"
                else:
                    stat, pval = mannwhitneyu(high, low, alternative="two-sided")
                    test_name = "mann-whitney"
                rows.append(
                    {
                        "comparison": "high_vs_low_traffic",
                        "variable": var,
                        "test": test_name,
                        "stat": float(stat),
                        "p_value": float(pval),
                    }
                )

        if "sea_lane_flag" in df.columns:
            lane = x[df["sea_lane_flag"] == 1].dropna()
            bg = x[df["sea_lane_flag"] == 0].dropna()
            if len(lane) >= 3 and len(bg) >= 3:
                stat, pval = mannwhitneyu(lane, bg, alternative="two-sided")
                rows.append(
                    {
                        "comparison": "sea_lane_vs_background",
                        "variable": var,
                        "test": "mann-whitney",
                        "stat": float(stat),
                        "p_value": float(pval),
                    }
                )

        if "distance_to_port" in df.columns:
            bins = [0, 5, 10, 20, 50, np.inf]
            labels = ["0-5", "5-10", "10-20", "20-50", ">50"]
            temp = df.copy()
            temp["distance_bin"] = pd.cut(temp["distance_to_port"], bins=bins, labels=labels)
            groups = [x[temp["distance_bin"] == b].dropna() for b in labels]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) >= 2:
                stat, pval = f_oneway(*groups)
                rows.append(
                    {
                        "comparison": "distance_to_port_bins",
                        "variable": var,
                        "test": "anova",
                        "stat": float(stat),
                        "p_value": float(pval),
                    }
                )

    return pd.DataFrame(rows)


def run_spatial_visualizations(df: pd.DataFrame, plot_dir: Path) -> None:
    lat_col, lon_col = infer_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        return

    for color_col in ["ndti", "vessel_density"]:
        if color_col not in df.columns:
            continue
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            pd.to_numeric(df[lon_col], errors="coerce"),
            pd.to_numeric(df[lat_col], errors="coerce"),
            c=pd.to_numeric(df[color_col], errors="coerce"),
            cmap="viridis",
            s=15,
            alpha=0.8,
        )
        plt.colorbar(scatter, label=color_col)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Spatial Distribution colored by {color_col}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"spatial_{color_col}.png", dpi=150)
        plt.close()


def run_outlier_detection(df: pd.DataFrame, plot_dir: Path) -> pd.DataFrame:
    rows = []
    for col in WATER_VARS + ["vessel_density"]:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if x.empty:
            continue
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((x < lower) | (x > upper)).sum())
        rows.append(
            {
                "variable": col,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": outlier_count,
            }
        )

        plt.figure(figsize=(7, 4))
        sns.boxplot(x=x)
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"boxplot_{col}.png", dpi=150)
        plt.close()

    return pd.DataFrame(rows)


def save_outputs(
    merged_df: pd.DataFrame,
    summary_tables: dict[str, pd.DataFrame],
    processed_dir: Path,
    reports_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned merged data
    merged_df.to_csv(processed_dir / "cleaned_merged_dataset.csv", index=False)
    if isinstance(merged_df, gpd.GeoDataFrame):
        merged_df.to_file(processed_dir / "cleaned_merged_dataset.geojson", driver="GeoJSON")

    # Save summary reports
    for name, table in summary_tables.items():
        table.to_csv(reports_dir / f"{name}.csv", index=False)

    # Compact JSON summary for quick inspection
    json_summary = {
        name: table.to_dict(orient="records") for name, table in summary_tables.items()
    }
    with open(reports_dir / "summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, default=str)


def run_pipeline(project_root: Path) -> None:
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    plots_dir = project_root / "outputs" / "plots"
    reports_dir = project_root / "outputs" / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    datasets = load_datasets(data_dir)
    if not datasets:
        print("[STOP] No datasets available. Add files to /data and rerun.")
        return

    cleaned = {name: clean_dataset(df, name) for name, df in datasets.items()}

    sentinel = cleaned.get("sentinel")
    vessel = cleaned.get("vessel")
    ports = cleaned.get("ports")

    if sentinel is not None and vessel is not None:
        merged = spatial_nearest_merge(sentinel, vessel, right_suffix="vessel")
        print("[OK] Sentinel-2 and vessel density merged via nearest-neighbor spatial join.")
    else:
        # Fallback: use the first available cleaned dataset
        fallback_key = next(iter(cleaned))
        merged = cleaned[fallback_key]
        print(f"[INFO] Merge fallback used: '{fallback_key}' (missing sentinel/vessel pair).")

    if ports is not None:
        merged = add_distance_to_port(merged, ports)

    merged = engineer_features(merged)

    distribution_stats = run_distribution_analysis(merged, plots_dir)
    normality_stats = run_normality_tests(merged)
    corr = run_correlation_analysis(merged, plots_dir)
    group_tests = run_group_comparisons(merged)
    run_spatial_visualizations(merged, plots_dir)
    outlier_stats = run_outlier_detection(merged, plots_dir)

    save_outputs(
        merged_df=merged,
        summary_tables={
            "distribution_stats": distribution_stats,
            "normality_tests": normality_stats,
            "pearson_correlation": corr["pearson"].reset_index().rename(
                columns={"index": "variable"}
            ),
            "spearman_correlation": corr["spearman"].reset_index().rename(
                columns={"index": "variable"}
            ),
            "group_comparisons": group_tests,
            "outlier_stats": outlier_stats,
        },
        processed_dir=processed_dir,
        reports_dir=reports_dir,
    )
    print("[DONE] Pipeline completed successfully.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_pipeline(root)
