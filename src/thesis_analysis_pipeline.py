from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, mannwhitneyu, spearmanr, ttest_ind
from sklearn.neighbors import BallTree


CORE_WATER_VARS = ["ndci", "ndti", "ndwi"]
EXTRA_WATER_VARS = ["red_green_ratio", "fai"]
ALL_WATER_VARS = CORE_WATER_VARS + EXTRA_WATER_VARS


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    rename_map = {
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "density_total": "vessel_density",
        "total_vessel_density": "vessel_density",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def load_input_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Only CSV input is supported for this pipeline.")


def haversine_nearest_distance_km(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    # BallTree with haversine expects radians and returns radians.
    src = np.deg2rad(np.c_[source_lat, source_lon])
    tgt = np.deg2rad(np.c_[target_lat, target_lon])
    tree = BallTree(tgt, metric="haversine")
    dist_rad, _ = tree.query(src, k=1)
    return dist_rad[:, 0] * 6371.0088


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    needed = ["latitude", "longitude", "vessel_density"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for exposure features: {missing}")

    out = ensure_numeric(out, ["latitude", "longitude", "vessel_density"])
    out = out.dropna(subset=["latitude", "longitude", "vessel_density"])

    out["density_total_log"] = np.log1p(out["vessel_density"])
    q90 = out["density_total_log"].quantile(0.90)
    q75 = out["density_total_log"].quantile(0.75)
    q25 = out["density_total_log"].quantile(0.25)
    out["sea_lane_flag"] = (out["density_total_log"] >= q90).astype(int)
    out["traffic_group_25_75"] = np.where(
        out["density_total_log"] >= q75,
        "high",
        np.where(out["density_total_log"] <= q25, "low", "mid"),
    )

    lane_points = out[out["sea_lane_flag"] == 1][["latitude", "longitude"]].dropna()
    if len(lane_points) > 0:
        out["distance_to_lane_km"] = haversine_nearest_distance_km(
            out["latitude"].values,
            out["longitude"].values,
            lane_points["latitude"].values,
            lane_points["longitude"].values,
        )
    else:
        out["distance_to_lane_km"] = np.nan

    if "distance_to_port" in out.columns:
        out["distance_to_port"] = pd.to_numeric(out["distance_to_port"], errors="coerce")
    elif {"port_latitude", "port_longitude"}.issubset(out.columns):
        out = ensure_numeric(out, ["port_latitude", "port_longitude"])
        valid = out.dropna(subset=["port_latitude", "port_longitude"])
        out["distance_to_port"] = np.nan
        if len(valid) > 0:
            out.loc[valid.index, "distance_to_port"] = haversine_nearest_distance_km(
                valid["latitude"].values,
                valid["longitude"].values,
                valid["port_latitude"].values,
                valid["port_longitude"].values,
            )

    return out


def mann_whitney_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    lane = df[df["sea_lane_flag"] == 1]
    bg = df[df["sea_lane_flag"] == 0]
    for col in cols:
        if col not in df.columns:
            continue
        a = pd.to_numeric(lane[col], errors="coerce").dropna()
        b = pd.to_numeric(bg[col], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "variable": col,
                "median_lane": float(a.median()),
                "median_background": float(b.median()),
                "u_stat": float(stat),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)


def restricted_lane_tests(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    specs = [
        ("all_points", np.full(len(df), True)),
        (
            "distance_to_port_10_50",
            (df.get("distance_to_port", pd.Series(np.nan, index=df.index)) >= 10)
            & (df.get("distance_to_port", pd.Series(np.nan, index=df.index)) <= 50),
        ),
        (
            "distance_to_port_20_50",
            (df.get("distance_to_port", pd.Series(np.nan, index=df.index)) >= 20)
            & (df.get("distance_to_port", pd.Series(np.nan, index=df.index)) <= 50),
        ),
        ("swir_b11_le_0_05", df.get("b11", pd.Series(np.nan, index=df.index)) <= 0.05),
    ]

    out = []
    for name, mask in specs:
        subset = df[mask.fillna(False)] if isinstance(mask, pd.Series) else df[mask]
        if subset.empty:
            continue
        table = mann_whitney_table(subset, cols)
        if table.empty:
            continue
        table.insert(0, "subset", name)
        table.insert(1, "n", len(subset))
        out.append(table)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def distance_bin_anova(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if "distance_to_port" not in df.columns:
        return pd.DataFrame()
    bins = [0, 5, 10, 20, 50, np.inf]
    labels = ["0-5", "5-10", "10-20", "20-50", ">50"]
    temp = df.copy()
    temp["port_distance_bin"] = pd.cut(temp["distance_to_port"], bins=bins, labels=labels)
    rows = []
    for col in cols:
        if col not in temp.columns:
            continue
        groups = [
            pd.to_numeric(temp.loc[temp["port_distance_bin"] == b, col], errors="coerce").dropna()
            for b in labels
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        stat, p = f_oneway(*groups)
        rows.append({"variable": col, "f_stat": float(stat), "p_value": float(p)})
    return pd.DataFrame(rows)


def high_low_ttest(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if "traffic_group_25_75" not in df.columns:
        return pd.DataFrame()
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        high = pd.to_numeric(df.loc[df["traffic_group_25_75"] == "high", col], errors="coerce").dropna()
        low = pd.to_numeric(df.loc[df["traffic_group_25_75"] == "low", col], errors="coerce").dropna()
        if len(high) < 3 or len(low) < 3:
            continue
        stat, p = ttest_ind(high, low, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "variable": col,
                "high_mean": float(high.mean()),
                "low_mean": float(low.mean()),
                "t_stat": float(stat),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    shipping_vars = ["vessel_density", "density_total_log", "distance_to_port", "distance_to_lane_km"]
    rows = []
    for s in shipping_vars:
        if s not in df.columns:
            continue
        for w in cols:
            if w not in df.columns:
                continue
            tmp = df[[s, w]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(tmp) < 3:
                continue
            r, p = spearmanr(tmp[s], tmp[w])
            rows.append({"shipping_var": s, "water_var": w, "spearman_r": float(r), "p_value": float(p)})
    return pd.DataFrame(rows)


def run(input_csv: Path, project_root: Path) -> None:
    reports = project_root / "outputs" / "reports"
    processed = project_root / "data" / "processed"
    reports.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    df = load_input_table(input_csv)
    df = normalize_columns(df)
    df = build_exposure_features(df)
    df.to_csv(processed / "master_dataset_enriched.csv", index=False)

    vars_present = [v for v in ALL_WATER_VARS if v in df.columns]
    if not vars_present:
        raise ValueError(
            "No thesis water variables found. Expected one or more of: "
            f"{ALL_WATER_VARS}"
        )

    lane_tests = restricted_lane_tests(df, vars_present)
    lane_tests.to_csv(reports / "lane_vs_background_tests.csv", index=False)

    t_tests = high_low_ttest(df, vars_present)
    t_tests.to_csv(reports / "high_vs_low_traffic_ttests.csv", index=False)

    anova = distance_bin_anova(df, vars_present)
    anova.to_csv(reports / "distance_bin_anova.csv", index=False)

    spear = spearman_table(df, vars_present)
    spear.to_csv(reports / "spearman_shipping_water.csv", index=False)

    summary = pd.DataFrame(
        {
            "rows_total": [len(df)],
            "sea_lane_points": [int((df["sea_lane_flag"] == 1).sum())],
            "background_points": [int((df["sea_lane_flag"] == 0).sum())],
            "q90_threshold": [float(df["density_total_log"].quantile(0.90))],
        }
    )
    summary.to_csv(reports / "thesis_pipeline_summary.csv", index=False)
    print("[DONE] Thesis-aligned outputs written to outputs/reports")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis-aligned lane/background analysis pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/master_dataset.csv",
        help="Path to merged point-level CSV with Sentinel + vessel variables",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    run(root / args.input, root)


if __name__ == "__main__":
    main()
