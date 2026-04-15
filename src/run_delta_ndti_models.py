"""
Time-aware modeling and evaluation for:
  - Primary: delta_ndti (NDTI(t+1) − NDTI(t))
  - Secondary: ndti_next (NDTI at t+1), same features / split / models for comparison

No random row shuffle; split by calendar week. Features use ≤ t only; targets use t+1 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# Excluded from feature matrix X (targets / metadata)
META_COLS = {"grid_cell_id", "week_start_utc", "delta_ndti", "has_valid_delta_ndti", "ndti_next_target"}


def load_modeling_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.loc[df["has_valid_delta_ndti"] == True].copy()  # noqa: E712
    df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    df = df.sort_values(["grid_cell_id", "week_start_utc"]).reset_index(drop=True)
    # NDTI(t+1); same row set as delta task when has_valid_delta_ndti (both endpoints non-null)
    df["ndti_next_target"] = df.groupby("grid_cell_id", sort=False)["sentinel_ndti_mean_t"].shift(-1)
    both = df["sentinel_ndti_mean_t"].notna() & df["ndti_next_target"].notna()
    df = df.loc[both].copy()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == "boolean" or str(X[c].dtype) == "bool":
            X[c] = X[c].astype(np.float64)
        elif str(X[c].dtype).startswith("Int"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def time_aware_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Any], list[Any]]:
    """First train_frac of distinct weeks -> train; remainder -> test."""
    weeks = sorted(df["week_start_utc"].dropna().unique())
    n = len(weeks)
    if n < 3:
        raise ValueError("Need at least 3 distinct weeks for time split.")
    n_train_w = max(1, int(np.floor(n * train_frac)))
    n_train_w = min(n_train_w, n - 1)
    train_weeks = weeks[:n_train_w]
    test_weeks = weeks[n_train_w:]
    tw_set = set(pd.Timestamp(w) for w in train_weeks)
    te_set = set(pd.Timestamp(w) for w in test_weeks)
    train_df = df[df["week_start_utc"].isin(tw_set)].copy()
    test_df = df[df["week_start_utc"].isin(te_set)].copy()
    return train_df, test_df, [str(w) for w in train_weeks], [str(w) for w in test_weeks]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "rmse": float(root_mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "n": int(mask.sum()),
    }


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    """Median imputation + scaling + Ridge (linear baseline; NaNs not supported natively)."""
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)
    return pred_train, pred_test, pipe


def fit_hgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    max_iter: int = 200,
    max_depth: int = 8,
    learning_rate: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, HistGradientBoostingRegressor]:
    """HistGradientBoosting handles missing values in numeric inputs natively."""
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return pred_train, pred_test, model


def permutation_importance_hgb(
    model: HistGradientBoostingRegressor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    names: list[str],
    *,
    max_samples: int = 2500,
    n_repeats: int = 8,
) -> list[dict[str, Any]]:
    """HGB in recent sklearn has no feature_importances_; use permutation importance."""
    n = min(max_samples, len(X_train))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), size=n, replace=False)
    Xs = X_train.iloc[idx]
    ys = y_train[idx]
    r = permutation_importance(
        model,
        Xs,
        ys,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    order = np.argsort(-r.importances_mean)
    return [
        {"feature": names[i], "importance_mean": float(r.importances_mean[i]), "importance_std": float(r.importances_std[i])}
        for i in order
    ]


def summarize_group_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    by: str,
) -> list[dict[str, Any]]:
    """RMSE / MAE / n per group (e.g. week or grid)."""
    tmp = df[[by]].copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    tmp["sq_err"] = (tmp["y_true"] - tmp["y_pred"]) ** 2
    tmp["abs_err"] = (tmp["y_true"] - tmp["y_pred"]).abs()
    rows = []
    for key, g in tmp.groupby(by, sort=False):
        rows.append(
            {
                by: str(key),
                "n": len(g),
                "rmse": float(np.sqrt(g["sq_err"].mean())),
                "mae": float(g["abs_err"].mean()),
            }
        )
    return rows


def expanding_window_validation(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: list[str],
    *,
    min_train_weeks: int,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    """
    Within training timeline only: expand train weeks, hold out next week as test.
    Refits HGB only (fast); reports RMSE on held-out week.
    """
    weeks = sorted(train_df["week_start_utc"].unique())
    if len(weeks) < min_train_weeks + n_folds:
        return []
    folds: list[dict[str, Any]] = []
    # Anchor: start after min_train_weeks consecutive weeks from start
    start_idx = min_train_weeks - 1
    for k in range(n_folds):
        split_at = start_idx + k
        if split_at + 1 >= len(weeks):
            break
        tr_w = weeks[: split_at + 1]
        te_w = weeks[split_at + 1]
        mask_tr = train_df["week_start_utc"].isin(set(tr_w))
        mask_te = train_df["week_start_utc"] == te_w
        if mask_te.sum() == 0:
            continue
        X_tr = X_train.loc[mask_tr.values]
        y_tr = y_train[mask_tr.values]
        X_te = X_train.loc[mask_te.values]
        y_te = y_train[mask_te.values]
        model = HistGradientBoostingRegressor(
            max_iter=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        folds.append(
            {
                "fold": k + 1,
                "train_end_week": str(tr_w[-1]),
                "test_week": str(te_w),
                "n_train": int(mask_tr.sum()),
                "n_test": int(mask_te.sum()),
                "rmse": float(root_mean_squared_error(y_te, pred)),
                "mae": float(mean_absolute_error(y_te, pred)),
            }
        )
    return folds


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
    label: str,
    *,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
) -> None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true[mask], y_pred[mask], alpha=0.35, s=8, label=label)
    lim = [
        float(np.nanmin([y_true[mask].min(), y_pred[mask].min()])),
        float(np.nanmax([y_true[mask].max(), y_pred[mask].max()])),
    ]
    pad = (lim[1] - lim[0]) * 0.05 + 1e-9
    lim = [lim[0] - pad, lim[1] + pad]
    plt.plot(lim, lim, "k--", lw=1, label="y = x")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
) -> None:
    res = y_true - y_pred
    mask = np.isfinite(res)
    plt.figure(figsize=(6, 4))
    plt.hist(res[mask], bins=40, color="steelblue", edgecolor="white", alpha=0.9)
    plt.xlabel("Residual (actual − predicted)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_importance(top: list[dict[str, Any]], path: Path, title: str, n: int = 20) -> None:
    top = top[:n]
    names = [x["feature"] for x in top][::-1]
    vals = [x.get("importance_mean", x.get("importance", 0.0)) for x in top][::-1]
    plt.figure(figsize=(8, max(4, n * 0.2)))
    plt.barh(range(len(names)), vals, color="darkgreen", alpha=0.75)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_task_comparison_table(
    path: Path,
    *,
    m_ridge_d: dict[str, float],
    m_hgb_d: dict[str, float],
    m_ridge_n: dict[str, float],
    m_hgb_n: dict[str, float],
) -> None:
    """Thesis-friendly table: test metrics for both tasks × both models."""
    rows: list[dict[str, Any]] = []
    for task, mr, mh in (
        ("delta_ndti", m_ridge_d, m_hgb_d),
        ("ndti_next", m_ridge_n, m_hgb_n),
    ):
        for model_name, m in (("ridge_median_impute_scaled", mr), ("hist_gradient_boosting", mh)):
            rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "split": "test",
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "n": m["n"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_task_comparison_figure(
    path: Path,
    *,
    m_ridge_d: dict[str, float],
    m_hgb_d: dict[str, float],
    m_ridge_n: dict[str, float],
    m_hgb_n: dict[str, float],
) -> None:
    """Side-by-side test R² and test RMSE: ΔNDTI vs NDTI(t+1)."""
    labels = ["ΔNDTI\nRidge", "ΔNDTI\nHGB", "NDTI(t+1)\nRidge", "NDTI(t+1)\nHGB"]
    r2s = [m_ridge_d["r2"], m_hgb_d["r2"], m_ridge_n["r2"], m_hgb_n["r2"]]
    rmses = [m_ridge_d["rmse"], m_hgb_d["rmse"], m_ridge_n["rmse"], m_hgb_n["rmse"]]
    x = np.arange(len(labels))
    colors = ["#c44e52", "#c44e52", "#4c72b0", "#4c72b0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(x, r2s, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("R² (test)")
    ax1.set_title("Test coefficient of determination")
    ax1.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, rmses, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("RMSE (test)")
    ax2.set_title("Test root mean squared error")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Task comparison (identical time-based split & features): change vs level",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _importance_groups(imp: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "vessel_density_sum": float(
            sum(x.get("importance_mean", 0.0) for x in imp if x["feature"].startswith("vessel_density"))
        ),
        "lagged_spectral_sum": float(
            sum(
                x.get("importance_mean", 0.0)
                for x in imp
                if x["feature"].startswith("sentinel_") and "observation" not in x["feature"]
            )
        ),
        "sentinel_observation_count_t": float(
            sum(x.get("importance_mean", 0.0) for x in imp if "observation_count" in x["feature"])
        ),
        "temporal_week_encoding_sum": float(
            sum(x.get("importance_mean", 0.0) for x in imp if x["feature"] in ("week_of_year", "week_sin", "week_cos"))
        ),
        "static_geo_sum": float(
            sum(
                x.get("importance_mean", 0.0)
                for x in imp
                if x["feature"] in ("grid_res_deg", "grid_centroid_lat", "grid_centroid_lon")
            )
        ),
        "has_flags_sum": float(
            sum(x.get("importance_mean", 0.0) for x in imp if x["feature"].startswith("has_"))
        ),
    }


def run(
    *,
    input_path: Path,
    results_json: Path,
    predictions_path: Path,
    plots_dir: Path,
    train_frac: float,
) -> None:
    df = load_modeling_data(input_path)
    feature_cols = feature_columns(df)
    X = prepare_X(df, feature_cols)
    y_delta = df["delta_ndti"].to_numpy(dtype=float)
    y_ndti_next = df["ndti_next_target"].to_numpy(dtype=float)

    train_df, test_df, train_weeks, test_weeks = time_aware_split(df, train_frac=train_frac)
    X_train = prepare_X(train_df, feature_cols)
    X_test = prepare_X(test_df, feature_cols)
    y_train_d = train_df["delta_ndti"].to_numpy(dtype=float)
    y_test_d = test_df["delta_ndti"].to_numpy(dtype=float)
    y_train_n = train_df["ndti_next_target"].to_numpy(dtype=float)
    y_test_n = test_df["ndti_next_target"].to_numpy(dtype=float)

    pred_tr_ridge, pred_te_ridge, ridge_pipe = fit_ridge(X_train, y_train_d, X_test, y_test_d)
    pred_tr_hgb, pred_te_hgb, hgb = fit_hgb(X_train, y_train_d, X_test, y_test_d)

    imp = permutation_importance_hgb(hgb, X_train, y_train_d, feature_cols)

    pred_tr_ridge_n, pred_te_ridge_n, ridge_pipe_n = fit_ridge(X_train, y_train_n, X_test, y_test_n)
    pred_tr_hgb_n, pred_te_hgb_n, hgb_n = fit_hgb(X_train, y_train_n, X_test, y_test_n)

    imp_n = permutation_importance_hgb(hgb_n, X_train, y_train_n, feature_cols)

    pred_all_ridge = ridge_pipe.predict(X)
    pred_all_hgb = hgb.predict(X)
    pred_all_ridge_n = ridge_pipe_n.predict(X)
    pred_all_hgb_n = hgb_n.predict(X)

    split_train = df["week_start_utc"].isin(set(train_df["week_start_utc"]))
    split_test = df["week_start_utc"].isin(set(test_df["week_start_utc"]))
    split_label = np.where(split_train, "train", np.where(split_test, "test", "unknown"))

    pred_df = pd.DataFrame(
        {
            "grid_cell_id": df["grid_cell_id"].values,
            "week_start_utc": df["week_start_utc"].values,
            "y_true_delta_ndti": y_delta,
            "y_true_ndti_next": y_ndti_next,
            "split": split_label,
            "pred_ridge_delta_ndti": pred_all_ridge,
            "pred_hist_gradient_boosting_delta_ndti": pred_all_hgb,
            "pred_ridge_ndti_next": pred_all_ridge_n,
            "pred_hist_gradient_boosting_ndti_next": pred_all_hgb_n,
            # Backward-compatible aliases (same as delta_ndti columns)
            "pred_ridge": pred_all_ridge,
            "pred_hist_gradient_boosting": pred_all_hgb,
        }
    )

    results: dict[str, Any] = {
        "objective": "Predict delta_ndti and ndti_next (NDTI at t+1); time-aware train/test split; same features",
        "n_rows_valid_target": int(len(df)),
        "ndti_next_target_note": (
            "ndti_next_target = sentinel_ndti_mean_t at week t+1 (groupby grid shift -1). "
            "Rows require non-null NDTI at t and t+1 (aligned with delta_ndti support)."
        ),
        "split": {
            "method": "time_aware_by_week",
            "train_frac": train_frac,
            "n_distinct_weeks_total": int(df["week_start_utc"].nunique()),
            "n_train_weeks": len(train_weeks),
            "n_test_weeks": len(test_weeks),
            "train_week_range": [train_weeks[0], train_weeks[-1]] if train_weeks else [],
            "test_week_range": [test_weeks[0], test_weeks[-1]] if test_weeks else [],
        },
        "metrics": {
            "ridge_median_impute_scaled": {
                "train": metrics_dict(y_train_d, pred_tr_ridge),
                "test": metrics_dict(y_test_d, pred_te_ridge),
            },
            "hist_gradient_boosting": {
                "train": metrics_dict(y_train_d, pred_tr_hgb),
                "test": metrics_dict(y_test_d, pred_te_hgb),
            },
        },
        "baseline_note": (
            "Ridge uses SimpleImputer(median) fit on train only, then StandardScaler + Ridge. "
            "Required because OLS/Ridge do not accept NaN; this is minimal single-split imputation, not iterative or model-based."
        ),
        "tree_note": (
            "HistGradientBoostingRegressor uses default early stopping on a validation fraction; "
            "missing values in numeric features are handled natively (no imputation)."
        ),
        "feature_importance_hist_gradient_boosting": imp,
        "feature_importance_method": "sklearn.inspection.permutation_importance (neg RMSE; train subsample)",
        "importance_groups": _importance_groups(imp),
        "error_analysis_test": {
            "by_week": summarize_group_errors(
                test_df.reset_index(drop=True),
                y_test_d,
                pred_te_hgb,
                "week_start_utc",
            ),
            "by_grid": summarize_group_errors(
                test_df.reset_index(drop=True),
                y_test_d,
                pred_te_hgb,
                "grid_cell_id",
            )[:50],
            "by_grid_note": "First 50 grids by group key order (for file size); full metrics computable from predictions.parquet",
        },
        "rolling_validation_train_timeline": expanding_window_validation(
            train_df.reset_index(drop=True),
            X_train.reset_index(drop=True),
            y_train_d,
            feature_cols,
            min_train_weeks=max(8, len(train_weeks) // 4),
            n_folds=3,
        ),
    }

    m_ridge_d = results["metrics"]["ridge_median_impute_scaled"]["test"]
    m_hgb_d = results["metrics"]["hist_gradient_boosting"]["test"]
    rte = m_ridge_d["rmse"]
    hte = m_hgb_d["rmse"]
    results["test_comparison"] = {
        "lower_rmse_is_better": (
            "hist_gradient_boosting" if hte < rte else "ridge_median_impute_scaled"
        ),
        "rmse_ridge_test": rte,
        "rmse_hist_gradient_boosting_test": hte,
        "rmse_delta_ridge_minus_hgb_test": float(rte - hte),
    }

    metrics_n = {
        "ridge_median_impute_scaled": {
            "train": metrics_dict(y_train_n, pred_tr_ridge_n),
            "test": metrics_dict(y_test_n, pred_te_ridge_n),
        },
        "hist_gradient_boosting": {
            "train": metrics_dict(y_train_n, pred_tr_hgb_n),
            "test": metrics_dict(y_test_n, pred_te_hgb_n),
        },
    }
    results["metrics_ndti_next"] = metrics_n

    m_ridge_n = metrics_n["ridge_median_impute_scaled"]["test"]
    m_hgb_n = metrics_n["hist_gradient_boosting"]["test"]

    results["ndti_next_comparison"] = {
        "target": "ndti_next = NDTI at week t+1 (sentinel_ndti_mean_t shifted −1 within grid)",
        "same_features_and_split_as_delta_ndti": True,
        "test_metrics_side_by_side": {
            "delta_ndti": {
                "ridge": {k: m_ridge_d[k] for k in ("rmse", "mae", "r2", "n")},
                "hist_gradient_boosting": {k: m_hgb_d[k] for k in ("rmse", "mae", "r2", "n")},
            },
            "ndti_next": {
                "ridge": {k: m_ridge_n[k] for k in ("rmse", "mae", "r2", "n")},
                "hist_gradient_boosting": {k: m_hgb_n[k] for k in ("rmse", "mae", "r2", "n")},
            },
        },
        "which_lower_test_rmse": {
            "ridge": (
                "ndti_next" if m_ridge_n["rmse"] < m_ridge_d["rmse"] else "delta_ndti"
            ),
            "hist_gradient_boosting": (
                "ndti_next" if m_hgb_n["rmse"] < m_hgb_d["rmse"] else "delta_ndti"
            ),
        },
        "feature_importance_hist_gradient_boosting": imp_n,
        "importance_groups": _importance_groups(imp_n),
        "interpretation_notes": [
            "Change prediction (delta_ndti) targets week-to-week differences; level prediction (ndti_next) targets absolute NDTI one week ahead.",
            "If ndti_next has lower RMSE, absolute turbidity level may be smoother or more learnable from lagged spectra and traffic than incremental changes (which can be noisier).",
            "If delta_ndti has lower RMSE, dynamics / residuals may be more predictable than levels once seasonality is partialled out—compare in context of negative R² and non-stationarity.",
            "Do not equate lower RMSE with causal interpretation; both tasks are associative forecasts under the same leakage-safe feature rule.",
        ],
        "thesis_artifacts": {
            "comparison_table_csv": str(plots_dir / "task_comparison_test.csv"),
            "comparison_figure_png": str(plots_dir / "task_comparison_test_metrics.png"),
            "note": "CSV and bar chart generated each run; import CSV into thesis table.",
        },
    }

    results["thesis_notes"] = [
        "Rows use has_valid_delta_ndti and non-null NDTI at t and t+1; weeks and rows are not randomly shuffled.",
        "Train weeks are strictly earlier than test weeks on the master timeline (time-aware split).",
        "If test R² is negative, the model is outperformed by predicting the mean target on the holdout—report and discuss shift or misspecification.",
        "HistGradientBoostingRegressor may use a random internal validation split only for early stopping; it does not inject future test weeks into training.",
        "Secondary task ndti_next uses identical features and split as delta_ndti for controlled comparison (see ndti_next_comparison).",
    ]

    save_task_comparison_table(
        plots_dir / "task_comparison_test.csv",
        m_ridge_d=m_ridge_d,
        m_hgb_d=m_hgb_d,
        m_ridge_n=m_ridge_n,
        m_hgb_n=m_hgb_n,
    )
    plot_task_comparison_figure(
        plots_dir / "task_comparison_test_metrics.png",
        m_ridge_d=m_ridge_d,
        m_hgb_d=m_hgb_d,
        m_ridge_n=m_ridge_n,
        m_hgb_n=m_hgb_n,
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(predictions_path, index=False)

    results_json.parent.mkdir(parents=True, exist_ok=True)
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Plots (test set, HGB and Ridge) — delta_ndti
    plot_pred_vs_actual(
        y_test_d,
        pred_te_ridge,
        "Test: Ridge — predicted vs actual delta_ndti",
        plots_dir / "pred_vs_actual_test_ridge.png",
        "Ridge",
        xlabel="Actual delta_ndti",
        ylabel="Predicted delta_ndti",
    )
    plot_pred_vs_actual(
        y_test_d,
        pred_te_hgb,
        "Test: HistGradientBoosting — predicted vs actual delta_ndti",
        plots_dir / "pred_vs_actual_test_hgb.png",
        "HGB",
        xlabel="Actual delta_ndti",
        ylabel="Predicted delta_ndti",
    )
    plot_residuals(
        y_test_d,
        pred_te_hgb,
        "Test residuals (HGB, delta_ndti)",
        plots_dir / "residuals_test_hgb.png",
    )
    plot_importance(
        imp,
        plots_dir / "feature_importance_hgb.png",
        "HistGradientBoosting feature importance (delta_ndti)",
    )

    # Plots — ndti_next
    plot_pred_vs_actual(
        y_test_n,
        pred_te_ridge_n,
        "Test: Ridge — predicted vs actual NDTI(t+1)",
        plots_dir / "pred_vs_actual_test_ridge_ndti_next.png",
        "Ridge",
        xlabel="Actual ndti_next",
        ylabel="Predicted ndti_next",
    )
    plot_pred_vs_actual(
        y_test_n,
        pred_te_hgb_n,
        "Test: HistGradientBoosting — predicted vs actual NDTI(t+1)",
        plots_dir / "pred_vs_actual_test_hgb_ndti_next.png",
        "HGB",
        xlabel="Actual ndti_next",
        ylabel="Predicted ndti_next",
    )
    plot_importance(
        imp_n,
        plots_dir / "feature_importance_hgb_ndti_next.png",
        "HistGradientBoosting feature importance (ndti_next)",
    )

    print(f"Wrote {results_json}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote plots under {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delta NDTI modeling (time-aware split).")
    parser.add_argument("--input", type=Path, default=Path("data/modeling_dataset.parquet"))
    parser.add_argument("--results", type=Path, default=Path("data/model_results.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions.parquet"))
    parser.add_argument("--plots-dir", type=Path, default=Path("outputs/modeling"))
    parser.add_argument("--train-frac", type=float, default=0.75, help="Fraction of distinct weeks in train")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    inp = args.input if args.input.is_absolute() else root / args.input
    res = args.results if args.results.is_absolute() else root / args.results
    pred = args.predictions if args.predictions.is_absolute() else root / args.predictions
    plots = args.plots_dir if args.plots_dir.is_absolute() else root / args.plots_dir

    run(
        input_path=inp,
        results_json=res,
        predictions_path=pred,
        plots_dir=plots,
        train_frac=args.train_frac,
    )


if __name__ == "__main__":
    main()
