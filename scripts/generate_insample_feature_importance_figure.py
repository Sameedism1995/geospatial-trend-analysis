#!/usr/bin/env python3
"""
In-sample (training-data-only) ML feature importance figure for the thesis.

Uses the same time-aware week split and models as ``src/run_delta_ndti_models.py``.
Permutation importance and Ridge coefficients are computed on **training rows only**
and labelled explicitly as in-sample diagnostics (not out-of-sample validation).

Outputs::
    outputs/final_figures/fig_ml_insample_feature_importance.{png,pdf}
    outputs/final_figures/fig_ml_insample_feature_importance_table.csv
    outputs/final_figures/fig_ml_insample_feature_importance_caption.md
    outputs/final_thesis_figures/fig_ml_insample_feature_importance.{png,pdf}  (mirror)

Usage::
    python3 scripts/generate_insample_feature_importance_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_delta_ndti_models as dm  # noqa: E402

OUT_FINAL = ROOT / "outputs" / "final_figures"
OUT_THESIS = ROOT / "outputs" / "final_thesis_figures"
INPUT_DEFAULT = ROOT / "data" / "modeling_dataset.parquet"
TRAIN_FRAC = 0.75
TOP_N = 18
TOP_SPATIAL = 12
DPI = 400

CATEGORY_COLORS = {
    "spatial": "#2563eb",
    "maritime": "#0891b2",
    "temporal": "#64748b",
    "spectral": "#16a34a",
    "metadata": "#94a3b8",
}


def feature_category(name: str) -> str:
    n = str(name)
    if n in {"grid_centroid_lat", "grid_centroid_lon", "grid_res_deg"}:
        return "spatial"
    if n.startswith("vessel_density") or n == "has_emodnet":
        return "maritime"
    if n in {"week_of_year", "week_sin", "week_cos"}:
        return "temporal"
    if n.startswith("has_") or "observation_count" in n:
        return "metadata"
    if n.startswith("sentinel_"):
        return "spectral"
    return "metadata"


def ridge_train_importance(
    pipe,
    feature_cols: list[str],
) -> pd.DataFrame:
    ridge = pipe.named_steps["ridge"]
    coef = np.asarray(ridge.coef_, dtype=float)
    return (
        pd.DataFrame({"feature": feature_cols, "importance": np.abs(coef)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def hgb_train_permutation(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: list[str],
    *,
    max_samples: int = 2500,
    n_repeats: int = 10,
) -> pd.DataFrame:
    n = min(max_samples, len(X_train))
    rng = np.random.RandomState(dm.RANDOM_STATE)
    idx = rng.choice(len(X_train), size=n, replace=False)
    Xs = X_train.iloc[idx]
    ys = y_train[idx]
    r = permutation_importance(
        model,
        Xs,
        ys,
        n_repeats=n_repeats,
        random_state=dm.RANDOM_STATE,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": r.importances_mean,
                "importance_std": r.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _barh_panel(ax, imp: pd.DataFrame, title: str, top_n: int) -> None:
    sub = imp.head(top_n).iloc[::-1]
    names = sub["feature"].astype(str).tolist()
    vals = sub["importance"].to_numpy(dtype=float)
    colors = [CATEGORY_COLORS.get(feature_category(n), "#94a3b8") for n in names]
    ys = np.arange(len(names))
    ax.barh(ys, vals, color=colors, edgecolor="#1e293b", linewidth=0.35, height=0.72)
    ax.set_yticks(ys, labels=names, fontsize=8.5)
    ax.set_xlabel("Relative weight (in-sample)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)


def build_figure(
    ridge_imp: pd.DataFrame,
    hgb_imp: pd.DataFrame,
    split_meta: dict[str, Any],
    train_metrics: dict[str, float],
) -> plt.Figure:
    spatial_mask = lambda s: feature_category(s) in {"spatial", "maritime"}
    ridge_sp = ridge_imp[ridge_imp["feature"].map(spatial_mask)].head(TOP_SPATIAL)
    hgb_sp = hgb_imp[hgb_imp["feature"].map(spatial_mask)].head(TOP_SPATIAL)

    fig = plt.figure(figsize=(13.8, 9.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.38, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _barh_panel(
        ax_a,
        ridge_imp,
        f"Ridge regression · |standardized β| (top {TOP_N})",
        TOP_N,
    )
    _barh_panel(
        ax_b,
        hgb_imp,
        f"HistGradientBoosting · permutation ΔRMSE (top {TOP_N})",
        TOP_N,
    )
    _barh_panel(
        ax_c,
        ridge_sp,
        f"Ridge · spatial & maritime predictors (top {TOP_SPATIAL})",
        TOP_SPATIAL,
    )
    _barh_panel(
        ax_d,
        hgb_sp,
        f"HistGradientBoosting · spatial & maritime (top {TOP_SPATIAL})",
        TOP_SPATIAL,
    )

    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[k], label=k.capitalize())
        for k in ("spatial", "maritime", "temporal", "spectral", "metadata")
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    tw = split_meta.get("n_train_weeks", "?")
    te = split_meta.get("n_test_weeks", "?")
    n_tr = train_metrics.get("n", "?")
    r2_tr = train_metrics.get("r2", float("nan"))

    fig.suptitle(
        "In-sample feature importance · training weeks only (ΔNDTI target)",
        fontsize=14.2,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        (
            "Diagnostics computed on the temporal training partition only "
            f"({tw} calendar weeks, n={n_tr} cell-weeks). "
            "Not out-of-sample predictive validation "
            f"(train R²={r2_tr:.3f} on held-in training rows; test R² reported separately in §5.9)."
        ),
        ha="center",
        fontsize=9.2,
        color="#334155",
        style="italic",
    )

    plt.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.07)
    return fig


def write_caption(path: Path, table_path: Path, split_meta: dict[str, Any]) -> None:
    tw = split_meta.get("train_week_range", ["?", "?"])
    path.write_text(
        f"""# In-sample ML feature importance (training data only)

**Figure.** Feature weights for predicting **ΔNDTI** under the thesis time-aware split (
first {split_meta.get('train_frac', TRAIN_FRAC):.0%} of distinct calendar weeks → train).
**Left:** Ridge absolute standardized coefficients fit on training rows only.
**Right:** HistGradientBoosting permutation importance (ΔRMSE, 10 repeats) evaluated on a
training subsample only. Lower panels isolate **spatial** (`grid_centroid_lat/lon`, `grid_res_deg`)
and **maritime** (`vessel_density_t` lags) predictors. Bar colours encode feature family.

**Interpretation:** Shows which inputs the models emphasised **within the training window**—a
legitimate explainability contribution. Negative test R² under temporally separated holdout
(Chapter 5 ML validation) confirms these weights must not be read as forecast skill.

**Table:** `{table_path.name}`

**Train weeks:** {tw[0]} → {tw[-1]} ({split_meta.get('n_train_weeks')} weeks).
""",
        encoding="utf-8",
    )


def save_outputs(fig: plt.Figure, combined: pd.DataFrame, split_meta: dict[str, Any]) -> None:
    for out_dir in (OUT_FINAL, OUT_THESIS):
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / "fig_ml_insample_feature_importance"
        fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), dpi=DPI, bbox_inches="tight", facecolor="white")

    table_path = OUT_FINAL / "fig_ml_insample_feature_importance_table.csv"
    combined.to_csv(table_path, index=False)

    meta_path = OUT_FINAL / "fig_ml_insample_feature_importance_meta.json"
    meta_path.write_text(json.dumps(split_meta, indent=2, default=str), encoding="utf-8")

    write_caption(OUT_FINAL / "fig_ml_insample_feature_importance_caption.md", table_path, split_meta)
    plt.close(fig)
    print(f"Wrote {OUT_FINAL / 'fig_ml_insample_feature_importance.png'}")


def main() -> int:
    input_path = INPUT_DEFAULT
    if not input_path.is_file():
        print(f"Missing {input_path}", file=sys.stderr)
        return 1

    df = dm.load_modeling_data(input_path)
    feature_cols = dm.feature_columns(df)
    train_df, test_df, train_weeks, test_weeks = dm.time_aware_split(df, train_frac=TRAIN_FRAC)

    X_train = dm.prepare_X(train_df, feature_cols)
    X_test = dm.prepare_X(test_df, feature_cols)
    y_train = train_df["delta_ndti"].to_numpy(dtype=float)
    y_test = test_df["delta_ndti"].to_numpy(dtype=float)

    _, _, ridge_pipe = dm.fit_ridge(X_train, y_train, X_test, y_test)
    pred_tr_hgb, _, hgb = dm.fit_hgb(X_train, y_train, X_test, y_test)

    ridge_imp = ridge_train_importance(ridge_pipe, feature_cols)
    hgb_imp = hgb_train_permutation(hgb, X_train, y_train, feature_cols)

    train_metrics = dm.metrics_dict(y_train, pred_tr_hgb)
    split_meta = {
        "method": "time_aware_by_week",
        "train_frac": TRAIN_FRAC,
        "n_train_weeks": len(train_weeks),
        "n_test_weeks": len(test_weeks),
        "train_week_range": train_weeks,
        "test_week_range": test_weeks,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "target": "delta_ndti",
        "label": "in_sample_training_only",
    }

    combined = ridge_imp.rename(columns={"importance": "ridge_abs_coef"}).merge(
        hgb_imp.rename(columns={"importance": "hgb_perm_mean", "importance_std": "hgb_perm_std"}),
        on="feature",
        how="outer",
    )
    combined["category"] = combined["feature"].map(feature_category)
    combined = combined.sort_values("hgb_perm_mean", ascending=False, na_position="last")

    fig = build_figure(ridge_imp, hgb_imp, split_meta, train_metrics)
    save_outputs(fig, combined, split_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
