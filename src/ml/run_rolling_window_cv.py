#!/usr/bin/env python3
"""
Expanding-window (rolling temporal) cross-validation on the thesis ML setup.

Imports the same data loading, feature construction, Ridge pipeline, and
HistGradientBoostingRegressor configuration as ``run_delta_ndti_models.py``.
No row shuffling; splits use ordered distinct calendar ``week_start_utc`` only.

Imputation and scaling are fit **only** on rows in each training fold (Ridge
``Pipeline.fit`` on the fold train subset). HGB is fit on the fold train subset only
(early stopping uses an internal withheld fraction of **that** fold's train rows).

Outputs under ``outputs/ml_cv_results/`` (CSV, JSON, PNG).

Usage::
    python src/ml/run_rolling_window_cv.py
    python src/ml/run_rolling_window_cv.py --input /path/to/modeling_dataset.parquet
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Imports: thesis module lives beside this package under ``src/``.
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import run_delta_ndti_models as dm  # noqa: E402

OUT_REL = Path("outputs/ml_cv_results")

# Thesis fold schedule: expanding train window; next block(s) as test.
# Indices = counts of consecutive distinct weeks from timeline start (sorted ``week_start_utc``).
EXPANDING_FOLD_SCHEDULE: list[tuple[int, int | None]] = [
    (20, 5),       # Fold 1: train 20 weeks, test following 5
    (25, 5),
    (30, 5),
    (35, 5),
    (40, None),    # Fold 5: train 40 weeks; test = all remaining weeks
]

TARGET_COLS = {
    "delta_ndti": "delta_ndti",
    "ndti_next": "ndti_next_target",
}


def _norm_week_ts(w: Any) -> pd.Timestamp:
    return pd.Timestamp(w)


def build_expanding_week_folds(sorted_weeks: list[Any]) -> list[dict[str, Any]]:
    """
    Build train/test week lists for each fold from EXPANDING_FOLD_SCHEDULE.

    Folds with empty train or test are skipped; metadata records ``skipped: true``.
    """
    weeks = [_norm_week_ts(x) for x in sorted_weeks]
    n = len(weeks)
    folds: list[dict[str, Any]] = []

    for fold_id, (n_train_weeks, n_test_weeks) in enumerate(EXPANDING_FOLD_SCHEDULE, start=1):
        if n_train_weeks > n:
            folds.append(
                {
                    "fold_id": fold_id,
                    "skipped": True,
                    "reason": "n_distinct_weeks < n_train_weeks",
                    "n_train_weeks_requested": n_train_weeks,
                    "n_test_weeks_requested": n_test_weeks,
                },
            )
            continue

        train_weeks_list = weeks[:n_train_weeks]
        if n_test_weeks is None:
            test_weeks_list = weeks[n_train_weeks:]
        else:
            hi = n_train_weeks + n_test_weeks
            test_weeks_list = weeks[n_train_weeks:hi]

        if not test_weeks_list:
            folds.append(
                {
                    "fold_id": fold_id,
                    "skipped": True,
                    "reason": "no test weeks remaining after train slice",
                    "n_train_weeks_effective": len(train_weeks_list),
                    "n_weeks_total": n,
                },
            )
            continue

        folds.append(
            {
                "fold_id": fold_id,
                "skipped": False,
                "n_train_weeks_effective": len(train_weeks_list),
                "n_test_weeks_effective": len(test_weeks_list),
                "train_weeks": train_weeks_list,
                "test_weeks": test_weeks_list,
                "train_week_range": [str(train_weeks_list[0]), str(train_weeks_list[-1])],
                "test_week_range": [str(test_weeks_list[0]), str(test_weeks_list[-1])],
            },
        )
    return folds


def slice_xy_for_weeks(
    df: pd.DataFrame,
    feature_cols: list[str],
    y_col: str,
    train_weeks: list[pd.Timestamp],
    test_weeks: list[pd.Timestamp],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Disjoint train/test matrices and targets by calendar week membership.

    ``week_start_utc`` is normalized to ``pd.Timestamp`` for set membership checks.
    """
    tw_set = {pd.Timestamp(t) for t in train_weeks}
    te_set = {pd.Timestamp(t) for t in test_weeks}
    wk = df["week_start_utc"].map(lambda z: pd.Timestamp(z))
    mt = wk.isin(tw_set)
    me = wk.isin(te_set)
    tr_df = df.loc[mt].copy().reset_index(drop=True)
    te_df = df.loc[me].copy().reset_index(drop=True)
    # Identical preprocessing to thesis: ``prepare_X`` on fold subsets only (no leakage).
    X_train = dm.prepare_X(tr_df, feature_cols)
    X_test = dm.prepare_X(te_df, feature_cols)
    y_train = tr_df[y_col].to_numpy(dtype=float)
    y_test = te_df[y_col].to_numpy(dtype=float)
    return X_train, y_train, X_test, y_test


def evaluate_one_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """
    Fit Ridge (median impute + scaler → Ridge) and HistGradientBoosting on train only.

    Test metrics exclude test rows from fitting (no lookahead). HGB retains thesis
    ``early_stopping`` / ``validation_fraction`` behaviour (sklearn withholds part of *train*
    for validation); this matches ``run_delta_ndti_models.fit_hgb`` intentionally.
    """
    pred_tr_r, pred_te_r, _ = dm.fit_ridge(X_train, y_train, X_test, y_test)
    pred_tr_h, pred_te_h, _ = dm.fit_hgb(X_train, y_train, X_test, y_test)

    return {
        "ridge": {
            "train": dm.metrics_dict(y_train, pred_tr_r),
            "test": dm.metrics_dict(y_test, pred_te_r),
        },
        "hist_gradient_boosting": {
            "train": dm.metrics_dict(y_train, pred_tr_h),
            "test": dm.metrics_dict(y_test, pred_te_h),
        },
    }


def rows_to_flat_records(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (fold × target × model) with flattened train/test metrics."""
    flat: list[dict[str, Any]] = []
    for row in rows:
        fid = row["fold_id"]
        target = row["target"]
        for model_key, display_name in (
            ("ridge", "ridge_median_impute_scaled"),
            ("hist_gradient_boosting", "hist_gradient_boosting"),
        ):
            m = row["models"][model_key]
            tr = m["train"]
            te = m["test"]
            flat.append(
                {
                    "fold_id": fid,
                    "target": target,
                    "model": display_name,
                    "n_weeks_train": row["n_weeks_train"],
                    "n_weeks_test": row["n_weeks_test"],
                    "train_week_range_start": row["train_week_range_start"],
                    "train_week_range_end": row["train_week_range_end"],
                    "test_week_range_start": row["test_week_range_start"],
                    "test_week_range_end": row["test_week_range_end"],
                    "n_train_rows": row["n_train_rows"],
                    "n_test_rows": row["n_test_rows"],
                    "train_rmse": tr["rmse"],
                    "train_r2": tr["r2"],
                    "test_rmse": te["rmse"],
                    "test_r2": te["r2"],
                },
            )
    return pd.DataFrame(flat)


def aggregate_averages(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of train and test RMSE & R² over folds per (target, model)."""
    g = df_flat.groupby(["target", "model"], sort=False)
    return (
        g.agg(
            n_folds=("fold_id", "count"),
            train_rows_mean=("n_train_rows", "mean"),
            test_rows_mean=("n_test_rows", "mean"),
            train_rmse_mean=("train_rmse", "mean"),
            train_rmse_std=("train_rmse", "std"),
            train_r2_mean=("train_r2", "mean"),
            train_r2_std=("train_r2", "std"),
            test_rmse_mean=("test_rmse", "mean"),
            test_rmse_std=("test_rmse", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
        )
        .reset_index()
    )


def best_worst_by_metric(
    df_flat: pd.DataFrame,
    metric_col: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    """Per (target, model): fold IDs for best/worst *test* metric."""
    out: dict[str, Any] = {}
    for (target, model), grp in df_flat.groupby(["target", "model"], sort=False):
        if grp.empty or grp[metric_col].isna().all():
            continue
        key = f"{target}::{model}"
        if lower_is_better:
            i_best = grp[metric_col].idxmin()
            i_worst = grp[metric_col].idxmax()
        else:
            i_best = grp[metric_col].idxmax()
            i_worst = grp[metric_col].idxmin()
        out[key] = {
            "best_fold": int(grp.loc[i_best, "fold_id"]),
            "best_value": float(grp.loc[i_best, metric_col]),
            "worst_fold": int(grp.loc[i_worst, "fold_id"]),
            "worst_value": float(grp.loc[i_worst, metric_col]),
        }
    return out


def _boxplot_compat(ax: Any, data: list[np.ndarray], tick_labels: list[str]) -> Any:
    """Matplotlib >=3.9 renamed ``labels`` → ``tick_labels`` for boxplot."""
    kwargs: dict[str, Any] = {"patch_artist": True}
    params = inspect.signature(ax.boxplot).parameters
    if "tick_labels" in params:
        kwargs["tick_labels"] = tick_labels
    else:
        kwargs["labels"] = tick_labels
    return ax.boxplot(data, **kwargs)


def plot_boxplot_single_metric(
    df_flat: pd.DataFrame,
    col: str,
    y_label: str,
    out_png: Path,
) -> None:
    """Separate PNG boxplot for RMSE and for R² across folds (one panel per target)."""
    targets = sorted(df_flat["target"].unique())
    models = ["ridge_median_impute_scaled", "hist_gradient_boosting"]
    palette = {"ridge_median_impute_scaled": "#c44e52", "hist_gradient_boosting": "#31708f"}

    ncols = max(1, len(targets))
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 5.0), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray([axes])

    for j, tgt in enumerate(targets):
        ax = axes[j]
        sub = df_flat[df_flat["target"].eq(tgt)]
        boxes: list[np.ndarray] = []
        tick_labels: list[str] = []
        colors: list[str] = []
        for m in models:
            sl = sub[sub["model"].eq(m)]
            if not sl.empty:
                boxes.append(sl[col].to_numpy(dtype=float))
                tick_labels.append(m.replace("_", "\n"))
                colors.append(palette[m])
        if boxes:
            bp = _boxplot_compat(ax, boxes, tick_labels)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
        if col == "test_r2":
            ax.axhline(0.0, color="#333333", lw=0.85, linestyle=":")
        ax.set_ylabel(y_label)
        ax.set_title(f"{tgt}\n{y_label} across folds")

    fig.savefig(out_png, dpi=175, bbox_inches="tight")
    plt.close(fig)


def plot_fold_lines(df_flat: pd.DataFrame, out_dir: Path) -> None:
    """Line plots per target: fold_id vs train/test RMSE and R² (separate PNG)."""
    palette = {"ridge_median_impute_scaled": "#c44e52", "hist_gradient_boosting": "#31708f"}
    model_order = ["ridge_median_impute_scaled", "hist_gradient_boosting"]

    for tgt in sorted(df_flat["target"].unique()):
        sub = df_flat[df_flat["target"].eq(tgt)].copy()
        sub = sub.sort_values(["fold_id", "model"])
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.8), constrained_layout=True)
        for model in model_order:
            g = sub[sub["model"].eq(model)]
            if g.empty:
                continue
            c = palette[model]
            axes[0, 0].plot(g["fold_id"], g["train_rmse"], marker="o", label=model, color=c)
            axes[0, 1].plot(g["fold_id"], g["test_rmse"], marker="o", label=model, color=c)
            axes[1, 0].plot(g["fold_id"], g["train_r2"], marker="s", label=model, color=c)
            axes[1, 1].plot(g["fold_id"], g["test_r2"], marker="s", label=model, color=c)
        axes[0, 0].set_title("Train RMSE")
        axes[0, 1].set_title("Test RMSE")
        axes[1, 0].set_title("Train R²")
        axes[1, 1].set_title("Test R²")
        for ax in axes.flat:
            ax.set_xlabel("Fold")
            ax.grid(True, alpha=0.35)
            ax.legend(fontsize=7.75)
        for ax in (axes[1, 0], axes[1, 1]):
            ax.axhline(0.0, color="#555555", lw=0.75, linestyle=":")
        fig.suptitle(f"Fold-wise performance — {tgt}", fontsize=12)
        fig.savefig(out_dir / f"rolling_window_lines_foldwise_{tgt}.png", dpi=175, bbox_inches="tight")
        plt.close(fig)


def print_interpretation(avg_df: pd.DataFrame, df_flat: pd.DataFrame, summary_bw: dict[str, Any]) -> None:
    """Concise thesis-oriented synthesis printed to stderr/stdout."""

    lines: list[str] = []
    lines.append("\n" + "=" * 72)
    lines.append("ROLLING WINDOW CV — INTERPRETATION SUMMARY")
    lines.append("=" * 72 + "\n")

    lines.append("HGB vs Ridge (mean test RMSE / R² across folds)")
    lines.append("")
    for _, rr in avg_df.iterrows():
        tgt, mod = rr["target"], rr["model"]
        lines.append(f"  {tgt} · {mod}")
        lines.append(
            f"    test RMSE: {rr['test_rmse_mean']:.6f} ± {rr['test_rmse_std']:.6f}  "
            f"(train {rr['train_rmse_mean']:.6f} ± {rr['train_rmse_std']:.6f})",
        )
        lines.append(
            f"    test R² : {rr['test_r2_mean']:.6f} ± {rr['test_r2_std']:.6f}  "
            f"(train {rr['train_r2_mean']:.6f} ± {rr['train_r2_std']:.6f})",
        )
        lines.append("")

    # Fold-by-fold: HGB lower test RMSE?
    lines.append("Cross-fold dominance (test RMSE: HistGradientBoosting < Ridge)")
    for tgt in sorted(df_flat["target"].unique()):
        sub = df_flat[df_flat["target"].eq(tgt)]
        n_comp = 0
        n_hgb_wins = 0
        for fid in sorted(sub["fold_id"].unique()):
            r = sub[(sub["fold_id"] == fid) & (sub["model"].eq("ridge_median_impute_scaled"))]
            h = sub[(sub["fold_id"] == fid) & (sub["model"].eq("hist_gradient_boosting"))]
            if r.empty or h.empty:
                continue
            n_comp += 1
            if h.iloc[0]["test_rmse"] < r.iloc[0]["test_rmse"]:
                n_hgb_wins += 1
        pct = (100 * n_hgb_wins / n_comp) if n_comp else float("nan")
        lines.append(
            f"  {tgt}: HGB beats Ridge on {n_hgb_wins}/{n_comp} folds ({pct:.1f}% of comparable folds)."
        )

    lines.append("")
    rel_cv = avg_df.assign(
        _rel=(avg_df["test_rmse_std"] / avg_df["test_rmse_mean"].replace(0, np.nan)).replace(np.inf, np.nan),
    )

    lines.append("Per-target median of (Ridge, HGB) **mean test R²** rows (read targets separately):")
    for tgt in sorted(avg_df["target"].unique()):
        subm = avg_df.loc[avg_df["target"].eq(tgt), "test_r2_mean"]
        lines.append(f"  {tgt}: median = {subm.median():.4f}")

    lines.append("")
    lines.append(
        "Interpretation bullets (discussion framing): • If HGB wins most folds on RMSE and/or R², "
        "it **consistently** outperforms the linear ridge baseline under temporal extrapolation.",
    )

    unstable = float(rel_cv["_rel"].max()) > 0.25 if rel_cv["_rel"].notna().any() else False

    cv_max = float(rel_cv["_rel"].max()) if rel_cv["_rel"].notna().any() else float("nan")
    lines.append(
        "• Temporal instability / distribution-shift: "
        + (
            "fold-to-fold test RMSE shows **moderate/high** dispersion (relative std/max mean ≈ {:.2f}"
            "); this supports **environmental variability** when paired with drifting drivers."
            if unstable
            else "relative fold-to-fold variation in mean test RMSE is **modest** (max std/mean ≈ {:.2f}); "
            "still inspect single worst folds."
        ).format(cv_max),
    )

    dsub = avg_df.loc[avg_df["target"].eq("delta_ndti"), "test_r2_mean"]
    nsub = avg_df.loc[avg_df["target"].eq("ndti_next"), "test_r2_mean"]
    dm = float(dsub.median()) if len(dsub) else float("nan")
    nm = float(nsub.median()) if len(nsub) else float("nan")

    def _tier(x: float) -> str:
        if not np.isfinite(x):
            return "n/a"
        if x <= 0.05:
            return "weak/near baseline"
        if x < 0.35:
            return "moderate"
        return "strong (check target definition & train–test gaps)"

    lines.append(
        "• Out-of-sample generalisation (tiered per target via median across the two models' mean test R²): "
        f"delta_ndti ≈ {_tier(dm)}; ndti_next ≈ {_tier(nm)}.",
    )
    lines.append(
        "    Contrast fold-wise train vs test curves in the PNGs for overfitting pockets or instability.",
    )

    lines.append("")
    lines.append(f"Best/worst folds (test RMSE, lower better): {json.dumps(summary_bw, indent=2)}")

    print("\n".join(lines))


def run_cv(input_parquet: Path, output_dir: Path) -> None:
    # -------------------------------------------------------------------------
    # 1) Load panel (thesis filtering + sorted — same panel as thesis models).
    # -------------------------------------------------------------------------
    df = dm.load_modeling_data(input_parquet)
    # Strict calendar-time row order for reproducible IO and thesis narrative (no shuffle).
    df = df.sort_values(["week_start_utc", "grid_cell_id"], kind="mergesort").reset_index(drop=True)

    weeks_sorted = sorted(df["week_start_utc"].dropna().unique(), key=lambda w: pd.Timestamp(w))
    fold_defs = build_expanding_week_folds(weeks_sorted)

    feature_cols = dm.feature_columns(df)

    rows_out: list[dict[str, Any]] = []

    for spec in fold_defs:
        if spec.get("skipped"):
            continue

        train_weeks = list(spec["train_weeks"])
        test_weeks = list(spec["test_weeks"])

        tw_set = {pd.Timestamp(t) for t in train_weeks}
        te_set = {pd.Timestamp(t) for t in test_weeks}
        # Ensure strict temporal isolation: calendar weeks disjoint.
        overlap = tw_set & te_set
        if overlap:
            raise RuntimeError(f"Leak: train/test week overlap fold {spec['fold_id']}: {overlap}")

        earliest_test = min(test_weeks)
        latest_train = max(train_weeks)
        if earliest_test <= latest_train:
            raise RuntimeError(
                "Expected test weeks strictly after last train week; "
                f"fold={spec['fold_id']} train_end={latest_train} test_start={earliest_test}",
            )

        for target_label, y_col in TARGET_COLS.items():
            X_train, y_train, X_test, y_test = slice_xy_for_weeks(
                df,
                feature_cols,
                y_col,
                train_weeks,
                test_weeks,
            )

            models_eval = evaluate_one_fold(X_train, y_train, X_test, y_test)

            rows_out.append(
                {
                    "fold_id": spec["fold_id"],
                    "target": target_label,
                    "n_weeks_train": spec["n_train_weeks_effective"],
                    "n_weeks_test": spec["n_test_weeks_effective"],
                    "train_week_range_start": spec["train_week_range"][0],
                    "train_week_range_end": spec["train_week_range"][1],
                    "test_week_range_start": spec["test_week_range"][0],
                    "test_week_range_end": spec["test_week_range"][1],
                    "n_train_rows": len(y_train),
                    "n_test_rows": len(y_test),
                    "models": models_eval,
                },
            )

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_flat = rows_to_flat_records(rows_out)

    df_flat.to_csv(out_dir / "rolling_window_metrics.csv", index=False)

    avg_df = aggregate_averages(df_flat)
    avg_df.to_csv(out_dir / "rolling_window_average_metrics.csv", index=False)

    skipped = [f for f in fold_defs if f.get("skipped")]
    summary_bw = {
        "test_rmse_lower_better": best_worst_by_metric(df_flat, "test_rmse", lower_is_better=True),
        "test_r2_higher_better": best_worst_by_metric(df_flat, "test_r2", lower_is_better=False),
    }

    summary: dict[str, Any] = {
        "pipeline_notes": (
            "Reuses run_delta_ndti_models: load_modeling_data, feature_columns, prepare_X; "
            "fit_ridge (median SimpleImputer + StandardScaler + Ridge(alpha=1.0)); "
            "fit_hgb (HistGradientBoostingRegressor thesis defaults). Fit only on each fold train rows."
        ),
        "n_rows_panel": len(df),
        "n_distinct_calendar_weeks": len(weeks_sorted),
        "week_order_constraint": (
            "DataFrame sorted by calendar week ascending for readability; folds use distinct week buckets only."
        ),
        "fold_schedule": [{"train_weeks": a, "test_weeks": b} for a, b in EXPANDING_FOLD_SCHEDULE],
        "fold_definitions_meta": fold_defs,
        "skipped_folds": skipped,
        "best_worst_folds": summary_bw,
        "average_metrics_records": avg_df.round(8).to_dict(orient="records"),
    }

    with (out_dir / "rolling_window_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Split boxplots exactly as requested: RMSE figure and R² figure.
    plot_boxplot_single_metric(df_flat, "test_rmse", "Test RMSE", out_dir / "rolling_window_boxplot_rmse.png")
    plot_boxplot_single_metric(df_flat, "test_r2", "Test R²", out_dir / "rolling_window_boxplot_r2.png")
    plot_fold_lines(df_flat, out_dir)

    print(f"Wrote {out_dir / 'rolling_window_metrics.csv'}")
    print(f"Wrote {out_dir / 'rolling_window_summary.json'}")
    print(f"Wrote {out_dir / 'rolling_window_average_metrics.csv'}")
    print(f"Wrote PNGs under {out_dir}")

    print("\n=== Average test metrics per model (mean ± std across folds) ===")
    disp = avg_df[
        ["target", "model", "n_folds", "test_rmse_mean", "test_rmse_std", "test_r2_mean", "test_r2_std"]
    ]
    print(disp.to_string(index=False))

    print("\n=== Best / worst fold (test RMSE, lower better) ===")
    for key, bw in summary_bw["test_rmse_lower_better"].items():
        print(f"  {key}: best fold {bw['best_fold']} (RMSE={bw['best_value']:.6f}), ", end="")
        print(f"worst fold {bw['worst_fold']} (RMSE={bw['worst_value']:.6f})")

    print("\n=== Best / worst fold (test R², higher better) ===")
    for key, bw in summary_bw["test_r2_higher_better"].items():
        print(f"  {key}: best fold {bw['best_fold']} (R²={bw['best_value']:.6f}), ", end="")
        print(f"worst fold {bw['worst_fold']} (R²={bw['worst_value']:.6f})")

    print_interpretation(avg_df, df_flat, summary_bw["test_rmse_lower_better"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding-window rolling temporal CV — thesis pipelines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data" / "modeling_dataset.parquet",
        help="Path to modeling parquet (same source as thesis script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / OUT_REL,
        help="Output directory for artefacts.",
    )
    args = parser.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(
            f"Input parquet missing: {inp}\nSpecify --input to modeling_dataset.parquet.",
        )

    run_cv(inp, args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
