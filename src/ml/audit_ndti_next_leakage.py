#!/usr/bin/env python3
"""
Diagnostics-only audit for potential target leakage in the ``ndti_next`` task.

Uses the same ``load_modeling_data``, ``feature_columns``, and ``prepare_X`` as
``run_delta_ndti_models.py`` / ``run_rolling_window_cv.py`` without changing them.

Writes reports under ``outputs/ml_cv_results/leakage_audit/``.

Usage::
    python src/ml/audit_ndti_next_leakage.py
    python src/ml/audit_ndti_next_leakage.py --input data/modeling_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def pearson_safe(x: np.ndarray, y_: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.nanstd(x) < 1e-15 or np.nanstd(y_) < 1e-15:
        return float("nan"), float("nan")
    try:
        r, p = stats.pearsonr(x, y_)
        return float(r), float(p)
    except ValueError:
        return float("nan"), float("nan")


def spearman_safe(x: np.ndarray, y_: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.nanstd(x) < 1e-15:
        return float("nan"), float("nan")
    try:
        r, p = stats.spearmanr(x, y_)
        return float(r), float(p)
    except ValueError:
        return float("nan"), float("nan")


def df_to_fence(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """Markdown-friendly table without optional ``tabulate`` dependency."""
    d = df.head(max_rows) if max_rows is not None else df
    body = d.to_string(index=False)
    return "```text\n" + body + "\n```"

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import run_delta_ndti_models as dm  # noqa: E402


OUT_SUBDIR = Path("outputs/ml_cv_results/leakage_audit")

SUSPICIOUS_NAME_PATTERNS = [
    re.compile(r"plus|\+1|_t\s*\+\s*1|tplus|forward|lead|future|next_", re.I),
    re.compile(r"ndti_next|target", re.I),
]


def prepare_X_audit(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numerical copies: raw-aligned (same index as df) for correlation / equality checks."""
    X = dm.prepare_X(df.reset_index(drop=True), feature_cols)
    y_df = df["ndti_next_target"].astype(float).reset_index(drop=True)
    return X, y_df.to_frame(name="ndti_next_target")


def validate_target_shift_identity(df_model: pd.DataFrame) -> dict[str, Any]:
    """Check that ndti_next_target matches group-wise shift(-1) of sentinel_ndti_mean_t."""
    g = df_model.groupby("grid_cell_id", sort=False)["sentinel_ndti_mean_t"]
    expected = g.shift(-1)
    y = df_model["ndti_next_target"].to_numpy(dtype=float)
    exp = expected.to_numpy(dtype=float)
    m = np.isfinite(y) & np.isfinite(exp)
    diff = np.abs(y[m] - exp[m])
    return {
        "statement": (
            "ndti_next_target is defined as groupby(grid_cell_id).sentinel_ndti_mean_t.shift(-1); "
            "row filter keeps both sentinel_ndti_mean_t and ndti_next_target non-null."
        ),
        "n_rows_checked": int(m.sum()),
        "max_abs_error": float(diff.max()) if diff.size else None,
        "mean_abs_error": float(diff.mean()) if diff.size else None,
        "fraction_exact_float_match": (
            float((diff <= np.finfo(float).resolution * np.maximum(np.abs(y[m]), 1.0)).mean()) if diff.size else None
        ),
    }


def correlations_with_target(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    yv = y.to_numpy(dtype=float)
    for col in X.columns:
        xv = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv)
        n = int(m.sum())
        if n < 3:
            rows.append(
                dict(
                    feature=col,
                    n_complete=n,
                    pearson_r=np.nan,
                    pearson_p=np.nan,
                    spearman_rho=np.nan,
                    spearman_p=np.nan,
                ),
            )
            continue
        pr, pp = pearson_safe(xv[m], yv[m])
        sr, sp = spearman_safe(xv[m], yv[m])
        rows.append(
            dict(
                feature=col,
                n_complete=n,
                pearson_r=float(pr),
                pearson_p=float(pp),
                spearman_rho=float(sr),
                spearman_p=float(sp),
            ),
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False, key=lambda s: np.abs(s))


def equality_like_audit(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-feature overlap with target: exact ratios and near-constant residuals."""
    yv = y.to_numpy(dtype=float)
    out_rows: list[dict[str, Any]] = []
    for col in X.columns:
        xv = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv)
        if not m.any():
            out_rows.append(
                dict(feature=col, n_complete=0, frac_exact_xy_equal=np.nan, max_abs_residual=np.nan),
            )
            continue
        xv_ = xv[m]
        yv_ = yv[m]
        exact = xv_ == yv_
        # Allow tiny float noise — “near identity” diagonal
        res = xv_ - yv_
        out_rows.append(
            dict(
                feature=col,
                n_complete=int(m.sum()),
                frac_exact_xy_equal=float(exact.mean()),
                max_abs_residual=float(np.max(np.abs(res))),
                median_abs_residual=float(np.median(np.abs(res))),
                rmse_residual_vs_target=float(np.sqrt(np.mean(res**2))),
            ),
        )
    return pd.DataFrame(out_rows)


def pairwise_high_corr(X: pd.DataFrame, *, threshold: float = 0.999) -> pd.DataFrame:
    """Upper triangle |r|>=threshold among numeric columns (complete-case per pair)."""
    cols = X.columns.to_list()
    num = pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce") for c in cols})
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            m = num[a].notna() & num[b].notna()
            if m.sum() < 5:
                continue
            xa = num.loc[m, a].to_numpy(dtype=float)
            xb = num.loc[m, b].to_numpy(dtype=float)
            r, _ = pearson_safe(xa, xb)
            if not np.isfinite(r) or abs(r) < threshold:
                continue
            pairs.append(dict(feature_a=a, feature_b=b, n_complete=int(m.sum()), pearson_r=float(r)))
    return pd.DataFrame(pairs)


def univariate_r2(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """R² of OLS y ~ 1 + x (complete cases)."""
    m = np.isfinite(y) & np.isfinite(x)
    if int(m.sum()) < 5:
        return {"n": float(m.sum()), "r2_univariate_ols_intercept": float("nan")}
    yy = y[m].astype(np.float64)
    xx = x[m].astype(np.float64)
    X_design = np.column_stack([np.ones_like(xx), xx])
    beta, *_ = np.linalg.lstsq(X_design, yy, rcond=None)
    pred = X_design @ beta
    sse = np.sum((yy - pred) ** 2)
    sst = np.sum((yy - yy.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 1e-12 else float("nan")
    return {"n": float(m.sum()), "r2_univariate_ols_intercept": float(r2), "intercept": float(beta[0]), "slope": float(beta[1])}


def duplicate_panel_keys(df: pd.DataFrame) -> dict[str, Any]:
    k = df.groupby(["grid_cell_id", "week_start_utc"], sort=False).size()
    dup = int((k > 1).sum())
    return dict(
        duplicate_key_pairs=int(dup),
        max_duplicate_count=int(k.max()) if len(k) else 0,
    )


def flagged_feature_names(features: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"regex_hits": []}
    hits: list[str] = []
    for f in features:
        if any(p.search(f) for p in SUSPICIOUS_NAME_PATTERNS):
            hits.append(f)
    out["regex_hits"] = hits
    return out


def static_pipeline_audit_text() -> str:
    return """### Static code audit (no data)

**Target construction** (`run_delta_ndti_models.load_modeling_data`):

1. Filter `has_valid_delta_ndti == True`.
2. Sort by `grid_cell_id`, `week_start_utc`.
3. `ndti_next_target = groupby(grid_cell_id).sentinel_ndti_mean_t.shift(-1)`.
4. Keep rows with non-null `sentinel_ndti_mean_t` and `ndti_next_target`.

**Feature matrix** (`feature_columns`): every column not in `META_COLS` = `{
  "grid_cell_id", "week_start_utc", "delta_ndti", "has_valid_delta_ndti", "ndti_next_target"
}`.

So **`sentinel_ndti_mean_t` (NDTI at week *t*) is an explicit predictor** while **`ndti_next_target` is NDTI at *t+1***.
This is allowed under the leakage rule documented in `docs/modeling_dataset_schema.md` (features ≤ *t*,
target uses *t+1* constructed at load time — not read from parquet as a column).

**Rolling CV** (`src/ml/run_rolling_window_cv.py`):

1. Rows are partitioned by disjoint sets of **calendar weeks** (`week_start_utc`); train weeks precede test weeks on the timeline.
2. Each fold builds `X_train = prepare_X(train_df)`, `X_test = prepare_X(test_df)` — no global fit before split.
3. `fit_ridge` / `fit_hgb` receive **training fold rows only**; `SimpleImputer` + `StandardScaler` are fitted **inside** Ridge’s `pipe.fit(X_train, y_train)` — test fold never participates in scaling/imputation fitting.

Therefore there is **no obvious train/test scaler leakage** introduced by rolling CV.


"""


def write_markdown_report(
    *,
    path: Path,
    shift_audit: dict[str, Any],
    dup_keys: dict[str, Any],
    forbidden_checks: dict[str, Any],
    name_flags: dict[str, list[str]],
    top_corr: pd.DataFrame,
    eq_audit: pd.DataFrame,
    near_dup_pairs: pd.DataFrame,
    univar_sentinel: dict[str, float],
    univar_delta_col: dict[str, float],
    n_rows: int,
    n_features: int,
    calendar_reconciliation: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# `ndti_next` target leakage audit\n")
    lines.append("**Scope:** diagnosis only; pipelines unchanged.\n")
    lines.append("## Executive summary\n")
    lines.append(
        "The thesis modeling code **does not place `ndti_next_target` in X**, and parquet features are constrained to lag "
        "*t*, *t−1*, *t−2* per `docs/modeling_dataset_schema.md`.\n\n"
        "**Current-week turbidity `sentinel_ndti_mean_t` is an explicit predictor** while **y encodes next-week NDTI** "
        "(see target construction). That is **consistent** with the leakage rule (predictors use information up to *t* "
        "only; *t+1* appears only via the constructed label). It is **not**, by itself, a bug — but it implies the "
        "**level-forecasting task can be easy** if NDTI evolves gradually and joint spectral predictors align. "
        "**Always check marginal correlations (Section 5)** and **whether your rolling CV metrics correspond to the "
        "same parquet/week span** you audit.\n\n"
        "Chronological holdouts in `data/model_results.json` can show **much worse** ndti_next test skill than short "
        "expanding windows — that pattern is often **regime / sample-size** shift rather than train/test leakage in code.\n",
    )
    lines.append("### Calendar reconciliation (rolling_window_metrics.csv vs this audit)\n\n")
    if calendar_reconciliation:
        lines.append(json.dumps(calendar_reconciliation, indent=2))
    else:
        lines.append("_No reconciliation dict passed._")
    lines.append("\n")

    lines.append("## 1. Target alignment & duplicate keys\n")
    lines.append(json.dumps(shift_audit, indent=2))
    lines.append("\n\nDuplicate (`grid_cell_id`, `week_start_utc`) groups:\n")
    lines.append(json.dumps(dup_keys, indent=2))
    lines.append("\n")

    lines.append("## 2. Forbidden targets in feature list\n")
    lines.append(json.dumps(forbidden_checks, indent=2))
    lines.append("\nRegExp-based name scan (forward-looking / target-like substrings):\n")
    lines.append(json.dumps(name_flags, indent=2))
    lines.append("\n")

    lines.append("## 3. Univariate explanatory power (full modeling panel after `load_modeling_data`)\n")
    lines.append("`sentinel_ndti_mean_t` only (OLS y ~ intercept +):\n")
    lines.append(json.dumps(univar_sentinel, indent=2))
    lines.append("\nOptional: `delta_ndti` correlation with y (**not used as predictor**):\n")
    lines.append(json.dumps(univar_delta_col, indent=2))
    lines.append("\n")

    lines.append(f"## 4. Rows & feature count post-filter\n")
    lines.append(f"- n_rows: **{n_rows}**\n- n_predictors |X|: **{n_features}**\n\n")

    lines.append("## 5. Top predictors correlated with `ndti_next_target`\n")
    lines.append("See `correlations_with_ndti_next.csv` (full table) and ")
    lines.append("`correlation_audit_top_predictors_matrix.csv` (pairwise Pearson among top |r| drivers). ")
    lines.append("Top 15 by |Pearson| versus y:\n\n")
    if top_corr.shape[0] > 0:
        lines.append(df_to_fence(top_corr.head(15)))
    lines.append("\n\n")

    lines.append("## 6. Near-duplicate / near-identity features vs target\n")
    lines.append("See `equality_audit_vs_target.csv` (exact match ratio between x and y, residual RMSE).\n\n")
    if not eq_audit.empty:
        risky = eq_audit.sort_values("frac_exact_xy_equal", ascending=False).head(8)
        lines.append(df_to_fence(risky))

    lines.append("\n\n## 7. Feature pairs with |Pearson r| ≥ 0.999\n")
    if near_dup_pairs.empty:
        lines.append("_None detected at this threshold._\n")
    else:
        lines.append(df_to_fence(near_dup_pairs))

    lines.append("\n\n## 8. Static preprocessing / split order\n")
    lines.append(static_pipeline_audit_text())

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ndti_next for feature/target leakage (diagnostics only).")
    parser.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "data" / "modeling_dataset.parquet",
        help="Modeling parquet (same input as thesis ML scripts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / OUT_SUBDIR,
        help="Directory for leakage audit artefacts.",
    )
    args = parser.parse_args()

    inp = args.input.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.is_file():
        stub = dict(
            error=f"Missing parquet: {inp}",
            recommendation="Provide --input to modeling_dataset.parquet; static audit text still useful.",
            static_pipeline_audit=static_pipeline_audit_text(),
        )
        (out_dir / "leakage_report.md").write_text(
            "# Leakage audit — data missing\n\n" + json.dumps(stub, indent=2),
            encoding="utf-8",
        )
        print(f"No data at {inp}; wrote stub report only.")
        sys.exit(0)

    df_raw = dm.load_modeling_data(inp)

    forbidden_checks = {
        "META_COLS": sorted(dm.META_COLS),
        "feature_list_includes_ndti_next_target": False,
        "feature_list_includes_delta_ndti": False,
        "feature_list_includes_has_valid_delta_ndti": False,
    }

    feats = dm.feature_columns(df_raw)
    forbidden_checks["feature_list_includes_ndti_next_target"] = "ndti_next_target" in feats
    forbidden_checks["feature_list_includes_delta_ndti"] = "delta_ndti" in feats
    forbidden_checks["feature_list_includes_has_valid_delta_ndti"] = "has_valid_delta_ndti" in feats

    (out_dir / "feature_list_used_for_training.json").write_text(json.dumps(sorted(feats), indent=2), encoding="utf-8")

    X, yfrm = prepare_X_audit(df_raw, feats)
    y = yfrm["ndti_next_target"]

    corr_tbl = correlations_with_target(X, y)
    corr_tbl.to_csv(out_dir / "correlations_with_ndti_next.csv", index=False)
    corr_tbl.head(25).to_csv(out_dir / "top_correlations_with_ndti_next_truncated.csv", index=False)

    top_cols = corr_tbl.assign(_a=np.abs(corr_tbl["pearson_r"])).nlargest(min(18, len(corr_tbl)), "_a")["feature"]
    top_cols_list = top_cols.tolist()
    sort_top = sorted(top_cols_list)
    corr_block = X[sort_top].corr(method="pearson")
    corr_block.to_csv(out_dir / "correlation_audit_top_predictors_matrix.csv")

    eq_audit = equality_like_audit(X, y)
    eq_audit.to_csv(out_dir / "equality_audit_vs_target.csv", index=False)

    near_dup = pairwise_high_corr(X, threshold=0.999)
    near_dup.to_csv(out_dir / "near_duplicate_feature_pairs.csv", index=False)

    shift_audit = validate_target_shift_identity(df_raw)
    (out_dir / "target_shift_verification.json").write_text(json.dumps(shift_audit, indent=2), encoding="utf-8")

    dup_keys = duplicate_panel_keys(df_raw)
    (out_dir / "duplicate_panel_key_summary.json").write_text(json.dumps(dup_keys, indent=2), encoding="utf-8")

    name_flags = flagged_feature_names(feats)
    (out_dir / "suspicious_feature_name_hits.json").write_text(json.dumps(name_flags, indent=2), encoding="utf-8")

    # Univariate R² for current-level NDTI vs next-level NDTI
    st = pd.to_numeric(X["sentinel_ndti_mean_t"], errors="coerce").to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    univar_sentinel = univariate_r2(yv, st)

    # delta_ndti is NOT in X but present on df — relationship diagnostic only
    delta = df_raw["delta_ndti"].to_numpy(dtype=float)
    univar_delta_col = univariate_r2(yv, delta)

    r_sent = corr_tbl.loc[corr_tbl["feature"].eq("sentinel_ndti_mean_t"), "pearson_r"]
    r_val = float(r_sent.iloc[0]) if len(r_sent) > 0 and np.isfinite(r_sent.iloc[0]) else float("nan")
    if np.isfinite(r_val) and abs(r_val) > 0.6:
        diag_text = (
            f"Strong linear association (Pearson r≈{r_val:.3f}) between current-week NDTI and next-week NDTI explains "
            "high out-of-sample R² when training and test windows share a stable seasonal regime."
        )
    elif np.isfinite(r_val) and abs(r_val) > 0.35:
        diag_text = (
            f"Moderate linear association (Pearson r≈{r_val:.3f}) between NDTI(t) and NDTI(t+1); multivariate ridge may "
            "still achieve high folded R² via joint spectral proxies, or metrics may derive from another parquet/calendar slice."
        )
    else:
        diag_text = (
            f"Pooled Pearson r≈{r_val:.3f} between sentinel_ndti_mean_t and ndti_next_target after load_modeling_data — "
            "not near-perfect linear persistence at full-panel scale; very high folded R² is unlikely from this marginal "
            "relationship alone unless the evaluated parquet/window differs (see rolling_metrics_calendar_reconciliation.json) "
            "or small-fold finite-sample/collinearity inflates sklearn R²."
        )

    leakage_diagnosis_summary = {
        "illegal_target_ndti_next_target_in_predictor_columns": forbidden_checks[
            "feature_list_includes_ndti_next_target"
        ],
        "illegal_delta_ndti_in_predictor_columns": forbidden_checks["feature_list_includes_delta_ndti"],
        "pearson_corr_sentinel_ndti_mean_t_with_ndti_next_target": r_val,
        "univariate_ols_r2_y_on_sentinel_ndti_mean_t_only": univar_sentinel,
        "diagnosis_plain_language": diag_text,
    }
    (out_dir / "leakage_diagnosis_summary.json").write_text(
        json.dumps(leakage_diagnosis_summary, indent=2),
        encoding="utf-8",
    )

    roll_csv = _ROOT / Path("outputs/ml_cv_results/rolling_window_metrics.csv")
    recon: dict[str, Any]
    if roll_csv.is_file():
        rw = pd.read_csv(roll_csv, nrows=1)
        tr0 = pd.to_datetime(rw["train_week_range_start"].iloc[0], utc=True, errors="coerce")
        wk_here = sorted(df_raw["week_start_utc"].dropna().unique())
        recon = {
            "rolling_metrics_csv_example_train_week_from_row1": str(tr0),
            "current_parquet_week_min_after_load_modeling_data": str(wk_here[0]),
            "current_parquet_week_max_after_load_modeling_data": str(wk_here[-1]),
            "overlap_interpretation": (
                "If week ranges disagree, rolling_window_metrics.csv was generated from "
                "a different modeling parquet or panel than audited here — compare artefacts before inferring leakage."
            ),
        }
    else:
        recon = {
            "note": "outputs/ml_cv_results/rolling_window_metrics.csv not found — skipped reconciliation.",
        }
    (out_dir / "rolling_metrics_calendar_reconciliation.json").write_text(json.dumps(recon, indent=2), encoding="utf-8")

    write_markdown_report(
        path=out_dir / "leakage_report.md",
        shift_audit=shift_audit,
        dup_keys=dup_keys,
        forbidden_checks=forbidden_checks,
        name_flags=name_flags,
        top_corr=corr_tbl,
        eq_audit=eq_audit,
        near_dup_pairs=near_dup,
        univar_sentinel=univar_sentinel,
        univar_delta_col=univar_delta_col,
        n_rows=len(df_raw),
        n_features=len(feats),
        calendar_reconciliation=recon,
    )

    (out_dir / "rolling_cv_preprocessing_note.md").write_text(
        "**Rolling CV preprocessing order (confirmed in source):**\n\n"
        "1. `load_modeling_data` constructs `ndti_next_target`; filter drops tail rows lacking t+1.\n"
        "2. For each temporal fold: `prepare_X(train_df)`, `prepare_X(test_df)` — no fit.\n"
        "3. `fit_ridge` fits `Pipeline(imputer→scaler→ridge)` on **X_train, y_train** only.\n"
        "4. `fit_hgb` fits on **X_train, y_train** only (internally withholds validation fraction "
        "**from training rows**, not test fold).\n",
        encoding="utf-8",
    )

    print(f"Leakage audit complete — wrote artefacts under:\n  {out_dir}")


if __name__ == "__main__":
    main()
