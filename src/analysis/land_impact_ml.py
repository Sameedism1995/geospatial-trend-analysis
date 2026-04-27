"""
Land Impact ML extension (LAND IMPACT EXTENSION LAYER, Step 5).

Trains TWO Random Forest regressors that predict NDVI (`ndvi_mean`) and
reports each one with grouped cross-validation by `grid_cell_id`:

1. **Primary (maritime-only)** — features: vessel density, NO2, S1
   disturbance, exposure, distance-to-vessel-hotspot, interaction indices,
   and lagged products. This is the model the thesis cites because it
   isolates the maritime-pressure → land-NDVI hypothesis.

2. **Leakage ceiling (with S2 spectral siblings)** — adds `fai_mean`,
   `ndwi_mean`, `ndti_mean`, `ndci_mean`, `b11_mean`. These are derived from
   the same Sentinel-2 product as the target and would inflate R² via
   inter-band coherence rather than maritime signal. Reported only as a
   sanity-check ceiling in the appendix.

Cross-validation:
    GroupKFold by `grid_cell_id` (up to 5 folds) so the same cell never
    appears in both train and test. Mean ± std of test R² is the headline.

Outputs (under outputs/reports/):
    land_impact_ml_model_summary.json           # primary (maritime-only)
    land_impact_ml_feature_importance.csv       # primary
    land_impact_ml_model_summary_with_s2_bands.json  # leakage ceiling
    land_impact_ml_feature_importance_with_s2_bands.csv  # leakage ceiling
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("land_impact_ml")

DEFAULT_TARGET = "ndvi_mean"

# Maritime + atmospheric + S1 features. NO Sentinel-2 reflectance bands here —
# those are spectral siblings of NDVI and would leak.
MARITIME_FEATURES: tuple[str, ...] = (
    "vessel_density",
    "NO2_mean",
    "oil_slick_probability_t",
    "detection_score",
    "coastal_exposure_score",
    "distance_to_nearest_high_vessel_density_cell",
    "maritime_pressure_index",
    "atmospheric_transfer_index",
    "vessel_x_no2",
    "vessel_x_ndvi_lag1",
    "vessel_x_ndvi_lag2",
    "vessel_x_ndvi_lag3",
)

# Same Sentinel-2 product as the target — using these as predictors is a
# leakage ceiling, not a maritime signal.
S2_LEAKAGE_FEATURES: tuple[str, ...] = (
    "fai_mean",
    "ndwi_mean",
    "ndti_mean",
    "ndci_mean",
    "b11_mean",
)

# Backwards-compat alias (the original module exported this name).
DEFAULT_FEATURE_CANDIDATES: tuple[str, ...] = MARITIME_FEATURES + S2_LEAKAGE_FEATURES

MIN_NON_NULL_PER_FEATURE = 50
MIN_USABLE_ROWS = 60
MAX_GROUP_FOLDS = 5


def _select_feature_columns(
    df: pd.DataFrame, candidates: Iterable[str], target: str
) -> list[str]:
    cols: list[str] = []
    for c in candidates:
        if c == target or c not in df.columns:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().sum() >= MIN_NON_NULL_PER_FEATURE:
            cols.append(c)
    return cols


def _empty_payload(target: str, label: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "target": target,
        "model_label": label,
        "features_used": [],
        "n_train": None,
        "n_test": None,
        "r2_train": None,
        "r2_test": None,
        "mae_test": None,
        "r2_cv_mean": None,
        "r2_cv_std": None,
        "n_cv_folds": None,
        "n_groups": None,
        "permutation_importance": [],
        "notes": "",
    }


def _train_one_model(
    features: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    label: str,
    *,
    random_state: int = 42,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Train a single RF model with grouped CV and a held-out test split."""
    payload = _empty_payload(target, label)

    if len(feature_cols) < 2:
        payload["notes"] = f"insufficient feature columns: {feature_cols}"
        return payload

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import GroupKFold, train_test_split
    except ImportError as exc:  # pragma: no cover - environment-dependent
        payload["notes"] = f"scikit-learn unavailable: {exc!r}"
        return payload

    y = pd.to_numeric(features[target], errors="coerce")
    X = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    groups_full = features.get("grid_cell_id")

    valid = y.notna() & X.notna().all(axis=1)
    if groups_full is not None:
        valid &= groups_full.notna()
    X = X[valid]
    y = y[valid]
    groups = groups_full[valid] if groups_full is not None else None

    n = int(len(y))
    payload["n_groups"] = int(groups.nunique()) if groups is not None else None
    if n < MIN_USABLE_ROWS:
        payload["notes"] = f"insufficient labelled rows after NA filter (n={n})"
        payload["n_train"] = n
        return payload

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=random_state
        )
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        r2_train = float(r2_score(y_train, y_pred_train))
        r2_test = float(r2_score(y_test, y_pred_test))
        mae_test = float(mean_absolute_error(y_test, y_pred_test))

        # Grouped CV — only meaningful with multiple distinct groups.
        r2_cv_mean: float | None = None
        r2_cv_std: float | None = None
        n_cv_folds_used: int | None = None
        if groups is not None and groups.nunique() >= 2:
            n_folds = int(min(MAX_GROUP_FOLDS, groups.nunique()))
            try:
                gkf = GroupKFold(n_splits=n_folds)
                fold_r2: list[float] = []
                X_arr = X.to_numpy()
                y_arr = y.to_numpy()
                g_arr = groups.to_numpy()
                for tr_idx, te_idx in gkf.split(X_arr, y_arr, groups=g_arr):
                    if len(te_idx) < 2:
                        continue
                    fold_model = RandomForestRegressor(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    fold_model.fit(X_arr[tr_idx], y_arr[tr_idx])
                    yp = fold_model.predict(X_arr[te_idx])
                    fold_r2.append(float(r2_score(y_arr[te_idx], yp)))
                if fold_r2:
                    r2_cv_mean = float(np.mean(fold_r2))
                    r2_cv_std = float(np.std(fold_r2))
                    n_cv_folds_used = int(len(fold_r2))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[LAND-IMPACT-ML][%s] GroupKFold failed: %s", label, exc)

        n_repeats = 5 if len(X_test) >= 200 else 10
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
        importance_rows = [
            {
                "feature": feature_cols[i],
                "permutation_importance_mean": float(perm.importances_mean[i]),
                "permutation_importance_std": float(perm.importances_std[i]),
                "rf_impurity_importance": float(model.feature_importances_[i]),
            }
            for i in range(len(feature_cols))
        ]
        importance_rows.sort(key=lambda r: r["permutation_importance_mean"], reverse=True)

        payload.update(
            {
                "status": "ok",
                "features_used": feature_cols,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "r2_train": r2_train,
                "r2_test": r2_test,
                "mae_test": mae_test,
                "r2_cv_mean": r2_cv_mean,
                "r2_cv_std": r2_cv_std,
                "n_cv_folds": n_cv_folds_used,
                "permutation_importance": importance_rows,
                "notes": "",
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "error"
        payload["notes"] = f"training failed: {exc!r}"

    return payload


def _persist(
    payload: dict[str, Any],
    reports_dir: Path,
    *,
    summary_filename: str,
    importance_filename: str,
    logger: logging.Logger,
) -> None:
    summary_path = reports_dir / summary_filename
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("[LAND-IMPACT-ML] Wrote %s", summary_path)

    if payload.get("permutation_importance"):
        fi_df = pd.DataFrame(payload["permutation_importance"])
    else:
        fi_df = pd.DataFrame(
            columns=[
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "rf_impurity_importance",
            ]
        )
    fi_path = reports_dir / importance_filename
    fi_df.to_csv(fi_path, index=False)
    logger.info("[LAND-IMPACT-ML] Wrote %s", fi_path)


def _format_r2(value: float | None) -> str:
    return "nan" if value is None or pd.isna(value) else f"{value:.4f}"


def run_land_impact_ml(
    features: pd.DataFrame,
    *,
    reports_dir: Path,
    target: str = DEFAULT_TARGET,
    feature_candidates: Iterable[str] | None = None,  # kept for backwards-compat
    test_size: float = 0.2,  # noqa: ARG001 — accepted for backwards-compat
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Train both maritime-only (primary) and S2-leakage-ceiling models.

    Returns the primary payload (so existing callers see the same shape they
    got before). The leakage-ceiling payload is persisted alongside but not
    returned.
    """
    _ = feature_candidates  # unused; kept so older callers don't break
    logger = logger or LOGGER
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if features.empty or target not in features.columns:
        primary = _empty_payload(target, "primary_maritime_only")
        primary["notes"] = f"target '{target}' missing or empty dataframe"
        _persist(
            primary,
            reports_dir,
            summary_filename="land_impact_ml_model_summary.json",
            importance_filename="land_impact_ml_feature_importance.csv",
            logger=logger,
        )
        return primary

    # Resolve feature columns for each tier from what's actually in the frame.
    primary_cols = _select_feature_columns(features, MARITIME_FEATURES, target)
    leakage_cols = _select_feature_columns(
        features, list(MARITIME_FEATURES) + list(S2_LEAKAGE_FEATURES), target
    )

    primary_payload = _train_one_model(
        features,
        primary_cols,
        target,
        "primary_maritime_only",
        random_state=random_state,
        logger=logger,
    )
    leakage_payload = _train_one_model(
        features,
        leakage_cols,
        target,
        "leakage_ceiling_with_s2_bands",
        random_state=random_state,
        logger=logger,
    )

    _persist(
        primary_payload,
        reports_dir,
        summary_filename="land_impact_ml_model_summary.json",
        importance_filename="land_impact_ml_feature_importance.csv",
        logger=logger,
    )
    _persist(
        leakage_payload,
        reports_dir,
        summary_filename="land_impact_ml_model_summary_with_s2_bands.json",
        importance_filename="land_impact_ml_feature_importance_with_s2_bands.csv",
        logger=logger,
    )

    logger.info(
        "[LAND-IMPACT-ML] primary: status=%s n=%s features=%d r2_test=%s r2_cv=%s±%s | leakage_ceiling: status=%s n=%s features=%d r2_test=%s r2_cv=%s±%s",
        primary_payload.get("status"),
        primary_payload.get("n_train"),
        len(primary_payload.get("features_used", []) or []),
        _format_r2(primary_payload.get("r2_test")),
        _format_r2(primary_payload.get("r2_cv_mean")),
        _format_r2(primary_payload.get("r2_cv_std")),
        leakage_payload.get("status"),
        leakage_payload.get("n_train"),
        len(leakage_payload.get("features_used", []) or []),
        _format_r2(leakage_payload.get("r2_test")),
        _format_r2(leakage_payload.get("r2_cv_mean")),
        _format_r2(leakage_payload.get("r2_cv_std")),
    )

    return primary_payload


__all__ = [
    "run_land_impact_ml",
    "DEFAULT_TARGET",
    "DEFAULT_FEATURE_CANDIDATES",
    "MARITIME_FEATURES",
    "S2_LEAKAGE_FEATURES",
]
