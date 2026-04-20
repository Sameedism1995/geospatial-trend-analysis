"""
Land Impact ML extension (LAND IMPACT EXTENSION LAYER, Step 5).

Trains a Random Forest regressor to predict NDVI (`ndvi_mean`) from maritime,
atmospheric, S1-disturbance, and water-quality features. Reports a held-out R²
plus **permutation importance** (sklearn built-in) as an interpretable feature
ranking.

This is an OPTIONAL, NON-BLOCKING analytical stage. If dependencies are missing,
if there is insufficient labelled data, or if the model fails to train, the
module logs and returns a structured empty payload rather than raising.

Outputs:
    outputs/reports/land_impact_ml_feature_importance.csv
    outputs/reports/land_impact_ml_model_summary.json
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
DEFAULT_FEATURE_CANDIDATES: tuple[str, ...] = (
    "vessel_density",
    "NO2_mean",
    "oil_slick_probability_t",
    "detection_score",
    "ndwi_mean",
    "ndti_mean",
    "ndci_mean",
    "fai_mean",
    "b11_mean",
    "coastal_exposure_score",
    "distance_to_nearest_high_vessel_density_cell",
    "maritime_pressure_index",
    "atmospheric_transfer_index",
    "vessel_x_no2",
    "vessel_x_ndvi_lag1",
    "vessel_x_ndvi_lag2",
    "vessel_x_ndvi_lag3",
)


def _select_feature_columns(df: pd.DataFrame, candidates: Iterable[str], target: str) -> list[str]:
    cols: list[str] = []
    for c in candidates:
        if c == target or c not in df.columns:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().sum() >= 50:
            cols.append(c)
    return cols


def _log_summary(logger: logging.Logger, payload: dict[str, Any]) -> None:
    logger.info(
        "[LAND-IMPACT-ML] status=%s | target=%s | n=%s | features=%d | r2_test=%s",
        payload.get("status"),
        payload.get("target"),
        payload.get("n_train"),
        len(payload.get("features_used", []) or []),
        payload.get("r2_test"),
    )


def run_land_impact_ml(
    features: pd.DataFrame,
    *,
    reports_dir: Path,
    target: str = DEFAULT_TARGET,
    feature_candidates: Iterable[str] = DEFAULT_FEATURE_CANDIDATES,
    test_size: float = 0.2,
    random_state: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Train RF on `target` and save permutation importance + summary."""
    logger = logger or LOGGER
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "status": "skipped",
        "target": target,
        "features_used": [],
        "n_train": None,
        "n_test": None,
        "r2_train": None,
        "r2_test": None,
        "mae_test": None,
        "permutation_importance": [],
        "notes": "",
    }

    if features.empty or target not in features.columns:
        payload["notes"] = f"target '{target}' missing or empty dataframe"
        _log_summary(logger, payload)
        _persist(payload, reports_dir, logger)
        return payload

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - environment-dependent
        payload["notes"] = f"scikit-learn unavailable: {exc!r}"
        _log_summary(logger, payload)
        _persist(payload, reports_dir, logger)
        return payload

    feature_cols = _select_feature_columns(features, feature_candidates, target)
    if len(feature_cols) < 2:
        payload["notes"] = f"insufficient feature columns: {feature_cols}"
        _log_summary(logger, payload)
        _persist(payload, reports_dir, logger)
        return payload

    y = pd.to_numeric(features[target], errors="coerce")
    X = features[feature_cols].apply(pd.to_numeric, errors="coerce")

    valid = y.notna() & X.notna().all(axis=1)
    X = X[valid]
    y = y[valid]
    if len(y) < 60:
        payload["notes"] = f"insufficient labelled rows after NA filter (n={len(y)})"
        payload["n_train"] = int(len(y))
        _log_summary(logger, payload)
        _persist(payload, reports_dir, logger)
        return payload

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(), y.to_numpy(), test_size=test_size, random_state=random_state
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

        payload.update({
            "status": "ok",
            "features_used": feature_cols,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_test": mae_test,
            "permutation_importance": importance_rows,
            "notes": "",
        })
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "error"
        payload["notes"] = f"training failed: {exc!r}"

    _log_summary(logger, payload)
    _persist(payload, reports_dir, logger)
    return payload


def _persist(payload: dict[str, Any], reports_dir: Path, logger: logging.Logger) -> None:
    summary_path = reports_dir / "land_impact_ml_model_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("[LAND-IMPACT-ML] Wrote %s", summary_path)

    if payload.get("permutation_importance"):
        fi_df = pd.DataFrame(payload["permutation_importance"])
    else:
        fi_df = pd.DataFrame(columns=[
            "feature",
            "permutation_importance_mean",
            "permutation_importance_std",
            "rf_impurity_importance",
        ])
    fi_path = reports_dir / "land_impact_ml_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info("[LAND-IMPACT-ML] Wrote %s", fi_path)


__all__ = ["run_land_impact_ml", "DEFAULT_TARGET", "DEFAULT_FEATURE_CANDIDATES"]
