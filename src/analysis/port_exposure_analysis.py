"""Aggregate port exposure metrics by nearest named port (thesis summaries)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("port_exposure_analysis")


def generate_port_summary(
    df: pd.DataFrame,
    project_root: Path,
    *,
    coastal_impact_csv: Path | None = None,
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Group by nearest_port; write outputs/reports/port_city_exposure_summary.csv.

    Optionally left-merge coastal_impact_score from coastal_impact_score.csv if provided.
    """
    log = logger or LOGGER
    out_path = output_path or (project_root / "outputs" / "reports" / "port_city_exposure_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or "nearest_port" not in df.columns:
        log.warning("[PORT SUMMARY] Empty frame or missing nearest_port; writing empty summary.")
        empty = pd.DataFrame(
            columns=[
                "nearest_port",
                "mean_vessel_density",
                "mean_NO2_mean",
                "mean_coastal_impact_score",
                "mean_port_exposure_score",
                "n_grid_week_rows",
            ]
        )
        empty.to_csv(out_path, index=False)
        return empty

    work = df.copy()
    work["nearest_port"] = work["nearest_port"].astype(str)

    if coastal_impact_csv is not None and coastal_impact_csv.exists():
        try:
            cis = pd.read_csv(coastal_impact_csv)
            if {"grid_cell_id", "week_start_utc", "coastal_impact_score"}.issubset(cis.columns):
                cis["week_start_utc"] = pd.to_datetime(cis["week_start_utc"], utc=True, errors="coerce")
                work["week_start_utc"] = pd.to_datetime(work["week_start_utc"], utc=True, errors="coerce")
                work = work.merge(
                    cis[["grid_cell_id", "week_start_utc", "coastal_impact_score"]],
                    on=["grid_cell_id", "week_start_utc"],
                    how="left",
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("[PORT SUMMARY] Could not merge coastal impact: %s", exc)

    g = work.groupby("nearest_port", dropna=False)
    out = pd.DataFrame({"n_grid_week_rows": g.size()})

    vcol = next((c for c in ("vessel_density", "vessel_density_t") if c in work.columns), None)
    if vcol:
        out["mean_vessel_density"] = g[vcol].mean()
    else:
        out["mean_vessel_density"] = pd.NA

    if "NO2_mean" in work.columns:
        out["mean_NO2_mean"] = g["NO2_mean"].mean()
    else:
        out["mean_NO2_mean"] = pd.NA

    if "coastal_impact_score" in work.columns:
        out["mean_coastal_impact_score"] = g["coastal_impact_score"].mean()
    else:
        out["mean_coastal_impact_score"] = pd.NA

    if "port_exposure_score" in work.columns:
        out["mean_port_exposure_score"] = g["port_exposure_score"].mean()
    else:
        out["mean_port_exposure_score"] = pd.NA

    out = out.reset_index()
    col_order = [
        "nearest_port",
        "mean_vessel_density",
        "mean_NO2_mean",
        "mean_coastal_impact_score",
        "mean_port_exposure_score",
        "n_grid_week_rows",
    ]
    out = out[[c for c in col_order if c in out.columns]]
    out.to_csv(out_path, index=False)
    log.info("[PORT SUMMARY] Wrote %s (%d ports)", out_path, len(out))
    return out
