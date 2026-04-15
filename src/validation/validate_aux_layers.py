"""
Read-only validation for NO₂ and Sentinel-1 oil auxiliary layers + cross-layer integrity.

Reads: modeling_dataset_human_impact.parquet (preferred) or modeling_dataset.parquet,
        data/aux/no2_grid_week.parquet, data/aux/sentinel1_oil_slicks.parquet (or oil_slicks.parquet),
        optional data/aux/no2_gee_validation.json

Never modifies data/aux/* or modeling outputs. Writes only JSON under data/validation/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def _playground_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_context_panel(root: Path) -> tuple[pd.DataFrame | None, str]:
    hip = root / "data" / "modeling_dataset_human_impact.parquet"
    mod = root / "data" / "modeling_dataset.parquet"
    if hip.exists():
        df = pd.read_parquet(hip)
        df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
        return df, "modeling_dataset_human_impact.parquet"
    if mod.exists():
        df = pd.read_parquet(mod)
        df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
        return df, "modeling_dataset.parquet"
    return None, "none"


def resolve_oil_parquet(root: Path, explicit: Path | None) -> Path:
    """Prefer sentinel1_oil_slicks.parquet; fall back to oil_slicks.parquet."""
    if explicit is not None:
        return explicit
    p1 = root / "data" / "aux" / "sentinel1_oil_slicks.parquet"
    p2 = root / "data" / "aux" / "oil_slicks.parquet"
    if p1.exists():
        return p1
    return p2


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(m) < 5:
        return None, None
    if scipy_stats:
        r, p = scipy_stats.spearmanr(m["x"], m["y"])
        return float(r), float(p)
    r = m["x"].corr(m["y"], method="spearman")
    return float(r) if not math.isnan(r) else None, None


def validate_no2(
    root: Path,
    no2_parquet: Path,
    no2_meta: Path,
) -> dict[str, Any]:
    flags: list[str] = []
    inputs = {
        "no2_parquet": str(no2_parquet),
        "no2_gee_validation_json": str(no2_meta),
        "context_panel": "modeling_dataset_human_impact.parquet (preferred) or modeling_dataset.parquet",
    }

    if not no2_parquet.exists():
        return {
            "status": "skipped",
            "coverage_percent": None,
            "distribution": {},
            "temporal": {},
            "spatial": {},
            "flags": ["no_data"],
            "summary": f"NO₂ parquet not found: {no2_parquet}",
            "inputs": inputs,
        }

    try:
        no2 = pd.read_parquet(no2_parquet)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "coverage_percent": None,
            "distribution": {},
            "temporal": {},
            "spatial": {},
            "flags": ["read_error"],
            "summary": str(e),
            "inputs": inputs,
        }

    no2["week_start_utc"] = pd.to_datetime(no2["week_start_utc"], utc=True, errors="coerce")
    meta: dict[str, Any] = {}
    if no2_meta.exists():
        with no2_meta.open(encoding="utf-8") as f:
            meta = json.load(f)

    n_rows = len(no2)
    n_valid = int(no2["no2_mean_t"].notna().sum()) if "no2_mean_t" in no2.columns else 0
    cov_recalc = 100.0 * n_valid / n_rows if n_rows else 0.0
    cov_meta = meta.get("coverage_percent")
    missing_per_grid = meta.get("missing_weeks_per_grid", {})

    if cov_recalc < 50.0 and n_rows > 0:
        flags.append("low_coverage")
    if n_valid == 0:
        flags.append("no_data")
    if isinstance(missing_per_grid, dict) and len(missing_per_grid) > 1:
        vals = list(missing_per_grid.values())
        if len(set(vals)) == 1 and n_rows > 0:
            flags.append("systemic_failure")

    gmean = float(no2["no2_mean_t"].mean(skipna=True)) if n_valid else None
    gstd = float(no2["no2_mean_t"].std(skipna=True)) if n_valid > 1 else None
    nu = int(no2["no2_mean_t"].nunique(dropna=True)) if n_valid else 0

    distribution = {
        "nunique_no2_mean_t": nu,
        "global_std": gstd,
        "global_mean": gmean,
        "metadata_coverage_percent": cov_meta,
        "total_grid_week_rows": n_rows,
        "non_null_no2_rows": n_valid,
    }
    if gstd is not None and gstd < 1e-12 and n_valid > 1:
        flags.append("constant_field")
    if nu <= 1 and n_valid > 0:
        flags.append("degenerate_distribution")

    var_by_grid = (
        no2.groupby("grid_cell_id", sort=False)["no2_mean_t"].var()
        if "grid_cell_id" in no2.columns
        else pd.Series(dtype=float)
    )
    zero_var = var_by_grid[var_by_grid.fillna(0) == 0].index.astype(str).tolist()
    n_grids = int(var_by_grid.shape[0])
    frac_zero_var = len(zero_var) / n_grids if n_grids else 0.0
    if n_grids and frac_zero_var > 0.5:
        flags.append("temporal_freeze")

    temporal = {
        "variance_statistics": {
            "min": float(var_by_grid.min()) if len(var_by_grid) else None,
            "max": float(var_by_grid.max()) if len(var_by_grid) else None,
            "mean": float(var_by_grid.mean()) if len(var_by_grid) else None,
            "median": float(var_by_grid.median()) if len(var_by_grid) else None,
        },
        "n_grids": n_grids,
        "n_grids_zero_variance": len(zero_var),
        "grids_zero_variance_sample_max20": zero_var[:20],
    }

    ctx, ctx_name = load_context_panel(root)
    spatial: dict[str, Any] = {"context_source": ctx_name, "skipped": True}
    if ctx is not None and "distance_to_coast_km" in ctx.columns and n_valid > 0:
        merged = no2.merge(
            ctx[["grid_cell_id", "week_start_utc", "distance_to_coast_km"]].drop_duplicates(),
            on=["grid_cell_id", "week_start_utc"],
            how="inner",
        )
        sub = merged.dropna(subset=["no2_mean_t", "distance_to_coast_km"])
        rho, pval = _spearman(sub["distance_to_coast_km"], sub["no2_mean_t"])
        spatial = {
            "skipped": False,
            "spearman_rho_distance_vs_no2": rho,
            "spearman_p_value": pval,
            "n_aligned_rows": int(len(sub)),
            "expected": "negative rho (higher NO₂ nearer coast / lower distance)",
        }
        if rho is None or abs(rho) < 0.05:
            flags.append("no_spatial_structure")
        if rho is not None and rho > 0.05:
            flags.append("unexpected_spatial_pattern")
    else:
        spatial["reason"] = "distance_to_coast_km not in context panel"

    status = "ok"
    if n_valid == 0 and n_rows > 0:
        status = "failed"

    summary = (
        f"NO₂ validation: {n_valid}/{n_rows} non-null rows ({cov_recalc:.2f}% coverage). "
        f"Distribution nunique={nu}. "
        f"Spatial ρ(dist,NO₂)={spatial.get('spearman_rho_distance_vs_no2') if isinstance(spatial, dict) else 'n/a'}. "
        f"Flags: {flags or 'none'}."
    )

    return {
        "status": status,
        "coverage_percent": round(cov_recalc, 4),
        "distribution": distribution,
        "temporal": temporal,
        "spatial": spatial,
        "flags": sorted(set(flags)),
        "summary": summary,
        "inputs": inputs,
    }


def validate_oil(
    root: Path,
    oil_parquet: Path,
    oil_meta: Path,
) -> dict[str, Any]:
    flags: list[str] = []
    inputs = {
        "oil_parquet": str(oil_parquet),
        "metadata_json_optional": str(oil_meta),
    }

    if not oil_parquet.exists():
        return {
            "status": "skipped",
            "plausibility": {},
            "temporal": {},
            "spatial": {},
            "distribution": {},
            "event_sparsity_ratio": None,
            "nonzero_oil_probability_fraction": None,
            "spatial_clustering_score_vs_shipping": None,
            "anomaly_flags": ["no_data"],
            "summary": f"Oil parquet not found: {oil_parquet}",
            "inputs": inputs,
        }

    try:
        oil = pd.read_parquet(oil_parquet)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "failed",
            "plausibility": {},
            "temporal": {},
            "spatial": {},
            "distribution": {},
            "event_sparsity_ratio": None,
            "nonzero_oil_probability_fraction": None,
            "spatial_clustering_score_vs_shipping": None,
            "anomaly_flags": ["read_error"],
            "summary": str(e),
            "inputs": inputs,
        }

    oil["week_start_utc"] = pd.to_datetime(oil["week_start_utc"], utc=True, errors="coerce")
    meta: dict[str, Any] = {}
    if oil_meta.exists():
        with oil_meta.open(encoding="utf-8") as f:
            meta = json.load(f)

    prob = pd.to_numeric(oil.get("oil_slick_probability_t"), errors="coerce")
    n = len(oil)
    n_zero = int(((prob.fillna(0) == 0) & prob.notna()).sum())
    frac_zero = float(n_zero / n) if n else 0.0
    frac_hi = float((prob > 0.5).sum()) / n if n else 0.0
    mean_p = float(prob.mean(skipna=True)) if prob.notna().any() else None
    nu = int(prob.nunique(dropna=True))
    sat = float((prob > 0.9).sum()) / n if n else 0.0

    high_sparsity_ok = bool(frac_zero >= 0.4) if n else False
    plausibility = {
        "mean_oil_slick_probability_t": mean_p,
        "fraction_zero_values": round(frac_zero, 4),
        "fraction_above_0_5": round(frac_hi, 4),
        "high_sparsity_ok": high_sparsity_ok,
        "note": "Dark-pixel proxy should be sparse (mostly zeros) — flag if overdense",
    }
    if n > 100 and frac_zero < 0.35:
        flags.append("insufficient_sparsity")
    if frac_hi > 0.5 and n > 50:
        flags.append("overdense_signal")
    mean_by_grid = oil.groupby("grid_cell_id")["oil_slick_probability_t"].mean()
    if mean_by_grid.nunique(dropna=True) <= 1 and len(mean_by_grid) > 3 and prob.notna().any():
        flags.append("constant_field")

    var_by_grid = (
        oil.groupby("grid_cell_id", sort=False)["oil_slick_probability_t"].var()
        if "grid_cell_id" in oil.columns
        else pd.Series(dtype=float)
    )
    zv = var_by_grid[var_by_grid.fillna(0) == 0].index.astype(str).tolist()
    ng = int(var_by_grid.shape[0])
    if ng and len(zv) / ng > 0.7:
        flags.append("no_temporal_variation")

    temporal = {
        "n_grids": ng,
        "n_grids_zero_variance": len(zv),
        "variance_across_weeks_note": "SAR sampling is discontinuous; many zero-variance grids can be normal",
    }

    ctx, _ = load_context_panel(root)
    spatial: dict[str, Any] = {"skipped": True}
    rho_v: float | None = None
    if ctx is not None and "vessel_density_t" in ctx.columns and n > 0:
        m = oil.merge(
            ctx[["grid_cell_id", "week_start_utc", "vessel_density_t"]],
            on=["grid_cell_id", "week_start_utc"],
            how="inner",
        )
        m = m.dropna(subset=["oil_slick_probability_t", "vessel_density_t"])
        rho_v, pval = _spearman(m["vessel_density_t"], m["oil_slick_probability_t"])
        uniform = mean_by_grid.nunique(dropna=True) <= 1 and len(mean_by_grid) > 3
        spatial = {
            "skipped": False,
            "spearman_rho_vessel_vs_oil_probability": rho_v,
            "spearman_p_value": pval,
            "n_aligned_rows": int(len(m)),
            "mean_oil_prob_nunique_across_grids": int(mean_by_grid.nunique(dropna=True)),
        }
        if rho_v is not None and abs(rho_v) < 0.05 and len(m) > 30:
            flags.append("no_spatial_signal")
        if uniform:
            flags.append("spatial_flatness")
    else:
        spatial["reason"] = "vessel_density_t not available in context panel"

    distribution = {
        "nunique_oil_slick_probability_t": nu,
        "saturation_fraction_above_0_9": round(sat, 4),
        "metadata_from_json": {k: meta.get(k) for k in ("oil_coverage_percent", "rows_with_oil_data")},
    }
    if nu <= 3 and n > 20 and prob.notna().any():
        flags.append("quantization_artifact")

    event_sparsity_ratio = round(frac_zero, 4) if n else None
    nonzero_frac = round(1.0 - frac_zero, 4) if n else None

    summary = (
        f"Oil proxy: mean p={mean_p}, frac_zero={frac_zero:.3f}, nunique={nu}. "
        f"Vessel–oil Spearman ρ={rho_v}. Flags: {flags or 'none'}."
    )

    status = "ok"
    if n == 0 or (prob.notna().sum() == 0):
        status = "failed"

    return {
        "status": status,
        "plausibility": plausibility,
        "temporal": temporal,
        "spatial": spatial,
        "distribution": distribution,
        "event_sparsity_ratio": event_sparsity_ratio,
        "nonzero_oil_probability_fraction": nonzero_frac,
        "spatial_clustering_score_vs_shipping": rho_v,
        "anomaly_flags": sorted(set(flags)),
        "summary": summary,
        "inputs": inputs,
    }


def integrated_validation(
    root: Path,
    no2_parquet: Path,
    oil_parquet: Path,
) -> dict[str, Any]:
    flags: list[str] = []
    ctx, ctx_name = load_context_panel(root)
    if ctx is None:
        return {
            "status": "skipped",
            "spearman_pairs": {},
            "correlation_matrix_spearman": {},
            "integrity_flags": ["no_context_panel"],
            "scientific_validity_score_heuristic": None,
            "ready_for_thesis_discussion": False,
            "science_gate": {"passes_integrity_rules": False, "conditions": {}},
            "summary": "No modeling_dataset_human_impact.parquet or modeling_dataset.parquet found.",
        }

    parts = []
    if no2_parquet.exists():
        no2 = pd.read_parquet(no2_parquet)
        no2["week_start_utc"] = pd.to_datetime(no2["week_start_utc"], utc=True, errors="coerce")
        parts.append(no2[["grid_cell_id", "week_start_utc", "no2_mean_t"]].rename(columns={"no2_mean_t": "no2"}))
    if oil_parquet.exists():
        oil = pd.read_parquet(oil_parquet)
        oil["week_start_utc"] = pd.to_datetime(oil["week_start_utc"], utc=True, errors="coerce")
        parts.append(
            oil[["grid_cell_id", "week_start_utc", "oil_slick_probability_t"]].rename(
                columns={"oil_slick_probability_t": "oil_prob"}
            )
        )

    if not parts:
        return {
            "status": "skipped",
            "spearman_pairs": {},
            "correlation_matrix_spearman": {},
            "integrity_flags": ["no_auxiliary_parquets"],
            "scientific_validity_score_heuristic": None,
            "ready_for_thesis_discussion": False,
            "science_gate": {"passes_integrity_rules": False, "conditions": {}},
            "summary": "Neither NO₂ nor oil parquet present.",
        }

    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p, on=["grid_cell_id", "week_start_utc"], how="outer")

    ctx_slim = ctx[
        [c for c in ("grid_cell_id", "week_start_utc", "vessel_density_t", "distance_to_coast_km") if c in ctx.columns]
    ].copy()
    m = m.merge(ctx_slim, on=["grid_cell_id", "week_start_utc"], how="inner")

    spearman_pairs: dict[str, Any] = {}
    for a, b, label in (
        ("distance_to_coast_km", "no2", "distance_to_coast_km_vs_no2"),
        ("vessel_density_t", "oil_prob", "vessel_density_t_vs_oil_slick_probability_t"),
        ("no2", "oil_prob", "no2_vs_oil_slick_probability_t"),
    ):
        if a not in m.columns or b not in m.columns:
            continue
        sub = m[[a, b]].dropna()
        if len(sub) < 8:
            spearman_pairs[label] = {"n": len(sub), "spearman_rho": None}
            continue
        rho, pval = _spearman(sub[a], sub[b])
        spearman_pairs[label] = {"n": len(sub), "spearman_rho": rho, "p_value": pval}

    pair_ns = [int(v.get("n", 0)) for v in spearman_pairs.values() if isinstance(v, dict)]
    if not pair_ns or max(pair_ns) < 8:
        flags.append("no_cross_layer_observations")

    cm_cols = [c for c in ("distance_to_coast_km", "no2", "oil_prob", "vessel_density_t") if c in m.columns]
    corr_matrix: dict[str, Any] = {}
    if len(cm_cols) >= 2 and len(m.dropna(how="all")) >= 5:
        cm = m[cm_cols].dropna()
        if len(cm) >= 5:
            corr_matrix = cm.corr(method="spearman").round(4).to_dict()

    # Integrity flags
    rhos_abs = []
    for v in spearman_pairs.values():
        r = v.get("spearman_rho") if isinstance(v, dict) else None
        if r is not None:
            rhos_abs.append(abs(r))
    if any(abs(v.get("spearman_rho", 0) or 0) > 0.95 for v in spearman_pairs.values() if isinstance(v, dict)):
        flags.append("suspected_leakage")
    if len(rhos_abs) >= 3 and all(x < 0.05 for x in rhos_abs):
        flags.append("weak_signal_structure")
    if len(cm_cols) >= 3 and corr_matrix:
        # high average |ρ| off-diagonal
        vals = []
        for ki, row in corr_matrix.items():
            for kj, val in row.items():
                if ki != kj and val is not None and not (isinstance(val, float) and math.isnan(val)):
                    vals.append(abs(float(val)))
        if len(vals) > 3 and np.mean(vals) > 0.85:
            flags.append("feature_collapse")

    # Score [0,1]
    score = 1.0
    dn = spearman_pairs.get("distance_to_coast_km_vs_no2", {})
    r_dn = dn.get("spearman_rho") if isinstance(dn, dict) else None
    vo = spearman_pairs.get("vessel_density_t_vs_oil_slick_probability_t", {})
    r_vo = vo.get("spearman_rho") if isinstance(vo, dict) else None
    no = spearman_pairs.get("no2_vs_oil_slick_probability_t", {})
    r_no = no.get("spearman_rho") if isinstance(no, dict) else None

    flat = True
    for col in ("no2", "oil_prob", "vessel_density_t"):
        if col in m.columns and m[col].notna().sum() > 5 and m[col].nunique(dropna=True) > 2:
            flat = False
    if flat:
        score -= 0.35
    if "no_cross_layer_observations" in flags:
        score -= 0.5
    if "suspected_leakage" in flags:
        score -= 0.3
    if "feature_collapse" in flags:
        score -= 0.2
    if r_dn is not None and r_dn > 0.08:
        score -= 0.12
    if r_vo is not None and r_vo < -0.4:
        score -= 0.08
    if r_no is not None and abs(r_no) > 0.85:
        score -= 0.12
        flags.append("no2_oil_correlation_unexpectedly_high")
    elif r_no is not None and abs(r_no) > 0.75:
        score -= 0.06
    score = max(0.0, min(1.0, score))

    critical_failure = (
        "no_cross_layer_observations" in flags
        or "suspected_leakage" in flags
        or "feature_collapse" in flags
        or "no2_oil_correlation_unexpectedly_high" in flags
    )
    ready = bool(score >= 0.55 and not critical_failure)

    science_gate = {
        "passes_integrity_rules": ready,
        "conditions": {
            "scientific_validity_score_ge_0_55": score >= 0.55,
            "no_suspected_leakage": "suspected_leakage" not in flags,
            "no_feature_collapse": "feature_collapse" not in flags,
            "no_excessive_no2_oil_coupling": "no2_oil_correlation_unexpectedly_high" not in flags,
            "no2_vs_coast_negative_or_near_zero_ok": r_dn is None or r_dn <= 0.05,
            "oil_vs_vessel_not_strongly_negative": r_vo is None or r_vo > -0.5,
        },
    }

    summary = (
        f"Integrity: context={ctx_name}, merged n={len(m)}. "
        f"ρ(dist,NO₂)={r_dn}, ρ(vessel,oil)={r_vo}, ρ(NO₂,oil)={r_no}. "
        f"Score={score:.3f}. ready_for_thesis_discussion={ready}. Flags={sorted(set(flags))}."
    )

    return {
        "status": "ok",
        "context_panel": ctx_name,
        "aligned_grid_week_rows": int(len(m)),
        "spearman_pairs": spearman_pairs,
        "correlation_matrix_spearman": corr_matrix,
        "integrity_flags": sorted(set(flags)),
        "scientific_validity_score_heuristic": round(score, 4),
        "ready_for_thesis_discussion": ready,
        "science_gate": science_gate,
        "summary": summary,
        "inputs": {
            "no2_parquet": str(no2_parquet),
            "oil_parquet": str(oil_parquet),
        },
    }


def main() -> None:
    root = _playground_root()
    p = argparse.ArgumentParser(description="Validate NO₂ + oil aux layers (read-only)")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument("--no2-parquet", type=Path, default=root / "data" / "aux" / "no2_grid_week.parquet")
    p.add_argument("--no2-meta", type=Path, default=root / "data" / "aux" / "no2_gee_validation.json")
    p.add_argument(
        "--oil-parquet",
        type=Path,
        default=None,
        help="Default: data/aux/sentinel1_oil_slicks.parquet if present else oil_slicks.parquet",
    )
    p.add_argument("--oil-meta", type=Path, default=root / "data" / "aux" / "sentinel1_oil_validation.json")
    p.add_argument("--out-dir", type=Path, default=root / "data" / "validation")
    args = p.parse_args()
    root = args.root

    no2_path = args.no2_parquet if args.no2_parquet.is_absolute() else root / args.no2_parquet
    no2_meta = args.no2_meta if args.no2_meta.is_absolute() else root / args.no2_meta
    if args.oil_parquet is None:
        oil_path = resolve_oil_parquet(root, None)
    else:
        oil_path = args.oil_parquet if args.oil_parquet.is_absolute() else root / args.oil_parquet
    oil_meta = args.oil_meta if args.oil_meta.is_absolute() else root / args.oil_meta

    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    r_no2 = validate_no2(root, no2_path, no2_meta)
    r_oil = validate_oil(root, oil_path, oil_meta)
    r_int = integrated_validation(root, no2_path, oil_path)

    for path, obj in (
        (out_dir / "no2_validation_report.json", r_no2),
        (out_dir / "sentinel1_oil_validation_report.json", r_oil),
        (out_dir / "human_impact_integrity_report.json", r_int),
    ):
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
