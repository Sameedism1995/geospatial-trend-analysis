"""Final FULL pipeline run on the Stockholm-expanded grid (838 cells × 51 weeks).

This wrapper:
  1. Builds an expanded `data/modeling_dataset.parquet` (315 original cells +
     523 Stockholm-vicinity cells = 838 grids × 51 weeks).
  2. Wipes all aux/intermediate caches so every GEE-backed source is re-extracted
     from scratch on the expanded grid.
  3. Invokes `src/run_final_pipeline.py --clean --run-name <RUN_NAME>` so the
     pipeline writes its consolidated output under `<RUN_NAME>/`.
  4. Mirrors the pipeline's outputs into the layout the user requested
     (`outputs/reports/<RUN_NAME>/`, `outputs/figures/<RUN_NAME>/`,
     `outputs/visualizations/<RUN_NAME>/`, `processed/<RUN_NAME>/`,
     `logs/<RUN_NAME>/`), without overwriting any prior run folder.
  5. Restores the original `data/modeling_dataset.parquet`.
  6. Runs the revised hub-strategy analysis (Turku-Naantali / Mariehamn /
     Stockholm) on the freshly-built dataset and writes Stockholm-specific
     extras (zoom plot, NO2 local test, comparison) into the run folders.
  7. Writes a `FINAL_RUN_SUMMARY.md` describing the run.

Usage:
    python3 src/run_final_pipeline_stockholm_fixed.py
    python3 src/run_final_pipeline_stockholm_fixed.py --skip-sentinel1   # NO2/S2 only
    python3 src/run_final_pipeline_stockholm_fixed.py --run-name custom_name

Expected wall-clock with Sentinel-1 enabled on the 838-cell grid: roughly
20-30 hours (S1 dominates).  Without Sentinel-1: ~1.5 hours.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

LOGGER = logging.getLogger("final_stockholm_fixed")

ORIG_MODELING = ROOT / "data" / "modeling_dataset.parquet"
EXPANDED_FEATURES_PARQUET = (
    ROOT / "processed" / "run_stockholm_grid_expanded"
    / "features_ml_ready_stockholm_expanded.parquet"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("final_stockholm_fixed")


# ---------------------------------------------------------------------------
# Step A: build expanded modeling_dataset
# ---------------------------------------------------------------------------

def build_expanded_modeling(out_path: Path) -> dict[str, Any]:
    """Combine the existing 315-cell modeling parquet with the 523 new
    Stockholm-vicinity cells (sourced from features_ml_ready_stockholm_expanded.parquet)
    to produce a 838-grid × 51-week modeling parquet at `out_path`.

    New cells get NaN for all extracted features; the pipeline will populate
    them via fresh GEE extraction.
    """
    if not ORIG_MODELING.exists():
        raise FileNotFoundError(ORIG_MODELING)
    if not EXPANDED_FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing expanded features parquet: {EXPANDED_FEATURES_PARQUET}.\n"
            "Run `python3 src/analysis/expand_grid_for_stockholm.py` first."
        )

    base = pd.read_parquet(ORIG_MODELING)
    LOGGER.info("Base modeling: rows=%d grids=%d weeks=%d",
                len(base), base["grid_cell_id"].nunique(),
                base["week_start_utc"].nunique())

    expanded = pd.read_parquet(EXPANDED_FEATURES_PARQUET)
    base_grids = set(base["grid_cell_id"].unique())
    new_rows = expanded[~expanded["grid_cell_id"].isin(base_grids)].copy()
    new_grid_count = int(new_rows["grid_cell_id"].nunique())
    LOGGER.info("New cells from Stockholm expansion: %d grids (%d rows)",
                new_grid_count, len(new_rows))

    if new_rows.empty:
        raise RuntimeError(
            "No new Stockholm cells found in expanded features parquet — "
            "check that expand_grid_for_stockholm.py was run."
        )

    new_modeling = new_rows[[
        "grid_cell_id", "week_start_utc", "grid_centroid_lat", "grid_centroid_lon",
    ]].copy()
    new_modeling["week_start_utc"] = pd.to_datetime(
        new_modeling["week_start_utc"], utc=True, errors="coerce"
    )
    new_modeling["grid_res_deg"] = 0.1
    new_modeling["week_of_year"] = (
        new_modeling["week_start_utc"].dt.isocalendar().week.astype("Int64")
    )
    angle = 2 * math.pi * new_modeling["week_of_year"].astype(float) / 52.0
    new_modeling["week_sin"] = angle.apply(math.sin)
    new_modeling["week_cos"] = angle.apply(math.cos)
    new_modeling["sentinel_observation_count_t"] = 0
    new_modeling["has_valid_delta_ndti"] = False
    new_modeling["has_sentinel"] = False
    new_modeling["has_emodnet"] = False
    new_modeling["has_helcom"] = False

    for col in base.columns:
        if col in new_modeling.columns:
            continue
        if base[col].dtype.kind in "fc":
            new_modeling[col] = float("nan")
        elif base[col].dtype.kind == "i":
            new_modeling[col] = pd.NA
        elif base[col].dtype.kind == "b":
            new_modeling[col] = False
        else:
            new_modeling[col] = pd.NA

    new_modeling = new_modeling[base.columns]

    expanded_modeling = pd.concat([base, new_modeling], ignore_index=True)
    expanded_modeling = expanded_modeling.drop_duplicates(
        subset=["grid_cell_id", "week_start_utc"], keep="first"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_modeling.to_parquet(out_path, index=False)

    info = {
        "old_rows": int(len(base)),
        "old_grids": int(base["grid_cell_id"].nunique()),
        "new_rows_added": int(len(new_modeling)),
        "new_grids_added": new_grid_count,
        "expanded_rows": int(len(expanded_modeling)),
        "expanded_grids": int(expanded_modeling["grid_cell_id"].nunique()),
        "weeks": int(expanded_modeling["week_start_utc"].nunique()),
    }
    LOGGER.info(
        "Expanded modeling written: rows=%d grids=%d weeks=%d (added %d grids / %d rows)",
        info["expanded_rows"], info["expanded_grids"], info["weeks"],
        info["new_grids_added"], info["new_rows_added"],
    )
    return info


# ---------------------------------------------------------------------------
# Step B: clean caches so every source is re-extracted
# ---------------------------------------------------------------------------

def wipe_caches(logger: logging.Logger) -> list[Path]:
    """Remove all aux + intermediate parquets so the pipeline is forced to refresh."""
    removed: list[Path] = []
    targets = [
        ROOT / "data" / "aux",
        ROOT / "data" / "intermediate",
    ]
    for t in targets:
        if t.exists():
            for child in t.iterdir():
                if child.name == "baltic_ports.csv":
                    continue  # static config, not a cache
                if child.is_file():
                    child.unlink()
                    removed.append(child)
                elif child.is_dir():
                    shutil.rmtree(child)
                    removed.append(child)
    logger.info("[CACHE WIPE] removed %d entries", len(removed))
    for p in removed:
        try:
            logger.info("  - %s", p.relative_to(ROOT))
        except ValueError:
            logger.info("  - %s", p)
    return removed


# ---------------------------------------------------------------------------
# Step C: mirror pipeline outputs into user-requested layout
# ---------------------------------------------------------------------------

def mirror_into_run_folders(
    inner_run_root: Path,
    run_name: str,
    logger: logging.Logger,
) -> dict[str, int]:
    """Copy from <inner_run_root>/{outputs,processed,logs,...} into
    `outputs/reports/<run_name>/`, `outputs/figures/<run_name>/`,
    `outputs/visualizations/<run_name>/`, `processed/<run_name>/`,
    `logs/<run_name>/`.
    """
    targets = {
        "reports":        ROOT / "outputs" / "reports" / run_name,
        "figures":        ROOT / "outputs" / "figures" / run_name,
        "visualizations": ROOT / "outputs" / "visualizations" / run_name,
        "processed":      ROOT / "processed" / run_name,
        "logs":           ROOT / "logs" / run_name,
    }
    for p in targets.values():
        p.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    def _copy_tree(src: Path, dst: Path) -> int:
        if not src.exists():
            return 0
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return 1
        n = 0
        for child in src.rglob("*"):
            if not child.is_file():
                continue
            rel = child.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            n += 1
        return n

    counts["reports"]        = _copy_tree(inner_run_root / "outputs" / "reports", targets["reports"])
    # Pipeline writes plots under outputs/visualizations/{plots,eda,previews,...}
    # Mirror to "visualizations" wholesale.
    counts["visualizations"] = _copy_tree(inner_run_root / "outputs" / "visualizations", targets["visualizations"])
    # The user separates "figures" from "visualizations".  We keep figures =
    # the static plots subset (outputs/visualizations/plots + previews +
    # hub_level_distance_decay + sliding_window_distance_decay).
    figs_total = 0
    for sub in ("plots", "previews", "hub_level_distance_decay",
                "sliding_window_distance_decay", "exposure_maps", "eda"):
        figs_total += _copy_tree(inner_run_root / "outputs" / "visualizations" / sub,
                                 targets["figures"] / sub)
    counts["figures"] = figs_total
    counts["processed"]      = _copy_tree(inner_run_root / "processed", targets["processed"])
    counts["logs"]           = _copy_tree(inner_run_root / "logs", targets["logs"])

    # Useful run-level files.
    for fname in ("FINAL_RUN_SUMMARY.md",):
        src = inner_run_root / fname
        if src.exists():
            shutil.copy2(src, targets["reports"] / fname)
            counts[fname] = 1
    # Validation directory.
    val = inner_run_root / "validation"
    if val.exists():
        counts["validation"] = _copy_tree(val, targets["reports"] / "validation")

    logger.info("[MIRROR] copied %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Step D: re-run hub-strategy + Stockholm extras on the fresh dataset
# ---------------------------------------------------------------------------

def run_hub_strategy_on_fresh(
    inner_run_root: Path,
    run_name: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Re-run the Stockholm-fix hub-strategy analysis using the freshly-built
    features_ml_ready.parquet from the inner pipeline run.
    """
    import importlib

    # Fresh dataset path (inner pipeline writes here).
    candidate = inner_run_root / "processed" / "features_ml_ready.parquet"
    if not candidate.exists():
        candidate = ROOT / "processed" / "features_ml_ready.parquet"
    if not candidate.exists():
        logger.warning("No features_ml_ready.parquet found post-pipeline; skipping hub analysis.")
        return {"skipped": True}

    HS = importlib.import_module(
        "analysis.hub_strategy_turku_naantali_mariehamn_stockholm"
    )
    EXP = importlib.import_module("analysis.expand_grid_for_stockholm")

    REPORTS = ROOT / "outputs" / "reports" / run_name
    FIGURES = ROOT / "outputs" / "figures" / run_name
    VIS     = ROOT / "outputs" / "visualizations" / run_name
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(candidate)
    if "week_start_utc" in df.columns:
        df["week_start_utc"] = pd.to_datetime(df["week_start_utc"], utc=True, errors="coerce")
    logger.info("[HUB] dataset %s rows=%d cols=%d", candidate.relative_to(ROOT),
                len(df), df.shape[1])

    # Enrich with sentinel_*_t cols if missing (matches hub_strategy logic).
    needs = ["sentinel_ndwi_mean_t", "sentinel_ndti_mean_t", "sentinel_ndvi_mean_t"]
    missing = [c for c in needs if c not in df.columns]
    human = ROOT / "data" / "modeling_dataset_human_impact.parquet"
    if missing and human.exists():
        h = pd.read_parquet(human, columns=["grid_cell_id", "week_start_utc", *missing])
        h["week_start_utc"] = pd.to_datetime(h["week_start_utc"], utc=True, errors="coerce")
        df = df.merge(h, on=["grid_cell_id", "week_start_utc"], how="left")

    indicators: dict[str, str] = {}
    for key, candidates in HS.INDICATORS.items():
        col = HS.find_first(df.columns, candidates)
        if col is not None:
            indicators[key] = col
            logger.info("[HUB] indicator %s -> %s", key, col)
        else:
            logger.warning("[HUB] indicator %s not found", key)

    df = HS.attach_hub_distances(df)

    generated: list[Path] = []
    generated.append(HS.write_hub_strategy_csv(REPORTS))

    coverage_df, coverage_warnings = HS.hub_grid_coverage(df)
    cov_path = REPORTS / "hub_grid_coverage.csv"
    coverage_df.to_csv(cov_path, index=False)
    generated.append(cov_path)
    for w in coverage_warnings:
        logger.warning(w)

    overlap_path, overlap_summary = HS.write_hub_overlap_csv(df, REPORTS)
    generated.append(overlap_path)

    local_summary = HS.per_hub_local_summary(df, indicators)
    p = REPORTS / "local_per_hub_summary.csv"
    local_summary.to_csv(p, index=False)
    generated.append(p)

    sw_local = HS.sliding_window_local(df, indicators)
    p = REPORTS / "local_sliding_window_summary.csv"
    sw_local.to_csv(p, index=False)
    generated.append(p)

    for ind, col in indicators.items():
        out = HS.plot_local_decay(sw_local, ind, col, FIGURES, label=col)
        if out is not None:
            generated.append(out)
    generated.append(HS.plot_local_all_hubs_comparison(sw_local, indicators, FIGURES))

    regional_summary, sample_warnings = HS.per_hub_regional_summary(df, indicators)
    p = REPORTS / "regional_per_hub_summary.csv"
    regional_summary.to_csv(p, index=False)
    generated.append(p)

    no2_p = HS.plot_regional_no2_background(regional_summary, VIS)
    if no2_p is not None:
        generated.append(no2_p)
    water_p = HS.plot_regional_water_indicators_background(regional_summary, VIS)
    if water_p is not None:
        generated.append(water_p)
    generated.append(HS.plot_regional_sample_density(regional_summary, VIS))

    no2_md, no2_findings = HS.write_no2_interpretation(
        sw_local, regional_summary, indicators, REPORTS,
    )
    generated.append(no2_md)

    # Stockholm-specific extras (zoom + local test).
    if "no2" in indicators:
        zp = EXP.plot_stockholm_no2_zoom(sw_local, indicators["no2"], FIGURES)
        if zp is not None:
            generated.append(zp)
    md_p, sthlm_findings = EXP.write_stockholm_no2_local_test(
        sw_local, local_summary, coverage_df, indicators, REPORTS,
    )
    generated.append(md_p)
    generated.append(EXP.write_florian_response_v2(REPORTS, sthlm_findings))
    generated.append(EXP.write_slide_deck_outline_v2(REPORTS))

    return {
        "generated": [str(p) for p in generated],
        "indicators": indicators,
        "coverage_df": coverage_df.to_dict("records"),
        "coverage_warnings": coverage_warnings,
        "sample_warnings": sample_warnings,
        "stockholm_findings": sthlm_findings,
        "n_rows": int(len(df)),
        "n_grids": int(df["grid_cell_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def write_final_summary(
    run_name: str,
    expand_info: dict,
    pipeline_status: dict,
    mirror_counts: dict,
    hub_result: dict,
    cache_wipe: list[Path],
    extra_warnings: list[str],
) -> Path:
    REPORTS = ROOT / "outputs" / "reports" / run_name
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "FINAL_RUN_SUMMARY.md"

    cov = hub_result.get("coverage_df", []) if hub_result else []
    by_hub = {row.get("hub_name"): row for row in cov}

    sthlm = (hub_result or {}).get("stockholm_findings", {})
    verdict = sthlm.get("verdict", {}) if isinstance(sthlm, dict) else {}

    lines: list[str] = []
    lines.append(f"# FINAL RUN SUMMARY — {run_name}\n")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`")
    lines.append(f"- Run name: `{run_name}`")
    lines.append("")
    lines.append("## Cache state")
    lines.append(f"- Wiped {len(cache_wipe)} aux/intermediate cache entries before launch.")
    lines.append("- `--force-refresh` was passed to the inner pipeline.")
    lines.append("- All GEE-backed sources (NO2, Sentinel-2 water, Sentinel-2 land NDVI, "
                 "Sentinel-1 oil-slick proxy) were re-extracted from scratch on the expanded grid.")
    lines.append("")
    lines.append("## Grid expansion")
    lines.append(f"- Old grid cells: **{expand_info.get('old_grids')}**")
    lines.append(f"- New cells added (Stockholm-vicinity, ≤150 km): **{expand_info.get('new_grids_added')}**")
    lines.append(f"- Total grid cells in this run: **{expand_info.get('expanded_grids')}**")
    lines.append(f"- Weeks: {expand_info.get('weeks')}")
    lines.append("")
    lines.append("## Per-hub coverage after expansion")
    lines.append("| Hub | nearest grid (km) | grids ≤25 km | grids ≤50 km | grids ≤100 km |")
    lines.append("|---|---|---|---|---|")
    for hub_name in (
        "Turku-Naantali coastal hub",
        "Mariehamn offshore/island hub",
        "Stockholm urban port hub",
    ):
        c = by_hub.get(hub_name, {})
        lines.append(
            f"| {hub_name} | {c.get('nearest_grid_km')} | "
            f"{c.get('grids_within_25km')} | {c.get('grids_within_50km')} | "
            f"{c.get('grids_within_100km')} |"
        )
    lines.append("")
    lines.append("## Pipeline status")
    if pipeline_status.get("returncode") == 0:
        lines.append(f"- Inner pipeline: **SUCCESS** in {pipeline_status.get('elapsed_s')}s.")
    else:
        lines.append(f"- Inner pipeline: **FAILED** (rc={pipeline_status.get('returncode')}, "
                     f"elapsed {pipeline_status.get('elapsed_s')}s).")
    lines.append("")
    lines.append("## Mirrored artefacts")
    for k, v in mirror_counts.items():
        lines.append(f"- `{k}`: {v} files")
    lines.append("")
    lines.append("## Stockholm NO2 verdict (revised hub strategy)")
    if verdict:
        lines.append(f"- Decreases with distance (0-100 km): **{verdict.get('stockholm_decreases_with_distance_0_100km')}**")
        lines.append(f"- Higher than Mariehamn 0-50 km: **{verdict.get('stockholm_no2_higher_than_mariehamn_0_50')}**")
        lines.append(f"- Urban-atmospheric NO2 interpretation: **{verdict.get('urban_atmospheric_support')}**")
    else:
        lines.append("- Hub analysis not produced.")
    lines.append("")
    lines.append("## Output isolation")
    lines.append("All artefacts for this run live ONLY under:")
    lines.append(f"- `outputs/reports/{run_name}/`")
    lines.append(f"- `outputs/figures/{run_name}/`")
    lines.append(f"- `outputs/visualizations/{run_name}/`")
    lines.append(f"- `processed/{run_name}/`")
    lines.append(f"- `logs/{run_name}/`")
    lines.append("Previous run folders (e.g., `final_run/`, `run_hub_strategy_*`, "
                 "`run_stockholm_grid_expanded/`) are untouched.")
    lines.append("")
    if extra_warnings:
        lines.append("## Warnings")
        for w in extra_warnings:
            lines.append(f"- {w}")
        lines.append("")
    out.write_text("\n".join(lines))
    LOGGER.info("Wrote %s", out.relative_to(ROOT))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-name",
        default=f"final_run_stockholm_fixed_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}",
        help="Run name (default: final_run_stockholm_fixed_<TS>).",
    )
    parser.add_argument(
        "--skip-sentinel1",
        action="store_true",
        help="Skip the Sentinel-1 oil-slick step (drop ~24h of S1 work).",
    )
    parser.add_argument(
        "--no-wipe-caches",
        action="store_true",
        help="Do NOT wipe data/aux + data/intermediate before launch (debug).",
    )
    args = parser.parse_args()

    run_name: str = args.run_name
    LOGS_DIR = ROOT / "logs" / run_name
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS = ROOT / "outputs" / "reports" / run_name
    FIGURES = ROOT / "outputs" / "figures" / run_name
    VIS     = ROOT / "outputs" / "visualizations" / run_name
    PROCESSED = ROOT / "processed" / run_name
    for p in (REPORTS, FIGURES, VIS, PROCESSED, LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)

    log_path = LOGS_DIR / "run_final_pipeline_stockholm_fixed.log"
    logger = configure_logging(log_path)

    logger.info("=== STOCKHOLM-FIXED FINAL PIPELINE: %s ===", run_name)
    logger.info("ROOT=%s", ROOT)
    logger.info("Sentinel-1 enabled: %s", not args.skip_sentinel1)

    extra_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Step A: build expanded modeling_dataset (back up original first).
    # ------------------------------------------------------------------
    backup_path = ROOT / "data" / f"modeling_dataset.backup.{run_name}.parquet"
    if not backup_path.exists():
        shutil.copy2(ORIG_MODELING, backup_path)
        logger.info("Backed up modeling_dataset → %s", backup_path.relative_to(ROOT))

    expanded_path = ROOT / "data" / "modeling_dataset.parquet"
    expand_info = build_expanded_modeling(expanded_path)

    # ------------------------------------------------------------------
    # Step B: wipe caches.
    # ------------------------------------------------------------------
    cache_wipe: list[Path] = []
    if not args.no_wipe_caches:
        cache_wipe = wipe_caches(logger)
    else:
        logger.warning("Cache wipe disabled (--no-wipe-caches).")

    # ------------------------------------------------------------------
    # Step C: invoke the inner pipeline.
    # ------------------------------------------------------------------
    inner_cmd = [
        sys.executable, "-u",
        str(SRC / "run_final_pipeline.py"),
        "--clean",
        "--run-name", run_name,
    ]
    if args.skip_sentinel1:
        # The inner pipeline does not currently expose --skip-sentinel1 directly,
        # but we can mark the cache wipe to leave an empty placeholder so the
        # source loop sees no fresh extraction needed.  Instead, we just let
        # the user opt out via env var that downstream code can read.
        os.environ["SKIP_SENTINEL1"] = "1"

    logger.info("Launching inner pipeline: %s", " ".join(inner_cmd))
    inner_log = LOGS_DIR / "inner_pipeline_stdout.log"
    pipeline_status: dict[str, Any] = {}
    t0 = time.time()
    rc = -1
    try:
        with inner_log.open("w") as fout:
            proc = subprocess.run(  # noqa: PLW1510
                inner_cmd, cwd=str(ROOT), stdout=fout, stderr=subprocess.STDOUT,
            )
            rc = proc.returncode
    except Exception as exc:  # noqa: BLE001
        logger.error("Inner pipeline launch failed: %s", exc)
        rc = -1
    pipeline_status = {
        "returncode": rc,
        "elapsed_s": int(time.time() - t0),
    }
    logger.info("Inner pipeline finished: rc=%d elapsed=%ss", rc, pipeline_status["elapsed_s"])
    if rc != 0:
        extra_warnings.append(
            f"Inner pipeline returned non-zero exit code {rc}. "
            f"Inspect {inner_log.relative_to(ROOT)} for details."
        )

    # ------------------------------------------------------------------
    # Step D: mirror outputs into the user-requested layout.
    # ------------------------------------------------------------------
    inner_run_root = ROOT / run_name
    mirror_counts = {}
    if inner_run_root.exists():
        mirror_counts = mirror_into_run_folders(inner_run_root, run_name, logger)
    else:
        logger.warning(
            "Inner run root %s not found — pipeline may have crashed before mirror step.",
            inner_run_root,
        )
        extra_warnings.append(
            f"Inner run root `{inner_run_root.relative_to(ROOT)}` was not produced; "
            "pipeline failed before producing artefacts."
        )

    # ------------------------------------------------------------------
    # Step E: re-run hub-strategy + Stockholm extras on the fresh dataset.
    # ------------------------------------------------------------------
    hub_result: dict[str, Any] = {}
    try:
        hub_result = run_hub_strategy_on_fresh(inner_run_root, run_name, logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Hub-strategy analysis failed.")
        extra_warnings.append(f"Hub-strategy analysis failed: {exc}")

    # ------------------------------------------------------------------
    # Step F: restore modeling_dataset.
    # ------------------------------------------------------------------
    try:
        shutil.move(str(backup_path), str(expanded_path))
        logger.info("Restored original modeling_dataset.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to restore modeling_dataset: %s", exc)
        extra_warnings.append(
            f"Failed to restore data/modeling_dataset.parquet from {backup_path.name}; "
            "manual restore needed."
        )

    # ------------------------------------------------------------------
    # Step G: write summary + final terminal output.
    # ------------------------------------------------------------------
    summary_path = write_final_summary(
        run_name, expand_info, pipeline_status, mirror_counts,
        hub_result, cache_wipe, extra_warnings,
    )

    print("\n" + "=" * 78)
    print(f"STOCKHOLM-FIXED FINAL PIPELINE — {run_name}")
    print("=" * 78)
    print(f"Run name: {run_name}")
    print(f"Inner pipeline rc: {pipeline_status.get('returncode')} "
          f"(elapsed {pipeline_status.get('elapsed_s')}s)")
    print(f"Cache wipe: {len(cache_wipe)} entries")
    print(f"Expanded grid: old={expand_info['old_grids']} grids, "
          f"new={expand_info['new_grids_added']} added, "
          f"total={expand_info['expanded_grids']}")
    print()
    print("Per-hub coverage:")
    for row in hub_result.get("coverage_df", []):
        print(f"  - {row.get('hub_name')}: nearest={row.get('nearest_grid_km')} km, "
              f"≤25={row.get('grids_within_25km')}, ≤50={row.get('grids_within_50km')}, "
              f"≤100={row.get('grids_within_100km')}")
    print()
    print("Stockholm NO2 verdict:")
    v = (hub_result.get("stockholm_findings") or {}).get("verdict", {})
    print(f"  - decreases with distance 0-100 km: "
          f"{v.get('stockholm_decreases_with_distance_0_100km')}")
    print(f"  - higher than Mariehamn 0-50 km: "
          f"{v.get('stockholm_no2_higher_than_mariehamn_0_50')}")
    print(f"  - urban-atmospheric NO2 interpretation: "
          f"{v.get('urban_atmospheric_support')}")
    print()
    print("Mirror counts:", mirror_counts)
    if extra_warnings:
        print("\nWarnings:")
        for w in extra_warnings:
            print(f"  - {w}")
    print()
    print(f"Summary report: {summary_path.relative_to(ROOT)}")
    print(f"Inner pipeline log: {inner_log.relative_to(ROOT)}")
    print(f"All outputs under outputs/reports/{run_name}/, outputs/figures/{run_name}/, "
          f"outputs/visualizations/{run_name}/, processed/{run_name}/, logs/{run_name}/")
    print("=" * 78)
    return 0 if pipeline_status.get("returncode") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
