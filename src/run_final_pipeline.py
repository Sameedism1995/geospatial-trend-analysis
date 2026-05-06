"""Final, reproducible end-to-end pipeline run.

Produces a self-contained `final_run/` deliverable with:
  - data/                    (raw / aux / intermediate snapshots)
  - processed/               (merged_dataset + features_ml_ready)
  - outputs/reports/         (correlations, lags, feature interactions, hub analysis CSVs, NO2 diagnostic)
  - outputs/visualizations/  (EDA, plots, hub-level + sliding-window plots, exposure maps)
  - logs/full_pipeline.log
  - validation/              (validation reports, including the new final_dataset_validation)
  - config_snapshot/         (copy of all config/*.yaml|json)
  - FINAL_RUN_SUMMARY.md

Usage:
    python3 src/run_final_pipeline.py
    python3 src/run_final_pipeline.py --quick-test
    python3 src/run_final_pipeline.py --no-force-refresh   (reuse cached aux parquets)

`--force-refresh` is the default (clears reuse so ingestion runs fresh). If GEE
auth or remote APIs are unavailable, individual sources will fail per-source,
the orchestrator continues, and a clear warning is written to FINAL_RUN_SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FINAL_RUN_ROOT = ROOT / "final_run"

LOGGER = logging.getLogger("final_pipeline")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    # Configure handlers on the root logger only so child loggers (final_pipeline,
    # full_pipeline, hub_distance_decay, ...) inherit them via propagation
    # without duplicating output.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)
    logger = logging.getLogger("final_pipeline")
    logger.handlers.clear()
    logger.propagate = True
    return logger


def make_skeleton(base: Path) -> dict[str, Path]:
    paths = {
        "data": base / "data",
        "data_aux": base / "data" / "aux",
        "data_intermediate": base / "data" / "intermediate",
        "processed": base / "processed",
        "outputs": base / "outputs",
        "reports": base / "outputs" / "reports",
        "visualizations": base / "outputs" / "visualizations",
        "viz_plots": base / "outputs" / "visualizations" / "plots",
        "viz_eda": base / "outputs" / "visualizations" / "eda",
        "viz_previews": base / "outputs" / "visualizations" / "previews",
        "logs": base / "logs",
        "validation": base / "validation",
        "config_snapshot": base / "config_snapshot",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def snapshot_configs(target_dir: Path, logger: logging.Logger) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    sources: list[Path] = []
    if (ROOT / "config").exists():
        sources.extend(sorted((ROOT / "config").rglob("*")))
    sources.extend([ROOT / "requirements.txt", ROOT / "README.md", ROOT / ".gitignore"])
    for src in sources:
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".yaml", ".yml", ".json", ".txt", ".md", ".cfg", ".toml", ".env"} and src.parent != ROOT:
            continue
        rel = src.relative_to(ROOT) if src.is_relative_to(ROOT) else Path(src.name)
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        saved.append(dst)
    logger.info("[CONFIG SNAPSHOT] copied %d files to %s", len(saved), target_dir)
    return saved


# ---------------------------------------------------------------------------
# Pre-flight: GEE auth probe
# ---------------------------------------------------------------------------

def gee_probe(logger: logging.Logger) -> dict[str, Any]:
    """Try to initialise Earth Engine. Return status dict."""
    info: dict[str, Any] = {"available": False}
    try:
        import ee  # type: ignore

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("EE_PROJECT")
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            info["available"] = True
            info["project"] = project or "(default)"
            logger.info("[GEE PROBE] ok project=%s", info["project"])
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)[:300]
            logger.warning("[GEE PROBE] auth failed: %s", info["error"])
    except ImportError:
        info["error"] = "earthengine-api not installed"
        logger.warning("[GEE PROBE] earthengine-api not installed")
    return info


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_full_pipeline(
    *,
    quick_test: bool,
    force_refresh: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run pipeline.run_full_pipeline.main() with our chosen flags. Captures result."""
    logger.info("[PIPELINE] Importing pipeline.run_full_pipeline")
    import pipeline.run_full_pipeline as rfp  # type: ignore

    argv = ["run_full_pipeline"]
    if quick_test:
        argv.append("--quick-test")
    if force_refresh:
        argv.append("--force-refresh")
    argv.extend(
        [
            "--feature-interaction-map",
            "--scientific-validation",
            "--anomaly-detection",
            "--coastal-impact-score",
            "--final-visualization",
            "--land-impact",
        ]
    )
    logger.info("[PIPELINE] argv=%s", argv)
    saved_argv = sys.argv[:]
    sys.argv = argv
    t0 = time.time()
    status: dict[str, Any] = {"started": t0}
    try:
        rfp.main()
        status["status"] = "ok"
    except SystemExit as exc:
        status["status"] = "system_exit"
        status["code"] = int(getattr(exc, "code", 0) or 0)
        logger.warning("[PIPELINE] exited via SystemExit code=%s", status["code"])
    except Exception as exc:  # noqa: BLE001
        status["status"] = "error"
        status["error"] = str(exc)
        logger.exception("[PIPELINE] crashed: %s", exc)
    finally:
        sys.argv = saved_argv
        status["elapsed_s"] = round(time.time() - t0, 2)
    logger.info("[PIPELINE] finished status=%s elapsed=%ss", status.get("status"), status.get("elapsed_s"))
    return status


# ---------------------------------------------------------------------------
# Mirror project-root outputs into final_run/
# ---------------------------------------------------------------------------

def _copy_tree(src: Path, dst: Path, logger: logging.Logger) -> int:
    if not src.exists():
        logger.info("[MIRROR] skip (missing): %s", src)
        return 0
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return 1
    n = 0
    for child in src.rglob("*"):
        if child.is_dir():
            continue
        rel = child.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, target)
        n += 1
    logger.info("[MIRROR] %s → %s (%d files)", src, dst, n)
    return n


def mirror_outputs(paths: dict[str, Path], logger: logging.Logger) -> dict[str, int]:
    counts: dict[str, int] = {}
    counts["data_aux"] = _copy_tree(ROOT / "data" / "aux", paths["data_aux"], logger)
    counts["data_intermediate"] = _copy_tree(ROOT / "data" / "intermediate", paths["data_intermediate"], logger)
    counts["data_validation"] = _copy_tree(ROOT / "data" / "validation", paths["validation"], logger)
    counts["modeling_input"] = _copy_tree(ROOT / "data" / "modeling_dataset.parquet", paths["data"] / "modeling_dataset.parquet", logger)
    counts["processed"] = _copy_tree(ROOT / "processed", paths["processed"], logger)
    counts["outputs_reports"] = _copy_tree(ROOT / "outputs" / "reports", paths["reports"], logger)
    counts["outputs_eda"] = _copy_tree(ROOT / "outputs" / "eda", paths["viz_eda"], logger)
    eda_md = ROOT / "outputs" / "eda_summary.md"
    if eda_md.exists():
        shutil.copy2(eda_md, paths["reports"] / "eda_summary.md")
        counts["eda_summary_md"] = 1
    counts["outputs_plots"] = _copy_tree(ROOT / "outputs" / "plots", paths["viz_plots"], logger)
    counts["outputs_previews"] = _copy_tree(ROOT / "outputs" / "previews", paths["viz_previews"], logger)
    counts["outputs_visualizations"] = _copy_tree(ROOT / "outputs" / "visualizations", paths["visualizations"], logger)
    pipeline_log = ROOT / "logs" / "pipeline_run.log"
    if pipeline_log.exists():
        shutil.copy2(pipeline_log, paths["logs"] / "pipeline_run.log")
        counts["pipeline_log"] = 1
    return counts


# ---------------------------------------------------------------------------
# Plot QA
# ---------------------------------------------------------------------------

def plot_quality_check(viz_root: Path, logger: logging.Logger) -> dict[str, Any]:
    """Walk all PNG plots under viz_root; flag suspicious ones (very small file size)."""
    results = {"total": 0, "small_files": [], "min_size_bytes": 1024}
    for png in viz_root.rglob("*.png"):
        results["total"] += 1
        size = png.stat().st_size
        if size < results["min_size_bytes"]:
            results["small_files"].append({"path": str(png.relative_to(viz_root)), "size": int(size)})
    if results["small_files"]:
        logger.warning("[PLOT QA] %d plots smaller than %d bytes (likely empty/nan-only)", len(results["small_files"]), results["min_size_bytes"])
        for entry in results["small_files"][:10]:
            logger.warning("  %s (%d bytes)", entry["path"], entry["size"])
    else:
        logger.info("[PLOT QA] %d plots checked, all above %d bytes", results["total"], results["min_size_bytes"])
    return results


# ---------------------------------------------------------------------------
# Final summary writer
# ---------------------------------------------------------------------------

def write_final_summary(
    base: Path,
    *,
    pipeline_status: dict[str, Any],
    gee_status: dict[str, Any],
    mirror_counts: dict[str, int],
    validation_report: dict[str, Any] | None,
    hub_summary: dict[str, Any] | None,
    plot_qa: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    out = base / "FINAL_RUN_SUMMARY.md"
    lines: list[str] = []
    lines.append("# FINAL RUN SUMMARY")
    lines.append("")
    lines.append(f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`")
    lines.append(f"- Final-run root: `{base.relative_to(ROOT)}`")
    lines.append(f"- Pipeline status: **{pipeline_status.get('status')}** (elapsed {pipeline_status.get('elapsed_s')}s)")
    if pipeline_status.get("error"):
        lines.append(f"- Pipeline error: `{pipeline_status['error']}`")
    lines.append("")

    lines.append("## Earth Engine pre-flight")
    if gee_status.get("available"):
        lines.append(f"- GEE auth OK (project=`{gee_status.get('project')}`).")
    else:
        lines.append(f"- GEE auth NOT available — `{gee_status.get('error')}`.")
        lines.append(
            "- Consequence: GEE-backed sources (Sentinel-1 oil, Sentinel-2 water/land, NO2) cannot be re-extracted "
            "in this environment. The orchestrator therefore reused the most recent cached `data/aux/*.parquet` "
            "files (which **are** the latest successful GEE extraction, just committed to the repo). "
            "Set `GOOGLE_CLOUD_PROJECT` or `EE_PROJECT` and re-run with `--force-refresh` (the default) to truly re-extract."
        )
    lines.append(f"- Requested force_refresh: **{pipeline_status.get('requested_force_refresh')}** | effective: **{pipeline_status.get('effective_force_refresh')}**")
    lines.append("")

    lines.append("## Mirrored artefacts (project root → final_run/)")
    for k, v in mirror_counts.items():
        lines.append(f"- `{k}`: {v} files")
    lines.append("")

    lines.append("## Dataset")
    if validation_report:
        rows, cols = validation_report.get("shape", [None, None])
        ds_path = Path(validation_report.get("dataset_path", ""))
        try:
            ds_rel = ds_path.relative_to(ROOT)
        except ValueError:
            ds_rel = ds_path
        lines.append(f"- ML-ready features parquet: `{ds_rel}` ({rows} × {cols})")
        miss = validation_report.get("missing_values", {})
        high = miss.get("high_missing_columns", {})
        lines.append(f"- High-missing columns (>{miss.get('high_missing_threshold_percent')}%): {len(high)}")
        for k, v in sorted(high.items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"  - `{k}`: {v:.2f}%")
        temporal = validation_report.get("temporal", {})
        lines.append(f"- Temporal coverage: {temporal.get('first_week')} → {temporal.get('last_week')}, "
                     f"{temporal.get('n_unique_weeks')}/{temporal.get('n_expected_weeks')} weeks present, "
                     f"{temporal.get('n_missing_weeks')} missing")
        spatial = validation_report.get("spatial", {})
        lines.append(f"- Spatial: {spatial.get('n_unique_grid_cells')} grid cells, "
                     f"{spatial.get('n_grid_week_pairs')} grid-week pairs, "
                     f"{spatial.get('n_duplicate_rows')} duplicates")
        corr = validation_report.get("correlation_sanity", {})
        lines.append(f"- Correlation flags (|r|≥{corr.get('threshold')}): {corr.get('n_flagged_pairs')}")
        warnings = validation_report.get("warnings") or []
        if warnings:
            lines.append("- Validation warnings:")
            for w in warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("- Validation warnings: none.")
    else:
        lines.append("- Validation report not produced.")
    lines.append("")

    lines.append("## Hub distance-decay analysis")
    if hub_summary:
        lines.append(f"- Hubs analysed: **{len(hub_summary.get('hubs', []))}** ({hub_summary.get('hubs')}).")
        lines.append(f"- Indicators used: `{hub_summary.get('indicators_used')}`.")
        no2 = hub_summary.get("no2_findings") or {}
        if no2:
            lines.append(
                f"- NO2 200–500 km mean: **{no2.get('band_mean'):.3e}** (n={no2.get('band_n')}); "
                f"<200 km = {no2.get('near_mean'):.3e}; >500 km = {no2.get('far_mean'):.3e}."
            )
            lines.append(f"- 200–500 km hub composition: `{no2.get('band_hub_counts')}`.")
        out_dirs = hub_summary.get("out_dirs", {})
        for k, v in out_dirs.items():
            lines.append(f"- `{k}`: `{Path(v).relative_to(ROOT) if Path(v).is_relative_to(ROOT) else v}`")
    else:
        lines.append("- Hub analysis not produced.")
    lines.append("")

    lines.append("## Plot quality check")
    lines.append(f"- Plots scanned: {plot_qa.get('total')}")
    if plot_qa.get("small_files"):
        lines.append(f"- Suspicious small files (< {plot_qa.get('min_size_bytes')} bytes): {len(plot_qa['small_files'])}")
        for entry in plot_qa["small_files"][:10]:
            lines.append(f"  - `{entry['path']}` ({entry['size']} bytes)")
    else:
        lines.append("- All plots above size threshold.")
    lines.append("")

    lines.append("## Known limitations")
    lines.append("- Only 3 distinct nearest ports in the dataset (Mariehamn / Naantali / Turku); the 200–500 km distance "
                 "band is single-hub (Mariehamn) so cross-hub claims at that range should be made cautiously.")
    lines.append("- NDVI is coastal-only; expect low coverage on this column (this is normal, not a defect).")
    lines.append("- Sentinel-2 / Sentinel-1 / NO2 sources require valid Earth Engine project credentials for true "
                 "fresh re-extraction; the run uses cached aux parquets when GEE is unavailable.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("- Configs snapshot: `config_snapshot/`")
    lines.append("- Full log: `logs/full_pipeline.log` (and `logs/pipeline_run.log` if produced).")
    lines.append("- Re-run: `python3 src/run_final_pipeline.py`")
    lines.append("")

    out.write_text("\n".join(lines))
    logger.info("Wrote %s", out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-test", action="store_true", help="Use sample-mode pipeline (small samples).")
    parser.add_argument("--no-force-refresh", action="store_true", help="Reuse cached aux parquets instead of forcing fresh extraction.")
    parser.add_argument("--clean", action="store_true", help="Delete the run folder before starting (otherwise overwrites in place).")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom name for the run folder.  When set, the orchestrator writes "
             "to <project>/<run-name>/ instead of <project>/final_run/.  Use this "
             "to keep multiple runs isolated.",
    )
    args = parser.parse_args()

    global FINAL_RUN_ROOT  # noqa: PLW0603
    if args.run_name:
        FINAL_RUN_ROOT = ROOT / args.run_name
    if args.clean and FINAL_RUN_ROOT.exists():
        shutil.rmtree(FINAL_RUN_ROOT)
    paths = make_skeleton(FINAL_RUN_ROOT)
    log_path = paths["logs"] / "full_pipeline.log"
    logger = setup_logger(log_path)
    logger.info("=== FINAL RUN START ===")
    logger.info("ROOT=%s | FINAL_RUN_ROOT=%s", ROOT, FINAL_RUN_ROOT)
    logger.info("force_refresh=%s | quick_test=%s", not args.no_force_refresh, args.quick_test)

    snapshot_configs(paths["config_snapshot"], logger)

    gee_status = gee_probe(logger)
    requested_force_refresh = not args.no_force_refresh
    force_refresh = requested_force_refresh
    if requested_force_refresh and not gee_status.get("available"):
        logger.warning(
            "[FINAL RUN] GEE unavailable — disabling --force-refresh to avoid empty merged dataset. "
            "Pipeline will reuse the cached aux parquets (most recent successful GEE extraction)."
        )
        force_refresh = False

    pipeline_status = run_full_pipeline(quick_test=args.quick_test, force_refresh=force_refresh, logger=logger)
    pipeline_status["requested_force_refresh"] = requested_force_refresh
    pipeline_status["effective_force_refresh"] = force_refresh

    logger.info("=== Mirroring outputs into final_run/ ===")
    mirror_counts = mirror_outputs(paths, logger)

    logger.info("=== Final dataset validation ===")
    features_path = paths["processed"] / "features_ml_ready.parquet"
    validation_report: dict[str, Any] | None = None
    if features_path.exists():
        try:
            from validation.final_dataset_validation import run_validation  # type: ignore

            validation_report = run_validation(features_path, paths["validation"], logger=logger)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[VALIDATION] Failed: %s", exc)
    else:
        logger.warning("[VALIDATION] features parquet not found at %s; skipping", features_path)

    logger.info("=== Hub distance-decay analysis on final_run dataset ===")
    hub_summary: dict[str, Any] | None = None
    if features_path.exists():
        try:
            from analysis.hub_distance_decay_analysis import run_analysis  # type: ignore

            hub_summary = run_analysis(dataset=features_path, out_root=FINAL_RUN_ROOT)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[HUB ANALYSIS] Failed: %s", exc)
    else:
        logger.warning("[HUB ANALYSIS] features parquet not found; skipping")

    logger.info("=== Plot quality check ===")
    plot_qa = plot_quality_check(paths["visualizations"], logger)

    summary_path = write_final_summary(
        FINAL_RUN_ROOT,
        pipeline_status=pipeline_status,
        gee_status=gee_status,
        mirror_counts=mirror_counts,
        validation_report=validation_report,
        hub_summary=hub_summary,
        plot_qa=plot_qa,
        logger=logger,
    )

    logger.info("=== FINAL RUN COMPLETE ===")
    logger.info("Summary: %s", summary_path)
    logger.info("All artefacts under: %s", FINAL_RUN_ROOT)

    # Print key paths for the user
    logger.info("Generated outputs (top-level + key files):")
    seen: set[Path] = set()
    for child in sorted(FINAL_RUN_ROOT.rglob("*")):
        if not child.is_file() or child.suffix not in {".md", ".json", ".csv", ".parquet"}:
            continue
        rel = child.relative_to(ROOT)
        if rel in seen:
            continue
        seen.add(rel)
        logger.info("  %s", rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
