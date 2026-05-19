#!/usr/bin/env python3
"""
Orchestrate presentation-quality exports into outputs/presentation_graphs/.

Run from repo root:

    python scripts/run_presentation_graphs.py

Does not modify thesis pipeline scripts; may invoke rolling CV subprocess if metrics CSV absent.
"""

from __future__ import annotations

import traceback
from collections import defaultdict
from pathlib import Path

from .common import OUT_ROOT, ROOT, LOG, apply_style, ensure_dirs, log_result
from .dataset_environment import (
    env_maps_features,
    env_maps_modeling,
    fig_correlation_heatmap,
    fig_data_integration_architecture,
    fig_environmental_distributions,
    fig_missingness_heatmap,
    fig_port_locations,
    fig_spatial_coverage_density,
    fig_temporal_coverage,
    fig_weekly_environmental_variability,
    fig_weekly_observation_timeline,
    fig_indicator_boxplots,
    fig_baltic_study_extent_grid,
)
from .extended import ALL_EXTENDED_JOBS


DATASET_TASKS = [
    fig_baltic_study_extent_grid,
    fig_port_locations,
    fig_data_integration_architecture,
    fig_temporal_coverage,
    fig_weekly_observation_timeline,
    fig_missingness_heatmap,
    fig_spatial_coverage_density,
]

ENV_TASKS = [
    env_maps_modeling,
    env_maps_features,
    fig_environmental_distributions,
    fig_weekly_environmental_variability,
    fig_indicator_boxplots,
    fig_correlation_heatmap,
]


def _safe(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        log_result(category="FAILED", stem=label, ok=False, error=traceback.format_exc()[-1800:])


def write_readme(n_png: int, n_pdf: int, failures: list[str]) -> None:
    lines = [
        "# Presentation graphs bundle",
        "",
        f"Generated under `{OUT_ROOT.relative_to(ROOT)}`.",
        "",
        "## Figure directories",
        "",
        "| Folder | Contents |",
        "|--------|-----------|",
        "| `dataset_overview/` | Baltic extent, ports, temporal coverage, missingness, integration sketch |",
        "| `environmental_indicators/` | Spectral / vessel maps, distributions, correlations |",
        "| `exposure_analysis/` | MPI / coastal scores, decay curves, hotspots |",
        "| `wind_regime/` | Alignment diagnostics, atmospheric indices |",
        "| `machine_learning/` | Split metrics, predictions, residuals, importance |",
        "| `temporal_lag_persistence/` | Lag autocorr, persistence scatter |",
        "| `validation/` | Split diagrams, rolling trends, comparison graphic |",
        "| `anomaly_detection/` | Weekly exceedances, median z-score timeline |",
        "| `comparison_analysis/` | Cross-port standardised contrasts |",
        "| `summary_maps/` | Executive tiles + NDVI footprint summary |",
        "",
        "## Source code",
        "",
        "- Driver: `scripts/run_presentation_graphs.py`",
        "- Modules: `scripts/presentation_graphs/` (`common.py`, `dataset_environment.py`, `extended.py`, `generate_all.py`)",
        "",
        "## Slide placement hints",
        "",
        "- **Opening**: dataset_overview Baltic grid + integration diagram.",
        "- **Methods / data**: temporal coverage + missingness.",
        "- **Results environmental**: environmental_indicators maps + distributions.",
        "- **Exposure story**: exposure_analysis decay + hotspots.",
        "- **Wind / atmosphere**: wind_regime alignment plots.",
        "- **Temporal structure**: temporal_lag_persistence heatmaps.",
        "- **ML**: machine_learning predicted-vs-actual + metrics bars.",
        "- **Robustness**: validation rolling vs single-split comparison.",
        "- **Stress events**: anomaly_detection timelines.",
        "- **Ports**: comparison_analysis bars.",
        "- **Closing**: summary_maps infographic.",
        "",
        "## Export statistics",
        "",
        f"- PNG files: **{n_png}**",
        f"- PDF files: **{n_pdf}**",
        f"- Logged failures: **{len(failures)}**",
        "",
    ]
    if failures:
        lines.append("### Failures")
        lines.extend(f"- {f}" for f in failures)
    lines.append("")
    lines.append("## Generated PNG inventory")
    inv: dict[str, list[str]] = defaultdict(list)
    for p in sorted(OUT_ROOT.rglob("*.png")):
        inv[p.parent.name].append(p.name)
    for folder in sorted(inv.keys()):
        lines.append(f"### `{folder}/`")
        for name in sorted(inv[folder]):
            lines.append(f"- `{name}`")
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    apply_style()
    ensure_dirs()
    LOG.clear()

    for fn in DATASET_TASKS:
        _safe(fn.__name__, fn)
    for fn in ENV_TASKS:
        _safe(fn.__name__, fn)
    for fn in ALL_EXTENDED_JOBS:
        _safe(fn.__name__, fn)

    pngs = sorted(OUT_ROOT.rglob("*.png"))
    pdfs = sorted(OUT_ROOT.rglob("*.pdf"))
    fails = [f'{x.get("stem")}: {(str(x.get("error") or ""))[:220]}' for x in LOG if not x.get("ok")]
    write_readme(len(pngs), len(pdfs), fails)

    print("=== Presentation graphs summary ===")
    print(f"PNG count : {len(pngs)}")
    print(f"PDF count : {len(pdfs)}")
    print(f"Failures  : {len(fails)}")
    print(f"Root      : {OUT_ROOT}")
    if fails:
        print("Failed tasks (see README):", "; ".join(fails[:12]))
    return 1 if fails else 0
