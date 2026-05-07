#!/usr/bin/env python3
"""
Final thesis outputs (Turku vs Mariehamn only): two distance methodologies.

A) Fixed-band distance decay — all standard annuli; gaps = missing data (not imputed).
B) Shared-valid-annulus comparison — per-indicator nearest band with both ports and both wind strata (n≥1).

Writes under: outputs/final_run_turku_mariehamn_thesis/

  python3 src/analysis/run_final_thesis_turku_mariehamn.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns

class _TeeStream:
    """Write to multiple streams (stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC))

from analysis.run_coastal_exposure_analysis import (  # noqa: E402
    build_indices,
    merge_wind_vectors,
    prepare_panel,
)
from analysis.run_portwise_coastal_exposure import (  # noqa: E402
    FIGURE_PORTS,
    FOCAL_PORTS,
    WIND_REGIME_ZONE_PRIORITY,
    aggregate_long_table,
    attach_focal_port_distances,
    build_rankings,
    metric_definitions,
    plot_decay_curves,
    plot_index_ranking,
    plot_wind_regime_bars,
    _safe_slug,
)

from analysis.final_thesis_spatiotemporal import run_spatiotemporal_block  # noqa: E402

THESIS_ROOT = _ROOT / "outputs" / "final_run_turku_mariehamn_thesis"
THESIS_FIG = THESIS_ROOT / "figures"
THESIS_REP = THESIS_ROOT / "reports"
THESIS_NOTES = THESIS_ROOT / "thesis_notes"

DEFAULT_INPUT = _ROOT / "outputs" / "processed" / "features_ml_ready_coastal_wind.parquet"
WIND_CSV = _ROOT / "outputs" / "reports" / "run_coastal_wind_transport" / "coastal_wind_alignment_features.csv"
NE = _ROOT / "data" / "aux" / "natural_earth_coast_cache"


def write_shared_annulus_methodology_md() -> None:
    p = THESIS_REP / "shared_annulus_methodology.md"
    p.write_text(
        "\n".join(
            [
                "# Shared-valid-annulus methodology (Turku vs Mariehamn)",
                "",
                "## Why two distance modes?",
                "",
                "### Fixed-band distance decay",
                "All port-centred annuli in the standard list are **always** retained: "
                "**0–3 km (coastal core)**, **0–3 km**, **3–7 km**, **7–15 km**, **15–30 km**. "
                "Empty or undefined strata appear as **gaps** in the curves or missing bootstrap intervals — **no imputation**.",
                "",
                "### Shared-valid-annulus cross-port comparison",
                "For **shoreward vs non-shoreward comparison between Turku and Mariehamn**, forcing one fixed band for every composite "
                "can mislead when coverage or valid rank-inputs differ by band (e.g. maritime exposure needs intra-week ranks of "
                "vessel density and wind alignment). Comparing bars without aligning supported annuli implies comparability the grid may not support.",
                "",
                "So **per composite indicator**, we take the **nearest** annulus in order "
                "**0–3 km → 3–7 km → 7–15 km → 15–30 km** such that **both** ports have **shoreward** and **non-shoreward** strata with "
                "**n ≥ 1** in the decay table. If none qualify, that panel is **n/a** — **no fabricated values**.",
                "",
                "## Data integrity",
                "",
                "- Missing bands remain visible in **fixed-band** decay figures.",
                "- Selections are logged in `shared_annulus_selection.csv`.",
                "- **Stockholm** is excluded from this thesis comparative bundle.",
                "",
            ],
        ),
        encoding="utf-8",
    )


def write_thesis_notes_text() -> None:
    p = THESIS_NOTES / "shared_annulus_methodology_text.md"
    p.write_text(
        "\n".join(
            [
                "## Thesis wording (cross-port directional comparison)",
                "",
                "For cross-port directional comparisons of composite indices, **each indicator** was evaluated within the **nearest shared distance annulus** "
                "containing **valid observations for both Turku and Mariehamn** and for **both** shoreward and non-shoreward wind regimes (**n ≥ 1** per stratum). "
                "This avoids **artificial imputation** while preserving **fair comparability** on the available grid. "
                "**Fixed-band distance-decay analysis** was retained **separately**: all standard annuli are reported and **missing values are explicit** — **no values were fabricated**.",
                "",
                "Association / structuring language only — not causal transport or deposition.",
                "",
            ],
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--ne-cache", type=Path, default=NE)
    ap.add_argument("--wind-features-csv", type=Path, default=WIND_CSV)
    args = ap.parse_args()

    log_path = THESIS_ROOT / "logs" / "final_temporal_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _logf = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — script lifecycle
    _old_out, _old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(_old_out, _logf)
    sys.stderr = _TeeStream(_old_err, _logf)

    try:
        return _main_inner(args)
    finally:
        sys.stdout = _old_out
        sys.stderr = _old_err
        _logf.close()


def _main_inner(args: argparse.Namespace) -> int:

    if not args.input.is_file():
        print(f"[FATAL] missing {args.input}")
        return 1

    for d in (THESIS_ROOT, THESIS_FIG, THESIS_REP, THESIS_NOTES):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df = merge_wind_vectors(df, args.wind_features_csv)
    df = attach_focal_port_distances(df)
    df = prepare_panel(df, Path(args.ne_cache))
    df = build_indices(df)

    metrics = metric_definitions(df)
    all_rows: list = []
    for pname in FOCAL_PORTS:
        all_rows.extend(aggregate_long_table(df, pname, f"dist_{_safe_slug(pname)}_km", metrics))

    decay_tbl = pd.DataFrame(all_rows)
    decay_tm = decay_tbl.loc[decay_tbl["port"].isin(FIGURE_PORTS.keys())].copy()
    decay_tbl.to_csv(THESIS_REP / "port_distance_decay_statistics_full_with_stockholm.csv", index=False)
    decay_tm.to_csv(THESIS_REP / "port_distance_decay_statistics_turku_mariehamn.csv", index=False)

    ports = list(FIGURE_PORTS.keys())
    print("[validation] Band priority (shared-valid-annulus):", WIND_REGIME_ZONE_PRIORITY)

    sel_path = THESIS_REP / "shared_annulus_selection.csv"
    plot_wind_regime_bars(
        decay_tm,
        save_path=THESIS_FIG / "Fig_wind_regime_shared_valid_annulus_turku_mariehamn.png",
        diagnostics_path=sel_path,
        log_selection=True,
    )
    print("[validation] Shared annulus diagnostics ->", sel_path)

    plot_decay_curves(
        decay_tm,
        save_dir=THESIS_FIG,
        filename_pattern="final_distance_decay_{slug}.png",
        suptitle_template=(
            "{port}: fixed-band distance decay — all standard annuli (gaps = missing; bootstrap 95% CI; not causal)"
        ),
    )

    rank_df = build_rankings(decay_tbl, df, list(FIGURE_PORTS.keys()))
    rank_df.to_csv(THESIS_REP / "port_exposure_ranking_turku_mariehamn.csv", index=False)

    sns.set_theme(style="whitegrid", font_scale=0.95)
    plot_index_ranking(rank_df, save_path=THESIS_FIG / "Fig1_port_exposure_index_ranking_turku_mariehamn.png")

    write_shared_annulus_methodology_md()
    write_thesis_notes_text()

    print("[spatiotemporal] Lag / anomaly / synthesis / validation ->", THESIS_ROOT)
    run_spatiotemporal_block(df, THESIS_ROOT)

    print(f"[OK] Thesis bundle -> {THESIS_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
