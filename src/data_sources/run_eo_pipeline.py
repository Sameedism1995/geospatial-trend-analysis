"""
Run order: NO₂ GEE → Sentinel-1 oil GEE → read-only validation (science gate).

Exit code 1 if human_impact_integrity_report.json indicates failure of thesis readiness
(see science_gate.passes_integrity_rules).

Does not run human_impact_distance_analysis or modeling — EO extraction + validation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_sources.gee_grid_utils import DEFAULT_BUFFER_DEG_NO2, DEFAULT_BUFFER_DEG_S1  # noqa: E402
from data_sources.no2_gee_pipeline import run_pipeline as run_no2  # noqa: E402
from data_sources.sentinel1_oil_pipeline import run_pipeline as run_oil  # noqa: E402


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    r = _root()
    p = argparse.ArgumentParser(description="Run NO₂ + S1 oil GEE pipelines then validation")
    p.add_argument("--input", type=Path, default=r / "data" / "modeling_dataset.parquet")
    p.add_argument("--skip-no2", action="store_true")
    p.add_argument("--skip-oil", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--fail-on-validation", action="store_true", help="Exit 1 if science gate fails")
    args = p.parse_args()

    inp = args.input if args.input.is_absolute() else r / args.input
    no2_out = r / "data" / "aux" / "no2_grid_week.parquet"
    no2_val = r / "data" / "aux" / "no2_gee_validation.json"
    oil_out = r / "data" / "aux" / "sentinel1_oil_slicks.parquet"
    oil_val = r / "data" / "aux" / "sentinel1_oil_validation.json"
    val_dir = r / "data" / "validation"
    integ_path = val_dir / "human_impact_integrity_report.json"

    if not args.skip_no2:
        print("=== NO₂ (Sentinel-5P / GEE) ===")
        run_no2(inp, no2_out, no2_val, buffer_deg=DEFAULT_BUFFER_DEG_NO2)
    if not args.skip_oil:
        print("=== Oil proxy (Sentinel-1 GRD / GEE) ===")
        run_oil(inp, oil_out, oil_val, buffer_deg=DEFAULT_BUFFER_DEG_S1)

    if not args.skip_validation:
        print("=== Validation (read-only) ===")
        from validation.validate_aux_layers import (  # noqa: E402
            integrated_validation,
            resolve_oil_parquet,
            validate_no2,
            validate_oil,
        )

        no2_p = no2_out
        oil_p = resolve_oil_parquet(r, None)
        no2_meta = r / "data" / "aux" / "no2_gee_validation.json"
        oil_meta = r / "data" / "aux" / "sentinel1_oil_validation.json"
        val_dir.mkdir(parents=True, exist_ok=True)
        reports = (
            ("no2_validation_report.json", validate_no2(r, no2_p, no2_meta)),
            ("sentinel1_oil_validation_report.json", validate_oil(r, oil_p, oil_meta)),
            ("human_impact_integrity_report.json", integrated_validation(r, no2_p, oil_p)),
        )
        for name, obj in reports:
            out_p = val_dir / name
            with out_p.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=str)
            print(f"Wrote {out_p.resolve()}")

        if args.fail_on_validation and integ_path.exists():
            with integ_path.open(encoding="utf-8") as f:
                rep = json.load(f)
            gate = rep.get("science_gate", {})
            ok = gate.get("passes_integrity_rules", rep.get("ready_for_thesis_discussion", False))
            if not ok:
                print("ERROR: Science gate failed — do not proceed to modeling until data are fixed.")
                sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
