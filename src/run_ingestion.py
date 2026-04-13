from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion.orchestrator import IngestionOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-grade ingestion pipeline runner")
    parser.add_argument(
        "--mode",
        choices=["scheduled", "on_demand"],
        default="scheduled",
        help="Run mode: scheduled daily or on-demand query.",
    )
    parser.add_argument(
        "--config",
        default="config/ingestion_sources.yaml",
        help="Path to ingestion source config YAML.",
    )
    parser.add_argument(
        "--sources",
        default="emodnet,helcom,sentinel",
        help="Comma-separated source list.",
    )
    parser.add_argument("--bbox", default="", help="BBox as minLon,minLat,maxLon,maxLat")
    parser.add_argument("--time-from", default="", help="Start datetime/date in ISO format")
    parser.add_argument("--time-to", default="", help="End datetime/date in ISO format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing partition files.")
    parser.add_argument("--sentinel-limit", type=int, default=100)
    parser.add_argument("--sentinel-max-pages", type=int, default=1)
    parser.add_argument("--sentinel-max-cloud", type=int, default=30)
    parser.add_argument("--helcom-page-size", type=int, default=50)
    parser.add_argument("--helcom-max-pages", type=int, default=1)
    parser.add_argument("--helcom-max-records", type=int, default=10)
    parser.add_argument("--emodnet-max-assets", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    orchestrator = IngestionOrchestrator(project_root=project_root, config_path=config_path)

    bbox = None
    if args.bbox:
        bbox = [float(x.strip()) for x in args.bbox.split(",")]
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly 4 comma-separated values")

    params = {
        "sources": [s.strip() for s in args.sources.split(",") if s.strip()],
        "bbox": bbox,
        "time_from": args.time_from or None,
        "time_to": args.time_to or None,
        "sentinel_limit": args.sentinel_limit,
        "sentinel_max_pages": args.sentinel_max_pages,
        "sentinel_max_cloud": args.sentinel_max_cloud,
        "helcom_page_size": args.helcom_page_size,
        "helcom_max_pages": args.helcom_max_pages,
        "helcom_max_records": args.helcom_max_records,
        "emodnet_max_assets": args.emodnet_max_assets,
    }
    params = {k: v for k, v in params.items() if v is not None}

    result = orchestrator.run(mode=args.mode, run_params=params, overwrite=args.overwrite)
    summary = {
        "run_id": result.run_id,
        "raw_files_count": len(result.raw_files),
        "processed_files_count": len(result.processed_files),
        "failures": result.failures,
        "manifests": [str(p) for p in result.manifests],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

