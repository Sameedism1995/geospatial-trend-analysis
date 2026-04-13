from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .emodnet_client import EmodnetClient
from .helcom_client import HelcomClient
from .logging import ApiCallLogger, RunManifestLogger
from .sentinel_client import SentinelHubClient
from .standardize import standardize_result
from .storage import DataStorage
from .validation import validate_no_fake_data


@dataclass
class IngestionRunResult:
    run_id: str
    manifests: list[Path]
    raw_files: list[Path]
    processed_files: list[Path]
    failures: int


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class IngestionOrchestrator:
    def __init__(self, project_root: Path, config_path: Path) -> None:
        self.project_root = project_root
        self.config = load_config(config_path)
        self.storage = DataStorage(project_root)
        logs_dir = project_root / "data" / "raw" / "_logs"
        self.call_logger = ApiCallLogger(logs_dir)
        self.manifest_logger = RunManifestLogger(logs_dir)

    def _build_clients(self) -> dict[str, Any]:
        sources = self.config.get("sources", {})
        return {
            "emodnet": EmodnetClient(sources.get("emodnet", {})),
            "helcom": HelcomClient(sources.get("helcom", {})),
            "sentinel": SentinelHubClient(sources.get("sentinel", {})),
        }

    def run(self, mode: str, run_params: dict[str, Any], overwrite: bool = False) -> IngestionRunResult:
        if mode not in {"scheduled", "on_demand"}:
            raise ValueError("mode must be one of: scheduled, on_demand")
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        ingestion_date = datetime.now(UTC).date().isoformat()

        clients = self._build_clients()
        selected = run_params.get("sources", ["emodnet", "helcom", "sentinel"])
        raw_files: list[Path] = []
        processed_files: list[Path] = []
        failures = 0
        call_summaries: list[dict[str, Any]] = []

        for source_name in selected:
            client = clients.get(source_name)
            if client is None:
                continue
            source_results = client.fetch(run_params=run_params)
            for result in source_results:
                self.call_logger.log_call(result)
                raw_path = self.storage.write_raw_result(
                    result=result, ingestion_date=ingestion_date, overwrite=overwrite
                )
                raw_files.append(raw_path)

                standardized_records = standardize_result(result)
                processed_path = self.storage.write_processed_records(
                    source=result.source,
                    dataset=result.dataset,
                    ingestion_date=ingestion_date,
                    request_signature=result.request_signature,
                    records=standardized_records,
                    overwrite=overwrite,
                )
                processed_files.append(processed_path)
                if not result.success:
                    failures += 1

                call_summaries.append(
                    {
                        "source": result.source,
                        "dataset": result.dataset,
                        "request_signature": result.request_signature,
                        "status": result.status,
                        "success": result.success,
                        "raw_file": str(raw_path),
                        "processed_file": str(processed_path),
                        "records_count": result.records_count,
                        "error_message": result.error_message,
                    }
                )

        manifest = self.manifest_logger.write_manifest(
            run_id=run_id,
            payload={
                "run_id": run_id,
                "mode": mode,
                "selected_sources": selected,
                "config_snapshot": self.config,
                "run_params": run_params,
                "raw_files": [str(p) for p in raw_files],
                "processed_files": [str(p) for p in processed_files],
                "call_summaries": call_summaries,
                "failures": failures,
                "validation": validate_no_fake_data(processed_files),
            },
        )
        return IngestionRunResult(
            run_id=run_id,
            manifests=[manifest],
            raw_files=raw_files,
            processed_files=processed_files,
            failures=failures,
        )

