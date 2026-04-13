from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .base import ApiCallResult


class DataStorage:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.raw_root = project_root / "data" / "raw"
        self.processed_root = project_root / "data" / "processed"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)

    def _ensure_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def raw_output_path(self, source: str, dataset: str, ingestion_date: str, signature: str) -> Path:
        return (
            self.raw_root
            / source
            / dataset
            / ingestion_date
            / f"{signature[:16]}.parquet"
        )

    def processed_output_path(
        self, source: str, dataset: str, ingestion_date: str, signature: str
    ) -> Path:
        return (
            self.processed_root
            / source
            / dataset
            / ingestion_date
            / f"{signature[:16]}.parquet"
        )

    def write_raw_result(self, result: ApiCallResult, ingestion_date: str, overwrite: bool) -> Path:
        path = self.raw_output_path(
            result.source, result.dataset, ingestion_date, result.request_signature
        )
        if path.exists() and not overwrite:
            return path

        self._ensure_parquet(path)
        row = {
            "source": result.source,
            "dataset": result.dataset,
            "endpoint": result.endpoint,
            "request_signature": result.request_signature,
            "fetched_at_utc": result.fetched_at_utc,
            "status": result.status,
            "success": result.success,
            "status_code": result.status_code,
            "request_params": json.dumps(result.request_params, default=str),
            "response_payload": json.dumps(result.response_payload, default=str),
            "error_message": result.error_message,
            "records_count": result.records_count,
        }
        pd.DataFrame([row]).to_parquet(path, index=False)
        return path

    def write_processed_records(
        self,
        source: str,
        dataset: str,
        ingestion_date: str,
        request_signature: str,
        records: list[dict[str, Any]],
        overwrite: bool,
    ) -> Path:
        path = self.processed_output_path(source, dataset, ingestion_date, request_signature)
        if path.exists() and not overwrite:
            return path

        self._ensure_parquet(path)
        if records:
            df = pd.DataFrame(records)
        else:
            # Explicitly preserve a file artifact with no fake rows.
            df = pd.DataFrame(
                columns=[
                    "source",
                    "dataset",
                    "record_timestamp_utc",
                    "ingested_at_utc",
                    "latitude",
                    "longitude",
                    "geometry_wkt",
                    "crs",
                    "grid_id",
                    "raw_record_ref",
                    "payload",
                ]
            )
        df.to_parquet(path, index=False)
        return path

