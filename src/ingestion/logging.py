from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import ApiCallResult, utc_now_iso


class ApiCallLogger:
    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.calls_log = self.logs_dir / "api_calls.jsonl"

    def log_call(self, result: ApiCallResult) -> None:
        row = {
            "timestamp_utc": result.fetched_at_utc,
            "source": result.source,
            "dataset": result.dataset,
            "endpoint": result.endpoint,
            "request_signature": result.request_signature,
            "request_params": result.request_params,
            "status": result.status,
            "success": result.success,
            "status_code": result.status_code,
            "records_count": result.records_count,
            "error_message": result.error_message,
            "retriable": result.retriable,
        }
        with open(self.calls_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


class RunManifestLogger:
    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, run_id: str, payload: dict[str, Any]) -> Path:
        path = self.logs_dir / f"run_manifest_{run_id}.json"
        payload = {"written_at_utc": utc_now_iso(), **payload}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

