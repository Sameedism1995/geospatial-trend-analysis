from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, Optional

import requests


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def stable_signature(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ApiRequest:
    source: str
    dataset: str
    method: str
    url: str
    params: dict[str, Any] = field(default_factory=dict)
    json_body: Optional[dict[str, Any]] = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 120
    page: Optional[int] = None
    cursor: Optional[str] = None

    def signature(self) -> str:
        return stable_signature(
            {
                "source": self.source,
                "dataset": self.dataset,
                "method": self.method,
                "url": self.url,
                "params": self.params,
                "json_body": self.json_body,
                "page": self.page,
                "cursor": self.cursor,
            }
        )


@dataclass
class ApiCallResult:
    source: str
    dataset: str
    endpoint: str
    request_signature: str
    fetched_at_utc: str
    status: str
    success: bool
    status_code: Optional[int]
    request_params: dict[str, Any]
    response_payload: Any
    error_message: Optional[str] = None
    records_count: int = 0
    retriable: bool = False


class BaseApiClient:
    def __init__(
        self,
        source_name: str,
        requests_per_second: float = 1.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
    ) -> None:
        self.source_name = source_name
        self.requests_per_second = max(requests_per_second, 0.1)
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._min_interval_s = 1.0 / self.requests_per_second
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

    def _record_count(self, payload: Any) -> int:
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in ("features", "items", "results", "metadata", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return len(value)
        return 1 if payload else 0

    def execute(self, req: ApiRequest) -> ApiCallResult:
        attempt = 0
        last_error: Optional[str] = None
        status_code: Optional[int] = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                self._throttle()
                response = requests.request(
                    method=req.method.upper(),
                    url=req.url,
                    params=req.params or None,
                    json=req.json_body,
                    headers=req.headers or None,
                    timeout=req.timeout_s,
                )
                self._last_call_ts = time.time()
                status_code = response.status_code
                retriable = status_code in {429, 500, 502, 503, 504}
                if response.ok:
                    payload: Any
                    content_type = response.headers.get("content-type", "")
                    if "json" in content_type.lower():
                        payload = response.json()
                    else:
                        payload = {"text": response.text, "content_type": content_type}
                    return ApiCallResult(
                        source=req.source,
                        dataset=req.dataset,
                        endpoint=req.url,
                        request_signature=req.signature(),
                        fetched_at_utc=utc_now_iso(),
                        status="success",
                        success=True,
                        status_code=status_code,
                        request_params=req.params,
                        response_payload=payload,
                        records_count=self._record_count(payload),
                    )

                last_error = f"HTTP {status_code}: {response.text[:500]}"
                if not retriable or attempt > self.max_retries:
                    break
                time.sleep(self.backoff_base_s * (2 ** (attempt - 1)))
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt > self.max_retries:
                    break
                time.sleep(self.backoff_base_s * (2 ** (attempt - 1)))

        return ApiCallResult(
            source=req.source,
            dataset=req.dataset,
            endpoint=req.url,
            request_signature=req.signature(),
            fetched_at_utc=utc_now_iso(),
            status="failed",
            success=False,
            status_code=status_code,
            request_params=req.params,
            response_payload={},
            error_message=last_error,
            retriable=True if status_code in {429, 500, 502, 503, 504} else False,
        )

    def build_missing_endpoint_result(
        self,
        dataset: str,
        endpoint: str,
        request_params: Optional[dict[str, Any]],
        reason: str,
    ) -> ApiCallResult:
        req = ApiRequest(
            source=self.source_name,
            dataset=dataset,
            method="GET",
            url=endpoint,
            params=request_params or {},
        )
        return ApiCallResult(
            source=self.source_name,
            dataset=dataset,
            endpoint=endpoint,
            request_signature=req.signature(),
            fetched_at_utc=utc_now_iso(),
            status="missing_endpoint",
            success=False,
            status_code=None,
            request_params=request_params or {},
            response_payload={},
            error_message=reason,
            retriable=False,
        )

    @staticmethod
    def chunked(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

