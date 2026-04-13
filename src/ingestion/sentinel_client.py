from __future__ import annotations

import os
from typing import Any

from .base import ApiCallResult, ApiRequest, BaseApiClient


class SentinelHubClient(BaseApiClient):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(
            source_name="sentinel",
            requests_per_second=float(config.get("requests_per_second", 1.0)),
            max_retries=int(config.get("max_retries", 3)),
            backoff_base_s=float(config.get("backoff_base_s", 1.0)),
        )
        self.config = config

    def _token(self) -> str | None:
        client_id = os.getenv("SENTINELHUB_CLIENT_ID")
        client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
        token_url = self.config.get("oauth_token_url")
        if not client_id or not client_secret or not token_url:
            return None

        req = ApiRequest(
            source="sentinel",
            dataset="oauth_token",
            method="POST",
            url=token_url,
            params={},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        # requests.request with json_body is not ideal for form auth, so direct call here.
        import requests

        try:
            resp = requests.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=60,
            )
            if not resp.ok:
                return None
            payload = resp.json()
            return payload.get("access_token")
        except Exception:  # noqa: BLE001
            return None

    def fetch(self, run_params: dict[str, Any]) -> list[ApiCallResult]:
        results: list[ApiCallResult] = []
        search_url = self.config.get("catalog_search_url")
        if not search_url:
            results.append(
                self.build_missing_endpoint_result(
                    dataset="catalog",
                    endpoint="",
                    request_params={},
                    reason="Sentinel Hub catalog search URL missing in config",
                )
            )
            return results

        token = self._token()
        if not token:
            results.append(
                self.build_missing_endpoint_result(
                    dataset="catalog",
                    endpoint=search_url,
                    request_params={},
                    reason="Missing/invalid Sentinel Hub OAuth credentials",
                )
            )
            return results

        bbox = run_params.get("bbox", self.config.get("default_bbox"))
        time_from = run_params.get("time_from", self.config.get("default_time_from"))
        time_to = run_params.get("time_to", self.config.get("default_time_to"))
        limit = min(int(run_params.get("sentinel_limit", self.config.get("limit", 100))), 100)
        collection = run_params.get("sentinel_collection", self.config.get("collection", "sentinel-2-l2a"))
        max_cloud = int(run_params.get("sentinel_max_cloud", self.config.get("max_cloud_coverage", 30)))

        body = {
            "bbox": bbox,
            "datetime": f"{time_from}/{time_to}",
            "collections": [collection],
            "limit": limit,
            "filter": f"eo:cloud_cover < {max_cloud}",
            "filter-lang": "cql2-text",
        }
        req = ApiRequest(
            source="sentinel",
            dataset="catalog",
            method="POST",
            url=search_url,
            json_body=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        res = self.execute(req)
        results.append(res)

        # Cursor pagination if next token is exposed.
        max_pages = int(run_params.get("sentinel_max_pages", self.config.get("max_pages", 1)))
        current = res
        page = 1
        while page < max_pages and current.success:
            payload = current.response_payload
            next_href = None
            if isinstance(payload, dict):
                links = payload.get("links", [])
                if isinstance(links, list):
                    for link in links:
                        if isinstance(link, dict) and link.get("rel") == "next":
                            next_href = link.get("href")
                            break
            if not next_href:
                break
            page += 1
            next_req = ApiRequest(
                source="sentinel",
                dataset="catalog",
                method="GET",
                url=next_href,
                params={},
                headers={"Authorization": f"Bearer {token}"},
                page=page,
            )
            current = self.execute(next_req)
            results.append(current)

        return results

