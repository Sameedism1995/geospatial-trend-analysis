from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .base import ApiCallResult, ApiRequest, BaseApiClient


class HelcomClient(BaseApiClient):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(
            source_name="helcom",
            requests_per_second=float(config.get("requests_per_second", 1.0)),
            max_retries=int(config.get("max_retries", 3)),
            backoff_base_s=float(config.get("backoff_base_s", 1.0)),
        )
        self.config = config

    @staticmethod
    def _extract_record_spatial(xml_text: str, record_id: str) -> dict[str, Any]:
        ns = {
            "gmd": "http://www.isotc211.org/2005/gmd",
            "gco": "http://www.isotc211.org/2005/gco",
        }
        root = ET.fromstring(xml_text)
        title = root.findtext(".//gmd:title/gco:CharacterString", default="", namespaces=ns)

        west = root.findtext(
            ".//gmd:EX_GeographicBoundingBox/gmd:westBoundLongitude/gco:Decimal",
            default="",
            namespaces=ns,
        )
        east = root.findtext(
            ".//gmd:EX_GeographicBoundingBox/gmd:eastBoundLongitude/gco:Decimal",
            default="",
            namespaces=ns,
        )
        south = root.findtext(
            ".//gmd:EX_GeographicBoundingBox/gmd:southBoundLatitude/gco:Decimal",
            default="",
            namespaces=ns,
        )
        north = root.findtext(
            ".//gmd:EX_GeographicBoundingBox/gmd:northBoundLatitude/gco:Decimal",
            default="",
            namespaces=ns,
        )
        bbox = [None, None, None, None]
        try:
            bbox = [float(west), float(south), float(east), float(north)]
        except Exception:  # noqa: BLE001
            pass
        return {"record_id": record_id, "title": title, "bbox": bbox, "crs": "EPSG:4326"}

    def fetch(self, run_params: dict[str, Any]) -> list[ApiCallResult]:
        results: list[ApiCallResult] = []
        formatter_url = self.config.get("record_formatter_xml_endpoint")
        dataset_ids = run_params.get("helcom_dataset_ids", self.config.get("dataset_ids", []))
        if not formatter_url or not dataset_ids:
            results.append(
                self.build_missing_endpoint_result(
                    dataset="spatial_records",
                    endpoint=str(formatter_url),
                    request_params={"dataset_ids": dataset_ids},
                    reason="HELCOM formatter endpoint or dataset IDs missing in config",
                )
            )
            return results

        max_records = int(run_params.get("helcom_max_records", self.config.get("max_records", 10)))
        for record_id in dataset_ids[:max_records]:
            req = ApiRequest(
                source="helcom",
                dataset="spatial_records",
                method="GET",
                url=formatter_url.format(record_id=record_id),
                params={},
                headers={"Accept": "application/xml"},
            )
            res = self.execute(req)
            if res.success and isinstance(res.response_payload, dict) and isinstance(
                res.response_payload.get("text"), str
            ):
                try:
                    record = self._extract_record_spatial(res.response_payload["text"], record_id)
                    res.response_payload = {"records": [record]}
                    res.records_count = 1
                except Exception as exc:  # noqa: BLE001
                    res.success = False
                    res.status = "failed"
                    res.error_message = f"HELCOM XML parse failed: {exc}"
                    res.records_count = 0
            results.append(res)
        return results

