from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from .base import ApiCallResult, ApiRequest, BaseApiClient


class EmodnetClient(BaseApiClient):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(
            source_name="emodnet",
            requests_per_second=float(config.get("requests_per_second", 1.0)),
            max_retries=int(config.get("max_retries", 3)),
            backoff_base_s=float(config.get("backoff_base_s", 1.0)),
        )
        self.config = config

    @staticmethod
    def _extract_layers_from_wms(xml_text: str) -> list[dict[str, Any]]:
        ns = {"wms": "http://www.opengis.net/wms"}
        root = ET.fromstring(xml_text)
        out: list[dict[str, Any]] = []

        for layer in root.findall(".//wms:Layer", ns):
            name_el = layer.find("wms:Name", ns)
            title_el = layer.find("wms:Title", ns)
            ex = layer.find("wms:EX_GeographicBoundingBox", ns)
            if name_el is None:
                continue
            minx = miny = maxx = maxy = None
            if ex is not None:
                try:
                    minx = float(ex.findtext("wms:westBoundLongitude", default="", namespaces=ns))
                    miny = float(ex.findtext("wms:southBoundLatitude", default="", namespaces=ns))
                    maxx = float(ex.findtext("wms:eastBoundLongitude", default="", namespaces=ns))
                    maxy = float(ex.findtext("wms:northBoundLatitude", default="", namespaces=ns))
                except Exception:  # noqa: BLE001
                    minx = miny = maxx = maxy = None
            if minx is None:
                # Fallback: any BoundingBox with numeric bounds.
                for bb in layer.findall("wms:BoundingBox", ns):
                    try:
                        minx = float(bb.attrib.get("minx"))
                        miny = float(bb.attrib.get("miny"))
                        maxx = float(bb.attrib.get("maxx"))
                        maxy = float(bb.attrib.get("maxy"))
                        break
                    except Exception:  # noqa: BLE001
                        continue
            out.append(
                {
                    "name": name_el.text,
                    "title": title_el.text if title_el is not None else None,
                    "bbox": [minx, miny, maxx, maxy],
                    "crs": "EPSG:4326",
                }
            )
        return out

    def fetch(self, run_params: dict[str, Any]) -> list[ApiCallResult]:
        results: list[ApiCallResult] = []
        wms_urls = self.config.get("wms_capabilities_urls", [])
        if not wms_urls:
            results.append(
                self.build_missing_endpoint_result(
                    dataset="wms_layers",
                    endpoint="",
                    request_params={},
                    reason="EMODnet WMS capabilities URLs missing in config",
                )
            )
            return results

        for url in wms_urls:
            req = ApiRequest(
                source="emodnet",
                dataset="wms_layers",
                method="GET",
                url=url,
                params={},
                headers={"Accept": "text/xml"},
            )
            res = self.execute(req)
            if res.success and isinstance(res.response_payload, dict) and isinstance(
                res.response_payload.get("text"), str
            ):
                try:
                    layers = self._extract_layers_from_wms(res.response_payload["text"])
                    res.response_payload = {"layers": layers, "service_url": url}
                    res.records_count = len(layers)
                except Exception as exc:  # noqa: BLE001
                    res.success = False
                    res.status = "failed"
                    res.error_message = f"EMODnet WMS parse failed: {exc}"
                    res.records_count = 0
            results.append(res)
        return results

