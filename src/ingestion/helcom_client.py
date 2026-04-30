from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from typing import Any

from .base import ApiCallResult, ApiRequest, BaseApiClient


# ISO 19139 / GeoNetwork namespaces used by HELCOM metadata XML.
_XML_NS = {
    "gmd": "http://www.isotc211.org/2005/gmd",
    "gco": "http://www.isotc211.org/2005/gco",
    "gmx": "http://www.isotc211.org/2005/gmx",
}


def _findall_texts(root: ET.Element, path: str, ns: dict[str, str]) -> list[str]:
    out: list[str] = []
    for el in root.findall(path, namespaces=ns):
        text = (el.text or "").strip()
        if text:
            out.append(text)
    return out


def _normalize_iso_date(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC).isoformat()
    except Exception:  # noqa: BLE001
        try:
            ts = datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=UTC)
            return ts.isoformat()
        except Exception:  # noqa: BLE001
            return None


def _polygon_wkt_from_bbox(bbox: list[float | None]) -> str | None:
    try:
        minx, miny, maxx, maxy = (float(v) for v in bbox[:4])
    except Exception:  # noqa: BLE001
        return None
    # WGS84 polygon, counter-clockwise, closed.
    coords = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]
    coord_str = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in coords)
    return f"POLYGON (({coord_str}))"


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
        """Parse a HELCOM GeoNetwork ISO-19139 record.

        Returns a rich, spatially-grounded dict. Previous versions only captured
        `title` + `bbox`; this version additionally surfaces abstract, keywords,
        topic category, organisation, lineage, publication/revision dates and a
        real POLYGON WKT built from the ISO bounding box. Every field is
        optional and only appears when the XML carries it.
        """
        ns = _XML_NS
        try:
            root = ET.fromstring(xml_text)
        except Exception:  # noqa: BLE001
            return {"record_id": record_id, "title": "", "bbox": [None, None, None, None], "crs": "EPSG:4326"}

        title = root.findtext(".//gmd:identificationInfo//gmd:citation//gmd:title/gco:CharacterString", default="", namespaces=ns) or root.findtext(
            ".//gmd:title/gco:CharacterString", default="", namespaces=ns
        )
        abstract = root.findtext(".//gmd:identificationInfo//gmd:abstract/gco:CharacterString", default="", namespaces=ns)
        purpose = root.findtext(".//gmd:identificationInfo//gmd:purpose/gco:CharacterString", default="", namespaces=ns)

        topic_category = root.findtext(
            ".//gmd:identificationInfo//gmd:topicCategory/gmd:MD_TopicCategoryCode",
            default="",
            namespaces=ns,
        )

        organisation = root.findtext(
            ".//gmd:contact//gmd:CI_ResponsibleParty/gmd:organisationName/gco:CharacterString",
            default="",
            namespaces=ns,
        )

        lineage = root.findtext(
            ".//gmd:dataQualityInfo//gmd:LI_Lineage//gmd:statement/gco:CharacterString",
            default="",
            namespaces=ns,
        )

        keywords = _findall_texts(
            root,
            ".//gmd:descriptiveKeywords//gmd:keyword/gco:CharacterString",
            ns,
        )

        # ISO gmd:dateStamp is the file-level metadata date; citation dates are
        # the dataset publication/revision/creation dates.
        dataset_stamp = _normalize_iso_date(
            root.findtext(".//gmd:dateStamp/gco:Date", default="", namespaces=ns)
            or root.findtext(".//gmd:dateStamp/gco:DateTime", default="", namespaces=ns)
        )
        citation_dates: dict[str, str | None] = {}
        for ci in root.findall(".//gmd:identificationInfo//gmd:citation//gmd:date/gmd:CI_Date", namespaces=ns):
            code_el = ci.find(".//gmd:CI_DateTypeCode", namespaces=ns)
            code = (code_el.get("codeListValue") if code_el is not None else None) or "date"
            iso = _normalize_iso_date(
                ci.findtext(".//gmd:date/gco:Date", default="", namespaces=ns)
                or ci.findtext(".//gmd:date/gco:DateTime", default="", namespaces=ns)
            )
            if iso:
                citation_dates[str(code)] = iso

        # Primary record timestamp preference:
        #   publication > revision > creation > metadata dateStamp.
        record_timestamp = (
            citation_dates.get("publication")
            or citation_dates.get("revision")
            or citation_dates.get("creation")
            or dataset_stamp
        )

        # Bounding box.
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
        bbox: list[float | None] = [None, None, None, None]
        try:
            bbox = [float(west), float(south), float(east), float(north)]
        except Exception:  # noqa: BLE001
            pass

        # Temporal extent (observation period).
        temporal_begin = _normalize_iso_date(
            root.findtext(".//gmd:EX_TemporalExtent//gml:beginPosition", default="", namespaces={**ns, "gml": "http://www.opengis.net/gml"})
        )
        temporal_end = _normalize_iso_date(
            root.findtext(".//gmd:EX_TemporalExtent//gml:endPosition", default="", namespaces={**ns, "gml": "http://www.opengis.net/gml"})
        )

        geometry_wkt = _polygon_wkt_from_bbox(bbox)

        payload: dict[str, Any] = {
            "record_id": record_id,
            "title": title,
            "abstract": abstract,
            "purpose": purpose,
            "topic_category": topic_category,
            "organisation": organisation,
            "lineage": lineage,
            "keywords": keywords,
            "bbox": bbox,
            "crs": "EPSG:4326",
            "geometry_wkt": geometry_wkt,
            "record_timestamp_utc": record_timestamp,
            "metadata_stamp_utc": dataset_stamp,
            "citation_dates": citation_dates,
            "temporal_extent": {"begin": temporal_begin, "end": temporal_end},
        }
        return payload

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
