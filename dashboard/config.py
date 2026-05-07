"""
Central registry for dashboard data sources — extend EXTRA_DATA_SOURCES in sources_local.py
or append to CORE_DATA_SOURCES here.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = project_root()

# Distance-band display order used in plots
ZONE_ORDER = ["0-3 km", "3-7 km", "7-15 km", "15-30 km"]

CORE_DATA_SOURCES: list[dict] = [
    {
        "id": "port_decay",
        "rel_path": "outputs/reports/port_distance_decay_statistics.csv",
        "kind": "csv",
        "port_column": "port",
        "category": "exposure",
        "label": "Port distance decay (full pipeline)",
    },
    {
        "id": "port_ranking",
        "rel_path": "outputs/reports/port_exposure_ranking.csv",
        "kind": "csv",
        "port_column": "port",
        "category": "exposure",
        "label": "Port exposure ranking",
    },
    {
        "id": "thesis_decay_tm",
        "rel_path": "outputs/final_run_turku_mariehamn_thesis/reports/port_distance_decay_statistics_turku_mariehamn.csv",
        "kind": "csv",
        "port_column": "port",
        "category": "thesis",
        "label": "Thesis decay (Turku–Mariehamn)",
    },
    {
        "id": "thesis_ranking",
        "rel_path": "outputs/final_run_turku_mariehamn_thesis/reports/port_exposure_ranking_turku_mariehamn.csv",
        "kind": "csv",
        "port_column": "port",
        "category": "thesis",
        "label": "Thesis port ranking",
    },
    {
        "id": "lag_relationships",
        "rel_path": "outputs/final_run_turku_mariehamn_thesis/tables/final_temporal_relationships.csv",
        "kind": "csv",
        "port_column": None,
        "category": "thesis",
        "label": "Temporal / lag relationships",
    },
    {
        "id": "anomaly_overlap",
        "rel_path": "outputs/final_run_turku_mariehamn_thesis/tables/final_anomaly_overlap_table.csv",
        "kind": "csv",
        "port_column": "port",
        "category": "thesis",
        "label": "Anomaly overlap",
    },
    {
        "id": "key_findings",
        "rel_path": "outputs/final_run_turku_mariehamn_thesis/tables/final_key_findings_table.csv",
        "kind": "csv",
        "port_column": None,
        "category": "thesis",
        "label": "Key findings table",
    },
    {
        "id": "features_panel",
        "rel_path": "outputs/processed/features_ml_ready_coastal_wind.parquet",
        "kind": "parquet",
        "port_column": "nearest_port",
        "category": "grid",
        "label": "ML-ready coastal panel (weekly grid)",
        "sample_rows": 8000,
    },
]

FIGURE_ROOTS = [
    "outputs/figures",
    "outputs/final_run_turku_mariehamn_thesis/figures",
    "outputs/final_run_turku_mariehamn_thesis/maps",
    "outputs/visualizations",
]

try:
    from dashboard.sources_local import EXTRA_DATA_SOURCES, EXTRA_FIGURE_ROOTS  # type: ignore
except ImportError:
    EXTRA_DATA_SOURCES = []
    EXTRA_FIGURE_ROOTS = []


def merged_data_sources() -> list[dict]:
    merged = [*CORE_DATA_SOURCES]
    ids = {s["id"] for s in merged}
    for s in EXTRA_DATA_SOURCES:
        if s.get("id") and s["id"] not in ids:
            merged.append(s)
            ids.add(s["id"])
        elif not s.get("id"):
            merged.append(s)
    return merged


def merged_figure_roots() -> list[str]:
    ordered: list[str] = []
    for r in [*FIGURE_ROOTS, *EXTRA_FIGURE_ROOTS]:
        if r not in ordered:
            ordered.append(r)
    return ordered


def resolve_path(rel_path: str) -> Path:
    return ROOT / rel_path


ALL_PORTS_DEFAULT = ["Turku", "Mariehamn", "Stockholm"]
