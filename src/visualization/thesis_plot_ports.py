"""Port filters for thesis figures — Stockholm excluded from all plots."""

from __future__ import annotations

import pandas as pd

EXCLUDED_PORTS: tuple[str, ...] = ("Stockholm",)
THESIS_PORTS: tuple[str, ...] = ("Turku", "Mariehamn")
THESIS_PORTS_WITH_NAANTALI: tuple[str, ...] = ("Turku", "Naantali", "Mariehamn")

PORT_COORDS: dict[str, tuple[float, float]] = {
    "Turku": (60.435, 22.225),
    "Mariehamn": (60.0973, 19.9348),
    "Naantali": (60.454, 22.094),
}

PORT_COLORS: dict[str, str] = {
    "Turku": "#e69f00",
    "Mariehamn": "#009e73",
    "Naantali": "#cc79a7",
}


def exclude_ports(df: pd.DataFrame, col: str = "nearest_port") -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df.loc[~df[col].astype(str).isin(EXCLUDED_PORTS)].copy()


def filter_port_list(ports: list[str] | tuple[str, ...]) -> list[str]:
    return [p for p in ports if p not in EXCLUDED_PORTS]
