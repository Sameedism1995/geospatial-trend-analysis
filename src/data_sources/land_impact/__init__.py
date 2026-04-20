"""Land Impact Extension Layer — data sources subpackage.

Additive, non-invasive modules that extend the maritime/atmospheric/water-quality
pipeline with land-side observations (NDVI-on-land) to enable land-sea interaction
analysis. Nothing in this package modifies existing water or S1/NO2 pipelines.
"""

from __future__ import annotations

__all__ = ["sentinel2_land_metrics"]
