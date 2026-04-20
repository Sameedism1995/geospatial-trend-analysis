from __future__ import annotations

import ee
import logging


PROJECT_ID = "clawd-bot-485923"
LOGGER = logging.getLogger("ee_init")
_EE_INITIALIZED = False


def init_ee() -> None:
    """
    Safe Earth Engine initialization.
    Must be used by ALL EE-based data sources.
    """
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Earth Engine initialization failed: {exc}") from exc


def safe_initialize_ee(ee_module=None) -> None:
    """
    Initialize Earth Engine exactly once per process.
    Must never be wrapped in timeout wrappers.
    """
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        print("[EE] Initialization successful")
        return

    ee_lib = ee_module or ee
    try:
        ee_lib.Initialize(project=PROJECT_ID)
        _EE_INITIALIZED = True
        LOGGER.info("[EE] Initialization successful")
        print("[EE] Initialization successful")
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("[EE] Initialization failed: %s", str(exc))
        print(f"[EE] Initialization failed: {exc}")
        raise RuntimeError(f"Earth Engine initialization failed: {exc}") from exc
