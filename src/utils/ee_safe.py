from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def safe_ee_call(
    func: Callable[[], T],
    *,
    timeout: int = 120,
    context: str = "EE_CALL",
    max_retries: int = 2,
    backoffs_seconds: tuple[float, ...] = (2.0, 5.0),
    logger: logging.Logger | None = None,
) -> T | None:
    attempts = max(1, 1 + int(max_retries))
    for attempt in range(1, attempts + 1):
        if logger is not None:
            logger.info("[EE][START] %s attempt=%d/%d", context, attempt, attempts)
        box: dict[str, object] = {"done": False, "value": None, "error": None}

        def _runner() -> None:
            try:
                box["value"] = func()
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                box["done"] = True

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()
        worker.join(timeout)

        if not box["done"]:
            if logger is not None:
                logger.warning("[EE][TIMEOUT] %s attempt=%d/%d", context, attempt, attempts)
        elif box["error"] is None:
            if logger is not None:
                logger.info("[EE][SUCCESS] %s attempt=%d/%d", context, attempt, attempts)
            return box["value"]  # type: ignore[return-value]
        elif logger is not None:
            logger.warning("[EE][ERROR] %s attempt=%d/%d error=%s", context, attempt, attempts, repr(box["error"]))

        if attempt < attempts:
            time.sleep(backoffs_seconds[min(attempt - 1, len(backoffs_seconds) - 1)])
    return None
