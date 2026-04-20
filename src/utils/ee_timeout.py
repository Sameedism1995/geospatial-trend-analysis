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
    timeout_seconds: int = 120,
    max_retries: int = 2,
    backoffs_seconds: tuple[float, ...] = (2.0, 5.0),
    logger: logging.Logger | None = None,
    context: str = "",
) -> T | None:
    """Run an EE call with timeout + bounded retries; return None on failure."""
    attempts = max(1, 1 + int(max_retries))
    backoffs = list(backoffs_seconds) if backoffs_seconds else [2.0, 5.0]

    for attempt in range(1, attempts + 1):
        if logger is not None:
            logger.info("[EE][START] context=%s attempt=%d/%d", context or "EE_CALL", attempt, attempts)
        result_box: dict[str, object] = {"done": False, "value": None, "error": None}

        def _runner() -> None:
            try:
                result_box["value"] = func()
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc
            finally:
                result_box["done"] = True

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()
        worker.join(timeout_seconds)
        try:
            if not result_box["done"]:
                if logger is not None:
                    logger.warning("[EE][TIMEOUT] context=%s attempt=%d/%d", context or "EE_CALL", attempt, attempts)
            elif result_box["error"] is None:
                result = result_box["value"]
                if logger is not None:
                    logger.info("[EE][SUCCESS] context=%s attempt=%d/%d", context or "EE_CALL", attempt, attempts)
                return result  # type: ignore[return-value]
            else:
                raise result_box["error"]  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning("[EE][ERROR] context=%s attempt=%d/%d error=%s", context or "EE_CALL", attempt, attempts, repr(exc))
        if attempt < attempts:
            sleep_s = backoffs[min(attempt - 1, len(backoffs) - 1)]
            time.sleep(sleep_s)
    return None
