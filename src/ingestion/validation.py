from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_PLACEHOLDER_TOKENS = {"synthetic", "placeholder", "fake_data"}


def validate_no_fake_data(processed_files: list[Path]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    checked_files = 0
    for path in processed_files:
        if not path.exists():
            continue
        checked_files += 1
        df = pd.read_parquet(path)
        if df.empty:
            continue
        if "payload" not in df.columns:
            continue
        for idx, payload in enumerate(df["payload"].astype(str)):
            lower = payload.lower()
            hits = [token for token in FORBIDDEN_PLACEHOLDER_TOKENS if token in lower]
            if hits:
                violations.append(
                    {
                        "file": str(path),
                        "row_index": idx,
                        "matched_tokens": hits,
                    }
                )
    return {
        "checked_files": checked_files,
        "violations_count": len(violations),
        "violations": violations,
        "status": "pass" if not violations else "fail",
    }

