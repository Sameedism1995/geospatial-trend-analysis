"""
Export **all** cells from every CSV under `outputs/reports/`, every flattened
leaf from JSON there, plus `outputs/eda/eda_stats.json` and
`data/validation/full_pipeline_validation_report.json`, into one long table.

    PYTHONPATH=src python3 src/export_all_statistics_summary.py

Smaller export (only statistic-like column names):

    PYTHONPATH=src python3 src/export_all_statistics_summary.py --stat-columns-only

Outputs (default under outputs/reports/):
  - ALL_STATISTICS_LONG.csv
  - ALL_STATISTICS_INVENTORY.md
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Used only with --stat-columns-only: limit to inferential / summary-style names.
STAT_PATTERN = re.compile(
    r"(^|_)(pvalue|p_value|p_val|prob|sig|"
    r"welch|mann|whitney|anova|ttest|f_stat|t_stat|"
    r"cohen|effect|mean|median|mode|std|sem|var|rho|tau|"
    r"pearson|spearman|kendall|correlation|corr|"
    r"r2|r_squared|rmse|mae|mse|mape|"
    r"ci_lower|ci_upper|ci_|q25|q75|iqr|"
    r"statistic|value|score|rate|ratio|diff)"
    r"|(_p$|^_p_|^rho|^r$|^n$|^df$)",
    re.I,
)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _flatten_json(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(_flatten_json(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            rows.extend(_flatten_json(v, f"{prefix}[{i}]"))
    else:
        rows.append((prefix or "root", obj))
    return rows


def collect_from_csv(path: Path, *, stat_columns_only: bool) -> tuple[pd.DataFrame, int, str | None]:
    """Melt CSV to long form: every column (default) or stat-like columns only."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), 0, str(exc)[:200]
    n = len(df)
    if df.empty:
        return pd.DataFrame(), n, "empty"

    if stat_columns_only:
        use_cols = [c for c in df.columns if STAT_PATTERN.search(str(c))]
        if not use_cols:
            return pd.DataFrame(), n, "no_stat_columns_matched"
    else:
        use_cols = list(df.columns)

    rel = _rel(path)
    slim = df[use_cols].copy()
    # Avoid melt name clash if a source column is literally named "cell_value"
    renamed = {c: f"__src__{c}" for c in slim.columns}
    slim = slim.rename(columns=renamed)
    slim.insert(0, "_row", range(len(slim)))
    melted = slim.melt(id_vars=["_row"], var_name="column", value_name="cell_value")
    melted["column"] = melted["column"].str.replace(r"^__src__", "", regex=True)
    melted["source_file"] = rel
    melted["source_format"] = "csv"
    return melted.rename(columns={"_row": "row_index"}), n, None


def collect_from_json(path: Path, *, stat_keys_only: bool) -> pd.DataFrame:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    flat = _flatten_json(raw)
    rows = []
    rel = _rel(path)
    for key, val in flat:
        if stat_keys_only and not STAT_PATTERN.search(str(key)):
            continue
        if isinstance(val, (dict, list)):
            continue
        rows.append(
            {
                "source_file": rel,
                "source_format": "json",
                "row_index": 0,
                "column": key,
                "cell_value": val,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "reports",
        help="Directory for ALL_STATISTICS_* outputs.",
    )
    parser.add_argument(
        "--skip-eda-stats",
        action="store_true",
        help="Do not merge outputs/eda/eda_stats.json.",
    )
    parser.add_argument(
        "--stat-columns-only",
        action="store_true",
        help="Only include CSV/JSON fields whose names match p-value / mean / correlation-style patterns (smaller file).",
    )
    parser.add_argument(
        "--no-include-report-json",
        action="store_true",
        help="Skip *.json under outputs/reports (still includes eda_stats + validation JSON unless disabled).",
    )
    args = parser.parse_args()
    include_report_json = not args.no_include_report_json

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_roots = [
        ROOT / "outputs" / "reports",
    ]

    all_parts: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for root in report_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if path.name.startswith("ALL_STATISTICS"):
                continue
            long_df, n_rows, note = collect_from_csv(path, stat_columns_only=args.stat_columns_only)
            entry: dict[str, Any] = {
                "path": _rel(path),
                "rows_in_file": n_rows,
                "unique_columns": int(long_df["column"].nunique()) if not long_df.empty else 0,
                "cells_exported": int(len(long_df)),
            }
            if long_df.empty:
                entry["note"] = note or ""
                if note and note not in ("empty", "no_stat_columns_matched"):
                    skipped.append((_rel(path), note))
            else:
                all_parts.append(long_df)
            inventory.append(entry)

    j_kw = {"stat_keys_only": args.stat_columns_only}

    eda_stats = ROOT / "outputs" / "eda" / "eda_stats.json"
    if not args.skip_eda_stats and eda_stats.exists():
        jdf = collect_from_json(eda_stats, **j_kw)
        if not jdf.empty:
            all_parts.append(jdf)
        inventory.append(
            {
                "path": _rel(eda_stats),
                "rows_in_file": 1,
                "unique_columns": int(jdf["column"].nunique()) if not jdf.empty else 0,
                "cells_exported": len(jdf),
                "note": "eda_stats.json (flattened leaves)",
            }
        )

    validation_json = ROOT / "data" / "validation" / "full_pipeline_validation_report.json"
    if validation_json.exists():
        jdf = collect_from_json(validation_json, **j_kw)
        if not jdf.empty:
            all_parts.append(jdf)
        inventory.append(
            {
                "path": _rel(validation_json),
                "rows_in_file": 1,
                "unique_columns": int(jdf["column"].nunique()) if not jdf.empty else 0,
                "cells_exported": len(jdf),
            }
        )

    if include_report_json:
        report_json_root = ROOT / "outputs" / "reports"
        if report_json_root.exists():
            for jpath in sorted(report_json_root.rglob("*.json")):
                jdf = collect_from_json(jpath, **j_kw)
                entry = {
                    "path": _rel(jpath),
                    "rows_in_file": 1,
                    "unique_columns": int(jdf["column"].nunique()) if not jdf.empty else 0,
                    "cells_exported": len(jdf),
                }
                if jdf.empty:
                    entry["note"] = "no_primitive_leaves" if not args.stat_columns_only else "no_stat_keys_matched"
                else:
                    all_parts.append(jdf)
                inventory.append(entry)

    if not all_parts:
        stub = pd.DataFrame(
            columns=["source_file", "row_index", "column", "cell_value", "source_format"]
        )
        stub.to_csv(out_dir / "ALL_STATISTICS_LONG.csv", index=False)
        inv = out_dir / "ALL_STATISTICS_INVENTORY.md"
        inv.write_text(
            "# Statistics export\n\nNo statistical columns found under outputs/reports/.\n",
            encoding="utf-8",
        )
        print("[WARN] No data aggregated. Run analyses or check paths.")
        return 1

    out = pd.concat(all_parts, ignore_index=True)
    out["cell_value"] = out["cell_value"].apply(lambda x: x if pd.isna(x) else str(x))
    out.to_csv(out_dir / "ALL_STATISTICS_LONG.csv", index=False)

    mode = "**stat-like columns only**" if args.stat_columns_only else "**all columns / all JSON leaves**"
    inv_lines = [
        "# All pipeline report results (long export)",
        "",
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Mode: {mode}",
        f"- Long-table rows: **{len(out)}** (one row per CSV cell or JSON leaf)",
        f"- Source files scanned: **{len(inventory)}**",
        "",
        "## How to read `ALL_STATISTICS_LONG.csv`",
        "",
        "- **source_file**: report path relative to project root.",
        "- **row_index**: row number in the original CSV (0-based); JSON sources use 0.",
        "- **column**: original column name or flattened JSON key path.",
        "- **cell_value**: cell value (stringified).",
        "",
        "By default this file includes **every** column from every CSV under "
        "`outputs/reports/`, all matching `*.json` there, plus `outputs/eda/eda_stats.json` "
        "and `data/validation/full_pipeline_validation_report.json`. "
        "Use `--stat-columns-only` for a smaller table that keeps only names matching "
        "typical statistics (p-values, means, correlations, etc.).",
        "",
        "## Source file inventory",
        "",
        "| File | Rows in file | Unique columns | Cells exported | Note |",
        "|------|-------------:|---------------:|---------------:|------|",
    ]
    for row in sorted(inventory, key=lambda r: str(r.get("path", ""))):
        p = row.get("path", "")
        rc = row.get("rows_in_file", "")
        nu = row.get("unique_columns", "")
        nc = row.get("cells_exported", "")
        note = row.get("note", "")
        inv_lines.append(f"| `{p}` | {rc} | {nu} | {nc} | {note} |")

    inv_lines.extend(
        [
            "",
            "## Files skipped or with parse errors",
            "",
        ]
    )
    if skipped:
        for p, msg in skipped[:50]:
            inv_lines.append(f"- `{p}`: {msg}")
        if len(skipped) > 50:
            inv_lines.append(f"- … and {len(skipped) - 50} more")
    else:
        inv_lines.append("- (none)")

    (out_dir / "ALL_STATISTICS_INVENTORY.md").write_text("\n".join(inv_lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote {out_dir / 'ALL_STATISTICS_LONG.csv'} ({len(out)} rows)")
    print(f"[OK] Wrote {out_dir / 'ALL_STATISTICS_INVENTORY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
