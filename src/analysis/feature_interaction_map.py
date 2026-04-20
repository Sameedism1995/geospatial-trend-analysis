from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_category_columns(df: pd.DataFrame, feature_registry: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    category_columns: dict[str, list[str]] = {}
    for category, entries in feature_registry.items():
        cols: list[str] = []
        for entry in entries:
            col = entry.get("resolved_column")
            if col and col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                if series.notna().any() and float(series.var(skipna=True)) != 0.0:
                    cols.append(col)
        # Keep stable order while removing duplicates.
        category_columns[category] = list(dict.fromkeys(cols))
    return category_columns


def _build_cross_category_pairs(category_columns: dict[str, list[str]]) -> list[tuple[str, str, str, str]]:
    categories = [c for c, cols in category_columns.items() if cols]
    pairs: list[tuple[str, str, str, str]] = []
    for i, left_cat in enumerate(categories):
        for right_cat in categories[i + 1 :]:
            for left_col in category_columns[left_cat]:
                for right_col in category_columns[right_cat]:
                    pairs.append((left_col, right_col, left_cat, right_cat))
    return pairs


def _interpret_label(value: float) -> str:
    abs_v = abs(value)
    if value >= 0.6:
        return "strong positive"
    if abs_v >= 0.3:
        return "moderate"
    if abs_v >= 0.1:
        return "weak"
    return "negligible"


def _plot_heatmap(
    corr_df: pd.DataFrame,
    ordered_columns: list[str],
    column_category: dict[str, str],
    output_path: Path,
) -> None:
    matrix = corr_df.loc[ordered_columns, ordered_columns].values
    fig, ax = plt.subplots(figsize=(12, 10))
    img = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered_columns)))
    ax.set_xticklabels(ordered_columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered_columns)))
    ax.set_yticklabels(ordered_columns, fontsize=8)
    ax.set_title("Feature Interaction Map (Cross-Category Correlation)")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")

    # Draw separators between category groups.
    category_boundaries: list[int] = []
    last = None
    for idx, col in enumerate(ordered_columns):
        cat = column_category[col]
        if last is not None and cat != last:
            category_boundaries.append(idx - 0.5)
        last = cat
    for boundary in category_boundaries:
        ax.axhline(boundary, color="black", linewidth=0.8)
        ax.axvline(boundary, color="black", linewidth=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _plot_top_scatter(
    df: pd.DataFrame,
    top_rows: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in top_rows.iterrows():
        left = row["feature_1"]
        right = row["feature_2"]
        corr = float(row["correlation_value"])
        subset = df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < 3:
            continue
        x = subset[left].values
        y = subset[right].values

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.5, s=14)
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, linewidth=1.5)
        ax.set_xlabel(left)
        ax.set_ylabel(right)
        ax.set_title(f"{left} vs {right} (r={corr:.3f})")
        fig.tight_layout()
        safe_name = f"scatter_{left}_vs_{right}".replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"{safe_name}.png", dpi=220)
        plt.close(fig)


def run_feature_interaction_map(
    df: pd.DataFrame,
    feature_registry: dict[str, list[dict[str, Any]]],
    logger,
    *,
    exclude_within_category: bool = True,
) -> None:
    category_columns = _extract_category_columns(df, feature_registry)
    active_categories = [c for c, cols in category_columns.items() if cols]
    if len(active_categories) < 2:
        logger.warning("[FEATURE_INTERACTION] Skipped: fewer than 2 categories with usable numeric features.")
        return

    column_category = {col: cat for cat, cols in category_columns.items() for col in cols}
    ordered_columns = [col for cat in active_categories for col in category_columns[cat]]
    numeric_df = df[ordered_columns].apply(pd.to_numeric, errors="coerce")
    corr_df = numeric_df.corr(method="pearson")

    pairs = _build_cross_category_pairs(category_columns)
    if not pairs:
        logger.warning("[FEATURE_INTERACTION] Skipped: no cross-category feature pairs available.")
        return

    ranked_rows: list[dict[str, Any]] = []
    for left_col, right_col, left_cat, right_cat in pairs:
        corr_value = corr_df.loc[left_col, right_col]
        if pd.isna(corr_value):
            continue
        ranked_rows.append(
            {
                "feature_1": left_col,
                "feature_2": right_col,
                "correlation_value": float(corr_value),
                "category_pair": f"{left_cat} ↔ {right_cat}",
                "interpretation_label": _interpret_label(float(corr_value)),
            }
        )

    if not ranked_rows:
        logger.warning("[FEATURE_INTERACTION] Skipped: all cross-category correlations are NaN.")
        return

    ranked = pd.DataFrame(ranked_rows)
    ranked = ranked.reindex(ranked["correlation_value"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    reports_dir = Path("outputs/reports")
    plots_dir = Path("outputs/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(reports_dir / "feature_interactions_ranked.csv", index=False)

    heatmap_df = corr_df.copy()
    if exclude_within_category:
        for left in ordered_columns:
            for right in ordered_columns:
                if left == right:
                    continue
                if column_category[left] == column_category[right]:
                    heatmap_df.loc[left, right] = np.nan
    _plot_heatmap(
        corr_df=heatmap_df,
        ordered_columns=ordered_columns,
        column_category=column_category,
        output_path=plots_dir / "feature_interaction_map.png",
    )

    _plot_top_scatter(
        df=df,
        top_rows=ranked.head(3),
        output_dir=plots_dir / "feature_interactions",
    )

    logger.info("[FEATURE_INTERACTION] Wrote ranked interactions: %s", reports_dir / "feature_interactions_ranked.csv")
    logger.info("[FEATURE_INTERACTION] Wrote interaction map: %s", plots_dir / "feature_interaction_map.png")
