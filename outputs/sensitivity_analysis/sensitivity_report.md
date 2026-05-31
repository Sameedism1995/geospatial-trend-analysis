# Methodological sensitivity analysis

Exploratory robustness checks for the thesis composite **Environmental Stress Index (ESI)** 
and **shoreward wind-regime** classification. Computed from `processed/features_ml_ready.parquet` 
with wind fields merged from `outputs/reports/run_coastal_wind_transport/coastal_wind_alignment_features.csv` when present.

---

## Sensitivity Test 1: Environmental Stress Index (ESI)

### 1.1 Current (original) ESI calculation

**Location:** `src/analysis/run_coastal_exposure_analysis.py`, function `build_indices()`.

For each grid-week, the pipeline builds **weekly z-scores** (within calendar week) for:

- `local_no2_excess` (NO₂ relative to 15–30 km shipping-band weekly baseline)
- `vessel_density_t`
- `ndti_weekly_anomaly` (when available)
- `oil_slick_probability_t` (when available)
- `(rank_pct(coastal_wind_alignment_score mapped to [0,1]) − 0.5)` — wind alignment term

The **original ESI** is the **rank percentile** (0–1) of the **equal-weight arithmetic mean** 
of those components (`environmental_stress_index`).

> **Note:** Figure 5.8 in `scripts/generate_thesis_sections_5_5_to_5_10.py` uses a *different* 
> six-variable global z-mean (MEI, NO₂, FAI, NDTI, vessel density, coastal exposure score). 
> This sensitivity test uses the **coastal-exposure pipeline ESI** above, which matches RQ evidence 
> enrichment and `build_indices` outputs.

### 1.2 Alternative ESI (weighted, no wind term)

Weekly z-scores of the same four environmental/maritime inputs, combined as a **weighted sum**:

| Component | Weight |
|-----------|--------|
| `local_no2_excess` (NO₂) | 0.35 |
| `vessel_density_t` | 0.35 |
| `ndti_weekly_anomaly` (NDTI) | 0.15 |
| `oil_slick_probability_t` | 0.15 |

Then converted to rank percentile → `environmental_stress_index_alt` (0–1 scale, comparable to original).

### 1.3 Comparison (grid-week level)

- Panel rows with both indices: **2,050** (full panel **16,065**; coastal_panel **1,479**)
- **Spearman ρ:** 0.9187
- **Pearson r:** 0.9157
- **Mean absolute difference** (0–1 scale): 0.0930

### 1.4 Hotspot overlap (top 10% cells)

Cell-level means of each ESI were computed per `grid_cell_id`; cells in the **top 10%** of each 
formulation were compared.

- Cells with both means: **231**
- Hot cells (original / alternative): **24** / **24**
- Overlap count: **15**
- Overlap as fraction of original hotspots: **62.5%**
- Jaccard index: **0.455**

### 1.5 Turku vs Mariehamn rankings (mean cell ESI)

**Original ESI** (`environmental_stress_index`):
- Turku: mean=0.5737, median=0.6069, n_cells=45, rank=1
- Mariehamn: mean=0.4740, median=0.4518, n_cells=114, rank=2

**Alternative ESI** (`environmental_stress_index_alt`):
- Turku: mean=0.6065, median=0.5612, n_cells=40, rank=1
- Mariehamn: mean=0.4065, median=0.3626, n_cells=109, rank=2

**Ranking stability:** Turku ranks above Mariehamn under **both** formulations (original: Turku higher; alternative: Turku higher).

---

## Sensitivity Test 2: Wind-regime classification

### 2.1 Current threshold

**Location:** `src/analysis/run_coastal_wind_transport.py` (and `prepare_panel` in coastal exposure).

- `coastal_wind_alignment_score = cos(angle_between_wind_to_direction and bearing_cell_to_coast)`
- **Shoreward** if `coastal_wind_alignment_score ≥ cos(45°) ≈ 0.7071` (`coastal_wind_shoreward_45deg = 1`)
- **Non-shoreward** otherwise (among rows with valid alignment)

### 2.2 Alternative thresholds

Same rule with **±30°** (`cos(30°) ≈ 0.866`) and **±60°** (`cos(60°) = 0.5`).

Analysis subset: all grid-weeks with finite `coastal_wind_alignment_score` (**1,479** rows at 45°; see CSV for counts per threshold).

### 2.3 Mann–Whitney (shoreward vs non-shoreward)

#### Threshold ±30°
- Shoreward: **288** · Non-shoreward: **1,191**
- **NO2 (no2_mean_t)**: mean diff (shoreward − non) = -6.343e-07, p = 0.2463 (n.s.), direction = shoreward_lower
- **ACEI**: mean diff (shoreward − non) = 0.3545, p = 4.103e-70 (p<0.05), direction = shoreward_higher
- **ESI (original)**: mean diff (shoreward − non) = 0.1154, p = 2.893e-18 (p<0.05), direction = shoreward_higher

#### Threshold ±45°
- Shoreward: **442** · Non-shoreward: **1,037**
- **NO2 (no2_mean_t)**: mean diff (shoreward − non) = -1.496e-06, p = 0.007281 (p<0.05), direction = shoreward_lower
- **ACEI**: mean diff (shoreward − non) = 0.3688, p = 1.537e-98 (p<0.05), direction = shoreward_higher
- **ESI (original)**: mean diff (shoreward − non) = 0.1055, p = 2.144e-20 (p<0.05), direction = shoreward_higher

#### Threshold ±60°
- Shoreward: **566** · Non-shoreward: **913**
- **NO2 (no2_mean_t)**: mean diff (shoreward − non) = -1.38e-06, p = 0.007611 (p<0.05), direction = shoreward_lower
- **ACEI**: mean diff (shoreward − non) = 0.3922, p = 1.64e-124 (p<0.05), direction = shoreward_higher
- **ESI (original)**: mean diff (shoreward − non) = 0.109, p = 6.699e-24 (p<0.05), direction = shoreward_higher

### 2.4 Directional consistency across thresholds

- **NO2 (no2_mean_t):** directions = ['shoreward_lower', 'shoreward_lower', 'shoreward_lower']; significant at α=0.05 = [False, True, True]
- **ACEI:** directions = ['shoreward_higher', 'shoreward_higher', 'shoreward_higher']; significant at α=0.05 = [True, True, True]
- **ESI (original):** directions = ['shoreward_higher', 'shoreward_higher', 'shoreward_higher']; significant at α=0.05 = [True, True, True]

---

## Overall conclusion

Main **spatial patterns** are **moderately robust** to alternative ESI weights (ρ≥0.7), but hotspot membership and magnitudes shift somewhat; thesis language should keep ESI **experimental** and avoid over-interpreting precise hotspot boundaries.

Wind-threshold sensitivity: the **sign** of the shoreward vs non-shoreward NO₂ contrast is **consistent** across ±30°, ±45°, and ±60° (shoreward_lower). Stricter thresholds (±30°) classify fewer weeks as shoreward; looser thresholds (±60°) classify more.

**Thesis-safe summary:** Distance-decay, Turku–Mariehamn vessel-density contrast, negative ML test R², and associative (non-causal) framing are **not overturned** by these reasonable methodological variants. Claims tied to **exact ESI hotspot maps** or **wind-conditioned ESI differences** should remain hedged.

---

## Artefacts

- `esi_gridweek_comparison.csv` — row-level original vs alternative ESI
- `esi_cell_means.csv` — per-cell means for hotspot analysis
- `esi_port_rankings.csv` — Turku / Mariehamn mean cell ESI and ranks
- `wind_threshold_mannwhitney.csv` — counts and tests per threshold

Regenerate: `python3 scripts/run_sensitivity_analysis.py`
