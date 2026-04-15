# Modeling Dataset Schema

`data/modeling_dataset.parquet` is a **weekly panel** derived from `data/master_dataset.parquet` for supervised learning with a **delta turbidity (NDTI)** target. One row per (`grid_cell_id`, `week_start_utc`).

This file is **not** a trained model output; it contains only features, target, and metadata. No scaling, normalization, feature selection, or imputation is applied here.

---

## Grain and ordering

- **Row grain:** one row per grid cell × week (same balanced panel as the master dataset).
- **Temporal ordering:** weeks are Monday-start UTC, spaced by 7 days, consistent with the master panel.
- **Strict leakage rule:** all feature columns use information from **week t and earlier** (`t`, `t−1`, `t−2` relative to that row’s `week_start_utc`). The only use of week **t+1** is inside the **target** definition below (not as a feature).

---

## Target

| Column | Definition |
|--------|------------|
| `delta_ndti` | `sentinel_ndti_mean(t+1) − sentinel_ndti_mean(t)` for the same `grid_cell_id`, where both values are **non-null**. |
| `has_valid_delta_ndti` | Boolean: true iff `delta_ndti` is defined (finite). Rows with false typically include the **last week per grid** (no t+1) or missing NDTI at t or t+1. |

**Not used as a target:** NDCI (and any other index) is not a label in this dataset.

There is **no interpolation** of missing spectral values for the target: if either endpoint is missing, `delta_ndti` is null.

---

## Feature construction (lags)

For each spectral variable, master column `sentinel_<index>_mean` is turned into three columns:

| Suffix | Meaning |
|--------|---------|
| `_t` | Value at the row’s week **t** |
| `_t_minus_1` | Value at **t − 1 week** (same grid) |
| `_t_minus_2` | Value at **t − 2 weeks** (same grid) |

**Spectral indices (Sentinel-2 means, from Statistics API):**

- `sentinel_ndvi_mean_*`
- `sentinel_ndwi_mean_*`
- `sentinel_evi_mean_*`
- `sentinel_ndti_mean_*`

**Observation count (current week only):**

- `sentinel_observation_count_t` — integer count at **t** (missing counts treated as 0 when building from master).

**Vessel traffic (raster-derived):**

- `vessel_density_t`, `vessel_density_t_minus_1`, `vessel_density_t_minus_2`

These are obtained by **sampling EMODnet Human Activities vessel density GeoTIFFs** (annual layers) at the grid cell centroid: for each row, the **calendar year** of week **t**, **t−1**, and **t−2** selects the corresponding `vesseldensity_*_<year>.tif` stacks. Values from all vessel-category rasters for that year are **summed** at the centroid to approximate total sampled density (see script `src/build_modeling_dataset.py`). If TIFFs are absent under `data/downloads/emodnet_extracted/`, these columns are all null.

**Static geography (replicated each week, no temporal leakage):**

- `grid_res_deg`
- `grid_centroid_lat`, `grid_centroid_lon`

**Temporal encodings:**

- `week_of_year` — ISO week number (typically 1–53).
- `week_sin`, `week_cos` — `sin` / `cos` of `2π × (week_of_year − 1) / 53` for smooth seasonality (53 used as a stable normalizer).

---

## Excluded / not in this version

- **Bathymetry / seabed substrate:** not present in the current master ingestion; no columns added. They can be joined later **per grid** from EMODnet or other sources without time leakage if they are static.
- **NDCI:** not used as a target or feature.
- **Future information:** no features use t+1 or later.

---

## Missing values (policy)

- **No forward- or backward-fill** and **no model-based imputation** in the builder.
- Missing spectral or vessel values remain **NaN** so that downstream modeling can apply explicit masks (e.g. train only where features and target are observed).
- Rows are **not** dropped globally for missingness; the full panel is kept where the master panel exists.

---

## Provenance script

Generate or refresh the file:

```bash
python src/build_modeling_dataset.py \
  --input data/master_dataset.parquet \
  --output data/modeling_dataset.parquet \
  --tif-root data/downloads/emodnet_extracted
```

---

## Column list (29)

**Identifiers / static:** `grid_cell_id`, `week_start_utc`, `grid_res_deg`, `grid_centroid_lat`, `grid_centroid_lon`

**Temporal:** `week_of_year`, `week_sin`, `week_cos`

**Vessel:** `vessel_density_t`, `vessel_density_t_minus_1`, `vessel_density_t_minus_2`

**Spectral lags:** twelve columns `sentinel_{ndvi,ndwi,evi,ndti}_mean_{t,t_minus_1,t_minus_2}`

**Observations:** `sentinel_observation_count_t`

**Target / mask:** `delta_ndti`, `has_valid_delta_ndti`

**Context flags (from master):** `has_sentinel`, `has_emodnet`, `has_helcom`
