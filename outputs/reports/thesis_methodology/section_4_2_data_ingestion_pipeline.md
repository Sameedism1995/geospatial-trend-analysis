# Section 4.2 — Data Ingestion Pipeline (implementation notes)

## A. Vessel density and modelling spine (feeds the main merge)

- **Source:** EMODnet Human Activities vessel-density **GeoTIFFs** (annual `vesseldensity_*_<year>.tif` stacks) sampled at **0.1° grid centroids**; values **summed across vessel-category rasters** for the calendar year aligned to each week (see `src/build_modeling_dataset.py`, `docs/modeling_dataset_schema.md`).
- **Output:** `data/modeling_dataset.parquet` ( consumed by `load_base_vessel_source()` in `run_full_pipeline.py` as the **vessels** frame).
- **Spatial logic:** WGS84 lon/lat → `rasterio` sample via CRS transform.
- **Temporal logic:** Week-based row index with year lookup for TIFF vintage (t, t−1, t−2 where applicable per builder).

## B. Google Earth Engine (primary satellite / NO₂ path in `run_full_pipeline`)

**Authentication:** `earthengine authenticate`; `ee.Initialize()` with optional cloud project (`safe_initialize_ee` / `utils/ee_init.py`).

**Shared grid definition:** `data_sources/gee_grid_utils.py` loads **`grid_cell_id`**, **`week_start_utc`**, centroids from `modeling_dataset.parquet`; builds **axis-aligned rectangles** centroid **± `buffer_deg`** (default **0.1°** for NO₂/S1); study **bounding box** = min/max grid ± pad.

**Temporal logic:** For each distinct UTC week anchor, **`filterDate([week_start, week_start + 7 days))`** (`iter_week_utc_bounds`).

**Spatial statistics:** `reduceRegions` over each grid feature; reducer scales: e.g. NO₂ **3500 m**, Sentinel-1 **30 m**, Sentinel-2 water-quality **60 m** (per module constants).

| Module | Collection | Output parquet |
|--------|------------|----------------|
| `no2_gee_pipeline.py` | `COPERNICUS/S5P/NRTI/L3_NO2` (band `tropospheric_NO2_column_number_density`) | `data/aux/no2_grid_week.parquet` |
| `sentinel1_oil_pipeline.py` | `COPERNICUS/S1_GRD` (weekly VH median composite; dark-water fraction proxy) | `data/aux/sentinel1_oil_slicks.parquet` |
| `sentinel2_water_quality.py` | `COPERNICUS/S2_SR_HARMONIZED` (SCL cloud mask; indices B3,B4,B5,B8,B11) | `data/aux/sentinel2_water_quality.parquet` |
| `land_impact/sentinel2_land_metrics.py` | `COPERNICUS/S2_SR_HARMONIZED` (NDVI on land after water mask) | `data/aux/sentinel2_land_metrics.parquet` |

**Caching:** If `data/aux/<layer>.parquet` exists and **`--force-refresh`** is off, extractors **read cache**; `--force-refresh` triggers `wipe_extraction_aux_and_intermediate()` (except whitelisted `baltic_ports.csv`).

## C. Sentinel Hub (two distinct usages)

1. **Catalogue / STAC-style search (ingestion package)** — `config/ingestion_sources.yaml`: **`oauth_token_url`** `https://services.sentinel-hub.com/oauth/token`, **`catalog_search_url`** `https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search`. Auth: env **`SENTINELHUB_CLIENT_ID`** / **`SENTINELHUB_CLIENT_SECRET`**. Body: **bbox**, **datetime** interval, **`collections`: [`sentinel-2-l2a`]**, **CQL2** `eo:cloud_cover` filter. Implemented in **`src/ingestion/sentinel_client.py`**. Results standardized and written under **`data/raw/`** / **`data/processed/`** via **`src/ingestion/orchestrator.py`**.
2. **Statistics API (standalone script)** — **`src/extract_sentinel_weekly_features.py`**: OAuth to same token URL; **POST** `https://services.sentinel-hub.com/api/v1/statistics` with evalscript over **Sentinel-2 L2A**, weekly aggregation — **not** invoked by `run_full_pipeline.main()`.

## D. STAC / Microsoft Planetary Computer

- **Optional utility:** **`src/fetch_online_data.py`** — `fetch_sentinel2_stac()` POSTs to **`https://planetarycomputer.microsoft.com/api/stac/v1/search`**, writes **`data/raw/sentinel2_stac_items.json`** (and digest CSV). This is **metadata catalogue** ingestion, **not** the processor for weekly reflectance used in `merged_dataset.parquet`.

## E. EMODnet (ingestion client)

- **`src/ingestion/emodnet_client.py`** — GET **WMS GetCapabilities** URLs from YAML (e.g. bathymetry and human-activities services), parses **Layer** metadata + **EX_GeographicBoundingBox** into structured records. **Does not** replace the GeoTIFF-based vessel pipeline in **`build_modeling_dataset.py`** unless separately wired.

## F. HELCOM (ingestion client)

- **`src/ingestion/helcom_client.py`** — Fetches **GeoNetwork** formatter XML per configured **`record_id`**: base URL pattern `https://metadata.helcom.fi/geonetwork/srv/api/records/{record_id}/formatters/xml`. Parses ISO 19139 fields (title, abstract, bbox, dates, etc.). **Metadata / provenance** enrichment path; not the direct source of weekly grid values in the ML parquet.

## G. Open-Meteo (downstream analysis, not main parquet merge)

- **`src/analysis/run_land_pollution_drivers_wind.py`** and **`src/analysis/run_coastal_wind_transport.py`** call **`https://archive-api.open-meteo.com/v1/archive`** for **ERA5-backed** wind when no user CSV is supplied.

## H. Parquet generation sequence (execution order in `run_full_pipeline.main()`)

1. **Vessels** → `data/intermediate/vessels.parquet`
2. For each **`discover_sources()`** entry → `data/intermediate/{source}.parquet` + aux path
3. **`merge_sources`** → **`processed/merged_dataset.parquet`**
4. **`feature_engineering`** + **port proximity + port exposure** → **`processed/features_ml_ready.parquet`** (first write)
5. **`run_land_impact_extension`** (if `--land-impact`): **buffering → `add_land_sea_interactions` → rewrite** `features_ml_ready.parquet` → lag/ML/report side effects under **`outputs/reports/`**

## I. Scripts vs artefacts (quick map)

| Artefact / output | Generator |
|-------------------|-----------|
| `merged_dataset.parquet` | `pipeline/run_full_pipeline.py` (`merge_sources` + `to_parquet`) |
| `features_ml_ready.parquet` | Same module: `feature_engineering`, port layers, **`run_land_impact_extension`** |
| Coastal wind CSV / summaries | **`analysis/run_coastal_wind_transport.py`** → e.g. `outputs/reports/run_coastal_wind_transport/` |
| Exposure indices / figures | **`analysis/run_coastal_exposure_analysis.py`**; coastal impact composite: **`analysis/coastal_impact_score.py`** (inside pipeline when flag on) |
| Anomaly CSV | **`analysis/anomaly_detection.py`** → `outputs/reports/anomaly_scores.csv` (+ temporal companion when enabled) |
