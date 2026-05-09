# Whole feature inventory — before preprocessing / merging

This document lists **signals as they originate from providers** (Tier A), **standard ingestion tables** written by the ingestion layer (Tier B), **first-stage Earth Engine extractions aligned to grid × ISO week** (Tier C — still upstream of ML-ready parquet fusion), plus **geometry and catalogue side tables**. Anything in `outputs/processed/features_ml_ready*.parquet`, `master_dataset*.csv`, and engineered deltas is **not** covered here — see [`DATASET_FEATURES_LIST.md`](DATASET_FEATURES_LIST.md).

**Machine-readable export:** [`source_feature_inventory_before_preprocessing.csv`](source_feature_inventory_before_preprocessing.csv) — same inventory as one row per field (tiers A/B/C/aux/ref, scope, parent path, formula notes).

**Legend**

| Tier | Meaning |
|------|---------|
| **A** | Native observable: band name, NetCDF/API field, raster value, XML/STAC leaf field. |
| **B** | First tabular envelope your code writes (`data/raw/**/*.parquet`, `data/processed/{sentinel,emodnet,helcom}/**/*.parquet`). |
| **C** | Zonal weekly statistics from GEE / resampled grids — one modality per file, **before** full multi-source join and coastal-wind augmentation. |

---

## 1. Summary table (sources → artefacts)

| Source | Tier A (native) | Tier B files | Tier C (first grid-week extract) |
|--------|-----------------|--------------|----------------------------------|
| HELCOM GeoNetwork ISO-19139 XML | Parsed metadata keys (below) | `data/processed/helcom/.../*.parquet` | — |
| EMODnet WMS GetCapabilities | Layer `name`, `title`, `bbox`, `crs` | `data/processed/emodnet/.../*.parquet` | — |
| Sentinel Hub Catalogue (STAC search) | STAC Feature `properties`, `assets`, geometry | `data/processed/sentinel/.../*.parquet` | — |
| Planetary Computer STAC export (`sentinel2_stac_items`) | Same STAC richness in JSON (+ optional 9-column CSV digest) | `data/raw/sentinel2_stac_items.json/.csv` | — |
| EMODnet Human Activities vessel density GeoTIFF | One band vessel density raster | — | Weekly `vessel_density_t` in `data/intermediate/vessels.parquet` (after resample to grid) |
| Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED` | SR bands below + `SCL` + scene metadata | — | [`sentinel2_water_quality.py`](../src/data_sources/sentinel2_water_quality.py): water indices statistics; [`sentinel2_land_metrics.py`](../src/data_sources/land_impact/sentinel2_land_metrics.py): land NDVI |
| GEE `COPERNICUS/S5P/NRTI/L3_NO2` | `tropospheric_NO2_column_number_density` | — | `no2_mean_t`, `no2_std_t` |
| GEE `COPERNICUS/S1_GRD` | SAR `VH` (+ other GRD bands in product) | — | `oil_slick_probability_t`, `oil_slick_count_t` (dark-water proxy, not labelled oil) |
| Open‑Meteo ERA5 archive API | hourly `wind_speed_10m`, `wind_direction_10m`, `time` | — | Script-derived weekly `wind_u_mean`, `wind_v_mean`, etc. [`run_land_pollution_drivers_wind.py`](../src/analysis/run_land_pollution_drivers_wind.py) |
| Natural Earth 110m coast | GeoJSON coastline features | Cached under `data/aux/natural_earth_coast_cache/` | Used for bearings / shoreline logic downstream |
| Port list (`baltic_ports.csv`) | `port_name`, `latitude`, `longitude` | `data/aux/baltic_ports.csv` | — |

---

## 2. Tier B — ingestion `raw_result` parquet (per API call envelope)

[`DataStorage.write_raw_result`](../src/ingestion/storage.py) persists one row per HTTP/API outcome:

| Feature | Meaning |
|---------|---------|
| `source` | `sentinel`, `emodnet`, `helcom` |
| `dataset` | e.g. `catalog`, `wms_layers`, `spatial_records`, `oauth_token` |
| `endpoint` | URL or logical endpoint |
| `request_signature` | Hash id for deduplication paths |
| `fetched_at_utc` | Ingest clock time |
| `status`, `success`, `status_code` | Call outcome |
| `request_params` | JSON-serialised parameters |
| `response_payload` | JSON-serialised body (often nested: STAC GeoJSON, WMS XML text wrapper, HELCOM XML text, token JSON) |
| `error_message` | Failure detail |
| `records_count` | Row / feature / layer count reported by client |

Paths follow `data/raw/<source>/<dataset>/<ingestion_date>/<signature>.parquet`.

---

## 3. Tier B — standardized `processed` record parquet (spatial catalogue rows)

[`write_processed_records`](../src/ingestion/storage.py) + [`standardize_result`](../src/ingestion/standardize.py) produce **exactly ten** top-level columns; variable provider fields live inside **`payload` (JSON string)**.

| Feature | Meaning |
|---------|---------|
| `source`, `dataset` | Provenance |
| `record_timestamp_utc` | Best-effort scene / dataset time (`TIME_KEYS` in standardizer + HELCOM citations) |
| `ingested_at_utc` | Ingest timestamp |
| `latitude`, `longitude` | Representative point (explicit fields or centroid of `bbox`) |
| `geometry_wkt` | WKT **or** JSON string of GeoJSON geometry (Sentinel Hub path) |
| `crs` | e.g. `EPSG:4326`, `EPSG:32634` from `proj:epsg` when present |
| `grid_id` | Layer name, Sentinel `tile_id` / record `id`, or HELCOM stable id/title surrogate |
| `raw_record_ref` | Request signature / batch key |
| `payload` | **Full canonical provider record JSON** |

Paths: `data/processed/<source>/<dataset>/<ingestion_date>/*.parquet`.

---

### 3.1 HELCOM — keys inside `payload` (Tier A inside JSON)

Parsed in [`HelcomClient._extract_record_spatial`](../src/ingestion/helcom_client.py):  
`record_id`, `title`, `abstract`, `purpose`, `topic_category`, `organisation`, `lineage`, `keywords` (array), `bbox`, `crs`, `geometry_wkt`, `record_timestamp_utc`, `metadata_stamp_utc`, `citation_dates` (nested dict), `temporal_extent` (`begin`, `end`).

---

### 3.2 EMODnet WMS layers — keys inside each `payload` item

From [`EmodnetClient._extract_layers_from_wms`](../src/ingestion/emodnet_client.py):  
`name`, `title`, `bbox` [min lon, min lat, max lon, max lat], `crs`.

---

### 3.3 Sentinel catalogue (STAC features) — structure inside `payload`

Each catalogue hit is typically a STAC **Feature**:

- Top-level identifiers: **`id`, `bbox`, `type`, `geometry`, `collection`, `links`, `assets`, `properties`, `stac_extensions`, `stac_version`**.
- **`assets`**: keyed product parts (spectral GeoTIFFs at multiple resolutions plus auxiliary rasters such as **`AOT`**, **`WVP`**, **`SCL`**, **`visual`**, **`preview`**, **`safe-manifest`**, etc.; band keys follow product naming **`B01`–`B12`**, **`TCI`**, and resolution variants where applicable).

**Example `properties` keys** observed in workspace STAC export `data/raw/sentinel2_stac_items.json` (Planetary Computer; Hub responses are analogous):

`datetime`, `platform`, `proj:epsg`, `instruments`, `s2:mgrs_tile`, `constellation`, `s2:granule_id`, `eo:cloud_cover`, `s2:datatake_id`, `s2:product_uri`, `s2:datastrip_id`, `s2:product_type`, `sat:orbit_state`, `s2:datatake_type`, `s2:generation_time`, `sat:relative_orbit`, `s2:water_percentage`, `s2:mean_solar_zenith`, `s2:mean_solar_azimuth`, `s2:processing_baseline`, `s2:snow_ice_percentage`, `s2:vegetation_percentage`, `s2:thin_cirrus_percentage`, `s2:cloud_shadow_percentage`, `s2:nodata_pixel_percentage`, `s2:unclassified_percentage`, `s2:dark_features_percentage`, `s2:not_vegetated_percentage`, `s2:degraded_msi_data_percentage`, `s2:high_proba_clouds_percentage`, `s2:reflectance_conversion_factor`, `s2:medium_proba_clouds_percentage`, `s2:saturated_defective_pixel_percentage`.

**Flattened STAC catalogue digest** (`data/raw/sentinel2_stac_items.csv`) — **9** columns:

`id`, `datetime`, `eo_cloud_cover`, `platform`, `proj_epsg`, `bbox_min_lon`, `bbox_min_lat`, `bbox_max_lon`, `bbox_max_lat`

---

### 3.4 Other small raw catalogues (`data/raw/*.csv`)

| File | Columns |
|------|---------|
| `emodnet_vessel_density_links.csv` | `name`, `description`, `url`, `protocol` |
| `helcom_records_page1.csv` | `uuid`, `title` |

---

## 4. Tier A — EMODnet vessel density raster (GeoTIFF)

- **Stored as:** COG-style GeoTIFF under download/extract dirs (paths vary by machine); summary rows in **`data/processed/emodnet_tiff_summary.csv`** include derivative QC columns (**not** raster pixels):  
  `file`, `vessel_code`, `year`, `width`, `height`, `crs`, `pixel_count_*`, percentile stats `*_density`, flags `sea_lane_flag`, `traffic_*`, etc.
- **Native pixel content:** effectively **one band** interpreted as vessel density (**`float32`** in extracts described in earlier runs); CRS encoded in file (often EPSG:3035 in product packaging).

---

## 5. Tier A — Google Earth Engine collections (pixels / image metadata)

Configured in codebase:

### 5.1 `COPERNICUS/S2_SR_HARMONIZED`

**Native image bands read in this repo:** **`B3`, `B4`, `B5`, `B8`, `B11`**, **`SCL`** (scene classification).

**Additional scene/image metadata used:** `CLOUDY_PIXEL_PERCENTAGE`, `system:time_start`.

**Reflectance scale:** DN ÷ **`10000.0`** to physical reflectance \[0–1\] used in formulae ([`sentinel2_water_quality.py`](../src/data_sources/sentinel2_water_quality.py), land module).

**Derived analytics (computed in EE before export — still “upstream” of ML parquet):**

| Derived “band” name in reducer | Formula (conceptual) |
|--------------------------------|----------------------|
| `ndwi` | (B3 − B8) / (B3 + B8) McFeeters |
| `ndti` | (B4 − B3) / (B4 + B3) turbidity-related |
| `ndci` | (B5 − B4) / (B5 + B4) |
| `fai` | floating algae index from B8, B4, B11 linear blend |
| `b11` | SWIR scaled reflectance |
| `ndvi` land module only | (B8 − B4) / (B8 + B4); masked with clouds + McFeeters NDWI ≤ **0** for land |

`SCL` classes treated as opaque mask: **`{3, 8, 9, 10}`** excluded (shadow / cloud / cirrus family).

---

### 5.2 `COPERNICUS/S5P/NRTI/L3_NO2`

- **Tier A band exported at L3:** **`tropospheric_NO2_column_number_density`** (single band image collection; temporal mean inside each week).

---

### 5.3 `COPERNICUS/S1_GRD`

- **Tier A polarization used:** **`VH`** (weekly median composite, regional percentile threshold pipeline — see [`sentinel1_oil_pipeline.py`](../src/data_sources/sentinel1_oil_pipeline.py)).

---

## 6. Tier C — per-modality Parquet schemas (aligned `grid_cell_id`, `week_start_utc`)

These are **`data/intermediate/*.parquet`** (and copies under **`data/aux/`** with identical column names).

| Artefact | Features |
|---------|----------|
| `vessels.parquet` | `grid_cell_id`, `week_start_utc`, `vessel_density_t`, `grid_centroid_lat`, `grid_centroid_lon` |
| `no2.parquet` (`no2_grid_week.parquet`) | `grid_cell_id`, `week_start_utc`, `no2_mean_t`, `no2_std_t` |
| `sentinel1.parquet` (`sentinel1_oil_slicks.parquet`) | `grid_cell_id`, `week_start_utc`, `oil_slick_probability_t`, `oil_slick_count_t` |
| `sentinel2_water_quality.parquet` | `grid_cell_id`, `week_start_utc`, for each of `ndwi`, `ndti`, `ndci`, `fai`, `b11`: `{band}_mean`, `{band}_median`, `{band}_std` (**17** series fields + keys) |
| `land_impact_ndvi.parquet` (`sentinel2_land_metrics.parquet`) | `grid_cell_id`, `week_start_utc`, `ndvi_mean`, `ndvi_median`, `ndvi_std` |

Model grid geometry for EE reducers loads with [`gee_grid_utils`](../src/data_sources/gee_grid_utils.py): at minimum `grid_cell_id`, `week_start_utc`, `grid_centroid_lat`, `grid_centroid_lon` in the grids table passed to pipelines.

---

## 7. Tier A — Open‑Meteo ERA5 archive hourly JSON

[`fetch_open_meteo_cluster_wind`](../src/analysis/run_land_pollution_drivers_wind.py) requests:

`latitude`, `longitude`, `start_date`, `end_date`, `hourly='wind_speed_10m,wind_direction_10m'`, `timezone='UTC'`.

**`hourly` object keys:** `time`, `wind_speed_10m`, `wind_direction_10m`.

**Weekly aggregates emitted by code** (already a transform; list for traceability):

`cluster_id`, `week_start_utc`, `wind_u_mean`, `wind_v_mean`, `wind_speed_mean`, `wind_direction_to_degrees`, `wind_direction_from_degrees`.

Wind **from-direction** meteorological convention is converted to u/v before weekly means.

---

## 8. Reference geometry — Natural Earth coastline cache

[`ne_110m_coastline.geojson`](../data/aux/natural_earth_coast_cache/ne_110m_coastline.geojson) — root: `type`, `name`, `crs`, `features`, `bbox`.

Per **feature**: `type`, `properties`, `bbox`, `geometry` (`LineString` in sample).

**Properties** (tier A attributes on vector): **`scalerank`**, **`featurecla`**, **`min_zoom`**.

---

## 9. Reference table — Baltic ports (`data/aux/baltic_ports.csv`)

**Features:** `port_name`, `latitude`, `longitude`

---

## 10. Supplementary illustrative / QA tables (not primary sensors)

| Item | Role |
|------|------|
| `data/sample_water_quality.csv` | **7**: `date`, `latitude`, `longitude`, `salinity`, `temperature`, `turbidity`, `vessel_count` — illustrative unless you attach provenance. |
| `data/event_regime_summary.csv` | **9**: `activity_regime`, `mean_ndvi`, …, `n_rows` — exploratory summary. |
| `data/distance_decay_summary.csv` | **8**: binned coastline/shipping distances vs means. |
| `data/aux/*_validation.json`, `no2_gee_validation.json` | Structured QA payloads (nested JSON, not rectangular features). |

`data/processed/master_dataset*.csv` and `cleaned_merged_dataset.csv` are **already merged preprocessing products** → see [`DATASET_FEATURES_LIST.md`](DATASET_FEATURES_LIST.md).

---

## 11. Contrast boundary

Anything that **concatenates** modalities with focal-port buffers, shoreline bearings, Open‑Meteo wind overlays, anomalies, cross terms (`vessel_x_no2`), and nearest‑land NDVI linkage lives in **`outputs/processed/features_ml_ready_coastal_wind.parquet`** and related pipelines — **downstream** of this catalog.

For **executive routing** across scripts and artefacts, see [`CODE_AND_RESULTS_EXECUTIVE_INDEX.md`](CODE_AND_RESULTS_EXECUTIVE_INDEX.md).
