# Master Dataset Schema

`data/master_dataset.parquet` is a weekly panel table with one row per (`grid_cell_id`, `week_start_utc`).

## Grain
- One row per `week_start_utc` x `grid_cell_id` (balanced panel)
- Weekly bucket uses Monday start (UTC)
- Fixed geographic grid resolution: `0.1` degrees
- Same set of grid cells repeats across all canonical weeks (consistent row count per grid cell over time).

## Core Keys
- `week_start_utc` (timestamp, UTC)
- `grid_cell_id` (string, fixed-grid ID)
- `grid_res_deg` (float)
- `grid_centroid_lat` (float, nullable)
- `grid_centroid_lon` (float, nullable)

## Source Presence Flags
- `has_sentinel` (bool)
- `has_emodnet` (bool)
- `has_helcom` (bool)

## Sentinel Weekly Aggregates (Temporal)
- `sentinel_observation_count` (int, source observation count in that week+grid)
- `sentinel_record_count` (alias for panel compatibility)
- `sentinel_datasets` (list[str])
- `sentinel_raw_refs` (list[str])
- `sentinel_feature_ids` (list[str])
- `sentinel_first_record_ts` (timestamp, UTC)
- `sentinel_last_record_ts` (timestamp, UTC)
- `sentinel_<numeric_property>_mean` (float, per week+grid)
- `sentinel_<numeric_property>_median` (float, per week+grid)
  - Example currently available from Sentinel catalog properties: `eo_cloud_cover`.
  - Band-level means/medians are included automatically if band numeric properties are present in ingested payloads.

## EMODnet Static Features (Replicated Across Weeks)
- `emodnet_record_count_static` (int per grid from full EMODnet source)
- `emodnet_raw_refs_static` (list[str])
- `emodnet_feature_ids_static` (list[str])
- `emodnet_first_record_ts_static` (timestamp, UTC)
- `emodnet_last_record_ts_static` (timestamp, UTC)
- `emodnet_record_count` (replicated static count on each week row)

## HELCOM Weekly Aggregates (Nearest-Week Assignment)
- `helcom_record_count` (int)
- `helcom_raw_refs` (list[str])
- `helcom_feature_ids` (list[str])
- `helcom_first_record_ts` (timestamp, UTC)
- `helcom_last_record_ts` (timestamp, UTC)
- HELCOM records are assigned to the nearest canonical week from the panel time axis.

## Provenance
- `provenance_json` (json string)
  - Per-source record counts
  - Raw request signatures (`raw_refs`)
  - Feature identifiers captured from source payloads

## Temporal Consistency Rules
- No mixing of granularities:
  - Sentinel -> weekly aggregated
  - HELCOM -> nearest canonical week
  - EMODnet -> static replicated across weeks
- Canonical weeks are derived from Sentinel observation weeks (or available temporal sources fallback).

## Notes
- No feature engineering or model targets are created here.
- Records without valid spatial coordinates are excluded from panel construction (no inferred spatial mapping).
