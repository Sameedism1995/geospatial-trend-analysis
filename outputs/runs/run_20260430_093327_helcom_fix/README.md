# Run snapshot — `run_20260430_093327_helcom_fix`

Re-run of the complete pipeline **after repairing the HELCOM integration**.
Every earlier stage (ingestion, merge, features, EDA, correlations, anomaly,
coastal impact, land impact, delta-NDTI modeling, human-impact distance
analysis, QA) was executed end-to-end against the freshly-rebuilt
`data/master_dataset.parquet`.

## What changed vs. the previous run (`run_20260430_070441`)

| Metric | Before | After |
|---|---|---|
| `has_helcom=True` rows in master | **6** (0.04 %) | **12 342** (76.83 %) |
| Distinct grid cells with HELCOM coverage | **4** | **242** |
| Weeks with HELCOM coverage | **1** | **51** (all) |
| `helcom_titles` surfaced per cell | n/a | 3 – 10 dataset titles |
| `geometry_wkt` populated on HELCOM spatial records | 0 % | **100 %** (real POLYGON WKT) |
| HELCOM payload fields promoted to columns | only `title` | `bbox`, `record_title`, `abstract`, `purpose`, `topic_category`, `organisation`, `lineage`, `keywords_csv`, `temporal_extent_*` |

### Root causes fixed

1. **`src/ingestion/helcom_client.py`** — previously captured only title + bbox.
   Rewritten to parse ISO-19139 citation dates (publication / revision /
   creation), abstract, keywords, topic category, organisation, lineage,
   temporal extent, and to emit a real WGS84 `POLYGON` WKT from the bbox.
2. **`src/ingestion/standardize.py`** — standardiser now propagates
   `geometry_wkt` and `record_timestamp_utc` from the enriched payload
   instead of hard-coding them to `None`.
3. **`src/enrich_helcom_records.py`** (new) — offline re-enricher that
   upgrades every existing `data/processed/helcom/**/*.parquet` with
   polygon WKT + promoted columns, without needing a new network call.
4. **`src/build_master_dataset.py`** — HELCOM aggregator rewritten from
   `nearest-week + bbox-centroid grid join` (which collapsed every dataset
   onto a single grid cell) to a proper **polygon-grid intersection**
   broadcast across all canonical weeks. New column `helcom_titles` lists
   the covering datasets per cell.

## How it was run

```bash
# 1) One-shot enrichment of existing HELCOM parquets
python src/enrich_helcom_records.py

# 2) Rebuild master + modeling panels with the new builder
python src/build_master_dataset.py
python src/build_modeling_dataset.py

# 3) Full multi-source pipeline with every analytical stage enabled
python src/pipeline/run_full_pipeline.py \
    --feature-interaction-map \
    --scientific-validation \
    --anomaly-detection \
    --coastal-impact-score \
    --final-visualization \
    --land-impact

# 4) All side scripts into $RUN_DIR …
python src/run_delta_ndti_models.py           ...
python src/visualize_dataset.py               ...
python src/visualize_global_maps.py           ...
python src/human_impact_distance_analysis.py  ...
python src/master_dataset_qa_report.py        ...
```

## Folder layout

Same structure as the earlier snapshot:

```
plots/                    reports/             eda/          previews/
modeling/delta_ndti/      human_impact/        visualizations/
logs/                     merged_dataset.parquet
features_ml_ready.parquet master_dataset.parquet  modeling_dataset.parquet
README.md                 RUN_MANIFEST.json
```

## HELCOM-specific artefacts in this run

- **`master_dataset.parquet`** now carries 6 HELCOM panel columns
  populated on 76.83 % of rows (`helcom_record_count`, `helcom_raw_refs`,
  `helcom_feature_ids`, `helcom_titles`, `helcom_first_record_ts`,
  `helcom_last_record_ts`).
- **`reports/master_dataset_qa_report.json`** reflects the new coverage.
- **`data/processed/helcom/**/*.parquet`** were upgraded in place: every
  record now has a real `geometry_wkt` polygon, a `bbox` column, and
  promoted metadata fields (`record_title`, `abstract`, `keywords_csv`,
  `topic_category`, `organisation`, `lineage`, `temporal_extent_*`).

## Remaining caveat

`record_timestamp_utc` is still NULL for HELCOM records. The raw XML was
discarded by the older ingestion run before being persisted, so the ISO
`CI_Date` elements cannot be recovered offline. The enrichment script
gracefully leaves those columns NULL; a future live HELCOM ingestion run
(with the updated client) will populate them automatically.

## Headline pipeline results (unchanged — HELCOM is catalog metadata,
not a modelled signal)

- Strongest Spearman: `ndwi_mean vs ndci_mean = -0.876`, `ndwi_mean vs b11_mean = -0.842`, `ndci_mean vs fai_mean = 0.796`
- Land-sea lag: `NO2_mean → ndvi_mean` peaks at 4 weeks, Spearman = 0.66 (strong)
- Land-impact ML: r²_test = 0.845, r²_cv = 0.677 ± 0.203
- Coastal-impact top zone: `g0.100_r1503_c2022` (60.35°N, 22.25°E) 2023-04-09, score = 0.622
