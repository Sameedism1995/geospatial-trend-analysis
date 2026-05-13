# Pipeline orchestration and architecture diagrams

Mermaid source for **Section 4** (or appendix). Render in [Mermaid Live Editor](https://mermaid.live), VS Code (Mermaid extension), GitHub-flavoured Markdown, or export to SVG/PNG from those tools.

---

## 1. End-to-end orchestration (`run_final_pipeline.py`)

```mermaid
flowchart TB
  subgraph ENTRY["Entry: run_final_pipeline.py"]
    A[Snapshot configs to run_folder/config_snapshot]
    B[GEE probe: ee.Initialize + project env]
    C{GEE unavailable AND force-refresh requested?}
    D[Disable force-refresh → reuse data/aux parquets]
    E[Invoke pipeline.run_full_pipeline.main]
  end

  subgraph PIPE["pipeline/run_full_pipeline.py"]
    F[Optional wipe data/aux + intermediate]
    G[Load vessels from modeling_dataset.parquet]
    H[For each GEE source: extract or load aux parquet]
    I[merge_sources → merged_dataset.parquet]
    J[feature_engineering + port proximity + port exposure]
    K[Write features_ml_ready.parquet v1]
    L{--land-impact?}
    M[land_sea_buffering + land_sea_interactions]
    N[Rewrite features_ml_ready.parquet]
    O[global_validation JSON]
    P[EDA + correlation + optional stages]
    Q[anomaly_detection → anomaly_scores.csv]
    R[coastal_impact_score → coastal_impact_score.csv]
    S[final visualization + port map]
  end

  subgraph POST["Post-pipeline (final_run mirror)"]
    T[Mirror processed/, outputs/, validation/]
    U[final_dataset_validation on features_ml_ready]
    V[hub_distance_decay_analysis]
    W[FINAL_RUN_SUMMARY.md]
  end

  A --> B --> C
  C -->|yes| D --> E
  C -->|no| E
  E --> F --> G --> H --> I --> J --> K --> L
  L -->|yes| M --> N
  L -->|no| O
  N --> O
  O --> P --> Q --> R --> S
  S --> T --> U --> V --> W
```

---

## 2. Data acquisition and merge (operational ML path only)

```mermaid
flowchart LR
  subgraph UPSTREAM["Upstream (not run_full_pipeline)"]
    MD[(modeling_dataset.parquet\nEMODnet GeoTIFF sampling)]
  end

  subgraph GEE["Google Earth Engine → data/aux/"]
    NO2[no2_grid_week.parquet\nS5P L3 NO2]
    S1[sentinel1_oil_slicks.parquet\nS1 GRD VH proxy]
    S2W[sentinel2_water_quality.parquet\nS2 SR indices]
    S2L[sentinel2_land_metrics.parquet\noptional NDVI land]
  end

  subgraph MERGE["Harmonization"]
    INT[data/intermediate/*.parquet]
    MGD[(merged_dataset.parquet)]
  end

  MD --> INT
  NO2 --> INT
  S1 --> INT
  S2W --> INT
  S2L --> INT
  INT --> MGD
```

---

## 3. Feature engineering order (after merge)

```mermaid
flowchart TD
  MG[(merged_dataset.parquet)]
  FE[feature_engineering:\nNO2_mean, NO2_trend, vessel_density,\ndetection_score, nan_ratio_row, …]
  PP[port_proximity:\nnearest_port, distance_to_port_km]
  PE[port_exposure_score:\nvessel / 1+d_km]
  W1[(features_ml_ready.parquet\nfirst write)]
  LI{land-impact\nenabled?}
  BUF[land_sea_buffering:\ndistance_to_high_vessel,\nbands, coastal_exposure_score]
  INT[land_sea_interactions:\nindices, vessel_x_no2,\nvessel_x_ndvi_lag1–3]
  W2[(features_ml_ready.parquet\nfinal write)]

  MG --> FE --> PP --> PE --> W1 --> LI
  LI -->|yes| BUF --> INT --> W2
  LI -->|no| DONE[Downstream validation / EDA / analysis]
  W2 --> DONE
```

---

## 4. Parallel research ingestion (optional)

**Not** invoked by `run_full_pipeline.main()`. Separate entry: `python3 src/run_ingestion.py`.

```mermaid
flowchart TB
  YAML[config/ingestion_sources.yaml]
  ORC[ingestion/orchestrator.py]
  EMOD[EmodnetClient\nWMS GetCapabilities]
  HEL[HelcomClient\nGeoNetwork XML]
  SH[SentinelHubClient\nSTAC POST + OAuth]
  RAW[data/raw/]
  PR[data/processed/]
  MAN[data/raw/_logs manifests]

  YAML --> ORC
  ORC --> EMOD
  ORC --> HEL
  ORC --> SH
  EMOD --> RAW --> PR
  HEL --> RAW
  SH --> RAW
  ORC --> MAN
```

---

## 5. Downstream coastal wind / exposure (manual or separate runs)

```mermaid
flowchart LR
  PANEL[(features_ml_ready.parquet\nor augmented parquet)]
  CWT[run_coastal_wind_transport.py]
  ALIGN[coastal_wind_alignment_features.csv\nOpen-Meteo ERA5 or wind CSV]
  CEA[run_coastal_exposure_analysis.py]
  EXP[Exposure indices, band stats,\nfigures under outputs/]

  PANEL --> CWT --> ALIGN
  ALIGN --> CEA
  PANEL --> CEA
  CEA --> EXP
```

---

## 6. ASCII quick reference (orchestration spine)

```
run_final_pipeline.py
    │
    ├─► config_snapshot/
    ├─► gee_probe ──► (may fall back to cached data/aux/)
    │
    └─► pipeline.run_full_pipeline.main()
            │
            ├─► [vessels] modeling_dataset ──► intermediate/vessels.parquet
            ├─► [no2, s1, s2, land?] GEE ──► data/aux/*.parquet + intermediate/
            ├─► merge ──► processed/merged_dataset.parquet
            ├─► feature_engineering + ports ──► processed/features_ml_ready.parquet (v1)
            ├─► land_impact? ──► buffering + interactions ──► same parquet (final)
            ├─► validation ──► data/validation/*.json
            └─► EDA / correlation / anomaly / coastal score / viz ──► outputs/
    │
    ├─► mirror ──► final_run/ (or --run-name)
    ├─► final_dataset_validation
    ├─► hub_distance_decay_analysis
    └─► FINAL_RUN_SUMMARY.md
```

---

*Generated for thesis documentation. Adjust node labels to match your chapter terminology.*
