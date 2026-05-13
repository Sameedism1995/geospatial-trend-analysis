# ML-Based Geospatial Analysis of Coastal Water Quality and Maritime Traffic

This project provides a clean, reproducible Python setup for geospatial data analysis and machine learning workflows focused on coastal water quality and maritime traffic.

## Project Structure

```text
project_root/
│── data/
│── notebooks/
│── src/
│── outputs/
│── requirements.txt
│── README.md
```

## Environment Setup

### 1) Create virtual environment

```bash
python3 -m venv .venv
```

### 2) Activate virtual environment

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Notebooks

```bash
jupyter notebook
```

Open `notebooks/01_data_loading_and_eda.ipynb` and run cells in order.

## Run EDA Pipeline

Place thesis datasets in `data/` (CSV or GeoJSON), then run:

```bash
python src/eda_pipeline.py
```

The pipeline will:
- load and clean available datasets
- attempt nearest-neighbor spatial merge for Sentinel and vessel data
- run core statistical and spatial EDA
- save outputs to:
  - `data/processed/`
  - `outputs/plots/`
  - `outputs/reports/`

## Run Thesis-Aligned Statistical Tables

Use this when your merged point-level dataset is ready (Sentinel + vessel + coordinates).

Required columns (minimum):
- `latitude`, `longitude`
- `vessel_density` (or `density_total`)
- at least one of: `ndci`, `ndti`, `ndwi`, `red_green_ratio`, `fai`

Optional columns:
- `distance_to_port` or (`port_latitude`, `port_longitude`)
- `b11` (for low-SWIR restricted tests)

Run:

```bash
python src/thesis_analysis_pipeline.py --input data/processed/master_dataset.csv
```

Outputs:
- `data/processed/master_dataset_enriched.csv`
- `outputs/reports/lane_vs_background_tests.csv`
- `outputs/reports/high_vs_low_traffic_ttests.csv`
- `outputs/reports/distance_bin_anova.csv`
- `outputs/reports/spearman_shipping_water.csv`
- `outputs/reports/thesis_pipeline_summary.csv`

## Run Research Ingestion Pipeline

This pipeline ingests source-native data from EMODnet, HELCOM, and Sentinel Hub, stores raw and standardized parquet outputs, and logs every API call and run manifest for reproducibility.

Set Sentinel Hub credentials (required for sentinel source):

```bash
export SENTINELHUB_CLIENT_ID="your_client_id"
export SENTINELHUB_CLIENT_SECRET="your_client_secret"
```

Scheduled daily-style run (fixed defaults from config):

```bash
python src/run_ingestion.py --mode scheduled
```

On-demand run (custom AOI/time):

```bash
python src/run_ingestion.py \
  --mode on_demand \
  --sources emodnet,helcom,sentinel \
  --bbox 19.5,59.7,22.5,60.7 \
  --time-from 2023-01-01T00:00:00Z \
  --time-to 2023-12-31T23:59:59Z
```

Outputs:
- Raw parquet: `data/raw/<source>/<dataset>/<ingestion_date>/`
- Standardized parquet: `data/processed/<source>/<dataset>/<ingestion_date>/`
- API call log: `data/raw/_logs/api_calls.jsonl`
- Run manifest: `data/raw/_logs/run_manifest_<run_id>.json`

### Sentinel Ingestion Smoke Test

Run a focused Sentinel-2 test that verifies:
- OAuth auth
- catalog query success
- small raster retrieval and local save

```bash
export SENTINELHUB_CLIENT_ID="your_client_id"
export SENTINELHUB_CLIENT_SECRET="your_client_secret"
python src/sentinel_test.py
```

Test outputs are written to:
- `data/raw/sentinel_test/sentinel_test_metadata_<timestamp>.json`
- `data/raw/sentinel_test/sentinel_test_catalog_<timestamp>.json`
- `data/raw/sentinel_test/sentinel_test_preview_<timestamp>.png`

## Build Master Dataset

Create a unified weekly fixed-grid dataset from ingested source outputs:

```bash
python src/build_master_dataset.py
```

Outputs:
- `data/master_dataset.parquet`
- `docs/master_dataset_schema.md`

## Run Full Multi-Source Pipeline

Run the integrated extraction, merge, feature engineering, validation, EDA, and correlation-analysis workflow:

```bash
python src/pipeline/run_full_pipeline.py
```

Optional flags:

```bash
python src/pipeline/run_full_pipeline.py --quick-test
python src/pipeline/run_full_pipeline.py --skip-validation
python src/pipeline/run_full_pipeline.py --skip-eda
python src/pipeline/run_full_pipeline.py --skip-correlation
```

Correlation outputs:
- `outputs/reports/pearson_correlation.csv`
- `outputs/reports/spearman_correlation.csv`
- `outputs/reports/correlation_evaluation.csv`
- `outputs/plots/correlations/`

## Notes

- A sample dataset is included at `data/sample_water_quality.csv`.
- Store generated figures, model artifacts, and exports in `outputs/`.
