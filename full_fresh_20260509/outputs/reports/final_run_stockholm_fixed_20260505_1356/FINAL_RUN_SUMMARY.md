# FINAL RUN SUMMARY — final_run_stockholm_fixed_20260505_1356

- Generated: `2026-05-05 23:04:11 +0500`
- Run name: `final_run_stockholm_fixed_20260505_1356`

## Cache state
- Wiped 0 aux/intermediate cache entries before launch.
- `--force-refresh` was passed to the inner pipeline.
- All GEE-backed sources (NO2, Sentinel-2 water, Sentinel-2 land NDVI, Sentinel-1 oil-slick proxy) were re-extracted from scratch on the expanded grid.

## Grid expansion
- Old grid cells: **315**
- New cells added (Stockholm-vicinity, ≤150 km): **523**
- Total grid cells in this run: **838**
- Weeks: 51

## Per-hub coverage after expansion
| Hub | nearest grid (km) | grids ≤25 km | grids ≤50 km | grids ≤100 km |
|---|---|---|---|---|
| Turku-Naantali coastal hub | 2.11 | 30 | 65 | 157 |
| Mariehamn offshore/island hub | 5.92 | 17 | 75 | 242 |
| Stockholm urban port hub | 2.53 | 31 | 124 | 458 |

## Pipeline status
- Inner pipeline: **SUCCESS** in 32848s.

## Mirrored artefacts
- `reports`: 55 files
- `visualizations`: 65 files
- `figures`: 56 files
- `processed`: 3 files
- `logs`: 2 files
- `FINAL_RUN_SUMMARY.md`: 1 files
- `validation`: 12 files

## Stockholm NO2 verdict (revised hub strategy)
- Decreases with distance (0-100 km): **True**
- Higher than Mariehamn 0-50 km: **True**
- Urban-atmospheric NO2 interpretation: **SUPPORTED**

## Output isolation
All artefacts for this run live ONLY under:
- `outputs/reports/final_run_stockholm_fixed_20260505_1356/`
- `outputs/figures/final_run_stockholm_fixed_20260505_1356/`
- `outputs/visualizations/final_run_stockholm_fixed_20260505_1356/`
- `processed/final_run_stockholm_fixed_20260505_1356/`
- `logs/final_run_stockholm_fixed_20260505_1356/`
Previous run folders (e.g., `final_run/`, `run_hub_strategy_*`, `run_stockholm_grid_expanded/`) are untouched.
