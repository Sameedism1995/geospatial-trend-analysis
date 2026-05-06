# Source audit — vessel & oil

Source dataset: `final_run/processed/features_ml_ready.parquet`

## Vessel layer
- Decision: **spatial_proxy**
- Coverage > 70%: True (actual: 75.873%)
- Grids temporally varying > 50%: False (actual: 0.0)
- Weekly-mean std: 1.1212702919885052e-16

  - `vessel_density_t`: {'column': 'vessel_density_t', 'present': True, 'dtype': 'float64', 'non_null_count': 12189, 'coverage_percent': 75.873, 'n_unique': 219, 'zero_percent': 6.6667, 'mean': 0.5254662464773204, 'std': 1.8281379089166958, 'min': 0.0, 'max': 25.710631121415645, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 1.0, 'weekly_mean_std': 1.1212702919885052e-16, 'weekly_mean_unique': 1}
  - `vessel_density`: {'column': 'vessel_density', 'present': True, 'dtype': 'float64', 'non_null_count': 12189, 'coverage_percent': 75.873, 'n_unique': 219, 'zero_percent': 6.6667, 'mean': 0.5254662464773204, 'std': 1.8281379089166958, 'min': 0.0, 'max': 25.710631121415645, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 1.0, 'weekly_mean_std': 1.1212702919885052e-16, 'weekly_mean_unique': 1}
  - `maritime_pressure_index`: {'column': 'maritime_pressure_index', 'present': True, 'dtype': 'float64', 'non_null_count': 12189, 'coverage_percent': 75.873, 'n_unique': 219, 'zero_percent': 6.6667, 'mean': 0.020437703142947506, 'std': 0.07110435758202567, 'min': 0.0, 'max': 1.0, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 1.0, 'weekly_mean_std': 0.0, 'weekly_mean_unique': 1}
  - `port_exposure_score`: {'column': 'port_exposure_score', 'present': True, 'dtype': 'float64', 'non_null_count': 12189, 'coverage_percent': 75.873, 'n_unique': 219, 'zero_percent': 6.6667, 'mean': 0.0286439628124826, 'std': 0.1777921640482113, 'min': 0.0, 'max': 2.436831213049178, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 1.0, 'weekly_mean_std': 0.0, 'weekly_mean_unique': 1}
  - `distance_to_nearest_high_vessel_density_cell`: {'column': 'distance_to_nearest_high_vessel_density_cell', 'present': True, 'dtype': 'float64', 'non_null_count': 16065, 'coverage_percent': 100.0, 'n_unique': 205, 'zero_percent': 7.619, 'mean': 682.0397682572504, 'std': 1425.1616268977448, 'min': 0.0, 'max': 6930.603517463788, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 1.0, 'weekly_mean_std': 0.0, 'weekly_mean_unique': 1}

## Oil / Sentinel-1 layer
- Decision: **unusable**
- Has non-zero oil signal: False
- Detection score present: False
- Extra SAR/VV/VH columns: (none)
  - `oil_slick_probability_t`: {'column': 'oil_slick_probability_t', 'present': True, 'dtype': 'float64', 'non_null_count': 0, 'coverage_percent': 0.0, 'n_unique': 0, 'zero_percent': 0.0, 'mean': None, 'std': None, 'min': None, 'max': None, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 0.0, 'weekly_mean_std': None, 'weekly_mean_unique': 0}
  - `detection_score`: {'column': 'detection_score', 'present': True, 'dtype': 'float64', 'non_null_count': 0, 'coverage_percent': 0.0, 'n_unique': 0, 'zero_percent': 0.0, 'mean': None, 'std': None, 'min': None, 'max': None, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 0.0, 'weekly_mean_std': None, 'weekly_mean_unique': 0}
  - `oil_slick_count_t`: {'column': 'oil_slick_count_t', 'present': True, 'dtype': 'float64', 'non_null_count': 16065, 'coverage_percent': 100.0, 'n_unique': 1, 'zero_percent': 100.0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'fraction_grids_temporally_varying': 0.0, 'median_unique_values_per_grid': 1.0, 'weekly_mean_std': 0.0, 'weekly_mean_unique': 1}