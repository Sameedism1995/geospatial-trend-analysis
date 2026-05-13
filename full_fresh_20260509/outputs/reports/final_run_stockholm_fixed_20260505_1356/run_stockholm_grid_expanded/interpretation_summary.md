# Hub strategy interpretation summary (Stockholm grid expanded)

- Old grid cells: 315
- New grid cells added: 523
- Total grid cells now: 838

## Per-hub coverage after expansion
| Hub | nearest grid (km) | grids ≤25 km | grids ≤50 km | grids ≤100 km |
|---|---|---|---|---|
| Turku-Naantali coastal hub | 2.11 | 30 | 65 | 157 |
| Mariehamn offshore/island hub | 5.92 | 17 | 75 | 242 |
| Stockholm urban port hub | 2.53 | 31 | 124 | 458 |

## Indicators detected
- vessel_density: `vessel_density_t`
- ndwi: `sentinel_ndwi_mean_t`
- ndti: `sentinel_ndti_mean_t`
- ndvi: `sentinel_ndvi_mean_t`
- no2: `no2_mean_t`

## GEE extraction meta (new Stockholm cells)
```json
{
  "no2": {
    "rows": 26673,
    "non_null_pct": 89.45000562366438
  },
  "sentinel2_water": {
    "rows": 26673,
    "non_null_ndwi_pct": 74.64852097626814
  },
  "sentinel2_land": {
    "rows": 26673,
    "non_null_ndvi_pct": 67.45772878941251
  }
}
```

## Stockholm NO2 verdict
- Decreases with distance 0-100 km: **True**
- Higher than Mariehamn 0-50 km: **True**
- Urban-atmospheric NO2 interpretation: **SUPPORTED**

## Thesis-safe statements
- Stockholm now has true local 0-50 km NO2 coverage; the urban-NO2 hypothesis is testable, with the verdict above.
- Vessel-density at Stockholm is missing for new cells; do not interpret the near-Stockholm vessel comparison as a temporal AIS signal.
- Water indices (NDWI/NDTI/NDVI) for new Stockholm cells include land pixels and should be interpreted as 'mixed land/water context' rather than pure water.
- All distances >100 km in figures are explicitly REGIONAL BACKGROUND, not port impact.