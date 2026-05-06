# Stockholm NO2 local test (expanded grid)

## 1. Does Stockholm now have valid 0-25 / 0-50 / 0-100 km coverage?
- Stockholm nearest grid: **2.5 km**
- Grids within 25 km: **31**
- Grids within 50 km: **124**
- Grids within 100 km: **458**

## 2. NO2 sample stats per band
| Hub | Band | sample_count | valid_obs | mean | median | missing% |
|---|---|---|---|---|---|---|
| Stockholm urban port hub | 0-25 | 1581 | 1382 | 1.562e-05 | 1.724e-05 | 12.59 |
| Stockholm urban port hub | 0-50 | 6324 | 5600 | 1.448e-05 | 1.574e-05 | 11.45 |
| Stockholm urban port hub | 0-100 | 23358 | 20832 | 1.277e-05 | 1.408e-05 | 10.81 |
| Mariehamn offshore/island hub | 0-25 | 867 | 762 | 1.062e-05 | 1.148e-05 | 12.11 |
| Mariehamn offshore/island hub | 0-50 | 3825 | 3363 | 1.102e-05 | 1.209e-05 | 12.08 |
| Mariehamn offshore/island hub | 0-100 | 12342 | 10788 | 1.165e-05 | 1.273e-05 | 12.59 |

## 3. NO2 distance trend within 0-100 km

- **Stockholm urban port hub**: slope = -5.117e-08 per km, n_windows = 9, mean(near third) = 1.5027832584089968e-05, mean(far third) = 1.1916768669620677e-05
- **Mariehamn offshore/island hub**: slope = 2.226e-08 per km, n_windows = 9, mean(near third) = 1.0958943983609507e-05, mean(far third) = 1.2214525109047203e-05

## 4. Verdict

- Stockholm NO2 decreases with distance (0-100 km): **True**
- Stockholm 0-50 km NO2 higher than Mariehamn 0-50 km: **True**
- Urban-atmospheric NO2 interpretation: **SUPPORTED**

## 5. Honest caveats

- Sentinel-5P NO2 is a tropospheric column proxy; it integrates over a vertical column and is sensitive to weather, season, cloud cover.
- 0-25/0-50 km windows around Stockholm cover Stockholm's urban land area, industrial areas, road traffic and shipping lanes.  We cannot separate urban-source from ship-source NO2 with column data alone.
- Conclusion is association, not causality: NO2 is *consistent with* an urban background pattern; a definitive ship-NO2 attribution would need plume tracking, wind-direction conditioning, or an in-situ campaign.
- Vessel-density data is not available for the new Stockholm cells (HELCOM coverage stops at the original grid); near-Stockholm vessel-density comparisons are limited to the 5 pre-existing cells within ~100 km.