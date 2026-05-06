# NO2 interpretation check — hub strategy

Goal: test whether NO2 within 0-50 km is dominated by urban/atmospheric background (Stockholm strong, Mariehamn weak) rather than a pure port-proximity signal.

> **DATA-COVERAGE WARNING**: the existing grid does not contain any cells within 50 km of Stockholm (nearest grid ~81 km).  The 0-50 km Stockholm test cannot be performed in this dataset; we report Stockholm 50-200 km as the closest substitute and flag the interpretation as INCONCLUSIVE.

## Local 0-50 km NO2 trend per hub

| Hub | n windows | slope (NO2 per km) | mean near | mean far |
|---|---|---|---|---|
| Stockholm urban port hub | 0 | n/a (no data) | n/a | n/a |
| Mariehamn offshore/island hub | 4 | 8.308e-09 | 1.062e-05 | 1.093e-05 |
| Turku-Naantali coastal hub | 4 | -1.262e-07 | 1.741e-05 | 1.357e-05 |

## Stockholm 50-200 km (substitute band)

| Hub | n windows | slope (NO2 per km) | mean near | mean far |
|---|---|---|---|---|
| Stockholm 50-200 km | 3 | -1.342e-08 | 1.226e-05 | 1.199e-05 |

## Verdict

- Stockholm has data 0-50 km: **False**
- Stockholm NO2 decreases with distance (0-50 km): **False**
- Stockholm NO2 higher near port than 50 km out: **False**
- Stockholm NO2 decreases with distance (50-200 km, substitute): **True**
- Mariehamn NO2 0-50 km signal weaker than Stockholm: **False**

**Interpretation**: INCONCLUSIVE for 0-50 km: dataset has no grids within 50 km of Stockholm, so the urban-NO2 hypothesis cannot be tested at the intended scale.  In the next available band (50-200 km) NO2 decreases with distance from Stockholm, which is consistent with — but does not prove — an urban-source pattern.

## Caveats
- NO2 here is a tropospheric column proxy from Sentinel-5P; it is sensitive to weather, season, cloud cover, and land-source emissions.
- The existing grid was built around the Mariehamn / Turku / Naantali area and does not extend close to Stockholm; this prevents the direct 0-50 km Stockholm test.  Future runs should either re-grid around the Stockholm port or import an external near-Stockholm grid sample.
- 0-50 km windows around each hub include grids over both land and sea; land contributions inflate NO2 around urban hubs.
- Use this section as **interpretation guidance**, not as a causal test.
