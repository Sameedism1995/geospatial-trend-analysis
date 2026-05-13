# NO2 interpretation check — hub strategy

Goal: test whether NO2 within 0-50 km is dominated by urban/atmospheric background (Stockholm strong, Mariehamn weak) rather than a pure port-proximity signal.

## Local 0-50 km NO2 trend per hub

| Hub | n windows | slope (NO2 per km) | mean near | mean far |
|---|---|---|---|---|
| Stockholm urban port hub | 4 | -6.258e-08 | 1.562e-05 | 1.371e-05 |
| Mariehamn offshore/island hub | 4 | 1.460e-08 | 1.062e-05 | 1.112e-05 |
| Turku-Naantali coastal hub | 4 | -1.262e-07 | 1.741e-05 | 1.357e-05 |

## Stockholm 50-200 km (substitute band)

| Hub | n windows | slope (NO2 per km) | mean near | mean far |
|---|---|---|---|---|
| Stockholm 50-200 km | 5 | -2.877e-08 | 1.297e-05 | 1.181e-05 |

## Verdict

- Stockholm has data 0-50 km: **True**
- Stockholm NO2 decreases with distance (0-50 km): **True**
- Stockholm NO2 higher near port than 50 km out: **False**
- Stockholm NO2 decreases with distance (50-200 km, substitute): **True**
- Mariehamn NO2 0-50 km signal weaker than Stockholm: **True**

**Interpretation**: NO2 behaves more like an urban/regional atmospheric signal than a pure port-proximity signal.

## Caveats
- NO2 here is a tropospheric column proxy from Sentinel-5P; it is sensitive to weather, season, cloud cover, and land-source emissions.
- The existing grid was built around the Mariehamn / Turku / Naantali area and does not extend close to Stockholm; this prevents the direct 0-50 km Stockholm test.  Future runs should either re-grid around the Stockholm port or import an external near-Stockholm grid sample.
- 0-50 km windows around each hub include grids over both land and sea; land contributions inflate NO2 around urban hubs.
- Use this section as **interpretation guidance**, not as a causal test.
