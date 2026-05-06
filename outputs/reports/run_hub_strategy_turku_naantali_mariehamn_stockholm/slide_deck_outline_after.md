# Slide deck outline (after revision)

1. **Why revise the hub strategy**
   - Florian: previous setup mixed local port impact with regional background.
   - Turku and Naantali (~15 km) were double-counted as separate hubs.
   - 0-1000 km curves blurred local vs. regional patterns.

2. **Revised hubs**
   - Turku-Naantali coastal hub (combined).
   - Mariehamn offshore/island hub (independent).
   - Stockholm urban port hub (added for urban contrast).

3. **Local analysis (0-100 km)**
   - Sliding window: 25 km / step 10 km.
   - Show: local_vessel_density_decay.png and local_all_hubs_comparison.png.
   - Headline: vessel density decays within 0-100 km, strongest for the
     Finnish coastal hub; Mariehamn shows island-scale decay; Stockholm
     decay is moderate.

4. **NO2 interpretation**
   - Show: local_no2_decay.png + regional_no2_background.png side-by-side.
   - Stockholm NO2 elevates within 0-50 km, Mariehamn NO2 is flat.
   - Conclusion: NO2 is dominated by urban/atmospheric background, not
     ship-source emissions.

5. **Regional background**
   - 100-1000 km bands shown only as background context, never as port
     impact.  Title: "REGIONAL BACKGROUND PATTERN - NOT PORT IMPACT".

6. **Comparison with previous run**
   - Pull comparison_with_previous_runs.md.
   - Show: previous 0-1000 km NO2 curve vs. new local 0-100 km + regional
     background bars.

7. **Limitations and next steps**
   - Single-year (2023) panel; no winter/summer break-down.
   - Sentinel-1 oil-slick proxy still being re-extracted; results will
     be incorporated in next iteration.
   - Need urban/road-traffic NO2 covariate to fully isolate ship NO2.
