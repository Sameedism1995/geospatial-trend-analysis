# Slide deck outline (after Stockholm grid fix)

1. **Why we re-gridded**
   - Previous Stockholm hub had no cells within 50 km (nearest was 81 km).
   - The urban-NO2 contrast hypothesis could not be tested locally.

2. **What we did**
   - Built a 0.1-deg grid around Stockholm within 150 km.
   - Extracted fresh Sentinel-5P NO2 and Sentinel-2 NDWI/NDTI/NDVI for new cells via GEE.
   - Vessel density: not available for new cells; reported as missing.
   - Re-ran the hub-strategy analysis on the expanded dataset.

3. **Hubs**
   - Turku-Naantali coastal hub (combined).
   - Mariehamn offshore/island hub.
   - Stockholm urban port hub (now properly covered).

4. **Local 0-100 km results**
   - Show: local_all_hubs_comparison.png, local_stockholm_no2_zoom.png.
   - Stockholm NO2 distance trend reported with explicit slope and
     sample counts.

5. **Stockholm vs. Mariehamn NO2 contrast**
   - Show: stockholm_no2_local_test.md table.
   - Verdict: SUPPORTED / PARTIAL / NOT SUPPORTED for the urban-NO2
     interpretation.

6. **Regional background**
   - Show: regional_no2_background.png + regional_water_indicators_background.png.
   - Title says "REGIONAL BACKGROUND PATTERN - NOT PORT IMPACT".

7. **Limitations**
   - Vessel density at Stockholm needs an external AIS source.
   - Tropospheric NO2 column integrates over the vertical; cannot separate
     ship-source from urban-source within column data.
   - Single-year (2023) panel.

8. **Next steps**
   - Wire EMODnet vessel-density raster sampling into new cells.
   - Add wind-direction conditioning for NO2 (downwind / upwind windows).
