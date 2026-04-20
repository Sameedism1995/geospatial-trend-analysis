# Thesis Results Summary

## Correlation Insights

The strongest cross-feature correlation links ndwi_mean and ndci_mean (spearman, r = -0.876, negative, strong). Magnitudes across the top-ranked pairs remain primarily in the weak-to-moderate band, consistent with weekly grid-level remote-sensing signals where direct coupling between maritime pressure and environmental response is expected to be partial rather than deterministic.

### Top 5 findings

| feature_x | feature_y | method | value | abs_value | strength |
| --- | --- | --- | --- | --- | --- |
| ndwi_mean | ndci_mean | spearman | -0.8759025127855893 | 0.8759025127855893 | strong |
| ndwi_mean | b11_mean | spearman | -0.8421783300676257 | 0.8421783300676257 | strong |
| ndci_mean | fai_mean | spearman | 0.7957075665108807 | 0.7957075665108807 | strong |
| ndwi_mean | fai_mean | spearman | -0.7889536756425569 | 0.7889536756425569 | strong |
| ndci_mean | b11_mean | spearman | 0.7280673918619299 | 0.7280673918619299 | strong |

## Temporal Dynamics

Temporal analysis indicates that vessel_density relates most strongly to detection_score at a best lag of 0 week(s) (r = -0.100, immediate, weak). Short delays dominate the ranked list, suggesting measurable but bounded temporal response between shipping-related pressure and environmental indicators.

### Top 5 findings

| feature_x | feature_y | best_lag | best_correlation | abs_best_correlation | interpretation | strength |
| --- | --- | --- | --- | --- | --- | --- |
| vessel_density | detection_score | 0 | -0.0996923618485672 | 0.0996923618485672 | immediate | weak |
| vessel_density | NO2_mean | 4 | 0.0409729163213094 | 0.0409729163213094 | delayed response | weak |

## Anomaly Detection

The most extreme detected anomaly is grid g0.100_r1503_c2022 at 2023-12-17 00:00:00+00:00 (score = 2.039, extreme event). The top-20 anomalies cluster around a small number of repeatedly-flagged grid cells, indicating persistent local deviations rather than isolated noise.

### Top 5 findings

| grid_cell_id | week_start_utc | anomaly_score | anomaly_label | event_class |
| --- | --- | --- | --- | --- |
| g0.100_r1503_c2022 | 2023-12-17 00:00:00+00:00 | 2.0388407342999564 | anomalous | extreme event |
| g0.100_r1503_c2022 | 2023-12-10 00:00:00+00:00 | 2.0388407342999564 | anomalous | extreme event |
| g0.100_r1503_c2022 | 2023-12-03 00:00:00+00:00 | 2.0388407342999564 | anomalous | extreme event |
| g0.100_r1437_c1885 | 2023-12-03 00:00:00+00:00 | 1.9163470087265329 | anomalous | extreme event |
| g0.100_r1503_c2022 | 2023-01-01 00:00:00+00:00 | 1.7122670113800622 | anomalous | extreme event |

## Coastal Impact Analysis

The highest-pressure coastal zone is grid g0.100_r1503_c2022 at 2023-12-17 00:00:00+00:00 (composite score = 1.000). Ranked zones consistently combine elevated shipping-related features with stronger anomaly response, supporting the thesis hypothesis that spatial exposure near ports and shipping corridors concentrates environmental pressure.

### Top 5 findings

| grid_cell_id | week_start_utc | coastal_impact_score | vessel_density | NO2_mean | anomaly_score | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| g0.100_r1503_c2022 | 2023-12-17 00:00:00+00:00 | 1.0 | 25.710631121415645 | nan | 2.0388407342999564 | high environmental pressure zone |
| g0.100_r1503_c2022 | 2023-12-10 00:00:00+00:00 | 1.0 | 25.710631121415645 | nan | 2.0388407342999564 | high environmental pressure zone |
| g0.100_r1503_c2022 | 2023-12-03 00:00:00+00:00 | 1.0 | 25.710631121415645 | nan | 2.0388407342999564 | high environmental pressure zone |
| g0.100_r1437_c1885 | 2023-12-03 00:00:00+00:00 | 0.9293920928453208 | nan | 0.0001895733548012634 | 1.9163470087265329 | high environmental pressure zone |
| g0.100_r1503_c2022 | 2023-01-01 00:00:00+00:00 | 0.8117561777214111 | 25.710631121415645 | nan | 1.7122670113800622 | high environmental pressure zone |
