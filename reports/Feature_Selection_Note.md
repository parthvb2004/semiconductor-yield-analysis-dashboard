# Feature Selection Note — EDA vs. Random Forest Divergence

## Finding

The sensors identified as most visually separable in the Phase 5 EDA (univariate,
standardized mean-gap between Pass/Fail distributions) showed minimal overlap with the
Phase 6 Random Forest importance ranking — only **2 sensors** (Sensor_574, Sensor_132)
appeared in both lists.

## Interpretation

This divergence is expected and analytically meaningful, not an error in either method:

- **The EDA check is univariate** — it evaluates each sensor in isolation, comparing its
  own Pass-group mean to its own Fail-group mean. With only ~103 Fail-class wafers (6.6%
  of the dataset), this metric is sensitive to noise: a small number of atypical Fail
  wafers can produce an apparently large separation on a sensor that isn't actually a
  reliable predictor.
- **Random Forest importance is multivariate** — it reflects how much each sensor
  contributes to correctly classifying wafers *in combination* with every other sensor in
  the model, and it was computed on a correlation- and mutual-information-filtered subset
  (redundant sensors removed first). A sensor can be individually well-separated but
  redundant with another sensor already carrying that signal, causing the model to assign
  it low importance. Conversely, a sensor with modest individual separation can still be
  highly informative when combined with others (an interaction effect invisible to a
  single-sensor view).
- Random Forest importance was additionally validated with 5-fold stratified
  cross-validation (ROC-AUC), giving it a stronger statistical basis than the single-pass
  EDA separability check.

## Conclusion

Random Forest importance is treated as the authoritative ranking for downstream work
(KPI sensor deep-dives, dashboard "Top Contributing Sensors" visual, and any future
predictive modeling). The EDA separability check remains valuable as an independent,
easy-to-interpret cross-reference — the fact that it identifies a largely different set of
sensors is itself evidence that failure signatures in this dataset are driven more by
multivariate interactions than by any single dominant sensor, which is a realistic pattern
for a complex manufacturing process with hundreds of interacting tool parameters.
