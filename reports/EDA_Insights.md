# EDA Insights — SECOM Yield Analysis

## Dataset Snapshot

- Total wafer runs: **1,567**
- Sensors retained after cleaning: **432** (434 total columns minus `Timestamp` and `Label`)
- Class balance: **93.4% Pass / 6.6% Fail**
- Date range covered: **~Aug 2008 – Oct 2008**

## Top 5 EDA Insights

1. **Class Balance confirms the fab-typical ~93/7 split.**
   Pass/Fail came out to 93.4% / 6.6%, closely matching the imbalance expected in real
   semiconductor yield data. This confirms accuracy would be a misleading evaluation metric
   going forward (a model predicting "Pass" every time would score ~93.4% while catching zero
   failures) — ROC-AUC / PR-AUC with stratified cross-validation is the right approach for
   Phase 6 and any later modeling work.

2. **The most separable sensors are Sensor_60, Sensor_104, Sensor_511, Sensor_349, Sensor_432,
   and Sensor_435** — and they share a consistent pattern: it's **not** a simple mean shift
   between Pass and Fail, it's a **variance/tail difference**. In every one of the six
   distribution plots, the Pass class forms a tight, narrow peak around a central value, while
   the Fail class is flatter and extends into a long right-hand tail (e.g., Sensor_432 and
   Sensor_435 both show Pass sharply peaked near 0 while Fail spreads out to 400–500). This
   suggests failing wafers are associated with increased *process variability* on these sensors,
   not just an offset reading — a meaningful distinction for the root-cause story in the
   dashboard.

3. **Sensor redundancy is concentrated in a few tight clusters, not spread evenly.** The
   correlation heatmap (top 30 highest-variance sensors) shows two clear redundant groups:
   {Sensor_163, Sensor_162, Sensor_298, Sensor_25, Sensor_297, Sensor_24} and
   {Sensor_160, Sensor_22, Sensor_161, Sensor_295, Sensor_296}, both with correlations
   approaching 1.0 within-group. A smaller pair, {Sensor_205, Sensor_141, Sensor_341}, also
   correlates strongly. Interestingly, Sensor_68 and Sensor_23 show a **negative** correlation
   (~-0.6) — worth a closer look, since inversely-related sensors can indicate a shared
   underlying process mechanism (e.g., one rising as the other falls during the same step). The
   majority of the remaining high-variance sensors (Sensor_512, 420, 500, 501, 419, 487, 483,
   488, 140, 489, 469) show near-zero correlation with everything else — they're statistically
   independent signals, not redundant with each other.

4. **Failure rate was highest and most volatile in the earliest weeks of data (late Aug 2008),
   then stabilized through September and October.** The daily failure rate spiked as high as
   ~40–50% in the Aug 21–31 window (including one day at 100%, likely a low-volume day),
   whereas from mid-September onward, peaks rarely exceeded ~20–25% and many days sat near 0%.
   This is consistent with either a process ramp-up/stabilization effect or early tooling
   issues that were resolved over time — a plausible, dashboard-worthy "yield improved over
   time" narrative, though it should be labeled as an observed pattern rather than a confirmed
   root cause without further investigation.

5. **PCA shows heavy overlap between Pass and Fail — no clean linear separation.** In the 2D
   PCA scatter, the vast majority of both Pass and Fail points sit tightly clustered near the
   origin, fully overlapping. A handful of Fail points do appear among the small number of
   extreme outliers (PC1 ~70–105), but most outliers in that region are actually Pass wafers, so
   PC1/PC2 position alone isn't a reliable failure indicator. This supports the plan (already
   set in the roadmap) to use **Random Forest importance** rather than a linear/PCA-based
   method for Phase 6 — the failure signal here is likely non-linear and sensor-specific rather
   than a broad structural difference visible in reduced dimensions.

## Implications for Phase 6 (Feature Selection)

- **Correlation filter candidates:** one sensor from each of {Sensor_163/162/298/25/297/24} and
  {Sensor_160/22/161/295/296} should be dropped as redundant; keep Sensor_68 and Sensor_23
  separately despite their correlation, since the negative relationship (rather than
  redundancy) may itself carry signal.
- **Expected to rank highly in Random Forest importance:** Sensor_60, Sensor_104, Sensor_511,
  Sensor_349, Sensor_432, Sensor_435 — this EDA-stage "quick check" ranking gives a benchmark
  to sanity-check the Phase 6 output against. If these sensors *don't* show up near the top of
  the Random Forest ranking, that's worth investigating (may indicate the simple standardized
  mean-gap metric was misleading, or that Random Forest is picking up different, interaction-
  based signal).
- Given PCA showed no linear separation, prioritize the embedded/model-based stage (Random
  Forest) over any linear filter method as the final ranking authority.

## Charts Referenced

- `reports/figures/class_balance.png`
- `reports/figures/top_sensor_distributions.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/failure_rate_over_time.png`
- `reports/figures/pca_scatter.png`
