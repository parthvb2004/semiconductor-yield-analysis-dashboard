# SECOM Dataset & Semiconductor Yield — Domain Notes

## Business Problem Statement

This project simulates a fab's yield engineering workflow: identifying which of 590
in-line sensor signals are most predictive of wafer failure, calculating shift/machine/lot-level
yield KPIs, and surfacing actionable insights via an interactive Power BI dashboard — mirroring
the SPC and yield-analysis work done by process/yield engineers at semiconductor manufacturers.

## Dataset Structure

| Element | Description |
|---|---|
| Rows | 1,567 — each row is one individual wafer production run |
| Columns | 590 sensor readings, anonymized as `Feature_1` ... `Feature_590` |
| Label | Pass/fail outcome per run (separate label file) |
| Label encoding (raw UCI) | **-1 = Pass, 1 = Fail** — remap to 0 = Pass, 1 = Fail on load; easy to invert by mistake |
| Timestamp | Real timestamp per run — genuine data, not synthetic |
| Missingness | Heavy; some sensors missing >50% of values (sensor not wired in, tool not used, or invalid reading) |
| Class balance | ~93-94% Pass, ~6-7% Fail — mirrors real fab pass rates; the Fail class is the rare, high-value-to-detect minority |

## Core Domain Vocabulary

| Term | Plain-English Explanation |
|---|---|
| Wafer | A thin silicon slice that hundreds of chips ("dies") are built onto simultaneously |
| Lot | A batch of wafers (typically 25) processed together through the line |
| Yield | % of wafers/dies that pass final test — the single most important fab KPI, directly tied to revenue and cost per good chip |
| Process Step / Tool | A wafer passes through dozens to hundreds of steps (etch, deposition, lithography, CMP, etc.), each on a specific machine/tool |
| In-line Sensor / SPC Data | Sensors on each tool continuously measure parameters (temp, pressure, gas flow) — this is what SECOM's 590 features represent |
| SPC (Statistical Process Control) | Monitoring sensor signals against control limits to catch drift before it causes failures |
| Excursion | A sensor reading (or group of readings) that drifts outside normal control limits — an early warning sign |
| Root Cause Analysis (RCA) | Investigating which sensor(s)/tool/step caused a failure spike |
| Fab (Fabrication Plant) | The factory where wafers are manufactured |
| Shift | Fabs run 24/7 in shifts (e.g., Day/Swing/Night) — shift-level yield comparison is a standard KPI |
| Operator | Technician/engineer responsible for a tool/lot during a shift |
| Chamber / Tool ID | Even "identical" machines vary slightly (chamber-to-chamber matching) — simulated via synthetic `Machine_ID` |

## Key Modeling Implications

- **Class imbalance (~93/7):** Never evaluate models on plain accuracy — a model predicting
  "Pass" for everything scores ~93% while being useless. Use ROC-AUC / PR-AUC and stratified
  cross-validation instead.
- **High dimensionality, low row count (590 features, 1,567 rows):** Risk of overfitting and
  spurious correlation if all features are used — feature selection (Phase 6) is essential,
  not optional.
- **Missing data is itself informative:** A sensor going dark can indicate a process anomaly,
  not just a data quality issue — worth flagging, not just imputing away.
- **Label encoding must be verified early:** Misreading -1/1 silently inverts every yield
  number downstream.

## Sources
- UCI Machine Learning Repository — SECOM Dataset: https://archive.ics.uci.edu/dataset/179/secom
- General semiconductor manufacturing terminology reference: SEMI.org
