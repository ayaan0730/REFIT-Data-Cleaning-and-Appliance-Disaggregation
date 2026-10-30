# Data: coverage, quality findings, cleaning, remaining concerns

## Source

REFIT Electrical Load Measurements (University of Strathclyde), 20 UK households in the Loughborough area, 2013-2014. Per house: one whole-home aggregate clamp plus nine Individual Appliance Monitors (IAMs). Active power in watts, polled every 6-8 seconds, not synchronised across sensors. Timestamps are UTC (Unix time). Documentation: the dataset's `CLEAN_READ_ME_081116.txt`.

**Cleaned version (used for Sections 2-3)**, per the readme: UK daylight-saving time corrected; IAM sections moved to the right appliance after resets or householder moves; NaNs forward-filled; IAM spikes above 4000 W replaced by zeros; an `Issues` column set to 1 when the IAM sum exceeds the aggregate.

## Households and channels used

| House | Washing-machine channel | Model (from the readme) | Usable data span (1-min, after cleaning) | Usable minutes |
|---|---|---|---|---|
| 1 | Appliance5 | Beko WMC6140 | 2013-10-09 to 2015-07-10 | 756,424 |
| 2 | Appliance2 | LG F1289TD | 2013-09-17 to 2015-05-28 | 628,843 |
| 5 | Appliance3 | AEG L99695HWD | 2013-09-26 to 2015-07-06 | 689,050 |
| 6 | Appliance2 | Bosch Classixx 1200 Express | 2013-11-28 to 2015-06-28 | 663,128 |
| 7 | Appliance5 | Bosch (model unknown) | 2013-11-01 to 2015-07-08 | 681,884 |

Mapping comes from the readme's appliance list, which says occupants may have moved monitors ("we cannot guarantee"). House 1 also has a separate washer-dryer on Appliance4; Houses 5 and 7 have separate tumble dryers. Sampling interval after processing: 1 minute. Units: watts (energy reported in kWh).

## Section 1: raw House 1 inspection and cleaning
 Script: `src/01_clean_raw_house1.py`.

**Findings** (from `outputs/house1_raw_audit_table.csv`):

| Check | Finding |
|---|---|
| Timestamp parsing and ordering | 0 unparseable timestamps. There are 2 negative timestamp jumps (Δt < 0s), occurring primarily at the boundary where _Part1 and _Part2 overlap. |
| Duplicate timestamps | 1,211,048 (14.22%) |
| Missing readings | Aggregate 2.2% NaN (188,973 readings); washing machine (Appliance5) 5.8%; fridge 5.1%; the electric heater (Appliance9) is worst at 22.4%. |
| Irregular intervals | Median sampling interval is 7.0s (mean 7.55s). There are 1,066,352 intervals exceeding 10s due to polling jitter, and 390 outages lasting >5 minutes. The longest single outage is 998.4 hours (41.6 days) in February–March 2014. |
| Negative values | None in any channel (0 in the audit table). |
| Impossible or extreme values | Raw readings reach 29,159 W on the aggregate and up to 98,301 W on appliance channels, which is impossible for these appliances; the electric heater's maximum of 65,535 W equals the 16-bit limit, which suggests a sensor overflow value. Counts of extreme spikes above [> 20,000 W for the aggregate and > 4,000 W for individual appliances]: aggregate 127, fridge 846, freezer(1) 40, freezer(2) 118, washer-dryer 2,085, washing machine 221, dishwasher 36, computer 374, television 179, electric heater 50. |

**Cleaning pipeline, in order** 
1. Sort by timestamp.
2. Remove duplicate timestamps, keeping last reason: (reason: Part 2 of the raw data was concatenated after Part 1; keeping the last entry preserves the corrected Part 2 rows where missing sensors are properly marked as NaN instead of ambiguous 0s)..
3. Resample to 1-minute means.
4. Interpolate gaps of at most 5 minutes only.
5. Gaps longer than that stay NaN and are flagged as outages, not filled (no evidence about what the load did).
6. Spikes: Aggregate readings > 20,000 W and individual appliance readings > 4,000 W are masked as NaN before resampling. Note: The cleaned fridge channel still shows occasional 1.5–3 kW spikes, representing either genuine compressor startup surges or remaining sensor transients below the 4,000 W cutoff.

Plots: `outputs/figures/sec1_house1_before_after_overview.png`, `sec1_house1_zoomed_cleaning.png`. Figure 1a (Overview) shows that removing the 65–98 kW 16-bit register-overflow spikes makes the genuine ~2.3 kW washing-machine heating peaks and ~100 W fridge cycles visible. Figure 1b (Zoomed) illustrates a 24-hour window where a >22 kW raw spike on the washing machine and a ~1.4 kW spike on the fridge were removed, preserving the real wash cycle that occurred later that morning.

## Processing for Sections 2-3

1. Load the cleaned CSV, resample to 1-minute means, interpolate gaps up to 5 minutes.
2. Rows with `Issues == 1` are set to NaN.
3. Rows where the washing machine exceeds the aggregate by more than 50 W are set to NaN (physically impossible).
4. Clip the aggregate to 0-20,000 W and the washing machine to 0-3,500 W.
5. Remove flat "plateau" segments (below).

### Plateau segments

Rule (`src/plateau_rule.py`): washing-machine power between 100 and 1000 W, rolling 30-minute standard deviation under 25 W, lasting at least 120 minutes.

Audit result (`outputs/plateau_summary_all_houses.csv`, `house5_plateau_runs.csv`): only House 5 is affected, with two runs (2014-09-29 and 2015-04-10, 8.5 hours total, about 1 kWh, 0.5% of its washing-machine energy). In both runs the values are exactly constant (100% identical minutes) and every other channel reads exactly 0 W, so these look like logger dropouts rather than real load. They are excluded from training, validation and testing. Results with and without exclusion are compared in RESULTS.md; the difference is small.

## Remaining concerns

- **Forward-filling can hide outages.** In the cleaned data a dropout may look like a flat run, not a gap. I only checked this for House 5's washing-machine channel (plateau audit above); I did not run a repeated-value check on every channel of every house.
- **House 5 long flat ~500 W stretches.** Separate from the dropouts above. In the longest false negative (2015-05-02, 01:42-05:43 UTC) the washing-machine channel sat at about 500 W for hours, with no heating burst in the 3 hours before it. During the same run the tumble-dryer channel read exactly 475 W (maximum equals mean), which looks like a stale or forward-filled value, and about 24 of the 241 clock minutes are missing, possibly rows removed by the `Issues` flag (IAM sum above the aggregate); I did not verify that. So it is unclear whether such stretches are genuine washing or a monitoring artefact. They are kept in the data and reported as a limitation. See RESULTS.md.
- **Channel mapping risk.** Monitors may have been moved by householders; the readme cannot guarantee the labels.
- **Confusers in the aggregate.** Tumble dryers, dishwashers and electric showers overlap with washing-machine signatures.
- **Only five homes**, all in one UK area, 2013-2015.
- **Outage periods** (notably February 2014) reduce usable data and break windows.
- **Splits use row position**, which approximates time because missing-data rows were dropped.