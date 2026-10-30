# Results: washing-machine EDA, model, validation, failure analysis

## 1. Washing-machine EDA (Section 2)

**Homes and periods.** Houses 1, 2, 5, 6, 7 from the official cleaned REFIT data, 1-minute resolution, spans in [DATA.md](DATA.md).

**Active-cycle rule, in plain language.** A cycle starts when the machine draws at least 20 W and ends when it has stayed below 20 W for 15 minutes; pauses shorter than 15 minutes inside a wash do not split it; cycles shorter than 15 minutes or under 0.05 kWh (50 Wh) are discarded. This is because physical wash programmes pause with near-zero power draw between active wash, rinse, and spin phases, and splitting them would artificially inflate the cycle count.

**Availability and quality** (from `outputs/sec2_washing_machine_summary.csv`):

| House | Date range | Missing intervals | Zero/off time | Detected cycles |
|---|---|---|---|---|
| 1 | 2013-10-09 to 2015-07-10 | 12.41% | 98.12% | 374 |
| 2 | 2013-09-17 to 2015-05-28 | 24.41% | 95.31% | 322 |
| 5 | 2013-09-26 to 2015-07-06 | 8.91% | 92.11% | 707 |
| 6 | 2013-11-28 to 2015-06-28 | 14.96% | 98.12% | 120 |
| 7 | 2013-11-01 to 2015-07-08 | 12.45% | 91.62% | 834 |

State the cycle counts: some houses have few cycles, so their weekday patterns are noisy.

**Plot (i): representative 24-hour trace** (`sec2_plot1_representative_24h_trace.png`, House 1, 2014-04-29). The washing machine is a small part of the day: one wash around midday (a ~2.4 kW heating block of roughly 15 minutes, then low-power activity), while the aggregate also shows a ~10 kW spike at about 08:00 and several ~2.5-3.3 kW events, including a ~2.5 kW block near 13:40 when the washing machine was off. Those other loads overlap the washing machine's power range, which is why disaggregating it from the aggregate alone is hard.

**Plot (ii): detected cycles** (`sec2_plot2_detected_cycles_anatomy.png`). House 1's cycle (2013-12-27, 88 min, 0.58 kWh) has two ~2 kW heating blocks with low-power activity between them; House 2's (2013-10-15, 86 min, 0.50 kWh) has one ~2 kW block of about 15 minutes followed by low-power activity. I describe only what the power trace shows (heating blocks, low-power stretches, near-zero pauses), not named wash phases. The House 1 example is longer than that house's typical cycle (median about 33 minutes), so it is an illustration, not an average cycle.

**Plot (iii): cross-house comparison** (`sec2_plot3_cross_house_comparison.png`). House 6 peaks sharply around 10:00 UTC (>31% of cycles), House 7 has morning (07:00) and evening (18:00) peaks, while House 1 shows an overnight timer peak (04:00-05:00 UTC). Weekday patterns vary: Houses 1 and 2 distribute evenly across weekdays, whereas House 5 runs 59.8% of its laundry on Fridays and Saturdays. Median cycle duration ranges from 33-53 minutes (Houses 1, 5) to 97-130 minutes (Houses 2, 6). Energy per cycle differs about 2x between homes, around 0.32 kWh in Houses 1 and 5 against 0.55-0.68 kWh in Houses 2, 6 and 7. Say plainly that small samples (e.g., House 6, with only 120 cycles) make some weekday shares unreliable.

## 2. Model (Section 3)

### Task and data
Estimate the washing machine's power each minute from the whole-home aggregate only (the appliance channel is used as the training target and for scoring, never as an input). Houses 1, 2, 5, 6, 7; spans in DATA.md.

### Models
| Model | What it is |
|---|---|
| Baseline 1 | Always off (0 W). |
| Baseline 2 | Mean washing-machine power by day-of-week and hour, learned on the training set. |
| **Model 1 (main)** | Two-stage hurdle LightGBM: a classifier decides ON/OFF (threshold tuned on validation), a regressor with L1 loss predicts power when ON; output capped at the aggregate. Features: baseload-subtracted ("net") aggregate; rolling mean/std/max/range over 5-61 minutes; past versus future half-window asymmetry; minutes spent in the 1650-3200 W heating band; lags; time of day and weekday. |
| Model 2 (comparison) | Seq2Point 1D-CNN (Zhang et al., 2018 idea) on a 61-minute window of net aggregate power; L1 loss; best epoch chosen on validation over all rows. |

### Validation design and leakage control
**Split A, temporal.** For every house, the first 70% of usable minutes train, the next 15% validate, the last 15% test, with a **210-minute buffer on each side of each boundary** (a 420-minute gap). A row's features can see up to about 210 minutes (baseload window +-180 min plus +-30 min windows and lags), so the gap means no aggregate sample is used by both a training row and an evaluation row. One pooled model is trained on all houses; results are reported pooled and per house.

**Split B, unseen house.** Train on Houses 1, 2, 5; validate on House 6; test on House 7, a home the model never saw.

**Other leakage controls.** Features use only the aggregate. They are computed within each house, and CNN windows are built per house, so none straddles two houses. The classifier threshold, early stopping and the CNN's best epoch use the validation set only; the test set is scored once. **Why not a random split:** neighbouring minutes overlap heavily, so random splits would leak. **Caveat:** splits use row position, which approximates time because missing-data rows were dropped.

### Metrics and why
- **Overall MAE and RMSE.** Reported but misleading alone: the machine is idle most of the time, so always-off already scores well on MAE.
- **Active-ON MAE.** Error only when the machine is truly on; this is where models differ.
- **Relative energy error (SAE).** Matters for energy reporting and savings estimates. Pooled values can hide cancellation between houses, so read it per house.
- **NDE.** Squared-error relative to the signal's energy; 1.0 means no better than zero.
- **ON/OFF precision, recall, F1** at the active threshold of 20 W: whether usage is detected at the right time.

### Results (final run: plateau minutes excluded)

Pooled, temporal split (true energy 133.90 kWh):

| Model | Overall MAE | RMSE | Active MAE | Pred. kWh | Energy err. | NDE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Always-off | 15.70 | 150.21 | 365.16 | 0.00 | 100.0% | 1.000 | 0.000 | 0.000 | 0.000 |
| Time-of-day | 24.81 | 149.11 | 349.35 | 90.47 | 32.4% | 0.993 | 0.082 | 0.370 | 0.135 |
| **LightGBM** | **11.87** | 110.00 | 198.12 | 111.62 | 16.6% | 0.732 | 0.697 | 0.543 | **0.611** |
| CNN | 15.18 | 110.20 | **195.00** | 140.01 | 4.6% | 0.734 | 0.390 | 0.716 | 0.505 |

Unseen house, House 7 (true energy 284.55 kWh):

| Model | Overall MAE | RMSE | Active MAE | Pred. kWh | Energy err. | NDE | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Always-off | 25.04 | 202.27 | 463.37 | 0.00 | 100.0% | 1.000 | 0.000 | 0.000 | 0.000 |
| Time-of-day | 33.59 | 200.80 | 447.69 | 118.45 | 58.4% | 0.993 | 0.116 | 0.304 | 0.168 |
| **LightGBM** | **22.56** | 178.20 | 322.39 | 200.07 | 29.7% | 0.881 | 0.698 | 0.477 | **0.567** |
| CNN | 37.97 | 189.74 | **245.45** | 493.93 | 73.6% | 0.938 | 0.247 | 0.951 | 0.392 |

Per house, temporal test set (overall MAE / active MAE / energy error / F1):

| House | Always-off | Time-of-day | LightGBM | CNN |
|---|---|---|---|---|
| 1 | 4.28 / 342.9 / 100% / 0 | 14.57 / 327.0 / 150% / 0.035 | **3.50 / 186.1 / 17.7% / 0.782** | 7.27 / 172.6 / 80.2% / 0.339 |
| 2 | 14.74 / 453.7 / 100% / 0 | 24.11 / 439.3 / 29.3% / 0.081 | **14.34 / 256.6 / 1.3% / 0.546** | 20.54 / 254.0 / 47.2% / 0.456 |
| 5 | 21.29 / 254.2 / 100% / 0 | 29.66 / 239.7 / 48.3% / 0.232 | 21.90 / 225.3 / 62.0% / 0.264 | 21.45 / 214.7 / 59.2% / 0.451 |
| 6 | 4.45 / 337.8 / 100% / 0 | 14.61 / 321.1 / 138.6% / 0.031 | **2.42 / 98.7 / 11.4% / 0.637** | 6.10 / 100.1 / 91.7% / 0.312 |
| 7 | 34.53 / 464.4 / 100% / 0 | 41.81 / 446.8 / 70.4% / 0.240 | **17.92 / 163.2 / 2.2% / 0.847** | 21.50 / 169.6 / 6.2% / 0.699 |

(Source: `outputs/sec3_per_house_metrics_excl_plateaus.csv`.)

**Held-out test periods** (last 15% of each house, after the buffer; output of `src/05_failure_checks.py`):

| House | Test period | Test minutes |
|---|---|---|
| 1 | 2015-03-24 to 2015-07-10 | 113,254 |
| 2 | 2015-03-02 to 2015-05-28 | 94,117 |
| 5 | 2015-03-20 to 2015-07-06 | 103,148 |
| 6 | 2015-03-11 to 2015-06-28 | 99,260 |
| 7 | 2015-03-21 to 2015-07-08 | 102,073 |

Training and validation use each house's earlier data (about 70% and 15% of its usable minutes); for the unseen-house split the test set is all of House 7.

### Sensitivity to excluding plateau minutes
| Run | Split | Model | Overall MAE | Active MAE | Energy err. | F1 |
|---|---|---|---|---|---|---|
| All data | Temporal | LightGBM | 11.66 | 198.68 | 20.4% | 0.604 |
| Excluded | Temporal | LightGBM | 11.87 | 198.12 | 16.6% | 0.611 |
| All data | Unseen | LightGBM | 22.62 | 329.66 | 33.9% | 0.556 |
| Excluded | Unseen | LightGBM | 22.56 | 322.39 | 29.7% | 0.567 |

The exclusion changes little because it removes 0.5% of House 5's washing-machine energy. ("All data" figures are from the run before the exclusion was added; reproduce with `--with-plateaus`.)

### Actual versus predicted
`outputs/figures/sec3_actual_vs_predicted_typical_excl_plateaus.png` shows a median-energy window in House 1's test period; `sec3_actual_vs_predicted_test_excl_plateaus.png` shows the window with the most washing-machine energy (flattering, so it is supplementary). In the median window (House 1, 2015-05-05) LightGBM finds the cycle but its peak (~600 W) is far below the true ~2300 W heating block, and the CNN raises a spurious ~1100 W false positive at about 08:30, when the aggregate shows a ~3.3 kW event unrelated to the washing machine. In the busiest window (2015-06-06) both models follow the three main ~2 kW heating blocks but miss an ~800 W start at about 09:00, just after a ~10 kW aggregate spike, and a ~480 W burst at about 15:15.

### Feature importance
`sec3_feature_importance_excl_plateaus.png`. The classifier relies on local variability (`c_range_15m`), minutes in the heating band and short-term jumps, which matches the physics of a wash. Baseload and hour-of-day also rank highly; these describe the household rather than the appliance and may contribute to weaker performance on unseen homes. I did not run an ablation without these features, so this is a hypothesis.

### Interpretation
- LightGBM clearly beats both baselines in Houses 1, 2, 6 and 7. In House 5 its overall MAE (21.90 W) is no better than always-off (21.29 W).
- The CNN has a similar active-power error but lower precision, and is unstable across runs: its pooled energy error moved from 0.55% to 4.6% between two near-identical runs, and its validation loss swings widely between epochs. On the unseen house its overall MAE is worse than always-off. I do not claim CNN-versus-LightGBM differences beyond this. The CNN also lacks the validation-tuned ON threshold that LightGBM gets, which handicaps its precision.
- LightGBM under-predicts total energy (17% temporal, 30% unseen). On the temporal test this is almost entirely House 5: its shortfall is 22.7 kWh against a pooled shortfall of 22.3 kWh, while the other four houses together are 0.4 kWh over. Low pooled or per-house energy error can also hide cancellation: House 2's 1.3% energy error comes with 8.3 kWh (36%) of its washing-machine energy missed (6.6 kWh of it in minutes at or above 1000 W), offset by over-prediction elsewhere. Use differences between groups, not absolute levels, for savings estimates.
- Generalisation to a new home is the weak point: House 7's F1 is 0.85 when seen in training and 0.57 when unseen.

## 3. Failure analysis

Figure: `sec3_failure_modes_analysis_excl_plateaus.png`. Cases were selected by rule (longest run of each error type), not by eye.

**Missed energy by house** (temporal test, LightGBM, minutes where the machine truly draws at least 20 W but the model predicts under 20 W; `src/05_failure_checks.py`):

| House | Total kWh | Missed kWh | Missed % | Missed below 1000 W | Missed at or above 1000 W |
|---|---|---|---|---|---|
| 1 | 8.09 | 1.64 | 20.2% | 0.53 | 1.11 |
| 2 | 23.13 | 8.29 | 35.8% | 1.70 | 6.59 |
| 5 | 36.59 | 24.96 | 68.2% | 20.14 | 4.81 |
| 6 | 7.36 | 0.58 | 7.8% | 0.53 | 0.05 |
| 7 | 58.73 | 4.51 | 7.7% | 1.95 | 2.56 |

**Failure 1: false positive, House 7, 2015-04-23 20:38 UTC.** For 24 minutes the model predicted up to about 1.9 kW while the washing machine was off. Within 45 minutes of that point the aggregate peaked at 4.9 kW (mean 1.5 kW), the tumble dryer channel peaked at 3.4 kW (mean 410 W) and the dishwasher at 2.3 kW (mean 494 W). So two other heating appliances were running, and their power overlaps the ~2 kW heating band the model associates with a wash. Interpretation: when other heating loads run, the aggregate alone cannot tell them apart from a washing-machine heating block.

**Failure 2: false negative, House 5, 2015-05-02 01:42-05:43 UTC.** For 217 recorded minutes (a four-hour span with some minutes missing) the washing-machine channel stayed at about 400-530 W (median 498 W) and LightGBM predicted 0 W. What I checked:
- **It is not a baseload problem.** I first suspected the 6-hour baseload feature had absorbed the load. The numbers say otherwise: median aggregate 1113 W and baseload feature about 768 W, so the net power the model sees is about 420 W, close to the true 498 W (baseload recomputed from the test rows only, so approximate).
- **It is not the tail of a heating wash.** In the 3 hours before the run the channel peaked at 284 W and never reached 1000 W.
- **The load is flat.** The classifier's strongest features are local variability (`c_range_15m`), one-minute jumps (`diff_1m`) and minutes in the 1650-3200 W heating band. A flat ~500 W stretch has none of these, so it looks like steady household load.
- **The label itself is doubtful.** During the run the House 5 tumble-dryer channel reads exactly 475 W (maximum equals mean), which looks like a stale or forward-filled value, and about 24 of the 241 clock minutes are missing, possibly rows removed by the `Issues` flag. I did not verify this.
- **It is not a one-off.** LightGBM misses 24.96 kWh of House 5's 36.59 kWh (68%) in the test period, 20.14 kWh of it in minutes below 1000 W. Most of House 5's failure is low-power activity.

Interpretation: the model recognises washing by heating bursts and changes in power, not by steady low-power draw, so it fails where the machine's load is flat and low. Whether those stretches are genuine washing (for example a long programme) or a monitoring artefact could not be settled from the data. A fix would add features for sustained plateaus; I did not do this.

**Other situations where it will struggle:** a washing-machine start masked by a large simultaneous load (a ~10 kW aggregate spike coincides with a missed ~800 W start in House 1); homes not in the training set; any home whose washing programmes lack the ~2 kW heating signature.