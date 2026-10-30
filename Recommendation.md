# Recommendation: using the washing-machine disaggregation model

**For:** utility analytics and product teams. **Model:** two-stage LightGBM estimating washing-machine power each minute from 1-minute whole-home active power. Trained and tested on five UK homes (REFIT, 2013-2015).

## What it can do
Detects when a washing machine is running and estimates its power, better than a time-of-day guess in most homes: ON/OFF F1 0.61 pooled, 0.78 to 0.85 in the best homes (Houses 1 and 7), and active-power error 46% lower than always-off on the temporal test (198 vs 365 W) and 30% lower on the unseen house (322 vs 463 W).

## Suitable uses
- **Portfolio-level insight:** when and how often washing happens across many homes.
- **Relative comparison** of cycle energy and duration between households or cohorts.
- **Finding candidate homes** for energy-saving messages, to be tested in a controlled trial (below).

## Not suitable (yet)
- **Billing, or precise appliance-level energy for one home.** Total-energy error was 17% on the temporal test and 30% on an unseen house, and ranged from 1% to 62% by home.
- **Homes the model has not seen, without calibration.** F1 fell from 0.85 to 0.57 for House 7 when it was unseen.
- **Homes with long low-power programmes.** House 5 (F1 0.26) has long flat low-power stretches (about 500 W for hours) that the model misses; it is unclear whether they are genuine washing.
- **Data coarser than 1 minute.**
- **Indian homes**, without retraining and local validation (limitations below).

## Before deployment
Sub-meter a sample of target-market homes; calibrate or fine-tune per home or cohort; rerun the validation on unseen homes; monitor drift.

---

# Section 4 answers

## 4.1 One practical opportunity, and how to measure it

**Opportunity: identify high-energy-per-cycle households and nudge them toward lower-temperature or eco programmes.** In this data, energy per cycle differs about 2x between homes (roughly 0.3 kWh in Houses 1 and 5, 0.55-0.7 kWh in Houses 2, 6 and 7; (Confirmed from sec2_washing_machine_summary.csv)), and the ~2 kW heating blocks account for the majority of the energy consumed in a cycle. Differences between homes may also reflect machine size or load, so this picks candidates; it does not prove the cause.

**How I would measure whether it worked.**
1. Randomise households into tip and control groups.
2. Collect 4-8 weeks of baseline, then run the intervention for 8 weeks.
3. Primary outcome: difference-in-differences of mean estimated kWh per cycle (tip versus control, after versus before).
4. Guardrails: cycles per week (to catch people washing more) and heating-block minutes per cycle.
5. Success threshold, stated in advance as an assumption: a 10% reduction in mean kWh per cycle (a target to agree with the utility before the trial, which would be sized to detect it).
6. The model under-predicts total energy by 17-30%, so rely on group differences, which cancel a stable bias, and spot-check a sub-metered subset.

## 4.2 Limitations
- **Overlapping appliance signatures.** A tumble dryer and a dishwasher running at the same time in House 7 produced a 24-minute false positive: other heating loads of about 2 kW or more look like a wash.
- **Missing data.** Outages (notably February 2014), forward-filled values that can hide dropouts, and 8.5 hours of logger dropout in House 5.
- **Household differences.** Performance ranges from F1 0.85 (House 7, seen) to 0.26 (House 5); an unseen house scored 0.57.
- **UK to India.**
  - Washing in India commonly uses cold water and top-loading or semi-automatic machines (to be verified), so the ~2 kW heating signature the model relies on may be weaker or absent.
  - Large air-conditioner loads and voltage fluctuation change the aggregate.
  - Usage times and household routines differ, and I trained on only five UK homes.
  - I would not deploy there without local sub-metered data to retrain and validate.

## 4.3 What gets harder at 15- or 30-minute data
A wash in these homes lasts about 35-130 minutes, and its ~2 kW heating bursts last only about 5-10 minutes. At 15 or 30 minutes a whole cycle becomes one to four readings, and each burst is averaged into its neighbours, so the sharp edges, short lags and heating-band features this model depends on disappear. ON/OFF detection at 20 W stops being meaningful, overlapping appliances can no longer be told apart, and per-cycle counting and duration become unreliable. What remains is estimating interval energy and usage patterns, with much lower accuracy. I did not test this on resampled data; the argument comes from the structure of the signal, not from an experiment.