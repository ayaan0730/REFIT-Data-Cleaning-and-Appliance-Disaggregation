# Washing-machine disaggregation from whole-home power (REFIT)

Take-home assignment for Flock Energy. Cleans raw REFIT smart-meter data, explores washing-machine usage across five UK households, and estimates washing-machine power from the whole-home aggregate at 1-minute resolution.

**Headline result.** A two-stage LightGBM model (ON/OFF classifier, then power regressor) clearly beats an always-off baseline and a time-of-day profile baseline, but it generalises less well to an unseen house, and it fails on one house with long low-power programmes. Details and failure analysis are in [RESULTS.md](RESULTS.md).

| Split (test set) | Model | Overall MAE (W) | Active-ON MAE (W) | ON F1 | Energy error |
|---|---|---|---|---|---|
| Temporal, last 15% of each house | Always-off | 15.70 | 365.2 | 0.000 | 100% |
| | Time-of-day profile | 24.81 | 349.4 | 0.135 | 32.4% |
| | **Two-stage LightGBM** | **11.87** | **198.1** | **0.611** | 16.6% |
| | Seq2Point 1D-CNN | 15.18 | 195.0 | 0.505 | 4.6% (see caveat in RESULTS.md) |
| Unseen house (House 7) | Always-off | 25.04 | 463.4 | 0.000 | 100% |
| | Time-of-day profile | 33.59 | 447.7 | 0.168 | 58.4% |
| | **Two-stage LightGBM** | **22.56** | 322.4 | **0.567** | 29.7% |
| | Seq2Point 1D-CNN | 37.97 | 245.5 | 0.392 | 73.6% |

Source: `outputs/sec3_model_evaluation_metrics_excl_plateaus.csv`.

## Repository layout

```text
README.md  DATA.md  RESULTS.md  Recommendation.md
requirements.txt
src/
  01_clean_raw_house1.py        Section 1: raw House 1 inspection and cleaning
  02_washing_machine_eda.py     Section 2: washing-machine EDA, cycle detection
  03_disaggregation_model.py    Section 3: baselines, LightGBM, CNN, evaluation, plots
  04_house5_plateau_audit.py    audit of flat segments on House 5's washing-machine channel
  05_failure_checks.py          evidence for the failure-case analysis (reads cached predictions)
  plateau_rule.py               shared rule for flagging flat "plateau" segments
outputs/                        result tables and figures produced by the scripts
data/                           NOT committed (see Setup)

## Setup

Tested with Python 3.13 on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Download the data (Strathclyde REFIT, CC BY 4.0) and extract into `data/`:

- Cleaned REFIT: https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z  ->  `data/CLEAN_REFIT_081116/`
- Raw REFIT (Section 1 only): https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z  ->  'data/REFIT_RAW_081116/'

## Run order

Run every script from the repository root.

```bash
python src/01_clean_raw_house1.py
python src/02_washing_machine_eda.py
python src/03_disaggregation_model.py          # about 2-3 mins on CPU; caches predictions
python src/04_house5_plateau_audit.py
python src/05_failure_checks.py                # needs the cache written by script 03
# optional sensitivity run that keeps the flagged plateau minutes:
python src/03_disaggregation_model.py --with-plateaus
```

Outputs are written to `outputs/` (tables) and `outputs/figures/`. Script 03 tags its files `_excl_plateaus` or `_all_data` so the two runs never overwrite each other.

## What I built

1. **Cleaning (Section 1).**  Parsed UTC timestamps and deduplicated overlapping records by keeping the last valid entry. Handled missingness by interpolating short gaps (≤ 5 mins) and leaving long outages (> 5 mins) as NaN to prevent artificial flatlines. Removed physical impossibility spikes (like 16-bit register overflows at 65,535 W) before resampling to 1-minute intervals. See DATA.md.
2. **EDA (Section 2).** Cycle detection on the washing-machine channel of five houses, usage by hour and weekday, cross-house cycle comparison. See [RESULTS.md](RESULTS.md).
3. **Model (Section 3).** Per-minute washing-machine power from the aggregate. Two baselines, a two-stage hurdle LightGBM (main model) and a Seq2Point 1D-CNN (comparison), two validation designs.
4. **Practical implications (Section 4)** are in [Recommendation.md](Recommendation.md).

## Key decisions and assumptions

Decision,Why
1-minute resolution,Required by the brief; short enough to see the ~2 kW heating blocks that identify a wash.
Active threshold 20 W and merge rule 15 minutes,"A physical wash programme pauses with near-zero draw between wash, rinse, and spin phases; bridging ≤ 15 min gaps prevents fragmenting a single physical laundry load into multiple events. The same rule is used for EDA and for the model's ON/OFF target."
"Hurdle model (classify, then regress)",The machine is idle most of the time and bursty when on; one regressor smears predictions toward zero.
Temporal split with a 210-minute buffer on each side of every boundary,"A row's features can look about 210 minutes away, so the gap prevents train and evaluation rows sharing aggregate samples."
"Unseen-house split (train 1, 2, 5 / validate 6 / test 7)",Tests what a utility actually needs: a home the model has never seen.
"Flat ""plateau"" minutes excluded","8.5 h in House 5 (0.5% of its washing-machine energy) are exactly constant and every other channel reads 0 W, which looks like logger dropout. Results barely change (RESULTS.md)."

## Trade-offs

- A simple model with hand-built features is easy to explain and fast to rerun; the CNN needs more tuning and is unstable across runs.
- Excluding plateau minutes is a judgment; I report results with and without it.
- Physical Interpretability vs. Deep Automation: Choosing explicit feature engineering (rolling lags, past vs. future asymmetry) over an end-to-end Deep Learning approach (Seq2Point) traded deep automated feature extraction for physical interpretability, faster CPU training, and precise zero-power idle predictions.

## AI tools used

I used LLMs (Gemini / Claude) as a thought partner and coding assistant: for scaffolding boilerplate matplotlib code, restructuring the Section 3 script for memory optimization (chunked float32 reads and zero-copy NumPy sliding_window_view), writing the plateau audit/failure-check diagnostic scripts, and formatting markdown tables. I ran all code on the REFIT data myself locally, reviewed and adjusted the design choices listed above (e.g., hysteresis thresholds and leakage buffers), and authored the physical and operational interpretations in the deliverables.

## Possible improvements

- Add features that capture sustained low-power plateaus (for example a long-window minimum of net power and time since the last large change), and check whether such flat stretches are genuine washing. The House 5 failure analysis (RESULTS.md) ruled out a simple baseload-window explanation: the model misses flat low-power load because it identifies washing by heating bursts and power changes.
- Per-home calibration with a short sub-metered period for new houses.
- Tune the CNN's ON threshold on validation (the LightGBM gate gets one; the CNN does not) and average several seeds.
- Test the model on 15- and 30-minute resampled data to quantify the resolution loss.
- Evaluate on more homes and on non-UK data.

## References

- Murray, D., et al. (2015). A data management platform for personalised real-time energy feedback. *8th International Conference on Energy Efficiency in Domestic Appliances and Lighting* (the citation requested in the REFIT readme).
- Murray, D., Stankovic, L., Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. *Scientific Data*, 4, 160122.
- Zhang, C., Zhong, M., Wang, Z., Goddard, N., Sutton, C. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. *AAAI*. 
- Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*. 