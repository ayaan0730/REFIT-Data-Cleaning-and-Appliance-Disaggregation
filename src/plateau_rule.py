"""
Shared rule for flagging "stationary plateau" segments on the washing-machine channel.

Why: a real wash programme is bursty (≈2 kW heating blocks, motor ramps, spin, pauses).
A channel that sits at one low power level (100-1000 W) with almost no variation for
2+ hours does not look like a wash programme. It may be another load on that monitor,
a stuck/forward-filled value, or a mislabelled channel.

This is a stated ASSUMPTION, not a fact. The diagnostic script measures what these
segments look like; the modelling script reports results with and without them.

Both scripts import this file so the rule is defined in exactly one place.
"""
import numpy as np
import pandas as pd

PLATEAU_MIN_W = 100.0        # lower power bound of a plateau (W)
PLATEAU_MAX_W = 1000.0       # upper power bound of a plateau (W)
PLATEAU_STD_MAX_W = 25.0     # max rolling std (W) to count as "flat"
PLATEAU_STD_WINDOW = 30      # rolling window for the std (minutes)
PLATEAU_MIN_MINUTES = 120    # a flat stretch must last at least this long (minutes)


def find_plateau_runs(wm: pd.Series):
    """
    wm: washing-machine power at a regular 1-minute index (NaN allowed).
    Returns (mask, runs):
      mask : np.ndarray[bool], True for every minute inside a plateau run
      runs : list of (start_pos, end_pos_exclusive) positional ranges
    """
    rolling_std = wm.rolling(PLATEAU_STD_WINDOW, min_periods=PLATEAU_STD_WINDOW, center=True).std()
    cand = ((wm >= PLATEAU_MIN_W) & (wm <= PLATEAU_MAX_W) & (rolling_std < PLATEAU_STD_MAX_W)).values

    d = np.diff(np.concatenate([[0], cand.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    keep = (ends - starts) >= PLATEAU_MIN_MINUTES
    runs = list(zip(starts[keep].tolist(), ends[keep].tolist()))

    mask = np.zeros(len(wm), dtype=bool)
    for s, e in runs:
        mask[s:e] = True
    return mask, runs
