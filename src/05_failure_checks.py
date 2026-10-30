"""
Step 3c: evidence for the failure-case write-up. Reads the CACHED predictions (no retraining).

Run (from the repo root):  python src/05_failure_checks.py [path_to_cache.pkl]
      (default cache: ./outputs/sec3_cache_excl_plateaus.pkl)

It prints:
  A) the held-out test period of every house            -> "state the homes and periods"
  B) how much washing-machine energy LightGBM misses, split by true power band, per house
  C) the longest false-negative run: what the aggregate, the baseload feature and the target
     look like inside it, and what the channel did in the 3 h BEFORE it
  D) which other appliance channels were running during the false-negative and false-positive cases
and saves ./outputs/figures/failure_fn_context.png
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib


def _load_audit_module():
    # module names that start with a digit cannot be used in an `import` statement, hence importlib
    for name in ("04_house5_plateau_audit", "step3b_house5_diagnostic", "step3bhouse5"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise
    raise ImportError("Could not find 04_house5_plateau_audit.py. Run from the repo root: python src/05_failure_checks.py")


_audit = _load_audit_module()
load_house_1min, HOUSES = _audit.load_house_1min, _audit.HOUSES

warnings.filterwarnings("ignore")
CACHE = sys.argv[1] if len(sys.argv) > 1 else "./outputs/sec3_cache_excl_plateaus.pkl"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)
ON_W = 20.0

# Channel names from the CLEAN_READ_ME (Appliance1..9) for the houses used here
NAMES = {
    1: ["Fridge", "Freezer(1)", "Freezer(2)", "Washer Dryer", "Washing Machine", "Dishwasher", "Computer", "Television Site", "Electric Heater"],
    2: ["Fridge-Freezer", "Washing Machine", "Dishwasher", "Television Site", "Microwave", "Toaster", "Hi-Fi", "Kettle", "Overhead Fan"],
    5: ["Fridge-Freezer", "Tumble Dryer", "Washing Machine", "Dishwasher", "Desktop Computer", "Television Site", "Microwave", "Kettle", "Toaster"],
    6: ["Freezer", "Washing Machine", "Dishwasher", "MJY Computer", "TV/Satellite", "Microwave", "Kettle", "Toaster", "PGM Computer"],
    7: ["Fridge", "Freezer(1)", "Freezer(2)", "Tumble Dryer", "Washing Machine", "Dishwasher", "Television Site", "Toaster", "Kettle"],
}


def run_bounds(mask: np.ndarray, houses: np.ndarray):
    """(start, end_exclusive) of the longest run of True; runs never cross a house boundary."""
    m = mask.astype(np.int8)
    change = np.flatnonzero((np.diff(m) != 0) | (np.diff(houses) != 0)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(m)]])
    sel = m[starts] == 1
    if not sel.any():
        return None
    lens = (ends - starts)[sel]
    k = int(np.argmax(lens))
    return int(starts[sel][k]), int(ends[sel][k])


def channel_check(house: int, ts: pd.Timestamp, half_min: int):
    d = load_house_1min(house, HOUSES[house], all_channels=True)
    w = d.loc[ts - pd.Timedelta(minutes=half_min): ts + pd.Timedelta(minutes=half_min)]
    labels = {"Aggregate": "Aggregate", "WashingMachine": "WashingMachine (target channel)"}
    for i, n in enumerate(NAMES.get(house, []), start=1):
        labels.setdefault(f"Appliance{i}", f"Appliance{i} {n}")
    out = pd.DataFrame({"max_W": w.max(), "mean_W": w.mean()}).round(0)
    out.index = [labels.get(i, i) for i in out.index]
    print(out.sort_values("max_W", ascending=False).to_string())


def main():
    c = pickle.load(open(CACHE, "rb"))
    t = c["test_slim"].copy()
    t["pred"] = c["preds"]["TwoStage_LightGBM"]
    houses = t["house"].values
    wm = t["WashingMachine"].values
    pred = t["pred"].values

    # ---- A) test periods ----
    print("A) HELD-OUT TEST PERIOD PER HOUSE (temporal split)")
    for h, g in t.groupby("house"):
        print(f"   House {int(h)}: {g.index.min():%Y-%m-%d} -> {g.index.max():%Y-%m-%d}   ({len(g):,} min)")

    # ---- B) missed energy by power band ----
    print("\nB) LightGBM MISSED ENERGY (truth >= 20 W, prediction < 20 W), by true power band")
    rows = []
    for h, g in t.groupby("house"):
        w, q = g["WashingMachine"].values, g["pred"].values
        miss = (w >= ON_W) & (q < ON_W)
        tot = w.sum() / 60000
        rows.append({"House": int(h), "Total_kWh": round(tot, 2),
                     "Missed_kWh": round(w[miss].sum() / 60000, 2),
                     "Missed_%": round(100 * w[miss].sum() / 60000 / max(tot, 1e-9), 1),
                     "Missed_<1000W_kWh": round(w[miss & (w < 1000)].sum() / 60000, 2),
                     "Missed_>=1000W_kWh": round(w[miss & (w >= 1000)].sum() / 60000, 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- C) longest false-negative run ----
    fn = run_bounds((wm > 400) & (pred < 50), houses)
    if fn is None:
        print("\nC) No false-negative run (truth > 400 W while prediction < 50 W) found.")
    else:
        s, e = fn
        h = int(houses[s])
        ts0, ts1 = t.index[s], t.index[e - 1]
        g = t[t["house"] == h]
        # same definition as the model's baseload feature (approximate: built from test rows only)
        base = g["agg_raw"].rolling(360, min_periods=1, center=True).quantile(0.10)
        seg = g.loc[ts0:ts1]
        net = (seg["agg_raw"] - base.loc[seg.index]).clip(lower=0)
        print(f"\nC) LONGEST FALSE-NEGATIVE RUN: House {h}, {ts0:%Y-%m-%d %H:%M} -> {ts1:%H:%M} UTC ({e - s} min)")
        print(f"   median aggregate            : {seg['agg_raw'].median():7.0f} W")
        print(f"   median baseload feature     : {base.loc[seg.index].median():7.0f} W")
        print(f"   median net (agg - baseload) : {net.median():7.0f} W   <- what the model's net-power features see")
        print(f"   median TRUE machine power   : {seg['WashingMachine'].median():7.0f} W")
        pre = g.loc[ts0 - pd.Timedelta(minutes=180): ts0 - pd.Timedelta(minutes=1), "WashingMachine"]
        print(f"   channel in the 3 h BEFORE the run: max {pre.max():.0f} W | minutes >= 1000 W: {(pre >= 1000).sum()} "
              f"| minutes >= 20 W: {(pre >= ON_W).sum()}")
        print("   (heating blocks >= 1000 W just before the plateau suggest it is the tail of a wash programme)")
        print("\n   Every 20 min inside the run:")
        tab = pd.DataFrame({"aggregate": seg["agg_raw"], "baseload": base.loc[seg.index], "net": net,
                            "true_WM": seg["WashingMachine"], "pred_LGBM": seg["pred"]}).iloc[::20].round(0)
        print(tab.to_string())

        ctx = g.loc[ts0 - pd.Timedelta(minutes=180): ts1 + pd.Timedelta(minutes=180)]
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(ctx.index, ctx["agg_raw"], color="#888888", lw=1.1, ls=":", label="Aggregate")
        ax.plot(ctx.index, base.loc[ctx.index], color="#7570b3", lw=1.6, label="Baseload feature (6 h rolling 10th pct)")
        ax.plot(ctx.index, ctx["WashingMachine"], color="#2ca25f", lw=1.8, label="Washing-machine channel")
        ax.plot(ctx.index, ctx["pred"], color="#e41a1c", lw=1.6, ls="--", label="LightGBM prediction")
        ax.axvspan(ts0, ts1, color="#fdae6b", alpha=0.2, label="False-negative run")
        ax.set_title(f"House {h}: longest false-negative run in context (±3 h)", fontweight="bold")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(FIG_DIR, "failure_fn_context.png")
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"\n   Saved: {p}")
        try:
            print(f"\nD1) OTHER CHANNELS DURING THE FALSE-NEGATIVE RUN (House {h})")
            channel_check(h, ts0 + (ts1 - ts0) / 2, int((e - s) // 2))
        except Exception as ex:
            print("   channel check skipped:", ex)

    # ---- D) false-positive case: which other appliance was running? ----
    fp = run_bounds((wm < 10) & (pred > 200), houses)
    if fp is None:
        print("\nD2) No false-positive run found.")
    else:
        s, e = fp
        h = int(houses[s])
        mid = t.index[(s + e) // 2]
        print(f"\nD2) FALSE-POSITIVE CASE: House {h}, around {mid:%Y-%m-%d %H:%M} UTC ({e - s} min); channels ±45 min")
        try:
            channel_check(h, mid, 45)
        except Exception as ex:
            print("   channel check skipped:", ex)


if __name__ == "__main__":
    main()