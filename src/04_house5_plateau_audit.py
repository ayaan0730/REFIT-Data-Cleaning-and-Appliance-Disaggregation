"""
Step 3b: Washing-machine channel diagnostic (built for House 5, run on all 5 houses).

Question: what are the long, flat ~500 W stretches on House 5's washing-machine channel,
and how much of the House 5 results do they drive?

Outputs (./outputs):
  plateau_summary_all_houses.csv      does the same rule flag anything in the other houses?
  house5_plateau_runs.csv             one row per flagged run (when, how long, aggregate response)
  figures/house5_plateau_diagnostics.png   timeline by month + duration histogram
  figures/house5_plateau_examples.png      the longest / median / shortest runs in context

Run:  python 04_house5_plateau_audit.py     (needs plateau_rule.py in the same folder)
"""
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plateau_rule import (find_plateau_runs, PLATEAU_MIN_W, PLATEAU_MAX_W,
                          PLATEAU_STD_MAX_W, PLATEAU_STD_WINDOW, PLATEAU_MIN_MINUTES)

warnings.filterwarnings("ignore")

CLEAN_DATA_DIR = "./data/CLEAN_REFIT_081116"
OUTPUT_DIR = "./outputs"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

HOUSES = {1: "Appliance5", 2: "Appliance2", 5: "Appliance3", 6: "Appliance2", 7: "Appliance5"}
BUFFER_MINS = 210   # same as the modelling script
ON_W = 20.0


def find_clean_house_file(clean_dir: str, house_num: int) -> str:
    for f in glob.glob(os.path.join(clean_dir, "**", "*.csv"), recursive=True):
        if re.search(rf"(house[_\s-]*{house_num}|h{house_num})(?!\d)", os.path.basename(f), re.IGNORECASE):
            return f
    raise FileNotFoundError(f"Could not find CLEAN CSV for House {house_num}")


def load_house_1min(house_num: int, wm_col: str, all_channels: bool = False) -> pd.DataFrame:
    """1-minute frame with Aggregate + WashingMachine (same cleaning rules as the modelling script)."""
    fpath = find_clean_house_file(CLEAN_DATA_DIR, house_num)
    cols = list(pd.read_csv(fpath, nrows=5).columns)
    unix_col, agg_col = cols[1], cols[2]
    issues_col = cols[-1] if "issue" in cols[-1].lower() else None
    app_cols = [c for c in cols[3:] if c != issues_col] if all_channels else [wm_col]

    num_cols = [agg_col] + app_cols + ([issues_col] if issues_col else [])
    dtypes = {c: "float32" for c in num_cols}
    chunks = list(pd.read_csv(fpath, usecols=[unix_col] + num_cols, dtype=dtypes, chunksize=1_000_000))
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    df[unix_col] = pd.to_numeric(df[unix_col], errors="coerce")
    df = df.dropna(subset=[unix_col])
    df["timestamp"] = pd.to_datetime(df[unix_col], unit="s", utc=True)
    df = (df.drop(columns=[unix_col]).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").set_index("timestamp"))

    agg = {c: "mean" for c in [agg_col] + app_cols}
    if issues_col:
        agg[issues_col] = "max"
    d1 = df.resample("1min").agg(agg)
    d1[[agg_col] + app_cols] = d1[[agg_col] + app_cols].interpolate(method="time", limit=5, limit_direction="forward")
    if issues_col:
        d1.loc[d1[issues_col] == 1, [agg_col] + app_cols] = np.nan
        d1 = d1.drop(columns=[issues_col])
    d1.loc[d1[wm_col] > d1[agg_col] + 50, [agg_col, wm_col]] = np.nan

    d1 = d1.rename(columns={agg_col: "Aggregate", wm_col: "WashingMachine"})
    d1["Aggregate"] = d1["Aggregate"].clip(0, 20000)
    d1["WashingMachine"] = d1["WashingMachine"].clip(0, 3500)
    return d1


def mean_slice(arr: np.ndarray, a: int, b: int) -> float:
    a, b = max(a, 0), min(b, len(arr))
    return float(np.nanmean(arr[a:b])) if b > a else np.nan


def main():
    print("=" * 70)
    print("WASHING-MACHINE CHANNEL DIAGNOSTIC: stationary-plateau segments")
    print(f"Rule: {PLATEAU_MIN_W:.0f}-{PLATEAU_MAX_W:.0f} W, rolling {PLATEAU_STD_WINDOW}-min std < "
          f"{PLATEAU_STD_MAX_W:.0f} W, lasting >= {PLATEAU_MIN_MINUTES} min")
    print("=" * 70)

    # ---------- 1) Does the rule flag anything in the other houses? ----------
    rows = []
    for h, col in HOUSES.items():
        d = load_house_1min(h, col)
        wm = d["WashingMachine"]
        mask, runs = find_plateau_runs(wm)
        total_kwh = float(wm.sum()) / 60000.0
        plat_kwh = float(wm[mask].sum()) / 60000.0
        rows.append({
            "House": h, "Runs": len(runs), "Plateau_hours": round(mask.sum() / 60, 1),
            "Plateau_kWh": round(plat_kwh, 2), "Total_WM_kWh": round(total_kwh, 2),
            "Share_of_WM_energy_%": round(100 * plat_kwh / max(total_kwh, 1e-9), 1),
        })
        del d
    summ = pd.DataFrame(rows)
    print("\n1) PLATEAU SEGMENTS BY HOUSE (same rule, all houses)")
    print(summ.to_string(index=False))
    summ.to_csv(os.path.join(OUTPUT_DIR, "plateau_summary_all_houses.csv"), index=False)

    # ---------- 2) House 5 in detail ----------
    d5 = load_house_1min(5, HOUSES[5], all_channels=True)
    wm = d5["WashingMachine"]
    agg = d5["Aggregate"].values
    wmv = wm.values
    mask, runs = find_plateau_runs(wm)
    if not runs:
        print("\nNo plateau runs found in House 5 under this rule. Nothing to exclude.")
        return

    run_rows = []
    for s, e in runs:
        seg = wmv[s:e]
        # The flagged run is ~15 min shorter than the real plateau at each end (centred 30-min std window),
        # so compare inside-plateau windows against outside windows placed beyond that eroded margin.
        step_start = mean_slice(agg, s + 10, s + 40) - mean_slice(agg, s - 60, s - 25)
        step_end = mean_slice(agg, e - 40, e - 10) - mean_slice(agg, e + 25, e + 60)
        run_rows.append({
            "start_utc": d5.index[s], "end_utc": d5.index[e - 1], "minutes": e - s,  # ~30 min shorter than the true plateau (edge erosion)
            "mean_W": round(float(np.nanmean(seg)), 1), "std_W": round(float(np.nanstd(seg)), 1),
            "start_hour_london": int(d5.index[s].tz_convert("Europe/London").hour),
            "pct_identical_minutes": round(100 * float(np.mean(np.diff(seg) == 0)), 1),
            "agg_step_at_start_W": round(step_start, 0), "agg_step_at_end_W": round(step_end, 0),
        })
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(os.path.join(OUTPUT_DIR, "house5_plateau_runs.csv"), index=False)

    print(f"\n2) HOUSE 5: {len(runs)} plateau runs, {mask.sum() / 60:.1f} h in total")
    print(runs_df.describe().loc[["min", "50%", "max"], ["minutes", "mean_W", "std_W", "pct_identical_minutes",
                                                          "agg_step_at_start_W", "agg_step_at_end_W"]].round(1).to_string())
    print("\n   Longest 8 runs:")
    print(runs_df.sort_values("minutes", ascending=False).head(8).to_string(index=False))
    hours = runs_df["start_hour_london"].value_counts().sort_index()
    print("\n   Run start hour (Europe/London):", dict(hours))

    monthly = pd.Series(mask.astype(int), index=d5.index.tz_localize(None))
    monthly = (monthly.groupby(monthly.index.to_period("M")).sum() / 60).round(1)
    monthly = monthly[monthly > 0]
    print("\n   Plateau hours by month:")
    print(monthly.to_string())

    # ---------- 3) What do the other channels do during plateaus? ----------
    chan = [c for c in d5.columns if c != "Aggregate"]
    cmp_df = pd.DataFrame({"mean_W_inside_plateaus": d5.loc[mask, chan].mean(),
                           "mean_W_outside": d5.loc[~mask, chan].mean()}).round(1)
    print("\n3) MEAN POWER PER CHANNEL, INSIDE vs OUTSIDE PLATEAUS "
          "(a channel that drops to ~0 inside may be the real source)")
    print(cmp_df.to_string())

    # ---------- 4) How much of the test-period results do plateaus drive? ----------
    valid = d5.index[d5["WashingMachine"].notna() & d5["Aggregate"].notna()]
    test_idx = valid[int(len(valid) * 0.85) + BUFFER_MINS:]
    ms = pd.Series(mask, index=d5.index).loc[test_idx]
    wm_t = d5.loc[test_idx, "WashingMachine"]
    on = wm_t >= ON_W
    print("\n4) HOUSE 5 TEMPORAL TEST PERIOD (approximate)")
    print(f"   plateau minutes            : {ms.mean() * 100:.1f}% of test minutes")
    print(f"   plateau share of ON-minutes: {100 * (ms & on).sum() / max(on.sum(), 1):.1f}%")
    print(f"   plateau share of WM energy : {100 * wm_t[ms].sum() / max(wm_t.sum(), 1e-9):.1f}%")

    # ---------- plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(monthly.index.astype(str), monthly.values, color="#d95f02")
    axes[0].set_title("House 5: plateau hours per month", fontweight="bold")
    axes[0].set_ylabel("Hours")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(runs_df["minutes"], bins=20, color="#7570b3")
    axes[1].set_title("House 5: plateau run durations", fontweight="bold")
    axes[1].set_xlabel("Minutes")
    axes[1].set_ylabel("Runs")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "house5_plateau_diagnostics.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    order = runs_df.sort_values("minutes").index.tolist()
    picks = sorted({order[-1], order[len(order) // 2], order[0]})
    fig, axes = plt.subplots(len(picks), 1, figsize=(14, 3.6 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, picks):
        s, e = runs[k]
        a, b = max(0, s - 180), min(len(d5), e + 180)
        sub = d5.iloc[a:b]
        ax.plot(sub.index, sub["Aggregate"], color="#888888", lw=1.0, ls=":", label="Aggregate")
        ax.plot(sub.index, sub["WashingMachine"], color="#2ca25f", lw=1.8, label="Washing-machine channel")
        ax.axvspan(d5.index[s], d5.index[e - 1], color="#fdae6b", alpha=0.25, label="Flagged plateau")
        ax.set_title(f"Run {k}: {e - s} min, mean {np.nanmean(wmv[s:e]):.0f} W, starts {d5.index[s]:%Y-%m-%d %H:%M} UTC",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Power (W)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "house5_plateau_examples.png")
    plt.savefig(p2, dpi=200)
    plt.close()

    print(f"\nSaved: {p1}\nSaved: {p2}")
    print("Saved CSVs: plateau_summary_all_houses.csv, house5_plateau_runs.csv")
    print("\nHOW TO READ THIS")
    print(" - agg_step_at_start/end close to mean_W  -> the load is real: the house really draws that power.")
    print("   (the open question is only WHICH appliance it is)")
    print(" - agg_step near 0                          -> likely a sensor artefact / stuck value.")
    print(" - pct_identical_minutes near 100           -> forward-filled stale readings.")
    print(" - plateaus in other houses ~0 h            -> the problem is specific to House 5's channel.")


if __name__ == "__main__":
    main()
