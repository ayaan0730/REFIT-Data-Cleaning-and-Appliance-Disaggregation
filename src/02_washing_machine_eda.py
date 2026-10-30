import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURATION & HOUSEHOLD SELECTION
CLEAN_DATA_DIR = "./data/CLEAN_REFIT_081116"
OUTPUT_DIR = "./outputs"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 5 Selected Houses with known Washing Machine channels (from CLEAN_READ_ME_081116.txt)
# Exceeds the "at least 4 households" requirement.
SELECTED_HOUSES = {
    1: {"wm_col": "Appliance5", "brand": "Beko WMC6140", "occupants": 2, "type": "Detached (4 bed)"},
    2: {"wm_col": "Appliance2", "brand": "LG F1289TD", "occupants": 4, "type": "Semi-detached (3 bed)"},
    5: {"wm_col": "Appliance3", "brand": "AEG L99695HWD", "occupants": 4, "type": "Mid-terrace (4 bed)"},
    6: {"wm_col": "Appliance2", "brand": "Bosch Classixx 1200", "occupants": 2, "type": "Detached (4 bed)"},
    7: {"wm_col": "Appliance5", "brand": "Bosch Unknown", "occupants": 4, "type": "Detached (3 bed)"},
}

# Cycle Detection Rule Parameters (Citing NILMTK / Batra et al., 2014)
ON_POWER_THRESHOLD_W = 20.0     # Power >= 20W indicates active drum/pump/heater (ignores standby <5W)
MAX_OFF_GAP_MINS = 15           # Bridge pauses <= 15 mins (soak/drain transitions within a single wash)
MIN_CYCLE_DURATION_MINS = 15    # Real wash cycles last at least 15 minutes
MIN_CYCLE_ENERGY_KWH = 0.05     # Must consume at least 0.05 kWh (50 Wh) to filter momentary spikes


def find_clean_house_file(clean_dir: str, house_num: int) -> str:
    """Finds CLEAN_House{N}.csv inside CLEAN_DATA_DIR."""
    all_csvs = glob.glob(os.path.join(clean_dir, "**", "*.csv"), recursive=True)
    for f in all_csvs:
        fname = os.path.basename(f)
        if re.search(rf"(house[_\s-]*{house_num}|h{house_num})(?!\d)", fname, re.IGNORECASE):
            return f
    raise FileNotFoundError(f"Could not find CLEAN CSV for House {house_num} in {clean_dir}")


def load_clean_house_1min(house_num: int, meta: dict) -> dict:
    """Loads official cleaned REFIT file for a house and resamples to 1-minute resolution."""
    fpath = find_clean_house_file(CLEAN_DATA_DIR, house_num)
    wm_col = meta["wm_col"]
    print(f"Loading House {house_num} ({os.path.basename(fpath)}) -> {wm_col} ({meta['brand']})...")
    
    # Only load columns we need to make reading 5 houses super fast.
    sample = pd.read_csv(fpath, nrows=5)
    cols = list(sample.columns)
    time_col, unix_col, agg_col = cols[0], cols[1], cols[2]
    issues_col = cols[-1] if "issue" in cols[-1].lower() else None
    
    usecols = [unix_col, agg_col, wm_col] + ([issues_col] if issues_col else [])
    df = pd.read_csv(fpath, usecols=usecols, low_memory=False)
    
    rename_map = {unix_col: "Unix", agg_col: "Aggregate", wm_col: "WashingMachine"}
    if issues_col:
        rename_map[issues_col] = "Issues"
    df = df.rename(columns=rename_map)
    
    df["Unix"] = pd.to_numeric(df["Unix"], errors="coerce")
    df = df.dropna(subset=["Unix"])
    df["timestamp"] = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")
    
    # Resample to 1-minute grid
    agg_dict = {"Aggregate": "mean", "WashingMachine": "mean"}
    if "Issues" in df.columns:
        agg_dict["Issues"] = "max"
    df_1min = df.resample("1min").agg(agg_dict)
    
    # Interpolate tiny polling gaps <= 3 mins (since official clean data is ~8s polling)
    df_1min[["Aggregate", "WashingMachine"]] = df_1min[["Aggregate", "WashingMachine"]].interpolate(
        method="time", limit=3, limit_direction="forward"
    )
    
    return {"house": house_num, "df": df_1min, "meta": meta}


def detect_washing_machine_cycles(df_1min: pd.DataFrame, house_num: int) -> pd.DataFrame:
    """
    Identifies active washing-machine operating cycles using a 3-step physical hysteresis rule:
    1. Thresholding: Power >= 20W is flagged as active.
    2. Gap Merging: Short inactive pauses <= 15 minutes are bridged into the same cycle.
    3. Validation: Retain cycles with duration >= 15 mins and total energy >= 0.05 kWh.
    """
    s = df_1min["WashingMachine"].fillna(0.0)
    is_on = (s >= ON_POWER_THRESHOLD_W).astype(int)
    
    # Step 2: Bridge OFF gaps <= MAX_OFF_GAP_MINS (15 mins) between ON segments
    # Identify contiguous OFF blocks
    off_mask = (is_on == 0)
    off_groups = (~off_mask).cumsum()
    off_lengths = off_mask.groupby(off_groups).transform("sum")
    
    # Do not bridge the very first or very last OFF block of the dataset
    first_group = off_groups.iloc[0] if off_mask.iloc[0] else -1
    last_group = off_groups.iloc[-1] if off_mask.iloc[-1] else -1
    bridge_mask = off_mask & (off_lengths <= MAX_OFF_GAP_MINS) & (~off_groups.isin([first_group, last_group]))
    
    active_bridged = is_on.copy()
    active_bridged[bridge_mask] = 1
    
    # Step 3: Extract contiguous active blocks and filter by min duration & energy
    cycle_id = (active_bridged.diff().fillna(0) != 0).cumsum()
    active_df = pd.DataFrame({
        "power": s,
        "aggregate": df_1min["Aggregate"],
        "active": active_bridged,
        "cycle_id": cycle_id
    })
    
    on_blocks = active_df[active_df["active"] == 1]
    if on_blocks.empty:
        return pd.DataFrame()
    
    cycles = []
    for cid, grp in on_blocks.groupby("cycle_id"):
        start_ts = grp.index[0]
        end_ts = grp.index[-1]
        duration_mins = len(grp)
        # Energy in kWh: sum(Watts * (1/60 hour)) / 1000
        energy_kwh = (grp["power"].sum() / 60.0) / 1000.0
        peak_power_w = grp["power"].max()
        mean_power_w = grp["power"].mean()
        
        if duration_mins >= MIN_CYCLE_DURATION_MINS and energy_kwh >= MIN_CYCLE_ENERGY_KWH:
            cycles.append({
                "house": f"House {house_num}",
                "house_num": house_num,
                "start_time": start_ts,
                "end_time": end_ts,
                "duration_mins": duration_mins,
                "energy_kwh": energy_kwh,
                "peak_power_w": peak_power_w,
                "mean_power_w": mean_power_w,
                "hour_of_day": start_ts.hour,
                "day_of_week": start_ts.dayofweek,
                "day_name": start_ts.day_name(),
            })
            
    return pd.DataFrame(cycles)


def generate_eda_plots(house_data: dict, all_cycles_df: pd.DataFrame):
    """Generates the 3 required figures for Section 2."""
    print("\n" + "="*55)
    print("GENERATING SECTION 2 EDA PLOTS")
    print("="*55)
    
    # PLOT (i): Representative 24-Hour Aggregate + Washing Machine Trace
    # Pick a clear day in House 1 with 1 or 2 clean washing machine cycles
    h1_df = house_data[1]["df"]
    h1_cycles = all_cycles_df[all_cycles_df["house_num"] == 1]
    rep_cycle = h1_cycles[(h1_cycles["duration_mins"] >= 60) & (h1_cycles["peak_power_w"] >= 1800)].iloc[10]
    day_start = rep_cycle["start_time"].floor("D")
    day_end = day_start + pd.Timedelta(hours=24)
    day_slice = h1_df.loc[day_start:day_end]
    
    fig, ax1 = plt.subplots(figsize=(14, 5.5))
    ax1.plot(day_slice.index, day_slice["Aggregate"], color="#4daf4a", lw=1.2, alpha=0.8, label="Whole-Home Aggregate (W)")
    ax1.fill_between(day_slice.index, 0, day_slice["WashingMachine"], color="#377eb8", alpha=0.65, label="Washing Machine Load (W)")
    ax1.plot(day_slice.index, day_slice["WashingMachine"], color="#1f4e78", lw=1.2)
    
    ax1.set_title(f"Plot (i): Representative 24-Hour Whole-Home Aggregate vs. Washing Machine Trace (House 1 — {day_start.strftime('%Y-%m-%d')})",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Active Power (Watts)", fontsize=11)
    ax1.set_xlabel("Time of Day (UTC)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    p1_path = os.path.join(FIG_DIR, "sec2_plot1_representative_24h_trace.png")
    plt.savefig(p1_path, dpi=200)
    plt.close()
    print(f"Saved Plot (i)   : {p1_path}")

    # PLOT (ii): Detailed Zoom of Detected Washing-Machine Cycles (Multi-Stage Signature)
    # Show 2 detected cycles side-by-side (e.g. House 1 Beko vs House 2 LG) with shaded detection boundaries
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sample_houses = [(1, "Beko WMC6140"), (2, "LG F1289TD")]
    
    for idx, (h_num, brand) in enumerate(sample_houses):
        hdf = house_data[h_num]["df"]
        h_cyc = all_cycles_df[(all_cycles_df["house_num"] == h_num) & 
                              (all_cycles_df["duration_mins"].between(65, 120)) & 
                              (all_cycles_df["peak_power_w"] > 1800)].iloc[5]
        
        c_start, c_end = h_cyc["start_time"], h_cyc["end_time"]
        pad_start = c_start - pd.Timedelta(minutes=20)
        pad_end = c_end + pd.Timedelta(minutes=20)
        sub = hdf.loc[pad_start:pad_end]
        
        ax = axes[idx]
        ax.plot(sub.index, sub["Aggregate"], color="#999999", lw=1.0, ls="--", alpha=0.7, label="Aggregate Power")
        ax.plot(sub.index, sub["WashingMachine"], color="#d95f02", lw=2.0, label="Washing Machine")
        ax.axvspan(c_start, c_end, color="#fee0b6", alpha=0.45, label=f"Detected Cycle ({h_cyc['duration_mins']}m, {h_cyc['energy_kwh']:.2f} kWh)")
        ax.axhline(ON_POWER_THRESHOLD_W, color="#e41a1c", ls=":", lw=1.2, label=f"Threshold ({ON_POWER_THRESHOLD_W:.0f}W)")
        
        ax.set_title(f"House {h_num} ({brand}) — Detected Cycle Anatomy\n{c_start.strftime('%Y-%m-%d %H:%M')} to {c_end.strftime('%H:%M')} UTC",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Power (Watts)")
        ax.set_xlabel("Time (UTC)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Plot (ii): Detected Washing-Machine Cycles Showing Heating (~2kW), Agitation, and Spin Phases",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p2_path = os.path.join(FIG_DIR, "sec2_plot2_detected_cycles_anatomy.png")
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print(f"Saved Plot (ii)  : {p2_path}")

    # PLOT (iii): Cross-House Comparison (Temporal Habits + Cycle Characteristics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Panel A: Hourly usage profile across houses (% of cycles starting at each hour)
    hourly_counts = all_cycles_df.groupby(["house", "hour_of_day"]).size().unstack(fill_value=0)
    hourly_pct = hourly_counts.div(hourly_counts.sum(axis=1), axis=0) * 100
    for h_label in hourly_pct.index:
        axes[0, 0].plot(hourly_pct.columns, hourly_pct.loc[h_label], marker="o", lw=1.8, label=h_label)
    axes[0, 0].set_title("A. Washing Machine Usage by Hour of Day (% of Cycles Started)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Hour of Day (0–23 UTC)")
    axes[0, 0].set_ylabel("% of Household Cycles")
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: Day of Week usage heatmap (% of cycles across Day of Week per House)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = all_cycles_df.groupby(["house", "day_name"]).size().unstack(fill_value=0)
    dow_counts = dow_counts.reindex(columns=day_order, fill_value=0)
    dow_pct = dow_counts.div(dow_counts.sum(axis=1), axis=0) * 100
    sns.heatmap(dow_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0, 1], cbar_kws={"label": "% of Cycles"})
    axes[0, 1].set_title("B. Day-of-Week Laundry Distribution by Household (%)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("")

    # Panel C: Cycle Duration Comparison (Minutes) across Houses
    sns.boxplot(
        data=all_cycles_df, x="house", y="duration_mins", hue="house",
        palette="Set2", showfliers=False, ax=axes[1, 0], legend=False
    )
    axes[1, 0].set_title("C. Cycle Duration Distribution Across Households (Minutes)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Household")
    axes[1, 0].set_ylabel("Cycle Duration (Minutes)")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel D: Energy per Cycle Comparison (kWh) across Houses
    sns.boxplot(
        data=all_cycles_df, x="house", y="energy_kwh", hue="house",
        palette="Set2", showfliers=False, ax=axes[1, 1], legend=False
    )
    axes[1, 1].set_title("D. Energy Consumed per Cycle Across Households (kWh)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Household")
    axes[1, 1].set_ylabel("Energy per Cycle (kWh)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Plot (iii): Cross-Household Comparison of Washing Machine Usage Patterns & Cycle Characteristics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p3_path = os.path.join(FIG_DIR, "sec2_plot3_cross_house_comparison.png")
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print(f"Saved Plot (iii) : {p3_path}")


if __name__ == "__main__":
    house_data = {}
    summary_rows = []
    all_cycles_list = []

    print("="*60)
    print("SECTION 2: WASHING MACHINE EXPLORATORY ANALYSIS (5 HOUSES)")
    print("="*60)

    for h_num, meta in SELECTED_HOUSES.items():
        res = load_clean_house_1min(h_num, meta)
        house_data[h_num] = res
        df_1m = res["df"]
        
        cycles_df = detect_washing_machine_cycles(df_1m, h_num)
        all_cycles_list.append(cycles_df)
        
        total_mins = len(df_1m)
        missing_mins = int(df_1m["WashingMachine"].isna().sum())
        valid_wm = df_1m["WashingMachine"].dropna()
        zero_or_off_pct = float((valid_wm < ON_POWER_THRESHOLD_W).mean() * 100)
        issues_pct = float((df_1m["Issues"] > 0).mean() * 100) if "Issues" in df_1m.columns else 0.0
        
        summary_rows.append({
            "House": f"House {h_num}",
            "Channel": meta["wm_col"],
            "Appliance_Brand": meta["brand"],
            "Start_Date": df_1m.index.min().strftime("%Y-%m-%d"),
            "End_Date": df_1m.index.max().strftime("%Y-%m-%d"),
            "Total_1Min_Rows": total_mins,
            "Missing_Mins": missing_mins,
            "Missing_Pct": round(missing_mins / total_mins * 100, 2),
            "Off_Standby_Pct (<20W)": round(zero_or_off_pct, 2),
            "Submeter_Issue_Pct": round(issues_pct, 2),
            "Detected_Cycles": len(cycles_df),
            "Cycles_Per_Week": round(len(cycles_df) / (total_mins / (60 * 24 * 7)), 2),
            "Median_Duration_Mins": round(float(cycles_df["duration_mins"].median()), 1) if not cycles_df.empty else 0,
            "Median_Energy_kWh": round(float(cycles_df["energy_kwh"].median()), 3) if not cycles_df.empty else 0,
            "Median_Peak_W": round(float(cycles_df["peak_power_w"].median()), 0) if not cycles_df.empty else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    all_cycles_df = pd.concat(all_cycles_list, ignore_index=True)

    print("\n--- CROSS-HOUSEHOLD WASHING MACHINE SUMMARY TABLE ---")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "sec2_washing_machine_summary.csv"), index=False)
    all_cycles_df.to_csv(os.path.join(OUTPUT_DIR, "sec2_detected_cycles_all_houses.csv"), index=False)

    generate_eda_plots(house_data, all_cycles_df)
    print("\nStep 2 Complete! Summary tables and 3 required EDA plots saved to ./outputs/")
