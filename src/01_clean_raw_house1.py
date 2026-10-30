import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIGURATION & PATHS (Matched to your folder)
RAW_DATA_DIR = "./data/REFIT_RAW_081116"
OUTPUT_DIR = "./outputs"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# House 1 Channel Mapping (from CLEAN_READ_ME_081116.txt)
HOUSE1_CHANNELS = {
    "Aggregate": "Aggregate",
    "Appliance1": "Fridge",
    "Appliance2": "Freezer_1",
    "Appliance3": "Freezer_2",
    "Appliance4": "Washer_Dryer",
    "Appliance5": "Washing_Machine",
    "Appliance6": "Dishwasher",
    "Appliance7": "Computer",
    "Appliance8": "Television_Site",
    "Appliance9": "Electric_Heater",
}

APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]
ALL_POWER_COLS = ["Aggregate"] + APPLIANCE_COLS

# Physical thresholds based on UK grid & REFIT README
MAX_AGGREGATE_WATTS = 20000.0  # UK residential single-phase max ~23kW; >20kW is sensor spike
MAX_APPLIANCE_WATTS = 4000.0   # Explicitly stated in REFIT README (>4000W = spike)
SHORT_GAP_LIMIT_MINS = 5       # Interpolate gaps <= 5 minutes; flag > 5 minutes as outage


def find_house1_files(raw_dir: str) -> list:
    """Recursively finds ONLY House 1 CSV files (avoiding House 10, 11, etc.)."""
    all_csvs = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    house1_files = []
    for f in all_csvs:
        fname = os.path.basename(f)
        # Match House1, House_1, H1, etc. where '1' is NOT followed by another digit (like 10, 11)
        if re.search(r"(house[_\s-]*1|h1)(?!\d)", fname, re.IGNORECASE):
            house1_files.append(f)
    return sorted(house1_files)


def load_raw_house1(raw_dir: str) -> pd.DataFrame:
    """Loads Raw House 1 CSV(s), handling single file or _Part1 / _Part2 splits."""
    files = find_house1_files(raw_dir)
    if not files:
        all_files = glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
        raise FileNotFoundError(
            f"Could not find House 1 CSV in {raw_dir}.\nFiles found in directory: {all_files[:15]}"
        )
    
    print(f"Found Raw House 1 file(s): {[os.path.basename(f) for f in files]}")
    dfs = []
    for f in files:
        print(f"Loading {os.path.basename(f)} ...")
        df_part = pd.read_csv(f, low_memory=False)
        
        # Standardize column names: Time/DateTime, Unix, Aggregate, Appliance1..9
        cols = list(df_part.columns)
        if len(cols) >= 11:
            rename_map = {
                cols[0]: "DateTime",
                cols[1]: "Unix",
                cols[2]: "Aggregate",
            }
            for i in range(1, 10):
                if 2 + i < len(cols):
                    rename_map[cols[2 + i]] = f"Appliance{i}"
            df_part = df_part.rename(columns=rename_map)
        
        keep_cols = ["DateTime", "Unix"] + [c for c in ALL_POWER_COLS if c in df_part.columns]
        df_part = df_part[keep_cols]
        dfs.append(df_part)
        
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows loaded: {len(df_raw):,}")
    return df_raw


def inspect_raw_data(df: pd.DataFrame) -> dict:
    """Audits raw data quality before any modifications (Requirement 1a)."""
    print("\n" + "="*55)
    print("1. RAW DATA QUALITY AUDIT (HOUSE 1)")
    print("="*55)
    report = {}
    
    # 1. Timestamp parsing check
    unix_numeric = pd.to_numeric(df["Unix"], errors="coerce")
    dt_parsed = pd.to_datetime(unix_numeric, unit="s", utc=True, errors="coerce")
    
    # Also check string DateTime parsing if Unix is missing
    if dt_parsed.isna().all():
        dt_parsed = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
        unix_numeric = dt_parsed.astype("int64") // 10**9

    report["total_raw_rows"] = len(df)
    report["unparseable_timestamps"] = int(dt_parsed.isna().sum())
    
    valid_ts = unix_numeric.dropna().values
    diffs = np.diff(valid_ts)
    
    # 2. Ordering & Duplicates
    report["out_of_order_timestamps"] = int(np.sum(diffs < 0))
    report["exact_duplicate_timestamps"] = int(pd.Series(valid_ts).duplicated().sum())
    
    # Sort timestamps to inspect true sampling intervals
    sorted_ts = np.sort(np.unique(valid_ts))
    sorted_diffs = np.diff(sorted_ts)
    report["median_sampling_interval_sec"] = float(np.median(sorted_diffs))
    report["mean_sampling_interval_sec"] = float(np.mean(sorted_diffs))
    report["irregular_intervals_gt_10s"] = int(np.sum(sorted_diffs > 10))
    report["outages_gt_5min_count"] = int(np.sum(sorted_diffs > 300))
    report["max_outage_hours"] = float(np.max(sorted_diffs) / 3600.0)
    
    # 3. Missing values & Extreme/Impossible values per channel
    channel_stats = {}
    for col in ALL_POWER_COLS:
        s = pd.to_numeric(df[col], errors="coerce")
        limit = MAX_AGGREGATE_WATTS if col == "Aggregate" else MAX_APPLIANCE_WATTS
        channel_stats[col] = {
            "missing_nan_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 3),
            "zero_pct": round(float((s == 0).mean() * 100), 2),
            "negative_values": int((s < 0).sum()),
            "extreme_spikes_gt_limit": int((s > limit).sum()),
            "max_raw_wattage": float(s.max()) if not s.dropna().empty else np.nan,
            "min_raw_wattage": float(s.min()) if not s.dropna().empty else np.nan,
        }
    report["channel_stats"] = channel_stats
    
    print(f"Total Raw Rows           : {report['total_raw_rows']:,}")
    print(f"Date Range (UTC)         : {dt_parsed.min()} to {dt_parsed.max()}")
    print(f"Unparseable Timestamps   : {report['unparseable_timestamps']:,}")
    print(f"Out-of-order Jumps (<0s) : {report['out_of_order_timestamps']:,}")
    print(f"Duplicate Timestamps     : {report['exact_duplicate_timestamps']:,}")
    print(f"Median Sampling Interval : {report['median_sampling_interval_sec']:.1f}s (Mean: {report['mean_sampling_interval_sec']:.2f}s)")
    print(f"Irregular Gaps (>10s)    : {report['irregular_intervals_gt_10s']:,}")
    print(f"Long Outages (>5 mins)   : {report['outages_gt_5min_count']:,} (Max Outage: {report['max_outage_hours']:.1f} hours)")
    
    stats_df = pd.DataFrame(channel_stats).T
    stats_df.index = [f"{c} ({HOUSE1_CHANNELS[c]})" for c in stats_df.index]
    print("\nChannel-Level Quality Summary:")
    print(stats_df.to_string())
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "house1_raw_audit_table.csv"))
    return report


def clean_and_resample_house1(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes documented cleaning pipeline (Requirement 1b)."""
    print("\n" + "="*55)
    print("2. CLEANING & 1-MINUTE RESAMPLING PIPELINE")
    print("="*55)
    df = df_raw.copy()
    
    # 1. Coerce numeric & parse timestamps
    df["Unix"] = pd.to_numeric(df["Unix"], errors="coerce")
    if df["Unix"].isna().all():
        df["timestamp"] = pd.to_datetime(df["DateTime"], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["Unix"], unit="s", utc=True, errors="coerce")
        
    df = df.dropna(subset=["timestamp"])
    for col in ALL_POWER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # 2. Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 3. Remove exact timestamp duplicates (Rule: keep 'last' logged reading at that timestamp)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    print(f"[Step 3] Removed {before_dedup - len(df):,} duplicate timestamps (Rule: keep='last').")
    
    df = df.set_index("timestamp")
    
    # Save a 1-minute raw snapshot BEFORE outlier removal/interpolation for before-vs-after plotting
    df_raw_1min = df[ALL_POWER_COLS].resample("1min").mean()
    
    # 4. Filter physically impossible values -> NaN
    for col in ALL_POWER_COLS:
        neg_mask = df[col] < 0
        if neg_mask.sum() > 0:
            print(f"[Step 4] Replaced {neg_mask.sum():,} negative values in {col} with NaN")
            df.loc[neg_mask, col] = np.nan
            
    agg_spike_mask = df["Aggregate"] > MAX_AGGREGATE_WATTS
    print(f"[Step 4] Replaced {agg_spike_mask.sum():,} Aggregate spikes > {MAX_AGGREGATE_WATTS}W with NaN")
    df.loc[agg_spike_mask, "Aggregate"] = np.nan
    
    for col in APPLIANCE_COLS:
        spike_mask = df[col] > MAX_APPLIANCE_WATTS
        if spike_mask.sum() > 0:
            print(f"[Step 4] Replaced {spike_mask.sum():,} spikes > {MAX_APPLIANCE_WATTS}W in {col} ({HOUSE1_CHANNELS[col]}) with NaN")
            df.loc[spike_mask, col] = np.nan
            
    # 5. Resample to 1-minute regular grid (mean power over each 1-min bucket)
    df_1min = df[ALL_POWER_COLS].resample("1min").mean()
    
    # 6. Identify gap lengths & handle short vs. long outages
    is_missing = df_1min["Aggregate"].isna()
    gap_groups = (~is_missing).cumsum()
    gap_sizes = is_missing.groupby(gap_groups).transform("sum")
    
    df_1min["is_long_outage"] = (is_missing & (gap_sizes > SHORT_GAP_LIMIT_MINS)).astype(int)
    df_1min["is_short_gap_filled"] = (is_missing & (gap_sizes <= SHORT_GAP_LIMIT_MINS)).astype(int)
    
    # Linearly interpolate ONLY short gaps (limit = 5 minutes); leave longer outages as NaN & flagged
    df_1min[ALL_POWER_COLS] = df_1min[ALL_POWER_COLS].interpolate(
        method="time", limit=SHORT_GAP_LIMIT_MINS, limit_direction="forward"
    )
    
    # 7. Flag sub-metering discrepancy (sum of IAMs > Aggregate)
    iam_sum = df_1min[APPLIANCE_COLS].sum(axis=1, min_count=1)
    df_1min["submeter_exceeds_agg_flag"] = (iam_sum > df_1min["Aggregate"]).astype(int)
    
    print(f"[Step 5] 1-Minute Grid Total Rows      : {len(df_1min):,}")
    print(f"[Step 6] Short gaps filled (<=5 mins)  : {df_1min['is_short_gap_filled'].sum():,} mins ({df_1min['is_short_gap_filled'].mean()*100:.2f}%)")
    print(f"[Step 6] Long outages flagged (>5 mins): {df_1min['is_long_outage'].sum():,} mins ({df_1min['is_long_outage'].mean()*100:.2f}%)")
    print(f"[Step 7] Sub-meter > Aggregate flagged : {df_1min['submeter_exceeds_agg_flag'].sum():,} mins ({df_1min['submeter_exceeds_agg_flag'].mean()*100:.2f}%)")
    
    return df_raw_1min, df_1min


def plot_before_after_cleaning(df_raw_1min: pd.DataFrame, df_clean_1min: pd.DataFrame):
    """Generates Requirement 1c plots: Aggregate + 2 Appliances before and after cleaning."""
    print("\n" + "="*55)
    print("3. GENERATING DIAGNOSTIC PLOTS")
    print("="*55)
    
    channels_to_plot = [
        ("Aggregate", "Whole-Home Aggregate"),
        ("Appliance5", "Appliance 5: Washing Machine (Beko WMC6140)"),
        ("Appliance1", "Appliance 1: Fridge (Hotpoint RLA50P)"),
    ]
    
    # Plot 1: Full Time-Series Overview (Before vs After)
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    for row_idx, (col, label) in enumerate(channels_to_plot):
        axes[row_idx, 0].plot(df_raw_1min.index, df_raw_1min[col], color="#d95f02", lw=0.5, alpha=0.85)
        axes[row_idx, 0].set_title(f"BEFORE Cleaning: {label}", fontsize=11, fontweight="bold")
        axes[row_idx, 0].set_ylabel("Power (Watts)")
        axes[row_idx, 0].grid(True, alpha=0.3)
        
        axes[row_idx, 1].plot(df_clean_1min.index, df_clean_1min[col], color="#1b9e77", lw=0.5, alpha=0.85)
        axes[row_idx, 1].set_title(f"AFTER Cleaning: {label}", fontsize=11, fontweight="bold")
        axes[row_idx, 1].set_ylabel("Power (Watts)")
        axes[row_idx, 1].grid(True, alpha=0.3)
        
    plt.suptitle("House 1: Raw vs. Cleaned 1-Minute Channels (Full Study Period)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path1 = os.path.join(FIG_DIR, "sec1_house1_before_after_overview.png")
    plt.savefig(fig_path1, dpi=200)
    plt.close()
    print(f"Saved full overview plot : {fig_path1}")

    # Plot 2: Zoomed 24-Hour Window Around an Extreme Spike / Anomaly
    # Find where raw and clean differ most (e.g., removed spike or interpolated short gap)
    # Plot 2: Zoomed 24-Hour Window Around an Extreme Spike on the Washing Machine
    wm_diff = (df_raw_1min["Appliance5"].fillna(0) - df_clean_1min["Appliance5"].fillna(0)).abs()
    center_ts = wm_diff.idxmax()
    win_start = center_ts - pd.Timedelta(hours=12)
    win_end = center_ts + pd.Timedelta(hours=12)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for idx, (col, label) in enumerate(channels_to_plot):
        raw_slice = df_raw_1min.loc[win_start:win_end, col]
        clean_slice = df_clean_1min.loc[win_start:win_end, col]
        
        axes[idx].plot(raw_slice.index, raw_slice, label="Raw 1-Min (Before Cleaning)", color="#e41a1c", lw=1.5, ls="--", alpha=0.75)
        axes[idx].plot(clean_slice.index, clean_slice, label="Cleaned 1-Min (After Cleaning)", color="#377eb8", lw=1.2)
        axes[idx].set_title(f"Zoomed Comparison: {label}", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("Power (W)")
        axes[idx].legend(loc="upper right")
        axes[idx].grid(True, alpha=0.3)
        
    plt.suptitle(f"House 1 Zoomed Cleaning Inspection ({win_start.strftime('%Y-%m-%d %H:%M')} to {win_end.strftime('%Y-%m-%d %H:%M')})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path2 = os.path.join(FIG_DIR, "sec1_house1_zoomed_cleaning.png")
    plt.savefig(fig_path2, dpi=200)
    plt.close()
    print(f"Saved zoomed plot        : {fig_path2}")
    # Zoom specifically into a day where the Washing Machine (Appliance5) had a raw spike removed.
    wm_diff = (df_raw_1min["Appliance5"].fillna(0) - df_clean_1min["Appliance5"].fillna(0)).abs()
    center_ts = wm_diff.idxmax()


if __name__ == "__main__":
    df_raw = load_raw_house1(RAW_DATA_DIR)
    audit_report = inspect_raw_data(df_raw)
    df_raw_1min, df_clean_1min = clean_and_resample_house1(df_raw)
    plot_before_after_cleaning(df_raw_1min, df_clean_1min)
    
    out_csv = os.path.join(OUTPUT_DIR, "house1_cleaned_1min.csv")
    df_clean_1min.to_csv(out_csv)
    print(f"\nStep 1 Complete! Cleaned dataset saved to: {out_csv}")
