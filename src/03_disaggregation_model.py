import os
import gc
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# CONFIGURATION & PATHS
CLEAN_DATA_DIR = "./data/CLEAN_REFIT_081116"
OUTPUT_DIR = "./outputs"
FIG_DIR = "./outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

SELECTED_HOUSES = {
    1: {"wm_col": "Appliance5", "brand": "Beko WMC6140"},
    2: {"wm_col": "Appliance2", "brand": "LG F1289TD"},
    5: {"wm_col": "Appliance3", "brand": "AEG L99695HWD"},
    6: {"wm_col": "Appliance2", "brand": "Bosch Classixx 1200"},
    7: {"wm_col": "Appliance5", "brand": "Bosch Unknown"},
}

ON_THRESHOLD_W = 20.0   # Active washing machine threshold (Watts)
SEQ_LEN = 61            # 61-minute centered window [t-30 ... t+30] for Seq2Point
HALF_W = SEQ_LEN // 2   # 30 minutes


# PYTORCH SEQ2POINT 1D-CNN ARCHITECTURE (Zhang et al., 2018)
class Seq2PointCNN(nn.Module):
    """
    1D-Convolutional Seq2Point Network adapted for 1-minute resolution NILM.
    Input : (Batch, Channels=1 [Net Aggregate], Window=61 mins)
    Output: Midpoint Washing Machine Power y_t (Watts)
    """
    def __init__(self, in_channels: int = 1, seq_len: int = 61):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 61 -> 30
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feats = self.conv_block(x)
        return self.head(feats).squeeze(-1)


def find_clean_house_file(clean_dir: str, house_num: int) -> str:
    all_csvs = glob.glob(os.path.join(clean_dir, "**", "*.csv"), recursive=True)
    for f in all_csvs:
        fname = os.path.basename(f)
        if re.search(rf"(house[_\s-]*{house_num}|h{house_num})(?!\d)", fname, re.IGNORECASE):
            return f
    raise FileNotFoundError(f"Could not find CLEAN CSV for House {house_num}")


def load_and_prepare_house(house_num: int, meta: dict) -> pd.DataFrame:
    """Loads cleaned REFIT house using memory-efficient float32/int64 dtypes and resamples to 1-min."""
    fpath = find_clean_house_file(CLEAN_DATA_DIR, house_num)
    wm_col = meta["wm_col"]
    sample = pd.read_csv(fpath, nrows=5)
    cols = list(sample.columns)
    unix_col, agg_col = cols[1], cols[2]
    issues_col = cols[-1] if "issue" in cols[-1].lower() else None

    usecols = [unix_col, agg_col, wm_col] + ([issues_col] if issues_col else [])
    dtype_map = {agg_col: "float32", wm_col: "float32"}
    if issues_col:
        dtype_map[issues_col] = "float32"

    # Read in chunks to prevent C-parser out-of-memory errors on large CSVs
    chunks = []
    for chunk in pd.read_csv(fpath, usecols=usecols, dtype=dtype_map, chunksize=1_000_000, low_memory=True):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    rename_map = {unix_col: "Unix", agg_col: "Aggregate", wm_col: "WashingMachine"}
    if issues_col:
        rename_map[issues_col] = "Issues"
    df = df.rename(columns=rename_map)

    df["Unix"] = pd.to_numeric(df["Unix"], errors="coerce")
    df = df.dropna(subset=["Unix"])
    df["timestamp"] = pd.to_datetime(df["Unix"], unit="s", utc=True)
    df = df.drop(columns=["Unix"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")

    # Resample to 1-minute intervals
    agg_dict = {"Aggregate": "mean", "WashingMachine": "mean"}
    if "Issues" in df.columns:
        agg_dict["Issues"] = "max"
    df_1m = df.resample("1min").agg(agg_dict)
    del df
    gc.collect()

    # Interpolate short gaps <= 5 mins
    df_1m[["Aggregate", "WashingMachine"]] = df_1m[["Aggregate", "WashingMachine"]].interpolate(
        method="time", limit=5, limit_direction="forward"
    )

    # Filter out corrupted rows where Issues == 1 or WashingMachine > Aggregate + 50W
    if "Issues" in df_1m.columns:
        df_1m.loc[df_1m["Issues"] == 1, ["Aggregate", "WashingMachine"]] = np.nan
    df_1m.loc[df_1m["WashingMachine"] > df_1m["Aggregate"] + 50, ["Aggregate", "WashingMachine"]] = np.nan

    # Clip physical bounds & keep as float32
    df_1m["Aggregate"] = df_1m["Aggregate"].clip(lower=0.0, upper=20000.0).astype("float32")
    df_1m["WashingMachine"] = df_1m["WashingMachine"].clip(lower=0.0, upper=3500.0).astype("float32")
    df_1m["house"] = np.int16(house_num)
    return df_1m


def engineer_nilm_features(df_1m: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Constructs compact, memory-efficient float32 features for LightGBM and stores
    a 61-minute window pointer for zero-copy Seq2Point 1D-CNN extraction.
    """
    p = df_1m["Aggregate"].astype("float32")

    # Estimate local household baseload (rolling 6-hour 10th percentile)
    baseload = p.rolling(window=360, min_periods=1, center=True).quantile(0.10).astype("float32")
    p_net = (p - baseload).clip(lower=0.0).astype("float32")

    feats = pd.DataFrame(index=df_1m.index)
    feats["agg_raw"] = p
    feats["agg_net"] = p_net
    feats["baseload"] = baseload

    # Centered rolling statistics across multiple temporal scales (5m, 15m, 31m, 61m)
    for w in [5, 15, 31, 61]:
        r = p_net.rolling(window=w, min_periods=max(1, w // 3), center=True)
        feats[f"c_mean_{w}m"] = r.mean().astype("float32")
        feats[f"c_std_{w}m"] = r.std().fillna(0.0).astype("float32")
        feats[f"c_max_{w}m"] = r.max().astype("float32")
        feats[f"c_min_{w}m"] = r.min().astype("float32")
        feats[f"c_range_{w}m"] = (feats[f"c_max_{w}m"] - feats[f"c_min_{w}m"]).astype("float32")

    # Past vs. Future half-window asymmetry (catches low-power spin tail after a 2kW heating block)
    for w in [15, 30, 45]:
        past_r = p_net.rolling(window=w, min_periods=1)
        fut_mean = p_net.iloc[::-1].rolling(window=w, min_periods=1).mean().iloc[::-1].astype("float32")
        fut_max = p_net.iloc[::-1].rolling(window=w, min_periods=1).max().iloc[::-1].astype("float32")
        feats[f"past_mean_{w}m"] = past_r.mean().astype("float32")
        feats[f"past_max_{w}m"] = past_r.max().astype("float32")
        feats[f"fut_mean_{w}m"] = fut_mean
        feats[f"fut_max_{w}m"] = fut_max
        feats[f"asym_mean_{w}m"] = (feats[f"past_mean_{w}m"] - fut_mean).astype("float32")

    # High-heat duration counters ([1650W, 3200W] band distinguishes 2-min kettle from 20-min wash heater)
    is_heating_band = ((p_net >= 1650.0) & (p_net <= 3200.0)).astype("float32")
    for w in [15, 31, 61]:
        feats[f"heat_band_mins_{w}m"] = is_heating_band.rolling(window=w, min_periods=1, center=True).sum().astype("float32")
    feats["past_45m_heat_mins"] = is_heating_band.rolling(window=45, min_periods=1).sum().astype("float32")

    # Key sparse lags for LightGBM (avoids storing all 61 columns in DataFrame.)
    for lag in [-30, -20, -15, -10, -5, -2, -1, 1, 2, 5, 10, 15, 20, 30]:
        feats[f"lag_{lag}m"] = p_net.shift(lag).astype("float32")

    feats["diff_1m"] = p_net.diff(1).abs().astype("float32")
    feats["diff_3m"] = p_net.diff(3).abs().astype("float32")

    # Cyclical temporal features
    hour = (df_1m.index.hour + df_1m.index.minute / 60.0).astype("float32")
    dow = df_1m.index.dayofweek.astype("float32")
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype("float32")
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype("float32")
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    feats["hour_int"] = df_1m.index.hour.astype("int8")
    feats["dow_int"] = df_1m.index.dayofweek.astype("int8")

    feature_cols = [c for c in feats.columns if c not in ["hour_int", "dow_int"]]
    feats["WashingMachine"] = df_1m["WashingMachine"].astype("float32")
    feats["house"] = df_1m["house"].astype("int16")

    # Drop rows where target or 30-min window edges are NaN
    feats = feats.dropna(subset=["WashingMachine", "agg_raw", "lag_-30m", "lag_30m"])
    gc.collect()
    return feats, feature_cols


def build_seq2point_windows(df: pd.DataFrame, idx: np.ndarray = None) -> np.ndarray:
    """
    Extracts 61-minute centered windows [t-30 ... t+30] from `agg_net` for the 1D-CNN.

    FIX: windows are built PER HOUSE. The previous version padded/windowed over the
    concatenated multi-house frame, so rows near a house boundary received windows
    that mixed the end of one house with the start of the next.

    The frame is assumed to be grouped contiguously by house (true for how the
    splits are built). Only the requested rows `idx` are materialised
    (all rows if idx is None).
    """
    arr = df["agg_net"].values.astype("float32")
    houses = df["house"].values

    # contiguous house blocks: [(start, end), ...]
    change = np.flatnonzero(np.diff(houses) != 0) + 1
    bounds = np.concatenate([[0], change, [len(arr)]])

    padded_blocks = []
    starts = np.empty(len(arr), dtype=np.int64)
    offset = 0
    for b in range(len(bounds) - 1):
        s, e = bounds[b], bounds[b + 1]
        block = np.pad(arr[s:e], (HALF_W, HALF_W), mode="edge")
        padded_blocks.append(block)
        starts[s:e] = offset + np.arange(e - s)   # window start for each row
        offset += len(block)

    big = np.concatenate(padded_blocks)
    view = sliding_window_view(big, window_shape=SEQ_LEN)  # zero-copy view
    sel = starts if idx is None else starts[idx]
    return view[sel]


def train_pytorch_seq2point(X_tr_seq: np.ndarray, y_tr: np.ndarray,
                            X_va_seq: np.ndarray, y_va: np.ndarray,
                            X_te_seq: np.ndarray, epochs: int = 10, batch_size: int = 512) -> np.ndarray:
    """Trains the PyTorch 1D-CNN Seq2Point model with L1 (MAE) loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2PointCNN(in_channels=1, seq_len=SEQ_LEN).to(device)

    X_tr_t = torch.tensor(np.array(X_tr_seq, copy=True) / 2000.0, dtype=torch.float32).unsqueeze(1)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(np.array(X_va_seq, copy=True) / 2000.0, dtype=torch.float32).unsqueeze(1)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    best_weights = None

    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_preds = []
            for i in range(0, len(X_va_t), 2048):
                batch_x = X_va_t[i:i+2048].to(device)
                va_preds.append(model(batch_x).cpu())
            va_preds = torch.cat(va_preds)
            val_loss = criterion(va_preds, y_va_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Predict on test set in memory-safe batches
    model.eval()
    te_preds = []
    with torch.no_grad():
        for i in range(0, len(X_te_seq), 4096):
            batch_np = np.array(X_te_seq[i:i+4096], copy=True) / 2000.0
            batch_x = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1).to(device)
            te_preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(te_preds)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split_name: str) -> dict:
    """Computes comprehensive NILM metrics (MAE, RMSE, Active-MAE, SAE, NDE, F1, Precision, Recall)."""
    y_pred = np.clip(y_pred, 0.0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    on_mask = y_true >= ON_THRESHOLD_W
    active_mae = mean_absolute_error(y_true[on_mask], y_pred[on_mask]) if on_mask.sum() > 0 else np.nan

    true_kwh = np.sum(y_true, dtype=np.float64) / 60000.0
    pred_kwh = np.sum(y_pred, dtype=np.float64) / 60000.0
    sae_rel_pct = abs(pred_kwh - true_kwh) / max(true_kwh, 1e-6) * 100.0
    nde = np.sqrt(np.sum((y_true - y_pred) ** 2, dtype=np.float64) / max(np.sum(y_true ** 2, dtype=np.float64), 1e-6))

    true_state = (y_true >= ON_THRESHOLD_W).astype(int)
    pred_state = (y_pred >= ON_THRESHOLD_W).astype(int)
    prec = precision_score(true_state, pred_state, zero_division=0)
    rec = recall_score(true_state, pred_state, zero_division=0)
    f1 = f1_score(true_state, pred_state, zero_division=0)

    return {
        "Split": split_name,
        "Model": model_name,
        "Overall_MAE_W": round(float(mae), 2),
        "RMSE_W": round(float(rmse), 2),
        "Active_ON_MAE_W": round(float(active_mae), 2),
        "True_Energy_kWh": round(float(true_kwh), 2),
        "Pred_Energy_kWh": round(float(pred_kwh), 2),
        "Rel_Energy_Err_SAE_%": round(float(sae_rel_pct), 2),
        "NDE": round(float(nde), 3),
        "ON_Precision": round(float(prec), 3),
        "ON_Recall": round(float(rec), 3),
        "ON_F1_Score": round(float(f1), 3),
    }


def train_and_eval_suite(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                         feature_cols: list, split_label: str) -> tuple[list, dict]:
    """Trains and evaluates Baselines, Two-Stage Hurdle LightGBM, PyTorch Seq2Point 1D-CNN, and Hybrid Ensemble."""
    print(f"\n--- Evaluating Split: {split_label} ---")
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Test rows: {len(test_df):,}")

    y_train = train_df["WashingMachine"].values
    y_val = val_df["WashingMachine"].values
    y_test = test_df["WashingMachine"].values

    results = []
    preds_dict = {}

    # 1. BASELINE 1: Always-Off (0 Watts)
    pred_zero = np.zeros_like(y_test)
    results.append(evaluate_predictions(y_test, pred_zero, "Baseline 1: Always-Off (0W)", split_label))
    preds_dict["Always-Off"] = pred_zero

    # 2. BASELINE 2: Time-of-Day & Day-of-Week Historical Profile
    profile_map = train_df.groupby(["dow_int", "hour_int"])["WashingMachine"].mean().to_dict()
    global_mean = float(y_train.mean())
    pred_profile = np.array([
        profile_map.get((d, h), global_mean)
        for d, h in zip(test_df["dow_int"].values, test_df["hour_int"].values)
    ], dtype="float32")
    results.append(evaluate_predictions(y_test, pred_profile, "Baseline 2: Time-of-Day Profile", split_label))
    preds_dict["Profile_Baseline"] = pred_profile

    # 3. MODEL 1: Two-Stage Hurdle LightGBM
    print("Training Model 1: Two-Stage Hurdle LightGBM...")
    X_tr = train_df[feature_cols].values
    X_va = val_df[feature_cols].values
    X_te = test_df[feature_cols].values

    y_tr_bin = (y_train >= ON_THRESHOLD_W).astype(int)
    y_va_bin = (y_val >= ON_THRESHOLD_W).astype(int)

    pos_idx = np.where(y_tr_bin == 1)[0]
    neg_idx = np.where(y_tr_bin == 0)[0]
    rng = np.random.default_rng(42)
    keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 10), replace=False)
    bal_idx = np.sort(np.concatenate([pos_idx, keep_neg]))

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample_for_bin=100000,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr[bal_idx], y_tr_bin[bal_idx],
        eval_set=[(X_va, y_va_bin)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    val_probs = clf.predict_proba(X_va)[:, 1]
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.25, 0.85, 25):
        score = f1_score(y_va_bin, (val_probs >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, th
    print(f"  Optimal Stage-1 probability threshold on Val: {best_th:.2f} (Val F1: {best_f1:.3f})")

    reg_neg_keep = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) // 2), replace=False)
    reg_idx = np.sort(np.concatenate([pos_idx, reg_neg_keep]))

    reg = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    val_active_mask = y_val >= ON_THRESHOLD_W
    reg.fit(
        X_tr[reg_idx], y_train[reg_idx],
        eval_set=[(X_va[val_active_mask], y_val[val_active_mask])],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    test_probs = clf.predict_proba(X_te)[:, 1]
    test_probs_smooth = pd.Series(test_probs).rolling(3, min_periods=1, center=True).mean().values
    test_on_gate = (test_probs_smooth >= best_th).astype("float32")

    raw_reg_pred = np.clip(reg.predict(X_te), 0.0, None).astype("float32")
    pred_lgbm = np.minimum(test_on_gate * raw_reg_pred, test_df["agg_raw"].values)

    results.append(evaluate_predictions(y_test, pred_lgbm, "Model 1: Two-Stage Hurdle LightGBM", split_label))
    preds_dict["TwoStage_LightGBM"] = pred_lgbm
    preds_dict["lgbm_clf"] = clf
    preds_dict["lgbm_reg"] = reg

    # 4. MODEL 2: PyTorch 1D-CNN Seq2Point (Zhang et al., 2018)
    print("Training Model 2: PyTorch 1D-CNN Seq2Point...")
    cnn_neg_keep = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 2), replace=False)
    cnn_idx = np.sort(np.concatenate([pos_idx, cnn_neg_keep]))

    X_tr_seq = build_seq2point_windows(train_df, cnn_idx)
    y_tr_seq = y_train[cnn_idx]
    val_active_idx = np.flatnonzero(val_active_mask)
    X_va_seq = build_seq2point_windows(val_df, val_active_idx)
    y_va_seq = y_val[val_active_idx]
    te_windows = build_seq2point_windows(test_df)

    raw_cnn_pred = train_pytorch_seq2point(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, te_windows, epochs=10, batch_size=512)
    raw_cnn_pred = np.clip(raw_cnn_pred, 0.0, None).astype("float32")
    pred_seq2point = np.minimum(raw_cnn_pred, test_df["agg_raw"].values)
    pred_seq2point[pred_seq2point < ON_THRESHOLD_W] = 0.0

    results.append(evaluate_predictions(y_test, pred_seq2point, "Model 2: PyTorch Seq2Point 1D-CNN", split_label))
    preds_dict["Seq2Point_CNN"] = pred_seq2point

    # 5. MODEL 3: State-Gated Hybrid Ensemble (LightGBM Gate + 50/50 LightGBM & Seq2Point 1D-CNN)
    pred_ensemble = np.minimum(test_on_gate * (0.5 * raw_reg_pred + 0.5 * raw_cnn_pred), test_df["agg_raw"].values)
    results.append(evaluate_predictions(y_test, pred_ensemble, "Model 3: Hybrid Gate + Seq2Point-LGBM Ensemble", split_label))
    preds_dict["Hybrid_Ensemble"] = pred_ensemble

    gc.collect()
    return results, preds_dict


def pick_failure_case(test_out: pd.DataFrame, mask: np.ndarray, fallback_pos: int):
    """
    Returns (timestamp, house) for a representative failure row.
    Uses POSITIONAL access so the house always belongs to the chosen row
    (test_out's index has duplicate timestamps across houses, so label lookups are ambiguous).
    """
    pos = np.flatnonzero(mask)
    p = int(pos[len(pos) // 2]) if len(pos) > 0 else int(fallback_pos)
    return test_out.index[p], int(test_out["house"].iloc[p])


def plot_evaluation_figures(test_df: pd.DataFrame, preds_temporal: dict, feature_cols: list):
    """Generates Requirement 3 Actual-vs-Predicted plots, Feature Importance, and Failure Case analysis."""
    print("\n" + "="*55)
    print("GENERATING SECTION 3 EVALUATION & FAILURE PLOTS")
    print("="*55)

    test_out = test_df[["agg_raw", "WashingMachine", "house"]].copy()
    test_out["Pred_LGBM"] = preds_temporal["TwoStage_LightGBM"]
    test_out["Pred_Seq2Point"] = preds_temporal["Seq2Point_CNN"]
    test_out["Pred_Ensemble"] = preds_temporal["Hybrid_Ensemble"]

    # PLOT 1: Actual vs. Predicted Disaggregation on Evaluation Data
    h1_test = test_out[test_out["house"] == 1].copy()
    rolling_true = h1_test["WashingMachine"].rolling(360).sum()
    best_end = rolling_true.idxmax()
    win_start = best_end - pd.Timedelta(hours=8)
    win_end = best_end + pd.Timedelta(hours=4)
    sub = h1_test.loc[win_start:win_end]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})

    axes[0].plot(sub.index, sub["agg_raw"], color="#666666", lw=1.2, alpha=0.85, label="Whole-Home Aggregate Signal (Model Input)")
    axes[0].fill_between(sub.index, 0, sub["WashingMachine"], color="#2b83ba", alpha=0.45, label="Ground Truth Washing Machine")
    axes[0].set_title("A. Model Input: Whole-Home Aggregate Power vs. Ground Truth Washing Machine (Test Set — House 1)",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Power (Watts)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(sub.index, 0, sub["WashingMachine"], color="#cccccc", alpha=0.6, label="Actual Washing Machine (Ground Truth)")
    axes[1].plot(sub.index, sub["WashingMachine"], color="#333333", lw=1.4)
    axes[1].plot(sub.index, sub["Pred_LGBM"], color="#d7191c", lw=2.0, label="Model 1: Two-Stage Hurdle LightGBM")
    axes[1].plot(sub.index, sub["Pred_Seq2Point"], color="#2b83ba", lw=1.5, ls="--", alpha=0.85, label="Model 2: PyTorch Seq2Point 1D-CNN")
    axes[1].plot(sub.index, sub["Pred_Ensemble"], color="#1b9e77", lw=1.8, ls="-.", label="Model 3: Hybrid Ensemble")
    axes[1].set_title("B. Disaggregation Output: Actual vs. Predicted Washing Machine Load (1-Minute Resolution)",
                      fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Washing Machine Power (W)")
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Section 3: Actual vs. Predicted Appliance Disaggregation on Held-Out Test Data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f1_path = os.path.join(FIG_DIR, "sec3_actual_vs_predicted_test.png")
    plt.savefig(f1_path, dpi=200)
    plt.close()
    print(f"Saved Actual vs Predicted plot : {f1_path}")

    # PLOT 2: Failure Modes Analysis
    wm = test_out["WashingMachine"].values
    pred = test_out["Pred_LGBM"].values
    err = pred - wm

    fp_ts, fp_house = pick_failure_case(test_out, (wm < 10) & (pred > 1500), int(np.argmax(err)))
    fn_ts, fn_house = pick_failure_case(test_out, (wm > 400) & (pred < 50), int(np.argmin(err)))

    # Filter by house FIRST: within a single house the index is unique and sorted,
    # so label-based time slicing is safe.
    fp_slice = test_out[test_out["house"] == fp_house].loc[
        fp_ts - pd.Timedelta(minutes=90): fp_ts + pd.Timedelta(minutes=90)
    ]
    fn_slice = test_out[test_out["house"] == fn_house].loc[
        fn_ts - pd.Timedelta(minutes=90): fn_ts + pd.Timedelta(minutes=90)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    axes[0].plot(fp_slice.index, fp_slice["agg_raw"], color="#888888", lw=1.1, ls=":", label="Aggregate Power")
    axes[0].plot(fp_slice.index, fp_slice["WashingMachine"], color="#2ca25f", lw=2.0, label="Actual Washing Machine")
    axes[0].plot(fp_slice.index, fp_slice["Pred_LGBM"], color="#e41a1c", lw=1.8, ls="--", label="Predicted (LightGBM)")
    axes[0].set_title(f"Failure Mode 1: Confuser Appliance False Positive (House {fp_house})\n"
                      f"Overlapping ~2kW Heating Signature (e.g., Dishwasher / Tumble Dryer)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Power (Watts)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fn_slice.index, fn_slice["agg_raw"], color="#888888", lw=1.1, ls=":", label="Aggregate Power")
    axes[1].plot(fn_slice.index, fn_slice["WashingMachine"], color="#2ca25f", lw=2.0, label="Actual Washing Machine")
    axes[1].plot(fn_slice.index, fn_slice["Pred_LGBM"], color="#e41a1c", lw=1.8, ls="--", label="Predicted (LightGBM)")
    axes[1].set_title(f"Failure Mode 2: Masked Low-Power Spin / Cold-Wash Cycle (House {fn_house})\n"
                      f"Low-Wattage Drum Activity Masked by High Household Baseload Noise", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Power (Watts)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Section 3: Diagnostic Analysis of Model Failure Cases on Held-Out Test Data", fontsize=13, fontweight="bold")
    plt.tight_layout()
    f2_path = os.path.join(FIG_DIR, "sec3_failure_modes_analysis.png")
    plt.savefig(f2_path, dpi=200)
    plt.close()
    print(f"Saved Failure Modes plot       : {f2_path}")

    # PLOT 3: Feature Importance
    clf = preds_temporal["lgbm_clf"]
    reg = preds_temporal["lgbm_reg"]
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Classifier_Gain": clf.booster_.feature_importance(importance_type="gain"),
        "Regressor_Gain": reg.booster_.feature_importance(importance_type="gain"),
    })
    imp_df["Classifier_Pct"] = imp_df["Classifier_Gain"] / imp_df["Classifier_Gain"].sum() * 100
    imp_df["Regressor_Pct"] = imp_df["Regressor_Gain"] / imp_df["Regressor_Gain"].sum() * 100
    top_clf = imp_df.sort_values("Classifier_Pct", ascending=True).tail(12)
    top_reg = imp_df.sort_values("Regressor_Pct", ascending=True).tail(12)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].barh(top_clf["Feature"], top_clf["Classifier_Pct"], color="#377eb8")
    axes[0].set_title("Stage 1 (ON/OFF State Classifier) — Top 12 Features (% Gain)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("% of Total Information Gain")
    axes[0].grid(True, alpha=0.3)

    axes[1].barh(top_reg["Feature"], top_reg["Regressor_Pct"], color="#ff7f00")
    axes[1].set_title("Stage 2 (Active Load Regressor) — Top 12 Features (% Gain)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("% of Total Information Gain")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Section 3: Two-Stage LightGBM Feature Importance (Physical Interpretability)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    f3_path = os.path.join(FIG_DIR, "sec3_feature_importance.png")
    plt.savefig(f3_path, dpi=200)
    plt.close()
    print(f"Saved Feature Importance plot  : {f3_path}")


if __name__ == "__main__":
    print("="*60)
    print("SECTION 3: APPLIANCE DISAGGREGATION MODELLING & VALIDATION")
    print("="*60)

    house_dfs = {}
    feature_cols = None
    for h_num, meta in SELECTED_HOUSES.items():
        print(f"Preparing features for House {h_num} ({meta['brand']})...")
        raw_1m = load_and_prepare_house(h_num, meta)
        feats_df, feature_cols = engineer_nilm_features(raw_1m)
        del raw_1m
        gc.collect()
        house_dfs[h_num] = feats_df

    # SPLIT 1: CHRONOLOGICAL TEMPORAL SPLIT (70% Train / 15% Val / 15% Test per house)
    tr_list, va_list, te_list = [], [], []
    BUFFER_MINS = 60
    for h_num, fdf in house_dfs.items():
        n = len(fdf)
        i_tr = int(n * 0.70)
        i_va = int(n * 0.85)
        tr_list.append(fdf.iloc[: i_tr - BUFFER_MINS])
        va_list.append(fdf.iloc[i_tr + BUFFER_MINS : i_va - BUFFER_MINS])
        te_list.append(fdf.iloc[i_va + BUFFER_MINS :])

    train_temp = pd.concat(tr_list)
    val_temp = pd.concat(va_list)
    test_temp = pd.concat(te_list)
    del tr_list, va_list, te_list
    gc.collect()

    res_temporal, preds_temporal = train_and_eval_suite(
        train_temp, val_temp, test_temp, feature_cols, "Temporal Split (70/15/15)"
    )

    plot_evaluation_figures(test_temp, preds_temporal, feature_cols)
    del train_temp, val_temp, test_temp, preds_temporal
    gc.collect()

    # SPLIT 2: LEAVE-ONE-HOUSE-OUT SPLIT (Train: Houses 1,2,5 | Val: House 6 | Test: Unseen House 7)
    train_loho = pd.concat([house_dfs[1], house_dfs[2], house_dfs[5]])
    val_loho = house_dfs[6]
    test_loho = house_dfs[7]

    res_loho, preds_loho = train_and_eval_suite(
        train_loho, val_loho, test_loho, feature_cols, "Unseen House Split (Test=House 7)"
    )

    all_metrics_df = pd.DataFrame(res_temporal + res_loho)
    print("\n" + "="*80)
    print("FINAL MODEL BENCHMARK RESULTS TABLE")
    print("="*80)
    print(all_metrics_df.to_string(index=False))

    metrics_path = os.path.join(OUTPUT_DIR, "sec3_model_evaluation_metrics.csv")
    all_metrics_df.to_csv(metrics_path, index=False)
    print(f"\nStep 3 Complete! Metrics saved to {metrics_path}")
