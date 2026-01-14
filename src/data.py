import os
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import yaml

@dataclass
class SplitMasks:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_frame(csv_path: str, ts_col: str, tz: str = None) -> pd.DataFrame:
    ext = os.path.splitext(csv_path)[1].lower()
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use a .csv path in config.")
    if ts_col not in df.columns:
        # try common fallbacks
        candidates = [c for c in df.columns if str(c).lower() in ["ts","timestamp","time","datetime","date"]]
        if len(candidates) == 1:
            df = df.rename(columns={candidates[0]: ts_col})
        else:
            raise ValueError(f"Timestamp column '{ts_col}' not found in {csv_path}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    if tz:
        if df[ts_col].dt.tz is None:
            df[ts_col] = df[ts_col].dt.tz_localize(tz)
        else:
            df[ts_col] = df[ts_col].dt.tz_convert(tz)
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df

def make_date_masks(df: pd.DataFrame, ts_col: str, cfg: dict) -> SplitMasks:
    s = cfg["splits"]
    def rng_mask(start, end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        return (df[ts_col] >= start) & (df[ts_col] < end)
    train_m = rng_mask(s["train_start"], s["train_end"]).to_numpy()
    val_m   = rng_mask(s["val_start"],   s["val_end"]).to_numpy()
    test_m  = rng_mask(s["test_start"],  s["test_end"]).to_numpy()
    if (train_m & val_m).any() or (train_m & test_m).any() or (val_m & test_m).any():
        raise ValueError("Split masks overlap — check your date ranges in config.yaml")
    return SplitMasks(train=train_m, val=val_m, test=test_m)

def save_index_lists(masks: SplitMasks, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for name, mask in [("train", masks.train), ("val", masks.val), ("test", masks.test)]:
        idx = np.where(mask)[0]
        pd.Series(idx).to_csv(os.path.join(out_dir, f"{name}_idx.csv"), index=False, header=False)

def get_feature_columns(df: pd.DataFrame, target: str, feature_cols: List[str]) -> List[str]:
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing: {missing}")
        return feature_cols
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cols = [c for c in num_cols if c != target]
    if not cols:
        raise ValueError("No numeric feature columns found. Provide feature_cols in config.")
    return cols

def fit_and_save_scaler(df: pd.DataFrame, masks: SplitMasks, features: List[str], scaler_cfg: dict):
    X_train = df.loc[masks.train, features].values
    stype = scaler_cfg.get("type", "standard").lower()
    if stype == "standard":
        scaler = StandardScaler()
    elif stype == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler.type must be 'standard' or 'minmax'")
    scaler.fit(X_train)
    out_path = scaler_cfg["save_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"scaler": scaler, "features": features}, out_path)
    return out_path
