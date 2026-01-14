import os, json, joblib, yaml
import numpy as np, pandas as pd
from typing import Tuple, Dict, List

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_locked_indices(artifacts_dir: str) -> Dict[str, np.ndarray]:
    splits_dir = os.path.join(artifacts_dir, "splits")
    train_idx = pd.read_csv(os.path.join(splits_dir, "train_idx.csv"), header=None)[0].to_numpy()
    val_idx   = pd.read_csv(os.path.join(splits_dir, "val_idx.csv"),   header=None)[0].to_numpy()
    test_idx  = pd.read_csv(os.path.join(splits_dir, "test_idx.csv"),  header=None)[0].to_numpy()
    return {"train": train_idx, "val": val_idx, "test": test_idx}

def load_scaler_bundle(path: str):
    bundle = joblib.load(path)
    return bundle["scaler"], bundle["features"]

def add_basic_time_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["hour"]  = ts.dt.hour
    df["dow"]   = ts.dt.dayofweek
    df["month"] = ts.dt.month
    return df

def windowize(X: np.ndarray, y: np.ndarray, T_in: int, T_out: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    N = len(y)
    winsX, winsY = [], []
    end = N - (T_in + T_out) + 1
    for start in range(0, max(end, 0), stride):
        i, j, k = start, start + T_in, start + T_in + T_out
        winsX.append(X[i:j, :]); winsY.append(y[j:k])
    if not winsX:
        return np.empty((0, T_in, X.shape[1])), np.empty((0, T_out))
    return np.stack(winsX, 0), np.stack(winsY, 0)

# --- replace in src/features_lstm.py ---

def build_split_windows(
    df: pd.DataFrame,
    ts_col: str,
    target_col: str,
    features: List[str],
    scaler,
    idx: np.ndarray,
    T_in: int,
    T_out: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean per-column (avoids pandas assignment shape errors), scale with locked scaler,
    and windowize.
    """
    # Keep only rows in this split, preserving order
    sub = df.iloc[idx].reset_index(drop=True).copy()

    # Dedup list of columns to clean (target may also appear in features)
    use_cols = list(dict.fromkeys(list(features) + [target_col]))

    # Per-column cleaning to avoid DataFrame assignment mismatch
    for c in use_cols:
        # Coerce to numeric -> non-numeric becomes NaN
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
        # Replace inf with NaN
        sub[c] = sub[c].replace([np.inf, -np.inf], np.nan)
        # Interpolate forward/backward to fill small gaps
        sub[c] = sub[c].interpolate(method="linear", limit_direction="both")

    # Final drop rows that still have NaN in any required column
    sub = sub.dropna(subset=use_cols)

    # Need enough contiguous rows to form at least one window
    if len(sub) < (T_in + T_out):
        return np.empty((0, T_in, len(features))), np.empty((0, T_out))

    # Scale features using TRAIN-fitted scaler
    X = sub[features].to_numpy(dtype=float)
    Xs = scaler.transform(X)

    # Target vector (unscaled) — predict true power
    y = sub[target_col].to_numpy(dtype=float)

    Xw, Yw = windowize(Xs, y, T_in=T_in, T_out=T_out, stride=stride)
    return Xw, Yw

def save_npz(path: str, **arrays):
    import os, numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def make_lstm_datasets(
    config_path: str,
    T_in: int = 168,
    T_out: int = 24,
    stride: int = 1,
    use_time_features: bool = False,
    out_name: str = "lstm_dataset",
) -> Dict[str, str]:
    """
    End-to-end:
      - load config, CSV, scaler, indices
      - (optional) add calendar features
      - clean NaNs
      - build windows for train/val/test
      - save npz files under artifacts/
    """
    cfg = load_config(config_path)
    csv_path   = cfg["data"]["csv_path"]
    ts_col     = cfg["data"]["timestamp_col"]
    target_col = cfg["data"]["target_col"]
    scaler_path = cfg["scaler"]["save_path"]
    artifacts_dir = os.path.dirname(scaler_path)

    # Load data
    df = pd.read_csv(csv_path)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Optional calendar features
    if use_time_features:
        df = add_basic_time_features(df, ts_col)

    # Load locked scaler + features, locked indices
    scaler, base_features = load_scaler_bundle(scaler_path)

    # If time features requested but scaler was trained without them, re-fit on TRAIN only
    features = base_features.copy()
    added = [c for c in ["hour","dow","month"] if use_time_features and c in df.columns and c not in features]
    if added:
        features = features + added
        idx = read_locked_indices(artifacts_dir)
        from sklearn.preprocessing import StandardScaler
        Xtrain = df.iloc[idx["train"]][features].replace([np.inf, -np.inf], np.nan)
        Xtrain = Xtrain.interpolate(method="linear", limit_direction="both").dropna()
        scaler = StandardScaler().fit(Xtrain.to_numpy(dtype=float))
        new_scaler_path = os.path.join(artifacts_dir, f"scaler_{out_name}.pkl")
        joblib.dump({"scaler": scaler, "features": features}, new_scaler_path)
        scaler_path = new_scaler_path

    idx = read_locked_indices(artifacts_dir)

    # Build windows per split
    Xtr, Ytr = build_split_windows(df, ts_col, target_col, features, scaler, idx["train"], T_in, T_out, stride)
    Xva, Yva = build_split_windows(df, ts_col, target_col, features, scaler, idx["val"],   T_in, T_out, stride)
    Xte, Yte = build_split_windows(df, ts_col, target_col, features, scaler, idx["test"],  T_in, T_out, stride)

    # Sanity check: ensure we actually have windows
    if Xtr.size == 0:
        raise RuntimeError(f"No training windows created. Check splits and T_in/T_out (need >= {T_in+T_out} rows after cleaning).")
    if Xva.size == 0:
        print("[warn] No validation windows; metrics will be NaN. Consider expanding val range.")
    if Xte.size == 0:
        print("[warn] No test windows; test metrics will be NaN. Consider expanding test range.")

    # Save datasets
    out_dir = os.path.join(artifacts_dir, "lstm")
    os.makedirs(out_dir, exist_ok=True)
    train_npz = os.path.join(out_dir, f"{out_name}_train.npz")
    val_npz   = os.path.join(out_dir, f"{out_name}_val.npz")
    test_npz  = os.path.join(out_dir, f"{out_name}_test.npz")
    meta_json = os.path.join(out_dir, f"{out_name}_meta.json")

    save_npz(train_npz, X=Xtr, Y=Ytr)
    save_npz(val_npz,   X=Xva, Y=Yva)
    save_npz(test_npz,  X=Xte, Y=Yte)

    meta = {
        "config_path": config_path,
        "scaler_path": scaler_path,
        "features": features,
        "T_in": T_in,
        "T_out": T_out,
        "stride": stride,
        "use_time_features": use_time_features,
        "shapes": {
            "train": [int(x) for x in Xtr.shape],
            "val":   [int(x) for x in Xva.shape],
            "test":  [int(x) for x in Xte.shape],
        },
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    return {"train": train_npz, "val": val_npz, "test": test_npz, "meta": meta_json}

