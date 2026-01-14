"""
Usage:
  python scripts/lock_data_and_scaler.py --config configs/config.yaml

This will:
  - Load your CSV
  - Create non-overlapping train/val/test masks by date
  - Save index lists to artifacts/splits/*.csv
  - Fit a scaler on TRAIN-only and save to artifacts/scaler.pkl
"""
import argparse
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import (
    load_config,
    load_frame,
    make_date_masks,
    save_index_lists,
    get_feature_columns,
    fit_and_save_scaler,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path   = cfg["data"]["csv_path"]
    ts_col     = cfg["data"]["timestamp_col"]
    tz         = cfg["data"]["timezone"]
    target_col = cfg["data"]["target_col"]
    feature_cols = cfg["data"].get("feature_cols", [])
    splits_dir = cfg["outputs"]["splits_dir"]

    print(f"[1/5] Loading frame: {csv_path}")
    df = load_frame(csv_path, ts_col, tz)

    print("[2/5] Building split masks from config dates...")
    masks = make_date_masks(df, ts_col, cfg)

    print(f"[3/5] Saving index lists to: {splits_dir}")
    save_index_lists(masks, splits_dir)

    print("[4/5] Resolving feature columns...")
    features = get_feature_columns(df, target_col, feature_cols)
    print(f"Using {len(features)} features. Example: {features[:8]}")

    print("[5/5] Fitting scaler on TRAIN-only and saving...")
    scaler_path = fit_and_save_scaler(df, masks, features, cfg["scaler"])
    print(f"Saved scaler to: {scaler_path}")
    print("Done.")

if __name__ == "__main__":
    main()
