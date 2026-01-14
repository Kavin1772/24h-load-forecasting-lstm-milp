# scripts/run_scheduler.py
import os
import sys
import argparse
import json
import numpy as np

import importlib, src.scheduling.heuristics as H
importlib.reload(H)
from src.scheduling.heuristics import (
    load_npz, load_best_model, predict_next24, load_yaml,
    schedule_day, schedule_naive, total_cost_with_appliances,
    schedule_to_csv, save_json
)


# ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scheduling.heuristics import (
    load_npz,
    load_best_model,
    predict_next24,
    load_yaml,
    schedule_day,
    schedule_naive,
    total_cost_with_appliances,
    schedule_to_csv,
    save_json,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="LSTM run tag (e.g., lstm_v4)")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--yaml_meta", required=True, help="Path to tariff_and_appliances.yaml")
    args = ap.parse_args()

    # --- Locate artifacts directory ---
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    artifacts_dir = os.path.join(os.path.dirname(cfg["scaler"]["save_path"]), "lstm")

    best_pt = os.path.join(artifacts_dir, f"{args.tag}_best.pt")
    meta_json = os.path.join(artifacts_dir, f"{args.tag}_t168_h24_meta.json")
    with open(meta_json, "r") as f:
        meta = json.load(f)

    test_npz = os.path.join(artifacts_dir, f"{args.tag}_t{meta['T_in']}_h{meta['T_out']}_test.npz")
    Xte, Yte = load_npz(test_npz)
    X_last = Xte[-1]
    in_features = X_last.shape[-1]

    # --- Predict next 24h base load ---
    model = load_best_model(best_pt, in_features=in_features, T_out=meta["T_out"])
    y_pred = predict_next24(model, X_last)  # base load forecast

    # --- Load tariff & appliance metadata ---
    meta_yaml = load_yaml(args.yaml_meta)
    prices = np.array(meta_yaml["tou"]["prices"], dtype=float)

    # --- Convert predicted load to kW (if in Watts) ---
    if np.nanmedian(y_pred) > 50:  # typical threshold
        y_pred_kw = y_pred / 1000.0
    else:
        y_pred_kw = y_pred

    # --- Build schedules ---
    sched_naive = schedule_naive(prices, meta_yaml)
    sched_opt = schedule_day(prices, meta_yaml)

    # --- Compute KPIs ---
    naive_cost, naive_peak, base_peak = total_cost_with_appliances(y_pred_kw, sched_naive, meta_yaml)
    opt_cost, opt_peak, _ = total_cost_with_appliances(y_pred_kw, sched_opt, meta_yaml)

    stats = {
        "currency": meta_yaml["tou"].get("currency", "USD"),
        "naive_cost": naive_cost,
        "opt_cost": opt_cost,
        "savings_$": naive_cost - opt_cost,
        "base_peak_kw": base_peak,
        "naive_peak_kw": naive_peak,
        "opt_peak_kw": opt_peak,
        "peak_reduction_kw": naive_peak - opt_peak,
    }

    # --- Export results ---
    out_dir = os.path.join(artifacts_dir, "schedules", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    schedule_to_csv(sched_opt, os.path.join(out_dir, "schedule_baseline.csv"))
    schedule_to_csv(sched_naive, os.path.join(out_dir, "schedule_naive.csv"))
    save_json(stats, os.path.join(out_dir, "schedule_stats_baseline.json"))

    print("Saved:")
    print(" -", os.path.join(out_dir, "schedule_baseline.csv"))
    print(" -", os.path.join(out_dir, "schedule_naive.csv"))
    print(" -", os.path.join(out_dir, "schedule_stats_baseline.json"))
    print("Stats:", stats)


if __name__ == "__main__":
    main()
