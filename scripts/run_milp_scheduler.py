# scripts/run_milp_scheduler.py
import os, sys, json, argparse, numpy as np

# ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scheduling.heuristics import (
    load_npz, load_best_model, predict_next24, load_yaml,
    schedule_naive, total_cost_with_appliances, schedule_to_csv, save_json
)
from src.scheduling.milp_scheduler import optimize_day_milp, schedule_cost_peak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="LSTM run tag (e.g., lstm_v4)")
    ap.add_argument("--config", required=True, help="Path to configs/config.yaml")
    ap.add_argument("--yaml_meta", required=True, help="Path to configs/tariff_and_appliances.yaml")
    args = ap.parse_args()

    # Locate artifacts
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    artifacts_dir = os.path.join(os.path.dirname(cfg["scaler"]["save_path"]), "lstm")

    best_pt   = os.path.join(artifacts_dir, f"{args.tag}_best.pt")
    meta_json = os.path.join(artifacts_dir, f"{args.tag}_t168_h24_meta.json")
    with open(meta_json, "r") as f:
        meta_ds = json.load(f)

    test_npz = os.path.join(artifacts_dir, f"{args.tag}_t{meta_ds['T_in']}_h{meta_ds['T_out']}_test.npz")
    Xte, Yte = load_npz(test_npz)
    X_last = Xte[-1]
    in_features = X_last.shape[-1]

    # Forecast next-24 base load
    model = load_best_model(best_pt, in_features=in_features, T_out=meta_ds["T_out"])
    y_pred = predict_next24(model, X_last)

    # Convert to kW if needed
    if np.nanmedian(y_pred) > 50:
        y_pred_kw = y_pred / 1000.0
    else:
        y_pred_kw = y_pred

    # Load tariff + appliances
    meta_tar = load_yaml(args.yaml_meta)
    prices = np.array(meta_tar["tou"]["prices"], dtype=float)

    # Naive baseline (for comparison)
    sched_naive = schedule_naive(prices, meta_tar)
    naive_cost, naive_peak, base_peak = total_cost_with_appliances(y_pred_kw, sched_naive, meta_tar)

    # MILP optimal
    sched_opt = optimize_day_milp(prices, meta_tar, y_pred_kw)
    opt_cost, opt_peak, _ = schedule_cost_peak(y_pred_kw, sched_opt, meta_tar)

    # Save
    out_dir = os.path.join(artifacts_dir, "schedules", args.tag, "milp")
    os.makedirs(out_dir, exist_ok=True)
    schedule_to_csv(sched_naive, os.path.join(out_dir, "schedule_naive.csv"))
    schedule_to_csv(sched_opt,   os.path.join(out_dir, "schedule_opt_milp.csv"))

    stats = {
        "currency": meta_tar["tou"].get("currency", "USD"),
        "naive_cost": naive_cost,
        "opt_cost": opt_cost,
        "savings_$": naive_cost - opt_cost,
        "base_peak_kw": base_peak,
        "naive_peak_kw": naive_peak,
        "opt_peak_kw": opt_peak,
        "peak_reduction_kw": naive_peak - opt_peak,
        "power_cap_kw": meta_tar.get("power_cap_kw", None),
        "precedence": meta_tar.get("precedence", []),
    }
    save_json(stats, os.path.join(out_dir, "schedule_stats_milp.json"))

    print("Saved:")
    print(" -", os.path.join(out_dir, "schedule_naive.csv"))
    print(" -", os.path.join(out_dir, "schedule_opt_milp.csv"))
    print(" -", os.path.join(out_dir, "schedule_stats_milp.json"))
    print("Stats:", stats)

if __name__ == "__main__":
    main()
