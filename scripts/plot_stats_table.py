# scripts/plot_stats_table.py
import os, sys, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_stats_table(stats_path: str, out_png: str):
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # Build DataFrame with formatted values
    rows = [
        ["Currency", stats.get("currency", "USD")],
        ["Naive Cost", f"{stats['naive_cost']:.3f}"],
        ["Optimized Cost", f"{stats['opt_cost']:.3f}"],
        ["Savings ($)", f"{stats['savings_$']:.3f}"],
        ["Base Peak (kW)", f"{stats['base_peak_kw']:.3f}"],
        ["Naive Peak (kW)", f"{stats['naive_peak_kw']:.3f}"],
        ["Optimized Peak (kW)", f"{stats['opt_peak_kw']:.3f}"],
        ["Peak Reduction (kW)", f"{stats['peak_reduction_kw']:.3f}"],
        ["Power Cap (kW)", str(stats.get("power_cap_kw", 'None'))],
        ["Precedence", str(stats.get("precedence", []))],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(6, len(rows) * 0.4))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print("Saved table:", out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="artifacts/lstm/schedules", help="Schedules directory root")
    args = ap.parse_args()

    stats_path = os.path.join(args.root, args.tag, "milp", "schedule_stats_milp.json")
    out_png = os.path.join(args.root, args.tag, "milp", "milp_stats_table.png")
    plot_stats_table(stats_path, out_png)

if __name__ == "__main__":
    main()
