# scripts/plot_schedule_milp.py
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure project root on path if you run this from elsewhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def _read_csv(p):
    df = pd.read_csv(p)
    if not {"appliance", "hour"}.issubset(df.columns):
        raise ValueError(f"{p} must contain columns: appliance,hour")
    return df

def _read_json(p):
    with open(p, "r") as f:
        return json.load(f)

def _build_blocks(df: pd.DataFrame):
    """Turn per-hour rows into contiguous blocks per appliance for a Gantt chart."""
    rows = []
    for name, g in df.groupby("appliance"):
        hours = sorted(int(h) for h in g["hour"].tolist())
        if not hours:
            continue
        start = hours[0]
        prev = start
        for h in hours[1:] + [None]:
            if h is None or h != prev + 1:
                rows.append((name, start, prev))
                if h is not None:
                    start = h
            prev = h if h is not None else prev
    return rows

def plot_gantt(schedule_csv: str, out_png: str, title: str):
    df = _read_csv(schedule_csv)
    blocks = _build_blocks(df)
    if not blocks:
        blocks = [("No Tasks", 0, 0)]
    names = sorted({b[0] for b in blocks})
    ymap = {n: i for i, n in enumerate(names)}

    fig = plt.figure(figsize=(10, 3 + 0.4 * max(1, len(names))))
    ax = plt.gca()
    for name, s, e in blocks:
        y = ymap[name]
        ax.barh(y, (e - s + 1), left=s, height=0.6)
        ax.text(s + (e - s + 1) / 2, y, name, ha="center", va="center", fontsize=9)

    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels(names)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.set_xlabel("Hour of day")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("Saved Gantt:", out_png)

def plot_bars(stats_json: str, out_png_prefix: str):
    s = _read_json(stats_json)
    labels = ["Naive", "Optimized"]
    cost = [s["naive_cost"], s["opt_cost"]]
    peak = [s["naive_peak_kw"], s["opt_peak_kw"]]
    currency = s.get("currency", "USD")

    # Cost bar
    fig1 = plt.figure(figsize=(5, 4))
    ax1 = plt.gca()
    ax1.bar(labels, cost)
    ax1.set_ylabel(f"Cost ({currency})")
    ax1.set_title("Daily Cost: Naive vs MILP")
    fig1.tight_layout()
    cost_png = out_png_prefix + "_cost.png"
    os.makedirs(os.path.dirname(cost_png), exist_ok=True)
    fig1.savefig(cost_png, dpi=150, bbox_inches="tight")
    print("Saved cost bars:", cost_png)

    # Peak bar
    fig2 = plt.figure(figsize=(5, 4))
    ax2 = plt.gca()
    ax2.bar(labels, peak)
    ax2.set_ylabel("Peak demand (kW)")
    ax2.set_title("Peak Demand: Naive vs MILP")
    fig2.tight_layout()
    peak_png = out_png_prefix + "_peak.png"
    fig2.savefig(peak_png, dpi=150, bbox_inches="tight")
    print("Saved peak bars:", peak_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Run tag used in training (e.g., lstm_v4)")
    ap.add_argument("--root", default="artifacts/lstm/schedules", help="Base schedules directory")
    args = ap.parse_args()

    base = os.path.join(args.root, args.tag, "milp")
    # Inputs
    opt_csv   = os.path.join(base, "schedule_opt_milp.csv")
    stats_json= os.path.join(base, "schedule_stats_milp.json")
    # Outputs
    gantt_png = os.path.join(base, "milp_schedule_gantt.png")
    bars_pref = os.path.join(base, "milp_bars")

    plot_gantt(opt_csv, gantt_png, title=f"{args.tag} – MILP Optimized 24h Schedule")
    plot_bars(stats_json, bars_pref)

if __name__ == "__main__":
    main()
