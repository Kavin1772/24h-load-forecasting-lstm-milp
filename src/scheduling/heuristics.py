import os
import json
import math
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple

# ---- Forecast utilities ----
def load_npz(npz_path: str):
    d = np.load(npz_path)
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)

def load_best_model(model_path: str, in_features: int, T_out: int):
    from src.models.lstm import Seq2HorizLSTM
    m = Seq2HorizLSTM(in_features=in_features, T_out=T_out)
    state = torch.load(model_path, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

@torch.no_grad()
def predict_next24(model, X_last: np.ndarray) -> np.ndarray:
    # X_last: (T_in, F)
    xb = torch.from_numpy(X_last[None, ...])
    yb = model(xb).cpu().numpy()[0]  # (24,)
    return yb

# ---- Tariff & appliance I/O ----
def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---- Scheduling helpers ----
def cheapest_hours(prices: np.ndarray, n: int, window: Tuple[int, int]) -> List[int]:
    """Return n hour indices with lowest price within [start,end)."""
    start, end = window
    hours = list(range(start, end))
    order = sorted(hours, key=lambda h: prices[h])
    return order[:n]

def place_non_interruptible(mask: np.ndarray, prices: np.ndarray, duration: int, window: Tuple[int, int]) -> int:
    """Return best start hour for a consecutive block of 'duration' minimizing price within window."""
    start, end = window
    best_h, best_cost = None, math.inf
    for s in range(start, end - duration + 1):
        if mask[s:s + duration].any():  # already occupied
            continue
        c = prices[s:s + duration].sum()
        if c < best_cost:
            best_cost, best_h = c, s
    return best_h

def schedule_day(prices: np.ndarray, meta: Dict) -> Dict[str, List[int]]:
    """
    Greedy cost-aware schedule:
      1) Sort non-interruptible by rated_kw desc, place the cheapest feasible block (price * kW).
      2) Sort interruptible by rated_kw desc, fill cheapest remaining hours inside their windows.
    """
    sched = {}
    occupied = np.zeros(24, dtype=bool)

    # 1) Non-interruptible (largest kW first)
    ni = [ap for ap in meta["appliances"] if not ap.get("interruptible", False)]
    ni.sort(key=lambda ap: float(ap["rated_kw"]), reverse=True)
    for ap in ni:
        dur = int(ap["duration_h"])
        startw = int(ap["earliest_start"])
        endw   = int(ap["latest_end"])
        best_h, best_cost = None, float("inf")
        for s in range(startw, endw - dur + 1):
            if occupied[s:s+dur].any():
                continue
            c = prices[s:s+dur].sum() * float(ap["rated_kw"])
            if c < best_cost:
                best_cost, best_h = c, s
        if best_h is None:
            sched[ap["name"]] = []
        else:
            hrs = list(range(best_h, best_h + dur))
            occupied[hrs] = True
            sched[ap["name"]] = hrs

    # 2) Interruptible (largest kW first → cheapest free hours)
    it = [ap for ap in meta["appliances"] if ap.get("interruptible", False)]
    it.sort(key=lambda ap: float(ap["rated_kw"]), reverse=True)
    for ap in it:
        startw = int(ap["earliest_start"])
        endw   = int(ap["latest_end"])
        dur    = int(ap["duration_h"])
        cand = [h for h in range(startw, endw) if not occupied[h]]
        hrs = sorted(cand, key=lambda h: prices[h])[:dur]
        occupied[hrs] = True
        sched[ap["name"]] = sorted(hrs)

    return sched


# ---- Naive baseline schedule ----
def schedule_naive(prices: np.ndarray, meta: Dict) -> Dict[str, List[int]]:
    """
    Baseline: earliest feasible start for each appliance (non-interruptible as a block),
    interruptible = earliest 'duration_h' distinct hours. Ignores price.
    Always tries to place exactly 'duration_h' hours if available in the window.
    """
    sched = {}
    occupied = np.zeros(24, dtype=bool)

    # Non-interruptible (earliest feasible block)
    for ap in meta["appliances"]:
        dur = int(ap["duration_h"])
        startw = int(ap["earliest_start"])
        endw   = int(ap["latest_end"])
        placed = None
        for s in range(startw, endw - dur + 1):
            if not occupied[s:s+dur].any():
                placed = s
                break
        if placed is None:
            sched[ap["name"]] = []
        else:
            hrs = list(range(placed, placed + dur))
            occupied[hrs] = True
            sched[ap["name"]] = hrs

    # Interruptible (earliest distinct hours)
    for ap in meta["appliances"]:
        if ap.get("interruptible", False):
            dur = int(ap["duration_h"])
            startw = int(ap["earliest_start"])
            endw   = int(ap["latest_end"])
            cand = [h for h in range(startw, endw) if not occupied[h]]
            hrs = sorted(cand)[:dur]
            # ensure we actually picked 'dur' hours if possible
            if len(cand) >= dur and len(hrs) < dur:
                hrs = sorted(cand)[:dur]
            occupied[hrs] = True
            sched[ap["name"]] = sorted(hrs)

    return sched

# ---- KPI computation ----
def total_cost_with_appliances(
    pred_load_kw: np.ndarray,
    sched: Dict[str, List[int]],
    meta: Dict
) -> Tuple[float, float, float]:
    """
    Returns (total_cost, peak_kw, base_peak_kw) using predicted base load in kW.
    """
    prices = np.array(meta["tou"]["prices"], dtype=float)  # $/kWh
    add = np.zeros(24, dtype=float)
    kw_map = {ap["name"]: float(ap["rated_kw"]) for ap in meta["appliances"]}
    for name, hours in sched.items():
        for h in hours:
            add[h] += kw_map[name]
    total_kw = pred_load_kw + add
    cost = float(np.sum(total_kw * prices))
    peak_kw = float(np.max(total_kw))
    base_peak_kw = float(np.max(pred_load_kw))
    return cost, peak_kw, base_peak_kw

# ---- Export helpers ----
def schedule_to_csv(sched: Dict[str, List[int]], out_csv: str):
    rows = []
    for name, hours in sched.items():
        for h in hours:
            rows.append({"appliance": name, "hour": h})
    df = pd.DataFrame(rows).sort_values(["appliance", "hour"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

def save_json(d: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
