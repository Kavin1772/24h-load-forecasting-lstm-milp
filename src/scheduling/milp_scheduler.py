import os
from typing import Dict, List, Tuple
import numpy as np
import pulp

def _read_optional(meta: Dict, key: str, default=None):
    return meta[key] if key in meta else default

def _window(ap: Dict) -> Tuple[int, int]:
    # window is [start, end) hours
    return int(ap["earliest_start"]), int(ap["latest_end"])

def _is_interruptible(ap: Dict) -> bool:
    return bool(ap.get("interruptible", False))

def optimize_day_milp(
    prices: np.ndarray,
    meta: Dict,
    base_load_kw: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Exact minimum-cost day-ahead scheduler with:
      - non-interruptible blocks (single contiguous start)
      - interruptible hours (choose any 'duration_h' hours)
      - optional power cap per hour
      - optional precedence (finish 'before' <= start 'after')
    prices:       shape (24,)   $/kWh per hour
    base_load_kw: shape (24,)   predicted base load in kW
    meta: parsed YAML with 'appliances', optional 'power_cap_kw', 'precedence'
    """
    H = 24
    appliances = meta["appliances"]
    power_cap = _read_optional(meta, "power_cap_kw", None)
    precedence = _read_optional(meta, "precedence", [])

    prob = pulp.LpProblem("min_cost_schedule", pulp.LpMinimize)

    # Decision variables
    y = {}   # non-interruptible start choices: y[(name, s)] in {0,1}
    x = {}   # interruptible hourly choices:   x[(name, h)] in {0,1}

    # Book-keeping to reconstruct hours & compute power per hour
    nonint_starts = {}  # map name -> list of feasible starts
    dur_map = {}
    kw_map = {}

    for ap in appliances:
        name = ap["name"]
        dur = int(ap["duration_h"])
        dur_map[name] = dur
        kw_map[name] = float(ap["rated_kw"])
        startw, endw = _window(ap)
        if _is_interruptible(ap):
            # hourly binaries in window
            for h in range(startw, endw):
                x[(name, h)] = pulp.LpVariable(f"x_{name}_{h}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            # must choose exactly dur hours if enough are available
            if endw - startw >= dur:
                prob += pulp.lpSum(x[(name, h)] for h in range(startw, endw)) == dur
            else:
                # window too small: choose at most available
                prob += pulp.lpSum(x[(name, h)] for h in range(startw, endw)) == (endw - startw)
        else:
            # non-interruptible: choose exactly one contiguous block start
            starts = list(range(startw, endw - dur + 1))
            nonint_starts[name] = starts
            for s in starts:
                y[(name, s)] = pulp.LpVariable(f"y_{name}_start_{s}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            if len(starts) > 0:
                prob += pulp.lpSum(y[(name, s)] for s in starts) == 1
            else:
                # no feasible start → force "none"
                # (no y vars; model will just skip this appliance)
                nonint_starts[name] = []

    # Hourly total load (base + appliances) and objective
    # Build active indicator for each appliance at each hour
    def active_noninterruptible(name: str, h: int):
        dur = dur_map[name]
        starts = nonint_starts.get(name, [])
        # Sum of starts that cover hour h
        return pulp.lpSum(y[(name, s)] for s in starts if (s <= h <= s + dur - 1))

    def active_interruptible(name: str, h: int):
        return x.get((name, h), 0.0)

    total_cost = 0.0
    for h in range(H):
        add_kw = []
        for ap in appliances:
            name = ap["name"]
            if _is_interruptible(ap):
                add_kw.append(kw_map[name] * active_interruptible(name, h))
            else:
                add_kw.append(kw_map[name] * active_noninterruptible(name, h))
        # total load at hour h
        total_kw_h = base_load_kw[h] + pulp.lpSum(add_kw)

        # Power cap (optional)
        if power_cap is not None:
            prob += total_kw_h <= float(power_cap)

        # Cost contribution (1-hour slots)
        total_cost += prices[h] * total_kw_h

    # Precedence constraints (optional)
    # Each precedence: finish(before) <= start(after)
    for rule in precedence:
        before = rule["before"]
        after = rule["after"]
        # For non-interruptible, get an integer start via "big-M" on y
        # Create aux integer vars for starts (only when appliance is non-interruptible)
        # For interruptible, approximate start/finish as min/max selected hour.
        # To keep model linear and compact, we enforce precedence only for NI→NI here.
        if (before in nonint_starts) and (after in nonint_starts):
            sb = pulp.LpVariable(f"start_{before}", lowBound=0, upBound=23, cat=pulp.LpInteger)
            sa = pulp.LpVariable(f"start_{after}",  lowBound=0, upBound=23, cat=pulp.LpInteger)
            # Link start integer to chosen y-combination
            if len(nonint_starts[before]) > 0:
                prob += sb == pulp.lpSum(s * y[(before, s)] for s in nonint_starts[before])
            if len(nonint_starts[after]) > 0:
                prob += sa == pulp.lpSum(s * y[(after, s)]  for s in nonint_starts[after])
            # finish(before) <= start(after)
            prob += sb + dur_map[before] <= sa
        # If you need precedence for interruptible devices as well, we can add
        # hour-wise implications—just say the word and I’ll extend it.

    prob += total_cost

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Reconstruct schedule
    sched: Dict[str, List[int]] = {}
    for ap in appliances:
        name = ap["name"]
        dur = dur_map[name]
        if _is_interruptible(ap):
            startw, endw = _window(ap)
            hrs = [h for h in range(startw, endw) if (name, h) in x and pulp.value(x[(name, h)]) > 0.5]
            sched[name] = sorted(hrs)
        else:
            starts = nonint_starts.get(name, [])
            if len(starts) == 0:
                sched[name] = []
            else:
                chosen = [s for s in starts if pulp.value(y[(name, s)]) > 0.5]
                if chosen:
                    s = int(round(chosen[0]))
                    sched[name] = list(range(s, s + dur))
                else:
                    sched[name] = []

    return sched

def schedule_cost_peak(pred_load_kw: np.ndarray, sched: Dict[str, List[int]], meta: Dict) -> Tuple[float, float, float]:
    prices = np.array(meta["tou"]["prices"], dtype=float)
    add = np.zeros(24, dtype=float)
    kw_map = {ap["name"]: float(ap["rated_kw"]) for ap in meta["appliances"]}
    for name, hours in sched.items():
        for h in hours:
            add[h] += kw_map[name]
    total_kw = pred_load_kw + add
    cost = float(np.sum(total_kw * prices))
    peak = float(np.max(total_kw))
    base_peak = float(np.max(pred_load_kw))
    return cost, peak, base_peak
