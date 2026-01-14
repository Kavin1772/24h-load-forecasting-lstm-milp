import os, sys, numpy as np, importlib
# Make sure project root is on sys.path
sys.path[:0] = [r"A:\AI in Engg\Project"]

import src.scheduling.heuristics as H
importlib.reload(H)

print("Module path:", H.__file__, "| version:", getattr(H, "HEURISTICS_VERSION", "unknown"))

meta = H.load_yaml(r"A:\AI in Engg\Project\configs\tariff_and_appliances.yaml")
prices = np.array(meta["tou"]["prices"], dtype=float)

print("Optimized schedule:", H.schedule_day(prices, meta))
print("Naive schedule    :", H.schedule_naive(prices, meta))
