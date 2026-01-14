import matplotlib.pyplot as plt

# Values from user's JSON
naive_cost = 2.567849
opt_cost = 2.811849

naive_peak = 3.720837
opt_peak = 3.719854

# 1. Cost comparison bar chart
plt.figure(figsize=(7,5))
plt.bar(["Naive", "Optimized"], [naive_cost, opt_cost])
plt.ylabel("Cost ($)")
plt.title("Energy Cost Comparison: Naive vs Optimized Scheduling")
plt.tight_layout()
plt.savefig('A:\AI in Engg\Project\outputscost_comparison.png', dpi=300)
plt.show()

# 2. Peak load comparison bar chart
plt.figure(figsize=(7,5))
plt.bar(["Naive", "Optimized"], [naive_peak, opt_peak])
plt.ylabel("Peak Load (kW)")
plt.title("Peak Load Comparison: Naive vs Optimized Scheduling")
plt.tight_layout()
plt.savefig('A:\AI in Engg\Project\outputscost_comparison.png', dpi=300)
plt.show()

"/mnt/data/cost_comparison.png", "/mnt/data/peak_comparison.png"
