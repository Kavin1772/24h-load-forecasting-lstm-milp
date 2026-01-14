import matplotlib.pyplot as plt

# ------------------------------
# 1. Hidden Size Comparison
# ------------------------------
hidden_sizes = [32, 64, 128, 256]
rmse_hidden = [294.095, 81.186, 71.477, 70.232]   # from your JSON

plt.figure(figsize=(8, 5))
plt.plot(hidden_sizes, rmse_hidden, marker='o')
plt.xlabel("Hidden Size")
plt.ylabel("RMSE")
plt.title("Effect of Hidden Size on LSTM Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig("hidden_size_vs_rmse.png", dpi=300)
plt.show()

# ------------------------------
# 2. Number of Layers Comparison
# ------------------------------
layers = [1, 2]
rmse_layers = [80.705, 117.460]

plt.figure(figsize=(8, 5))
plt.bar([str(l) for l in layers], rmse_layers)
plt.xlabel("Number of LSTM Layers")
plt.ylabel("RMSE")
plt.title("Effect of Number of Layers on LSTM Performance")
plt.tight_layout()
plt.savefig("layers_vs_rmse.png", dpi=300)
plt.show()

# ------------------------------
# 3. Dropout Comparison
# ------------------------------
dropouts = [0.0, 0.2, 0.5]
rmse_dropout = [95.219, 93.326, 92.845]

plt.figure(figsize=(8, 5))
plt.plot(dropouts, rmse_dropout, marker='o')
plt.xlabel("Dropout Rate")
plt.ylabel("RMSE")
plt.title("Effect of Dropout on LSTM Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig("dropout_vs_rmse.png", dpi=300)
plt.show()
