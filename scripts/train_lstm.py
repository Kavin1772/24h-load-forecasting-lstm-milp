# scripts/train_lstm.py
"""
Usage:
  python scripts/train_lstm.py --config configs/config.yaml ^
      --t_in 168 --t_out 24 --stride 1 --epochs 20 --batch_size 64 --lr 1e-3 ^
      --hidden 64 --layers 1 --dropout 0.0 --use_time_features 1 --tag run1

This will:
  - Build LSTM datasets from your locked splits & scaler
  - Train a simple LSTM forecaster
  - Save weights, metrics.json, and a sample forecast plot
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.features_lstm import make_lstm_datasets, load_config
from src.models.lstm import Seq2HorizLSTM

# ---- Dataset wrapper ----
class WindowDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def metric_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def metric_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def metric_mape(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total = 0.0
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = crit(pred, yb)
        total += loss.item() * xb.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(yb.cpu().numpy())
    if preds:
        P = np.concatenate(preds, axis=0)
        T = np.concatenate(trues, axis=0)
        rmse = metric_rmse(T, P)
        mae  = metric_mae(T, P)
        mape = metric_mape(T, P)
    else:
        rmse = mae = mape = float("nan")
    return total / max(1, len(loader.dataset)), rmse, mae, mape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--t_in", type=int, default=168)
    ap.add_argument("--t_out", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use_time_features", type=int, default=0)  # 1 = yes
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    scaler_dir = os.path.dirname(cfg["scaler"]["save_path"])
    out_dir = os.path.join(scaler_dir, "lstm")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build datasets (uses existing scaler; will write a new one if time features are added)
    ds_paths = make_lstm_datasets(
        config_path=args.config,
        T_in=args.t_in,
        T_out=args.t_out,
        stride=args.stride,
        use_time_features=bool(args.use_time_features),
        out_name=f"{args.tag}_t{args.t_in}_h{args.t_out}"
    )

    train_npz, val_npz, test_npz = ds_paths["train"], ds_paths["val"], ds_paths["test"]
    meta_json = ds_paths["meta"]

    # 2) Load datasets
    dtr = WindowDataset(train_npz)
    dva = WindowDataset(val_npz)
    dte = WindowDataset(test_npz)

    in_features = dtr.X.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Model / Optim / Loss
    model = Seq2HorizLSTM(in_features, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout, T_out=args.t_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    Ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    Lva = DataLoader(dva, batch_size=args.batch_size, shuffle=False, drop_last=False)
    Lte = DataLoader(dte, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 4) Train with simple early stopping
    best_val = float("inf")
    best_path = os.path.join(out_dir, f"{args.tag}_best.pt")
    patience, bad = 5, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, Ltr, crit, opt, device)
        va_loss, va_rmse, va_mae, va_mape = evaluate(model, Lva, crit, device)
        print(f"Epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | RMSE {va_rmse:.3f} | MAE {va_mae:.3f} | MAPE {va_mape:.2f}%")
        if np.isfinite(va_loss) and va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Ensure we have a checkpoint even if val_loss was NaN
    if not os.path.exists(best_path):
        print("[warn] No valid best checkpoint saved; saving last model instead.")
        torch.save(model.state_dict(), best_path)

    # 5) Load best and evaluate on test
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    te_loss, te_rmse, te_mae, te_mape = evaluate(model, Lte, crit, device)

    # 6) Save metrics + a sample plot
    metrics = {
        "val": {"loss": float(best_val)},
        "test": {"loss": float(te_loss), "rmse": te_rmse, "mae": te_mae, "mape": te_mape},
        "data": ds_paths,
        "in_features": int(in_features),
        "t_in": args.t_in,
        "t_out": args.t_out,
        "tag": args.tag,
    }
    with open(os.path.join(out_dir, f"{args.tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Optional: plot first test example
    try:
        import matplotlib.pyplot as plt
        sample = next(iter(Lte))
        xb, yb = sample
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        truth = yb.numpy()
        plt.figure()
        plt.plot(truth[0], label="truth")
        plt.plot(pred[0],  label="pred")
        plt.title(f"{args.tag}: first test window")
        plt.legend()
        fig_path = os.path.join(out_dir, f"{args.tag}_sample.png")
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        print(f"Saved sample plot: {fig_path}")
    except Exception as e:
        print(f"(plot skipped) {e}")

    print("Done.")

if __name__ == "__main__":
    main()
