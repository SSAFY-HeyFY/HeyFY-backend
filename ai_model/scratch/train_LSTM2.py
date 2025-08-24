
import os, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import TimeSeriesModel
from utils.data_utils import build_feature_target_arrays, make_sequences, get_scalers, split_timewise

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def main(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        C = json.load(f)

    data_path      = C["DATA_FILE_PATH"]
    date_col       = C.get("DATE_COL", "Date")
    target_col     = C.get("TARGET_COL", "ECOS_Close")
    feature_excl   = C.get("FEATURE_EXCLUDES", ["Date", "ECOS_Close"])
    window         = int(C.get("WINDOW_SIZE", 30))
    horizon        = int(C.get("PREDICTION_DAYS", 1))  # 단일스텝: 1
    model_type     = C.get("MODEL_TYPE", "LSTM")
    hidden_size    = int(C.get("HIDDEN_SIZE", 128))
    num_layers     = int(C.get("NUM_LAYERS", 2))
    dropout        = float(C.get("DROPOUT", 0.2))
    lr             = float(C.get("LR", 1e-3))
    batch_size     = int(C.get("BATCH_SIZE", 64))
    epochs         = int(C.get("EPOCHS", 100))
    patience       = int(C.get("PATIENCE", 10))
    val_ratio      = float(C.get("VAL_RATIO", 0.15))
    test_ratio     = float(C.get("TEST_RATIO", 0.15))
    scaler_type    = C.get("SCALER_TYPE", "standard")
    model_dir      = C.get("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load data
    if data_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    # Build arrays
    X_raw, y_raw, dates = build_feature_target_arrays(df, date_col, target_col, feature_excl)

    # Scalers
    feat_scaler, y_scaler = get_scalers(scaler_type)
    X_scaled = feat_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1,1)).reshape(-1)

    # Sequences (horizon=1)
    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=window, horizon=horizon)
    dates_seq = dates[window: window + len(X_seq)]  # 예측 시작일

    # Split
    (X_tr, y_tr, d_tr), (X_va, y_va, d_va), (X_te, y_te, d_te) = split_timewise(
        X_seq, y_seq, dates_seq, val_ratio=val_ratio, test_ratio=test_ratio
    )

    train_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SeqDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    model = TimeSeriesModel(
        input_size=X_tr.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pred_horizon=horizon,
        model_type=model_type
    ).to(device)

    crit = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf"); best_state=None; no_improve=0

    # Train
    for epoch in range(1, epochs+1):
        model.train(); total=0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item() * xb.size(0)
        tr_loss = total / len(train_loader.dataset)

        # val
        model.eval(); total=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = crit(pred, yb)
                total += loss.item() * xb.size(0)
        va_loss = total / len(val_loader.dataset)

        # KRW-scale RMSE (t+1) on val
        with torch.no_grad():
            preds=[]; trues=[]
            for xb, yb in val_loader:
                xb = xb.to(device)
                yhat = model(xb).cpu().numpy()  # (B,1)
                preds.append(yhat); trues.append(yb.numpy())
            preds = np.vstack(preds); trues = np.vstack(trues)
            inv_pred = y_scaler.inverse_transform(preds.reshape(-1,1)).reshape(-1)
            inv_true = y_scaler.inverse_transform(trues.reshape(-1,1)).reshape(-1)
            rmse_krw = math.sqrt(mean_squared_error(inv_true, inv_pred))

        print(f"[{epoch:03d}] train:{tr_loss:.6f} | val:{va_loss:.6f} | val RMSE₩:{rmse_krw:.3f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss; best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at {epoch}. best val:{best_val:.6f}")
            break

    # Save
    ckpt = os.path.join(model_dir, f"{model_type}_W{window}_P{horizon}_best_model.pt")
    torch.save(best_state, ckpt); print("Saved:", ckpt)
    joblib.dump({"feature_scaler": feat_scaler, "target_scaler": y_scaler},
                os.path.join(model_dir,"scalers.pkl"))
    with open(os.path.join(model_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(C, f, ensure_ascii=False, indent=2)

    # Evaluate on test
    model.load_state_dict(best_state)
    def predict(loader):
        model.eval(); out=[]
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device); out.append(model(xb).cpu().numpy())
        return np.vstack(out)  # (N,1)
    y_te_pred_s = predict(test_loader).reshape(-1,1)
    y_te_true_s = y_te.reshape(-1,1)

    y_te_pred = y_scaler.inverse_transform(y_te_pred_s).reshape(-1)
    y_te_true = y_scaler.inverse_transform(y_te_true_s).reshape(-1)
    d_te_reset = d_te.reset_index(drop=True)

    rmse = math.sqrt(mean_squared_error(y_te_true, y_te_pred))
    mae  = mean_absolute_error(y_te_true, y_te_pred)
    print(f"Test t+1 → RMSE₩:{rmse:.3f} | MAE₩:{mae:.3f}")

    # Plot: Test t+1 rolling
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig1 = os.path.join(model_dir, "test_tplus1_rolling.png")
    plt.figure(figsize=(12,4))
    plt.plot(d_te_reset, y_te_true, label="Actual (t+1)")
    plt.plot(d_te_reset, y_te_pred, linestyle="--", label="Pred (t+1)")
    plt.title("Test: next-day (t+1) prediction")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.legend(); plt.tight_layout(); plt.savefig(fig1, dpi=150); plt.close()
    print("Saved:", fig1)

if __name__ == "__main__":
    main()
