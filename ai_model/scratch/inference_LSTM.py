
import os, json, math
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import TimeSeriesModel
from utils.data_utils import build_feature_target_arrays, make_sequences, get_scalers

def main(config_path="config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        C = json.load(f)

    data_path      = C["DATA_FILE_PATH"]
    date_col       = C.get("DATE_COL", "Date")
    target_col     = C.get("TARGET_COL", "ECOS_Close")
    feature_excl   = C.get("FEATURE_EXCLUDES", ["Date", "ECOS_Close"])
    window         = int(C.get("WINDOW_SIZE", 30))
    horizon        = 1  # 단일스텝
    model_dir      = C.get("MODEL_DIR", "models")
    model_type     = C.get("MODEL_TYPE", "LSTM")
    hidden_size    = int(C.get("HIDDEN_SIZE", 128))
    num_layers     = int(C.get("NUM_LAYERS", 2))
    dropout        = float(C.get("DROPOUT", 0.2))

    # Load data
    if data_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    X_raw, y_raw, dates = build_feature_target_arrays(df, date_col, target_col, feature_excl)

    # Load scalers & model
    scalers = joblib.load(os.path.join(model_dir, "scalers.pkl"))
    feat_scaler, y_scaler = scalers["feature_scaler"], scalers["target_scaler"]

    X_scaled = feat_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw.reshape(-1,1)).reshape(-1)

    model = TimeSeriesModel(
        input_size=X_scaled.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pred_horizon=horizon,
        model_type=model_type
    )
    ckpt = os.path.join(model_dir, f"{model_type}_W{window}_P{horizon}_best_model.pt")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state); model.eval()

    # Make sequences
    X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=window, horizon=horizon)
    dates_seq = dates[window: window + len(X_seq)]

    # Predict all
    preds = []
    with torch.no_grad():
        for i in range(len(X_seq)):
            xb = torch.from_numpy(X_seq[i:i+1]).float()
            yhat = model(xb).numpy()[0,0]
            preds.append(yhat)
    preds = np.array(preds)
    preds_inv = y_scaler.inverse_transform(preds.reshape(-1,1)).reshape(-1)
    true_inv  = y_scaler.inverse_transform(y_seq.reshape(-1,1)).reshape(-1)

    # Plot last 200 points for clarity
    K = 200 if len(preds_inv) > 200 else len(preds_inv)
    d_plot = dates_seq.iloc[-K:]
    plt.figure(figsize=(12,4))
    plt.plot(d_plot, true_inv[-K:], label="Actual")
    plt.plot(d_plot, preds_inv[-K:], linestyle="--", label="Pred")
    plt.title("Rolling t+1 Forecast (last 200 points)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.legend(); plt.tight_layout()
    out = os.path.join(model_dir, "rolling_tplus1_last200.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()
