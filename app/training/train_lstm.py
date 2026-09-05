from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.lstm_model import (
    HIDDEN_SIZE,
    LSTMRegressor,
    SEQUENCE_FEATURES,
    SEQUENCE_LENGTH,
)



# CONFIGURATION

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 4
FUTURE_HORIZON = 20

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2023-12-31")
VAL_START = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-08-01")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nse_market_data.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "lstm"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)



# REPRODUCIBILITY


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



# DATA LOADING


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(
        DATA_PATH,
        header=[0, 1],
        index_col=0,
    )
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[~raw.index.isna()]

    stocks = []
    for ticker in raw.columns.get_level_values(0).unique():
        if ticker == "^NSEI":
            continue
        frame = raw[ticker].copy()
        frame = frame.rename_axis("Date").reset_index()
        frame["Ticker"] = ticker
        stocks.append(frame[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]])

    stock_df = pd.concat(stocks, ignore_index=True)

    nifty = raw["^NSEI"].copy()
    nifty = nifty.rename_axis("Date").reset_index()
    nifty = nifty[["Date", "Open", "High", "Low", "Close", "Volume"]]

    return stock_df, nifty



# LSTM-SPECIFIC FEATURE ENGINEERING


def build_lstm_features(stock: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    d = stock.sort_values("Date").reset_index(drop=True).copy()
    market = nifty.sort_values("Date").reset_index(drop=True).copy()

    # Daily returns are the base for all volatility calculations.
    d["return_1d"] = d["Close"].pct_change(fill_method=None)
    d["return_5d"] = d["Close"].pct_change(5, fill_method=None)
    d["return_20d"] = d["Close"].pct_change(20, fill_method=None)
    d["return_60d"] = d["Close"].pct_change(60, fill_method=None)

    # Past volatility: information available at prediction time.
    d["past_volatility_20d"] = d["return_1d"].rolling(20).std()
    d["past_volatility_60d"] = d["return_1d"].rolling(60).std()

    sma20 = d["Close"].rolling(20).mean()
    sma50 = d["Close"].rolling(50).mean()
    d["price_to_sma20"] = d["Close"] / sma20 - 1
    d["price_to_sma50"] = d["Close"] / sma50 - 1

    rolling_peak = d["Close"].rolling(60).max()
    d["drawdown_60d"] = d["Close"] / rolling_peak - 1

    market["market_return_1d"] = market["Close"].pct_change(fill_method=None)
    market["market_volatility_20d"] = market["market_return_1d"].rolling(20).std()

    d = d.merge(
        market[["Date", "market_volatility_20d"]],
        on="Date",
        how="left",
    )

    # Target uses ONLY the next 20 trading days.
    # For row t, target = std(return[t+1] ... return[t+20]).
    d["future_volatility_20d"] = (
        d["return_1d"]
        .shift(-1)
        .rolling(FUTURE_HORIZON)
        .std()
        .shift(-(FUTURE_HORIZON - 1))
    )

    return d.replace([np.inf, -np.inf], np.nan)


# SEQUENCE CREATION


def make_sequences(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = frame.sort_values("Date").reset_index(drop=True)
    valid = frame.dropna(subset=SEQUENCE_FEATURES + ["future_volatility_20d"]).reset_index(drop=True)

    X, y, dates = [], [], []

    for end_idx in range(SEQUENCE_LENGTH - 1, len(valid)):
        X.append(valid.loc[end_idx - SEQUENCE_LENGTH + 1:end_idx, SEQUENCE_FEATURES].to_numpy(dtype=np.float32))
        y.append(float(valid.loc[end_idx, "future_volatility_20d"]))
        dates.append(valid.loc[end_idx, "Date"])

    if not X:
        return (
            np.empty((0, SEQUENCE_LENGTH, len(SEQUENCE_FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    return np.stack(X), np.asarray(y, dtype=np.float32), np.asarray(dates)


def build_all_sequences(stock_df: pd.DataFrame, nifty_df: pd.DataFrame):
    X_parts, y_parts, date_parts, ticker_parts = [], [], [], []

    for ticker, group in stock_df.groupby("Ticker", sort=True):
        features = build_lstm_features(group, nifty_df)
        X, y, dates = make_sequences(features)

        if len(X) == 0:
            continue

        X_parts.append(X)
        y_parts.append(y)
        date_parts.append(dates)
        ticker_parts.extend([ticker] * len(X))

    return (
        np.concatenate(X_parts),
        np.concatenate(y_parts),
        np.concatenate(date_parts),
        np.asarray(ticker_parts),
    )



# TEMPORAL SPLIT + PURGE


def split_sequences(X, y, dates, tickers):
    # Purge 20 trading-day boundary regions so train/validation and
    # validation/test targets cannot overlap across the split.
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    val_mask = (dates >= VAL_START + pd.Timedelta(days=30)) & (dates <= VAL_END - pd.Timedelta(days=30))
    test_mask = (dates >= TEST_START + pd.Timedelta(days=30)) & (dates <= TEST_END)

    return (
        (X[train_mask], y[train_mask], dates[train_mask], tickers[train_mask]),
        (X[val_mask], y[val_mask], dates[val_mask], tickers[val_mask]),
        (X[test_mask], y[test_mask], dates[test_mask], tickers[test_mask]),
    )



# SCALING


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def transform_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = X.shape
    transformed = scaler.transform(X.reshape(-1, shape[-1]))
    return transformed.reshape(shape).astype(np.float32)


# TRAINING


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X).to(device)).detach().cpu().numpy()

    return {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
    }


def train_model(X_train, y_train, X_val, y_val, device):
    model = LSTMRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(xb)

        train_loss = running_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_prediction = model(torch.from_numpy(X_val).to(device))
            val_loss = loss_fn(val_prediction, torch.from_numpy(y_val).to(device)).item()

        print(f"Epoch {epoch:02d} | train_mse={train_loss:.8f} | val_mse={val_loss:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model



# MAIN


def main():
    set_seed()

    print(f"Loading data from: {DATA_PATH}")
    stock_df, nifty_df = load_market_data()
    print(f"Stock rows: {len(stock_df):,}")
    print(f"NIFTY rows: {len(nifty_df):,}")

    X, y, dates, tickers = build_all_sequences(stock_df, nifty_df)
    print(f"Sequences: {len(X):,}")
    print(f"Sequence shape: {X.shape[1:]}")

    (X_train, y_train, train_dates, _), (X_val, y_val, val_dates, _), (X_test, y_test, test_dates, _) = split_sequences(
        X, y, dates, tickers
    )

    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} | Test: {len(X_test):,}")

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise RuntimeError("One or more temporal splits are empty. Check the dataset date range.")

    scaler = fit_scaler(X_train)
    X_train = transform_sequences(X_train, scaler)
    X_val = transform_sequences(X_val, scaler)
    X_test = transform_sequences(X_test, scaler)

    device = get_device()
    print(f"Training device: {device}")

    model = train_model(X_train, y_train, X_val, y_val, device)

    val_metrics = evaluate(model, X_val, y_val, device)
    test_metrics = evaluate(model, X_test, y_test, device)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    # Always save CPU weights so the artifact is portable across Mac/Linux/CPU/GPU.
    torch.save(model.cpu().state_dict(), ARTIFACT_DIR / "model.pth")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.pkl")

    metadata = {
        "model_type": "LSTMRegressor",
        "sequence_features": SEQUENCE_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": HIDDEN_SIZE,
        "future_horizon": FUTURE_HORIZON,
        "target": "future_volatility_20d",
        "target_definition": "std(return[t+1:t+20])",
        "train_period": [str(TRAIN_START.date()), str(TRAIN_END.date())],
        "validation_period": [str(VAL_START.date()), str(VAL_END.date())],
        "test_period": [str(TEST_START.date()), str(TEST_END.date())],
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved artifacts to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
