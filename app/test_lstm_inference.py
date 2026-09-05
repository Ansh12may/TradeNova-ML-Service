from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from app.data.market_data import (
    get_stock_data,
    get_nifty_data,
)

from app.lstm_model import LSTMRegressor


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ARTIFACT_DIR = PROJECT_ROOT / "app" / "artifacts" / "lstm"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TICKER = "RELIANCE.NS"

SEQUENCE_FEATURES = [
    "past_volatility_20d",
    "past_volatility_60d",
    "return_5d",
    "return_20d",
    "return_60d",
    "price_to_sma20",
    "price_to_sma50",
    "drawdown_60d",
    "market_volatility_20d",
]

SEQUENCE_LENGTH = 120
HIDDEN_SIZE = 32


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    print("Loading TradeNova market data...")

    stock_df = get_stock_data(TICKER)
    nifty_df = get_nifty_data()

    print(f"Stock rows: {len(stock_df)}")
    print(f"NIFTY rows: {len(nifty_df)}")

    return stock_df, nifty_df


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def create_features(stock_df, nifty_df):

    df = stock_df.copy()

    # Daily return
    df["return_1d"] = df["Close"].pct_change(fill_method=None)

    # Multi-horizon returns
    df["return_5d"] = df["Close"].pct_change(5, fill_method=None)
    df["return_20d"] = df["Close"].pct_change(20, fill_method=None)
    df["return_60d"] = df["Close"].pct_change(60, fill_method=None)

    # Historical volatility
    df["past_volatility_20d"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )

    df["past_volatility_60d"] = (
        df["return_1d"]
        .rolling(60)
        .std()
    )

    # Moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # Relative price position
    df["price_to_sma20"] = (
        df["Close"] / df["SMA20"] - 1
    )

    df["price_to_sma50"] = (
        df["Close"] / df["SMA50"] - 1
    )

    # 60-day drawdown
    rolling_peak = df["Close"].rolling(60).max()

    df["drawdown_60d"] = (
        df["Close"] / rolling_peak - 1
    )

    # -----------------------------------------------------
    # NIFTY market volatility
    # -----------------------------------------------------

    nifty = nifty_df[["Date", "Close"]].copy()

    nifty["market_return_1d"] = (
        nifty["Close"].pct_change(fill_method=None)
    )

    nifty["market_volatility_20d"] = (
        nifty["market_return_1d"]
        .rolling(20)
        .std()
    )

    df = df.merge(
        nifty[["Date", "market_volatility_20d"]],
        on="Date",
        how="left",
    )

    return df


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

def load_model():

    model_path = ARTIFACT_DIR / "model.pth"
    scaler_path = ARTIFACT_DIR / "scaler.pkl"

    print(f"\nLoading model: {model_path}")
    print(f"Loading scaler: {scaler_path}")

    model = LSTMRegressor(
        input_size=len(SEQUENCE_FEATURES),
        hidden_size=HIDDEN_SIZE,
    )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(scaler_path)

    return model, scaler


# ---------------------------------------------------------
# Main inference
# ---------------------------------------------------------

def main():

    stock_df, nifty_df = load_data()

    df = create_features(
        stock_df,
        nifty_df,
    )

    # Keep only the features used by the model
    clean_df = df.dropna(
        subset=SEQUENCE_FEATURES
    ).copy()

    print(f"\nRows after feature engineering: {len(clean_df)}")

    if len(clean_df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough rows for a "
            f"{SEQUENCE_LENGTH}-day sequence."
        )

    # Latest 120 observations
    latest_window = clean_df[
        SEQUENCE_FEATURES
    ].tail(SEQUENCE_LENGTH).values

    print(
        f"Latest sequence shape: "
        f"{latest_window.shape}"
    )

    # Load scaler and model
    model, scaler = load_model()

    # -----------------------------------------------------
    # Apply EXACTLY the same scaler used during training
    # -----------------------------------------------------

    scaled_window = scaler.transform(
        latest_window
    )

    # Add batch dimension:
    #
    # (120, 9)
    #      ↓
    # (1, 120, 9)
    #
    X = torch.tensor(
        scaled_window,
        dtype=torch.float32,
    ).unsqueeze(0)

    print(f"Tensor shape: {tuple(X.shape)}")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        prediction = model(X)

    predicted_volatility = (
        prediction.item()
    )

    prediction_date = clean_df["Date"].iloc[-1]

    print("\n" + "=" * 50)
    print("LSTM INFERENCE RESULT")
    print("=" * 50)

    print(f"Ticker: {TICKER}")
    print(f"As of: {prediction_date.date()}")

    print(
        f"Predicted future 20-day volatility: "
        f"{predicted_volatility:.6f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()