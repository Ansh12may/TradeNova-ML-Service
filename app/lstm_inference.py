from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from app.data.market_data import get_nifty_data, get_stock_data
from app.lstm_model import LSTMRegressor

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

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts" / "lstm"

MODEL_PATH = ARTIFACT_DIR / "model.pth"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"



def _load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"LSTM scaler not found: {SCALER_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"LSTM metadata not found: {METADATA_PATH}"
        )

    model = LSTMRegressor(
        input_size=len(SEQUENCE_FEATURES),
        hidden_size=HIDDEN_SIZE,
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, scaler, metadata


# Lazy loading.
# This prevents FastAPI from loading the model unnecessarily
# during module import.
_lstm_model = None
_lstm_scaler = None
_lstm_metadata = None


def _ensure_model_loaded():

    global _lstm_model
    global _lstm_scaler
    global _lstm_metadata

    if _lstm_model is None:

        (
            _lstm_model,
            _lstm_scaler,
            _lstm_metadata,
        ) = _load_model()

    return (
        _lstm_model,
        _lstm_scaler,
        _lstm_metadata,
    )


# FEATURE ENGINEERING

def _build_lstm_features(
    stock_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
) -> pd.DataFrame:

    df = stock_df.copy()

    df = df.sort_values("Date").reset_index(drop=True)


    # Stock returns
   

    df["return_1d"] = (
        df["Close"].pct_change(fill_method=None)
    )

    df["return_5d"] = (
        df["Close"].pct_change(
            5,
            fill_method=None,
        )
    )

    df["return_20d"] = (
        df["Close"].pct_change(
            20,
            fill_method=None,
        )
    )

    df["return_60d"] = (
        df["Close"].pct_change(
            60,
            fill_method=None,
        )
    )


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

  
    # Trend
   

    sma20 = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    sma50 = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["price_to_sma20"] = (
        df["Close"] / sma20 - 1
    )

    df["price_to_sma50"] = (
        df["Close"] / sma50 - 1
    )

  
    # Drawdown
  
    rolling_peak = (
        df["Close"]
        .rolling(60)
        .max()
    )

    df["drawdown_60d"] = (
        df["Close"] / rolling_peak - 1
    )

  
    # NIFTY market volatility
   

    nifty = nifty_df.copy()

    nifty = (
        nifty
        .sort_values("Date")
        .reset_index(drop=True)
    )

    nifty["market_return_1d"] = (
        nifty["Close"].pct_change(
            fill_method=None
        )
    )

    nifty["market_volatility_20d"] = (
        nifty["market_return_1d"]
        .rolling(20)
        .std()
    )

    df = df.merge(
        nifty[
            [
                "Date",
                "market_volatility_20d",
            ]
        ],
        on="Date",
        how="left",
    )

    return df



# PREDICTION


def predict_future_volatility(
    ticker: str,
):
    """
    Predict future realized volatility for the next
    20 trading days using the trained LSTM.

    Returns
    -------
    prediction : float
        Predicted future 20-day realized volatility.

    prediction_date : pandas.Timestamp
        Latest date used by the model.
    """

    (
        model,
        scaler,
        metadata,
    ) = _ensure_model_loaded()

  
    # Validate model contract
   

    if metadata["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError(
            "LSTM sequence length does not match metadata."
        )

    if metadata["sequence_features"] != SEQUENCE_FEATURES:
        raise ValueError(
            "LSTM feature contract does not match metadata."
        )

    if metadata["future_horizon"] != 20:
        raise ValueError(
            "Unexpected LSTM forecast horizon."
        )

    
    # Load data
   

    stock_df = get_stock_data(ticker)
    nifty_df = get_nifty_data()

    if stock_df.empty:
        raise ValueError(
            f"No stock data available for {ticker}"
        )

    if nifty_df.empty:
        raise ValueError(
            "No NIFTY data available."
        )

  
    # Feature engineering
    

    features_df = _build_lstm_features(
        stock_df,
        nifty_df,
    )

    clean_df = (
        features_df
        .dropna(subset=SEQUENCE_FEATURES)
        .copy()
    )

    if len(clean_df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Not enough data for {ticker}. "
            f"Need {SEQUENCE_LENGTH} valid rows, "
            f"got {len(clean_df)}."
        )

  
    # Latest 120-day sequence
   

    latest_window = (
        clean_df[
            SEQUENCE_FEATURES
        ]
        .tail(SEQUENCE_LENGTH)
        .values
    )

    # Shape:
    # (120, 9)

    scaled_window = scaler.transform(
        latest_window
    )

    # Shape:
    # (1, 120, 9)

    X = torch.tensor(
        scaled_window,
        dtype=torch.float32,
    ).unsqueeze(0)

    
    # Inference
 

    with torch.no_grad():

        prediction = model(X)

    predicted_volatility = float(
        prediction.item()
    )

    prediction_date = (
        clean_df["Date"].iloc[-1]
    )

    return (
        predicted_volatility,
        prediction_date,
    )