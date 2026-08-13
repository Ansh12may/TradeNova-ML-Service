import numpy as np
import pandas as pd


def create_stock_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the per-stock technical features used during V5/V6/V7 training.

    Required input columns:
        Date, Close, Volume
    """

    d = (
        stock_df
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------
    # Returns
    # -----------------------------
    d["return_1d"] = d["Close"].pct_change(1)
    d["return_5d"] = d["Close"].pct_change(5)
    d["return_20d"] = d["Close"].pct_change(20)

    # -----------------------------
    # Volatility
    # -----------------------------
    d["volatility_20d"] = (
        d["return_1d"]
        .rolling(20)
        .std()
    )

    # -----------------------------
    # Momentum
    # -----------------------------
    d["momentum_60d"] = d["Close"].pct_change(60)

    # -----------------------------
    # Moving averages
    # -----------------------------
    d["SMA_20"] = (
        d["Close"]
        .rolling(20)
        .mean()
    )

    d["SMA_50"] = (
        d["Close"]
        .rolling(50)
        .mean()
    )

    d["price_to_sma20"] = (
        d["Close"] / d["SMA_20"]
    )

    d["price_to_sma50"] = (
        d["Close"] / d["SMA_50"]
    )

    # -----------------------------
    # Drawdown
    # -----------------------------
    d["running_max"] = (
        d["Close"]
        .cummax()
    )

    d["drawdown"] = (
        d["Close"] / d["running_max"]
    ) - 1

    # -----------------------------
    # Volume features
    # -----------------------------
    d["volume_change"] = (
        d["Volume"]
        .pct_change()
    )

    vol_mean = (
        d["Volume"]
        .rolling(20)
        .mean()
    )

    vol_std = (
        d["Volume"]
        .rolling(20)
        .std()
    )

    d["volume_zscore"] = (
        (d["Volume"] - vol_mean)
        / vol_std
    )

    return d


def add_atr(
    stock_df: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """
    Calculate ATR and ATR as a percentage of price.

    Required input columns:
        High, Low, Close
    """

    d = stock_df.copy()

    previous_close = d["Close"].shift(1)

    tr1 = d["High"] - d["Low"]

    tr2 = (
        d["High"] - previous_close
    ).abs()

    tr3 = (
        d["Low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    d["ATR_14"] = (
        true_range
        .rolling(
            window=window,
            min_periods=window
        )
        .mean()
    )

    d["ATR_percent"] = (
        d["ATR_14"] / d["Close"]
    )

    return d


def add_volatility_dynamics(
    stock_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add volatility relationship and volatility-dynamics features.

    Requires:
        volatility_20d
        market_volatility_20d
    """

    d = stock_df.copy()

    # Difference between stock and market volatility
    d["relative_volatility"] = (
        d["volatility_20d"]
        - d["market_volatility_20d"]
    )

    # Stock volatility relative to market volatility
    d["volatility_ratio"] = (
        d["volatility_20d"]
        / d["market_volatility_20d"]
    )

    # 5-day change in volatility
    d["volatility_change"] = (
        d["volatility_20d"]
        .pct_change(5)
    )

    # Change in the change of volatility
    d["volatility_acceleration"] = (
        d["volatility_change"]
        .diff()
    )

    return d