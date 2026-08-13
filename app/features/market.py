import numpy as np
import pandas as pd


def create_market_features(
    nifty_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create NIFTY market-context features.

    Required input columns:
        Date, Close
    """

    m = (
        nifty_df
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------
    # Market returns
    # -----------------------------
    m["market_return_1d"] = (
        m["Close"].pct_change(1)
    )

    m["market_return_5d"] = (
        m["Close"].pct_change(5)
    )

    m["market_return_20d"] = (
        m["Close"].pct_change(20)
    )

    # -----------------------------
    # Market volatility
    # -----------------------------
    m["market_volatility_20d"] = (
        m["market_return_1d"]
        .rolling(20)
        .std()
    )

    # -----------------------------
    # Market drawdown
    # -----------------------------
    m["market_running_max"] = (
        m["Close"].cummax()
    )

    m["market_drawdown"] = (
        m["Close"]
        / m["market_running_max"]
    ) - 1

    return m[
        [
            "Date",
            "market_return_1d",
            "market_return_5d",
            "market_return_20d",
            "market_volatility_20d",
            "market_drawdown",
        ]
    ]


def add_rolling_beta(
    stock_df: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Calculate rolling stock beta relative to NIFTY.

    Beta = Cov(stock return, market return)
           / Var(market return)
    """

    d = stock_df.copy()

    d["stock_beta_60d"] = np.nan

    valid = (
        d["return_1d"].notna()
        & d["market_return_1d"].notna()
    )

    v = d.loc[
        valid,
        [
            "Date",
            "return_1d",
            "market_return_1d",
        ],
    ].copy()

    covariance = (
        v["return_1d"]
        .rolling(
            window,
            min_periods=window
        )
        .cov(v["market_return_1d"])
    )

    market_variance = (
        v["market_return_1d"]
        .rolling(
            window,
            min_periods=window
        )
        .var()
    )

    beta = covariance / market_variance

    d.loc[
        v.index,
        "stock_beta_60d"
    ] = beta.values

    return d