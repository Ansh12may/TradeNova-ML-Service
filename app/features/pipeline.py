import numpy as np
import pandas as pd

from .technical import (
    create_stock_features,
    add_atr,
    add_volatility_dynamics,
)

from .market import (
    create_market_features,
    add_rolling_beta,
)

from .garch import (
    fit_garch_and_forecast,
)


V3_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "momentum_60d",
    "price_to_sma20",
    "price_to_sma50",
    "drawdown",
    "volume_change",
    "volume_zscore",
    "market_return_1d",
    "market_return_5d",
    "market_return_20d",
    "market_volatility_20d",
    "market_drawdown",
    "stock_beta_60d",
    "relative_volatility",
    "volatility_ratio",
    "ATR_14",
    "ATR_percent",
    "volatility_change",
    "volatility_acceleration",
]


V5_FEATURES = V3_FEATURES + [
    "garch_vol_forecast"
]


V6_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "volatility_change",
    "volatility_acceleration",
    "volume_change",
    "volume_zscore",
    "ATR_percent",
    "drawdown",
    "relative_volatility",
    "volatility_ratio",
    "garch_vol_forecast",
]


V7_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "momentum_60d",
    "drawdown",
    "relative_volatility",
    "volatility_ratio",
    "ATR_percent",
    "volatility_change",
    "volatility_acceleration",
    "stock_beta_60d",
]


def build_features_for_stock(
    stock_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    train_cutoff_idx: int | None = None,
) -> pd.DataFrame:
    """
    Build the complete feature dataframe for one stock.

    This reproduces the V5/V6/V7 feature engineering pipeline.
    """

    # ---------------------------------
    # 1. Stock technical features
    # ---------------------------------

    d = create_stock_features(
        stock_df
    )

    # ---------------------------------
    # 2. Market features
    # ---------------------------------

    market = create_market_features(
        nifty_df
    )

    d = d.merge(
        market,
        on="Date",
        how="left",
    )

    # ---------------------------------
    # 3. Rolling beta
    # ---------------------------------

    d = add_rolling_beta(d)

    # ---------------------------------
    # 4. ATR
    # ---------------------------------

    d = add_atr(d)

    # ---------------------------------
    # 5. Volatility dynamics
    # ---------------------------------

    d = add_volatility_dynamics(d)

    # ---------------------------------
    # 6. GARCH
    # ---------------------------------

    if train_cutoff_idx is not None:

        d["garch_vol_forecast"] = (
            fit_garch_and_forecast(
                d["return_1d"].values,
                train_cutoff_idx,
            )
        )

    else:

        d["garch_vol_forecast"] = np.nan

    # ---------------------------------
    # 7. Clean infinite values
    # ---------------------------------

    d = d.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return d


def select_model_features(
    features_df: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    """
    Select features in the exact order expected by
    the trained model.
    """

    missing = [
        feature
        for feature in feature_list
        if feature not in features_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    return features_df[
        feature_list
    ].copy()