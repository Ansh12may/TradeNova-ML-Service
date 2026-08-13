from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "nse_market_data.csv"
)


def load_raw_market_data() -> pd.DataFrame:
    """
    Load the yfinance MultiIndex CSV.

    Expected structure:

        Ticker       RELIANCE.NS
        Price        Open High Low Close Volume
        Date
        2019-01-01   ...

    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns.
    """

    df = pd.read_csv(
        DATA_PATH,
        header=[0, 1],
        index_col=0,
    )

    df.index = pd.to_datetime(
        df.index,
        errors="coerce",
    )

    df = df[
        ~df.index.isna()
    ]

    return df


def get_stock_data(
    ticker: str,
) -> pd.DataFrame:
    """
    Extract one stock from the MultiIndex market dataset.

    Returns columns:

        Date
        Open
        High
        Low
        Close
        Volume
    """

    df = load_raw_market_data()

    if ticker not in df.columns.get_level_values(0):
        raise ValueError(
            f"Ticker '{ticker}' not found in market data."
        )

    stock = df[ticker].copy()

    stock = stock.rename_axis(
        "Date"
    ).reset_index()

    stock["Ticker"] = ticker

    return stock[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Ticker",
        ]
    ]


def get_nifty_data() -> pd.DataFrame:
    """
    Extract NIFTY (^NSEI) from the market dataset.

    Returns:

        Date
        Open
        High
        Low
        Close
        Volume
    """

    df = load_raw_market_data()

    ticker = "^NSEI"

    if ticker not in df.columns.get_level_values(0):
        raise ValueError(
            "NIFTY (^NSEI) is not present in "
            "the local market dataset."
        )

    nifty = df[ticker].copy()

    nifty = nifty.rename_axis(
        "Date"
    ).reset_index()

    return nifty[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]