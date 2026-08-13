from pathlib import Path

import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "WIPRO.NS",
    "TATAPOWER.NS",
    "M&M.NS",
    "HCLTECH.NS",
    "AXISBANK.NS",
    "LT.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "BAJFINANCE.NS",
    "ADANIPORTS.NS",
]

MARKET_INDEX = "^NSEI"


def download_market_data(
    start: str = "2019-01-01",
    end: str | None = None,
):
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tickers = STOCKS + [MARKET_INDEX]

    print(
        f"Downloading {len(STOCKS)} stocks + NIFTY..."
    )

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    output_path = (
        RAW_DATA_DIR /
        "nse_market_data.csv"
    )

    data.to_csv(output_path)

    print(
        f"Saved market data to: {output_path}"
    )

    return data


if __name__ == "__main__":
    download_market_data()