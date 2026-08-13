from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "nse_market_data.csv"


def inspect_data():
    df = pd.read_csv(DATA_PATH, header=[0, 1], index_col=0)

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== COLUMNS ==========")
    print(df.columns)

    print("\n========== MISSING VALUES ==========")
    print(df.isna().sum().head(30))

    print("\n========== DATE RANGE ==========")
    print(df.index.min())
    print(df.index.max())


if __name__ == "__main__":
    inspect_data()