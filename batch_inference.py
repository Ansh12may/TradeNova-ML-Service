from __future__ import annotations

import pandas as pd

from app.data.market_data import (
    get_stock_data,
    get_nifty_data,
)

from app.features.pipeline import (
    build_features_for_stock,
    select_model_features,
)

from app.model_loader import (
    v5_model,
    v6_model,
    v6_scaler,
    v7_model,
    v7_scaler,
    v5_features,
    v6_features,
    v6_stock_thresholds,
    v7_cluster_mapping,
)

from app.risk.risk_engine import (
    calculate_volatility_risk,
    calculate_risk_score,
    classify_risk,
)

from app.llm.explainer import (
    generate_batch_explanations,
)


# ============================================================
# CONFIGURATION
# ============================================================

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

TRAIN_CUTOFF_DATE = pd.Timestamp(
    "2026-01-01"
)


# ============================================================
# V7 BASE FEATURES
# ============================================================

V7_BASE_FEATURES = [
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


# ============================================================
# V7 PROFILE
# ============================================================

def build_v7_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    profile = (
        df[V7_BASE_FEATURES]
        .agg(["mean", "std"])
    )

    row = {}

    for feature in V7_BASE_FEATURES:

        row[
            f"{feature}_mean"
        ] = profile.loc[
            "mean",
            feature,
        ]

        row[
            f"{feature}_std"
        ] = profile.loc[
            "std",
            feature,
        ]

    profile_df = pd.DataFrame(
        [row]
    )

    expected = list(
        v7_scaler.feature_names_in_
    )

    missing = [
        col
        for col in expected
        if col not in profile_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing V7 features: {missing}"
        )

    return profile_df[
        expected
    ]


# ============================================================
# V7 BEHAVIOR
# ============================================================

def get_v7_behavior(
    ticker: str,
    clean_df: pd.DataFrame,
):
    """
    Generate the stock-level V7 behavioral classification.
    """

    profile = build_v7_profile(
        clean_df
    )

    scaled = v7_scaler.transform(
        profile
    )

    cluster = int(
        v7_model.predict(
            scaled
        )[0]
    )

    mapping = v7_cluster_mapping[
        v7_cluster_mapping["Ticker"]
        == ticker
    ]

    behavioral_type = None

    if not mapping.empty:

        behavioral_type = mapping.iloc[0][
            "behavioral_type"
        ]

    return (
        cluster,
        behavioral_type,
    )


# ============================================================
# V6 ANOMALY
# ============================================================

def get_v6_anomaly(
    ticker: str,
    clean_df: pd.DataFrame,
):
    """
    Calculate the current V6 anomaly score.

    Uses the same decision_function-based logic
    as the original trained pipeline.
    """

    X_v6 = select_model_features(
        clean_df,
        v6_features,
    )

    latest_X = X_v6.iloc[
        [-1]
    ]

    scaled = v6_scaler.transform(
        latest_X
    )

    prediction = int(
        v6_model.predict(
            scaled
        )[0]
    )

    score = float(
        -v6_model.decision_function(
            scaled
        )[0]
    )

    if ticker not in v6_stock_thresholds.index:

        raise ValueError(
            f"No V6 thresholds for {ticker}"
        )

    p95 = float(
        v6_stock_thresholds.loc[
            ticker,
            "p95",
        ]
    )

    p99 = float(
        v6_stock_thresholds.loc[
            ticker,
            "p99",
        ]
    )

    if score >= p99:

        level = "EXTREME"

    elif score >= p95:

        level = "ELEVATED"

    else:

        level = "NORMAL"

    return (
        prediction,
        score,
        level,
    )


# ============================================================
# SINGLE STOCK INFERENCE
# ============================================================

def infer_stock(
    ticker: str,
    nifty_df: pd.DataFrame,
):
    """
    Run V5, V6 and V7 inference for one stock.

    This function produces deterministic ML outputs only.
    The LLM is invoked later after the final risk score
    has been calculated.
    """

    print(
        f"\nProcessing {ticker}..."
    )

    stock_df = get_stock_data(
        ticker
    )

    if stock_df.empty:

        raise ValueError(
            f"No data returned for {ticker}"
        )

    train_cutoff_idx = int(
        (
            stock_df["Date"]
            < TRAIN_CUTOFF_DATE
        ).sum()
    )

    features_df = build_features_for_stock(
        stock_df=stock_df,
        nifty_df=nifty_df,
        train_cutoff_idx=train_cutoff_idx,
    )

    required_features = sorted(
        set(
            v5_features
            + v6_features
        )
    )

    clean_df = features_df.dropna(
        subset=required_features
    ).copy()

    if clean_df.empty:

        raise ValueError(
            f"No valid feature rows for {ticker}"
        )

    latest = clean_df.iloc[-1]

    # ========================================================
    # V5 — VOLATILITY FORECAST
    # ========================================================

    X_v5 = select_model_features(
        clean_df,
        v5_features,
    )

    latest_X_v5 = X_v5.iloc[
        [-1]
    ]

    v5_prediction = float(
        v5_model.predict(
            latest_X_v5
        )[0]
    )

    # ========================================================
    # V6 — ANOMALY DETECTION
    # ========================================================

    (
        anomaly_prediction,
        anomaly_score,
        anomaly_level,
    ) = get_v6_anomaly(
        ticker=ticker,
        clean_df=clean_df,
    )

    # ========================================================
    # V7 — BEHAVIORAL PROFILE
    # ========================================================

    (
        cluster,
        behavioral_type,
    ) = get_v7_behavior(
        ticker=ticker,
        clean_df=clean_df,
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print(
        f"  Date: {latest['Date'].date()}"
    )

    print(
        f"  V5: {v5_prediction:.6f}"
    )

    print(
        f"  V6 score: {anomaly_score:.6f}"
    )

    print(
        f"  V6 level: {anomaly_level}"
    )

    print(
        f"  V7 cluster: {cluster}"
    )

    print(
        f"  Behavior: {behavioral_type}"
    )

    # ========================================================
    # STANDARDIZED ML OUTPUT CONTRACT
    # ========================================================

    return {
        "Ticker": ticker,
        "Date": latest["Date"],

        # V5
        "volatility_forecast": v5_prediction,

        # V6
        "anomaly_prediction": anomaly_prediction,
        "anomaly_score": anomaly_score,
        "anomaly_level": anomaly_level,

        # V7
        "behavioral_cluster": cluster,
        "behavioral_type": behavioral_type,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TradeNova ML — 20 Stock Batch Inference"
    )

    print("=" * 70)

    print(
        f"\nUniverse size: {len(STOCKS)}"
    )

    # ========================================================
    # LOAD NIFTY ONCE
    # ========================================================

    print(
        "\nLoading NIFTY..."
    )

    nifty_df = get_nifty_data()

    print(
        f"NIFTY rows: {len(nifty_df)}"
    )

    # ========================================================
    # RUN V5 / V6 / V7
    # ========================================================

    results = []

    for ticker in STOCKS:

        try:

            result = infer_stock(
                ticker=ticker,
                nifty_df=nifty_df,
            )

            results.append(
                result
            )

        except Exception as exc:

            print(
                f"  ERROR: {ticker}: {exc}"
            )

    if not results:

        raise RuntimeError(
            "No stocks were successfully processed."
        )

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # V5 CROSS-SECTIONAL RISK
    # ========================================================

    print(
        "\nCalculating V5 volatility risk..."
    )

    df["volatility_risk"] = (
        calculate_volatility_risk(
            df["volatility_forecast"]
        )
    )

    # ========================================================
    # DETERMINISTIC RISK ENGINE
    # ========================================================

    print(
        "Calculating final risk..."
    )

    df["risk_score"] = df.apply(
        lambda row: calculate_risk_score(
            volatility_risk=row[
                "volatility_risk"
            ],
            anomaly_level=row[
                "anomaly_level"
            ],
            cluster=row[
                "behavioral_cluster"
            ],
        ),
        axis=1,
    )

    df["risk_level"] = (
        df["risk_score"]
        .apply(
            classify_risk
        )
    )

    # ========================================================
    # SORT BEFORE LLM
    # ========================================================

    df = df.sort_values(
        "risk_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # ========================================================
    # GROQ EXPLANATION LAYER
    # ========================================================

    print(
        "\nGenerating AI risk explanations..."
    )

    print(
        "The LLM explains the deterministic ML output."
    )

    print(
        "The LLM does NOT calculate or modify risk."
    )

    df["risk_explanation"] = (
        generate_batch_explanations(
            df
        )
    )

    # ========================================================
    # FINAL SNAPSHOT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL 20-STOCK RISK SNAPSHOT"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "Ticker",
        "Date",
        "volatility_forecast",
        "volatility_risk",
        "anomaly_score",
        "anomaly_level",
        "behavioral_cluster",
        "behavioral_type",
        "risk_score",
        "risk_level",
    ]

    print(
        df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # AI EXPLANATIONS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "AI RISK EXPLANATIONS"
    )

    print(
        "=" * 70
    )

    for _, row in df.iterrows():

        print(
            f"\n{row['Ticker']}"
        )

        print(
            f"Risk: "
            f"{row['risk_score']:.2f} "
            f"({row['risk_level']})"
        )

        print(
            f"Explanation: "
            f"{row['risk_explanation']}"
        )

        print(
            "-" * 70
        )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        df[
            "risk_level"
        ].value_counts()
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        "batch_risk_snapshot.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: {output_path}"
    )

    # ========================================================
    # FINAL COLUMN CHECK
    # ========================================================

    print(
        "\nFinal columns:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()