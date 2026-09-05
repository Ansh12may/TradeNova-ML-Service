from __future__ import annotations
from typing import Any
import pandas as pd
from app.llm.explainer import generate_batch_explanations

from app.data.market_data import (
    get_stock_data,
    get_nifty_data,
)

from app.features.pipeline import (
    build_features_for_stock,
    select_model_features,
)

# IMPORTANT:
# Import the existing V6/V7 model stack BEFORE importing
# the PyTorch LSTM module. This avoids the native-library
# import-order issue observed on macOS.
from app.model_loader import (
    v6_model,
    v6_scaler,
    v7_model,
    v7_scaler,
    v6_features,
    v7_features,
    v6_stock_thresholds,
    v7_cluster_mapping,
)

from app.lstm_inference import (
    predict_future_volatility,
)

from app.risk.risk_engine import (
    calculate_volatility_risk,
    calculate_risk_score,
    classify_risk,
)

# CONFIGURATION

TRAIN_CUTOFF_DATE = pd.Timestamp(
    "2026-01-01"
)


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
    "AXISBANK.NS",
    "LT.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "BAJFINANCE.NS",
    "ADANIPORTS.NS",
    "HCLTECH.NS",
]

# V7 BASE FEATURES

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

# V7 PROFILE

def build_v7_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the stock-level behavioral profile
    required by the V7 clustering model.

    V7 expects mean and standard deviation
    statistics for each behavioral feature.
    """

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



# V7 BEHAVIOR


def get_v7_behavior(
    ticker: str,
    clean_df: pd.DataFrame,
):
    """
    Generate the stock-level V7 behavioral
    classification.
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



# V6 ANOMALY


def get_v6_anomaly(
    ticker: str,
    clean_df: pd.DataFrame,
):
    """
    Calculate the current V6 anomaly score.

    Uses the same decision_function-based
    logic as the original trained pipeline.
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



# SINGLE STOCK ML INFERENCE


def infer_stock(
    ticker: str,
    nifty_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run the complete ML inference pipeline
    for one stock.

    Models:
        LSTM -> future volatility
        V6   -> anomaly detection
        V7   -> behavioral profile

    This function does NOT calculate the
    final risk score.
    """

 
    # STOCK DATA


    stock_df = get_stock_data(
        ticker
    )

    if stock_df.empty:

        raise ValueError(
            f"No data returned for {ticker}"
        )


    # TRAINING CUTOFF
 

    train_cutoff_idx = int(
        (
            stock_df["Date"]
            < TRAIN_CUTOFF_DATE
        ).sum()
    )


    # EXISTING V6/V7 FEATURE PIPELINE
    #
    # IMPORTANT:
    # Keep the original pipeline contract.
  

    features_df = build_features_for_stock(
        stock_df=stock_df,
        nifty_df=nifty_df,
        train_cutoff_idx=train_cutoff_idx,
    )

    # ========================================================
    # REQUIRED FEATURES
    #
    # We retain the original V5 + V6 feature contract
    # because the existing feature pipeline was built around
    # these artifacts.
    #
    # V5 is no longer used for prediction.
    # LSTM has its own feature pipeline.
    # ========================================================

    required_features = sorted(
        set(
            list(
                v6_features
            )
            + list(
                v7_features
            )
            + V7_BASE_FEATURES
        )
    )

    # Only require features that are actually present.
    required_features = [
        feature
        for feature in required_features
        if feature in features_df.columns
    ]

    clean_df = features_df.dropna(
        subset=required_features
    ).copy()

    if clean_df.empty:

        raise ValueError(
            f"No valid feature rows for {ticker}"
        )

    latest = clean_df.iloc[-1]


    # LSTM — FUTURE 20-DAY VOLATILITY


    (
        lstm_prediction,
        lstm_prediction_date,
    ) = predict_future_volatility(
        ticker=ticker,
    )

   
    # V6 — ANOMALY DETECTION
 

    (
        anomaly_prediction,
        anomaly_score,
        anomaly_level,
    ) = get_v6_anomaly(
        ticker=ticker,
        clean_df=clean_df,
    )

    
    # V7 — BEHAVIORAL PROFILE
   

    (
        cluster,
        behavioral_type,
    ) = get_v7_behavior(
        ticker=ticker,
        clean_df=clean_df,
    )

   
    # STANDARDIZED ML OUTPUT
   

    return {
        "Ticker": ticker,

        "Date": lstm_prediction_date,

        
        # LSTM
    

        "volatility_forecast": float(
            lstm_prediction
        ),

      
        # V6
      

        "anomaly_prediction": (
            anomaly_prediction
        ),

        "anomaly_score": (
            anomaly_score
        ),

        "anomaly_level": (
            anomaly_level
        ),

        
        # V7
        

        "behavioral_cluster": (
            cluster
        ),

        "behavioral_type": (
            behavioral_type
        ),
    }



# BATCH RISK ANALYSIS


def run_batch_risk_analysis() -> list[dict[str, Any]]:
    """
    Run the complete TradeNova ML + risk
    pipeline for all supported stocks.
    """


    # LOAD NIFTY ONCE
  

    nifty_df = get_nifty_data()

    if nifty_df.empty:

        raise ValueError(
            "No NIFTY market data available."
        )

    results = []

    
    # PROCESS STOCKS
    

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
                f"Error processing {ticker}: {exc}"
            )

    # VALIDATE RESULTS
    if not results:

        raise ValueError(
            "Unable to generate risk analysis "
            "for any stock."
        )

    # DATAFRAME


    df = pd.DataFrame(
        results
    )

    # ========================================================
    # VOLATILITY RISK
    #
    # LSTM produces continuous volatility.
    #
    # Risk engine converts the predictions into
    # cross-sectional percentile risk.
    # ========================================================

    df["volatility_risk"] = (
        calculate_volatility_risk(
            df[
                "volatility_forecast"
            ]
        )
    )

    # FINAL RISK SCORE
    df["risk_score"] = df.apply(
        lambda row: calculate_risk_score(
            volatility_risk=float(
                row[
                    "volatility_risk"
                ]
            ),
            anomaly_level=str(
                row[
                    "anomaly_level"
                ]
            ),
            cluster=int(
                row[
                    "behavioral_cluster"
                ]
            ),
        ),
        axis=1,
    )


    # RISK CLASSIFICATION


    df["risk_level"] = (
        df[
            "risk_score"
        ].apply(
            classify_risk
        )
    )


    # SORT
   

    df = df.sort_values(
        "risk_score",
        ascending=False,
    )


    # JSON-SAFE OUTPUT


    return df.to_dict(
        orient="records"
    )


# SINGLE STOCK RISK


def get_risk_for_stock(
    ticker: str,
) -> dict[str, Any]:
    """
    Return the complete risk result for
    one supported stock.

    The complete universe is evaluated because
    volatility risk is cross-sectional.
    """

    ticker = ticker.upper().strip()

    if ticker not in STOCKS:

        raise ValueError(
            f"Unsupported ticker: {ticker}"
        )

    all_results = (
        run_batch_risk_analysis()
    )

    for result in all_results:

        if result["Ticker"] == ticker:

            return result

    raise ValueError(
        f"Risk result not available for {ticker}"
    )


# RISK EXPLANATION

def get_risk_explanation(
    ticker: str,
) -> dict[str, Any]:
    """
    Generate an LLM explanation for a single stock.

    The risk score and risk level are calculated
    deterministically by the ML/risk pipeline.

    The LLM only explains those results.
    """

    result = get_risk_for_stock(
        ticker=ticker,
    )

    result_df = pd.DataFrame(
        [result]
    )

    explanations = generate_batch_explanations(
        result_df
    )

    return {
        "Ticker": ticker,
        "risk_score": float(
            result["risk_score"]
        ),
        "risk_level": result["risk_level"],
        "explanation": explanations[0],
    }