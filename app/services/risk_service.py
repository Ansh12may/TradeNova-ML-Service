from __future__ import annotations
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


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CUTOFF_DATE = pd.Timestamp(
    "2026-01-01"
)


# ============================================================
# TRADE NOVA STOCK UNIVERSE
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


# ============================================================
# V7 BEHAVIOR
# ============================================================

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


# ============================================================
# V6 ANOMALY
# ============================================================

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


# ============================================================
# SINGLE STOCK ML INFERENCE
# ============================================================

def infer_stock(
    ticker: str,
    nifty_df: pd.DataFrame,
):
    """
    Run V5, V6 and V7 inference for one stock.

    This function produces deterministic ML
    outputs only.

    It does NOT calculate the final risk score.
    """

    stock_df = get_stock_data(
        ticker
    )

    if stock_df.empty:

        raise ValueError(
            f"No data returned for {ticker}"
        )

    # ========================================================
    # TRAINING CUTOFF
    # ========================================================

    train_cutoff_idx = int(
        (
            stock_df["Date"]
            < TRAIN_CUTOFF_DATE
        ).sum()
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    features_df = build_features_for_stock(
        stock_df=stock_df,
        nifty_df=nifty_df,
        train_cutoff_idx=train_cutoff_idx,
    )

    # ========================================================
    # REQUIRED FEATURES
    # ========================================================

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
    # RETURN STANDARDIZED ML OUTPUT
    # ========================================================

    return {
        "Ticker": ticker,

        "Date": latest[
            "Date"
        ],

        # ------------------------
        # V5
        # ------------------------

        "volatility_forecast": (
            v5_prediction
        ),

        # ------------------------
        # V6
        # ------------------------

        "anomaly_prediction": (
            anomaly_prediction
        ),

        "anomaly_score": (
            anomaly_score
        ),

        "anomaly_level": (
            anomaly_level
        ),

        # ------------------------
        # V7
        # ------------------------

        "behavioral_cluster": (
            cluster
        ),

        "behavioral_type": (
            behavioral_type
        ),
    }


# ============================================================
# BATCH RISK ANALYSIS
# ============================================================

def run_batch_risk_analysis():
    """
    Run the complete deterministic risk pipeline
    for the TradeNova stock universe.

    Pipeline:

        NIFTY
          ↓
        V5
        V6
        V7
          ↓
        Cross-sectional V5 risk
          ↓
        Final deterministic risk score
          ↓
        Risk classification

    Returns
    -------
    pd.DataFrame
        Final risk snapshot for all successfully
        processed stocks.
    """

    # ========================================================
    # LOAD MARKET INDEX ONCE
    # ========================================================

    nifty_df = get_nifty_data()

    if nifty_df.empty:

        raise RuntimeError(
            "Unable to load NIFTY market data."
        )

    # ========================================================
    # RUN ML INFERENCE
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
                f"ERROR processing "
                f"{ticker}: {exc}"
            )

    # ========================================================
    # VALIDATION
    # ========================================================

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

    df[
        "volatility_risk"
    ] = calculate_volatility_risk(
        df[
            "volatility_forecast"
        ]
    )

    # ========================================================
    # FINAL DETERMINISTIC RISK SCORE
    # ========================================================

    df[
        "risk_score"
    ] = df.apply(
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

    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    df[
        "risk_level"
    ] = df[
        "risk_score"
    ].apply(
        classify_risk
    )

    # ========================================================
    # SORT BY RISK
    # ========================================================

    df = df.sort_values(
        "risk_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# SINGLE STOCK RISK
# ============================================================

def get_risk_for_stock(
    ticker: str,
):
    """
    Get the deterministic risk result for one stock.

    IMPORTANT:
    V5 volatility risk is cross-sectional.

    Therefore we must first calculate the risk
    for the complete TradeNova universe and then
    select the requested ticker.

    This guarantees that:

        /risk/TCS.NS

    uses the same risk calculation as:

        /risk/batch
    """

    ticker = ticker.upper()

    # ========================================================
    # VALIDATE TICKER
    # ========================================================

    if ticker not in STOCKS:

        raise ValueError(
            f"Unsupported ticker: {ticker}"
        )

    # ========================================================
    # RUN COMPLETE UNIVERSE
    # ========================================================

    df = run_batch_risk_analysis()

    # ========================================================
    # FIND REQUESTED STOCK
    # ========================================================

    result = df[
        df["Ticker"] == ticker
    ]

    if result.empty:

        raise ValueError(
            f"No risk result found for {ticker}"
        )

    # ========================================================
    # RETURN SINGLE STOCK
    # ========================================================

    return result.iloc[
        0
    ].to_dict()





def get_risk_explanation(
    ticker: str,
):
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