from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# V6 — ANOMALY ADJUSTMENTS
# ============================================================

ANOMALY_ADJUSTMENT = {
    "NORMAL": 0,
    "ELEVATED": 10,
    "EXTREME": 20,
}


# ============================================================
# V7 — BEHAVIORAL ADJUSTMENTS
# ============================================================

CLUSTER_ADJUSTMENT = {
    0: 5,   # Higher Sensitivity / Higher Volatility
    1: 0,   # Lower Sensitivity / Lower Volatility
}


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(
    score: float,
) -> str:
    """
    Convert a deterministic risk score into
    the final risk category.

    Score ranges:

        0   - <25  -> LOW
        25  - <50  -> MODERATE
        50  - <75  -> HIGH
        75  - 100  -> VERY HIGH
    """

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# V5 — VOLATILITY RISK
# ============================================================

def calculate_volatility_risk(
    volatility_forecasts: pd.Series,
) -> pd.Series:
    """
    Convert V5 volatility forecasts into
    cross-sectional percentile risk scores.

    Original notebook logic:

        prediction.rank(pct=True) * 100

    The result is a 0–100 relative risk score
    across the stocks processed in the current batch.
    """

    if volatility_forecasts.empty:
        return pd.Series(
            dtype=float,
            index=volatility_forecasts.index,
        )

    return (
        volatility_forecasts
        .rank(pct=True)
        * 100
    )


# ============================================================
# V6 — STOCK-SPECIFIC ANOMALY CLASSIFICATION
# ============================================================

def classify_stock_anomaly(
    ticker: str,
    score: float,
    stock_thresholds: pd.DataFrame,
) -> str:
    """
    Classify an anomaly using stock-specific
    p95 and p99 thresholds.

    Threshold logic:

        score >= p99 -> EXTREME
        score >= p95 -> ELEVATED
        otherwise    -> NORMAL
    """

    if ticker not in stock_thresholds.index:

        raise ValueError(
            f"No V6 thresholds found for ticker: {ticker}"
        )

    p95 = float(
        stock_thresholds.loc[
            ticker,
            "p95",
        ]
    )

    p99 = float(
        stock_thresholds.loc[
            ticker,
            "p99",
        ]
    )

    if score >= p99:
        return "EXTREME"

    if score >= p95:
        return "ELEVATED"

    return "NORMAL"


# ============================================================
# FINAL RISK SCORE
# ============================================================

def calculate_risk_score(
    volatility_risk: float,
    anomaly_level: str,
    cluster: int,
) -> float:
    """
    Calculate the final deterministic risk score.

    Formula:

        risk_score =
            volatility_risk
            + anomaly_adjustment
            + behavioral_adjustment

    V6:

        NORMAL   -> +0
        ELEVATED -> +10
        EXTREME  -> +20

    V7:

        cluster 0 -> +5
        cluster 1 -> +0

    Final score is clipped to [0, 100].
    """

    anomaly_level = str(
        anomaly_level
    ).upper()

    cluster = int(
        cluster
    )

    anomaly_adjustment = (
        ANOMALY_ADJUSTMENT.get(
            anomaly_level,
            0,
        )
    )

    cluster_adjustment = (
        CLUSTER_ADJUSTMENT.get(
            cluster,
            0,
        )
    )

    score = (
        float(volatility_risk)
        + anomaly_adjustment
        + cluster_adjustment
    )

    return round(
        float(
            np.clip(
                score,
                0,
                100,
            )
        ),
        2,
    )


# ============================================================
# COMPLETE DETERMINISTIC RISK ENGINE
# ============================================================

def calculate_risk(
    volatility_forecasts: pd.Series,
    anomaly_levels: pd.Series,
    behavioral_clusters: pd.Series,
) -> pd.DataFrame:
    """
    Calculate deterministic risk for a batch of stocks.

    This function contains NO LLM logic.

    Parameters
    ----------
    volatility_forecasts:
        V5 volatility forecasts.

    anomaly_levels:
        V6 anomaly classifications:
        NORMAL / ELEVATED / EXTREME.

    behavioral_clusters:
        V7 cluster assignments.

    Returns
    -------
    DataFrame containing:

        volatility_forecast
        anomaly_level
        behavioral_cluster
        volatility_risk
        risk_score
        risk_level
    """

    result = pd.DataFrame(
        {
            "volatility_forecast": (
                volatility_forecasts
            ),
            "anomaly_level": (
                anomaly_levels
            ),
            "behavioral_cluster": (
                behavioral_clusters
            ),
        }
    ).copy()

    # ========================================================
    # V5 — VOLATILITY RISK
    # ========================================================

    result["volatility_risk"] = (
        calculate_volatility_risk(
            result[
                "volatility_forecast"
            ]
        )
    )

    # ========================================================
    # FINAL RISK SCORE
    # ========================================================

    result["risk_score"] = result.apply(
        lambda row: calculate_risk_score(
            volatility_risk=(
                row["volatility_risk"]
            ),
            anomaly_level=(
                row["anomaly_level"]
            ),
            cluster=(
                row["behavioral_cluster"]
            ),
        ),
        axis=1,
    )

    # ========================================================
    # FINAL RISK LEVEL
    # ========================================================

    result["risk_level"] = (
        result["risk_score"]
        .apply(
            classify_risk
        )
    )

    return result