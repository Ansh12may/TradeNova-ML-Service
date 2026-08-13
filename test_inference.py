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
    v7_cluster_mapping,
)


TICKER = "RELIANCE.NS"

TRAIN_CUTOFF_DATE = pd.Timestamp(
    "2026-01-01"
)


# --------------------------------------------------
# V7 behavioral-profile features
# --------------------------------------------------
#
# IMPORTANT:
# V7 was NOT trained on individual daily observations.
#
# It was trained on:
#
#     12 features
#         ×
#     mean + std
#
#     = 24 features
#
# These must match v7_scaler.feature_names_in_
# exactly.
# --------------------------------------------------

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


def build_v7_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the stock-level behavioral profile
    expected by the V7 scaler.

    For every V7 base feature:

        feature_mean
        feature_std

    are calculated across the historical observations.
    """

    profile = (
        df[V7_BASE_FEATURES]
        .agg(["mean", "std"])
    )

    profile_row = {}

    for feature in V7_BASE_FEATURES:

        profile_row[
            f"{feature}_mean"
        ] = profile.loc[
            "mean",
            feature,
        ]

        profile_row[
            f"{feature}_std"
        ] = profile.loc[
            "std",
            feature,
        ]

    profile_df = pd.DataFrame(
        [profile_row]
    )

    # ----------------------------------------------
    # Verify exact V7 feature contract
    # ----------------------------------------------

    expected_features = list(
        v7_scaler.feature_names_in_
    )

    missing = [
        feature
        for feature in expected_features
        if feature not in profile_df.columns
    ]

    extra = [
        feature
        for feature in profile_df.columns
        if feature not in expected_features
    ]

    if missing:
        raise ValueError(
            "Missing V7 features:\n"
            + "\n".join(missing)
        )

    if extra:
        raise ValueError(
            "Unexpected V7 features:\n"
            + "\n".join(extra)
        )

    # Force exact training order.
    profile_df = profile_df[
        expected_features
    ]

    return profile_df


def main():

    print("=" * 70)
    print(
        "TradeNova ML — Local End-to-End Inference Test"
    )
    print("=" * 70)

    # ==================================================
    # 1. LOAD MARKET DATA
    # ==================================================

    print(
        "\n[1] Loading market data..."
    )

    stock_df = get_stock_data(
        TICKER
    )

    nifty_df = get_nifty_data()

    print(
        f"Stock rows: {len(stock_df)}"
    )

    print(
        f"NIFTY rows: {len(nifty_df)}"
    )

    # ==================================================
    # 2. GARCH TRAINING CUTOFF
    # ==================================================

    train_cutoff_idx = int(
        (
            stock_df["Date"]
            < TRAIN_CUTOFF_DATE
        ).sum()
    )

    print(
        "\n[2] GARCH training cutoff:"
        f" {TRAIN_CUTOFF_DATE.date()}"
    )

    print(
        f"Training observations:"
        f" {train_cutoff_idx}"
    )

    # ==================================================
    # 3. BUILD FEATURES
    # ==================================================

    print(
        "\n[3] Building feature pipeline..."
    )

    features_df = build_features_for_stock(
        stock_df=stock_df,
        nifty_df=nifty_df,
        train_cutoff_idx=train_cutoff_idx,
    )

    print(
        "Feature dataframe shape:",
        features_df.shape,
    )

    print(
        "\nAvailable features:"
    )

    print(
        features_df.columns.tolist()
    )

    # ==================================================
    # 4. CLEAN DATA FOR V5/V6
    # ==================================================

    required_features = sorted(
        set(
            v5_features
            + v6_features
        )
    )

    clean_df = features_df.dropna(
        subset=required_features
    ).copy()

    print(
        "\n[4] Clean rows:",
        len(clean_df)
    )

    if clean_df.empty:
        raise RuntimeError(
            "No valid rows remain after "
            "feature engineering."
        )

    # ==================================================
    # 5. LATEST OBSERVATION
    # ==================================================

    latest = clean_df.iloc[-1]

    latest_date = latest["Date"]

    print(
        "\n[5] Latest valid observation:"
    )

    print(
        "Date:",
        latest_date
    )

    print(
        "Close:",
        latest["Close"]
    )

    # ==================================================
    # 6. V5 — VOLATILITY FORECAST
    # ==================================================

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

    print(
        "\n[6] V5 VOLATILITY"
    )

    print(
        "Prediction:",
        v5_prediction
    )

    # ==================================================
    # 7. V6 — ANOMALY DETECTION
    # ==================================================

    X_v6 = select_model_features(
        clean_df,
        v6_features,
    )

    latest_X_v6 = X_v6.iloc[
        [-1]
    ]

    X_v6_scaled = v6_scaler.transform(
        latest_X_v6
    )

    anomaly_prediction = int(
        v6_model.predict(
            X_v6_scaled
        )[0]
    )

    anomaly_score = float(
        -v6_model.score_samples(
            X_v6_scaled
        )[0]
    )

    print(
        "\n[7] V6 ANOMALY"
    )

    print(
        "Prediction:",
        anomaly_prediction
    )

    print(
        "Anomaly score:",
        anomaly_score
    )

    # ==================================================
    # 8. V7 — BEHAVIORAL PROFILE
    # ==================================================

    print(
        "\n[8] V7 BEHAVIORAL PROFILE"
    )

    v7_profile = build_v7_profile(
        clean_df
    )

    print(
        "V7 profile shape:",
        v7_profile.shape
    )

    print(
        "Expected:",
        len(
            v7_scaler.feature_names_in_
        ),
        "features"
    )

    # Scale using the exact scaler from training.

    X_v7_scaled = v7_scaler.transform(
        v7_profile
    )

    # Predict cluster.

    cluster = int(
        v7_model.predict(
            X_v7_scaled
        )[0]
    )

    print(
        "Cluster:",
        cluster
    )

    # ==================================================
    # 9. V7 STOCK MAPPING
    # ==================================================

    mapping = v7_cluster_mapping[
        v7_cluster_mapping["Ticker"]
        == TICKER
    ]

    behavioral_type = None

    if not mapping.empty:

        mapping_row = mapping.iloc[0]

        print(
            "\nBehavioral mapping:"
        )

        print(
            mapping_row.to_dict()
        )

        if "behavioral_type" in mapping.columns:
            behavioral_type = (
                mapping_row["behavioral_type"]
            )

    else:

        print(
            f"\nNo mapping found for {TICKER}"
        )

    # ==================================================
    # 10. FINAL RESULT
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL INFERENCE RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Ticker:",
        TICKER
    )

    print(
        "Date:",
        latest_date
    )

    print(
        "Close:",
        latest["Close"]
    )

    print(
        "V5 volatility prediction:",
        round(
            v5_prediction,
            6,
        )
    )

    print(
        "V6 anomaly prediction:",
        anomaly_prediction
    )

    print(
        "V6 anomaly score:",
        round(
            anomaly_score,
            6,
        )
    )

    print(
        "V7 cluster:",
        cluster
    )

    print(
        "V7 behavioral type:",
        behavioral_type
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()