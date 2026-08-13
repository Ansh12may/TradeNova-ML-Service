from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


# --------------------------------------------------
# Model artifacts
# --------------------------------------------------

V5_MODEL_PATH = (
    ARTIFACT_DIR /
    "v5_final_lightgbm_garch.joblib"
)

V5_FEATURES_PATH = (
    ARTIFACT_DIR /
    "v5_final_features.joblib"
)

V5_METADATA_PATH = (
    ARTIFACT_DIR /
    "v5_final_metadata.json"
)


V6_MODEL_PATH = (
    ARTIFACT_DIR /
    "v6_final_isolation_forest.joblib"
)

V6_SCALER_PATH = (
    ARTIFACT_DIR /
    "v6_final_scaler.joblib"
)

V6_FEATURES_PATH = (
    ARTIFACT_DIR /
    "v6_final_features.joblib"
)

V6_THRESHOLDS_PATH = (
    ARTIFACT_DIR /
    "v6_stock_thresholds.csv"
)

V6_METADATA_PATH = (
    ARTIFACT_DIR /
    "v6_final_metadata.json"
)


V7_MODEL_PATH = (
    ARTIFACT_DIR /
    "v7_final_kmeans.joblib"
)

V7_SCALER_PATH = (
    ARTIFACT_DIR /
    "v7_final_scaler.joblib"
)

V7_FEATURES_PATH = (
    ARTIFACT_DIR /
    "v7_final_features.joblib"
)

V7_MAPPING_PATH = (
    ARTIFACT_DIR /
    "v7_cluster_mapping.csv"
)

V7_METADATA_PATH = (
    ARTIFACT_DIR /
    "v7_final_metadata.json"
)


# --------------------------------------------------
# Load models
# --------------------------------------------------

v5_model = joblib.load(
    V5_MODEL_PATH
)

v6_model = joblib.load(
    V6_MODEL_PATH
)

v6_scaler = joblib.load(
    V6_SCALER_PATH
)

v7_model = joblib.load(
    V7_MODEL_PATH
)

v7_scaler = joblib.load(
    V7_SCALER_PATH
)


# --------------------------------------------------
# Load feature contracts
# --------------------------------------------------

v5_features = joblib.load(
    V5_FEATURES_PATH
)

v6_features = joblib.load(
    V6_FEATURES_PATH
)

v7_features = joblib.load(
    V7_FEATURES_PATH
)


# --------------------------------------------------
# Load additional artifacts
# --------------------------------------------------

v6_stock_thresholds = pd.read_csv(
    V6_THRESHOLDS_PATH,
    index_col="Ticker",
)

v7_cluster_mapping = pd.read_csv(
    V7_MAPPING_PATH
)


def get_model_info() -> dict:
    """
    Return basic information about the loaded
    production models.
    """

    return {
        "v5": {
            "model": type(v5_model).__name__,
            "features": len(v5_features),
        },
        "v6": {
            "model": type(v6_model).__name__,
            "features": len(v6_features),
            "threshold_rows": len(v6_stock_thresholds),
        },
        "v7": {
            "model": type(v7_model).__name__,
            "features": len(v7_features),
            "clusters": int(v7_model.n_clusters),
            "mapping_rows": len(v7_cluster_mapping),
        },
    }