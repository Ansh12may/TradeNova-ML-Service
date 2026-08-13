import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch_and_forecast(
    returns: np.ndarray,
    train_cutoff_idx: int,
) -> np.ndarray:
    """
    Fit GARCH(1,1) using only the training portion of returns
    and generate conditional volatility across the full series.

    This is the same methodology used during V5 training.

    Parameters
    ----------
    returns:
        Daily returns in decimal form.

    train_cutoff_idx:
        Number of observations belonging to the historical
        training period.

    Returns
    -------
    np.ndarray
        GARCH volatility forecast in decimal return units.
    """

    returns = np.asarray(
        returns,
        dtype=float
    )

    n = len(returns)

    # arch works more stably with percentage returns
    returns_pct = returns * 100

    # ---------------------------------
    # Training slice
    # ---------------------------------

    clean_train = returns_pct[:train_cutoff_idx]

    clean_train = clean_train[
        ~np.isnan(clean_train)
    ]

    # Need sufficient historical observations
    if len(clean_train) < 250:
        return np.full(
            n,
            np.nan
        )

    # ---------------------------------
    # Fit GARCH(1,1)
    # ---------------------------------

    try:

        model = arch_model(
            clean_train,
            vol="Garch",
            p=1,
            q=1,
            dist="normal",
            mean="Zero",
        )

        result = model.fit(
            disp="off"
        )

        omega = result.params["omega"]
        alpha = result.params["alpha[1]"]
        beta = result.params["beta[1]"]

    except Exception:
        return np.full(
            n,
            np.nan
        )

    # ---------------------------------
    # Conditional volatility recursion
    # ---------------------------------

    eps = np.nan_to_num(
        returns_pct,
        nan=0.0
    )

    sigma2 = np.zeros(n)

    sigma2[0] = np.nanvar(
        clean_train
    )

    for t in range(1, n):

        sigma2[t] = (
            omega
            + alpha * eps[t - 1] ** 2
            + beta * sigma2[t - 1]
        )

    # ---------------------------------
    # Convert back to decimal units
    # ---------------------------------

    volatility_pct = np.sqrt(
        sigma2
    )

    return volatility_pct / 100