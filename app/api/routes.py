from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.risk import (
    RiskResponse,
    BatchRiskResponse,
)
from app.schemas.risk import (
    RiskExplanationResponse,

)

from app.services.risk_service import (

    get_risk_explanation,

)

from app.services.risk_service import (
    get_risk_for_stock,
    run_batch_risk_analysis,
)


router = APIRouter(
    prefix="/api",
    tags=["Risk"],
)


# ============================================================
# SINGLE STOCK
# ============================================================

@router.get(
    "/risk/{ticker}",
    response_model=RiskResponse,
)
def get_stock_risk(
    ticker: str,
):
    """
    Calculate deterministic risk
    for a single stock.
    """

    try:

        result = get_risk_for_stock(
            ticker=ticker,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Risk calculation failed: {exc}",
        )


# ============================================================
# BATCH RISK
# ============================================================

@router.get(
    "/risk",
    response_model=BatchRiskResponse,
)
def get_batch_risk():
    """
    Calculate deterministic risk
    for the complete TradeNova universe.
    """

    try:

        stocks = run_batch_risk_analysis()

        return {
            "stocks": stocks,
            "total": len(stocks),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Batch risk calculation failed: {exc}",
        )


@router.get(
    "/risk/{ticker}/explanation",
    response_model=RiskExplanationResponse,
)
def get_stock_risk_explanation(
    ticker: str,
):

    try:

        return get_risk_explanation(
            ticker=ticker,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate explanation: {exc}",
        )