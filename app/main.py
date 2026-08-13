from fastapi import FastAPI, HTTPException
from app.api.routes import router
from app.services.risk_service import (
    get_risk_for_stock,
    run_batch_risk_analysis,
)


app = FastAPI(
    title="TradeNova ML Service",
    description="ML inference and risk analysis service for TradeNova",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():

    return {
        "service": "TradeNova ML Service",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


