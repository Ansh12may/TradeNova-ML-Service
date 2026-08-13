from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class RiskResponse(BaseModel):
    Ticker: str
    Date: datetime
    volatility_forecast: float
    anomaly_prediction: int
    anomaly_score: float
    anomaly_level: str
    behavioral_cluster: int
    behavioral_type: str | None
    volatility_risk: float
    risk_score: float
    risk_level: str

class BatchRiskResponse(BaseModel):
    stocks: list[RiskResponse]
    total: int

class RiskExplanationResponse(BaseModel):
    Ticker: str
    risk_score: float
    risk_level: str
    explanation: str