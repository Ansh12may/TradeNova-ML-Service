from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parents[2]

load_dotenv(
    BASE_DIR / ".env"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = (
    "llama-3.3-70b-versatile"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are the risk explanation assistant for TradeNova.

Your ONLY responsibility is to explain deterministic quantitative
risk results produced by TradeNova's machine-learning pipeline.

TradeNova uses:

- V5: volatility forecasting
- V6: anomaly detection
- V7: behavioral clustering
- Risk Engine: deterministic combination of V5, V6 and V7 signals

IMPORTANT ARCHITECTURE RULE:

The Risk Engine has already calculated the final risk score and
risk level.

You are an EXPLANATION layer only.

You MUST NOT calculate, modify, override, reinterpret, or contradict
the supplied risk score or risk level.

STRICT RULES:

1. Never change any supplied numerical value.

2. Never calculate a different risk score.

3. Never change the supplied risk level.

4. Never invent prices, returns, news, fundamentals, statistics,
   market events, or other information.

5. Never make a buy, sell, hold, trading, portfolio, or investment
   recommendation.

6. Never tell the user what action they should take.

7. Never predict future stock prices or future market movements.

8. Explain WHY the supplied deterministic risk classification
   was produced.

9. Clearly distinguish the three ML signals:
   - V5 = forecast volatility
   - V6 = anomaly detection
   - V7 = historical behavioral clustering

10. IMPORTANT:
    "volatility_risk" is a CROSS-SECTIONAL percentile within the
    current TradeNova stock universe being evaluated.

    For example:
    - 100 means the stock has the highest V5 forecast among the
      stocks in the current batch.
    - 50 means approximately the middle of the current batch.
    - 5 means the stock has a very low V5 forecast relative to
      the current batch.

    Do NOT describe this percentile as a historical percentile
    unless that information is explicitly supplied.

11. V7 behavioral profile is historical/contextual information.
    It describes the cluster to which the stock belongs based on
    its learned behavioral characteristics.

12. If anomaly_level is NORMAL, explicitly state that the anomaly
    detector did not identify unusually abnormal behavior.

13. If anomaly_level is ELEVATED or EXTREME, describe that only
    according to the supplied anomaly level. Do not invent causes.

14. Explain which supplied signals are the main contributors to
    the existing risk score.

15. Do not claim that V7 directly changed the risk score unless
    the supplied cluster corresponds to a known Risk Engine
    adjustment.

16. Keep the explanation concise, precise, and suitable for a
    financial-risk dashboard.

17. Use professional quantitative-finance and risk-management
    terminology.

18. Do not use phrases such as:
    - "investors should..."
    - "you should..."
    - "consider buying..."
    - "consider selling..."
    - "exercise caution..."
    - "good investment..."
    - "bad investment..."

19. The explanation must describe the model output, not provide
    financial advice.

The final response should be a natural-language explanation of
the supplied TradeNova risk result.
"""

# ============================================================
# GROQ CLIENT
# ============================================================

def _get_groq_client() -> Groq:
    """
    Create and return a Groq client.

    The API key is loaded from the GROQ_API_KEY
    environment variable.
    """

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY environment variable "
            "is not configured."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# SINGLE STOCK EXPLANATION
# ============================================================

def generate_risk_explanation(
    ticker: str,
    prediction: float,
    volatility_risk: float,
    anomaly_score: float,
    anomaly_level: str,
    behavioral_type: str,
    risk_score: float,
    risk_level: str,
) -> str:
    """
    Generate a natural-language explanation for an
    already calculated deterministic risk result.

    IMPORTANT:

    This function does NOT calculate risk.

    The supplied risk_score and risk_level are treated
    as authoritative outputs from the deterministic
    TradeNova Risk Engine.
    """

    client = _get_groq_client()

    user_prompt = user_prompt = f"""
Explain this deterministic TradeNova risk result.

Ticker:
{ticker}

V5 forecast volatility:
{prediction:.6f}

V5 cross-sectional volatility risk percentile:
{volatility_risk:.2f}

V6 anomaly score:
{anomaly_score:.6f}

V6 anomaly level:
{anomaly_level}

V7 behavioral profile:
{behavioral_type}

Final deterministic risk score:
{risk_score:.2f}

Final deterministic risk level:
{risk_level}

Explain the main factors behind the supplied final risk classification.

Interpretation requirements:

- The V5 volatility risk percentile is calculated across the
  current TradeNova stock universe, not from the stock's own
  historical percentile distribution.
- Explain V5 as the primary volatility signal.
- Explain V6 separately as anomaly detection.
- Explain V7 as historical behavioral context.
- If V6 is NORMAL, explicitly say that no unusually abnormal
  behavior was detected.
- Do not change or recalculate the final risk score.
- Do not change the final risk level.
- Do not invent information.
- Do not provide investment advice or trading recommendations.
"""
    
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_completion_tokens=250,
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "Groq returned an empty explanation."
        )

    return content.strip()


# ============================================================
# BATCH EXPLANATIONS
# ============================================================

def generate_batch_explanations(
    df,
) -> list[str]:
    """
    Generate one LLM explanation per stock.

    Expected DataFrame columns:

        Ticker
        volatility_forecast
        volatility_risk
        anomaly_score
        anomaly_level
        behavioral_type
        risk_score
        risk_level

    The LLM only explains the deterministic output.

    It does not calculate or modify risk.
    """

    explanations = []

    for _, row in df.iterrows():

        ticker = row["Ticker"]

        try:

            explanation = generate_risk_explanation(
                ticker=ticker,

                # ------------------------------------------------
                # V5
                # ------------------------------------------------
                prediction=float(
                    row["volatility_forecast"]
                ),

                volatility_risk=float(
                    row["volatility_risk"]
                ),

                # ------------------------------------------------
                # V6
                # ------------------------------------------------
                anomaly_score=float(
                    row["anomaly_score"]
                ),

                anomaly_level=str(
                    row["anomaly_level"]
                ),

                # ------------------------------------------------
                # V7
                # ------------------------------------------------
                behavioral_type=str(
                    row["behavioral_type"]
                ),

                # ------------------------------------------------
                # Risk Engine
                # ------------------------------------------------
                risk_score=float(
                    row["risk_score"]
                ),

                risk_level=str(
                    row["risk_level"]
                ),
            )

        except Exception as exc:

            print(
                f"  LLM ERROR for "
                f"{ticker}: {exc}"
            )

            explanation = (
                "Detailed AI explanation is "
                "currently unavailable."
            )

        explanations.append(
            explanation
        )

    return explanations