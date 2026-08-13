from app.llm.explainer import generate_risk_explanation


def main():

    explanation = generate_risk_explanation(
        ticker="TCS.NS",
        prediction=0.283487,
        volatility_risk=100.0,
        anomaly_score=-0.041972,
        anomaly_level="NORMAL",
        behavioral_type="Lower Sensitivity / Lower Volatility",
        risk_score=100.0,
        risk_level="VERY HIGH",
    )

    print("=" * 70)
    print("GROQ RISK EXPLANATION")
    print("=" * 70)
    print(explanation)
    print("=" * 70)


if __name__ == "__main__":
    main()