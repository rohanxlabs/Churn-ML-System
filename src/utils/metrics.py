from __future__ import annotations


def risk_level(churn_probability: float) -> str:
    if churn_probability >= 0.7:
        return "high"
    if churn_probability >= 0.4:
        return "medium"
    return "low"
