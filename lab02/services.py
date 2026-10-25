from schemas import FarmRequest
from storage import ALLOWED_REGIONS, MODEL_VERSION

from fastapi import HTTPException, status

def validate_business_rules(data: FarmRequest) -> None:
    # Pydantic проверяет типы, обязательность и диапазоны.
    # Здесь проверяются правила предметной области.

    if data.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {data.region}. "
                f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
            ),
        )

def risk_level(score: float) -> str:
    if score<0.3 :
        return "low"
    if score<0.7:
        return "medium"
    return "high"

def recommendation(level: str) -> str:
    if level == "low":
        return "Low risk level - safe option"
    if level == "medium":
        return "Medium risk level - please be advised when making a descision"
    return "High risk level - it is highly recommended to postpone this deal untill the risk level drops significantly"


def make_prediction_result(data: FarmRequest, score: float, request_id: str) -> dict:
    level = risk_level(score)

    return {
        "request_id": request_id,
        "farm_id": data.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation(level),
        "model_version": MODEL_VERSION,
    }
