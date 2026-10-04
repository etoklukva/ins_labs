from fastapi import FastAPI, HTTPException, Query, Request, status
import uuid, time, logging
from typing import Optional, List

from schemas import FarmRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from model import calculate_risk
from services import risk_level, recommendation
from storage import MODEL_NAME, MODEL_VERSION, MODEL_TYPE, MODEL_READY, ALLOWED_REGIONS, predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)



app = FastAPI(
    title = "Agro Culture API",
    description = "REST API for risk calcs",
    version = "1.0.0"
)



@app.get("/health", response_model = HealthResponse, summary = "status check",
    description=(
        "Используется для проверки того, "
        "что REST API запущен и отвечает."
    )
)
def health():
    return {
        "status": "ok"
    }

@app.get("/model-info", response_model = ModelInfoResponse,
    description=(
        "Возвращает название, версию, тип "
        "и текущее состояние модели."
    )
)
def model_info():
    model_status = (
        "ready"
        if MODEL_READY
        else "unavailable"
    )
    return {
        "model_name": MODEL_NAME,
        "model-version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": model_status
    }



@app.post("/predict", response_model = PredictionResponse, status_code = status.HTTP_200_OK,
    description=(
        "Принимает характеристики хозяйства, "
        "выполняет валидацию, инференс модели "
        "и возвращает оценку риска."
    )          
)
def predict (request: FarmRequest):
    if not MODEL_READY:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "Model is temporarily unavailable"
        )
    if request.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = (
                f"Unknown region: {request.region}."
                f"Allowed regions: "
                f"{sorted(ALLOWED_REGIONS)}"
            )
        )

    logger.info(f"Prediction request recieved / farm_id = {request.farm_id}")

    score = calculate_risk(request)
    level = risk_level(score)
    rec = recommendation(level)
    request_id = str(uuid.uuid4())

    result = {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": rec,
        "model_version": MODEL_VERSION
    }

    predictions[request_id] = result

    logger.info(
        "Prediction completed | "
        "request_id=%s | "
        "farm_id=%s | "
        "risk_score=%s | "
        "risk_level=%s",
        request_id,
        request.farm_id,
        score,
        level
    )
    return result



@app.get("/predictions", response_model = List[PredictionResponse], summary = "Get the list of completed predictions",
    description=(
        "Возвращает список выполненных прогнозов. "
        "Поддерживает ограничение количества результатов "
        "и фильтрацию по уровню риска."
    )         
)
def get_predictions(limit: int = Query(default=10, ge=1, le=100), risk_level: Optional[str] = Query(default=None)):
    
    values = list(predictions.values())
    allowed_levels = {
        "low",
        "medium",
        "high"
    }

    if risk_level is not None:
        if risk_level not in allowed_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=("risk_level must be 'low', 'medium' or 'high'")
            )
      
        values = [
            item
            for item in values
            if item["risk_level"] == risk_level
        ]
  
    return values[:limit]


@app.middleware("http")
async def add_process_time (request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = str(
        round(process_time, 6)
    )

    return response



@app.get("/predictions/{request_id}", response_model = PredictionResponse, summary = "Get prediction with request_id", 
    description=(
        "Возвращает сохраненный прогноз "
        "по его уникальному идентификатору."
    )
)
def get_prediction(request_id: str):

    if request_id not in predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
 
    return predictions[request_id]
