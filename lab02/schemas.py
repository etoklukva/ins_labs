from pydantic import BaseModel, Field
from typing import List, Optional

from storage import MAX_BATCH_SIZE, LLM_MAX_PROMPT_LENGTH, LLM_DEFAULT_MAX_TOKENS, LLM_MODEL, LLM_DEFAULT_TEMPERATURE

class FarmRequest (BaseModel):
    farm_id: str
    region: str
    crop_type: str
    temperature_avg: float
    precipitation_mm: float = Field(ge = 0)
    area_ha: float = Field(gt = 0)
    payment_delay_day: int = Field(ge = 0)
    previous_defaults: int = Field(ge = 0)
    debt: float = Field(ge = 0)


class PredictionResponse (BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class BatchRequest (BaseModel):
    items: list[FarmRequest] = Field(
        ..., 
        min_length = 1, 
        max_length = MAX_BATCH_SIZE, 
        description = "Список объектов для пакетного инференса"
    )

class BatchItemResponse (BaseModel):
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str

class BatchPredictionResponse (BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: List[BatchItemResponse]


class PaginationMeta (BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: Optional[str]
    prev_page: Optional[str]

class PaginatedPredictionsResponse(BaseModel):
    data: List[PredictionResponse]
    meta: PaginationMeta



class LLMChatRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=LLM_MAX_PROMPT_LENGTH,
        description="Текст запроса к LLM",
    )
    model: str = Field(
        default=LLM_MODEL,
        min_length=1,
        description="Модель Vireonix // Используется auto",
    )
    temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS,
        ge=1,
        le=2000,
    )

class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: Optional[str]
    answer: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    latency_ms: float

class PredictWithLLMResponse(BaseModel):
    """
    Комбинированный ответ:
    1. детерминированный результат агроскоринга;
    2. текстовое объяснение, сформированное LLM.
    """
    prediction: PredictionResponse
    llm: LLMChatResponse
