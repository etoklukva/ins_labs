from pydantic import BaseModel, Field

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