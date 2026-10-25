from pathlib import Path

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

MODEL_READY = True


ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol"
}


LLM_BASE_URL = "https://vireonix.ai/v1"
LLM_MODEL = "auto"

LLM_TIMEOUT_SECONDS = 60.0

LLM_MAX_PROMPT_LENGTH = 5000
LLM_DEFAULT_MAX_TOKENS = 300
LLM_DEFAULT_TEMPERATURE = 0.3


MAX_BATCH_SIZE = 8

MAX_BATCH_WAIT_MS = 50

QUEUE_MAX_SIZE = 100

INFERENCE_TIMEOUT_SECONDS = 2.0


MODEL_FIXED_OVERHEAD_SECONDS = 0.020
MODEL_PER_ITEM_SECONDS = 0.003


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "agro_scoring.db"

DB_BUSY_TIMEOUT_MS = 5000


WEEKDAY_MAP = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
}