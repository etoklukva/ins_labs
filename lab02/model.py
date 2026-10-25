from schemas import FarmRequest
from typing import List

from storage import MODEL_FIXED_OVERHEAD_SECONDS, MODEL_PER_ITEM_SECONDS

import asyncio

model_lock = asyncio.Lock()


async def model_predict_single(data: FarmRequest) -> float:
    # Одиночный вызов модели

    async with model_lock:
        await asyncio.sleep(MODEL_FIXED_OVERHEAD_SECONDS + MODEL_PER_ITEM_SECONDS)

        return calculate_risk(data)


async def model_predict_batch(items: List[FarmRequest]) -> List[float]:
    # Один пакетный вызов модели

    async with model_lock:
        await asyncio.sleep(MODEL_FIXED_OVERHEAD_SECONDS + MODEL_PER_ITEM_SECONDS * len(items))

        return [
            calculate_risk(item)
            for item in items
        ]

def calculate_risk(data: FarmRequest) -> float:
    score = 0.1 
    if data.payment_delay_day > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    return min(score, 1.0)


