from fastapi import HTTPException, status
from storage import LLM_MODEL, LLM_DEFAULT_TEMPERATURE, LLM_DEFAULT_MAX_TOKENS, LLM_BASE_URL

import time, httpx

import logging

logger = logging.getLogger(__name__)


async def call_vireonix_llm(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_DEFAULT_TEMPERATURE,
    max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
) -> dict:
    # Асинхронный запрос к Vireonix.

    url = f"{LLM_BASE_URL}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник сервиса агроскоринга. "
                    "Отвечай кратко и понятно. "
                    "Не изменяй переданный risk_score и risk_level. "
                    "Не придумывай причины, которых нет во входных данных."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()

    try:
        response = await client.post(
            url,
            headers={
                "Content-Type": "application/json",
            },
            json=payload,
        )

    except httpx.TimeoutException as exc:
        logger.exception(
            "LLM timeout | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timeout",
        ) from exc

    except httpx.RequestError as exc:
        logger.exception(
            "LLM connection error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is unavailable",
        ) from exc

    latency_ms = (time.perf_counter() - started) * 1000

    if response.status_code != status.HTTP_200_OK:
        logger.error(
            "LLM upstream error | request_id=%s | status=%s | body=%s",
            request_id,
            response.status_code,
            response.text[:500],
        )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "LLM provider returned an error "
                f"(HTTP {response.status_code})"
            ),
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider returned invalid JSON",
        ) from exc

    choices = data.get(
        "choices",
        [],
    )

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain choices",
        )

    choice = choices[0]

    message = choice.get(
        "message",
        {},
    )

    content = message.get(
        "content"
    )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain message content",
        )

    return {
        "request_id": request_id,
        "requested_model": model,
        "returned_model": data.get("model"),
        "answer": content,
        "reasoning": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "latency_ms": round(
            latency_ms,
            2,
        ),
    }


def build_scoring_explanation_prompt(prediction: dict) -> str:
    # Формирует промпт только из результата скорингового алгоритма.
    # Не отправляем внешний farm_id и исходные финансовые признаки в публичный LLM gateway.

    return (
        "Сформируй краткое объяснение для оператора банка "
        "по результату алгоритмического агроскоринга.\n\n"
        f"risk_score: {prediction['risk_score']}\n"
        f"risk_level: {prediction['risk_level']}\n"
        f"базовая рекомендация: {prediction['recommendation']}\n"
        f"версия скоринговой модели: {prediction['model_version']}\n\n"
        "Требования к ответу:\n"
        "1. 2-4 предложения.\n"
        "2. Не меняй категорию риска.\n"
        "3. Не придумывай дополнительные факторы риска.\n"
        "4. Укажи, что решение требует применения правил банка "
        "и при необходимости проверки оператором."
    )