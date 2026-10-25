from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError

import uuid, time, logging
from typing import Optional, List

from schemas import FarmRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from services import risk_level, recommendation, validate_business_rules, make_prediction_result
from storage import MODEL_NAME, MODEL_VERSION, MODEL_TYPE, MODEL_READY

import asyncio, httpx

from database import (
    init_db, 
    save_prediction, 
    save_predictions_batch, 
    get_predictions_from_db, 
    get_predictions_by_weekday_page, 
    seed_weekday_predictions,
    get_predictions_by_request_id
)
from batching import batch_worker, inference_queue, InferenceTask

from contextlib import asynccontextmanager
from dataclasses import dataclass

from model import model_predict_batch, model_predict_single
from schemas import BatchRequest, BatchPredictionResponse, PaginatedPredictionsResponse, LLMChatRequest, LLMChatResponse, PredictWithLLMResponse
from storage import (
    LLM_MODEL,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_DEFAULT_TEMPERATURE,
    MAX_BATCH_SIZE,
    MAX_BATCH_WAIT_MS,
    QUEUE_MAX_SIZE,
    INFERENCE_TIMEOUT_SECONDS,
    WEEKDAY_MAP
)
from llm_client import call_vireonix_llm, build_scoring_explanation_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    При старте:
    1. Инициализируем SQLite.
    2. Запускаем batch worker.

    При остановке:
    корректно завершаем worker.
    """

    await init_db()

    # Один переиспользуемый HTTP-клиент для внешнего LLM API.
    app.state.llm_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            LLM_TIMEOUT_SECONDS
        ),
    )

    worker = asyncio.create_task(
        batch_worker()
    )

    app.state.batch_worker = worker

    yield

    worker.cancel()

    try:
        await worker
    except asyncio.CancelledError:
        pass

    await app.state.llm_client.aclose()

app = FastAPI(
    title="Agro Scoring API",
    description=(
        "Agro Scoring API: request_id, SQLite/aiosqlite, "
        "single inference, batching и интеграция с Vireonix LLM."
    ),
    version="4.0.0",
    lifespan=lifespan,
)




@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": request_id,
        },
        headers=exc.headers,
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(
                exc.errors()
            ),
            "request_id": request_id,
        },
    )



@app.get("/health", response_model = HealthResponse, summary = "status check",
    description=(
        "Используется для проверки того, "
        "что REST API запущен и отвечает."
    )
)
async def health():
    return {
        "status": "ok"
    }

@app.get("/model-info", response_model = ModelInfoResponse,
    description=(
        "Возвращает название, версию, тип "
        "и текущее состояние модели."
    )
)
async def model_info():
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



@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, summary="Одиночный inference")
async def predict(data: FarmRequest, request: Request):
    """
    Один HTTP-request -> один inference -> одна запись SQLite.
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    validate_business_rules(data)

    request_id = (request.state.request_id)

    try:
        score = await asyncio.wait_for(
            model_predict_single(data),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout",
        )

    result = make_prediction_result(
        data,
        score,
        request_id,
    )

    # HTTP 200 возвращается только после успешного COMMIT.
    await save_prediction(
        result,
        source="single",
    )

    return result


@app.post("/batch-predict", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK, summary="Явный пакетный inference",)
async def batch_predict(batch: BatchRequest, request: Request):
    """
    Клиент сам передает несколько хозяйств одним HTTP-запросом.

    Все результаты batch сохраняются в SQLite одной транзакцией:
    executemany() + один commit().
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    for item in batch.items:
        validate_business_rules(item)

    request_id = (request.state.request_id)

    start_time = time.perf_counter()

    try:
        scores = await asyncio.wait_for(
            model_predict_batch(batch.items),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Batch inference timeout",
        )

    response_items = []

    for data, score in zip(batch.items,scores):
        level = risk_level(score)

        response_items.append(
            {
                "farm_id": data.farm_id,
                "risk_score": score,
                "risk_level": level,
                "recommendation": recommendation(level),
                "model_version": MODEL_VERSION,
            }
        )

    # Сохраняем весь batch одной транзакцией.
    await save_predictions_batch(
        request_id=request_id,
        items=response_items,
    )

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "request_id": request_id,
        "batch_size": len(
            batch.items
        ),
        "processing_time_ms": round(
            processing_time_ms,
            2,
        ),
        "items": response_items,
    }


@app.post("/predict-queued", response_model=PredictionResponse, status_code=status.HTTP_200_OK, summary="Inference через очередь и dynamic batching")
async def predict_queued(data: FarmRequest, request: Request):
    """
    С точки зрения клиента это одиночный HTTP-request.

    Сервер помещает его в asyncio.Queue и может объединить
    несколько запросов в один вызов модели.

    После получения score каждый запрос отдельно сохраняет
    свой результат в SQLite.
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    validate_business_rules(data)

    request_id = (request.state.request_id)

    # УПРАВЛЕНИЕ ПОТОКОМ ЗАПРОСОВ BACKPRESSURE

    if inference_queue.full():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Inference queue is full. "
                "Try again later."
            ),
            headers={"Retry-After": "1"},
        )

    loop = asyncio.get_running_loop()

    future = loop.create_future()

    task = InferenceTask(
        data=data,
        request_id=request_id,
        future=future,
    )

    try:
        inference_queue.put_nowait(task)

    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference queue is full",
            headers={
                "Retry-After": "1",
            },
        )

    logger.info(
        "Request queued | "
        "request_id=%s | queue_size=%s",
        request_id,
        inference_queue.qsize(),
    )

    try:
        score = await asyncio.wait_for(
            future,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        if not future.done():
            future.cancel()

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout",
        )

    result = make_prediction_result(
        data,
        score,
        request_id,
    )

    await save_prediction(
        result,
        source="queued",
    )

    return result


@app.get("/queue-status", summary="Состояние очереди и настройки batching")
async def queue_status():
    return {
        "queue_size": inference_queue.qsize(),
        "queue_max_size": QUEUE_MAX_SIZE,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_batch_wait_ms": MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
    }


@app.get("/predictions", response_model=List[PredictionResponse], summary="Получить историю прогнозов из SQLite")
async def get_predictions(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    risk_level: Optional[str] = Query(
        default=None,
        description="low, medium или high",
    ),
):
    if risk_level is not None:
        allowed_levels = {
            "low",
            "medium",
            "high",
        }

        if risk_level not in allowed_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=("risk_level must be low, medium or high"),
            )

    return await get_predictions_from_db(
        limit=limit,
        risk_level=risk_level,
    )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    request_id:
    - принимается из X-Request-ID;
    - если отсутствует, сервер создает UUID;
    - сохраняется в request.state;
    - возвращается клиенту в X-Request-ID.

    Также измеряется полное время HTTP-запроса.
    """

    start_time = time.perf_counter()

    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    logger.info(
        "Request started | "
        "request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled error | request_id=%s",
            request_id,
        )

        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id,
            },
        )

    process_time = (time.perf_counter() - start_time)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(round(process_time, 6))

    logger.info(
        "Request completed | "
        "request_id=%s | status=%s | time=%.4f",
        request_id,
        response.status_code,
        process_time,
    )

    return response





@app.post("/llm/chat", response_model=LLMChatResponse, status_code=status.HTTP_200_OK,
    summary="Прямой учебный запрос к Vireonix LLM",
    description=(
        "Отправляет prompt в OpenAI-совместимый Vireonix API. "
        "API-ключ и Authorization header не используются. "
        "Не передавайте реальные конфиденциальные данные."
    ),
)
async def llm_chat(data: LLMChatRequest, request: Request):
    request_id = (request.state.request_id)

    return await call_vireonix_llm(
        client=request.app.state.llm_client,
        prompt=data.prompt,
        request_id=request_id,
        model=data.model,
        temperature=data.temperature,
        max_tokens=data.max_tokens,
    )


@app.post("/predict-with-llm", response_model=PredictWithLLMResponse, status_code=status.HTTP_200_OK,
    summary="Агроскоринг с текстовым объяснением LLM",
    description=(
        "Сначала выполняет детерминированный агроскоринг и сохраняет "
        "результат в SQLite. Затем Vireonix LLM формирует только "
        "текстовое объяснение результата. LLM не определяет risk_score."
    ),
)
async def predict_with_llm(data: FarmRequest, request: Request):
    """
    Внешняя LLM не используется для расчета risk_score.
    """

    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )

    validate_business_rules(data)

    request_id = (request.state.request_id)

    try:
        score = await asyncio.wait_for(
            model_predict_single(data),
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout",
        )

    prediction = make_prediction_result(
        data,
        score,
        request_id,
    )

    # Сначала сохраняем детерминированный prediction.
    await save_prediction(
        prediction,
        source="single+llm",
    )

    prompt = build_scoring_explanation_prompt(prediction)

    llm_result = await call_vireonix_llm(
        client=request.app.state.llm_client,
        prompt=prompt,
        request_id=request_id,
        model=LLM_MODEL,
        temperature=LLM_DEFAULT_TEMPERATURE,
        max_tokens=LLM_DEFAULT_MAX_TOKENS,
    )

    return {
        "prediction": prediction,
        "llm": llm_result,
    }




@app.get("/requests/{request_id}/predictions",
    response_model=List[PredictionResponse],
    summary="Все прогнозы, созданные одним HTTP request",
)
async def get_request_predictions(request_id: str):
    """
    Для single / queued обычно возвращает один элемент.
    Для explicit batch может вернуть несколько элементов.
    """

    results = await get_predictions_by_request_id(request_id)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    return results


@app.get("/predictions/by-weekday", response_model=PaginatedPredictionsResponse,
    summary="Прогнозы по дню недели с пагинацией",
)
async def get_predictions_by_weekday(
    request: Request,

    weekday: str = Query(
        ...,
        description=(
            "День недели: monday, tuesday, wednesday, "
            "thursday, friday, saturday, sunday"
        ),
    ),

    page: int = Query(
        default=1,
        ge=1,
        description="Номер страницы",
    ),

    per_page: int = Query(
        default=5,
        ge=1,
        le=100,
        description="Количество записей на странице",
    ),
):
    """
    Пример:
        GET /predictions/by-weekday?weekday=monday&page=1&per_page=5

    Вернет первые 5 записей для понедельника.
    """

    weekday = weekday.lower()

    if weekday not in WEEKDAY_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "weekday must be one of: "
                "monday, tuesday, wednesday, thursday, "
                "friday, saturday, sunday"
            ),
        )

    data, total_items, total_pages = await get_predictions_by_weekday_page(
        weekday_number=WEEKDAY_MAP[weekday],
        page=page,
        per_page=per_page,
    )

    # Если записи существуют, но номер страницы слишком большой.
    if total_pages > 0 and page > total_pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Page {page} does not exist. "
                f"Total pages: {total_pages}"
            ),
        )

    # Если записей нет вообще, разрешаем только page=1.
    if total_pages == 0 and page > 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data for this weekday",
        )

    next_page = None

    if page < total_pages:
        next_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page + 1,
                per_page=per_page,
            )
        )

    prev_page = None

    if page > 1 and total_pages > 0:
        prev_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page - 1,
                per_page=per_page,
            )
        )

    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "next_page": next_page,
            "prev_page": prev_page,
        },
    }


@app.post("/demo/seed-weekdays",
    summary="Создать демонстрационные predictions за разные дни",
)
async def seed_demo_weekdays(
    count: int = Query(
        default=70,
        ge=7,
        le=1000,
        description="Количество демонстрационных записей",
    ),
):
    """
    Создает данные за последние 21 день, чтобы можно было
    протестировать выборку по дням недели и пагинацию.
    """

    created = await seed_weekday_predictions(count=count)

    return {
        "status": "ok",
        "created": created,
        "message": (
            "Demo predictions created. "
            "Use /predictions/by-weekday for pagination."
        ),
    }


@app.get("/predictions/{request_id}", response_model=PredictionResponse,
    summary="Получить одиночный прогноз по request_id",
)
async def get_prediction(request_id: str):
    """
    Сохраняет обратную совместимость старого API.

    Если request_id относится к batch и в БД несколько строк,
    возвращается 409 и предлагается использовать endpoint:
    /requests/{request_id}/predictions
    """

    results = await get_predictions_by_request_id(request_id)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found",
        )

    if len(results) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This request_id contains multiple predictions. "
                f"Use /requests/{request_id}/predictions"
            ),
        )

    return results[0]


@app.get("/")
async def root():
    return {
        "message": "Agro Scoring API is running",
        "docs": "/docs",
        "health": "/health"
    }


# Запуск
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
