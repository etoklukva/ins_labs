import math, uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import HTTPException, status

import aiosqlite
from contextlib import asynccontextmanager

from storage import BASE_DIR, DATABASE_PATH, DB_BUSY_TIMEOUT_MS, MODEL_VERSION

from services import recommendation, risk_level

import logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def get_db():
    db = await aiosqlite.connect(
        str(DATABASE_PATH),
        timeout=DB_BUSY_TIMEOUT_MS / 1000,
    )

    db.row_factory = aiosqlite.Row

    # busy_timeout задается для каждого нового соединения.
    await db.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS};")

    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    # Создает таблицу predictions при старте приложения

    async with get_db() as db:
        await db.execute(
            "PRAGMA journal_mode=WAL;"
        )

        await db.execute(
            "PRAGMA synchronous=NORMAL;"
        )

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_request_id
            ON predictions(request_id);
            """
        )

        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_risk_level
            ON predictions(risk_level);
            """
        )

        await db.commit()

    logger.info(
        "SQLite initialized | path=%s",
        DATABASE_PATH,
    )


async def save_prediction(result: dict, source: str) -> None:
    # Сохраняет один prediction.
    # request_id не используется как PRIMARY KEY:
    # один HTTP batch-request может породить несколько прогнозов
    # с одним и тем же request_id.

    try:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    result["request_id"],
                    result["farm_id"],
                    result["risk_score"],
                    result["risk_level"],
                    result["recommendation"],
                    result["model_version"],
                    source,
                ),
            )

            await db.commit()

    except aiosqlite.Error as exc:
        logger.exception(
            "Database write error | request_id=%s",
            result.get("request_id"),
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc



async def save_predictions_batch(request_id: str, items: List[dict]) -> None:
    # Сохраняет весь batch через executemany() и один COMMIT.

    rows = [
        (
            request_id,
            item["farm_id"],
            item["risk_score"],
            item["risk_level"],
            item["recommendation"],
            item["model_version"],
            "batch",
        )
        for item in items
    ]

    try:
        async with get_db() as db:
            await db.executemany(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )

            await db.commit()

    except aiosqlite.Error as exc:
        logger.exception(
            "Batch database write error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


async def get_predictions_from_db(limit: int, risk_level: Optional[str] = None) -> List[dict]:
    # Возвращает историю прогнозов из SQLite

    try:
        async with get_db() as db:
            if risk_level is None:
                cursor = await db.execute(
                    """
                    SELECT
                        request_id,
                        farm_id,
                        risk_score,
                        risk_level,
                        recommendation,
                        model_version
                    FROM predictions
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT
                        request_id,
                        farm_id,
                        risk_score,
                        risk_level,
                        recommendation,
                        model_version
                    FROM predictions
                    WHERE risk_level = ?
                    ORDER BY id DESC
                    LIMIT ?;
                    """,
                    (
                        risk_level,
                        limit,
                    ),
                )

            rows = await cursor.fetchall()
            await cursor.close()

        return [
            dict(row)
            for row in rows
        ]

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


async def get_predictions_by_weekday_page(weekday_number: str, page: int, per_page: int) -> tuple[List[dict], int, int]:
    # Возвращает одну страницу прогнозов для выбранного дня недели.

    offset = (page - 1) * per_page

    try:
        async with get_db() as db:
            # 1. Получаем записи текущей страницы.
            cursor = await db.execute(
                """
                SELECT
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version
                FROM predictions
                WHERE strftime('%w', created_at) = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?;
                """,
                (
                    weekday_number,
                    per_page,
                    offset,
                ),
            )

            rows = await cursor.fetchall()
            await cursor.close()

            # 2. Считаем общее количество записей этого дня недели.
            cursor = await db.execute(
                """
                SELECT COUNT(*)
                FROM predictions
                WHERE strftime('%w', created_at) = ?;
                """,
                (weekday_number,),
            )

            row = await cursor.fetchone()
            await cursor.close()

        total_items = row[0]

        total_pages = (
            math.ceil(total_items / per_page)
            if total_items > 0
            else 0
        )

        return (
            [dict(row) for row in rows],
            total_items,
            total_pages,
        )

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error | weekday=%s",
            weekday_number,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


async def seed_weekday_predictions(count: int = 70) -> int:
    # Создает демонстрационные predictions за разные дни.

    rows = []

    for i in range(count):
        request_id = str(uuid.uuid4())
        farm_id = f"DEMO-{i + 1:03d}"

        risk_score = round(
            min(0.1 + (i % 9) * 0.1, 1.0),
            2,
        )

        risk_level = risk_level(risk_score)

        rows.append(
            (
                request_id,
                farm_id,
                risk_score,
                risk_level,
                recommendation(risk_level),
                MODEL_VERSION,
                "seed",
                f"-{i % 21} days",
            )
        )

    try:
        async with get_db() as db:
            await db.executemany(
                """
                INSERT INTO predictions (
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version,
                    source,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    datetime('now', ?)
                );
                """,
                rows,
            )

            await db.commit()

        return count

    except aiosqlite.Error as exc:
        logger.exception(
            "Demo seed database error"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc


async def get_predictions_by_request_id(request_id: str) -> List[dict]:
    # Возвращает все прогнозы, связанные с HTTP request_id.

    try:
        async with get_db() as db:
            cursor = await db.execute(
                """
                SELECT
                    request_id,
                    farm_id,
                    risk_score,
                    risk_level,
                    recommendation,
                    model_version
                FROM predictions
                WHERE request_id = ?
                ORDER BY id;
                """,
                (request_id,),
            )

            rows = await cursor.fetchall()
            await cursor.close()

        return [dict(row) for row in rows]

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        ) from exc

    