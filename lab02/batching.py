import logging, asyncio, time
from dataclasses import dataclass

from schemas import FarmRequest
from storage import QUEUE_MAX_SIZE, MAX_BATCH_SIZE, MAX_BATCH_WAIT_MS, INFERENCE_TIMEOUT_SECONDS
from model import model_predict_batch

logger = logging.getLogger(__name__)


inference_queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


async def batch_worker() -> None:
    # Фоновый worker:
    # 1. Берет первый запрос.
    # 2. Копит дополнительные запросы до MAX_BATCH_SIZE или до истечения MAX_BATCH_WAIT_MS.
    # 3. Выполняет один batch inference.
    # 4. Возвращает каждому HTTP-запросу его score.

    logger.info("Batch worker started")

    while True:
        first_task = await inference_queue.get()

        if first_task.future.cancelled():
            inference_queue.task_done()
            continue

        batch = [first_task]

        deadline = (time.perf_counter() + MAX_BATCH_WAIT_MS / 1000)

        # Накопление dynamic batch

        while len(batch) < MAX_BATCH_SIZE:
            remaining_time = (deadline - time.perf_counter())

            if remaining_time <= 0:
                break

            try:
                task = await asyncio.wait_for(
                    inference_queue.get(),
                    timeout=remaining_time,
                )

                if task.future.cancelled():
                    inference_queue.task_done()
                    continue

                batch.append(
                    task
                )

            except asyncio.TimeoutError:
                break

        logger.info(
            "Dynamic batch formed | size=%s | queue_size=%s",
            len(batch),
            inference_queue.qsize(),
        )

        try:
            scores = await asyncio.wait_for(
                model_predict_batch(
                    [
                        task.data
                        for task in batch
                    ]
                ),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )

            for task, score in zip(
                batch,
                scores,
            ):
                if not task.future.done():
                    task.future.set_result(
                        score
                    )

        except Exception as exc:
            logger.exception(
                "Batch inference failed"
            )

            for task in batch:
                if not task.future.done():
                    task.future.set_exception(
                        exc
                    )

        finally:
            # На каждый queue.get() должен быть task_done().
            for _ in batch:
                inference_queue.task_done()