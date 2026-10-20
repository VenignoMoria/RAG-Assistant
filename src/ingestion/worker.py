import asyncio
import json
import os
from pathlib import Path
import httpx
import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from elasticsearch import AsyncElasticsearch, ConflictError, NotFoundError
from copilot.clients.models import ModelServices
from copilot.config import Settings
from copilot.domain import FeedbackEvent, KnowledgeEvent
from copilot.observability import setup_logging
from .chunking import chunks_for

log = structlog.get_logger()


class KnowledgeIndexer:
    def __init__(self, es: AsyncElasticsearch, index: str, models: ModelServices):
        self.es, self.index, self.models = es, index, models

    async def apply(self, event: KnowledgeEvent) -> None:
        # Kafka key=document_id обеспечивает последовательность в partition.
        # Только один consumer group имеет право писать этот индекс. Нельзя
        # параллельно запускать независимые группы writers: ES не даёт общей
        # транзакции между bulk chunks и version marker.
        try:
            current = await self.es.get(index=f"{self.index}-versions", id=event.document_id)
            if current["_source"]["version"] >= event.version:
                return
        except NotFoundError:
            pass
        chunks = chunks_for(event)
        vectors: list[list[float]] = []
        for offset in range(0, len(chunks), 16):
            vectors.extend(await self.models.embeddings([c.text for c in chunks[offset:offset+16]], "passage"))
        for offset in range(0, len(chunks), 100):
            operations = []
            for chunk, vector in zip(chunks[offset:offset+100], vectors[offset:offset+100], strict=True):
                operations.extend([{"index": {"_index": self.index, "_id": chunk.chunk_id}},
                                   {**chunk.model_dump(mode="json"), "embedding": vector}])
            result = await self.es.bulk(operations=operations, refresh="wait_for")
            if result["errors"]:
                raise RuntimeError("Bulk indexing failed; replay is idempotent")
        await self.es.delete_by_query(index=self.index, query={"bool": {"filter": [
            {"term": {"document_id": event.document_id}}, {"range": {"version": {"lt": event.version}}}]}},
            refresh=True, conflicts="proceed")
        # Marker записывается последним. При crash повторяем запись по тем же
        # chunk_id. На обновлении есть короткое окно видимости двух версий;
        # для атомарного snapshot нужен build-new-index + alias swap.
        await self.es.index(index=f"{self.index}-versions", id=event.document_id,
                            document={"version": event.version, "event_id": event.event_id}, refresh="wait_for")


async def run() -> None:
    setup_logging()
    settings = Settings()
    mode = os.getenv("WORKER_MODE", "knowledge")
    if mode not in {"knowledge", "feedback"}:
        raise ValueError("WORKER_MODE must be knowledge or feedback")
    topic = "knowledge-updates" if mode == "knowledge" else "operator-feedback"
    consumer = AIOKafkaConsumer(topic, bootstrap_servers=settings.kafka_bootstrap,
                               group_id=f"copilot-{mode}-v1", enable_auto_commit=False,
                               auto_offset_reset="earliest", max_poll_records=1,
                               max_poll_interval_ms=900000)
    producer = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap, acks="all", enable_idempotence=True)
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as http, \
            AsyncElasticsearch(settings.elasticsearch_url) as es:
        models = ModelServices(http, settings.embedding_url, settings.reranker_url, settings.nli_url)
        indexer = KnowledgeIndexer(es, settings.es_index, models)
        await producer.start()
        try:
            await consumer.start()
            marker = Path("/tmp/worker-ready")
            marker.touch()
            try:
                async for message in consumer:
                    for attempt in range(3):
                        try:
                            payload = json.loads(message.value)
                            if mode == "knowledge":
                                event = KnowledgeEvent.model_validate(payload)
                                if message.key != event.document_id.encode():
                                    raise ValueError("Knowledge event key must be document_id")
                                # Budget всех попыток меньше max_poll_interval,
                                # иначе rebalance может создать параллельного writer.
                                async with asyncio.timeout(180):
                                    await indexer.apply(event)
                            else:
                                feedback = FeedbackEvent.model_validate(payload)
                                try:
                                    await es.create(index="operator-feedback", id=feedback.event_id,
                                                    document=feedback.model_dump(mode="json"))
                                except ConflictError:
                                    pass  # Повтор события уже сохранён, второй insert не нужен.
                            break
                        except Exception as exc:
                            log.warning("ingestion_attempt_failed", attempt=attempt + 1, error_type=type(exc).__name__)
                            if attempt == 2:
                                # Original bytes передаются как строка только в закрытый DLQ;
                                # даже для ошибочных сообщений запрещено логирование payload.
                                envelope = {"topic": topic, "partition": message.partition, "offset": message.offset,
                                            "payload": message.value.decode(errors="replace"),
                                            "error_type": type(exc).__name__}
                                await producer.send_and_wait(f"{topic}.dlq", key=message.key,
                                                             value=json.dumps(envelope).encode())
                            else:
                                await asyncio.sleep(2 ** attempt)
                    # Commit только после успешной записи либо ACK от DLQ.
                    # При недоступности DLQ процесс падает и offset не сдвигает.
                    await consumer.commit()
            finally:
                marker.unlink(missing_ok=True)
                await consumer.stop()
        finally:
            await producer.stop()


if __name__ == "__main__":
    asyncio.run(run())
