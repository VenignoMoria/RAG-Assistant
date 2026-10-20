import hashlib
import json
from redis.asyncio import Redis
from .domain import Chunk


class PublicRetrievalCache:
    """Кэш документов только для фиксированного публичного каталога запросов."""
    CATALOG = {"общие правила возврата", "общие правила обмена", "как связаться с поддержкой"}

    def __init__(self, redis: Redis, version: str, ttl: int):
        self.redis, self.version, self.ttl = redis, version, ttl

    def key(self, original_message: str, history_present: bool, has_selection: bool) -> str | None:
        normalized = " ".join(original_message.lower().strip(" ?.! ").split())
        # Hash произвольного запроса не делает PII безопасными. Допускается
        # только точное совпадение с публичным каталогом без истории/билетов.
        if history_present or has_selection or normalized not in self.CATALOG:
            return None
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"public-retrieval:v1:{self.version}:{digest}"

    async def get(self, key: str) -> list[Chunk] | None:
        value = await self.redis.get(key)
        if value is None:
            return None
        return [Chunk.model_validate(c) for c in json.loads(value)]

    async def set(self, key: str, chunks: list[Chunk]) -> None:
        if all(c.public for c in chunks):
            await self.redis.set(key, json.dumps([c.model_dump(mode="json") for c in chunks]), ex=self.ttl)
