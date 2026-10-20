import argparse
import asyncio
from elasticsearch import AsyncElasticsearch
from copilot.config import Settings

MAPPINGS = {"dynamic": "strict", "properties": {
    "chunk_id": {"type": "keyword"}, "document_id": {"type": "keyword"},
    "version": {"type": "long"}, "title": {"type": "text", "analyzer": "russian"},
    "text": {"type": "text", "analyzer": "russian"}, "vertical": {"type": "keyword"},
    "public": {"type": "boolean"}, "source": {"type": "keyword"}, "score": {"type": "float"},
    "embedding": {"type": "dense_vector", "dims": 1024, "index": True, "similarity": "cosine"},
}}


async def create_indices(es: AsyncElasticsearch, index: str) -> None:
    for name, mappings in [(index, MAPPINGS), (f"{index}-versions", {"properties": {
        "version": {"type": "long"}, "event_id": {"type": "keyword"}}}),
        ("operator-feedback", {"dynamic": "strict", "properties": {
            "event_id": {"type": "keyword"}, "request_id": {"type": "keyword"},
            "outcome": {"type": "keyword"}, "reason_code": {"type": "keyword"}}})]:
        if not await es.indices.exists(index=name):
            await es.indices.create(index=name, mappings=mappings)


async def main() -> None:
    argparse.ArgumentParser(description="Create demonstration indices").parse_args()
    settings = Settings()
    async with AsyncElasticsearch(settings.elasticsearch_url) as es:
        await create_indices(es, settings.es_index)


if __name__ == "__main__":
    asyncio.run(main())
