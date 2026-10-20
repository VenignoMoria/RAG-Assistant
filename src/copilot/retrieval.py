import asyncio
import math
from elasticsearch import AsyncElasticsearch
from .clients.models import ModelServices
from .domain import Chunk


def rrf(rankings: list[list[Chunk]], k: int = 60, limit: int = 20) -> list[Chunk]:
    """RRF суммирует 1/(k+rank), не смешивая несопоставимые BM25/cosine scores."""
    scores: dict[str, float] = {}
    objects: dict[str, Chunk] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            objects[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return [objects[cid].model_copy(update={"score": scores[cid]}) for cid in ordered]


def search_filters(vertical: str | None) -> list[dict]:
    filters: list[dict] = [{"term": {"public": True}}]
    if vertical:
        filters.append({"terms": {"vertical": [vertical, "common"]}})
    return filters


class HybridRetriever:
    def __init__(self, es: AsyncElasticsearch, index: str, models: ModelServices, threshold: float):
        self.es, self.index, self.models, self.threshold = es, index, models, threshold

    async def search(self, query: str, vertical: str | None) -> list[Chunk]:
        query = " ".join(query.split())
        if not query:
            return []
        vector = (await self.models.embeddings([query], "query"))[0]
        filters = search_filters(vertical)
        sparse, dense = await asyncio.gather(
            self.es.search(index=self.index, size=50, source_excludes=["embedding"], query={
                "bool": {"must": [{"multi_match": {"query": query, "fields": ["title^2", "text"]}}],
                         "filter": filters}}),
            self.es.search(index=self.index, size=50, source_excludes=["embedding"], knn={
                "field": "embedding", "query_vector": vector, "k": 50, "num_candidates": 100,
                "filter": {"bool": {"filter": filters}}}),
        )
        rankings = [[Chunk.model_validate(hit["_source"]) for hit in result["hits"]["hits"]]
                    for result in (sparse, dense)]
        candidates = rrf(rankings, limit=20)
        if not candidates:
            return []
        reranked = await self.models.rerank(query, candidates)
        # Sigmoid(logit) удобен как сигнал, но НЕ является калиброванной
        # вероятностью релевантности. Порог подбирают на размеченной выборке.
        return [c for c in reranked if math.isfinite(c.score) and c.score >= self.threshold][:3]
