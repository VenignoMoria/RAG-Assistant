import math
import httpx
from ..domain import Chunk


class ModelServices:
    def __init__(self, http: httpx.AsyncClient, embedding_url: str, reranker_url: str, nli_url: str):
        self.http = http
        self.embedding_url, self.reranker_url, self.nli_url = embedding_url, reranker_url, nli_url

    async def embeddings(self, texts: list[str], kind: str) -> list[list[float]]:
        response = await self.http.post(f"{self.embedding_url}/embed", json={"texts": texts, "kind": kind})
        response.raise_for_status()
        vectors = response.json()["vectors"]
        if len(vectors) != len(texts) or any(len(v) != 1024 or not all(math.isfinite(x) for x in v) for v in vectors):
            raise ValueError("Invalid E5 embedding shape")
        return vectors

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        response = await self.http.post(f"{self.reranker_url}/rerank", json={
            "query": query, "texts": [c.text for c in chunks],
        })
        response.raise_for_status()
        scores = response.json()["scores"]
        if len(scores) != len(chunks):
            raise ValueError("Reranker score count mismatch")
        return sorted([c.model_copy(update={"score": float(s)}) for c, s in zip(chunks, scores, strict=True)],
                      key=lambda c: c.score, reverse=True)

    async def entailment(self, premise: str, hypothesis: str) -> float:
        response = await self.http.post(f"{self.nli_url}/nli", json={"premise": premise, "hypothesis": hypothesis})
        response.raise_for_status()
        score = float(response.json()["entailment"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid NLI probability")
        return score
