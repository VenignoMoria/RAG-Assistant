from typing import Protocol
from .domain import Chunk, DraftRequest, GeneratedDraft, RouteDecision, Trip


class OrderPort(Protocol):
    async def get_trip(self, trip_id: str, operator_id: str) -> Trip: ...


class RouterPort(Protocol):
    async def route(self, request: DraftRequest, trip: Trip) -> RouteDecision: ...


class RetrievalPort(Protocol):
    async def search(self, query: str, vertical: str | None) -> list[Chunk]: ...


class GeneratorPort(Protocol):
    async def generate(self, query: str, trip: Trip, chunks: list[Chunk]) -> GeneratedDraft: ...


class NLIPort(Protocol):
    async def entailment(self, premise: str, hypothesis: str) -> float: ...
