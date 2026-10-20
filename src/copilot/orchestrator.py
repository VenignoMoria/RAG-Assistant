import structlog
from redis.exceptions import RedisError
from .cache import PublicRetrievalCache
from .domain import DraftRequest, DraftResponse, Trip
from .ports import GeneratorPort, OrderPort, RetrievalPort, RouterPort
from .routing import validate_route
from .verification import DraftVerifier

log = structlog.get_logger()


def fallback(request_id: str, reason: str) -> DraftResponse:
    return DraftResponse(request_id=request_id, action="fallback",
                         answer="Недостаточно проверенных оснований. Оператору нужно проверить обращение вручную.",
                         reason=reason)


class Orchestrator:
    def __init__(self, orders: OrderPort, router: RouterPort, retrieval: RetrievalPort,
                 generator: GeneratorPort, verifier: DraftVerifier, cache: PublicRetrievalCache):
        self.orders, self.router, self.retrieval = orders, router, retrieval
        self.generator, self.verifier, self.cache = generator, verifier, cache

    async def draft(self, request: DraftRequest, operator_id: str, request_id: str) -> DraftResponse:
        trip = await self.orders.get_trip(request.trip_id, operator_id)
        if not set(request.selected_ticket_ids) <= {t.ticket_id for t in trip.tickets}:
            raise ValueError("Invalid explicit selection")
        route = validate_route(await self.router.route(request, trip), request, trip)
        if route.action == "fallback":
            return fallback(request_id, "router_fallback")
        if route.action == "clarify":
            return DraftResponse(request_id=request_id, action="clarify",
                                 answer=route.question or "Уточните билет и маршрут поездки.", reason="needs_clarification")
        selected = [t for t in trip.tickets if t.ticket_id in route.ticket_ids]
        vertical = selected[0].vertical if selected else None
        key = self.cache.key(request.message, bool(request.history), bool(request.selected_ticket_ids))             if route.action == "general" else None
        chunks = None
        if key:
            try:
                chunks = await self.cache.get(key)
            except (RedisError, ValueError):
                log.warning("cache_read_unavailable")
        if chunks is None:
            chunks = await self.retrieval.search(route.query, vertical)
            if key:
                try:
                    await self.cache.set(key, chunks)
                except RedisError:
                    log.warning("cache_write_unavailable")
        # При общем запросе о разных типах билетов нельзя создать единое
        # правило из авиа и ж/д. Common можно объединять с одной вертикалью.
        verticals = {str(c.vertical) for c in chunks} - {"common"}
        if len(verticals) > 1:
            common_chunks = [c for c in chunks if str(c.vertical) == "common"]
            if route.action == "general" and common_chunks:
                chunks = common_chunks
            else:
                return DraftResponse(request_id=request_id, action="clarify",
                                     answer="Для какого вида транспорта нужны общие правила?", reason="mixed_knowledge_verticals")
        if not chunks:
            return fallback(request_id, "low_retrieval_confidence")
        context = Trip(trip_id=trip.trip_id, tickets=selected)
        draft = await self.generator.generate(route.query, context, chunks)
        if not await self.verifier.verify(draft, chunks, context):
            return fallback(request_id, "unsupported_draft")
        cited = {cid for claim in draft.claims for cid in claim.citation_ids}
        return DraftResponse(request_id=request_id, action=route.action, answer=draft.answer,
                             citations=[c for c in chunks if c.chunk_id in cited],
                             ticket_ids=route.ticket_ids, reason="verified_draft")
