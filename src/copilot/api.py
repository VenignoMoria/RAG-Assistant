import asyncio
import hmac
from contextlib import asynccontextmanager
from uuid import uuid4
import httpx
import structlog
from elasticsearch import AsyncElasticsearch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from .cache import PublicRetrievalCache
from .clients.models import ModelServices
from .clients.order import MockOrderClient, OrderClient
from .clients.vllm import VLLMClient
from .config import Settings
from .domain import DraftRequest, DraftResponse, FeedbackEvent
from .generation import DraftGenerator
from .observability import FAILURES, LATENCY, REQUESTS, setup_logging
from .orchestrator import Orchestrator, fallback
from .retrieval import HybridRetriever
from .routing import ModelRouter
from .verification import DraftVerifier

log = structlog.get_logger()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = Settings()
    async with httpx.AsyncClient(timeout=settings.timeout_seconds) as http, \
            AsyncElasticsearch(settings.elasticsearch_url) as es:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        models = ModelServices(http, settings.embedding_url, settings.reranker_url, settings.nli_url)
        orders = MockOrderClient() if settings.mock_orders else OrderClient(
            http, settings.order_url, settings.order_token.get_secret_value())
        app.state.orchestrator = Orchestrator(
            orders, ModelRouter(VLLMClient(http, settings.router_url, settings.router_model)),
            HybridRetriever(es, settings.es_index, models, settings.rerank_threshold),
            DraftGenerator(VLLMClient(http, settings.generator_url, settings.generator_model)),
            DraftVerifier(models, settings.nli_threshold),
            PublicRetrievalCache(redis, settings.kb_version, settings.cache_ttl))
        app.state.settings, app.state.ready = settings, True
        try:
            yield
        finally:
            app.state.ready = False
            await redis.aclose()


app = FastAPI(title="Support Copilot RAG — учебный проект", lifespan=lifespan)


async def authorize(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security),
                    operator_id: str = Header(alias="X-Operator-ID", min_length=1, max_length=100)) -> str:
    expected = request.app.state.settings.service_token.get_secret_value()
    if not expected or not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="Invalid backend service credential")
    # Внешний ingress должен удалять X-Operator-ID клиента. Заголовок выставляет
    # только аутентифицированный backend; mock Order API ограничен demo identity.
    return operator_id


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
async def ready(request: Request) -> dict[str, bool]:
    # Это готовность самого процесса. Доступность внешних зависимостей
    # мониторится отдельно, чтобы сбой Redis не создавал каскад рестартов.
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Starting")
    return {"ready": True}


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/drafts", response_model=DraftResponse)
async def create_draft(payload: DraftRequest, request: Request, operator_id: str = Depends(authorize)) -> DraftResponse:
    request_id = str(uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    with LATENCY.time():
        try:
            async with asyncio.timeout(request.app.state.settings.total_timeout_seconds):
                result = await request.app.state.orchestrator.draft(payload, operator_id, request_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Trip access denied") from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403, 404):
                return fallback(request_id, "dependency_access_denied")
            FAILURES.labels(stage="http").inc()
            result = fallback(request_id, "dependency_error")
        except (TimeoutError, httpx.TimeoutException):
            FAILURES.labels(stage="timeout").inc()
            result = fallback(request_id, "timeout")
        except Exception as exc:
            # Не логируем str(exc), промпты, IDs поездок и HTTP bodies с PII.
            log.error("draft_failed", error_type=type(exc).__name__)
            FAILURES.labels(stage="pipeline").inc()
            result = fallback(request_id, "pipeline_error")
    REQUESTS.labels(action=result.action).inc()
    log.info("draft_finished", action=result.action, reason=result.reason)
    return result


@app.post("/v1/feedback", status_code=202)
async def feedback(event: FeedbackEvent, request: Request, operator_id: str = Depends(authorize)) -> dict[str, str]:
    from ingestion.producer import publish_event
    # Kafka используется только этим отдельным endpoint, не /v1/drafts.
    try:
        await publish_event(request.app.state.settings.kafka_bootstrap, "operator-feedback", event.event_id,
                            event.model_dump(mode="json"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Feedback queue unavailable; retry same event_id") from exc
    return {"status": "queued"}
