from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI
from .runtime import CrossEncoder
from .schemas import RerankRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = CrossEncoder()
    app.state.lock = Lock()
    yield


app = FastAPI(title="reranker", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": hasattr(app.state, "engine")}


@app.post("/rerank")
def infer(payload: RerankRequest) -> dict:
    # sync endpoint исполняется в threadpool. Lock ограничивает одновременное
    # использование модели; для throughput нужен отдельный batching scheduler.
    with app.state.lock:
        return {"scores": app.state.engine.score(payload.query, payload.texts)}
