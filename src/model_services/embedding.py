from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI
from .runtime import E5Encoder
from .schemas import EmbedRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = E5Encoder()
    app.state.lock = Lock()
    yield


app = FastAPI(title="embedding", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": hasattr(app.state, "engine")}


@app.post("/embed")
def infer(payload: EmbedRequest) -> dict:
    # sync endpoint исполняется в threadpool. Lock ограничивает одновременное
    # использование модели; для throughput нужен отдельный batching scheduler.
    with app.state.lock:
        return {"vectors": app.state.engine.encode(payload.texts, payload.kind)}
