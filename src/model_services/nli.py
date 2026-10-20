from contextlib import asynccontextmanager
import os
from threading import Lock
from fastapi import FastAPI
from .schemas import NLIRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend = os.getenv("NLI_BACKEND", "transformers")
    if backend == "onnx":
        from onnx_tools.inference import ONNXNLI
        app.state.engine = ONNXNLI(os.environ["ONNX_MODEL_DIR"])
    elif backend == "transformers":
        from .runtime import TransformersNLI
        app.state.engine = TransformersNLI()
    else:
        raise ValueError("Unsupported NLI_BACKEND")
    app.state.lock = Lock()
    yield


app = FastAPI(title="nli", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ready": hasattr(app.state, "engine")}


@app.post("/nli")
def infer(payload: NLIRequest) -> dict:
    # sync endpoint исполняется в threadpool. Lock ограничивает одновременное
    # использование модели; для throughput нужен отдельный batching scheduler.
    with app.state.lock:
        return {"entailment": app.state.engine.predict(payload.premise, payload.hypothesis)}
