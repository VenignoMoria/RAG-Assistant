import argparse
import asyncio
import json
from pathlib import Path
import httpx
import mlflow
from copilot.clients.vllm import VLLMClient
from copilot.domain import DraftRequest, RouteDecision, Trip
from copilot.routing import ModelRouter


def metrics(rows: list[dict], predictions: list[RouteDecision | None]) -> dict[str, float]:
    count = len(rows)
    if count == 0 or len(predictions) != count:
        raise ValueError("Empty or mismatched evaluation")
    correct = ambiguous = recalled = wrong = unnecessary = nonclarify = valid = 0
    for row, prediction in zip(rows, predictions, strict=True):
        target = RouteDecision.model_validate(row["target"])
        ambiguous += target.action == "clarify"
        nonclarify += target.action != "clarify"
        if prediction is None:
            continue
        valid += 1
        correct += prediction.action == target.action
        recalled += target.action == "clarify" and prediction.action == "clarify"
        unnecessary += target.action != "clarify" and prediction.action == "clarify"
        # Считаем любую выдачу лишнего ID, в том числе существующего в поездке,
        # ошибкой; подмножество верных ID отдельно отражает неполный recall.
        wrong += bool(set(prediction.ticket_ids) - set(target.ticket_ids))
    return {"action_accuracy": correct / count, "ambiguous_recall": recalled / max(1, ambiguous),
            "wrong_ticket_rate": wrong / count, "unnecessary_clarification_rate": unnecessary / max(1, nonclarify),
            "json_validity": valid / count, "sample_count": float(count)}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--lora-url", required=True)
    parser.add_argument("--baseline-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-model", default="router-merged")
    parser.add_argument("--data", default="training/data/sample_val.jsonl")
    parser.add_argument("--output", default="artifacts/router-comparison.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    results = {}
    async with httpx.AsyncClient(timeout=60) as http:
        for name, url, model in [("baseline", args.baseline_url, args.baseline_model),
                                  ("lora", args.lora_url, args.lora_model)]:
            router = ModelRouter(VLLMClient(http, url, model))
            predictions = []
            for row in rows:
                try:
                    predictions.append(await router.route(DraftRequest.model_validate(row["request"]),
                                                          Trip.model_validate(row["trip"])))
                except (httpx.RequestError, httpx.HTTPStatusError):
                    raise  # Сбой serving нельзя выдавать за низкое качество JSON.
                except (ValueError, KeyError, IndexError):
                    predictions.append(None)
            results[name] = metrics(rows, predictions)
            mlflow.set_experiment("support-copilot-router-eval")
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", model)
                mlflow.log_metrics(results[name])
                mlflow.log_artifact(args.data)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
