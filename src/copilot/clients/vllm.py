import json
from typing import TypeVar
import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class VLLMClient:
    """Общий HTTP транспорт без tool calling и доступа модели к Order API."""
    def __init__(self, http: httpx.AsyncClient, url: str, model: str):
        self.http, self.url, self.model = http, url.rstrip("/"), model

    async def structured(self, messages: list[dict[str, str]], schema: type[T], max_tokens: int = 1600) -> T:
        response = await self.http.post(f"{self.url}/chat/completions", json={
            "model": self.model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens,
            # Это расширение vLLM в теле HTTP, не устаревший guided_json.
            "structured_outputs": {"json": schema.model_json_schema()},
        })
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete structured generation")
        content = choice["message"]["content"]
        return schema.model_validate(json.loads(content))
