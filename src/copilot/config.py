from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=".env", extra="ignore")
    service_token: SecretStr
    mock_orders: bool = True
    order_url: str = "http://order-api:8080"
    order_token: SecretStr = SecretStr("")
    generator_url: str = "http://generator:8000/v1"
    router_url: str = "http://router:8000/v1"
    generator_model: str = "Qwen/Qwen2.5-14B-Instruct"
    router_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_url: str = "http://embedding:8000"
    reranker_url: str = "http://reranker:8000"
    nli_url: str = "http://nli:8000"
    elasticsearch_url: str = "http://elasticsearch:9200"
    es_index: str = "knowledge-v1"
    redis_url: str = "redis://redis:6379/0"
    kb_version: str = "demo-v1"
    kafka_bootstrap: str = "kafka:9092"
    rerank_threshold: float = Field(default=0.65, ge=0, le=1)
    nli_threshold: float = Field(default=0.8, ge=0, le=1)
    cache_ttl: int = Field(default=300, gt=0)
    timeout_seconds: float = Field(default=30, gt=0)
    total_timeout_seconds: float = Field(default=90, gt=0)
