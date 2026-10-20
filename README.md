# Support Copilot RAG

Учебный monorepo системы подготовки **черновиков для операторов** поддержки
туристического сервиса. Предметная область напоминает билетные сервисы, но
проект не связан с Туту, его API, данными или инфраструктурой. Все поездки,
документы и примеры синтетические. Результатов обучения и замеров production нет.

Система принимает доверенный `trip_id` от backend оператора, получает билеты через
Order API, выбирает `clarify / general / retrieve / fallback`, ищет правила и
формирует проверяемый черновик с цитатами. Конечного клиента здесь нет:
**оператор обязан утвердить черновик перед отправкой**, endpoint отправки отсутствует.

## Как читать проект

1. `src/copilot/domain.py` — контракты Pydantic v2; `ports.py` — границы компонентов.
2. `src/copilot/api.py` → `orchestrator.py` — сборка зависимостей и обработка запроса.
3. `routing.py` — Qwen router и детерминированные проверки билетов.
4. `retrieval.py` → `generation.py` → `verification.py` — весь путь RAG.
5. `src/model_services/` — E5, BGE, NLI; `src/onnx_tools/` — CPU вариант NLI.
6. `training/` и `evaluation/` — QLoRA, baseline, RAGAS и MLflow.
7. `src/ingestion/` — Kafka, chunking, ES versions и feedback.
8. `docker/`, `deploy/`, `.github/workflows/` — независимые жизненные циклы backend и моделей.

Подробности: [архитектура](docs/architecture.md), [поток запроса](docs/request_flow.md),
[retrieval](docs/retrieval.md), [обучение](docs/lo_ra_training.md),
[инфраструктура](docs/infrastructure.md), [CI/CD](docs/ci_cd.md), [ограничения](docs/limitations.md).

## Среда и зависимости

Python 3.11+, FastAPI, Pydantic v2, HTTPX, LangChain Core, Elasticsearch 8,
Redis, Kafka, PyTorch, Transformers, PEFT, TRL, bitsandbytes, vLLM,
Optimum ONNX, ONNX Runtime, RAGAS, MLflow и Prometheus.
Группы зависимостей разделены: CPU backend не устанавливает CUDA/training стек.
Указаны диапазоны и отдельные фиксированные API targets, **это не проверенный lockfile**.
TRL код ориентирован на 0.26.0, RAGAS на 0.2.15, vLLM image на 0.15.0.
Перед реальным запуском разрешите зависимости, проверьте CUDA/драйвер и зафиксируйте lock и image digests.

Ниже команды для самостоятельного изучения. При создании проекта они не выполнялись.

```bash
python -m venv .venv
# Активируйте .venv подходящей для вашей ОС командой.
python -m pip install -e '.[dev]'
cp .env.example .env
# Замените COPILOT_SERVICE_TOKEN случайным значением.
python scripts/check_syntax.py
pytest -q
```

Для model services: `pip install -e '.[models]'`; для QLoRA — `.[training]` в отдельной
GPU среде; для ONNX — `.[onnx]`; для RAGAS — отдельная среда `.[evaluation]`.
Разделение важно из-за различающихся constraints Transformers/LangChain/CUDA.
Скрипты запускайте из корня через `python -m ...` после установки editable package.

## Локальная инфраструктура

```bash
docker compose up -d elasticsearch redis kafka kafka-init
python -m ingestion.index
python -m ingestion.producer data/knowledge/refund_common.json
python -m ingestion.worker
```

При запуске Python с хоста задайте ES `http://localhost:9200`, Redis
`redis://localhost:6379/0`, Kafka `localhost:29092`; внутри Compose используются
имена сервисов. Прочие URLs в `.env.example` предназначены для container/Kubernetes DNS.
Сначала запустите E5 (`uvicorn model_services.embedding:app --port 8001`), затем
укажите `COPILOT_EMBEDDING_URL=http://localhost:8001` для ingestion. Скачивание
моделей при первом запуске требует сети и свободного места; здесь оно не выполнялось.

Compose поднимает локальные ES/Redis/Kafka. Профиль `app` добавляет только FastAPI,
а URLs остальных model services нужно настроить самостоятельно. Это **не воспроизведение GPU-кластера**.
Локальные plaintext порты привязаны к loopback, конфигурация не предназначена для публикации в сеть.

Для ручного model serving используйте соответствующие Dockerfile или отдельные GPU среды:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8010 --tensor-parallel-size 1 --max-model-len 8192
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8011 --max-model-len 4096
uvicorn model_services.reranker:app --port 8002
uvicorn model_services.nli:app --port 8003
uvicorn copilot.api:app --host 127.0.0.1 --port 8000
```

Переменные orchestrator для этого примера: generator `http://localhost:8010/v1`,
router `http://localhost:8011/v1`, embedding `http://localhost:8001`, reranker
`http://localhost:8002`, NLI `http://localhost:8003`. У двух vLLM процессов должны
быть разные выделенные GPU через `CUDA_VISIBLE_DEVICES`; эти команды не распределяют GPU автоматически.

## Пример API

```bash
curl http://localhost:8000/v1/drafts \
  -H 'Authorization: Bearer YOUR_LOCAL_TOKEN' \
  -H 'X-Operator-ID: demo-operator' -H 'Content-Type: application/json' \
  -d '{"trip_id":"demo-trip","message":"Как вернуть билет?","selected_ticket_ids":[]}'
```

Ожидаемая семантика — уточнение билета; реальный ответ router нужно подтвердить
после запуска модели. Для выбранного ж/д билета передайте `selected_ticket_ids:["rail-1"]`.
Общие правила допускают поиск без выбора. Любая ошибка зависимостей, JSON,
цитат или проверок приводит к явному fallback. Документов rail/avia в примере БЗ
нет: добавьте синтетические события по контракту `KnowledgeEvent`.

Feedback: `POST /v1/feedback` с `event_id`, `request_id`, `outcome`, `reason_code`.
API не принимает исправленный текст с персональными данными. При недоступной очереди
возвращается 503; повтор выполняется с тем же event_id.

## Обучение и оценка

```bash
python -m training.generate_synthetic_data
python -m training.train_qlora --config training/config.yaml
python -m training.merge_adapter
python -m training.evaluate_router --baseline-url http://baseline:8000/v1 \
  --lora-url http://candidate:8000/v1 --lora-model router-merged
python -m onnx_tools.export --output artifacts/nli-onnx
```

Сервисы baseline/candidate запускаются отдельно; evaluator не запускает GPU серверы.
Файлы sample_train/sample_val — примеры формата, а не достаточная обучающая выборка.
`evaluation/data/schema_example.jsonl` иллюстрирует RAGAS формат и не является benchmark.

## Что проверено при создании

Смотрите `docs/validation.md`: отдельно указаны AST Python, YAML и JSON.
Unit/integration tests, разрешение зависимостей, обучение, инференс, сборки,
security scans, workflows и инфраструктура при создании **не запускались**.
Файлы тестов включены для последующего запуска в подготовленной среде.
