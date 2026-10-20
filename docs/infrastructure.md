# Инфраструктура и эксплуатационные границы

Есть независимые образы и Deployments FastAPI, generator, router, embedding,
reranker, NLI и ingestion. Feedback использует образ ingestion, отдельный mode
и consumer group. Service создаётся для HTTP компонентов. ConfigMap содержит
публичные настройки, Secret example — только placeholders. Их нужно заменить;
никаких действующих паролей в репозитории нет.

## GPU и vLLM

Generator Deployment: 2 replicas, каждая запрашивает `nvidia.com/gpu: 1`, nodeSelector
NVIDIA-A100-SXM4-80GB. Это две независимые Qwen 14B реплики, а не tensor-parallel
модель на двух GPU. Router: отдельный Pod с одной NVIDIA-L4 (24 GB). Реальные
значения node labels зависят от GPU Feature Discovery, проверьте их на кластере.
Нужны драйвер, NVIDIA device plugin и минимум 2 A100 + 1 L4 доступных scheduler.
Embedding/reranker/NLI по умолчанию CPU; профиль ресурсных ограничений примерный.

vLLM хранит KV-cache attention для уже обработанных токенов: не пересчитывает
весь префикс при каждом новом токене. Continuous batching добавляет и убирает
запросы на шагах декодирования, повышая загрузку GPU. Paged KV memory снижает
фрагментацию, но длинные контексты и concurrency всё равно потребляют память.
`max-model-len`, gpu-memory-utilization и batch limits нужно измерять вместе.
Здесь нет обещаний throughput/latency, запас памяти не измерялся.

GPU Deployment использует maxSurge=0, maxUnavailable=1, чтобы обновление не
требовало дополнительной GPU. Generator временно работает на одной реплике,
router с одной репликой испытывает downtime. Rolling update не гарантирует
zero downtime. FastAPI maxSurge=1/maxUnavailable=0 обновляется независимо.

Readiness FastAPI проверяет инициализацию приложения, liveness — процесс.
Model startupProbe даёт время загрузить веса. Worker readiness marker создаётся
после подключения Kafka, liveness проверяет PID 1; это базовый probe, он не
обнаруживает все зависания. Kafka lag и timestamp последнего poll требуют
отдельного production мониторинга. Пробы не заменяют smoke test инференса.

RWX PVC хранит HF cache и должен быть подготовлен доступным storage provider;
permissions для vLLM root/non-root образов нужно проверить. Cache mount не
содержит уже скачанных моделей. В release образе router weights встроены,
image digest связывает serving код и модель. Базовые model images скачивают
checkpoint по имени и поэтому сами по себе не дают immutable model revision.

## Kafka и ingestion

Producer key=document_id сохраняет порядок обновлений одного документа в partition.
Один consumer group copilot-knowledge-v1 имеет исключительное право писать KB;
consumer group распределяет partitions между worker экземплярами. Число активных
consumer ограничено partitions. Несколько независимых writer groups нарушают
предположение о последовательности. Изменять partition count без миграции keys опасно.

Consumer извлекает plain text/HTML, удаляет script/style, делит текст с overlap,
кодирует passage embeddings и bulk пишет ES. Повторные версии пропускаются по
marker, повтор chunks идемпотентен по ID. Отсутствует транзакция ES/Kafka: доставка
at-least-once, crash после ES и до commit приводит к replay. После трёх попыток
DLQ должен подтвердить запись; если DLQ недоступен, offset не коммитится.
DLQ содержит raw payload и нуждается в ACL/retention/redaction, UI для него нет.
PDF/Office extraction не реализована и не заявляется поддержанной.

Feedback сохраняется отдельным consumer в ES, create(event_id) обрабатывает
повтор через conflict. Он не обучает модель автоматически. Для тренировки нужен
экспертный review, отдельная подготовка и контролируемый release.

## ONNX и наблюдаемость

Optimum экспортирует sequence-classification NLI checkpoint, не Qwen 14B.
ONNX Runtime CPU adapter загружает tokenizer, model.onnx и label2id metadata из
одной директории. Одинаковая токенизация, truncation, порядок классов и softmax
обязательны. Для смены backend задайте NLI_BACKEND=onnx и ONNX_MODEL_DIR; image
потребует группы зависимостей onnx и смонтированных экспортированных файлов.
Нужен parity test Transformers/ONNX на фиксированных парах до релиза; здесь он
не запускался. Поддержка ops/export выбранного checkpoint требует проверки.

Prometheus получает request counts по action, latency histogram и failures stage;
vLLM предоставляет собственный /metrics. Config в deploy/prometheus — пример
scrape, а не настроенный кластер мониторинга. Логи structlog JSON без PII.
Внешние ES, Redis, Kafka, Order API, MLflow, registry, ingress/TLS, autoscaling,
quotas, RBAC, backups и alerting оператор должен подготовить отдельно.
