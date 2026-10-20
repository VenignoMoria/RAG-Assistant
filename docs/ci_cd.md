# CI, CD и release моделей

CI отвечает на вопрос «соответствует ли изменение проверкам?». CD доставляет
конкретный проверенный артефакт в окружение. Созданные workflows не запускались.

`ci.yml`: push main / PR → Python 3.11 → установка .[dev] → Ruff → AST/YAML/JSON
→ pytest с fakes → Bandit → pip-audit. Это реальные проверки; уязвимость или
конфликт версий должны останавливать workflow, их нельзя игнорировать ради зелёного статуса.
Unit tests не требуют GPU, ES, Kafka или загруженных моделей. Они не заменяют
интеграционные проверки сервиса и GPU compatibility.

`fastapi-cd.yml` срабатывает только после успешного CI push в main; checkout
делается по head_sha этого CI. Buildx собирает исключительно FastAPI Dockerfile,
публикует GHCR image, deployment использует image@digest. В cluster вызывается
только `kubectl set image deployment/fastapi`, затем rollout status. Qwen Pods
не затрагиваются. Для требования «после merge» настройте protected main и запрет
прямых push: workflow сам не подменяет branch protection.

`model-evaluation.yml` запускается вручную на доверенном self-hosted runner.
Он обращается к заранее подготовленным baseline/candidate router endpoints,
сохраняет отчёт и provenance SHA/digest в artifact. Начальные quality gates
action_accuracy>=0.9, wrong_ticket_rate<=0, json_validity=1 иллюстративны и требуют
согласования; учебных val примеров недостаточно для release решения.
URL endpoints — доверенные operator inputs, runner не должен принимать PR forks.

`model-release.yml` — ручной контролируемый release **router serving**. Проверяет
успешный Model evaluation run, workflow event, commit SHA, совпадение candidate
digest и скачанного artifact; повторяет quality gate, ждёт environment approval,
обновляет только router. В GitHub нужно настроить required reviewers для
production-models. Сама YAML строка environment без reviewers не создаёт review.

Кандидат должен быть собран с реальными merged weights через
docker/router-release.Dockerfile; перед evaluation ему назначается то же
served-model-name, что будет использовать API. Digest binding отражает декларацию
оператора о запущенном кандидате: workflow не может доказать, что произвольный
endpoint действительно использует указанный image. Нужен доверенный staging
deploy/attestation для более строгой supply-chain гарантии.

Для generator, embedding, reranker, NLI нельзя слепо применять router accuracy
gate. Этот workflow намеренно не выпускает их: расширяйте отдельными retrieval,
RAGAS/NLI/parity reports и approvals. RAGAS evaluator включён как отдельная CLI
оценка; он не запускается без явно настроенных judge и embeddings endpoints.

Self-hosted runners copilot-evaluation/copilot-deploy, kubectl/gh, GHCR access,
KUBE_CONFIG/KUBE_CONFIG_MODELS и environments настраиваются владельцем.
Deployment credential backend должен иметь RBAC только на patch/get/watch
deployment/fastapi, model credential — только router. Кластерные YAML применяются
отдельно при provisioning; CD не делает `kubectl apply -f deploy/k8s`.
Rollback: выбрать известный хороший digest или `kubectl rollout undo` для одного
Deployment, повторить health/quality smoke check. Автоматический rollback пока не реализован.
