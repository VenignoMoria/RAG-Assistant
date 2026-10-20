import logging
import structlog
from prometheus_client import Counter, Histogram

REQUESTS = Counter("copilot_requests_total", "Draft requests", ["action"])
LATENCY = Histogram("copilot_request_seconds", "Draft latency")
FAILURES = Counter("copilot_failures_total", "Dependency or validation failures", ["stage"])


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ], logger_factory=structlog.stdlib.LoggerFactory())
