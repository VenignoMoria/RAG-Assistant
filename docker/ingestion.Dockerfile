FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home app
USER 10001
CMD ["python", "-m", "ingestion.worker"]
