FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/models/huggingface
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[models]" && useradd --uid 10001 --create-home app && mkdir /models && chown app /models
USER 10001
EXPOSE 8000
CMD ["uvicorn", "model_services.nli:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
