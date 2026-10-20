# Build context должен содержать проверенные artifacts/router-merged.
FROM vllm/vllm-openai:v0.15.0
COPY artifacts/router-merged /opt/router-model
CMD ["--model", "/opt/router-model", "--served-model-name", "Qwen/Qwen2.5-1.5B-Instruct", "--max-model-len", "4096", "--tensor-parallel-size", "1", "--host", "0.0.0.0", "--port", "8000"]
