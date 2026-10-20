FROM vllm/vllm-openai:v0.15.0
ENV HF_HOME=/models/huggingface
CMD ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--host", "0.0.0.0", "--port", "8000", "--tensor-parallel-size", "1", "--max-model-len", "4096", "--gpu-memory-utilization", "0.90"]
