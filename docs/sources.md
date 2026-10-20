# Документация API, использованная как ориентир

Это ссылки на первичные источники, а не утверждение о выполненном интеграционном тесте.
Версии API и образов зафиксированы в pyproject/Dockerfiles там, где это важно.

* [TRL 0.26.0 SFTTrainer](https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer) —
  processing_class, prompt/completion datasets, completion_only_loss, max_length.
* [vLLM 0.15.0 structured outputs](https://docs.vllm.ai/en/v0.15.0/features/structured_outputs/) —
  structured_outputs.json для OpenAI-compatible HTTP API.
* [Optimum ONNX examples](https://github.com/huggingface/optimum-onnx/tree/main/examples/onnxruntime) —
  отдельная export/runtime ветка для sequence classification.
* [multilingual-e5-large model card](https://huggingface.co/intfloat/multilingual-e5-large) —
  query/passage prefixes, masked mean pooling и нормализация.
* [bge-reranker-v2-m3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) —
  cross-encoder scoring query/document pairs.

Перед разрешением зависимостей сверяйте документацию именно выбранных версий.
