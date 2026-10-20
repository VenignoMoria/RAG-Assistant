# Обучение router: LoRA / QLoRA

Обучаем Qwen2.5-1.5B-Instruct выбирать действие, учитывать контекст и переписывать
запрос. Правила возврата не заучиваются: они версионируются в БЗ и извлекаются RAG.
Qwen 14B в этом pipeline не обучается. Два sample JSONL файла содержат только
синтетические примеры, раздельные по формулировкам train/val. Они демонстрируют
формат, а не дают статистически надёжную оценку.

`generate_synthetic_data.py` создаёт cases ambiguity/general/explicit/reference/
topic_switch/mixed/multi_same/unsafe/human/missing_facts. Каждый пример содержит
DraftRequest, Trip и целевой RouteDecision. `prepare_dataset.py` проверяет схему и
ID, использует реальный chat template tokenizer Qwen, строит prompt/completion,
отклоняет слишком длинные примеры вместо обрезания JSON target.

В TRL 0.26.0 `completion_only_loss=True` маскирует prompt labels через -100.
Это исключает loss для system/user токенов без зависимости от наличия generation
блоков в chat template. Перед training проверяется, что первый batch действительно
имеет masked/unmasked labels. Другой TRL требует повторной проверки collator.

## Настройки

| Параметр | Начальное значение |
|---|---|
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| target_modules | q,k,v,o projections + gate,up,down projections |
| quantization | 4-bit NF4 + double quantization |
| compute dtype | bf16 если GPU поддерживает, иначе fp16 |
| LR / epochs | 1e-4 / 3 |
| microbatch / accumulation | 2 / 8 |
| max_length | 2048 |
| warmup / scheduler / seed | 0.03 / cosine / 42 |

LoRA учит низкоранговые добавки к замороженным матрицам; QLoRA дополнительно хранит
базовые веса в 4 битах. NF4 — codebook, рассчитанный на примерно нормальное
распределение весов. Double quantization сжимает сами quantization constants.
Вычисления идут в bf16/fp16, а не в математике 4 бит для всех операций. Это снижает
память весов, но activation memory, optimizer и sequence length остаются важны.
Effective batch на одной GPU: 2*8=16; это учебная отправная точка, не оптимум.

`train_qlora.py` использует PyTorch, Transformers BitsAndBytesConfig, PEFT,
prepare_model_for_kbit_training, TRL SFTTrainer, gradient checkpointing и MLflow.
MLflow сохраняет config, параметры, training/eval loss. Loss не заменяет метрики
маршрутизации. `merge_adapter.py` загружает базу в float32 CPU, merge выполняет
из неквантованной базы; это требует памяти и времени. Артефакты не включены в repo.

## Baseline и оценка

Запустите исходную Qwen 1.5B и merged candidate в разных serving endpoints.
`evaluate_router.py` использует одинаковые промпты, temperature=0 и одну val
выборку. Метрики считаются до backend guard: иначе guard скрывал бы ошибки модели.

* action accuracy: корректное действие / все примеры;
* ambiguous recall: верные clarify / gold clarify;
* wrong-ticket rate: хотя бы один лишний относительно gold ID / все примеры;
* unnecessary clarification rate: clarify при gold != clarify / gold != clarify;
* JSON validity: JSON, прошедший RouteDecision validation / все примеры.

Неудачный HTTP serving останавливает оценку, а не записывается как invalid JSON.
Оценку полноты ticket selection, query rewriting и сквозного guarded поведения
нужно добавить на большой выборке; одни перечисленные метрики не покрывают всё.
Результатов baseline/LoRA сейчас нет, comparison JSON создаётся только evaluator.

В настоящем проекте аналитики проектируют taxonomy и sampling сложных случаев,
эксперты поддержки размечают неоднозначность/уточнения и проверяют запросы после
rewriting. PII удаляются до разметки; нужны least privilege, срок хранения,
двойная разметка спорных примеров и adjudication. Делить нужно по диалогам,
путешествиям, времени и шаблонам, чтобы избежать leakage. Synthetic train/val
не заменяют согласованную holdout выборку реальных обезличенных сценариев.

RAGAS оценивает faithfulness, answer relevancy, context precision/recall отдельно
от router. `evaluation/rag_eval.py` принимает сохранённые результаты и reference
answers, явный judge и OpenAI-compatible embeddings endpoint. Наш `/embed` имеет
другой контракт и напрямую для RAGAS OpenAIEmbeddings не подходит: нужен отдельный
совместимый evaluator endpoint. RAGAS — шумный LLM judge, а не проверка юридической
правильности правил. MLflow связывает настройки и оценки, но не доказывает качество.
