"""Оценка сохранённого RAG evaluation set с явно выбранным judge."""
import argparse
import json
from pathlib import Path
from datasets import Dataset
import mlflow
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="JSONL: question, answer, contexts, ground_truth")
    parser.add_argument("--judge-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--embedding-url", required=True, help="OpenAI-compatible embeddings endpoint")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--output", default="artifacts/ragas.json")
    args = parser.parse_args()
    # В evaluation .env указывает judge; сеть и передача данных должны быть
    # согласованы заранее. Не используем неявный платный OpenAI endpoint.
    llm = LangchainLLMWrapper(ChatOpenAI(base_url=args.judge_url, model=args.judge_model,
                                        api_key="local-evaluation", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(base_url=args.embedding_url,
                                                             model=args.embedding_model, api_key="local-evaluation",
                                                             check_embedding_ctx_length=False))
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines()]
    result = evaluate(Dataset.from_list(rows), metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                      llm=llm, embeddings=embeddings, raise_exceptions=True)
    frame = result.to_pandas()
    values = {name: float(frame[name].mean()) for name in
              ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    mlflow.set_experiment("support-copilot-rag-eval")
    with mlflow.start_run():
        mlflow.log_param("judge_model", args.judge_model)
        mlflow.log_metrics(values)
        mlflow.log_artifact(args.data)
        mlflow.log_artifact(str(target))


if __name__ == "__main__":
    main()
