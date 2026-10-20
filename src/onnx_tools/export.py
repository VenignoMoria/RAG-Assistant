import argparse
import json
from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--output", default="artifacts/nli-onnx")
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True,
                                                             provider="CPUExecutionProvider")
    model.save_pretrained(target)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(target)
    # Сохраняем семантику классов рядом с графом, а не зашиваем index=0.
    (target / "labels.json").write_text(json.dumps(model.config.label2id), encoding="utf-8")


if __name__ == "__main__":
    main()
