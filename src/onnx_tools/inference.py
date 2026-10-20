import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXNLI:
    def __init__(self, directory: str):
        path = Path(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.session = ort.InferenceSession(str(path / "model.onnx"), providers=["CPUExecutionProvider"])
        labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
        indices = [int(i) for label, i in labels.items() if "entail" in label.lower()]
        if len(indices) != 1:
            raise ValueError("Expected one semantic entailment class")
        self.entailment_index = indices[0]

    def predict(self, premise: str, hypothesis: str) -> float:
        batch = self.tokenizer(premise, hypothesis, return_tensors="np", padding=True,
                               truncation="only_first", max_length=512)
        inputs = {node.name: batch[node.name].astype(np.int64) for node in self.session.get_inputs()}
        outputs = self.session.run(["logits"], inputs)
        logits = outputs[0][0].astype(np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return float(probabilities[self.entailment_index])

    async def entailment(self, premise: str, hypothesis: str) -> float:
        import asyncio
        return await asyncio.to_thread(self.predict, premise, hypothesis)
