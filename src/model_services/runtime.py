import os
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def device() -> str:
    return os.getenv("MODEL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class E5Encoder:
    def __init__(self):
        name = os.getenv("MODEL_ID", "intfloat/multilingual-e5-large")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device()).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], kind: str) -> list[list[float]]:
        # Префиксы нужны даже для русского текста: они задают роли двух
        # сторон contrastive learning. Усредняем только токены attention_mask.
        batch = self.tokenizer([f"{kind}: {t}" for t in texts], padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device())
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().tolist()


class CrossEncoder:
    def __init__(self):
        name = os.getenv("MODEL_ID", "BAAI/bge-reranker-v2-m3")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device()).eval()

    @torch.inference_mode()
    def score(self, query: str, texts: list[str]) -> list[float]:
        batch = self.tokenizer([[query, text] for text in texts], padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device())
        # BGE отдаёт один logit на пару. Sigmoid сохраняет порядок рангов.
        return self.model(**batch).logits.float().view(-1).sigmoid().cpu().tolist()


def entailment_id(config) -> int:
    labels = {str(label).lower(): int(index) for label, index in config.label2id.items()}
    for label, index in labels.items():
        if "entail" in label:
            return index
    # Не предполагаем позицию entailment: другой checkpoint может поменять
    # порядок классов. При LABEL_0 нужны явные метаданные, иначе fail closed.
    raise ValueError(f"Missing semantic entailment label: {labels}")


class TransformersNLI:
    def __init__(self):
        name = os.getenv("MODEL_ID", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device()).eval()
        self.entailment_index = entailment_id(self.model.config)

    @torch.inference_mode()
    def predict(self, premise: str, hypothesis: str) -> float:
        batch = self.tokenizer(premise, hypothesis, truncation="only_first", padding=True,
                               max_length=512, return_tensors="pt").to(device())
        return float(self.model(**batch).logits.float().softmax(-1)[0, self.entailment_index].item())
