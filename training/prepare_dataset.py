import json
from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from copilot.domain import DraftRequest, RouteDecision, Trip
from copilot.routing import ROUTER_SYSTEM


def prepare(path: str, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        request = DraftRequest.model_validate(row["request"])
        trip = Trip.model_validate(row["trip"])
        target = RouteDecision.model_validate(row["target"])
        if not set(target.ticket_ids) <= {t.ticket_id for t in trip.tickets}:
            raise ValueError("Target uses unknown tickets")
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": json.dumps({"request": request.model_dump(), "trip": trip.model_dump()},
                                                   ensure_ascii=False)},
        ]
        # Qwen template добавляет assistant prefix. Prompt/completion формат
        # позволяет TRL маскировать весь prompt, не требуя {% generation %}
        # блоков, которых может не быть в исходном chat template Qwen.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target.model_dump_json()}],
                                             tokenize=False, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError("Chat template does not preserve assistant prefix")
        if len(tokenizer(full, add_special_tokens=False)["input_ids"]) > max_length:
            raise ValueError("Example exceeds max_length; do not truncate target JSON")
        result.append({"prompt": prompt, "completion": full[len(prompt):]})
    return Dataset.from_list(result)
