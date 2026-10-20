import math
import re
from .domain import Chunk, GeneratedDraft, Trip
from .ports import NLIPort


def atoms(text: str) -> set[str]:
    # Это консервативный лексический барьер, а не доказательство связи числа
    # с сущностью. Новые форматы дат/валют требуют расширения и тестов.
    return set(re.findall(r"\d+(?:[.,:/-]\d+)*%?", text))


def split_claims(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z])|[;\n]+", text) if part.strip()]


class DraftVerifier:
    def __init__(self, nli: NLIPort, threshold: float):
        self.nli, self.threshold = nli, threshold

    async def verify(self, draft: GeneratedDraft, chunks: list[Chunk], trip: Trip) -> bool:
        if draft.status != "draft" or not draft.claims:
            return False
        if draft.answer.strip() != " ".join(c.text.strip() for c in draft.claims):
            return False
        sources = {c.chunk_id: c for c in chunks}
        # Статусы заказа не выводятся из правил БЗ и не доверяются NLI.
        # В учебной версии любые утверждения о статусе направляются оператору;
        # отдельный подтверждённый Order API шаблон можно добавить позднее.
        status_pattern = r"\b(issued|refunded|cancelled|оформлен\w*|возвращ[её]н\w*|отмен[её]н\w*|оплачен\w*)\b"
        for claim in draft.claims:
            if not set(claim.citation_ids) <= sources.keys():
                return False
            evidence = [sources[cid].text for cid in claim.citation_ids]
            if re.search(status_pattern, claim.text, re.I):
                return False
            if not atoms(claim.text) <= atoms(" ".join(evidence)):
                return False
            dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", claim.text)
            allowed_dates = {t.departure_date for t in trip.tickets}
            if any(d not in allowed_dates for d in dates):
                return False
            sentences = split_claims(claim.text)
            if not sentences or any(not re.search(r"\w", sentence) for sentence in sentences):
                return False
            for sentence in sentences:
                # Проверяем поддержку хотя бы одним источником отдельно:
                # склейка правил разных документов может создавать ложный entailment.
                scores = [await self.nli.entailment(doc, sentence) for doc in evidence]
                if not any(math.isfinite(s) and s >= self.threshold for s in scores):
                    return False
        return True
