import pytest
from copilot.domain import Claim, GeneratedDraft, Trip
from copilot.verification import DraftVerifier
from .test_retrieval_cache import chunk


class NLI:
    async def entailment(self, premise: str, hypothesis: str) -> float:
        return 0.99


@pytest.mark.asyncio
async def test_unknown_citation_fails_even_when_nli_is_high():
    draft = GeneratedDraft(answer="Правило.", claims=[Claim(text="Правило.", citation_ids=["invented"])])
    assert not await DraftVerifier(NLI(), 0.8).verify(draft, [chunk("one")], Trip(trip_id="t", tickets=[]))


@pytest.mark.asyncio
async def test_unsupported_number_fails():
    draft = GeneratedDraft(answer="Возврат за 90 дней.",
                           claims=[Claim(text="Возврат за 90 дней.", citation_ids=["one"])])
    assert not await DraftVerifier(NLI(), 0.8).verify(draft, [chunk("one")], Trip(trip_id="t", tickets=[]))


@pytest.mark.asyncio
async def test_free_answer_cannot_bypass_claims():
    draft = GeneratedDraft(answer="Правило. Дополнительное обещание.",
                           claims=[Claim(text="Правило.", citation_ids=["one"])])
    assert not await DraftVerifier(NLI(), 0.8).verify(draft, [chunk("one")], Trip(trip_id="t", tickets=[]))


@pytest.mark.asyncio
async def test_order_status_is_not_inferred_from_knowledge():
    draft = GeneratedDraft(answer="Билет отменён.", claims=[Claim(text="Билет отменён.", citation_ids=["one"])])
    assert not await DraftVerifier(NLI(), 0.8).verify(draft, [chunk("one")], Trip(trip_id="t", tickets=[]))


@pytest.mark.asyncio
async def test_empty_sentence_sequence_fails_closed():
    draft = GeneratedDraft(answer=";", claims=[Claim(text=";", citation_ids=["one"])])
    assert not await DraftVerifier(NLI(), 0.8).verify(draft, [chunk("one")], Trip(trip_id="t", tickets=[]))


@pytest.mark.asyncio
async def test_explicit_abstention_is_not_a_verified_draft():
    draft = GeneratedDraft(status="insufficient", answer="", claims=[])
    assert not await DraftVerifier(NLI(), 0.8).verify(draft, [chunk("one")], Trip(trip_id="t", tickets=[]))
