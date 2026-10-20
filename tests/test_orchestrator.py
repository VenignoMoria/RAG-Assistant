import pytest
from copilot.domain import DraftRequest, GeneratedDraft, Claim, RouteDecision
from copilot.orchestrator import Orchestrator
from copilot.verification import DraftVerifier
from .test_routing import trip
from .test_retrieval_cache import chunk
from .test_verification import NLI


class Orders:
    async def get_trip(self, trip_id, operator_id):
        return trip()


class Router:
    async def route(self, request, selected_trip):
        return RouteDecision(action="retrieve", query="возврат", reason="test")


class Retrieval:
    def __init__(self):
        self.called = False

    async def search(self, query, vertical):
        self.called = True
        return [chunk("one")]


class Generator:
    async def generate(self, query, selected_trip, chunks):
        return GeneratedDraft(answer="Правило.", claims=[Claim(text="Правило.", citation_ids=["one"])])


class NoCache:
    def key(self, *args):
        return None


@pytest.mark.asyncio
async def test_clarification_stops_retrieval():
    retrieval = Retrieval()
    orchestrator = Orchestrator(Orders(), Router(), retrieval, Generator(), DraftVerifier(NLI(), 0.8), NoCache())
    result = await orchestrator.draft(DraftRequest(trip_id="trip", message="Как вернуть билет?"), "operator", "req")
    assert result.action == "clarify"
    assert not retrieval.called
    assert result.requires_operator_approval is True


@pytest.mark.asyncio
async def test_selected_ticket_produces_draft_with_citations():
    orchestrator = Orchestrator(Orders(), Router(), Retrieval(), Generator(), DraftVerifier(NLI(), 0.8), NoCache())
    result = await orchestrator.draft(DraftRequest(trip_id="trip", message="Возврат", selected_ticket_ids=["r1"]),
                                      "operator", "req")
    assert result.action == "retrieve"
    assert result.ticket_ids == ["r1"]
    assert [c.chunk_id for c in result.citations] == ["one"]
    assert result.requires_operator_approval is True
