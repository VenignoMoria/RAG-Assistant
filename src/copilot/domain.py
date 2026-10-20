from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Vertical(StrEnum):
    avia = "avia"
    rail = "rail"
    bus = "bus"
    common = "common"


class Ticket(StrictModel):
    ticket_id: str
    vertical: Literal["avia", "rail", "bus"]
    route: str
    departure_date: str
    status: str


class Trip(StrictModel):
    trip_id: str
    tickets: list[Ticket]

    @model_validator(mode="after")
    def unique_tickets(self) -> "Trip":
        if len({t.ticket_id for t in self.tickets}) != len(self.tickets):
            raise ValueError("Duplicate ticket IDs")
        return self


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class DraftRequest(StrictModel):
    trip_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=20)
    selected_ticket_ids: list[str] = Field(default_factory=list, max_length=20)


class RouteDecision(StrictModel):
    action: Literal["clarify", "general", "retrieve", "fallback"]
    query: str = Field(max_length=2000)
    ticket_ids: list[str] = Field(default_factory=list, max_length=20)
    question: str | None = Field(default=None, max_length=1000)
    reason: str = Field(max_length=1000)


class Chunk(StrictModel):
    chunk_id: str
    document_id: str
    version: int = Field(ge=1)
    title: str
    text: str
    vertical: Vertical
    public: bool = True
    source: str
    score: float = 0.0


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=1500)
    citation_ids: list[str] = Field(min_length=1, max_length=3)


class GeneratedDraft(StrictModel):
    # Итоговый текст собирается из проверенных утверждений, поэтому модель не
    # может спрятать неподтверждённую фразу в отдельном поле свободного ответа.
    status: Literal["draft", "insufficient"] = "draft"
    answer: str = Field(max_length=6000)
    claims: list[Claim] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def consistent_status(self) -> "GeneratedDraft":
        if self.status == "draft" and (not self.answer.strip() or not self.claims):
            raise ValueError("Draft requires answer and claims")
        if self.status == "insufficient" and (self.answer or self.claims):
            raise ValueError("Abstention must not contain an unverified answer")
        return self


class DraftResponse(StrictModel):
    request_id: str
    action: Literal["clarify", "general", "retrieve", "fallback"]
    answer: str
    citations: list[Chunk] = Field(default_factory=list)
    ticket_ids: list[str] = Field(default_factory=list)
    requires_operator_approval: Literal[True] = True
    reason: str


class FeedbackEvent(StrictModel):
    event_id: str
    request_id: str
    outcome: Literal["accepted", "edited", "rejected"]
    reason_code: str = Field(max_length=100)
    # Событие намеренно не содержит текст переписки, исправленный ответ и PII.


class KnowledgeEvent(StrictModel):
    event_id: str
    document_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    version: int = Field(ge=1)
    title: str = Field(max_length=300)
    content: str = Field(max_length=2000000)
    content_type: Literal["text/plain", "text/html"] = "text/plain"
    vertical: Vertical
    public: bool = True
    source: str
