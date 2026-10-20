import pytest
from copilot.domain import DraftRequest, RouteDecision, Ticket, Trip
from copilot.routing import validate_route


def trip() -> Trip:
    return Trip(trip_id="trip", tickets=[
        Ticket(ticket_id="r1", vertical="rail", route="A-B", departure_date="2030-01-01", status="issued"),
        Ticket(ticket_id="a1", vertical="avia", route="B-C", departure_date="2030-01-02", status="issued"),
    ])


def test_unknown_ticket_is_rejected():
    request = DraftRequest(trip_id="trip", message="Возврат")
    route = RouteDecision(action="retrieve", query="возврат", ticket_ids=["outside"], reason="test")
    with pytest.raises(ValueError):
        validate_route(route, request, trip())


def test_ambiguous_multi_ticket_needs_clarification():
    request = DraftRequest(trip_id="trip", message="Как вернуть билет?")
    route = RouteDecision(action="retrieve", query="возврат", ticket_ids=[], reason="test")
    assert validate_route(route, request, trip()).action == "clarify"


def test_valid_but_guessed_ticket_is_not_enough():
    request = DraftRequest(trip_id="trip", message="Как вернуть билет?")
    route = RouteDecision(action="retrieve", query="возврат", ticket_ids=["r1"], reason="guess")
    assert validate_route(route, request, trip()).action == "clarify"


def test_general_does_not_require_selection():
    request = DraftRequest(trip_id="trip", message="Общие правила возврата")
    route = RouteDecision(action="general", query="общие правила возврата", reason="general")
    assert validate_route(route, request, trip()).action == "general"


def test_mixed_selection_does_not_merge_verticals():
    request = DraftRequest(trip_id="trip", message="Возврат", selected_ticket_ids=["r1", "a1"])
    route = RouteDecision(action="retrieve", query="возврат", ticket_ids=["r1", "a1"], reason="test")
    assert validate_route(route, request, trip()).action == "clarify"


def test_router_cannot_override_operator():
    request = DraftRequest(trip_id="trip", message="Возврат", selected_ticket_ids=["r1"])
    route = RouteDecision(action="retrieve", query="возврат", ticket_ids=["a1"], reason="test")
    with pytest.raises(ValueError):
        validate_route(route, request, trip())


def test_model_cannot_downgrade_ambiguous_refund_to_general():
    request = DraftRequest(trip_id="trip", message="Как вернуть билет?")
    route = RouteDecision(action="general", query="общие правила", reason="wrong classification")
    assert validate_route(route, request, trip()).action == "clarify"


def test_model_cannot_drop_other_explicit_vertical():
    request = DraftRequest(trip_id="trip", message="Вернуть поезд и самолёт")
    route = RouteDecision(action="retrieve", query="возврат", ticket_ids=["r1"], reason="partial")
    assert validate_route(route, request, trip()).action == "clarify"
