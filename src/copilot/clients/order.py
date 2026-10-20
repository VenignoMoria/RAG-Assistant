import httpx
from ..domain import Ticket, Trip


class OrderClient:
    def __init__(self, http: httpx.AsyncClient, url: str, token: str):
        self.http, self.url, self.token = http, url.rstrip("/"), token

    async def get_trip(self, trip_id: str, operator_id: str) -> Trip:
        # Order API обязан проверить доступ оператора к trip_id. Идентификатор
        # оператора приходит из доверенного backend, а не из текста клиента.
        from urllib.parse import quote
        response = await self.http.get(
            f"{self.url}/trips/{quote(trip_id, safe='')}",
            headers={"Authorization": f"Bearer {self.token}", "X-Operator-ID": operator_id},
        )
        response.raise_for_status()
        trip = Trip.model_validate(response.json())
        if trip.trip_id != trip_id:
            raise ValueError("Order API returned another trip")
        return trip


class MockOrderClient:
    """Только синтетическое путешествие; корпоративной интеграции нет."""
    async def get_trip(self, trip_id: str, operator_id: str) -> Trip:
        if trip_id != "demo-trip" or operator_id != "demo-operator":
            raise PermissionError("Mock supports only demo operator and trip")
        return Trip(trip_id=trip_id, tickets=[
            Ticket(ticket_id="rail-1", vertical="rail", route="Город А — Город Б",
                   departure_date="2030-06-10", status="issued"),
            Ticket(ticket_id="avia-1", vertical="avia", route="Город Б — Город В",
                   departure_date="2030-06-12", status="issued"),
        ])
