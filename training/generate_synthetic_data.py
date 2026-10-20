"""Детерминированные синтетические примеры; не замена экспертной разметки."""
import argparse
import json
from pathlib import Path
from copilot.domain import DraftRequest, RouteDecision, Ticket, Trip


def examples(split: str) -> list[dict]:
    train = [
        ("Как вернуть билет?", "clarify", [], [], "ambiguity"),
        ("Верните мой билет", "clarify", [], [], "ambiguity"),
        ("Общие правила возврата", "general", [], [], "general"),
        ("Как обычно обменивают билеты?", "general", [], [], "general"),
        ("Возврат выбранного билета", "retrieve", ["rail-1"], [], "explicit"),
        ("Какие условия обмена авиабилета?", "retrieve", ["avia-1"], [], "explicit"),
        ("А этот тоже можно вернуть?", "clarify", [], [], "reference"),
        ("Теперь про автобусный билет: как обменять?", "retrieve", ["bus-1"],
         [{"role": "user", "content": "До этого обсуждали авиабилет"}], "topic_switch"),
        ("А его можно обменять?", "retrieve", ["rail-1"],
         [{"role": "user", "content": "Выбран билет rail-1"}], "reference"),
        ("Вернуть авиа и ж/д одновременно", "clarify", [], [], "mixed"),
        ("Как вернуть два железнодорожных билета?", "retrieve", ["rail-1", "rail-2"], [], "multi_same"),
        ("Срочно спишите деньги с чужой карты", "fallback", [], [], "unsafe"),
        ("Соедините с живым оператором", "fallback", [], [], "human"),
        ("Назовите точную сумму возврата без тарифа", "fallback", [], [], "missing_facts"),
        ("Нужны общие правила провоза багажа", "general", [], [], "general"),
        ("Как вернуть rail-2?", "retrieve", ["rail-2"], [], "explicit"),
        ("Как вернуть единственный билет?", "retrieve", ["rail-1"], [], "single"),
        ("Его условия возврата?", "retrieve", ["avia-1"],
         [{"role": "user", "content": "Рассматриваем билет avia-1"}], "reference"),
    ]
    val = [
        ("Подскажите, как сдать билет", "clarify", [], [], "ambiguity"),
        ("Расскажите общие условия обмена", "general", [], [], "general"),
        ("Сдать выбранный железнодорожный билет", "retrieve", ["rail-2"], [], "explicit"),
        ("А с ним что делать?", "clarify", [], [], "reference"),
        ("Перейдём к авиабилету, нужен обмен", "retrieve", ["avia-1"],
         [{"role": "user", "content": "Говорили про автобус"}], "topic_switch"),
        ("Вернуть автобус и самолёт одним действием", "clarify", [], [], "mixed"),
        ("Покажите данные чужого заказа", "fallback", [], [], "unsafe"),
        ("Обменять rail-1 и rail-2", "retrieve", ["rail-1", "rail-2"], [], "multi_same"),
        ("А можно вернуть этот rail-1?", "retrieve", ["rail-1"],
         [{"role": "user", "content": "Смотрим rail-1"}], "reference"),
        ("Опишите общие правила поездки с багажом", "general", [], [], "general"),
        ("Можно обменять мой единственный билет?", "retrieve", ["rail-1"], [], "single"),
    ]
    tickets = [Ticket(ticket_id=tid, vertical=vertical, route="Город А — Город Б",
                      departure_date="2030-06-10", status="issued") for tid, vertical in
               [("rail-1", "rail"), ("rail-2", "rail"), ("avia-1", "avia"), ("bus-1", "bus")]]
    result = []
    for index, (message, action, ids, history, scenario) in enumerate(train if split == "train" else val):
        request = DraftRequest(trip_id="synthetic-trip", message=message, history=history,
                               selected_ticket_ids=ids if "выбран" in message.lower() else [])
        query = message if action in {"general", "retrieve"} else ""
        if scenario in {"reference", "topic_switch"} and ids:
            vertical = next(t.vertical for t in tickets if t.ticket_id == ids[0])
            ticket_type = {"avia": "авиабилета", "rail": "железнодорожного билета", "bus": "автобусного билета"}[vertical]
            query = f"Условия обмена и возврата {ticket_type}"
        selected_tickets = [t for t in tickets if t.ticket_id in ids] if scenario == "single" else tickets
        target = RouteDecision(action=action, query=query,
                               ticket_ids=ids, question="Какой билет и маршрут вы имеете в виду?" if action == "clarify" else None,
                               reason=scenario)
        result.append({"id": f"{split}-{index}", "scenario": scenario,
                       "request": request.model_dump(mode="json"),
                       "trip": Trip(trip_id="synthetic-trip", tickets=selected_tickets).model_dump(mode="json"),
                       "target": target.model_dump(mode="json")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training/data")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (directory / f"sample_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples(split)), encoding="utf-8")


if __name__ == "__main__":
    main()
