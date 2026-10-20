import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--min-accuracy", type=float, required=True)
    parser.add_argument("--max-wrong-ticket-rate", type=float, required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    candidate = report["lora"]
    if candidate["action_accuracy"] < args.min_accuracy:
        raise SystemExit("Action accuracy gate failed")
    if candidate["wrong_ticket_rate"] > args.max_wrong_ticket_rate:
        raise SystemExit("Wrong-ticket gate failed")
    if candidate["json_validity"] < 1:
        raise SystemExit("JSON validity gate failed")


if __name__ == "__main__":
    main()
