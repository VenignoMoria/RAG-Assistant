import argparse
import asyncio
import json
from pathlib import Path
from aiokafka import AIOKafkaProducer
from copilot.config import Settings
from copilot.domain import KnowledgeEvent


async def publish_event(bootstrap: str, topic: str, key: str, payload: dict) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap, acks="all", enable_idempotence=True,
                               value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode())
    await producer.start()
    try:
        await producer.send_and_wait(topic, key=key.encode(), value=payload)
    finally:
        await producer.stop()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file")
    args = parser.parse_args()
    event = KnowledgeEvent.model_validate_json(Path(args.event_file).read_text(encoding="utf-8"))
    await publish_event(Settings().kafka_bootstrap, "knowledge-updates", event.document_id,
                        event.model_dump(mode="json"))


if __name__ == "__main__":
    asyncio.run(main())
