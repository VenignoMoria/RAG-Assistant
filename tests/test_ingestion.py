from copilot.domain import KnowledgeEvent
from ingestion.chunking import chunks_for


def test_chunk_ids_are_deterministic_and_versioned():
    event = KnowledgeEvent(event_id="e", document_id="doc", version=1, title="Demo",
                           content="Правило возврата. " * 200, vertical="common", source="demo://doc")
    one = chunks_for(event)
    two = chunks_for(event)
    assert [c.chunk_id for c in one] == [c.chunk_id for c in two]
    newer = chunks_for(event.model_copy(update={"version": 2}))
    assert not {c.chunk_id for c in one}.intersection(c.chunk_id for c in newer)


def test_html_scripts_are_not_indexed():
    event = KnowledgeEvent(event_id="e", document_id="doc", version=1, title="Demo",
                           content="<p>Правило</p><script>steal()</script>", content_type="text/html",
                           vertical="common", source="demo://doc")
    assert "steal" not in chunks_for(event)[0].text
