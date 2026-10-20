from copilot.cache import PublicRetrievalCache
from copilot.domain import Chunk
from copilot.retrieval import rrf, search_filters


def chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id=cid, version=1, title="Rule", text="Rule",
                 vertical="common", public=True, source="demo://rule")


def test_shared_candidate_beats_single_rank_one():
    result = rrf([[chunk("a"), chunk("b")], [chunk("c"), chunk("b")]])
    assert result[0].chunk_id == "b"


def test_duplicate_hit_does_not_get_extra_rrf_vote():
    result = rrf([[chunk("a"), chunk("a")], [chunk("b")]])
    assert result[0].score == result[1].score


def test_vertical_filter_includes_common():
    assert search_filters("rail")[-1] == {"terms": {"vertical": ["rail", "common"]}}
    assert len(search_filters(None)) == 1


def test_cache_rejects_free_text_history_and_selection():
    cache = PublicRetrievalCache(None, "v1", 300)
    assert cache.key("Заказ 123456 вернуть", False, False) is None
    assert cache.key("общие правила возврата", True, False) is None
    assert cache.key("общие правила возврата", False, True) is None
    assert cache.key("Общие правила возврата?", False, False) is not None


def test_kb_version_changes_cache_namespace():
    first = PublicRetrievalCache(None, "v1", 300)
    second = PublicRetrievalCache(None, "v2", 300)
    assert first.key("общие правила возврата", False, False) != second.key("общие правила возврата", False, False)
