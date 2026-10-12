"""Regress DOI ranking against Milvus similarity and distance metric directions."""

import json
from types import SimpleNamespace

import pytest

from m3s.db import milvus
from m3s.eval import recall, run_recall
from m3s.SubAgents.TagsQuery import RANKING_VERSION, TagsQueryHandler


@pytest.fixture
def make_handler(monkeypatch):
    def make(metric, hits, candidates, keywords=None):
        calls = []

        def search(**kwargs):
            calls.append(kwargs)
            return [hits]

        monkeypatch.setattr(milvus, "MilvusClient", lambda **kwargs: SimpleNamespace(
            search=search, use_database=lambda **kwargs: None))
        manager = milvus.MilvusClientManager(SimpleNamespace(
            url="http://milvus.example", username="test", password="test", database="test"))
        manager.update("search_params", {"metric_type": metric, "params": {}})
        neo4j = SimpleNamespace(neo4j_parameterized_query=lambda *args: {"node": []})
        database = SimpleNamespace(milvus=SimpleNamespace(client=manager),
                                   neo4j=SimpleNamespace(client=neo4j))
        handler = TagsQueryHandler(database, "Honeycomb structures", strict_errors=True)
        monkeypatch.setattr(handler, "_get_query_embedding", lambda: [1.0, 0.0])
        monkeypatch.setattr(handler, "_parse_keywords_with_ai", lambda: {
            "metamaterial_types": [], "performance_types": [], "keywords": []})
        monkeypatch.setattr(handler, "_filter_dois_by_tags", lambda *args: candidates)
        monkeypatch.setattr(handler, "_retrieve_doi_info_from_neo4j", lambda dois: {
            doi: {"title": doi, "abstract": ""} for doi in dois})
        monkeypatch.setattr(handler, "_keyword_search", lambda *args: keywords or {})
        return handler, calls

    return make


def hit(doi, distance, suffix=""):
    return {"id": f"{doi}/paper{suffix}", "distance": distance}


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("metric,better,worse", [
    ("COSINE", 0.9, 0.6), ("IP", 2.0, 1.0), ("L2", 0.1, 0.8),
    ("COSINE", -0.2, -0.8), ("IP", -2.0, -4.0),
])
def test_more_similar_paper_ranks_first_with_equal_keyword_scores(make_handler, fallback, metric, better, worse):
    hits = [hit("10.1/worse", worse), hit("10.1/better", better)]
    candidates = [] if fallback else ["10.1/worse", "10.1/better"]
    handler, calls = make_handler(metric, hits, candidates, {"10.1/worse": 2, "10.1/better": 2})
    result = handler.run(top_k=0)
    assert result["ranked_dois"] == ["10.1/better", "10.1/worse"]
    assert result["fallback_used"] == fallback
    assert len(calls) == 1
    assert calls[0]["search_params"]["metric_type"] == metric


@pytest.mark.parametrize("metric,values,expected", [
    ("COSINE", [0.9, 0.1], 0.9), ("IP", [-2.0, -5.0], -2.0),
    ("L2", [0.1, 5.0], 1 / 1.1),
])
def test_same_doi_keeps_its_best_hit_regardless_of_result_order(make_handler, metric, values, expected):
    handler, _ = make_handler(metric, [], ["10.1/a"])
    vectors = [hit("10.1/a", value, str(i)) for i, value in enumerate(values)]
    vectors.extend([hit("10.1/excluded", values[0]), {"id": "", "distance": values[0]}])
    for order in [vectors, list(reversed(vectors))]:
        handler._retrieve_entities_from_milvus = lambda embeddings: order
        assert handler._compute_semantic_scores(["10.1/a"]) == pytest.approx({"10.1/a": expected})


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0])
def test_equal_cosine_scores_are_finite_and_allow_keyword_ranking(make_handler, value):
    handler, _ = make_handler("COSINE", [hit("10.1/a", value), hit("10.1/b", value)],
                              ["10.1/a", "10.1/b"], {"10.1/a": 1, "10.1/b": 2})
    assert handler.run(top_k=0)["ranked_dois"] == ["10.1/b", "10.1/a"]


def test_missing_semantic_hit_does_not_outrank_negative_but_present_similarity(make_handler):
    handler, _ = make_handler("COSINE", [hit("10.1/a", -0.2), hit("10.1/b", -0.8)],
                              ["10.1/missing", "10.1/b", "10.1/a"])
    assert handler.run(top_k=0)["ranked_dois"][0] == "10.1/a"


def test_keyword_and_semantic_components_keep_equal_weight(make_handler):
    handler, _ = make_handler("COSINE", [hit("10.1/a", 0.9), hit("10.1/b", 0.6)],
                              ["10.1/a", "10.1/b"], {"10.1/a": 1, "10.1/b": 2})
    # A: 0.5 * 1 + 0.5 * 0.5 = 0.75; B: 0.5 * (2/3) + 0.5 * 1 = 5/6.
    assert handler.run(top_k=0)["ranked_dois"] == ["10.1/b", "10.1/a"]


def test_fallback_selects_top_fifty_unique_dois_after_aggregation(make_handler):
    hits = [hit(f"10.1/{index:02d}", index / 60) for index in range(60)]
    hits.extend([hit("10.1/00", 1.0, "best"), hit("10.1/00", 0.0, "worst")])
    handler, calls = make_handler("COSINE", hits, [])
    ranked = handler.run(top_k=0)["ranked_dois"]
    assert len(ranked) == len(set(ranked)) == 50
    assert ranked[0] == "10.1/00"
    assert "10.1/10" not in ranked and "10.1/11" in ranked
    assert len(calls) == 1


@pytest.mark.parametrize("distance", [None, float("nan"), float("inf"), float("-inf"), True, "invalid"])
def test_invalid_milvus_score_is_an_error(make_handler, distance):
    handler, _ = make_handler("COSINE", [hit("10.1/a", distance)], ["10.1/a"])
    with pytest.raises(ValueError, match="Invalid Milvus"):
        handler.run(top_k=0)


@pytest.mark.parametrize("metric,distance,message", [
    ("L2", -0.1, "nonnegative"), ("HAMMING", 1.0, "Unsupported"), ("UNKNOWN", 1.0, "Unsupported"),
])
def test_invalid_or_unsupported_metric_cannot_silently_use_cosine_rules(make_handler, metric, distance, message):
    handler, _ = make_handler(metric, [hit("10.1/a", distance)], ["10.1/a"])
    with pytest.raises(ValueError, match=message):
        handler.run(top_k=0)


def test_old_rank_cache_is_rejected_without_modifying_it_or_connecting(tmp_path, monkeypatch):
    cache = recall.Cache(tmp_path)
    query = "Honeycomb structures"
    record = {"query": query, "ranked_dois": ["10.1/wrong"]}
    cache.put("rank", query, record)
    monkeypatch.setattr(recall, "create_database_config_from_env", lambda: pytest.fail("Must not connect"))
    with pytest.raises(ValueError, match="new --out directory"):
        recall.retrieve(query, cache)
    assert cache.get("rank", query) == record
    current = {**record, "ranking_version": RANKING_VERSION}
    cache.put("rank", query, current)
    assert recall.retrieve(query, cache) == current


def test_completed_evaluation_from_old_ranking_is_rejected(tmp_path):
    item = {"query_id": "q001", "query": "Honeycomb structures", "kind": "short"}
    path = tmp_path / "q001.json"
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="semantic scoring version"):
        run_recall.cached_query(path, item)
    current = {**item, "ranking_version": RANKING_VERSION}
    path.write_text(json.dumps(current))
    assert run_recall.cached_query(path, item) == current


def test_old_rank_cache_stops_evaluation_before_metadata_writes_and_database_access(tmp_path, monkeypatch):
    query = "Honeycomb structures"
    cache = recall.Cache(tmp_path / "cache")
    cache.put("rank", query, {"query": query, "ranked_dois": ["10.1/wrong"]})
    metadata = tmp_path / "queries.json"
    metadata.write_text("previous metadata")
    monkeypatch.setattr(run_recall, "parse_args", lambda: SimpleNamespace(
        out=str(tmp_path), queries=None, expand=0, limit_queries=1, families=[], target_grid="benchmark", no_cache=False))
    monkeypatch.setattr(run_recall, "mongo_collection", lambda: pytest.fail("Must not connect"))
    with pytest.raises(ValueError, match="semantic scoring version"):
        run_recall.main()
    assert metadata.read_text() == "previous metadata"
