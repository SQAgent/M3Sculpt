"""Verify English retrieval terminology and preserve Chinese input compatibility."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from m3s.doi_workflow import parse_requirement
from m3s.eval import queries, run_recall
from m3s.nodes.parameters import material_parameters
from m3s.prompts import PROMPTS
from m3s.retrieval_vocabulary import PAPER_KEYWORDS, canonical_keywords
from m3s.screening import contradiction, verdict_from_json
from m3s.SubAgents import TagsQuery

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def test_keywords_and_queries_match_si_wording_and_order():
    paper = ROOT / "achievements/paper/SupportInformation.md"
    if not paper.is_file():
        pytest.skip("SI comparison is available in the research repository only")
    si = paper.read_text(encoding="utf-8")
    terms = re.findall(r"^- (.+)$", si.split("### S4.1. Domain-specific retrieval keywords")[1].split('<a id="s5"')[0], re.M)
    short = re.findall(r"^\d+\. (.+)$", si.split("### S5.1. Phrase-based queries")[1].split("### S5.2.")[0], re.M)
    long = re.findall(r"^\d+\. (.+)$", si.split("### S5.2. Mixed queries")[1].split("### S5.3.")[0], re.M)
    assert list(PAPER_KEYWORDS) == terms and len(terms) == 36
    assert queries.SHORT_QUERIES == short and len(short) == 16
    assert queries.LONG_QUERIES == long and len(long) == 14


def test_code_and_interface_have_no_literal_chinese():
    files = [p for folder in (ROOT / "src/m3s", ROOT / "tests") for p in folder.rglob("*")
             if p.is_file() and p.suffix in {".py", ".js", ".html", ".css", ".sh"}]
    assert not [str(p.relative_to(ROOT)) for p in files if HAN.search(p.read_text(encoding="utf-8"))]


def test_keyword_parser_uses_paper_terms_and_keeps_database_labels(monkeypatch):
    prompts = []

    def reply(prompt):
        prompts.append(prompt)
        return {"metamaterial_types": ["lattice"], "performance_types": ["negative_poisson"],
                "keywords": ["lattice", "negative poisson", "negative_poisson_ratio", "star-shaped"]}

    monkeypatch.setattr(TagsQuery, "chat_json", reply)
    handler = object.__new__(TagsQuery.TagsQueryHandler)
    handler.query, handler.strict_errors = "Star-shaped negative Poisson's ratio units", True
    result = handler._parse_keywords_with_ai()
    assert result == {"metamaterial_types": ["lattice"], "performance_types": ["negative_poisson"],
                      "keywords": ["Lattice", "Negative Poisson's Ratio", "star-shaped"]}
    assert all(term in prompts[0] for term in PAPER_KEYWORDS)
    assert not HAN.search(prompts[0])
    assert "English only" in prompts[0]


def test_nonenglish_keyword_output_is_not_used_silently():
    with pytest.raises(ValueError, match="in English"):
        canonical_keywords(["\u8702\u7a9d"])
    assert canonical_keywords(["compression_torsion coupling", "AUXETIC", "auxetic"]) == [
        "Compression-torsion Coupling", "Auxetic"]


def test_all_english_target_cases_are_accepted():
    for case in queries.target_cases():
        target = parse_requirement(case["text"])
        assert not HAN.search(case["text"])
        assert target["family"] == case["family"]
        assert target["target"] == pytest.approx(case["target"])
        assert target["tolerance"] == pytest.approx(case["tolerance"])


def test_english_constituent_and_directional_target_clauses():
    config, remaining = material_parameters(
        "constituent Young's modulus 2000 MPa; constituent Poisson's ratio 0.25; E_z 0.4 GPa, tolerance 10 MPa"
    )
    assert config["young_gpa"] == 2 and config["poisson"] == .25
    target = parse_requirement(remaining)
    assert target["direction"] == "E3" and target["tolerance"] == .01


def test_chinese_input_aliases_keep_their_original_meaning():
    config, remaining = material_parameters(
        "\u57fa\u4f53\u6768\u6c0f\u6a21\u91cf\uff1a2000 MPa\uff0c"
        "\u57fa\u4f53\u6cca\u677e\u6bd4 0.25\uff1b"
        "\u6768\u6c0f\u6a21\u91cf E_z 400 MPa\uff0c\u5bb9\u5dee\uff1a10 MPa"
    )
    assert config["young_gpa"] == 2 and config["poisson"] == .25
    target = parse_requirement(remaining)
    assert target["direction"] == "E3" and target["target"] == .4 and target["tolerance"] == .01
    clash = contradiction("\u9ad8\u521a\u5ea6", "\u8d1f\u521a\u5ea6\u8d85\u6750\u6599")
    assert clash and not HAN.search(clash["reason"])
    assert verdict_from_json({"is_relevant": "\u76f8\u5173"})["is_relevant"]


def test_english_expansion_classifies_short_phrases_by_words(monkeypatch):
    monkeypatch.setattr(queries, "chat_json", lambda prompt: {"queries": [
        "Honeycomb structures", "Lattice architectures designed to preserve isotropic stiffness under extreme multiaxial loads"
    ]})
    extra = queries.expand_queries([], 2)
    assert [item["kind"] for item in extra] == ["short", "long"]
    assert "English queries" in queries.EXPANSION_PROMPT
    assert "English" in PROMPTS["paper_relevance_screen"]


def test_cached_query_identifier_cannot_reuse_a_different_topic(tmp_path):
    file = tmp_path / "q001.json"
    item = {"query_id": "q001", "query": "Honeycomb structures", "kind": "short"}
    assert run_recall.cached_query(file, item) is None
    file.write_text(json.dumps({**item, "query": "\u8702\u7a9d\u7ed3\u6784"}))
    with pytest.raises(ValueError, match="new --out directory"):
        run_recall.cached_query(file, item)
    file.write_text(json.dumps(item))
    assert run_recall.cached_query(file, item) == item


def test_cache_mismatch_stops_before_metadata_writes_and_database_access(tmp_path, monkeypatch):
    trace = tmp_path / "per_query"
    trace.mkdir()
    item = {"query_id": "q001", "query": "Honeycomb structures", "kind": "short"}
    (trace / "q001.json").write_text(json.dumps({**item, "query": "A different old topic"}))
    metadata = tmp_path / "queries.json"
    metadata.write_text("previous query metadata")
    monkeypatch.setattr(run_recall, "parse_args", lambda: SimpleNamespace(
        out=str(tmp_path), queries=None, expand=0, limit_queries=1, families=[], target_grid="benchmark", no_cache=False))
    monkeypatch.setattr(run_recall, "mongo_collection", lambda: pytest.fail("Database access must not occur"))
    from m3s import configuration

    monkeypatch.setattr(configuration, "create_database_config_from_env", lambda: pytest.fail("Database access must not occur"))
    with pytest.raises(ValueError, match="Cached query"):
        run_recall.main()
    assert metadata.read_text() == "previous query metadata"
