"""Recall evaluation over 5 recommendation rounds; keeps every intermediate artifact."""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pymongo.errors import PyMongoError

from m3s.configuration import create_database_config_from_env
from m3s.doi_workflow import FIELDS, closest_row, source_paths, within_tolerance
from m3s.llm import chat_json
from m3s.prompts import PROMPTS
from m3s.screening import contradiction, verdict_from_json
from m3s.SubAgents.TagsQuery import RANKING_VERSION, TagsQueryHandler

SCREEN_LIMIT = 20  # Same bound as the serving workflow's per-round screening.


def _slug(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class Cache:
    """File-backed JSON cache so a rerun does not repeat model or database calls."""

    def __init__(self, root, enabled=True):
        self.root = Path(root)
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind, key):
        folder = self.root / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{_slug(key)}.json"

    def get(self, kind, key):
        target = self.path(kind, key)
        if not self.enabled or not target.is_file():
            return None
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, kind, key, value):
        self.path(kind, key).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return value


class Screener:
    """Rule-then-model relevance screening that records why each paper was judged."""

    def __init__(self, query, cache, use_llm=True):
        self.query = query
        self.cache = cache
        self.use_llm = use_llm
        self.log = []

    def _rule_verdict(self, doi, title, abstract):
        clash = contradiction(self.query, title, abstract)
        if clash:
            return {"doi": doi, "title": title, "stage": "rule", "rule": clash["rule"],
                    "is_relevant": False, "reason": clash["reason"]}
        if not self.use_llm:
            return {"doi": doi, "title": title, "stage": "skipped", "is_relevant": True,
                    "reason": "Model screening disabled"}
        return None

    def _llm_verdict(self, doi, title, abstract):
        """Judge one paper with the model, reusing the cached verdict when present."""
        key = f"{self.query}||{doi}"
        cached = self.cache.get("screen", key)
        if cached is not None:
            return cached
        prompt = PROMPTS["paper_relevance_screen"].format(
            user_query=self.query, title=title, abstract=(abstract or "")[:4000])
        started = time.time()
        verdict = verdict_from_json(chat_json(prompt))
        payload = {"doi": doi, "title": title, "stage": "llm", "seconds": round(time.time() - started, 2),
                   "is_relevant": True if verdict is None else verdict["is_relevant"],
                   "reason": "Model output unusable; retained by the offline benchmark" if verdict is None else verdict["reason"],
                   "verdict_missing": verdict is None}
        return self.cache.put("screen", key, payload)

    def verdicts(self, batch, workers=1):
        """Judge a batch of (doi, title, abstract) and return verdicts in input order."""
        pending, results = [], {}
        for doi, title, abstract in batch:
            early = self._rule_verdict(doi, title, abstract)
            if early is None:
                pending.append((doi, title, abstract))
            else:
                results[doi] = early
        if pending:
            if workers > 1 and len(pending) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(self._llm_verdict, *item): item[0] for item in pending}
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
            else:
                for item in pending:
                    results[item[0]] = self._llm_verdict(*item)
        return [results[doi] for doi, _, _ in batch]

    def judge(self, doi, title, abstract):
        """Return "" to keep the paper, or the rejection reason."""
        verdict = self.verdicts([(doi, title, abstract)])[0]
        self.log.append(verdict)
        return "" if verdict["is_relevant"] else verdict["reason"]


def require_current_ranking(record):
    """Reject cached rankings or completed evaluations from before the metric fix."""
    if record.get("ranking_version") != RANKING_VERSION:
        raise ValueError(
            "Cached ranking uses an outdated or unknown semantic scoring version. "
            "Use a new --out directory or explicitly use --no-cache to recompute."
        )


def retrieve(query, cache, use_llm=True):
    """Rank papers for one query exactly as the workflow does, without any target."""
    cached = cache.get("rank", query)
    if cached is not None:
        require_current_ranking(cached)
        return cached
    handler = TagsQueryHandler(create_database_config_from_env(), query, strict_errors=True)
    try:
        started = time.time()
        result = handler.run(top_k=0)
        payload = {"query": query, "seconds": round(time.time() - started, 2),
                   "ranking_version": RANKING_VERSION,
                   "ranked_dois": result.get("ranked_dois", []),
                   "fallback_used": result.get("fallback_used", False),
                   "titles": {doi: info.get("title", "") for doi, info in result.get("doi_info", {}).items()},
                   "abstracts": {doi: info.get("abstract", "") for doi, info in result.get("doi_info", {}).items()}}
    finally:
        handler.close()
    return cache.put("rank", query, payload)


def screen_rounds(ranked, titles, abstracts, screener, rounds, workers=1, batch=8):
    """Walk the ranking and collect the papers the workflow would recommend."""
    accepted, rejected, index = [], 0, 0
    budget = SCREEN_LIMIT * rounds  # Same screening budget the workflow spends per round.
    while len(accepted) < rounds and index < len(ranked) and rejected < budget:
        window = [(doi, titles.get(doi, ""), abstracts.get(doi, "")) for doi in ranked[index:index + batch]]
        index += len(window)
        for verdict in screener.verdicts(window, workers=workers):
            screener.log.append(verdict)
            if verdict["is_relevant"]:
                accepted.append(verdict["doi"])
                if len(accepted) >= rounds:
                    break
            else:
                rejected += 1
                if rejected >= budget:
                    break
    return accepted


def doi_evidence(doi, cache, neo4j_client, mongo_collection, retries=3):
    """Collect this DOI's SubImages and their existing engineering constants."""
    cached = cache.get("evidence", doi)
    if cached is not None:
        return cached
    query = """
    MATCH (d:DOI {has_SubFigure: true})-[]->(n:SubFigure)
    WHERE d.doi = $doi
    RETURN n.img_path AS img_path
    """
    result = neo4j_client.neo4j_parameterized_query(query, {"doi": doi})
    paths = list(dict.fromkeys(item["img_path"] for item in result.get("node", []) if item.get("img_path")))
    keys = source_paths(doi, paths)
    rows = []
    if keys:
        selector = {"design_space_id": os.getenv("MONGO_PERFORMANCE_DESIGN_SPACE", "DS3d_6"),
                    "$or": [{"structure_id": {"$regex": f"^{re.escape(key)}(?:_|$)"}} for key in keys]}
        projection = {"_id": 0, "structure_id": 1, "elastic_constants": 1}
        last = None
        for attempt in range(max(1, retries)):
            try:
                rows = list(mongo_collection.find(selector, projection))
                break
            except PyMongoError as exc:  # A restarted server must not look like a miss.
                last = exc
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to read existing computation records for {doi}: {last}")
    payload = {"doi": doi, "subimage_paths": paths, "source_keys": keys, "rows": rows,
               "stable_rows": sum(1 for row in rows if (row.get("elastic_constants") or {}).get("is_stable") is True)}
    return cache.put("evidence", doi, payload)


def check_case(evidence, requirement):
    """Return the nearest stable structure in this DOI plus the tolerance verdict."""
    best = closest_row(evidence["rows"], evidence["source_keys"], requirement)
    if best is None:
        return {"hit": False, "nearest_error": None, "structure_id": None,
                "matched_field": None, "matched_value": None, "image_path": None}
    return {"hit": within_tolerance(best["absolute_error"], requirement["tolerance"]),
            "nearest_error": best["absolute_error"], "structure_id": best["structure_id"],
            "matched_field": best["matched_field"], "matched_value": best["matched_value"],
            "image_path": best["image_path"]}


def covered_subimages(evidence, family):
    """Count SubImages of this DOI that carry a usable value for the property family."""
    fields = FIELDS[family]
    keys = set()
    for row in evidence["rows"]:
        constants = row.get("elastic_constants") or {}
        if constants.get("is_stable") is not True:
            continue
        if not any(isinstance(constants.get(field), (int, float)) and not isinstance(constants.get(field), bool)
                   for field in fields):
            continue
        sid = row.get("structure_id", "")
        key = next((item for item in evidence["source_keys"] if sid == item or sid.startswith(item + "_")), None)
        if key:
            keys.add(key)
    return keys
