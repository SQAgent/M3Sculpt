"""CLI: run the 5-round recall benchmark and write every intermediate artifact."""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from m3s.doi_workflow import FIELDS, parse_requirement
from m3s.eval import recall
from m3s.eval.queries import load_queries, requirement_text, target_cases

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def parse_args():
    """Read the benchmark options."""
    parser = argparse.ArgumentParser(description="Mechanical metamaterial retrieval recall benchmark (test mode, existing computation records)")
    parser.add_argument("--out", default="output/recall_eval", help="Result output directory")
    parser.add_argument("--rounds", type=int, default=5, help="Number of recommendation rounds per query")
    parser.add_argument("--queries", default="", help="Query list JSON; defaults to the built-in queries")
    parser.add_argument("--target-grid", choices=["benchmark", "paper"], default="benchmark",
                        help="paper uses 42 manuscript targets with absolute tolerance 0.01")
    parser.add_argument("--expand", type=int, default=0, help="Number of additional model-generated queries")
    parser.add_argument("--limit-queries", type=int, default=0, help="Run only the first N queries (0 means all)")
    parser.add_argument("--families", nargs="*", default=[], help="Run only selected properties, such as poisson_ratio")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel opencode relevance-screening workers")
    parser.add_argument("--no-llm-screen", action="store_true", help="Skip model relevance screening and use contradiction rules only")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing caches and recompute all results")
    return parser.parse_args()


def mongo_collection():
    """Open the existing-results collection used by test mode."""
    from pymongo import MongoClient
    uri = os.environ.get("MONGO_URL")
    if not uri:
        raise RuntimeError("Missing MONGO_URL")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    database = client[os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")]
    return client, database["StructureProperties"]


def evaluate_query(item, cases, evidences, cache, args, neo4j_client, collection):
    """Run one query: rank, screen, then check every target case round by round."""
    ranking = recall.retrieve(item["query"], cache, use_llm=True)
    screener = recall.Screener(item["query"], cache, use_llm=not args.no_llm_screen)
    accepted = recall.screen_rounds(ranking["ranked_dois"], ranking["titles"], ranking["abstracts"],
                                   screener, args.rounds, workers=args.workers)
    for doi in accepted:
        if doi not in evidences:
            evidences[doi] = recall.doi_evidence(doi, cache, neo4j_client, collection)
    rounds = []
    for index, doi in enumerate(accepted, start=1):
        evidence = evidences[doi]
        rounds.append({"round": index, "doi": doi, "title": ranking["titles"].get(doi, ""),
                       "subimages": len(evidence["source_keys"]),
                       "rows": len(evidence["rows"]), "stable_rows": evidence["stable_rows"]})
    results = []
    for case in cases:
        requirement = parse_requirement(case["text"])
        per_round, hit_round, covered = [], None, set()
        for index, doi in enumerate(accepted, start=1):
            evidence = evidences[doi]
            verdict = recall.check_case(evidence, requirement)
            covered |= recall.covered_subimages(evidence, case["family"])
            per_round.append({"round": index, "doi": doi, **verdict,
                              "subimages": len(evidence["source_keys"]),
                              "cumulative_covered_subimages": len(covered)})
            if verdict["hit"] and hit_round is None:
                hit_round = index
        results.append({"family": case["family"], "target": case["target"], "tolerance": case["tolerance"],
                        "requirement_text": case["text"], "hit": hit_round is not None,
                        "hit_round": hit_round, "rounds_available": len(accepted),
                        "covered_subimages": len(covered), "per_round": per_round})
    return {"query_id": item["query_id"], "query": item["query"], "kind": item["kind"],
            "ranking_version": ranking["ranking_version"],
            "ranked_total": len(ranking["ranked_dois"]), "fallback_used": ranking["fallback_used"],
            "recommended": rounds, "screen_log": screener.log, "cases": results}


def summarize(records, rounds):
    """Aggregate recall@k, per-family recall and SubImage coverage per round."""
    at_k = {k: {"hit": 0, "total": 0} for k in range(1, rounds + 1)}
    families, kinds, coverage = {}, {}, {k: [] for k in range(1, rounds + 1)}
    for record in records:
        for case in record["cases"]:
            for k in range(1, rounds + 1):
                at_k[k]["total"] += 1
                if case["hit_round"] is not None and case["hit_round"] <= k:
                    at_k[k]["hit"] += 1
            for bucket, key in ((families, case["family"]), (kinds, record["kind"])):
                slot = bucket.setdefault(key, {"hit": 0, "total": 0})
                slot["total"] += 1
                slot["hit"] += 1 if case["hit"] else 0
            for step in case["per_round"]:
                coverage[step["round"]].append(step["cumulative_covered_subimages"])
    def rate(slot):
        return round(slot["hit"] / slot["total"], 4) if slot["total"] else None
    return {"queries": len(records),
            "cases_per_query": len(records[0]["cases"]) if records else 0,
            "recall_at_round": {f"@{k}": {**at_k[k], "recall": rate(at_k[k])} for k in at_k},
            "recall_by_family": {name: {**slot, "recall": rate(slot)} for name, slot in families.items()},
            "recall_by_query_kind": {name: {**slot, "recall": rate(slot)} for name, slot in kinds.items()},
            "subimages_per_round": {f"@{k}": {"mean_cumulative": round(sum(values) / len(values), 2) if values else 0,
                                              "max_cumulative": max(values) if values else 0}
                                    for k, values in coverage.items()},
            "unique_subimages_per_query_round": {
                f"@{k}": round(sum(sum(item["subimages"] for item in record["recommended"][:k])
                                   for record in records) / len(records), 2) if records else 0
                for k in range(1, rounds + 1)}}


def cached_query(path, item):
    """Reject completed records from a different query or retrieval scoring version."""
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if any(record.get(key) != item[key] for key in ("query_id", "query", "kind")):
        raise ValueError(
            f"Cached query {item['query_id']} differs from the current query set. "
            "Use a new --out directory or explicitly use --no-cache to recompute."
        )
    recall.require_current_ranking(record)
    return record


def main():
    """Run the benchmark and write per-query records plus the summary."""
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = recall.Cache(out / "cache", enabled=not args.no_cache)
    queries = load_queries(args.queries or None, args.expand)
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    cases = target_cases(args.families or None)
    if args.target_grid == "paper":
        from m3s.eval.round_metrics import paper_targets
        cases = [{**case, "text": requirement_text(case["family"], case["target"], case["tolerance"])}
                 for case in paper_targets()
                 if not args.families or case["family"] in args.families]
    # Check before writing query metadata or connecting to any database.
    completed = {} if args.no_cache else {
        item["query_id"]: cached_query(out / "per_query" / f"{item['query_id']}.json", item)
        for item in queries
    }
    if not args.no_cache:
        for item in queries:
            ranking = cache.get("rank", item["query"])
            if ranking is not None:
                recall.require_current_ranking(ranking)
    (out / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "target_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    from m3s.configuration import create_database_config_from_env
    database = create_database_config_from_env()
    neo4j_client = database.neo4j.client
    client, collection = mongo_collection()
    records, evidences = [], {}
    trace = (out / "per_query").resolve()
    trace.mkdir(parents=True, exist_ok=True)
    try:
        for position, item in enumerate(queries, start=1):
            done = trace / f"{item['query_id']}.json"
            if completed.get(item["query_id"]) is not None:
                records.append(completed[item["query_id"]])
                print(f"[{position}/{len(queries)}] {item['query_id']} {item['query']} Reusing existing results", flush=True)
                continue
            started = time.time()
            record = evaluate_query(item, cases, evidences, cache, args, neo4j_client, collection)
            record["seconds"] = round(time.time() - started, 2)
            records.append(record)
            done.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            hits = sum(1 for case in record["cases"] if case["hit"])
            print(f"[{position}/{len(queries)}] {item['query_id']} {item['query']} "
                  f"matched {hits}/{len(record['cases'])} cases, recommended {len(record['recommended'])} papers, "
                  f"{record['seconds']}s", flush=True)
    finally:
        client.close()
    summary = summarize(records, args.rounds)
    summary["mode"] = "test"
    summary["matching_fields"] = FIELDS
    summary["target_grid"] = args.target_grid
    summary["design_space_id"] = os.getenv("MONGO_PERFORMANCE_DESIGN_SPACE", "DS3d_6")
    summary["rounds"] = args.rounds
    summary["llm_screening"] = not args.no_llm_screen
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["recall_at_round"], ensure_ascii=False, indent=2))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
