"""Random-order control arm: are randomly drawn papers topically usable at all?

The ranking is only worth its cost if the papers it returns are both on-topic and
numerically feasible, so the control arm must be screened by the same criterion.
"""
import argparse
import json
import os
import random
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from m3s.configuration import create_database_config_from_env
from m3s.eval import recall
from m3s.eval.round_metrics import (
    DEFAULT_TOLERANCE,
    doi_to_keys,
    feasible_keys,
    load_population,
    paper_targets,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def titles_for(neo4j_client, dois):
    """Fetch title and abstract for arbitrary DOIs drawn from the whole database."""
    query = """
    MATCH (d:DOI)
    WHERE d.doi IN $doi_list
    OPTIONAL MATCH (d)-[:SUMMARY]->(a:Abstract)
    RETURN d.doi AS doi, d.title AS title, a.abstract AS abstract
    """
    result = neo4j_client.neo4j_parameterized_query(query, {"doi_list": list(dois)})
    return {item["doi"]: {"title": item.get("title") or "", "abstract": item.get("abstract") or ""}
            for item in result.get("node", []) if item.get("doi")}


def parse_args():
    """Read the control-arm options."""
    parser = argparse.ArgumentParser(description="Random-order baseline: evaluate topic relevance of randomly sampled papers")
    parser.add_argument("--eval", default="output/recall_eval", help="run_recall output directory")
    parser.add_argument("--out", default="output/round_metrics", help="Output directory")
    parser.add_argument("--rounds", type=int, default=5, help="Number of papers sampled per query")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel opencode workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Absolute target tolerance (shared by all properties; default 0.01)")
    return parser.parse_args()


def main():
    """Draw random papers per query, screen them, and report joint success per round."""
    args = parse_args()
    from pymongo import MongoClient
    client = MongoClient(os.environ["MONGO_URL"], serverSelectionTimeoutMS=15000)
    database = client[os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")]
    try:
        population = load_population(database["StructureProperties"])
    finally:
        client.close()
    per_doi = doi_to_keys(population)
    cases = paper_targets(tolerance=args.tolerance)
    records = json.loads((Path(args.eval) / "records.json").read_text(encoding="utf-8"))
    cache = recall.Cache(Path(args.out) / "cache_baseline")
    neo4j_client = create_database_config_from_env().neo4j.client
    pool = sorted(per_doi)

    rows, arms = [], []
    for record in records:
        rng = random.Random(f"{args.seed}:{record['query_id']}")
        order = rng.sample(pool, min(args.rounds, len(pool)))
        info = titles_for(neo4j_client, order)
        screener = recall.Screener(record["query"], cache, use_llm=True)
        batch = [(doi, info.get(doi, {}).get("title", ""), info.get(doi, {}).get("abstract", ""))
                 for doi in order]
        verdicts = screener.verdicts(batch, workers=args.workers)
        relevant = [verdict["is_relevant"] for verdict in verdicts]
        arms.append({"query_id": record["query_id"], "query": record["query"], "order": order,
                     "verdicts": verdicts})
        seen, joint, numeric = set(), [], []
        for index in range(args.rounds):
            if relevant[index]:
                seen = seen | per_doi.get(order[index], set())
            joint.append(set(seen))
        seen_all = set()
        for index in range(args.rounds):
            seen_all = seen_all | per_doi.get(order[index], set())
            numeric.append(set(seen_all))
        for case in cases:
            rows.append({"family": case["family"], "target": case["target"],
                         "joint": [bool(feasible_keys(population, keys, case["family"], case["target"], case["tolerance"]))
                                   for keys in joint],
                         "numeric_only": [bool(feasible_keys(population, keys, case["family"], case["target"], case["tolerance"]))
                                          for keys in numeric]})
        print(f"{record['query_id']} {record['query']} relevant random papers {sum(relevant)}/{len(relevant)}", flush=True)

    relevance = mean(1 if verdict["is_relevant"] else 0 for arm in arms for verdict in arm["verdicts"])
    summary = {"queries": len(records), "targets": len(cases), "rounds": args.rounds, "seed": args.seed,
               "tolerance": args.tolerance,
               "random_relevance_rate": round(relevance, 4),
               "joint_success_rate": [round(mean(row["joint"][index] for row in rows), 4) for index in range(args.rounds)],
               "numeric_only_success_rate": [round(mean(row["numeric_only"][index] for row in rows), 4) for index in range(args.rounds)]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "baseline_arms.json").write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
