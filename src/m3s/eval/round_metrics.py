"""Round-wise metrics for the DOI-first retrieval scheme, computed from cached rankings.

Targets never enter retrieval, so every metric here is derived offline from the
recommendation order already recorded by `run_recall`; no model or ranking call is repeated.
"""
import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

from m3s.doi_workflow import FIELDS

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Target grid of the manuscript: value range, step and half-step tolerance.
PAPER_GRID = {
    "poisson_ratio": {"start": -1.5, "stop": 1.0, "step": 0.1},
    "young_modulus": {"start": 0.0, "stop": 0.8, "step": 0.1},
    "shear_modulus": {"start": 0.0, "stop": 0.30, "step": 0.05},
}
STRUCTURE_KEY = re.compile(r"^(?P<doi>.+?)_(?P<hash>[a-f0-9]{64})_(?P<index>\d+)(?:_.*)?$")
STEPS = {"poisson_ratio": 0.1, "young_modulus": 0.1, "shear_modulus": 0.05}
# Absolute tolerance shared by all three families: Poisson ratio is dimensionless,
# both moduli are in GPa, so one number is comparable across the grid.
DEFAULT_TOLERANCE = 0.01
TOLERANCE_SWEEP = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]


def paper_targets(grid=None, tolerance=DEFAULT_TOLERANCE):
    """Expand each family into its target values under one shared absolute tolerance."""
    cases = []
    for family, spec in (grid or PAPER_GRID).items():
        step = spec["step"]
        count = int(round((spec["stop"] - spec["start"]) / step)) + 1
        for index in range(count):
            cases.append({"family": family, "target": round(spec["start"] + index * step, 6),
                          "tolerance": tolerance})
    return cases


def load_population(collection):
    """Per SubImage, keep the usable component values of each property family.

    The SubImage key drops the sym/ero/con suffix so all computed variants of one
    figure collapse onto the figure a user would actually be shown.
    """
    projection = {"_id": 0, "structure_id": 1, "elastic_constants": 1}
    population = {}
    for document in collection.find({"design_space_id": os.getenv("MONGO_PERFORMANCE_DESIGN_SPACE", "DS3d_6")},
                                    projection):
        constants = document.get("elastic_constants") or {}
        if constants.get("is_stable") is not True:
            continue
        match = STRUCTURE_KEY.match(document.get("structure_id", ""))
        if not match:
            continue
        key = f"{match['doi']}_{match['hash']}_{match['index']}"
        slot = population.setdefault(key, {"doi": match["doi"], **{family: [] for family in FIELDS}})
        for family, fields in FIELDS.items():
            for field in fields:
                value = constants.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                slot[family].append(float(value))
    return population


def nearest_error(population, keys, family, target):
    """Smallest absolute error to the target across the given SubImages."""
    best = None
    for key in keys:
        for value in population[key][family]:
            error = abs(value - target)
            if best is None or error < best:
                best = error
    return best


def feasible_keys(population, keys, family, target, tolerance):
    """SubImages among `keys` that satisfy the target within tolerance."""
    return {key for key in keys
            if any(abs(value - target) <= tolerance for value in population[key][family])}


def doi_to_keys(population):
    """Map each DOI to its SubImage keys."""
    index = {}
    for key, slot in population.items():
        index.setdefault(slot["doi"], set()).add(key)
    return index


def evaluate_order(order, cases, population, per_doi, rounds):
    """Walk one recommendation order and record every metric round by round."""
    cumulative, seen = [], set()
    for index in range(rounds):
        if index < len(order):
            seen = seen | per_doi.get(order[index], set())
        cumulative.append(set(seen))
    results = []
    for case in cases:
        family, target, tolerance = case["family"], case["target"], case["tolerance"]
        steps, hit_round = [], None
        for index, keys in enumerate(cumulative, start=1):
            found = feasible_keys(population, keys, family, target, tolerance)
            error = nearest_error(population, keys, family, target)
            if found and hit_round is None:
                hit_round = index
            steps.append({"round": index, "success": bool(found), "feasible": len(found),
                          "explored": len(keys), "nearest_error": error})
        results.append({**case, "hit_round": hit_round, "per_round": steps})
    return results


def aggregate(evaluations, rounds):
    """Average each metric over all (query, target) pairs per round."""
    table = []
    for index in range(rounds):
        successes = [item["per_round"][index]["success"] for item in evaluations]
        feasible = [item["per_round"][index]["feasible"] for item in evaluations]
        explored = [item["per_round"][index]["explored"] for item in evaluations]
        errors = [item["per_round"][index]["nearest_error"] for item in evaluations
                  if item["per_round"][index]["nearest_error"] is not None]
        table.append({"round": index + 1,
                      "success_rate": round(sum(successes) / len(successes), 4) if successes else None,
                      "mean_feasible_subimages": round(mean(feasible), 2) if feasible else None,
                      "mean_explored_subimages": round(mean(explored), 2) if explored else None,
                      "median_nearest_error": round(median(errors), 6) if errors else None,
                      "mean_nearest_error": round(mean(errors), 6) if errors else None})
    return table


def by_family(evaluations, rounds):
    """Success rate per property family per round."""
    families = {}
    for item in evaluations:
        families.setdefault(item["family"], []).append(item)
    return {family: [round(sum(entry["per_round"][index]["success"] for entry in items) / len(items), 4)
                     for index in range(rounds)]
            for family, items in families.items()}


def reciprocal_rank(evaluations):
    """Mean reciprocal round of the first feasible recommendation."""
    values = [1.0 / item["hit_round"] if item["hit_round"] else 0.0 for item in evaluations]
    return round(mean(values), 4) if values else None


def oracle_rate(cases, population):
    """Share of targets any stable structure in the whole database can satisfy."""
    keys = set(population)
    reachable = sum(1 for case in cases
                    if feasible_keys(population, keys, case["family"], case["target"], case["tolerance"]))
    return round(reachable / len(cases), 4) if cases else None


def tolerance_sweep(orders, cases, population, per_doi, rounds, tolerances):
    """Success rate per round at several absolute tolerances, to expose the saturation point."""
    table = {}
    for tolerance in tolerances:
        columns = [[] for _ in range(rounds)]
        for order in orders:
            cumulative, seen = [], set()
            for index in range(rounds):
                if index < len(order):
                    seen = seen | per_doi.get(order[index], set())
                cumulative.append(set(seen))
            for case in cases:
                for index, keys in enumerate(cumulative):
                    columns[index].append(bool(feasible_keys(
                        population, keys, case["family"], case["target"], tolerance)))
        table[f"{tolerance:g}"] = [round(sum(column) / len(column), 4) for column in columns]
    return table


def sweep_by_family(orders, cases, population, per_doi, rounds, tolerances):
    """Same sweep split per property family, since the families share one absolute tolerance."""
    table = {}
    for tolerance in tolerances:
        columns = {}
        for order in orders:
            cumulative, seen = [], set()
            for index in range(rounds):
                if index < len(order):
                    seen = seen | per_doi.get(order[index], set())
                cumulative.append(set(seen))
            for case in cases:
                slot = columns.setdefault(case["family"], [[] for _ in range(rounds)])
                for index, keys in enumerate(cumulative):
                    slot[index].append(bool(feasible_keys(
                        population, keys, case["family"], case["target"], tolerance)))
        table[f"{tolerance:g}"] = {family: [round(sum(column) / len(column), 4) for column in slot]
                                   for family, slot in columns.items()}
    return table


def parse_args():
    """Read the offline analysis options."""
    parser = argparse.ArgumentParser(description="Calculate retrieval metrics by round using cached rankings offline")
    parser.add_argument("--eval", default="output/recall_eval", help="run_recall output directory")
    parser.add_argument("--out", default="output/round_metrics", help="Analysis output directory")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to analyze")
    parser.add_argument("--baseline-seeds", type=int, default=5, help="Number of random-order baseline repeats")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Absolute target tolerance (shared by all properties; default 0.01)")
    parser.add_argument("--sweep", nargs="*", type=float, default=TOLERANCE_SWEEP,
                        help="Absolute tolerances to sweep (default 0.01 0.015 ... 0.05)")
    return parser.parse_args()


def main():
    """Recompute round-wise metrics and the random-order baseline, then write them out."""
    args = parse_args()
    from pymongo import MongoClient
    uri = os.environ.get("MONGO_URL")
    if not uri:
        raise RuntimeError("Missing MONGO_URL")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    database = client[os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")]
    try:
        population = load_population(database["StructureProperties"])
    finally:
        client.close()
    per_doi = doi_to_keys(population)
    cases = paper_targets(tolerance=args.tolerance)
    records = json.loads((Path(args.eval) / "records.json").read_text(encoding="utf-8"))

    evaluations, per_query = [], []
    for record in records:
        order = [item["doi"] for item in record["recommended"]]
        results = evaluate_order(order, cases, population, per_doi, args.rounds)
        for item in results:
            item["query_id"] = record["query_id"]
            item["query"] = record["query"]
            item["kind"] = record["kind"]
        evaluations += results
        per_query.append({"query_id": record["query_id"], "query": record["query"], "kind": record["kind"],
                          "order": order, "cases": results})

    pool = sorted(per_doi)
    baseline = []
    for seed in range(args.baseline_seeds):
        rng = random.Random(seed)
        rows = []
        for record in records:
            order = rng.sample(pool, min(args.rounds, len(pool)))
            rows += evaluate_order(order, cases, population, per_doi, args.rounds)
        baseline.append(aggregate(rows, args.rounds))
    random_table = [{"round": index + 1,
                     "success_rate": round(mean(run[index]["success_rate"] for run in baseline), 4),
                     "mean_feasible_subimages": round(mean(run[index]["mean_feasible_subimages"] for run in baseline), 2),
                     "median_nearest_error": round(mean(run[index]["median_nearest_error"] for run in baseline), 6)}
                    for index in range(args.rounds)]

    summary = {"population_subimages": len(population), "population_dois": len(per_doi),
               "targets": len(cases), "queries": len(records), "rounds": args.rounds,
               "target_grid": PAPER_GRID,
               "tolerance": args.tolerance,
               "oracle_success_rate": oracle_rate(cases, population),
               "mean_reciprocal_round": reciprocal_rank(evaluations),
               "per_round": aggregate(evaluations, args.rounds),
               "success_by_family": by_family(evaluations, args.rounds),
               "random_order_baseline": random_table,
               "tolerance_sweep": tolerance_sweep([[item["doi"] for item in record["recommended"]]
                                                   for record in records],
                                                  cases, population, per_doi, args.rounds,
                                                  args.sweep),
               "tolerance_sweep_by_family": sweep_by_family(
                   [[item["doi"] for item in record["recommended"]] for record in records],
                   cases, population, per_doi, args.rounds, args.sweep),
               "tolerance_sweep_random": tolerance_sweep(
                   [random.Random(f"sweep:{record['query_id']}").sample(pool, min(args.rounds, len(pool)))
                    for record in records],
                   cases, population, per_doi, args.rounds, args.sweep)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("population_subimages", "population_dois", "targets", "oracle_success_rate",
                       "mean_reciprocal_round")}, ensure_ascii=False, indent=2))
    print("\nRound | Success rate | Random baseline | Feasible SubImages | Explored SubImages | Median nearest error")
    for row, rand in zip(summary["per_round"], random_table):
        print(f"{row['round']:4d} | {row['success_rate']:.4f} | {rand['success_rate']:.4f} | "
              f"{row['mean_feasible_subimages']:12.2f} | {row['mean_explored_subimages']:14.2f} | "
              f"{row['median_nearest_error']:.6f}")
    print("\nSuccess rate by property:", json.dumps(summary["success_by_family"], ensure_ascii=False))
    print(f"\nTolerance sweep (success rate; absolute tolerance; main-result tolerance {args.tolerance:g})")
    for tolerance in args.sweep:
        key = f"{tolerance:g}"
        real = summary["tolerance_sweep"][key]
        rand = summary["tolerance_sweep_random"][key]
        print(f"{key:>6} | " + "  ".join(f"{v:.3f}" for v in real)
              + " || " + "  ".join(f"{v:.3f}" for v in rand))
    print("\nTolerance sweep by property (final-round success rate)")
    for tolerance in args.sweep:
        key = f"{tolerance:g}"
        row = summary["tolerance_sweep_by_family"][key]
        print(f"{key:>6} | " + "  ".join(f"{family}={values[-1]:.3f}" for family, values in row.items()))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
