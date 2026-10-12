"""Ablation of the geometric-variant scope under the same RAG recommendation order.

Every arm reuses the ranking cached by `run_recall`, so retrieval and topical
screening are identical across arms; only the set of structural variants a paper
is allowed to contribute changes. That isolates what symmetrization and the
morphological erosion/dilation sweep add on top of the bare reconstruction.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, median

from dotenv import load_dotenv

from m3s.doi_workflow import FIELDS
from m3s.eval.round_metrics import DEFAULT_TOLERANCE, PAPER_GRID, paper_targets

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

STRUCTURE_KEY = re.compile(
    r"^(?P<doi>.+?)_(?P<hash>[a-f0-9]{64})_(?P<index>\d+)"
    r"(?:_sym(?P<sym>[01]{3})_(?P<op>ero|dil)(?P<steps>\d+)_con(?P<con>\d+))?$"
)

# Each arm is a predicate over the parsed suffix of a structure_id.
# Bare-reconstruction records have no suffix (sym is None).
SCOPES = {
    "bare_reconstruction": lambda v: v["sym"] is None,
    "symmetry_only": lambda v: v["sym"] is not None and v["op"] == "ero" and v["steps"] == 0,
    "morphology_only": lambda v: v["sym"] == "000",
    "all_stable_variants": lambda v: True,
}
SCOPE_NOTES = {
    "bare_reconstruction": "Reconstructed voxels without reflection or morphology (structure_id has no suffix)",
    "symmetry_only": "Reflection only: all reflection combinations, morphology fixed at ero0",
    "morphology_only": "Morphology only: fixed sym000, all erosion and dilation steps",
    "all_stable_variants": "All stable reflection and morphology variants (full workflow)",
}


def parse_structure_id(structure_id):
    """Split a structure_id into its SubImage key and its geometric-variant tags."""
    match = STRUCTURE_KEY.match(structure_id)
    if not match:
        return None
    return {"doi": match["doi"],
            "key": f"{match['doi']}_{match['hash']}_{match['index']}",
            "sym": match["sym"],
            "op": match["op"],
            "steps": int(match["steps"]) if match["steps"] is not None else None}


def load_scoped_populations(collection):
    """One database pass that fills every scope with its stable SubImage values.

    A SubImage is the figure a user is shown, so all variants of one figure collapse
    onto the same key; the scope decides which of those variants may contribute values.
    """
    projection = {"_id": 0, "structure_id": 1, "elastic_constants": 1}
    selector = {"design_space_id": os.getenv("MONGO_PERFORMANCE_DESIGN_SPACE", "DS3d_6")}
    scoped = {name: {} for name in SCOPES}
    variants = {name: 0 for name in SCOPES}
    for document in collection.find(selector, projection):
        constants = document.get("elastic_constants") or {}
        if constants.get("is_stable") is not True:
            continue
        tags = parse_structure_id(document.get("structure_id", ""))
        if not tags:
            continue
        values = {}
        for family, fields in FIELDS.items():
            usable = []
            for field in fields:
                value = constants.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                usable.append(float(value))
            values[family] = usable
        for name, predicate in SCOPES.items():
            if not predicate(tags):
                continue
            variants[name] += 1
            slot = scoped[name].setdefault(tags["key"], {"doi": tags["doi"],
                                                         **{family: [] for family in FIELDS}})
            for family, usable in values.items():
                slot[family] += usable
    return scoped, variants


def feasible_keys(population, keys, family, target, tolerance):
    return {key for key in keys
            if any(abs(value - target) <= tolerance for value in population[key][family])}


def nearest_error(population, keys, family, target):
    best = None
    for key in keys:
        for value in population[key][family]:
            err = abs(value - target)
            if best is None or err < best:
                best = err
    return best


def evaluate_order(order, cases, population, per_doi, rounds):
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
            err = nearest_error(population, keys, family, target)
            if found and hit_round is None:
                hit_round = index
            steps.append({"round": index, "success": bool(found), "feasible": len(found),
                          "explored": len(keys), "nearest_error": err})
        results.append({**case, "hit_round": hit_round, "per_round": steps})
    return results


def aggregate(evaluations, rounds):
    table = []
    for index in range(rounds):
        succ = [item["per_round"][index]["success"] for item in evaluations]
        feasible = [item["per_round"][index]["feasible"] for item in evaluations]
        explored = [item["per_round"][index]["explored"] for item in evaluations]
        errors = [item["per_round"][index]["nearest_error"] for item in evaluations
                  if item["per_round"][index]["nearest_error"] is not None]
        table.append({
            "round": index + 1,
            "success_rate": round(sum(succ) / len(succ), 4) if succ else None,
            "mean_feasible_subimages": round(mean(feasible), 2) if feasible else None,
            "mean_explored_subimages": round(mean(explored), 2) if explored else None,
            "median_nearest_error": round(median(errors), 6) if errors else None,
            "mean_nearest_error": round(mean(errors), 6) if errors else None,
        })
    return table


def by_family(evaluations, rounds):
    families = {}
    for item in evaluations:
        families.setdefault(item["family"], []).append(item)
    return {
        family: [round(sum(entry["per_round"][index]["success"] for entry in items) / len(items), 4)
                 for index in range(rounds)]
        for family, items in families.items()
    }


def reciprocal_rank(evaluations):
    values = [1.0 / item["hit_round"] if item["hit_round"] else 0.0 for item in evaluations]
    return round(mean(values), 4) if values else None


def mrr_by_family(evaluations):
    """Mean reciprocal round of the first feasible recommendation, per property family."""
    families = {}
    for item in evaluations:
        families.setdefault(item["family"], []).append(item)
    return {family: reciprocal_rank(items) for family, items in families.items()}


def oracle_rate(cases, population):
    """Share of targets this scope could satisfy if every paper in the database were read."""
    keys = set(population)
    reachable = sum(1 for case in cases
                    if feasible_keys(population, keys, case["family"], case["target"], case["tolerance"]))
    return round(reachable / len(cases), 4) if cases else None


def normalized_auc(rates):
    """Trapezoidal area of the cumulative curve over rounds 0..N, divided by the span."""
    values = [0.0] + list(rates)
    area = sum((values[index] + values[index + 1]) / 2 for index in range(len(rates)))
    return round(area / len(rates), 4) if rates else None


def target_coverage(cases, population, per_doi):
    """Share of DOIs that could satisfy each target on their own.

    A target no single paper can reach is a property of the database, not of the
    ranking, so this separates grid difficulty from retrieval quality.
    """
    coverage = {}
    for case in cases:
        reachable = sum(1 for keys in per_doi.values()
                        if feasible_keys(population, keys, case["family"], case["target"], case["tolerance"]))
        coverage[f"{case['family']}@{case['target']:g}"] = {
            "family": case["family"], "target": case["target"], "tolerance": case["tolerance"],
            "dois": reachable, "coverage": round(reachable / len(per_doi), 4) if per_doi else None}
    return coverage


def by_difficulty(evaluations, coverage, rounds, cut=0.5):
    """Split success by whether a single paper is likely to reach the target at all."""
    groups = {"easy": [], "hard": []}
    for item in evaluations:
        entry = coverage[f"{item['family']}@{item['target']:g}"]
        groups["easy" if entry["coverage"] >= cut else "hard"].append(item)
    result = {"cut": cut}
    for name, items in groups.items():
        result[name] = {
            "targets": len({(item["family"], item["target"]) for item in items}),
            "pairs": len(items),
            "per_round": [round(sum(entry["per_round"][index]["success"] for entry in items) / len(items), 4)
                          for index in range(rounds)] if items else None}
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Geometric-variant scope ablation: compare reflection and morphology success rates under the same RAG ranking")
    parser.add_argument("--eval", default="output/recall_eval", help="run_recall output directory")
    parser.add_argument("--out", default="output/variant_scope_rag", help="Output directory")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="Absolute target tolerance (shared by all properties; default 0.01)")
    return parser.parse_args()


def main():
    """Replay the cached RAG order under each variant scope and write the comparison."""
    args = parse_args()
    from pymongo import MongoClient
    uri = os.environ.get("MONGO_URL")
    if not uri:
        raise RuntimeError("Missing MONGO_URL")
    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    database = client[os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")]
    try:
        scoped, variant_counts = load_scoped_populations(database["StructureProperties"])
    finally:
        client.close()

    cases = paper_targets(tolerance=args.tolerance)
    records = json.loads((Path(args.eval) / "records.json").read_text(encoding="utf-8"))
    # run_recall only records papers that already passed topical screening, so the
    # recommended order is the topic-valid arm; every scope shares it exactly.
    orders = {record["query_id"]: [item["doi"] for item in record["recommended"]] for record in records}

    scopes, per_query_out, coverage = {}, {}, None
    for name, population in scoped.items():
        per_doi = {}
        for key, slot in population.items():
            per_doi.setdefault(slot["doi"], set()).add(key)
        evaluations, per_query = [], []
        for record in records:
            order = orders[record["query_id"]]
            rows = evaluate_order(order, cases, population, per_doi, args.rounds)
            for item in rows:
                item["query_id"] = record["query_id"]
                item["query"] = record["query"]
                item["kind"] = record["kind"]
            evaluations += rows
            per_query.append({"query_id": record["query_id"], "query": record["query"],
                              "kind": record["kind"], "order": order, "cases": rows})
        table = aggregate(evaluations, args.rounds)
        rates = [row["success_rate"] for row in table]
        scopes[name] = {"scope": name, "note": SCOPE_NOTES[name],
                        "variant_records": variant_counts[name],
                        "subimages": len(population), "dois": len(per_doi),
                        "oracle_success_rate": oracle_rate(cases, population),
                        "mean_reciprocal_round": reciprocal_rank(evaluations),
                        "normalized_auc": normalized_auc(rates),
                        "per_round": table,
                        "success_by_family": by_family(evaluations, args.rounds),
                        "mrr_by_family": mrr_by_family(evaluations)}
        per_query_out[name] = per_query
        if name == "all_stable_variants":
            coverage = target_coverage(cases, population, per_doi)
            scopes[name]["difficulty"] = by_difficulty(evaluations, coverage, args.rounds)

    baseline = scopes["bare_reconstruction"]["per_round"]
    full = scopes["all_stable_variants"]["per_round"]
    summary = {"targets": len(cases), "queries": len(records), "rounds": args.rounds,
               "target_grid": PAPER_GRID, "tolerance": args.tolerance,
               "topic_filter": "RAG ranking followed by topic screening (shared across all groups)",
               "scopes": scopes,
               "target_coverage": coverage,
               "morphology_gain": {
                   "absolute_pp": [round(100 * (full[index]["success_rate"] - baseline[index]["success_rate"]), 2)
                                   for index in range(args.rounds)],
                   "fold": [round(full[index]["success_rate"] / baseline[index]["success_rate"], 3)
                            if baseline[index]["success_rate"] else None
                            for index in range(args.rounds)]}}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "per_query.json").write_text(json.dumps(per_query_out, ensure_ascii=False, indent=2), encoding="utf-8")
    header = "round," + ",".join(SCOPES)
    lines = [header]
    for index in range(args.rounds):
        lines.append(f"{index + 1}," + ",".join(f"{scopes[name]['per_round'][index]['success_rate']:.4f}"
                                                for name in SCOPES))
    (out / "source_data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(cases)} targets x {len(records)} queries = {len(cases) * len(records)} cases; shared RAG recommendation order\n")
    print(f"{'Scope':<22}{'Variant records':>9}{'SubImage':>10}{'DOI':>6}  " +
          "  ".join(f"R{index + 1}" for index in range(args.rounds)) + "     AUC")
    for name, item in scopes.items():
        rates = "  ".join(f"{row['success_rate'] * 100:5.2f}%" for row in item["per_round"])
        print(f"{name:<22}{item['variant_records']:>9}{item['subimages']:>10}{item['dois']:>6}  "
              f"{rates}  {item['normalized_auc'] * 100:6.2f}%")
    print("\nMorphology + reflection gain relative to bare reconstruction: absolute pp " +
          str(summary["morphology_gain"]["absolute_pp"]) + ", ratio " +
          str(summary["morphology_gain"]["fold"]))
    print("\nFinal-round success rate by property and mean reciprocal first-hit round (MRR)")
    for name, item in scopes.items():
        print(f"  {name}  overall MRR={item['mean_reciprocal_round']}")
        for family, values in item["success_by_family"].items():
            print(f"    {family:<16}" + "  ".join(f"{value * 100:5.2f}%" for value in values) +
                  f"   MRR={item['mrr_by_family'][family]:.4f}")
    difficulty = scopes["all_stable_variants"]["difficulty"]
    print(f"\nTarget difficulty groups (per-paper feasibility threshold {difficulty['cut']:.0%}, full workflow)")
    for name in ("easy", "hard"):
        group = difficulty[name]
        print(f"  {name:<5} {group['targets']:2d} targets / {group['pairs']:4d} cases  " +
              "  ".join(f"{value * 100:5.2f}%" for value in group["per_round"]))
    print("\nPer-paper feasibility for each property target")
    for entry in coverage.values():
        print(f"  {entry['family']:<14}{entry['target']:6.2f}  {entry['coverage'] * 100:5.1f}%  ({entry['dois']} papers)")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
