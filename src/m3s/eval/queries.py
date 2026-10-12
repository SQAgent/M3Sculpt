"""Fixed query set and target grid for recall evaluation; optional LLM expansion."""
import json
from pathlib import Path

from m3s.llm import chat_json

# Phrase-based queries in the exact order and wording of SI S5.1.
SHORT_QUERIES = [
    'Honeycomb structures',
    'Chiral metamaterials',
    'Origami-inspired metamaterials',
    'Kirigami-inspired metamaterials',
    'Triply periodic minimal surface (TPMS) structures',
    'Graded lattice structures',
    'Foam materials',
    'Re-entrant honeycombs',
    'Rotating rigid-body mechanisms',
    'High-strength lattice structures',
    "Negative Poisson's ratio structures",
    'Auxetic metamaterials',
    'Pentamode metamaterials',
    'Lightweight truss metamaterials',
    "Star-shaped negative Poisson's ratio units",
    'High-stiffness cubic lattices',
]

# Sentence-based queries copied verbatim from SI S5.2.
LONG_QUERIES = [
    'Design of high-stiffness, lightweight 3D lattice metamaterials for load-bearing applications',
    "Mechanical response of re-entrant honeycombs with negative Poisson's ratio under tension",
    '3D metamaterials achieving tension--torsion coupling via chiral unit cells',
    'Anisotropic lattice metamaterials with tunable shear modulus',
    'High-strength, lightweight porous metal structures oriented toward additive manufacturing',
    "2D periodic structures with Poisson's ratio controlled through geometric parameters",
    'Closed-cell structures with isotropic elastic properties approaching theoretical upper bounds',
    "Plate-like lattice materials maintaining high Young's modulus at low densities",
    'Design of pentamode metamaterials with extremely low shear modulus',
    'Origami-inspired deployable metamaterials with programmable stiffness',
    'Elastic anisotropy of triply periodic minimal surface (TPMS) bone scaffolds',
    'Design methods for enhancing overall stiffness using graded-density lattices',
    "Stability analysis of negative Poisson's ratio structures under biaxial loading",
    'Calculation of equivalent elastic constants for honeycomb sandwich panels',
]

# property -> (targets, tolerances). Modulus targets and tolerances in GPa.
TARGET_GRID = {
    "poisson_ratio": {"targets": [-0.8, -0.3, 0.1, 0.3], "tolerances": [0.02, 0.05, 0.1]},
    "young_modulus": {"targets": [0.02, 0.1, 0.3], "tolerances": [0.01, 0.05, 0.1]},
    "shear_modulus": {"targets": [0.01, 0.05, 0.1], "tolerances": [0.005, 0.025, 0.05]},
}

FAMILY_LABEL = {"poisson_ratio": "Poisson ratio", "young_modulus": "Young modulus", "shear_modulus": "Shear modulus"}

EXPANSION_PROMPT = """You write search queries for a mechanical metamaterial paper retrieval benchmark.
Produce {count} new English queries that are NOT paraphrases of the existing ones below.
Half should be short phrases (2-8 words, e.g. Honeycomb structures, Chiral metamaterials), half longer topics naming a structure and a mechanical objective.
Do not mention any numeric target value or tolerance.

Existing queries:
{existing}

Output JSON only:
```json
{{"queries": ["...", "..."]}}
```
"""


def requirement_text(family, target, tolerance):
    """Render one target as the sentence the workflow parser accepts."""
    unit = "" if family == "poisson_ratio" else " GPa"
    return f"{FAMILY_LABEL[family]} {target:g}{unit}, tolerance {tolerance:g}{unit}"


def target_cases(families=None, grid=None):
    """Expand the grid into ordered (family, target, tolerance) cases."""
    grid = grid or TARGET_GRID
    cases = []
    for family in families or list(grid):
        spec = grid[family]
        for target in spec["targets"]:
            for tolerance in spec["tolerances"]:
                cases.append({"family": family, "target": float(target), "tolerance": float(tolerance),
                              "text": requirement_text(family, target, tolerance)})
    return cases


def base_queries():
    """Return the built-in query list, short phrases first."""
    return [{"query": q, "kind": "short"} for q in SHORT_QUERIES] + \
           [{"query": q, "kind": "long"} for q in LONG_QUERIES]


def expand_queries(existing, count):
    """Ask the model for extra queries; deduplicate against the existing ones."""
    seen = {q.strip() for q in existing}
    payload = chat_json(EXPANSION_PROMPT.format(count=count, existing="\n".join(f"- {q}" for q in existing)))
    fresh = []
    for item in payload.get("queries", []) if isinstance(payload, dict) else []:
        text = item.strip() if isinstance(item, str) else ""
        if not text or text in seen:
            continue
        seen.add(text)
        fresh.append({"query": text, "kind": "short" if len(text.split()) <= 8 else "long"})
        if len(fresh) >= count:
            break
    return fresh


def load_queries(path=None, expand=0):
    """Build the query set: file overrides the built-in list, then optional expansion."""
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data["queries"] if isinstance(data, dict) else data
        queries = [{"query": item, "kind": "short"} if isinstance(item, str) else dict(item) for item in items]
    else:
        queries = base_queries()
    if expand > 0:
        queries += expand_queries([item["query"] for item in queries], expand)
    for index, item in enumerate(queries, start=1):
        item.setdefault("kind", "long")
        item["query_id"] = f"q{index:03d}"
    return queries
