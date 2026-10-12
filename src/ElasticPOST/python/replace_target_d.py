"""Replace Figure5 d with the closest DS3d_6 principal Poisson component."""

import csv
import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "achievements/paper/plt/Figure5"
load_dotenv(ROOT / ".env")
db = MongoClient(os.getenv("MONGO_URL"))[
    os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")
]
fields = ["v12", "v13", "v23", "v21", "v31", "v32"]
pipeline = [
    {"$match": {"design_space_id": "DS3d_6", "elastic_constants.is_stable": True}},
    {
        "$project": {
            "structure_id": 1,
            "components": [
                {"field": f, "value": "$elastic_constants." + f} for f in fields
            ],
        }
    },
    {"$unwind": "$components"},
    {"$match": {"components.value": {"$type": "number"}}},
    {"$set": {"error": {"$abs": {"$add": ["$components.value", 1.5]}}}},
    {"$sort": {"error": 1, "structure_id": 1, "components.field": 1}},
    {"$limit": 1},
]
result = list(db.StructureProperties.aggregate(pipeline))[0]
result.pop("_id", None)
sid = result["structure_id"]
base_id = re.sub(r"_sym[01]{3}_(?:ero|dil)\d+_con\d+$", "", sid)
docs = []
for identifier in [base_id, sid]:
    r = db.StructureProperties.find_one(
        {"design_space_id": "DS3d_6", "structure_id": identifier}, {"_id": 0}
    )
    s = db.Structures.find_one(
        {"design_space_id": "DS3d_6", "structure_id": identifier}, {"_id": 0}
    )
    if not r or not s:
        raise ValueError("Missing record " + identifier)
    path = Path(s["Structure"]["NPY"]["file_path"])
    assert path.is_relative_to("/database/MetamaterialProject/DS3d_6")
    a = np.load(path)
    assert a.ndim == 3 and np.isin(a, [0, 1]).all()
    docs.append(
        {
            "record": r,
            "structure": s,
            "volume_fraction": float(np.count_nonzero(a) / a.size),
        }
    )
field = result["components"]["field"]
value = result["components"]["value"]
C = np.array(docs[1]["record"]["properties"]["CH"])
assert np.linalg.eigvalsh((C + C.T) / 2).min() > 0
S = np.linalg.inv(C)
i, j = int(field[1]) - 1, int(field[2]) - 1
np.testing.assert_allclose(-S[min(i, j), max(i, j)] / S[j, j], value, rtol=1e-8)
assert result["error"] <= 0.01
payload = json.loads((OUT / "selected_examples.json").read_text())
e = {
    "panel": "d",
    "query_id": "DB-direct",
    "query": "数据库全部稳定结构中最接近 −1.5 的主轴泊松比",
    "theme_en": "Closest database result",
    "family": "poisson_ratio",
    "target": -1.5,
    "tolerance": 0.01,
    "hit_round": None,
    "field": field,
    "value": value,
    "error": result["error"],
    "selection_source": "database_direct",
    "selection_rule": "Minimum absolute error among v12/v13/v23/v21/v31/v32 in all stable DS3d_6 records; no visual filter.",
    "base": docs[0],
    "selected": docs[1],
}
payload["examples"] = [
    e if item["panel"] == "d" else item for item in payload["examples"]
]
payload["mode"] = (
    "Panels a-c: existing Figure4 cached results. Panel d: direct DS3d_6 nearest-principal-Poisson query."
)
(OUT / "selected_examples.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2)
)
(OUT / "database_d_match.json").write_text(json.dumps(result, indent=2))
with (OUT / "source_data.csv").open(newline="") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    rows = list(reader)
for row in rows:
    if row["panel"] == "d":
        p = docs[1]["structure"]["parameters"]
        row.update(
            {
                "query_id": "DB-direct",
                "query": e["query"],
                "hit_round": "",
                "component": field,
                "value": value,
                "absolute_error": result["error"],
                "base_same_component": docs[0]["record"]["elastic_constants"][field],
                "volume_fraction": docs[1]["volume_fraction"],
                "structure_id": sid,
                "symmetry": p["orthogonal_symmetry"],
                "morphology": p["morph_operation"],
            }
        )
with (OUT / "source_data.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)
provenance = json.loads((OUT / "provenance.json").read_text())
provenance["panel_d_override"] = {
    "source": "MongoDB StructureProperties + Structures",
    "design_space": "DS3d_6",
    "selection_rule": e["selection_rule"],
    "match": result,
}
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
print(json.dumps(result, indent=2))
