"""Read-only database selection; compute actual occupied voxel fractions."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
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
query = {"design_space_id": "DS3d_6", "elastic_constants.is_stable": True}
rows = list(db.StructureProperties.find(query, {"_id": 0}))
structures = {
    r["structure_id"]: r
    for r in db.Structures.find({"design_space_id": "DS3d_6"}, {"_id": 0})
}
print("Stable records:", len(rows), flush=True)


def volume(row):
    s = structures[row["structure_id"]]
    p = s["Structure"]["NPY"]["file_path"]
    if not Path(p).is_relative_to("/database/MetamaterialProject/DS3d_6"):
        raise ValueError("Geometry outside DS3d_6: " + p)
    a = np.load(p, mmap_mode="r")
    if a.ndim != 3 or not np.isin(a, [0, 1]).all():
        raise ValueError(p)
    f = float(np.count_nonzero(a) / a.size)
    if not 0 < f <= 1:
        raise ValueError(p)
    return f


cache = OUT / "volume_fractions.json"
fractions = json.loads(cache.read_text()) if cache.exists() else {}
for row in rows:
    source = structures[row["structure_id"]]
    if source["design_space_id"] != "DS3d_6" or not Path(
        source["Structure"]["NPY"]["file_path"]
    ).is_relative_to("/database/MetamaterialProject/DS3d_6"):
        raise ValueError("Source outside DS3d_6")
missing = [r for r in rows if r["structure_id"] not in fractions]
with ThreadPoolExecutor(max_workers=12) as pool:
    for i, (r, f) in enumerate(zip(missing, pool.map(volume, missing))):
        fractions[r["structure_id"]] = f
        if i % 10000 == 0:
            print("Volumes:", i, "/", len(missing), flush=True)
cache.write_text(json.dumps(fractions))
criteria = [
    ("a", ["v12", "v13", "v23", "v21", "v31", "v32"], 1, False),
    ("b", ["v12", "v13", "v23", "v21", "v31", "v32"], -1, False),
    ("c", ["E1", "E2", "E3"], 1, True),
    ("d", ["G12", "G23", "G31"], 1, True),
]
selected = []
for panel, keys, sign, normalized in criteria:
    candidates = (
        (
            sign
            * r["elastic_constants"][key]
            / (fractions[r["structure_id"]] if normalized else 1),
            r,
            key,
        )
        for r in rows
        for key in keys
        if isinstance(r["elastic_constants"].get(key), (int, float))
        and np.isfinite(r["elastic_constants"][key])
    )
    score, r, key = max(candidates, key=lambda entry: entry[0])
    item = {
        "panel": panel,
        "criterion": key + ("/f" if normalized else ""),
        "engineering_component": key,
        "selection_value": score * sign,
        "volume_fraction": fractions[r["structure_id"]],
        "record": r,
        "structure": structures[r["structure_id"]],
    }
    selected.append(item)
    print(panel, r["structure_id"], key, score * sign, flush=True)
(OUT / "selected_examples.json").write_text(
    json.dumps(
        {
            "design_space": "DS3d_6",
            "stable_records": len(rows),
            "selection": "Principal-axis engineering constants only: six Poisson components, E1/E2/E3 divided by actual voxel fraction, G12/G23/G31 divided by actual voxel fraction. Directional extrema are never ranking criteria. No printability filter.",
            "examples": selected,
        },
        indent=2,
    )
)
