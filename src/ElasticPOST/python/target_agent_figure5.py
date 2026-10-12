"""Visualize existing Figure4 cache hits; never rerun retrieval or evaluation."""

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from pymongo import MongoClient
from skimage.measure import marching_cubes
from elasticpost import directional
from render_mesh import render, surface

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "achievements/paper/plt/Figure5"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "output/variant_scope_tol001/per_query.json"
FIELDS = {
    "young_modulus": ["E1", "E2", "E3"],
    "shear_modulus": ["G12", "G23", "G31"],
    "poisson_ratio": ["v12", "v13", "v23", "v21", "v31", "v32"],
}
CASES = [
    ("a", "q023", "young_modulus", 0.7, "Isotropic closed-cell structures"),
    ("b", "q027", "shear_modulus", 0.25, "TPMS scaffolds with elastic anisotropy"),
    ("c", "q022", "poisson_ratio", 1.0, "Geometrically tunable Poisson ratio"),
    ("d", "q013", "poisson_ratio", -1.5, "Re-entrant honeycombs"),
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": "#263238",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def resolve():
    """Resolve a saved first-hit error to its cached engineering-component witness."""
    pq = json.loads(CACHE.read_text())
    evidences = {}
    for path in (ROOT / "output/recall_eval/cache/evidence").glob("*.json"):
        d = json.loads(path.read_text())
        evidences[d["doi"]] = (path, d)
    load_dotenv(ROOT / ".env")
    db = MongoClient(os.getenv("MONGO_URL"))[
        os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")
    ]
    examples = []
    for letter, qid, family, target, theme in CASES:
        q = next(q for q in pq["all_stable_variants"] if q["query_id"] == qid)
        case = next(
            c for c in q["cases"] if c["family"] == family and c["target"] == target
        )
        rnd = case["hit_round"]
        assert rnd is not None
        saved_error = case["per_round"][rnd - 1]["nearest_error"]
        doi = q["order"][rnd - 1]
        ep, e = evidences[doi]
        matches = [
            (r["structure_id"], field, r)
            for r in e["rows"]
            if r["elastic_constants"].get("is_stable") is True
            for field in FIELDS[family]
            if math.isclose(
                abs(r["elastic_constants"][field] - target),
                saved_error,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ]
        if not matches:
            raise ValueError("No principal-component witness for cached result " + qid)
        sid, field, cached = min(matches, key=lambda m: (m[0], m[1]))
        base_id = re.sub(r"_sym[01]{3}_(?:ero|dil)\d+_con\d+$", "", sid)
        docs = []
        for identifier in [base_id, sid]:
            record = db.StructureProperties.find_one(
                {"design_space_id": "DS3d_6", "structure_id": identifier}, {"_id": 0}
            )
            geometry = db.Structures.find_one(
                {"design_space_id": "DS3d_6", "structure_id": identifier}, {"_id": 0}
            )
            if not record or not geometry:
                raise ValueError("Missing DS3d_6 record " + identifier)
            path = Path(geometry["Structure"]["NPY"]["file_path"])
            assert path.is_relative_to("/database/MetamaterialProject/DS3d_6")
            a = np.load(path)
            assert a.ndim == 3 and np.isin(a, [0, 1]).all()
            docs.append(
                {
                    "record": record,
                    "structure": geometry,
                    "volume_fraction": float(np.count_nonzero(a) / a.size),
                }
            )
        base, selected = docs
        value = selected["record"]["elastic_constants"][field]
        np.testing.assert_allclose(
            value, cached["elastic_constants"][field], rtol=0, atol=1e-12
        )
        assert abs(value - target) <= case["tolerance"]
        S = np.linalg.inv(selected["record"]["properties"]["CH"])
        if field[0] == "E":
            v = 1 / S[int(field[1]) - 1, int(field[1]) - 1]
        elif field[0] == "G":
            i = {"G12": 5, "G23": 3, "G31": 4}[field]
            v = 1 / S[i, i]
        else:
            i, j = int(field[1]) - 1, int(field[2]) - 1
            v = -S[min(i, j), max(i, j)] / S[j, j]
        np.testing.assert_allclose(v, value, rtol=1e-8, atol=1e-10)
        run_path = ROOT / f"output/recall_eval/per_query/{qid}.json"
        run = json.loads(run_path.read_text())
        recommendation = next(r for r in run["recommended"] if r["doi"] == doi)
        screening = next(
            r for r in run["screen_log"] if r["doi"] == doi and r["is_relevant"]
        )
        assert recommendation["round"] == rnd
        # Keep original trace and ablation result, without changing or rerunning either.
        scope_cases = {
            name: next(
                c
                for qq in queries
                if qq["query_id"] == qid
                for c in qq["cases"]
                if c["family"] == family and c["target"] == target
            )
            for name, queries in pq.items()
        }
        examples.append(
            {
                "panel": letter,
                "query_id": qid,
                "query": q["query"],
                "theme_en": theme,
                "family": family,
                "target": target,
                "tolerance": case["tolerance"],
                "hit_round": rnd,
                "field": field,
                "value": value,
                "error": saved_error,
                "cached_case": case,
                "cached_scope_cases": scope_cases,
                "recommendation": recommendation,
                "screening": screening,
                "source_files": [str(CACHE), str(ep), str(run_path)],
                "base": base,
                "selected": selected,
            }
        )
        print(
            letter,
            qid,
            field,
            value,
            "error",
            saved_error,
            "round",
            rnd,
            sid,
            flush=True,
        )
    (OUT / "selected_examples.json").write_text(
        json.dumps(
            {
                "design_space": "DS3d_6",
                "mode": "Saved Figure4 outcomes resolved to cached witnesses; no Agent/evaluation rerun.",
                "examples": examples,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    with (OUT / "source_data.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "panel",
                "query_id",
                "query",
                "hit_round",
                "target",
                "tolerance",
                "component",
                "value",
                "absolute_error",
                "base_same_component",
                "volume_fraction",
                "structure_id",
                "symmetry",
                "morphology",
            ]
        )
        for e in examples:
            s = e["selected"]
            p = s["structure"]["parameters"]
            w.writerow(
                [
                    e["panel"],
                    e["query_id"],
                    e["query"],
                    e["hit_round"],
                    e["target"],
                    e["tolerance"],
                    e["field"],
                    e["value"],
                    e["error"],
                    e["base"]["record"]["elastic_constants"][e["field"]],
                    s["volume_fraction"],
                    s["record"]["structure_id"],
                    p["orthogonal_symmetry"],
                    p["morph_operation"],
                ]
            )
    return examples


def mesh_image(doc):
    a = np.load(doc["structure"]["Structure"]["NPY"]["file_path"])
    v, f, _, _ = marching_cubes(np.pad(a, 1), 0.5)
    v -= (np.array(a.shape) + 1) / 2
    return render(v, f)


def component_label(field):
    return "$" + ({"v": r"\nu", "E": "E", "G": "G"}[field[0]]) + "_{" + field[1:] + "}$"


def panel(fig, e, ims, box, norm, cmap):
    x, y, w, h = box
    letter = e["panel"]
    unit = " GPa" if letter in "ab" else ""
    title = {
        "a": "Young’s modulus",
        "b": "Shear modulus",
        "c": "Positive Poisson’s ratio",
        "d": "Negative Poisson’s ratio",
    }[letter]
    fig.text(x, y + h - 0.025, f"({letter})", fontsize=23, weight="bold")
    fig.text(
        x + 0.075 * w,
        y + h - 0.022,
        f"{title}: {e['target']:g}{unit}",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        x + 0.015,
        y + h - 0.055,
        (
            f"{e['theme_en']}  ·  Round {e['hit_round']}"
            if e.get("hit_round") is not None
            else "Direct database selection · Closest principal Poisson ratio"
        ),
        fontsize=10,
        color="#59636D",
    )
    for im, offset, width in zip(ims, [0.0, 0.31, 0.62], [0.28, 0.28, 0.29]):
        ax = fig.add_axes([x + w * offset, y + 0.205, w * width, h - 0.29])
        ax.imshow(im)
        ax.axis("off")
    fig.text(x + w * 0.295, y + 0.28, "→", ha="center", fontsize=22, color="#7890AC")
    for offset, label in [
        (0.14, "Base reconstruction"),
        (0.45, "Selected variant"),
        (
            0.765,
            {
                "a": r"$E(\mathbf{n})$",
                "b": r"$G_{\max}(\mathbf{n})$",
                "c": r"$\nu_{\max}(\mathbf{n})$",
                "d": r"$\nu_{\min}(\mathbf{n})$",
            }[letter],
        ),
    ]:
        fig.text(x + w * offset, y + 0.19, label, ha="center", fontsize=12)
    cax = fig.add_axes([x + w * 0.94, y + 0.21, w * 0.032, h - 0.30])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = (
        [norm.vmin, 0, norm.vmax]
        if letter in "cd"
        else [norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax]
    )
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" for t in ticks])
    cb.ax.tick_params(labelsize=10, length=2)
    cb.outline.set_linewidth(0.6)
    if unit:
        cb.ax.set_title("GPa", fontsize=10, pad=8)
    field = e["field"]
    base = e["base"]["record"]["elastic_constants"][field]
    fig.text(
        x + 0.015,
        y + 0.155,
        f"{component_label(field)}: {base:.4f} → {e['value']:.4f}{unit}",
        fontsize=14,
    )
    fig.text(
        x + 0.015,
        y + 0.122,
        f"|error| = {e['error']:.6f}{unit} ≤ 0.01{unit}  ·  Target met",
        fontsize=11,
        color="#26768C",
    )
    p = e["selected"]["structure"]["parameters"]
    sym = p["orthogonal_symmetry"]
    m = p["morph_operation"]
    op = ("Erosion" if m.startswith("ero") else "Dilation") + " × " + m[3:]
    if m[3:] == "0":
        op = "None (0 iterations)"
    symtext = "None" if sym == "xxx" else f"Octant {sym} → mirror x, y, z"
    fig.text(x + 0.015, y + 0.088, "Symmetry: " + symtext, color="#7890AC", fontsize=11)
    fig.text(
        x + 0.015,
        y + 0.058,
        f"Morphology: {op} · 6-neighbor    $f$ = {e['selected']['volume_fraction']:.3f}",
        color="#26768C",
        fontsize=11,
    )
    fig.text(
        x + 0.015,
        y + 0.029,
        "DOI: " + p["DOI"] + "   [" + e["query_id"] + "]",
        fontsize=9,
        color="#59636D",
    )


def main():
    examples = (
        json.loads((OUT / "selected_examples.json").read_text())["examples"]
        if "--snapshot" in sys.argv
        else resolve()
    )
    th, ph = np.meshgrid(np.linspace(0, np.pi, 361), np.linspace(-np.pi, np.pi, 721))
    fields = [
        directional(e["selected"]["record"]["properties"]["CH"], th, ph)
        for e in examples
    ]
    keys = ["E", "G_max", "nu_max", "nu_min"]
    upper = float(np.ceil(max(fields[j][1][keys[j]].max() for j in [0, 1]) * 10) / 10)
    bound = float(
        np.ceil(max(abs(fields[j][1][keys[j]]).max() for j in [2, 3]) * 10) / 10
    )
    norms = [Normalize(0, upper)] * 2 + [TwoSlopeNorm(0, vmin=-bound, vmax=bound)] * 2
    maps = ["PuBuGn"] * 2 + ["BrBG_r"] * 2
    images = []
    audit = []
    for j, e in enumerate(examples):
        n, values = fields[j]
        v = values[keys[j]]
        ims = [
            mesh_image(e["base"]),
            mesh_image(e["selected"]),
            surface(n, v, maps[j], norms[j]),
        ]
        images.append(ims)
        for label, im in zip(["base", "variant", "field"], ims):
            plt.imsave(OUT / f"{e['panel']}_{label}.png", im)
        audit.append(
            {
                "panel": e["panel"],
                "field": keys[j],
                "min": float(v.min()),
                "max": float(v.max()),
                "color_min": norms[j].vmin,
                "color_max": norms[j].vmax,
            }
        )
    fig = plt.figure(figsize=(20, 14), facecolor="white")
    for e, im, box, norm, cmap in zip(
        examples,
        images,
        [
            (0.02, 0.525, 0.445, 0.455),
            (0.52, 0.525, 0.445, 0.455),
            (0.02, 0.055, 0.445, 0.455),
            (0.52, 0.055, 0.445, 0.455),
        ],
        norms,
        maps,
    ):
        panel(fig, e, im, box, norm, cmap)
    fig.text(
        0.5,
        0.016,
        "DS3d_6 · a–c: cached Agent cases; d: direct database match · Absolute tolerance 0.01 · Directional clouds provide context",
        ha="center",
        fontsize=11,
        color="#59636D",
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure5.{ext}", dpi=300)
    plt.close(fig)
    from PIL import Image

    with Image.open(OUT / "figure5.png") as full:
        for e, (x, y, w, h) in zip(
            examples,
            [
                (0, 0.52, 0.5, 0.48),
                (0.5, 0.52, 0.5, 0.48),
                (0, 0.045, 0.5, 0.48),
                (0.5, 0.045, 0.5, 0.48),
            ],
        ):
            full.crop(
                (
                    int(x * full.width),
                    int((1 - y - h) * full.height),
                    int((x + w) * full.width),
                    int((1 - y) * full.height),
                )
            ).save(OUT / f"figure5{e['panel']}.png")
    (OUT / "render_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
