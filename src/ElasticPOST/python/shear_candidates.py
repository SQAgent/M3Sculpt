"""Compare three principal-engineering-shear selection criteria in DS3d_6."""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pymongo import MongoClient
from skimage.measure import marching_cubes

from elasticpost import directional
from render_mesh import render, surface

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "achievements/paper/plt/Figure5/shear_comparison"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": "#263238",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def select():
    load_dotenv(ROOT / ".env")
    db = MongoClient(os.getenv("MONGO_URL"))[
        os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")
    ]
    fractions = json.loads((OUT.parent / "volume_fractions.json").read_text())
    rows = list(
        db.StructureProperties.find(
            {"design_space_id": "DS3d_6", "elastic_constants.is_stable": True},
            {"_id": 0},
        )
    )
    scores = []
    for r in rows:
        g = np.array([r["elastic_constants"][k] for k in ["G12", "G23", "G31"]], float)
        if not np.isfinite(g).all() or np.any(g <= 0):
            raise ValueError("Invalid engineering constants: " + r["structure_id"])
        f = fractions[r["structure_id"]]
        if not 0 < f <= 1:
            raise ValueError("Invalid volume fraction")
        scores.append(
            [float(g.min() / f), float(3 / np.sum(1 / g) / f), float(g.max() / g.min())]
        )
    a = np.array(scores)
    # Exact numerical ties only: prioritize weak-direction performance, then stable ID.
    choices = [
        min(range(len(rows)), key=lambda j: (-a[j, 0], rows[j]["structure_id"])),
        min(
            range(len(rows)),
            key=lambda j: (-a[j, 1], -a[j, 0], rows[j]["structure_id"]),
        ),
        min(
            range(len(rows)), key=lambda j: (a[j, 2], -a[j, 0], rows[j]["structure_id"])
        ),
    ]
    selected = []
    for panel, j in zip("abc", choices):
        r = rows[j]
        s = db.Structures.find_one(
            {"design_space_id": "DS3d_6", "structure_id": r["structure_id"]}, {"_id": 0}
        )
        path = Path(s["Structure"]["NPY"]["file_path"])
        assert path.is_relative_to("/database/MetamaterialProject/DS3d_6")
        vox = np.load(path)
        assert vox.ndim == 3 and np.isin(vox, [0, 1]).all()
        f = fractions[r["structure_id"]]
        np.testing.assert_allclose(
            np.count_nonzero(vox) / vox.size, f, rtol=0, atol=1e-12
        )
        S = np.linalg.inv(r["properties"]["CH"])
        np.testing.assert_allclose(
            [1 / S[k, k] for k in [5, 3, 4]],
            [r["elastic_constants"][k] for k in ["G12", "G23", "G31"]],
            rtol=1e-8,
        )
        item = {
            "panel": panel,
            "record": r,
            "structure": s,
            "volume_fraction": f,
            "weakest_over_f": a[j, 0],
            "harmonic_over_f": a[j, 1],
            "anisotropy": a[j, 2],
        }
        selected.append(item)
        print(panel, r["structure_id"], scores[j], f, flush=True)
    payload = {
        "design_space": "DS3d_6",
        "stable_records": len(rows),
        "selection": "Max min(G12,G23,G31)/f; max harmonic mean(G12,G23,G31)/f; min max(Gij)/min(Gij). Exact ties: larger weakest/f, then lexicographic structure_id. No density cap or printability filter.",
        "examples": selected,
    }
    (OUT / "selected_examples.json").write_text(json.dumps(payload, indent=2))
    with (OUT / "source_data.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "panel",
                "structure_id",
                "f",
                "G12_GPa",
                "G23_GPa",
                "G31_GPa",
                "weakest_over_f_GPa",
                "harmonic_over_f_GPa",
                "A_G",
                "symmetry",
                "morphology",
            ]
        )
        for i in selected:
            p = i["structure"]["parameters"]
            c = i["record"]["elastic_constants"]
            writer.writerow(
                [
                    i["panel"],
                    i["record"]["structure_id"],
                    i["volume_fraction"],
                    c["G12"],
                    c["G23"],
                    c["G31"],
                    i["weakest_over_f"],
                    i["harmonic_over_f"],
                    i["anisotropy"],
                    p["orthogonal_symmetry"],
                    p["morph_operation"],
                ]
            )
    return selected


def panel(fig, item, images, y, h, norm):
    letter = item["panel"]
    p = item["structure"]["parameters"]
    f = item["volume_fraction"]
    titles = {
        "a": "Maximum weakest-axis shear modulus / f",
        "b": "Maximum harmonic-mean shear modulus / f",
        "c": "Minimum principal shear anisotropy",
    }
    fig.text(
        0.025,
        y + h - 0.045,
        f"({letter})  " + titles[letter],
        fontsize=18,
        weight="bold",
    )
    for k, (x, width) in enumerate([(0.035, 0.235), (0.30, 0.235), (0.56, 0.235)]):
        ax = fig.add_axes([x, y + 0.13, width, h - 0.19])
        ax.imshow(images[k])
        ax.axis("off")
    for x, label in [
        (0.15, "Computed voxel structure"),
        (0.415, r"$G_{\min}(\mathbf{n})/f$"),
        (0.675, r"$G_{\max}(\mathbf{n})/f$"),
    ]:
        fig.text(x, y + 0.115, label, ha="center", fontsize=13)
    cax = fig.add_axes([0.82, y + 0.13, 0.018, h - 0.19])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="PuBuGn"), cax=cax)
    cb.set_ticks([0, norm.vmax / 2, norm.vmax])
    cb.ax.tick_params(labelsize=11, length=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.set_title("GPa", fontsize=11, pad=9)
    fig.text(0.90, y + h - 0.11, f"$f$ = {f:.4f}", fontsize=12)
    c = item["record"]["elastic_constants"]
    for k, key in enumerate(["G12", "G23", "G31"]):
        fig.text(
            0.90,
            y + h - 0.15 - 0.035 * k,
            f"$G_{{{key[1:]}}}/f$\n{c[key] / f:.4f}",
            fontsize=10,
        )
    sym = p["orthogonal_symmetry"]
    symtext = "None" if sym == "xxx" else f"Octant {sym} → mirror x, y, z"
    morph = p["morph_operation"]
    count = morph[3:]
    op = ("Erosion" if morph.startswith("ero") else "Dilation") + " × " + count
    if count == "0":
        op = "None (0 iterations)"
    fig.text(
        0.04,
        y + 0.077,
        f"Weakest/f = {item['weakest_over_f']:.4f} GPa    Harmonic/f = {item['harmonic_over_f']:.4f} GPa    $A_G$ = {item['anisotropy']:.6f}",
        fontsize=12,
    )
    con = p["structure_element"].replace("conn", "").replace("con", "")
    fig.text(0.04, y + 0.043, "Symmetry: " + symtext, color="#7890AC", fontsize=11)
    fig.text(
        0.49,
        y + 0.043,
        f"Morphology: {op} · {con}-neighbor",
        color="#26768C",
        fontsize=11,
    )
    fig.text(0.04, y + 0.016, "DOI: " + p["DOI"], color="#59636D", fontsize=9)


def main():
    items = select()
    th, ph = np.meshgrid(np.linspace(0, np.pi, 361), np.linspace(-np.pi, np.pi, 721))
    fields = [directional(i["record"]["properties"]["CH"], th, ph) for i in items]
    vmax = max(
        v["G_max"].max() / i["volume_fraction"] for i, (_, v) in zip(items, fields)
    )
    norm = Normalize(0, float(np.ceil(vmax * 20) / 20))
    images = []
    audit = []
    for item, (n, values) in zip(items, fields):
        a = np.load(item["structure"]["Structure"]["NPY"]["file_path"])
        v, faces, _, _ = marching_cubes(np.pad(a, 1), 0.5)
        v -= (np.array(a.shape) + 1) / 2
        f = item["volume_fraction"]
        im = [
            render(v, faces),
            surface(n, values["G_min"] / f, "PuBuGn", norm),
            surface(n, values["G_max"] / f, "PuBuGn", norm),
        ]
        images.append(im)
        for name, pixels in zip(["geometry", "Gmin", "Gmax"], im):
            plt.imsave(OUT / f"{item['panel']}_{name}.png", pixels)
        audit.append(
            {
                "panel": item["panel"],
                "directional_Gmin_over_f": float(values["G_min"].min() / f),
                "directional_Gmax_over_f": float(values["G_max"].max() / f),
            }
        )
    fig = plt.figure(figsize=(16, 17), facecolor="white")
    for j, (item, im) in enumerate(zip(items, images)):
        panel(fig, item, im, 0.035 + (2 - j) * 0.32, 0.31, norm)
    fig.text(
        0.5,
        0.012,
        "DS3d_6 · Selection uses G12, G23, G31 only · Directional envelopes are supplementary · Shared PuBuGn scale",
        ha="center",
        fontsize=10,
        color="#59636D",
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"shear_comparison.{ext}", dpi=300)
    plt.close(fig)
    for item, im in zip(items, images):
        # Crop the completed comparison so typography and spacing stay identical.
        from PIL import Image

        with Image.open(OUT / "shear_comparison.png") as source:
            j = "abc".index(item["panel"])
            y = 0.035 + (2 - j) * 0.32
            crop = source.crop(
                (
                    0,
                    int((1 - y - 0.31) * source.height),
                    source.width,
                    int((1 - y) * source.height),
                )
            )
            crop.save(OUT / f"shear_{item['panel']}.png")
    (OUT / "render_audit.json").write_text(json.dumps(audit, indent=2))
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
