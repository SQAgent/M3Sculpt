"""Render four database variants and directional elasticity surfaces."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from elasticpost import directional, verify
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from render_mesh import render, surface
from skimage.measure import marching_cubes

RENDER_CACHE = {}

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "achievements/paper/plt/Figure5"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": "#263238",
        "axes.labelcolor": "#263238",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def draw_panel(fig, item, box, fields, n, norm, cmap):
    x, y, w, h = box
    panel = item["panel"]
    vf = item["volume_fraction"]
    titles = {
        "a": "Largest principal Poisson’s ratio",
        "b": "Smallest principal Poisson’s ratio",
        "c": "Largest principal Young’s modulus / f",
        "d": "Largest principal shear modulus / f",
    }
    fig.text(x, y + h - 0.027, f"({panel})", fontsize=23, weight="bold")
    fig.text(x + 0.075 * w, y + h - 0.023, titles[panel], fontsize=15, weight="bold")
    key = {"a": "nu_max", "b": "nu_min", "c": "E", "d": "G_max"}[panel]
    values = fields[key] / (vf if panel in "cd" else 1)
    if panel not in RENDER_CACHE:
        a = np.load(item["structure"]["Structure"]["NPY"]["file_path"])
        vertices, faces, _, _ = marching_cubes(np.pad(a, 1), 0.5)
        vertices -= (np.array(a.shape) + 1) / 2
        RENDER_CACHE[panel] = (render(vertices, faces), surface(n, values, cmap, norm))
        plt.imsave(OUT / f"geometry_{panel}.png", RENDER_CACHE[panel][0])
        plt.imsave(OUT / f"field_{panel}.png", RENDER_CACHE[panel][1])
    ax = fig.add_axes([x + 0.005, y + 0.17, w * 0.40, h - 0.24])
    ax.imshow(RENDER_CACHE[panel][0])
    ax.axis("off")
    surface_ax = fig.add_axes([x + w * 0.43, y + 0.17, w * 0.46, h - 0.24])
    surface_ax.imshow(RENDER_CACHE[panel][1])
    surface_ax.axis("off")
    cax = fig.add_axes([x + w * 0.92, y + 0.17, w * 0.035, h - 0.24])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = (
        [norm.vmin, 0, norm.vmax]
        if panel in "ab"
        else [norm.vmin, (norm.vmin + norm.vmax) / 2, norm.vmax]
    )
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{v:.2f}" for v in ticks])
    cb.ax.tick_params(labelsize=13, length=3, width=0.7)
    cb.outline.set_linewidth(0.6)
    labels = {
        "a": r"$\nu_{\max}(\mathbf{n})$",
        "b": r"$\nu_{\min}(\mathbf{n})$",
        "c": r"$E(\mathbf{n})/f$ [GPa]",
        "d": r"$G_{\max}(\mathbf{n})/f$ [GPa]",
    }
    fig.text(x + w * 0.67, y + 0.16, labels[panel], ha="center", fontsize=15)
    component = item["engineering_component"]
    database_score = item["selection_value"]
    score = database_score
    symbol = {"v": r"\nu", "E": "E", "G": "G"}[component[0]]
    scorelabel = (
        "$"
        + symbol
        + "_{"
        + component[1:]
        + "}"
        + ("/f" if panel in "cd" else "")
        + "$"
    )
    fig.text(
        x + 0.015,
        y + 0.125,
        f"{scorelabel} = {score:.3f}"
        + (" GPa" if panel in "cd" else "")
        + f"     $f$ = {vf:.3f}",
        fontsize=14,
        weight="medium",
    )
    p = item["structure"]["parameters"]
    sym = p["orthogonal_symmetry"]
    morph = p["morph_operation"]
    con = p["structure_element"].replace("conn", "").replace("con", "")
    symtext = "None (xxx)" if sym == "xxx" else f"Octant {sym} → mirror x, y, z"
    mt = {"ero": "Erosion", "dil": "Dilation"}
    op = next(
        (v + " × " + morph[3:] for k, v in mt.items() if morph.startswith(k)), morph
    )
    if morph in ["ero0", "dil0"]:
        op = "None (0 iterations)"
    fig.text(x + 0.015, y + 0.088, "Symmetry: " + symtext, color="#7890AC", fontsize=12)
    fig.text(
        x + 0.015,
        y + 0.057,
        f"Morphology: {op} · {con}-neighbor",
        color="#26768C",
        fontsize=12,
    )
    fig.text(x + 0.015, y + 0.025, "DOI: " + p["DOI"], fontsize=10, color="#59636D")
    return {
        "panel": panel,
        "field_min": float(values.min()),
        "field_max": float(values.max()),
        "database_selection_value": database_score,
        "plotted_extreme": float(values.min() if panel == "b" else values.max()),
        "engineering_component": component,
        "volume_fraction": vf,
        "symmetric_C_min_eigenvalue": float(
            np.linalg.eigvalsh(
                (
                    np.array(item["record"]["properties"]["CH"])
                    + np.array(item["record"]["properties"]["CH"]).T
                )
                / 2
            ).min()
        ),
    }


def main():
    print(verify())
    data = json.loads((OUT / "selected_examples.json").read_text())
    items = data["examples"]
    assert data["design_space"] == "DS3d_6"
    for item in items:
        assert (
            item["record"]["design_space_id"]
            == item["structure"]["design_space_id"]
            == "DS3d_6"
        )
        assert Path(item["structure"]["Structure"]["NPY"]["file_path"]).is_relative_to(
            "/database/MetamaterialProject/DS3d_6"
        )
    # Verify the displayed principal constants against each raw database CH.
    for item in items:
        S = np.linalg.inv(np.asarray(item["record"]["properties"]["CH"], float))
        component = item["engineering_component"]
        if component.startswith("E"):
            axis = int(component[1]) - 1
            value = 1 / S[axis, axis]
        elif component.startswith("G"):
            axis = {"G12": 5, "G23": 3, "G31": 4}[component]
            value = 1 / S[axis, axis]
        else:
            i, j = int(component[1]) - 1, int(component[2]) - 1
            value = -S[min(i, j), max(i, j)] / S[j, j]
        np.testing.assert_allclose(
            value, item["record"]["elastic_constants"][component], rtol=1e-8, atol=1e-10
        )
        actual_f = float(
            np.count_nonzero(
                np.load(item["structure"]["Structure"]["NPY"]["file_path"])
            )
            / np.load(
                item["structure"]["Structure"]["NPY"]["file_path"], mmap_mode="r"
            ).size
        )
        np.testing.assert_allclose(
            actual_f, item["volume_fraction"], rtol=0, atol=1e-12
        )
    th, ph = np.meshgrid(np.linspace(0, np.pi, 361), np.linspace(-np.pi, np.pi, 721))
    calculated = [directional(i["record"]["properties"]["CH"], th, ph) for i in items]
    bound = max(
        abs(calculated[0][1]["nu_max"]).max(), abs(calculated[1][1]["nu_min"]).max()
    )
    bound = float(np.ceil(bound * 10) / 10)
    modulus_bound = max(
        (
            calculated[j][1]["E" if j == 2 else "G_max"] / items[j]["volume_fraction"]
        ).max()
        for j in [2, 3]
    )
    modulus_bound = float(np.ceil((modulus_bound - 1e-7) * 100) / 100)
    norms = [TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound)] * 2 + [
        Normalize(0, modulus_bound)
    ] * 2
    maps = ["BrBG_r", "BrBG_r", "PuBuGn", "PuBuGn"]
    boxes = [
        (0.025, 0.525, 0.455, 0.45),
        (0.525, 0.525, 0.455, 0.45),
        (0.025, 0.07, 0.455, 0.45),
        (0.525, 0.07, 0.455, 0.45),
    ]
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    audits = []
    for i, item in enumerate(items):
        audits.append(
            draw_panel(
                fig,
                item,
                boxes[i],
                calculated[i][1],
                calculated[i][0],
                norms[i],
                maps[i],
            )
        )
    fig.text(
        0.5,
        0.023,
        "Selected by principal engineering constants · Directional envelopes shown for context · Moduli divided by solid volume fraction f",
        ha="center",
        fontsize=11,
        color="#59636D",
    )
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"figure5.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    for i, item in enumerate(items):
        fig = plt.figure(figsize=(9, 6))
        draw_panel(
            fig,
            item,
            (0.05, 0.04, 0.86, 0.93),
            calculated[i][1],
            calculated[i][0],
            norms[i],
            maps[i],
        )
        fig.savefig(OUT / f"figure5{item['panel']}.png", dpi=300)
        plt.close(fig)
    (OUT / "render_audit.json").write_text(json.dumps(audits, indent=2))
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
