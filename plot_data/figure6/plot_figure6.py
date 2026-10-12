"""Render Figure 6 from the data beside this script."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.publication_layout import image_panels6, configure_output

DATA_DIR = Path(__file__).resolve().parent
NS = {"s": "http://www.w3.org/2000/svg"}
FONT = 10
KG_FONT = 10

from _shared import design_helpers as source


plt.rcParams.update({"text.color": "#263238"})


def main():
    data = json.loads((DATA_DIR / "source_data.json").read_text())
    for e in data["examples"]:
        for kind in ("base", "selected"):
            npy = e[kind]["structure"]["Structure"]["NPY"]
            npy["file_path"] = str(DATA_DIR / npy["file_path"])
    theta, phi = np.meshgrid(
        np.linspace(0, np.pi, 361), np.linspace(-np.pi, np.pi, 721)
    )
    fields = [
        source.directional(e["selected"]["record"]["properties"]["CH"], theta, phi)
        for e in data["examples"]
    ]
    keys = ["E", "G_max", "nu_max", "nu_min"]
    upper = float(np.ceil(max(fields[j][1][keys[j]].max() for j in [0, 1]) * 10) / 10)
    bound = float(
        np.ceil(max(abs(fields[j][1][keys[j]]).max() for j in [2, 3]) * 10) / 10
    )
    rows = []
    for j, (e, (n, values), key) in enumerate(zip(data["examples"], fields, keys)):
        norm = (
            Normalize(0, upper) if j < 2 else TwoSlopeNorm(0, vmin=-bound, vmax=bound)
        )
        cmap = "PuBuGn" if j < 2 else "BrBG_r"
        images = [
            source.geometry(e["base"]),
            source.geometry(e["selected"]),
            source.surface(n, values[key], cmap, norm),
        ]
        label = [
            r"$E(\mathbf{d})$",
            r"$G_{\max}(\mathbf{d})$",
            r"$\nu_{\max}(\mathbf{d})$",
            r"$\nu_{\min}(\mathbf{d})$",
        ][j]
        ticks = [0, upper / 2, upper] if j < 2 else [-bound, 0, bound]
        rows.append((images, label, norm, cmap, ticks, "GPa" if j < 2 else ""))
        print(f"Figure 6 rendered {e['panel']}", flush=True)
    image_panels6(rows)


if __name__ == "__main__":
    configure_output(DATA_DIR)
    main()
