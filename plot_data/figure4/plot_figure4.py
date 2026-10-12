"""Render Figure 4 from the data beside this script."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.publication_layout import image_rows, configure_output

DATA_DIR = Path(__file__).resolve().parent
NS = {"s": "http://www.w3.org/2000/svg"}
FONT = 10
KG_FONT = 10

from _shared import image_helpers as source


plt.rcParams.update({"text.color": "#263238", "axes.labelcolor": "#263238"})


def main():
    data = json.loads((DATA_DIR / "source_data.json").read_text())
    for example in data["examples"]:
        example["paths"] = {
            key: str(DATA_DIR / value) for key, value in example["paths"].items()
        }
    theta, phi = np.meshgrid(
        np.linspace(0, np.pi, 181), np.linspace(-np.pi, np.pi, 361)
    )
    fields = [
        source.directional(e["record"]["properties"]["CH"], theta, phi)
        for e in data["examples"]
    ]
    emax = source.nice_upper(max(v["E"].max() for _, v in fields))
    gmax = source.nice_upper(fields[3][1]["G_max"].max())
    nmax = source.nice_upper(max(np.abs(v["nu_min"]).max() for _, v in fields))
    norms = {
        "E": Normalize(0, emax),
        "G_max": Normalize(0, gmax),
        "nu_min": TwoSlopeNorm(vmin=-nmax, vcenter=0, vmax=nmax),
    }
    rows = []
    for e, (n, values), key in zip(
        data["examples"], fields, ["nu_min", "E", "E", "G_max"]
    ):
        cmap = "BrBG_r" if key == "nu_min" else "PuBuGn"
        norm = norms[key]
        images = [
            source.picture(e["paths"]["origin"]),
            source.picture(e["paths"]["processed"]),
            source.geometry(e["paths"]["voxel"]),
            source.surface(n, values[key], cmap, norm),
        ]
        label = {
            "E": r"$E(\mathbf{d})$",
            "G_max": r"$G_{\max}(\mathbf{d})$",
            "nu_min": r"$\nu_{\min}(\mathbf{d})$",
        }[key]
        ticks = (
            [norm.vmin, 0, norm.vmax]
            if key == "nu_min"
            else [0, norm.vmax / 2, norm.vmax]
        )
        rows.append(
            (images, label, norm, cmap, ticks, "" if key == "nu_min" else "GPa")
        )
        print(f"Figure 4 rendered {e['panel']}", flush=True)
    image_rows(
        4,
        rows,
        ["Original\nimage", "Processed\nimage", "Reconstruction", "Elastic\nresponse"],
    )


if __name__ == "__main__":
    configure_output(DATA_DIR)
    main()
