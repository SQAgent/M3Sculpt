"""Render Figure 5 from the data beside this script."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.publication_layout import canvas, save, configure_output

DATA_DIR = Path(__file__).resolve().parent
NS = {"s": "http://www.w3.org/2000/svg"}
FONT = 10
KG_FONT = 10

from _shared import tolerance_panels as source


def main():
    data = source.load_data(DATA_DIR)
    tolerances = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    grid = source.success_rate_grid(data, tolerances, list(range(1, 6)))
    fig = canvas(5)
    axes = [
        fig.add_axes(box)
        for box in [
            (0.12, 0.57, 0.33, 0.36),
            (0.65, 0.57, 0.32, 0.36),
            (0.12, 0.145, 0.33, 0.295),
            (0.65, 0.145, 0.32, 0.295),
        ]
    ]
    source.panel_heatmap(axes[0], fig, grid, tolerances, 5)
    source.panel_papers(axes[1], data, tolerances, 5)
    source.panel_violin(axes[2], data, 0.01, 5)
    source.panel_family_bars(axes[3], data, 0.01, 5)
    for a in (axes[0], axes[2], axes[3]):
        a.tick_params(axis="x", labelrotation=35)
        for label in a.get_xticklabels():
            label.set_ha("right")
    axes[1].legend(
        loc="upper right",
        handlelength=1.4,
        fontsize=FONT,
        labelspacing=0.5,
        borderaxespad=0.3,
    )
    axes[3].legend(loc="upper right", handlelength=1, fontsize=FONT, labelspacing=0.4)
    axes[3].set_ylim(0, 165)
    axes[1].set_xticks([0.01, 0.03, 0.05])
    axes[1].set_xticklabels(["0.01", "0.03", "0.05"])
    # Use smaller horizontal values above the bars, as requested.
    for text in axes[3].texts:
        text.set_rotation(0)
        text.set_gid("bar_value_4d")
        text.set_fontsize(8)
    for a, letter in zip(axes, "abcd"):
        a.text(-0.30, 1.045, f"({letter})", transform=a.transAxes)
    save(fig, 5)


if __name__ == "__main__":
    configure_output(DATA_DIR)
    main()
