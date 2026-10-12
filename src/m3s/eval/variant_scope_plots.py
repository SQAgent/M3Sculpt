"""Publication figure for the geometric-operation ablation of `variant_scope`.

Panel (a) carries the main claim; panel (b) resolves it by property and adds the
first-hit efficiency, so neither panel repeats the other's evidential role.
Follows the house paper-figure style in `.agent/skills/paper-figure-python`.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.lines import Line2D

STYLE_DIR = Path(__file__).resolve().parents[3] / ".agent/skills/paper-figure-python/scripts"
sys.path.insert(0, str(STYLE_DIR))
from paper_style import add_panel_label, apply_house_style, finalize_figure  # noqa: E402


def _register_arial_fallback():
    """Alias a metric-compatible sans-serif as Arial when Arial is missing.

    The house style requires Arial, but many Linux environments do not ship
    Microsoft fonts. Liberation Sans is metric-compatible with Arial, so
    registering it under the Arial name preserves both the intended metrics and
    the style guide's declared fallback behaviour.
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    if "Arial" in installed:
        return True
    registered = False
    for fname in (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    ):
        path = Path(fname)
        if not path.is_file():
            continue
        entry = font_manager.FontEntry(
            fname=str(path),
            name="Arial",
            style="italic" if "Italic" in path.name else "normal",
            weight="bold" if "Bold" in path.name else "normal",
        )
        font_manager.fontManager.ttflist.append(entry)
        registered = True
    return registered

SCOPE_LABEL = {
    "bare_reconstruction": "Bare",
    "symmetry_only": "+Sym",
    "morphology_only": "+Morph",
    "all_stable_variants": "All",
}
SCOPE_LEGEND = {
    "bare_reconstruction": "Bare reconstruction",
    "symmetry_only": "+ Symmetrization",
    "morphology_only": "+ Morphology",
    "all_stable_variants": "All variants",
}
SCOPE_STYLE = {
    "bare_reconstruction": {"color": "#6B7280", "marker": "s", "linestyle": (0, (4, 2))},
    "symmetry_only": {"color": "#E0A458", "marker": "^", "linestyle": (0, (5, 1.5, 1, 1.5))},
    "morphology_only": {"color": "#2A9D8F", "marker": "D", "linestyle": (0, (5, 1.5, 1, 1.5))},
    "all_stable_variants": {"color": "#0F4D92", "marker": "o", "linestyle": "-"},
}
FAMILY_LABEL = {
    "poisson_ratio": "Poisson's ratio",
    "young_modulus": "Young's modulus",
    "shear_modulus": "Shear modulus",
}
FAMILY_COLOR = {
    "poisson_ratio": "#0F4D92",
    "young_modulus": "#E0A458",
    "shear_modulus": "#2A9D8F",
}


def parse_args():
    """Read the plotting options."""
    parser = argparse.ArgumentParser(description="Plot publication figures for geometric-operation ablations (house style)")
    parser.add_argument("--metrics", default="output/variant_scope_rag", help="variant_scope output directory")
    parser.add_argument("--out", default="output/variant_scope_rag", help="Figure output directory")
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution")
    return parser.parse_args()


def panel_rounds(axes, rounds, scopes):
    """Hero panel: cumulative success per round under a shared RAG order."""
    for name, style in SCOPE_STYLE.items():
        values = [row["success_rate"] * 100 for row in scopes[name]["per_round"]]
        axes.plot(rounds, values, lw=1.5, ms=4.2, markeredgewidth=0,
                  label=SCOPE_LEGEND[name], **style)
    bare = [row["success_rate"] * 100 for row in scopes["bare_reconstruction"]["per_round"]]
    full = [row["success_rate"] * 100 for row in scopes["all_stable_variants"]["per_round"]]
    axes.fill_between(rounds, bare, full, color="#0F4D92", alpha=0.07, linewidth=0)
    axes.annotate(f"+{full[-1] - bare[-1]:.1f} pp", xy=(2.55, (full[1] + bare[1]) / 2),
                  ha="center", va="center", fontsize=7, color="#0F4D92")
    for name in SCOPE_STYLE:
        value = scopes[name]["per_round"][-1]["success_rate"] * 100
        axes.annotate(f"{value:.1f}", xy=(rounds[-1], value), xytext=(-3, 5),
                      textcoords="offset points", ha="right", va="bottom", fontsize=6.5,
                      color=SCOPE_STYLE[name]["color"])
    axes.set_xlabel("Recommendation round")
    axes.set_ylabel("Cumulative success rate (%)")
    axes.set_xticks(rounds)
    axes.set_yticks([0, 20, 40, 60, 80, 100])
    axes.set_xlim(0.82, 5.18)
    axes.set_ylim(0, 100)
    axes.legend(fontsize=6.5, loc="lower right", handlelength=2.4, borderaxespad=0.6)


def panel_family(axes, scopes, rounds):
    """Resolve the final round by property, with first-hit efficiency on the right axis."""
    names = list(SCOPE_STYLE)
    positions = np.arange(len(names))
    width = 0.26
    for offset, (family, label) in enumerate(FAMILY_LABEL.items()):
        centers = positions + (offset - 1) * width
        values = [scopes[name]["success_by_family"][family][-1] * 100 for name in names]
        axes.bar(centers, values, width=width, label=label, color=FAMILY_COLOR[family],
                 alpha=0.9, edgecolor="none")
        for center, value in zip(centers, values):
            axes.text(center, value + 1.5, f"{value:.0f}", ha="center", va="bottom", fontsize=6)
    axes.set_xticks(positions)
    axes.set_xticklabels([SCOPE_LABEL[name] for name in names])
    axes.set_ylabel(f"Round-{rounds[-1]} success rate (%)")
    axes.set_ylim(0, 118)
    axes.set_yticks([0, 20, 40, 60, 80, 100])

    twin = axes.twinx()
    for offset, family in enumerate(FAMILY_LABEL):
        centers = positions + (offset - 1) * width
        values = [scopes[name]["mrr_by_family"][family] for name in names]
        twin.plot(centers, values, marker="o", ms=3.6, lw=1.1, linestyle=(0, (1.5, 1.5)),
                  color=FAMILY_COLOR[family], markerfacecolor="white", markeredgewidth=1.0)
    twin.set_ylabel("MRR (first-hit round$^{-1}$)")
    twin.set_ylim(0, 1.18)
    twin.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    twin.spines["right"].set_visible(True)

    handles, labels = axes.get_legend_handles_labels()
    marker = Line2D([], [], color="#4B5563", marker="o", ms=3.6, lw=1.1,
                    linestyle=(0, (1.5, 1.5)), markerfacecolor="white", markeredgewidth=1.0)
    axes.legend(handles + [marker], labels + ["MRR (right axis)"], fontsize=6.5,
                ncol=2, loc="upper left", columnspacing=1.0, handlelength=1.8)


def main():
    """Render the two-panel ablation figure at final manuscript width."""
    args = parse_args()
    _register_arial_fallback()
    apply_house_style(font_size=7.0, spine_width=0.8)
    summary = json.loads((Path(args.metrics) / "summary.json").read_text(encoding="utf-8"))
    scopes = summary["scopes"]
    rounds = list(range(1, summary["rounds"] + 1))

    figure = plt.figure(figsize=(183 / 25.4, 72 / 25.4), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.42)
    axes_a = figure.add_subplot(grid[0, 0])
    axes_b = figure.add_subplot(grid[0, 1])
    panel_rounds(axes_a, rounds, scopes)
    panel_family(axes_b, scopes, rounds)
    add_panel_label(axes_a, "a", x=-0.16)
    add_panel_label(axes_b, "b", x=-0.16)

    saved = finalize_figure(figure, Path(args.out) / "variant_scope_rounds", dpi=args.dpi)
    print(f"Figures saved to {saved[0]}")


if __name__ == "__main__":
    main()
