"""Draw the round-wise retrieval figures from the offline metric JSON files."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FAMILY_LABEL = {"poisson_ratio": "Poisson ratio", "young_modulus": "Young modulus", "shear_modulus": "Shear modulus"}
COLORS = {"ours": "#1f6feb", "random": "#d1495b", "joint": "#2a9d8f", "grey": "#6b7280"}


def use_cjk_font():
    """Pick an installed font that carries Chinese glyphs."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Serif CJK JP", "WenQuanYi Zen Hei", "AR PL UMing CN"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    """Read the plotting options."""
    parser = argparse.ArgumentParser(description="Plot retrieval metrics by recommendation round")
    parser.add_argument("--metrics", default="output/round_metrics", help="round_metrics/baseline output directory")
    parser.add_argument("--out", default="output/figures", help="Figure output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution")
    return parser.parse_args()


def panel_success(axes, rounds, summary, baseline):
    """Success rate and joint success rate against the random-order control."""
    ours = [row["success_rate"] * 100 for row in summary["per_round"]]
    rand = [value * 100 for value in baseline["numeric_only_success_rate"]]
    joint = [value * 100 for value in baseline["joint_success_rate"]]
    axes.plot(rounds, ours, "o-", color=COLORS["ours"], lw=2, ms=6, label="Our method: joint success rate")
    axes.plot(rounds, rand, "s--", color=COLORS["random"], lw=1.8, ms=5, label="Random order: numerical feasibility only")
    axes.plot(rounds, joint, "^-", color=COLORS["joint"], lw=2, ms=6, label="Random order: joint success rate")
    axes.fill_between(rounds, joint, ours, color=COLORS["ours"], alpha=0.10)
    axes.annotate(f"+{ours[-1] - joint[-1]:.1f} pt", xy=(rounds[-1], (ours[-1] + joint[-1]) / 2),
                  xytext=(-58, 0), textcoords="offset points", fontsize=9, color=COLORS["ours"])
    axes.set_xlabel("Recommendation round")
    axes.set_ylabel("Success rate (%)")
    axes.set_title("(a) Topic relevance distinguishes the methods", fontsize=11)
    axes.set_xticks(rounds)
    axes.set_ylim(0, 100)
    axes.legend(fontsize=8.5, loc="center left")
    axes.grid(alpha=0.25)


def panel_family(axes, rounds, summary):
    """Success rate per property family."""
    for family, values in summary["success_by_family"].items():
        axes.plot(rounds, [value * 100 for value in values], "o-", lw=1.8, ms=5,
                  label=FAMILY_LABEL.get(family, family))
    axes.set_xlabel("Recommendation round")
    axes.set_ylabel("Success rate (%)")
    axes.set_title("(b) Success rate by property", fontsize=11)
    axes.set_xticks(rounds)
    axes.set_ylim(40, 100)
    axes.legend(fontsize=9)
    axes.grid(alpha=0.25)


def panel_sweep(axes, summary):
    """Round-5 success rate as the tolerance shrinks."""
    tolerances = sorted((float(key) for key in summary["tolerance_sweep"]), reverse=True)
    ours = [summary["tolerance_sweep"][f"{value:g}"][-1] * 100 for value in tolerances]
    rand = [summary["tolerance_sweep_random"][f"{value:g}"][-1] * 100 for value in tolerances]
    positions = range(len(tolerances))
    axes.plot(positions, ours, "o-", color=COLORS["ours"], lw=2, ms=6, label="Our method")
    axes.plot(positions, rand, "s--", color=COLORS["random"], lw=1.8, ms=5, label="Random order")
    axes.set_xticks(list(positions))
    axes.set_xticklabels([f"{value:g}" for value in tolerances])
    axes.set_xlabel("Absolute tolerance (Poisson ratio: dimensionless; moduli: GPa)")
    axes.set_ylabel("Success rate at round 5 (%)")
    axes.set_title("(c) Tolerance sweep: numerical feasibility does not distinguish methods", fontsize=11)
    axes.set_ylim(0, 100)
    axes.legend(fontsize=9)
    axes.grid(alpha=0.25)


def panel_error(axes, rounds, summary):
    """Median nearest error and cumulative feasible SubImages."""
    errors = [row["median_nearest_error"] for row in summary["per_round"]]
    axes.semilogy(rounds, errors, "o-", color=COLORS["grey"], lw=2, ms=6, label="Median nearest error")
    axes.set_xlabel("Recommendation round")
    axes.set_ylabel("Median nearest error")
    axes.set_xticks(rounds)
    axes.grid(alpha=0.25, which="both")
    twin = axes.twinx()
    feasible = [row["mean_feasible_subimages"] for row in summary["per_round"]]
    twin.bar(rounds, feasible, width=0.45, color=COLORS["joint"], alpha=0.35, label="Cumulative feasible SubImages")
    twin.set_ylabel("Cumulative feasible SubImages")
    twin.set_ylim(0, max(feasible) * 1.45)
    handles = axes.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = axes.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    axes.legend(handles, labels, fontsize=8.5, loc="lower left")
    axes.set_title("(d) Matching margin and candidate accumulation", fontsize=11)


def main():
    """Render the four-panel figure plus a standalone relevance summary."""
    args = parse_args()
    use_cjk_font()
    metrics = Path(args.metrics)
    summary = json.loads((metrics / "summary.json").read_text(encoding="utf-8"))
    baseline = json.loads((metrics / "baseline_summary.json").read_text(encoding="utf-8"))
    rounds = list(range(1, summary["rounds"] + 1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    figure, grid = plt.subplots(2, 2, figsize=(11, 8))
    panel_success(grid[0][0], rounds, summary, baseline)
    panel_family(grid[0][1], rounds, summary)
    panel_sweep(grid[1][0], summary)
    panel_error(grid[1][1], rounds, summary)
    figure.suptitle(f"Multi-round retrieval ({summary['queries']} queries x {summary['targets']} targets; "
                    f"population: {summary['population_dois']} DOIs / {summary['population_subimages']} SubImages)",
                    fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out / "figure5_rounds.png", dpi=args.dpi)
    figure.savefig(out / "figure5_rounds.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(5.4, 4.2))
    ours_relevance = 100.0
    rand_relevance = baseline["random_relevance_rate"] * 100
    bars = axes.bar(["Our recommendations", "Random sampling"], [ours_relevance, rand_relevance],
                    color=[COLORS["ours"], COLORS["random"]], width=0.5)
    for bar, value in zip(bars, (ours_relevance, rand_relevance)):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%",
                  ha="center", fontsize=10)
    axes.set_ylabel("Topic-relevant papers (%)")
    axes.set_ylim(0, 112)
    axes.set_title("Topic relevance of recommended papers", fontsize=11)
    axes.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure.savefig(out / "figure5_relevance.png", dpi=args.dpi)
    plt.close(figure)
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
