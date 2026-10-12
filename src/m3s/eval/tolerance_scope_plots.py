"""Publication figure: tolerance x variant-scope response of the multi-round agent.

Every number is recomputed offline from the nearest-error trace cached by
`variant_scope`, because a hit at tolerance d is exactly `nearest_error <= d`.
No database pass and no retrieval call is repeated here.

Panel roles (no panel repeats another's evidential job). The top row sweeps the
tolerance, the bottom row zooms into the operating point, so each row shares an
x-axis meaning:
  (a) heat map      - full tolerance x scope grid at the final round: the headline
                      result plus the evidence that the scope ordering never
                      inverts across the whole tolerance sweep.
  (b) papers-to-hit - mean number of papers examined before the first hit against
                      tolerance; "how much reading", not just "whether". Cases
                      never hit inside the budget are censored at the budget.
  (c) violins       - at the operating tolerance, the distribution of the round-5
                      nearest error normalised by the tolerance, on a log axis.
                      Thresholded rates cannot show that the whole distribution
                      shifts by ~2 decades, nor that hits carry margin instead of
                      sitting on the threshold.
  (d) family bars   - round-5 success resolved by property family, the only place
                      the per-property response to each operation is visible.

The symmetrization/morphology interaction term, previously shown as a stacked
attribution, moves to the supplementary material.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

STYLE_DIR = Path(__file__).resolve().parents[3] / ".agent/skills/paper-figure-python/scripts"
sys.path.insert(0, str(STYLE_DIR))
from paper_style import add_panel_label, apply_house_style, finalize_figure  # noqa: E402

SCOPES = ["bare_reconstruction", "symmetry_only", "morphology_only", "all_stable_variants"]
SCOPE_SHORT = {
    "bare_reconstruction": "Initial",
    "symmetry_only": "+Sym",
    "morphology_only": "+Morph",
    "all_stable_variants": "Expanded",
}
SCOPE_LEGEND = {
    "bare_reconstruction": "Initial",
    "symmetry_only": "+ Symmetrization",
    "morphology_only": "+ Morphology",
    "all_stable_variants": "Expanded",
}
SCOPE_STYLE = {
    "bare_reconstruction": {"color": "#8A939C", "marker": "s", "linestyle": (0, (4, 2))},
    "symmetry_only": {"color": "#7890AC", "marker": "^", "linestyle": (0, (5, 1.5, 1, 1.5))},
    "morphology_only": {"color": "#26768C", "marker": "D", "linestyle": (0, (5, 1.5, 1, 1.5))},
    "all_stable_variants": {"color": "#B26343", "marker": "o", "linestyle": "-"},
}
FAMILY_LABEL = {
    "poisson_ratio": "Poisson's ratio",
    "young_modulus": "Young's modulus",
    "shear_modulus": "Shear modulus",
}
FAMILY_COLOR = {
    "poisson_ratio": "#8C7A9F",
    "young_modulus": "#B26343",
    "shear_modulus": "#26768C",
}


def register_arial_fallback():
    """Alias metric-compatible Liberation Sans as Arial when Arial is absent."""
    if "Arial" in {font.name for font in font_manager.fontManager.ttflist}:
        return True
    registered = False
    for name in (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    ):
        path = Path(name)
        if not path.is_file():
            continue
        font_manager.fontManager.ttflist.append(
            font_manager.FontEntry(
                fname=str(path),
                name="Arial",
                style="italic" if "Italic" in path.name else "normal",
                weight="bold" if "Bold" in path.name else "normal",
            )
        )
        registered = True
    return registered


def load_data(metrics_dir):
    """Load the per-case nearest-error trace for every scope.

    Cases are kept whole rather than flattened per round, because the first-hit
    round behind MRR only exists at the level of one (query, target) pair.
    """
    pq = json.loads((Path(metrics_dir) / "per_query.json").read_text(encoding="utf-8"))
    data = {}
    for scope in SCOPES:
        cases = []
        for q in pq[scope]:
            for c in q["cases"]:
                cases.append({
                    "family": c["family"],
                    "target": c["target"],
                    "errors": [r["nearest_error"] for r in c["per_round"]],
                })
        data[scope] = cases
    return data


def success_rate(data, scope, tolerance, rnd):
    """Fraction of (query, target) pairs with nearest_error <= tolerance at round rnd."""
    cases = data[scope]
    ok = 0
    for case in cases:
        error = case["errors"][rnd - 1]
        if error is not None and error <= tolerance:
            ok += 1
    return ok / len(cases) if cases else 0.0


def first_hit_round(case, tolerance):
    """Earliest round whose nearest error already satisfies the tolerance."""
    for index, error in enumerate(case["errors"], start=1):
        if error is not None and error <= tolerance:
            return index
    return None


def mean_papers_to_hit(data, scope, tolerance, max_rounds=5):
    """Mean number of papers examined before the first hit.

    Cases that never hit within ``max_rounds`` are censored: they contribute
    ``max_rounds`` to the mean, so the result is a lower-bound estimate of the
    true expected number of papers needed. This avoids selection bias from
    conditioning on hits only.
    """
    cases = data[scope]
    total = 0.0
    for case in cases:
        hit = first_hit_round(case, tolerance)
        total += hit if hit else max_rounds
    return total / len(cases) if cases else 0.0


def reciprocal_rank(data, scope, tolerance):
    """Mean reciprocal first-hit round; cases never satisfied contribute zero."""
    cases = data[scope]
    total = 0.0
    for case in cases:
        hit = first_hit_round(case, tolerance)
        if hit:
            total += 1.0 / hit
    return total / len(cases) if cases else 0.0


def reciprocal_rank_grid(data, tolerances, scopes=None):
    """Return a dict scope -> tol -> MRR."""
    if scopes is None:
        scopes = SCOPES
    return {scope: {tol: reciprocal_rank(data, scope, tol) for tol in tolerances}
            for scope in scopes}


def success_rate_grid(data, tolerances, rounds, scopes=None):
    """Return a dict scope -> tol -> [rate per round]."""
    if scopes is None:
        scopes = SCOPES
    grid = {}
    for scope in scopes:
        grid[scope] = {}
        for tol in tolerances:
            grid[scope][tol] = [success_rate(data, scope, tol, r) for r in rounds]
    return grid


def reciprocal_rank_by_family(data, scope, tolerance):
    """Per-family MRR at a given tolerance."""
    groups = {}
    for case in data[scope]:
        groups.setdefault(case["family"], []).append(case)
    out = {}
    for family, cases in groups.items():
        total = sum(1.0 / hit for hit in
                    (first_hit_round(case, tolerance) for case in cases) if hit)
        out[family] = total / len(cases) if cases else 0.0
    return out


def success_rate_family(data, scope, family, tolerance, rnd):
    """Success rate filtered to one property family."""
    cases = [c for c in data[scope] if c["family"] == family]
    ok = 0
    for case in cases:
        error = case["errors"][rnd - 1]
        if error is not None and error <= tolerance:
            ok += 1
    return ok / len(cases) if cases else 0.0


def panel_heatmap(axes, figure, grid, tolerances, final_round):
    """(a) Full tolerance x scope grid at the final round."""
    matrix = np.array([[grid[scope][tol][final_round - 1] * 100 for scope in SCOPES]
                       for tol in tolerances])
    image = axes.imshow(matrix, cmap="PuBuGn", aspect="auto", vmin=30, vmax=95, origin="upper")
    axes.set_xticks(range(len(SCOPES)))
    axes.set_xticklabels([SCOPE_SHORT[s] for s in SCOPES])
    axes.set_yticks(range(len(tolerances)))
    axes.set_yticklabels([f"{t:g}" for t in tolerances])
    axes.set_ylabel("Tolerance (abs.)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axes.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6,
                      color="white" if value > 70 else "#1F2937")
    axes.tick_params(length=0)
    bar = figure.colorbar(image, ax=axes, fraction=0.075, pad=0.03, ticks=[30, 60, 90])
    bar.set_label(f"Success by round {final_round} (%)", fontsize=7)
    bar.ax.tick_params(labelsize=6, length=2)
    bar.outline.set_linewidth(0.8)


def panel_violin(axes, data, tolerance, final_round):
    """(c) Distribution of the round-5 nearest error normalised by the tolerance.

    Operating on log10-transformed error ratios with a manually scaled KDE keeps
    the violin shape meaningful on a log axis (matplotlib's stock violin plot
    computes KDE in linear space and then stretches under log scaling, which
    produces meaningless flat slabs).
    """
    from scipy.stats import gaussian_kde as _kde

    lo, hi = -5.0, 2.0
    grid = np.linspace(lo, hi, 400)
    x_coords = np.arange(1, len(SCOPES) + 1)
    half = 0.42

    for x, scope in zip(x_coords, SCOPES):
        ratio = np.array([c["errors"][final_round - 1] for c in data[scope]]) / tolerance
        lg = np.log10(np.clip(ratio, 10**lo, 10**hi))
        dens = _kde(lg, bw_method=0.25)(grid)
        dens = dens / dens.max() * half
        color = SCOPE_STYLE[scope]["color"]
        axes.fill_betweenx(grid, x - dens, x + dens, facecolor=color, alpha=0.78,
                           edgecolor=color, linewidth=0.5, zorder=3)
        q1, med, q3 = np.percentile(lg, [25, 50, 75])
        axes.vlines(x, q1, q3, color="#1F2937", lw=3.0, zorder=4)
        axes.plot(x, med, "o", ms=3.5, color="white", markeredgewidth=0, zorder=5)

    axes.axhline(0.0, color="#1F2937", lw=0.9, linestyle=(0, (2, 2)), zorder=6)

    axes.set_ylim(lo, hi)
    axes.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2])
    axes.set_xlim(x_coords[0] - 0.6, x_coords[-1] + 0.6)
    axes.set_xticks(x_coords)
    axes.set_xticklabels([SCOPE_SHORT[s] for s in SCOPES])
    axes.set_xlabel("Variant scope")
    axes.set_ylabel(r"log$_{10}$ ($\varepsilon\,/\,\delta$)")


def panel_papers(axes, data, tolerances, rounds_budget):
    """(b) Mean number of papers examined before the first hit, against tolerance."""
    for scope in SCOPES:
        values = [mean_papers_to_hit(data, scope, t, rounds_budget) for t in tolerances]
        axes.plot(tolerances, values, lw=1.5, ms=4.0, markeredgewidth=0,
                  label=SCOPE_LEGEND[scope], **SCOPE_STYLE[scope])
    axes.set_xlabel("Tolerance (abs.)")
    axes.set_ylabel("Mean papers examined")
    axes.set_xlim(tolerances[0] - 0.002, tolerances[-1] + 0.002)
    axes.set_ylim(1.0, rounds_budget + 0.3)
    axes.set_yticks(range(1, rounds_budget + 1))
    axes.set_xticks(tolerances[::2])
    axes.set_xticklabels([f"{t:g}" for t in tolerances[::2]])
    axes.legend(fontsize=6.0, loc="upper right", handlelength=2.2, borderaxespad=0.5)


def panel_family_bars(axes, data, tolerance, final_round):
    """(d) Round-5 success resolved by property family across the four scopes.

    Panels a-c aggregate the three families, which hides that the moduli and the
    Poisson ratio respond to different geometric operations; this panel is the
    only place that decomposition appears.
    """
    positions = np.arange(len(SCOPES))
    width = 0.26
    for offset, (family, label) in enumerate(FAMILY_LABEL.items()):
        centers = positions + (offset - 1) * width
        values = [success_rate_family(data, scope, family, tolerance, final_round) * 100
                  for scope in SCOPES]
        axes.bar(centers, values, width=width, label=label, color=FAMILY_COLOR[family],
                 alpha=0.92, edgecolor="white", linewidth=0.4)
        for center, value in zip(centers, values):
            axes.text(center, value + 1.5, f"{value:.0f}", ha="center", va="bottom",
                      fontsize=5.8, color="#1F2937")
    axes.set_xticks(positions)
    axes.set_xticklabels([SCOPE_SHORT[scope] for scope in SCOPES])
    axes.set_xlabel("Variant scope")
    axes.set_ylabel(f"Success by round {final_round} (%)")
    axes.set_ylim(0, 112)
    axes.set_yticks([0, 20, 40, 60, 80, 100])
    axes.legend(fontsize=6.0, loc="upper left", handlelength=1.4,
                columnspacing=0.9, borderaxespad=0.5)


def write_source_data(path, grid, tolerances, rounds):
    """Persist the exact numbers behind the figure."""
    lines = ["tolerance,scope,round,success_rate"]
    for tol in tolerances:
        for scope in SCOPES:
            for index, rnd in enumerate(rounds):
                lines.append(f"{tol:g},{scope},{rnd},{grid[scope][tol][index]:.4f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_data(path, data, tolerance, final_round):
    """Persist the per-family numbers behind panel (d), including MRR and n."""
    lines = ["tolerance,scope,family,n_pairs,success_rate,mrr"]
    for scope in SCOPES:
        mrr = reciprocal_rank_by_family(data, scope, tolerance)
        for family in FAMILY_LABEL:
            cases = [c for c in data[scope] if c["family"] == family]
            rate = success_rate_family(data, scope, family, tolerance, final_round)
            lines.append(f"{tolerance:g},{scope},{family},{len(cases)},"
                         f"{rate:.4f},{mrr[family]:.4f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_distribution_data(path, data, tolerance, final_round):
    """Persist the error-ratio quantiles behind panel (c)."""
    lines = ["tolerance,scope,n_pairs,p5,p25,median,p75,p95,success_rate"]
    for scope in SCOPES:
        ratio = np.array([c["errors"][final_round - 1] for c in data[scope]]) / tolerance
        p5, p25, med, p75, p95 = np.percentile(ratio, [5, 25, 50, 75, 95])
        rate = float(np.mean(ratio <= 1.0))
        lines.append(f"{tolerance:g},{scope},{len(ratio)},{p5:.4e},{p25:.4e},"
                     f"{med:.4e},{p75:.4e},{p95:.4e},{rate:.4f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_interaction_data(path, grid, tolerances, final_round):
    """Persist the symmetrization/morphology interaction term for the supplement.

    Panel (c) no longer shows this decomposition, so the numbers are kept here to
    remain reproducible from the figure's own source data.
    """
    lines = ["tolerance,bare,symmetry_only,morphology_only,all_variants,"
             "sym_gain,morph_gain,additive_prediction,interaction_pp"]
    for tol in tolerances:
        b = grid["bare_reconstruction"][tol][final_round - 1] * 100
        s = grid["symmetry_only"][tol][final_round - 1] * 100
        m = grid["morphology_only"][tol][final_round - 1] * 100
        a = grid["all_stable_variants"][tol][final_round - 1] * 100
        additive = b + (s - b) + (m - b)
        lines.append(f"{tol:g},{b:.2f},{s:.2f},{m:.2f},{a:.2f},"
                     f"{s - b:.2f},{m - b:.2f},{additive:.2f},{a - additive:+.2f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_papers_data(path, data, tolerances, rounds_budget):
    """Persist the mean papers-to-first-hit behind panel (b)."""
    lines = ["tolerance,scope,mean_papers_censored,hit_rate,mean_papers_hits_only"]
    for tol in tolerances:
        for scope in SCOPES:
            hits = [first_hit_round(c, tol) for c in data[scope]]
            n = len(hits)
            hit_only = [h for h in hits if h]
            censored = sum(h if h else rounds_budget for h in hits) / n
            cond = sum(hit_only) / len(hit_only) if hit_only else float("nan")
            lines.append(f"{tol:g},{scope},{censored:.4f},{len(hit_only) / n:.4f},{cond:.4f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    """Read the plotting options."""
    parser = argparse.ArgumentParser(
        description="Plot publication figures for tolerance x geometric-operation scope (house style)")
    parser.add_argument("--metrics", default="output/variant_scope_tol001",
                        help="variant_scope output directory (must contain per_query.json)")
    parser.add_argument("--out", default="paper/figures/tolerance_scope",
                        help="Figure output directory")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--tolerances", nargs="*", type=float,
                        default=[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
                        help="List of absolute tolerances")
    parser.add_argument("--operating-point", type=float, default=0.01,
                        help="Tolerance used for the main manuscript results")
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution")
    return parser.parse_args()


def main():
    """Render the four-panel tolerance x scope figure at final manuscript width."""
    args = parse_args()
    register_arial_fallback()
    apply_house_style(font_size=7.0, spine_width=0.8)

    data = load_data(args.metrics)
    rounds = list(range(1, args.rounds + 1))
    tolerances = list(args.tolerances)
    grid = success_rate_grid(data, tolerances, rounds)

    figure = plt.figure(figsize=(183 / 25.4, 140 / 25.4), facecolor="white")
    outer = figure.add_gridspec(2, 2, hspace=0.30, wspace=0.30,
                                height_ratios=[1.0, 1.0])
    axes_a = figure.add_subplot(outer[0, 0])
    axes_b = figure.add_subplot(outer[0, 1])
    axes_c = figure.add_subplot(outer[1, 0])
    axes_d = figure.add_subplot(outer[1, 1])

    panel_heatmap(axes_a, figure, grid, tolerances, args.rounds)
    panel_papers(axes_b, data, tolerances, args.rounds)
    panel_violin(axes_c, data, args.operating_point, args.rounds)
    panel_family_bars(axes_d, data, args.operating_point, args.rounds)

    add_panel_label(axes_a, "a", x=-0.20)
    add_panel_label(axes_b, "b", x=-0.16)
    add_panel_label(axes_c, "c", x=-0.16)
    add_panel_label(axes_d, "d", x=-0.16)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_source_data(out_dir / "source_data.csv", grid, tolerances, rounds)
    write_family_data(out_dir / "source_data_family.csv", data,
                      args.operating_point, args.rounds)
    write_distribution_data(out_dir / "source_data_distribution.csv", data,
                            args.operating_point, args.rounds)
    write_interaction_data(out_dir / "source_data_interaction.csv", grid,
                           tolerances, args.rounds)
    write_papers_data(out_dir / "source_data_mean_papers.csv", data,
                      tolerances, args.rounds)
    saved = finalize_figure(figure, out_dir / "tolerance_scope", dpi=args.dpi)
    print(f"Figures saved to {saved[0]}")
    print(f"Source data saved to {out_dir / 'source_data.csv'}")
    print(f"Distribution data saved to {out_dir / 'source_data_distribution.csv'}")
    print(f"Interaction data (supplementary) saved to {out_dir / 'source_data_interaction.csv'}")


if __name__ == "__main__":
    main()
