"""Original tolerance/scope plotting functions with local inputs only."""

import json
from pathlib import Path
import numpy as np

SCOPES = [
    "bare_reconstruction",
    "symmetry_only",
    "morphology_only",
    "all_stable_variants",
]

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
    "bare_reconstruction": {
        "color": "#8A939C",
        "marker": "s",
        "linestyle": (0, (4, 2)),
    },
    "symmetry_only": {
        "color": "#7890AC",
        "marker": "^",
        "linestyle": (0, (5, 1.5, 1, 1.5)),
    },
    "morphology_only": {
        "color": "#26768C",
        "marker": "D",
        "linestyle": (0, (5, 1.5, 1, 1.5)),
    },
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
                cases.append(
                    {
                        "family": c["family"],
                        "target": c["target"],
                        "errors": [r["nearest_error"] for r in c["per_round"]],
                    }
                )
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
    return {
        scope: {tol: reciprocal_rank(data, scope, tol) for tol in tolerances}
        for scope in scopes
    }


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
        total = sum(
            1.0 / hit
            for hit in (first_hit_round(case, tolerance) for case in cases)
            if hit
        )
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
    matrix = np.array(
        [
            [grid[scope][tol][final_round - 1] * 100 for scope in SCOPES]
            for tol in tolerances
        ]
    )
    image = axes.imshow(
        matrix, cmap="PuBuGn", aspect="auto", vmin=30, vmax=95, origin="upper"
    )
    axes.set_xticks(range(len(SCOPES)))
    axes.set_xticklabels([SCOPE_SHORT[s] for s in SCOPES])
    axes.set_yticks(range(len(tolerances)))
    axes.set_yticklabels([f"{t:g}" for t in tolerances])
    axes.set_ylabel("Tolerance (abs.)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axes.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value > 70 else "#1F2937",
            )
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
        ratio = (
            np.array([c["errors"][final_round - 1] for c in data[scope]]) / tolerance
        )
        lg = np.log10(np.clip(ratio, 10**lo, 10**hi))
        dens = _kde(lg, bw_method=0.25)(grid)
        dens = dens / dens.max() * half
        color = SCOPE_STYLE[scope]["color"]
        axes.fill_betweenx(
            grid,
            x - dens,
            x + dens,
            facecolor=color,
            alpha=0.78,
            edgecolor=color,
            linewidth=0.5,
            zorder=3,
        )
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
        axes.plot(
            tolerances,
            values,
            lw=1.5,
            ms=4.0,
            markeredgewidth=0,
            label=SCOPE_LEGEND[scope],
            **SCOPE_STYLE[scope],
        )
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
        values = [
            success_rate_family(data, scope, family, tolerance, final_round) * 100
            for scope in SCOPES
        ]
        axes.bar(
            centers,
            values,
            width=width,
            label=label,
            color=FAMILY_COLOR[family],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.4,
        )
        for center, value in zip(centers, values):
            axes.text(
                center,
                value + 1.5,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#1F2937",
            )
    axes.set_xticks(positions)
    axes.set_xticklabels([SCOPE_SHORT[scope] for scope in SCOPES])
    axes.set_xlabel("Variant scope")
    axes.set_ylabel(f"Success by round {final_round} (%)")
    axes.set_ylim(0, 112)
    axes.set_yticks([0, 20, 40, 60, 80, 100])
    axes.legend(
        fontsize=6.0,
        loc="upper left",
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.5,
    )
