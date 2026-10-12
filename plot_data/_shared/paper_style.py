"""Mandatory plotting style and export helpers for paper-figure-python."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_house_style(font_size: float = 7.0, spine_width: float = 0.8) -> None:
    """Apply the user's global paper-figure style before creating a figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.weight": "bold",
            "font.size": font_size,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": spine_width,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def style_axes(ax, spine_width: float = 0.8):
    """Enforce four spines, major ticks only, and bold text."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(spine_width)
    ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        bottom=False,
        left=False,
        top=False,
        right=False,
    )
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
        label.set_fontfamily("Arial")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontfamily("Arial")
        title = legend.get_title()
        title.set_fontweight("bold")
        title.set_fontfamily("Arial")
    return ax


def add_panel_label(
    ax,
    label: str,
    x: float = -0.12,
    y: float = 1.04,
    fontsize: float = 8.0,
    color: str = "black",
):
    """Add a bold, parenthesized panel label such as (a)."""
    clean = label.strip()
    if clean.startswith("(") and clean.endswith(")"):
        clean = clean[1:-1].strip()
    if not re.fullmatch(r"[A-Za-z]+", clean):
        raise ValueError("Panel label must contain English letters only.")
    return ax.text(
        x,
        y,
        f"({clean.lower()})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=color,
        fontsize=fontsize,
        fontfamily="Arial",
        fontweight="bold",
    )


def bold_all_text(fig) -> None:
    """Enforce Arial-first bold styling on every text artist in the figure."""
    for text in fig.findobj(match=mpl.text.Text):
        text.set_fontweight("bold")
        text.set_fontfamily("Arial")


def assert_house_style(fig) -> None:
    """Raise an error when a quantitative axes violates the mandatory style."""
    errors: list[str] = []
    for index, ax in enumerate(fig.axes, start=1):
        if not ax.get_visible() or not ax.axison:
            continue
        missing = [name for name, spine in ax.spines.items() if not spine.get_visible()]
        if missing:
            errors.append(f"axes {index}: hidden spines {missing}")
        for tick in ax.xaxis.get_major_ticks() + ax.xaxis.get_minor_ticks():
            if tick.tick2line.get_visible():
                errors.append(f"axes {index}: top x tick is visible")
        for tick in ax.yaxis.get_major_ticks() + ax.yaxis.get_minor_ticks():
            if tick.tick2line.get_visible():
                errors.append(f"axes {index}: right y tick is visible")
        for tick in ax.xaxis.get_minor_ticks():
            if tick.tick1line.get_visible() or tick.tick2line.get_visible():
                errors.append(f"axes {index}: minor x tick is visible")
        for tick in ax.yaxis.get_minor_ticks():
            if tick.tick1line.get_visible() or tick.tick2line.get_visible():
                errors.append(f"axes {index}: minor y tick is visible")
        for text in ax.findobj(match=mpl.text.Text):
            if text.get_text() and text.get_fontweight() not in ("bold", 700):
                errors.append(f"axes {index}: non-bold text {text.get_text()!r}")
    if errors:
        raise AssertionError("; ".join(errors))


def finalize_figure(
    fig,
    output_base,
    dpi: int = 600,
    formats=("png",),
    bbox_inches: str = "tight",
    close: bool = True,
):
    """Apply final styling and save PNG only unless formats are requested."""
    for ax in fig.axes:
        if ax.get_visible() and ax.axison:
            style_axes(ax)
    bold_all_text(fig)
    assert_house_style(fig)
    base = Path(output_base).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved
