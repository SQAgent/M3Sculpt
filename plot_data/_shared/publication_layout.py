"""Publication layout copied from the original 10-point plotting script."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.path import Path as MPath
from matplotlib.text import Text
from PIL import Image

from .paper_style import apply_house_style, style_axes

DPI, WIDTH, FONT, KG_FONT = 600, 4016, 10, 10
HEIGHTS = {3: 5400, 4: 3557, 5: 3600, 6: 2244}
OUT = None
for font in Path("/usr/share/fonts/truetype/msttcorefonts").glob("Arial*.ttf"):
    font_manager.fontManager.addfont(str(font))


def configure_output(data_dir):
    """Write generated images separately from the copied source data."""
    global OUT
    parser = argparse.ArgumentParser(description="Render the local manuscript figure")
    parser.add_argument("--out", type=Path, default=Path(data_dir) / "rendered")
    args = parser.parse_args()
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def canvas(number):
    apply_house_style(font_size=FONT)
    plt.rcParams.update(
        {
            "axes.titlesize": FONT,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial:bold",
            "mathtext.it": "Arial:italic:bold",
            "mathtext.bf": "Arial:bold",
        }
    )
    return plt.figure(figsize=(WIDTH / DPI, HEIGHTS[number] / DPI), facecolor="white")


def save(fig, number):
    for ax in fig.axes:
        if ax.axison and not getattr(ax, "_colorbar", None):
            style_axes(ax)
    for text in fig.findobj(match=Text):
        text.set_fontsize(
            8
            if text.get_gid() == "bar_value_4d"
            else KG_FONT
            if text.get_gid() == "kg_label"
            else FONT
        )
        text.set_fontfamily("Arial")
        text.set_fontweight("bold")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text():
            continue
        # Invisible axes have unused tick text; exclude it from the audit.
        if text.axes is not None and not text.axes.axison:
            continue
        box = text.get_window_extent(renderer)
        if (
            box.x0 < -1
            or box.y0 < -1
            or box.x1 > fig.bbox.width + 1
            or box.y1 > fig.bbox.height + 1
        ):
            clipped.append(text.get_text())
    if clipped:
        raise ValueError(f"Figure {number} clipped labels: {clipped}")
    assert all(
        t.get_fontsize()
        == (
            8
            if t.get_gid() == "bar_value_4d"
            else KG_FONT
            if t.get_gid() == "kg_label"
            else FONT
        )
        for t in fig.findobj(match=Text)
    )
    # Colorbars deliberately retain their outline instead of four chart spines.
    quantitative = [
        a for a in fig.axes if a.axison and not getattr(a, "_colorbar", None)
    ]
    for ax in quantitative:
        assert all(s.get_visible() for s in ax.spines.values())
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure{number}.{ext}", dpi=DPI, bbox_inches=None)
    with Image.open(OUT / f"figure{number}.png") as im:
        assert im.size == (WIDTH, HEIGHTS[number]), im.size
        preview = im.copy()
        preview.thumbnail((850, 1000))
        preview.save(OUT / f"preview{number}.png")
    plt.close(fig)
    print(
        f"Figure {number}: {(WIDTH, HEIGHTS[number])}, {FONT} pt, exported PNG/PDF/SVG",
        flush=True,
    )


def image_rows(number, rows, headings):
    fig = canvas(number)
    count = len(headings)
    lefts = [0.055, 0.27, 0.485, 0.70] if count == 4 else [0.07, 0.335, 0.60]
    widths = [0.17] * 4 if count == 4 else [0.225] * 3
    for x, width, label in zip(lefts, widths, headings):
        fig.text(x + width / 2, 0.962, label, ha="center", va="center")
    for j, (images, name, norm, cmap, ticks, unit) in enumerate(rows):
        y = 0.715 - j * 0.225
        fig.text(0.015, y + 0.095, f"({'abcd'[j]})", va="center")
        for k, (im, x, width) in enumerate(zip(images, lefts, widths)):
            a = fig.add_axes([x, y, width, 0.18])
            a.imshow(im)
            a.axis("off")
            if number == 4 and k < 2:
                from matplotlib.patches import Rectangle

                a.add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=a.transAxes,
                        fill=False,
                        edgecolor="#7890AC",
                        lw=1.1,
                        clip_on=False,
                    )
                )
            if k + 1 < count:
                fig.text(
                    (x + width + lefts[k + 1]) / 2,
                    y + 0.09,
                    "→",
                    ha="center",
                    va="center",
                    color="#7890AC",
                )
        fig.text(lefts[-1] + widths[-1] / 2, y + 0.19, name, ha="center")
        cax = fig.add_axes([0.89, y + 0.02, 0.017, 0.14])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{t:g}" if number == 4 else f"{t:.2f}" for t in ticks])
        bar.ax.tick_params(length=2)
        bar.outline.set_linewidth(0.6)
        if unit:
            bar.ax.set_title(unit, pad=6)
    save(fig, number)


def image_panels6(rows):
    """Keep the original two-by-two case arrangement and three images per case."""
    fig = canvas(6)
    for j, (images, name, norm, cmap, ticks, unit) in enumerate(rows):
        x, y = (0.015 + (j % 2) * 0.50, 0.51 if j < 2 else 0.025)
        fig.text(x, y + 0.435, f"({'abcd'[j]})", va="top")
        for im, offset, label in zip(
            images, [0, 0.14, 0.28], ["Initial", "Expanded", name]
        ):
            ax = fig.add_axes([x + offset, y + 0.10, 0.13, 0.29])
            ax.imshow(im)
            ax.axis("off")
            fig.text(x + offset + 0.065, y + 0.06, label, ha="center", va="center")
        fig.text(x + 0.138, y + 0.245, "→", ha="center", va="center", color="#7890AC")
        cax = fig.add_axes([x + 0.410, y + 0.125, 0.012, 0.24])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        bar.set_ticks(ticks)
        bar.set_ticklabels([f"{t:g}" for t in ticks])
        bar.ax.tick_params(length=1.5, pad=1)
        bar.outline.set_linewidth(0.6)
        if unit:
            bar.ax.set_title(unit, pad=6)
    save(fig, 6)


def svg_path(d):
    """Decode the absolute M/L/Q/C/z commands used by the saved Matplotlib SVG."""
    tokens = re.findall(r"[MLQCZmlqcz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", d)
    points, codes, i = [], [], 0
    mapping = {
        "M": (1, MPath.MOVETO),
        "L": (1, MPath.LINETO),
        "Q": (2, MPath.CURVE3),
        "C": (3, MPath.CURVE4),
    }
    while i < len(tokens):
        command = tokens[i]
        i += 1
        if command in ("z", "Z"):
            points.append((0, 0))
            codes.append(MPath.CLOSEPOLY)
            continue
        count, code = mapping[command]
        for _ in range(count):
            points.append((float(tokens[i]), float(tokens[i + 1])))
            codes.append(code)
            i += 2
    return MPath(points, codes)
