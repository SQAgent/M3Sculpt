"""Render Figure 3 from the data beside this script."""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.publication_layout import canvas, save, svg_path, configure_output

DATA_DIR = Path(__file__).resolve().parent
NS = {"s": "http://www.w3.org/2000/svg"}
FONT = 10
KG_FONT = 10

import xml.etree.ElementTree as ET


def main():
    root = ET.parse(DATA_DIR / "knowledge_graph.svg").getroot()

    def find(ident):
        return root.find(f'.//s:g[@id="{ident}"]', NS)

    names = [
        "".join(find(f"text_{k}").find(".//s:text", NS).itertext()).strip()
        for k in range(31, 41)
    ]
    names[5], names[9] = r"M$^3$LLM-7B", r"M$^3$LLM-27B"
    benchmark = dict(
        zip(
            names,
            [float(find(f"text_{k}").find("s:text", NS).text) for k in range(47, 57)],
        )
    )
    baselines = {
        "M3LLM-7B": benchmark["Qwen2.5-7B-Instruct"],
        "M3LLM-27B": benchmark["Qwen3.8-27B"],
    }
    data = json.loads((DATA_DIR / "source_data.json").read_text())
    new_scores = data["additional_benchmark_records"]
    benchmark.update({r["model"]: r["accuracy"] for r in new_scores})
    fig = canvas(3)
    ax = fig.add_axes([0.025, 0.595, 0.49, 0.32])
    graph = find("axes_1")
    patches = graph.findall("s:g", NS)[:37]
    ax.set(xlim=(32.163558, 345.030042), ylim=(278.388, 7.2))
    ax.axis("off")
    nodes, labels, original_bounds = [], [], []
    # Size each box against the actual 10 pt text, retaining its center and shape.
    for patch, group in zip(patches[-15:], graph.findall("s:g", NS)[37:]):
        element = patch.find("s:path", NS)
        path = svg_path(element.get("d"))
        bounds = path.get_extents()
        style = dict(
            s.strip().split(": ", 1)
            for s in element.get("style", "").split(";")
            if s.strip()
        )
        node = PathPatch(
            path,
            facecolor=style.get("fill", "none"),
            edgecolor=style.get("stroke", "none"),
            linewidth=1,
            zorder=3,
            clip_on=False,
        )
        ax.add_patch(node)
        texts = group.findall("s:text", NS)
        label = "\n".join("".join(t.itertext()) for t in texts)
        color = re.search(r"fill: ([^;]+)", texts[0].get("style", ""))
        text = ax.text(
            (bounds.x0 + bounds.x1) / 2,
            (bounds.y0 + bounds.y1) / 2,
            label,
            ha="center",
            va="center",
            linespacing=1.05,
            gid="kg_label",
            fontsize=KG_FONT,
            fontfamily="Arial",
            fontweight="bold",
            zorder=4,
            color=color.group(1) if color else "#28343B",
        )
        nodes.append(node)
        labels.append(text)
        original_bounds.append(bounds)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Affine2D

    padding = 2.5 * fig.dpi / 72  # 2.5 pt per side, independently of export DPI.
    for node, text, bounds in zip(nodes, labels, original_bounds):
        tb = text.get_window_extent(renderer)
        nb = node.get_window_extent(renderer)
        sx, sy = (
            max(1, (tb.width + 2 * padding) / nb.width),
            max(1, (tb.height + 2 * padding) / nb.height),
        )
        cx, cy = bounds.x0 + bounds.width / 2, bounds.y0 + bounds.height / 2
        transform = Affine2D().translate(-cx, -cy).scale(sx, sy).translate(cx, cy)
        node.set_path(transform.transform_path(node.get_path()))
    # Separate the lower-left Abstract/Figures boxes at the common 10 pt size.
    offset = (12, 8)
    nodes[6].set_path(Affine2D().translate(*offset).transform_path(nodes[6].get_path()))
    tx, ty = labels[6].get_position()
    labels[6].set_position((tx + offset[0], ty + offset[1]))
    # Reconnect the 22 existing links to the enlarged boxes so arrowheads stay visible.
    centers = np.array([text.get_position() for text in labels])

    def nearest_node(point):
        distances = [
            np.hypot(
                max(b.x0 - point[0], 0, point[0] - b.x1),
                max(b.y0 - point[1], 0, point[1] - b.y1),
            )
            for b in original_bounds
        ]
        return int(np.argmin(distances))

    for group in patches[:22]:
        paths = group.findall("s:path", NS)
        element = paths[0]
        path = svg_path(element.get("d"))
        source, target = nearest_node(path.vertices[0]), nearest_node(path.vertices[-1])
        assert source != target
        style = dict(
            s.strip().split(": ", 1)
            for s in element.get("style", "").split(";")
            if s.strip()
        )
        # Recover each link's bend from the original SVG quadratic control point.
        # Arc3 uses screen coordinates, so transform before measuring the signed bend.
        radius = 0.0
        if len(path.vertices) == 3 and path.codes[1] == MPath.CURVE3:
            start, control, end = ax.transData.transform(path.vertices)
            chord = end - start
            normal = np.array([chord[1], -chord[0]])
            radius = float(
                np.dot(control - (start + end) / 2, normal) / np.dot(chord, chord)
            )
            if abs(radius) < 1e-6:
                radius = 0.0
        ax.add_patch(
            FancyArrowPatch(
                centers[source],
                centers[target],
                connectionstyle=f"arc3,rad={radius}",
                patchA=nodes[source],
                patchB=nodes[target],
                arrowstyle="-|>" if len(paths) > 1 else "-",
                mutation_scale=6,
                shrinkA=1.5,
                shrinkB=1.5,
                color=style.get("stroke", "#8DA5AE"),
                linewidth=float(style.get("stroke-width", 1)),
                alpha=float(style.get("opacity", 1)),
                zorder=1,
            )
        )
    fig.canvas.draw()
    for node, text in zip(nodes, labels):
        nb, tb = node.get_window_extent(renderer), text.get_window_extent(renderer)
        assert nb.contains(tb.x0 - padding + 0.1, tb.y0 - padding + 0.1)
        assert nb.contains(tb.x1 + padding - 0.1, tb.y1 + padding - 0.1)
    fig.text(0.025, 0.94, "(a)", va="top")
    b = fig.add_axes([0.65, 0.60, 0.32, 0.28])
    curves = []
    for ident, color, marker, name in [
        ("line2d_12", "#B26343", "o", r"M$^3$LLM-27B"),
        ("line2d_13", "#26768C", "s", r"M$^3$LLM-7B"),
    ]:
        path = svg_path(find(ident).find("s:path", NS).get("d")).vertices
        # Reuse exported coordinates without inventing replacement raw data.
        values = 82 - (path[:, 1] - 43.3584) / 201.204 * 8
        curves.append(values.tolist())
        baseline = baselines["M3LLM-27B" if marker == "o" else "M3LLM-7B"]
        b.plot(
            np.arange(7),
            np.r_[baseline, values],
            color=color,
            marker=marker,
            lw=1.3,
            ms=4,
            label=name,
        )
        selected_epoch = 1 if marker == "o" else 3
        b.scatter(
            selected_epoch,
            values[selected_epoch - 1],
            s=65,
            facecolors="none",
            edgecolors=color,
            lw=1,
            zorder=5,
        )
    b.set(
        xlim=(-0.35, 6.35),
        ylim=(68, 82),
        xticks=range(7),
        yticks=range(68, 83, 2),
        xlabel="Epoch",
        ylabel="Mechanical accuracy (%)",
    )
    b.legend(loc="lower right", handlelength=1.5)
    fig.text(0.555, 0.94, "(b)", va="top")
    c = fig.add_axes([0.15, 0.22, 0.365, 0.28])
    names, values = zip(*sorted(benchmark.items(), key=lambda item: item[1]))
    colors = [
        "#26768C"
        if name == r"M$^3$LLM-7B"
        else "#B26343"
        if name == r"M$^3$LLM-27B"
        else "#C9CDD1"
        for name in names
    ]
    c.bar(range(len(names)), values, width=0.43, color=colors)
    c.set(
        xticks=range(len(names)),
        xticklabels=names,
        ylim=(65, 85),
        yticks=[65, 70, 75, 80, 85],
        ylabel="Mechanical accuracy (%)",
        xlim=(-0.6, len(names) - 0.4),
    )
    for label in c.get_xticklabels():
        label.set_rotation(55)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    for i, value in enumerate(values):
        c.text(
            i,
            value + 0.55,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            color="#28343B",
        )
    fig.text(0.025, 0.53, "(c)", va="top")
    mmlu_rows = data["mmlu_records"]
    d = fig.add_axes([0.65, 0.22, 0.32, 0.28])
    for model, color, marker, selected_epoch in [
        ("M3LLM-27B", "#B26343", "o", 1),
        ("M3LLM-7B", "#26768C", "s", 3),
    ]:
        series = sorted(
            (r for r in mmlu_rows if r["model"] == model), key=lambda r: r["epoch"]
        )
        x = [r["epoch"] for r in series]
        y = [r["accuracy"] for r in series]
        d.plot(
            x,
            y,
            color=color,
            marker=marker,
            lw=1.3,
            ms=4,
            label=r"M$^3$LLM-" + model.split("-")[1],
        )
        d.scatter(
            selected_epoch,
            y[x.index(selected_epoch)],
            s=65,
            facecolors="none",
            edgecolors=color,
            lw=1,
            zorder=5,
        )
        d.axhline(y[x.index(0)], color=color, lw=0.8, ls="--", alpha=0.65)
    d.set(
        xlabel="Epoch",
        ylabel="MMLU accuracy (%)",
        xlim=(-0.35, 6.35),
        xticks=range(7),
        ylim=(65, 84),
        yticks=[65, 70, 75, 80],
    )
    d.legend(loc="center right", frameon=False, handlelength=1.5)
    assert np.allclose(b.get_position().size, d.get_position().size)
    fig.text(0.555, 0.53, "(d)", va="top")
    save(fig, 3)


if __name__ == "__main__":
    configure_output(DATA_DIR)
    main()
