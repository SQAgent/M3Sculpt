"""Rebuild Figure 1 using Python geometry and vector drawing only.

The FCC illustration has corner/face-centre nodes joined at nearest-neighbour
distance a/sqrt(2). It is a conceptual exemplar, not a reconstruction result.
Morphology uses periodic binary erosion/dilation of the same 40^3 voxel array.
The homogenized stiffness icon contains symbols only, not simulated values.
Run: python plot_figure1.py
"""

from __future__ import annotations

import io
import itertools
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Arc,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent))
from _shared.paper_style import apply_house_style, finalize_figure
from _shared.publication_layout import configure_output
import json

OUT = DATA_DIR / "rendered"
INK = "#28343B"
TEAL, SAGE, BLUE, VIOLET, RUST = "#1D6A80", "#557C60", "#526F96", "#705888", "#A65032"
PALETTE = [TEAL, SAGE, BLUE, VIOLET, RUST]
SECTOR_GAP = 0.014
for font in Path("/usr/share/fonts/truetype/msttcorefonts").glob("Arial*.ttf"):
    font_manager.fontManager.addfont(str(font))
apply_house_style(font_size=11)


def tint(color, amount=0.93):
    return tuple(np.array(colors.to_rgb(color)) * (1 - amount) + amount)


def ring_arrow(theta, color):
    """One clockwise annular arrow with an integrated head and matching tail."""
    start, end = theta + 36, theta - 36

    def polar(radius, degrees):
        angle = np.deg2rad(degrees)
        return np.column_stack(
            (0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle))
        )

    points = np.vstack(
        (
            polar(0.180, np.linspace(start, end, 100)),
            polar(0.158, [end - 10]),
            polar(0.136, np.linspace(end, start, 100)),
            polar(0.158, [start - 10]),
        )
    )
    return Polygon(
        points,
        closed=True,
        facecolor=color,
        edgecolor="white",
        linewidth=2.0,
        joinstyle="round",
        zorder=3,
    )


def draw_sector(ax, theta, color):
    """Offset each shared radial boundary equally, creating parallel sides."""
    outer, inner, offset = 0.478, 0.192, SECTOR_GAP / 2
    outer_shift = np.degrees(np.arcsin(offset / outer))
    inner_shift = np.degrees(np.arcsin(offset / inner))
    lower, upper = theta - 36, theta + 36

    def arc_points(radius, start, end):
        angles = np.deg2rad(np.linspace(start, end, 160))
        return np.column_stack(
            (0.5 + radius * np.cos(angles), 0.5 + radius * np.sin(angles))
        )

    outer_arc = arc_points(outer, lower + outer_shift, upper - outer_shift)
    inner_arc = arc_points(inner, upper - inner_shift, lower + inner_shift)
    ax.add_patch(
        Polygon(
            np.vstack((outer_arc, inner_arc)),
            closed=True,
            facecolor=tint(color, 0.94),
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        Arc(
            (0.5, 0.5),
            2 * outer,
            2 * outer,
            theta1=lower + outer_shift,
            theta2=upper - outer_shift,
            color=color,
            lw=2.3,
        )
    )
    ax.add_patch(
        Arc(
            (0.5, 0.5),
            2 * inner,
            2 * inner,
            theta1=lower + inner_shift,
            theta2=upper - inner_shift,
            color=tint(color, 0.57),
            lw=0.8,
        )
    )
    sides = [
        np.vstack((inner_arc[-1], outer_arc[0])),
        np.vstack((inner_arc[0], outer_arc[-1])),
    ]
    for side in sides:
        ax.plot(*side.T, color=color, lw=2.3, solid_capstyle="butt", zorder=2)
    return sides


def fcc_geometry(n=40, radius=0.062):
    corners = list(itertools.product([0.0, 1.0], repeat=3))
    face_centres = []
    for axis in range(3):
        for boundary in (0.0, 1.0):
            p = [0.5] * 3
            p[axis] = boundary
            face_centres.append(p)
    nodes = np.array(corners + face_centres)
    edges = [
        (a, b)
        for a, b in itertools.combinations(range(len(nodes)), 2)
        if np.isclose(np.linalg.norm(nodes[a] - nodes[b]), np.sqrt(0.5))
    ]
    assert len(nodes) == 14 and len(edges) == 36
    xyz = np.moveaxis(np.indices((n, n, n), dtype=float), 0, -1)
    xyz = (xyz + 0.5) / n
    distance = np.full((n, n, n), np.inf)
    for a, b in edges:
        direction = nodes[b] - nodes[a]
        t = np.clip(
            np.sum((xyz - nodes[a]) * direction, axis=-1)
            / np.dot(direction, direction),
            0,
            1,
        )
        distance = np.minimum(
            distance, np.linalg.norm(xyz - nodes[a] - t[..., None] * direction, axis=-1)
        )
    return nodes, edges, distance <= radius


def periodic_morphology(mask, operation):
    padded = np.pad(mask, 1, mode="wrap")
    return operation(padded)[1:-1, 1:-1, 1:-1]


def exposed_faces(mask):
    """Return exterior unit-voxel faces, including the unit-cell boundary."""
    faces, normals = [], []
    for axis in range(3):
        for sign in (-1, 1):
            neighbours = np.roll(mask, -sign, axis=axis)
            boundary = [slice(None)] * 3
            boundary[axis] = -1 if sign == 1 else 0
            neighbours[tuple(boundary)] = False
            indices = np.argwhere(mask & ~neighbours)
            other = [j for j in range(3) if j != axis]
            offsets = np.zeros((4, 3))
            offsets[:, axis] = int(sign == 1)
            offsets[:, other] = [[0, 0], [1, 0], [1, 1], [0, 1]]
            faces.append((indices[:, None, :] + offsets[None, :, :]) / mask.shape[0])
            normal = np.zeros(3)
            normal[axis] = sign
            normals.append(np.tile(normal, (len(indices), 1)))
    return np.concatenate(faces), np.concatenate(normals)


def render_lattice(mask, color, nodes=None, edges=None):
    """Orthographic, deterministic render; returns a transparent in-memory image."""
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho")
    # A fine bounding cube distinguishes the conventional FCC cell from its
    # octahedral interior; these lines are guides, not additional lattice struts.
    box_nodes = np.array(list(itertools.product([0.0, 1.0], repeat=3)))
    for a, b in itertools.combinations(range(8), 2):
        if np.isclose(np.linalg.norm(box_nodes[a] - box_nodes[b]), 1):
            ax.plot(*box_nodes[[a, b]].T, color=tint(color, 0.52), lw=0.62, alpha=0.75)
    if mask is not None:
        faces, normals = exposed_faces(mask)
        light = np.array([-0.5, -0.8, 1.0])
        light /= np.linalg.norm(light)
        illumination = 0.73 + 0.27 * np.maximum(normals @ light, 0)
        rgb = np.array(colors.to_rgb(color))
        facecolors = np.clip(illumination[:, None] * rgb + 0.045, 0, 1)
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=facecolors,
                edgecolors=(*rgb * 0.63, 0.38),
                linewidths=0.13,
                antialiased=True,
                zsort="average",
            )
        )
    else:
        for a, b in edges:
            ax.plot(*nodes[[a, b]].T, color=color, lw=3.7, solid_capstyle="round")
        ax.scatter(*nodes.T, s=18, color=color, depthshade=False)
    ax.set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.015), zlim=(-0.015, 1.015))
    ax.set_box_aspect((1, 1, 1), zoom=1.22)
    ax.view_init(elev=24, azim=-53)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, transparent=True, pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    im = Image.open(buffer).convert("RGBA")
    return im.crop(im.getbbox())


def main():
    global OUT
    OUT = configure_output(DATA_DIR)
    spec = json.loads((DATA_DIR / "source_data.json").read_text())
    nodes, edges, base = fcc_geometry(
        n=spec["voxel_resolution"], radius=spec["strut_radius_fraction"]
    )
    eroded = periodic_morphology(base, binary_erosion)
    dilated = periodic_morphology(base, binary_dilation)
    for name, mask in [("eroded", eroded), ("base", base), ("dilated", dilated)]:
        assert label(mask)[1] == 1, f"Disconnected FCC geometry: {name}"
        for axis in range(3):
            assert np.array_equal(np.take(mask, 0, axis), np.take(mask, -1, axis))
        print(
            f"{name}: {mask.sum()} voxels, connected; opposite cell faces match",
            flush=True,
        )
    assert eroded.sum() < base.sum() < dilated.sum()
    images = {
        "query": render_lattice(base, SAGE),
        "source": render_lattice(None, BLUE, nodes, edges),
        "reconstruction": render_lattice(base, BLUE),
        "eroded": render_lattice(eroded, VIOLET),
        "base": render_lattice(base, VIOLET),
        "dilated": render_lattice(dilated, VIOLET),
        "homogenization": render_lattice(base, RUST),
    }
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_axis_off()

    def text(x, y, s, size=13, color=INK, **kwargs):
        return ax.text(
            x, y, s, fontsize=size, color=color, ha="center", va="center", **kwargs
        )

    def arrow(a, b, color, scale=12, lw=1.8, rad=0):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=scale,
                linewidth=lw,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    def picture(name, x, y, width):
        im = images[name]
        height = width * im.height / im.width
        ax.imshow(
            im,
            extent=(x - width / 2, x + width / 2, y - height / 2, y + height / 2),
            zorder=5,
        )

    sector_sides = []
    for k, color in enumerate(PALETTE):
        theta = 90 - 72 * k
        sector_sides.append(draw_sector(ax, theta, color))
        ax.add_patch(ring_arrow(theta, color))
        angle = np.deg2rad(theta)
        text(
            0.5 + 0.158 * np.cos(angle),
            0.5 + 0.158 * np.sin(angle),
            str(k + 1),
            13,
            "white",
            zorder=4,
        )

    for k in range(5):
        a, b = sector_sides[k][0], sector_sides[(k + 1) % 5][1]
        assert np.allclose(a[1] - a[0], b[1] - b[0]), "Nonparallel sector boundaries"
        assert np.allclose(np.linalg.norm(a - b, axis=1), SECTOR_GAP), (
            "Uneven sector gap"
        )

    text(0.5, 0.504, "M³Sculptor", 22)
    text(0.5, 0.468, "Knowledge-driven design", 8.2, "#59666E")

    # 1: literature -> connected multimodal knowledge.
    text(0.5, 0.875, "Knowledge graph\nconstruction", 17, TEAL, linespacing=1.12)
    for dx, dy in [(-0.008, -0.007), (0, 0)]:
        ax.add_patch(
            FancyBboxPatch(
                (0.363 + dx, 0.723 + dy),
                0.080,
                0.099,
                boxstyle="round,pad=0.003,rounding_size=0.004",
                fc="white",
                ec=tint(TEAL, 0.38),
                lw=1.0,
            )
        )
    for y, w in [(0.802, 0.047), (0.789, 0.055), (0.776, 0.048)]:
        ax.plot([0.377, 0.377 + w], [y, y], color=TEAL, lw=1.7, solid_capstyle="round")
    ax.add_patch(Rectangle((0.377, 0.737), 0.023, 0.025, color=tint(TEAL, 0.72), lw=0))
    ax.plot(
        [0.38, 0.388, 0.394, 0.398], [0.741, 0.75, 0.745, 0.755], color=TEAL, lw=1.1
    )
    for y in [0.758, 0.748, 0.738]:
        ax.plot([0.407, 0.430], [y, y], color=tint(TEAL, 0.2), lw=1.2)
    arrow((0.462, 0.773), (0.519, 0.773), TEAL)
    pts = np.array(
        [
            [0.568, 0.773],
            [0.617, 0.813],
            [0.640, 0.761],
            [0.604, 0.719],
            [0.541, 0.726],
            [0.545, 0.816],
        ]
    )
    pts[:, 1] += 0.015
    for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (3, 4)]:
        ax.plot(*pts[[a, b]].T, color=tint(TEAL, 0.45), lw=1.3)
    for j, p in enumerate(pts):
        ax.add_patch(
            Circle(
                p,
                0.010 if j else 0.017,
                facecolor=TEAL if j == 0 else tint(TEAL, 0.78),
                edgecolor=TEAL,
                lw=1.1,
                zorder=3,
            )
        )
    text(0.411, 0.702, "Literature", 8.5, TEAL)
    text(0.566, 0.711, "Knowledge graph", 8.5, TEAL)

    # 2: a clear natural-language bubble and a retrieved structural exemplar.
    text(0.790, 0.689, "Design\nrequirements", 17, SAGE, linespacing=1.12)
    ax.add_patch(
        FancyBboxPatch(
            (0.731, 0.597),
            0.157,
            0.052,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            fc="white",
            ec=SAGE,
            lw=1.3,
        )
    )
    ax.add_patch(
        Polygon(
            [[0.747, 0.591], [0.747, 0.577], [0.767, 0.591]],
            closed=True,
            fc="white",
            ec=SAGE,
            lw=1.3,
            zorder=3,
        )
    )
    ax.plot([0.748, 0.766], [0.591, 0.591], color="white", lw=2, zorder=4)
    text(0.809, 0.624, "Truss metamaterial", 9, SAGE)
    arrow((0.816, 0.586), (0.816, 0.571), SAGE, scale=11, lw=1.5)
    picture("query", 0.815, 0.506, 0.120)
    text(0.825, 0.434, "Retrieved exemplar", 8.5, SAGE)

    # 3: consistent source geometry and voxelized reconstruction.
    text(0.692, 0.300, "3D reconstruction", 17, BLUE)
    ax.add_patch(
        FancyBboxPatch(
            (0.579, 0.156),
            0.086,
            0.094,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            fc="white",
            ec=tint(BLUE, 0.45),
            lw=0.9,
        )
    )
    picture("source", 0.622, 0.204, 0.076)
    arrow((0.680, 0.204), (0.707, 0.204), BLUE, scale=12)
    picture("reconstruction", 0.765, 0.204, 0.103)
    text(0.622, 0.132, "Image", 8.5, BLUE)
    text(0.750, 0.132, "Voxels", 8.5, BLUE)

    # 4: periodic operations applied to the exact same binary FCC structure.
    text(0.322, 0.300, "Morphological\noperations", 17, VIOLET, linespacing=1.10)
    for name, x, w in [
        ("eroded", 0.235, 0.082),
        ("base", 0.338, 0.082),
        ("dilated", 0.441, 0.082),
    ]:
        picture(name, x, 0.207, w)
    arrow((0.295, 0.207), (0.278, 0.207), VIOLET, scale=9, lw=1.2)
    arrow((0.381, 0.207), (0.398, 0.207), VIOLET, scale=9, lw=1.2)
    for x, s in [(0.235, "Erosion"), (0.338, "Original"), (0.441, "Dilation")]:
        text(x, 0.143, s, 8.5, VIOLET)

    # 5: periodic specimen and symbolic homogenized stiffness matrix.
    text(0.190, 0.669, "Computational\nhomogenization", 17, RUST, linespacing=1.12)
    picture("homogenization", 0.126, 0.531, 0.137)
    arrow((0.200, 0.516), (0.231, 0.516), RUST, scale=10, lw=1.5)
    # Bracketed 6 x 6 symbolic matrix is a workflow icon, not a data heatmap.
    mx, my, step = 0.250, 0.534, 0.0067
    for i in range(6):
        for j in range(6):
            ax.add_patch(
                Rectangle(
                    (mx + j * step, my - i * step),
                    step * 0.70,
                    step * 0.70,
                    facecolor=RUST if i == j else tint(RUST, 0.70),
                    edgecolor="none",
                )
            )
    ax.plot(
        [mx - 0.005, mx - 0.009, mx - 0.009, mx - 0.005],
        [my + 0.009, my + 0.009, my - 0.040, my - 0.040],
        color=RUST,
        lw=1.1,
    )
    ax.plot(
        [mx + 0.043, mx + 0.047, mx + 0.047, mx + 0.043],
        [my + 0.009, my + 0.009, my - 0.040, my - 0.040],
        color=RUST,
        lw=1.1,
    )
    text(mx + 0.019, 0.571, r"$\mathbf{C}^{\mathrm{H}}$", 13, RUST)
    text(0.156, 0.437, "Effective properties", 9, RUST)

    # Run layout/typography checks and save a print-resolution final figure.
    fig.canvas.draw()
    for artist in ax.texts:
        assert artist.get_fontweight() == "bold"
        bbox = artist.get_window_extent(fig.canvas.get_renderer())
        assert fig.bbox.contains(bbox.x0, bbox.y0) and fig.bbox.contains(
            bbox.x1, bbox.y1
        ), artist.get_text()
        # Check the entire label against its own sector, including its diagonals
        # and curved edges; being inside the canvas alone is insufficient.
        if artist.get_color() in PALETTE:
            sector_angle = 90 - 72 * PALETTE.index(artist.get_color())
            box = bbox.transformed(ax.transData.inverted())
            corners = (
                np.array([[x, y] for x in (box.x0, box.x1) for y in (box.y0, box.y1)])
                - 0.5
            )
            radius = np.linalg.norm(corners, axis=1)
            angle = (
                np.degrees(np.arctan2(corners[:, 1], corners[:, 0]))
                - sector_angle
                + 180
            ) % 360 - 180
            clearance = np.minimum.reduce(
                (
                    radius - 0.192,
                    0.478 - radius,
                    radius * np.sin(np.deg2rad(36 - np.abs(angle))) - SECTOR_GAP / 2,
                )
            )
            assert clearance.min() >= 0.007, (artist.get_text(), clearance.min())
    outputs = finalize_figure(
        fig, OUT / "figure1", dpi=600, formats=("png", "svg"), bbox_inches=None
    )
    for path in outputs:
        print(path, flush=True)


if __name__ == "__main__":
    main()
