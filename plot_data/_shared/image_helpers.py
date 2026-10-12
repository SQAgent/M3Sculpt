"""Image and voxel rendering copied from the reconstruction figure."""

import numpy as np
from PIL import Image, ImageOps
from skimage.measure import marching_cubes
from .elasticpost import directional as directional
from .render_mesh import render, surface as surface


def picture(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        fitted = ImageOps.contain(
            bg.convert("RGB"), (944, 944), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", (1024, 1024), "white")
        canvas.paste(fitted, ((1024 - fitted.width) // 2, (1024 - fitted.height) // 2))
        return np.asarray(canvas)


def geometry(path):
    volume = np.load(path)
    vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(float), 1), 0.5)
    vertices -= (np.asarray(volume.shape) + 1) / 2
    return render(vertices, faces, size=1000)


def nice_upper(value):
    power = 10 ** np.floor(np.log10(value))
    return float(np.ceil(value / power * 2) / 2 * power)
