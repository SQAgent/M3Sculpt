"""Image and voxel rendering copied from the reconstruction figure."""

import numpy as np
from skimage.measure import marching_cubes
from .elasticpost import directional as directional
from .render_mesh import render, surface as surface


def geometry(doc):
    a = np.load(doc["structure"]["Structure"]["NPY"]["file_path"])
    vertices, faces, _, _ = marching_cubes(np.pad(a, 1), 0.5)
    vertices -= (np.array(a.shape) + 1) / 2
    return render(vertices, faces)
