"""Voxelisation helpers for reconstructed 3D structures."""

import os
from typing import Optional

import numpy as np
import trimesh
from skimage import io


def load_Structure2Voxel(file_path: str, pitch: Optional[float] = None,
                         resolution: Optional[int] = None) -> np.ndarray:
    """Load a 3D structure file and return a filled uint8 voxel matrix.

    Supported extensions: .glb, .stl, .obj, .tif, .npy. Either `pitch` or
    `resolution` must be given for mesh formats.
    """
    file_ext = os.path.splitext(file_path)[-1].lower()

    if file_ext in {".glb", ".stl", ".obj"}:
        mesh = trimesh.load(file_path, process=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        if not pitch:
            if not resolution:
                raise ValueError("Either pitch or resolution must be provided for voxelization.")
            bounds = mesh.bounds
            pitch = (bounds[1] - bounds[0]) / (resolution - 1)
        voxelized = mesh.voxelized(pitch).fill()
        return voxelized.matrix.astype(np.uint8) * 255

    if file_ext == ".tif":
        return io.imread(file_path).astype(np.uint8)

    if file_ext == ".npy":
        return np.load(file_path).astype(np.uint8)

    raise ValueError(".glb, .stl, .obj, .tif, .npy supported only !")
