"""Paper geometry: initial voxels, eight octant reflections and six-neighbour sweeps."""

import json
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np

# Sampled legacy DS3d_6 records include ero0..6 and dil1..8; these are explicit
# sweep budgets, not a claim that every historical evaluation used this budget.
DEFAULT_SWEEP = {"erosion_steps": 6, "dilation_steps": 8}


def voxelize_model(model_path, output_path, resolution=64):
    """Keep the largest connected mesh and pad scalar-pitch voxels to an exact cube."""
    import trimesh

    mesh = trimesh.load(model_path, force="mesh", process=True)
    if mesh.is_empty:
        raise ValueError("The reconstructed model is empty")
    components = mesh.split(only_watertight=False)
    if not len(components):
        raise ValueError("The reconstructed model has no connected components")
    mesh = max(
        components,
        key=lambda item: abs(item.volume) if item.is_watertight else item.area,
    )
    extent = float(max(mesh.extents))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid 3D model dimensions")
    mesh.apply_translation(-mesh.bounds[0])
    array = mesh.voxelized(extent / (resolution - 1)).fill().matrix.astype(bool)
    if not array.any() or max(array.shape) > resolution:
        raise ValueError(f"Invalid voxel dimensions: {array.shape}; each axis must be at most {resolution}")
    pads = [
        ((resolution - size) // 2, (resolution - size + 1) // 2) for size in array.shape
    ]
    array = np.pad(array, pads)
    np.save(output_path, array.astype(np.uint8))
    return str(output_path)


def _binary_cube(array):
    array = np.asarray(array)
    if array.ndim != 3 or len(set(array.shape)) != 1 or array.shape[0] % 2:
        raise ValueError("Candidate generation requires a cubic 3D voxel array with an even side length")
    if (
        not np.isfinite(array).all()
        or not np.isin(array, (0, 1)).all()
        or not array.any()
    ):
        raise ValueError("Initial voxels must be a nonempty finite binary array")
    return array.astype(bool)


def octant_reflections(array):
    """Mirror each selected octant across the three cell midplanes independently."""
    array = _binary_cube(array)
    half = array.shape[0] // 2
    results = []
    for bits in product((0, 1), repeat=3):
        slices = tuple(slice(bit * half, (bit + 1) * half) for bit in bits)
        reflected = array[slices].copy()
        for axis, bit in enumerate(bits):
            flipped = np.flip(reflected, axis=axis)
            reflected = np.concatenate(
                (flipped, reflected) if bit else (reflected, flipped), axis=axis
            )
        results.append(("".join(map(str, bits)), reflected))
    return results


def morphology_step(array, operation):
    """Apply a centre-plus-six-face stencil with periodic unit-cell boundaries."""
    array = np.asarray(array, dtype=bool)
    if operation not in {"erosion", "dilation"}:
        raise ValueError(f"Unknown morphology operation: {operation}")
    result = array.copy()
    for axis in range(3):
        for shift in (-1, 1):
            neighbour = np.roll(array, shift, axis=axis)
            if operation == "erosion":
                result &= neighbour
            else:
                result |= neighbour
    return result


def expand_candidates(initial_path, symmetry_candidates, output_dir, sweep=None):
    """Save unique nonempty initial, reflected and morphologically changed voxels."""
    budgets = {**DEFAULT_SWEEP, **(sweep or {})}
    for name, value in budgets.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 32
        ):
            raise ValueError(f"{name} must be an integer between 0 and 32")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    candidates, discarded, by_hash = [], [], {}

    def save(array, label, origin):
        if not array.any():
            discarded.append({"label": label, "reason": "empty_geometry", **origin})
            return
        digest = sha256(
            np.ascontiguousarray(array, dtype=np.uint8).tobytes()
        ).hexdigest()
        provenance = {"label": label, **origin}
        if digest in by_hash:
            by_hash[digest]["aliases"].append(provenance)
            return
        path = folder / f"{label}.npy"
        np.save(path, array.astype(np.uint8))
        record = {
            "label": label,
            "voxel_path": str(path),
            "sha256": digest,
            "relative_density": float(array.mean()),
            "aliases": [provenance],
        }
        candidates.append(record)
        by_hash[digest] = record

    initial = _binary_cube(np.load(initial_path, allow_pickle=False))
    bases = [("initial", initial, None)]
    for entry in symmetry_candidates:
        array = np.load(entry["voxel_path"], allow_pickle=False).astype(bool)
        if array.shape != initial.shape:
            raise ValueError("Reflected candidates must have the same dimensions as the initial voxels")
        bases.append((f"sym{entry['octant']}", array, entry["octant"]))
    for label, array, octant in bases:
        save(array, label, {"octant": octant, "operation": "none", "steps": 0})
        for operation, count in (
            ("erosion", budgets["erosion_steps"]),
            ("dilation", budgets["dilation_steps"]),
        ):
            current = array.copy()
            for step in range(1, count + 1):
                current = morphology_step(current, operation)
                save(
                    current,
                    f"{label}_{operation}{step}",
                    {"octant": octant, "operation": operation, "steps": step},
                )
    manifest = folder / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "resolution": initial.shape[0],
                "boundary": "periodic",
                "sweep": budgets,
                "candidates": candidates,
                "discarded": discarded,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return candidates, str(manifest)
