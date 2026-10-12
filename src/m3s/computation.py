"""Live pipeline adapters. No property-database access in this module."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def mode_for(state):
    mode = state.get("evaluation_mode", "normal")
    if mode not in {"normal", "test"}:
        raise ValueError("evaluation_mode must be normal or test")
    return mode


# Natural-language constituent overrides are validated separately from targets.
DEFAULT_CONFIG = {"young_gpa": 1.0, "poisson": 0.3, "resolution": 64, "symmetry": "orthorhombic",
                  "solver_backend": "python"}


def validate_config(config=None):
    """Fill the fixed base-material defaults, then check the resulting values."""
    merged = {**DEFAULT_CONFIG, "solver_backend": os.getenv("M3S_HOMOGENIZATION_BACKEND", "python"), **(config or {})}
    try:
        young = float(merged["young_gpa"])
        poisson = float(merged["poisson"])
        resolution = int(merged["resolution"])
        symmetry = merged["symmetry"]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid computation settings: constituent Young modulus (GPa), Poisson ratio, symmetry and voxel resolution are required.") from exc
    if not math.isfinite(young) or young <= 0 or not math.isfinite(poisson) or not -1 < poisson < .5:
        raise ValueError("Constituent Young modulus must be positive; Poisson ratio must be within (-1, 0.5).")
    if resolution != float(merged["resolution"]) or not 8 <= resolution <= 256 or resolution % 2:
        raise ValueError("Voxel resolution must be an even integer between 8 and 256.")
    if symmetry != "orthorhombic":
        raise ValueError("The octant reflection pipeline only supports orthorhombic symmetry.")
    backend = merged["solver_backend"]
    if backend not in {"python", "chfem"}:
        raise ValueError("The homogenization solver must be python or chfem")
    return {"young_gpa": young, "poisson": poisson, "resolution": resolution, "symmetry": symmetry,
            "solver_backend": backend}


def process_image(source, output):
    """Use the image-edit interface from the original ImageEnhance implementation."""
    import base64
    import mimetypes

    import requests
    from dashscope import MultiModalConversation
    key = os.environ.get("AL_API_KEY")
    if not key:
        raise RuntimeError("Image processing requires AL_API_KEY")
    mime = mimetypes.guess_type(source)[0] or "image/png"
    image = base64.b64encode(Path(source).read_bytes()).decode()
    response = MultiModalConversation.call(
        api_key=key, model="qwen-image-edit-plus", stream=False, n=1, watermark=False,
        messages=[{"role": "user", "content": [
            {"image": f"data:{mime};base64,{image}"},
            {"text": "Preserve the original structure and its geometric connectivity. Remove only unrelated text and annotations; do not invent structural features."}]}],
        prompt_extend=False, size="1024*1024",
    )
    if response.status_code != 200:
        raise RuntimeError(f"Image processing failed: {response.code} {response.message}")
    blocks = response.output.choices[0].message.content
    url = next((block.get("image") for block in blocks if block.get("image")), None)
    if not url:
        raise RuntimeError("Image processing returned no image")
    result = requests.get(url, timeout=300)
    result.raise_for_status()
    Path(output).write_bytes(result.content)
    return output


def reconstruct(source, output_dir):
    from m3s.client.m3hunyuan import M3HunyuanClient
    from m3s.configuration import HunyuanConfig
    config = HunyuanConfig(
        url=os.getenv("HUNYUAN_URL", "http://localhost:7860/generate"),
        output_path=output_dir, image_path=source, seed=42,
        octree_depth=7, guidance_scale=7.5, face_count=5000,
    )
    client = M3HunyuanClient(config)
    client.generate()
    if not Path(client.largest_model_path).is_file():
        raise RuntimeError("3D reconstruction did not produce a valid STL file")
    return client.largest_model_path


def symmetrize(model_path, output_path, config):
    import numpy as np
    import trimesh
    mesh = trimesh.load(model_path, force="mesh")
    extent = float(max(mesh.extents))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid 3D model dimensions")
    # Scalar pitch preserves aspect ratio. Center-padding defines a cubic cell.
    array = mesh.voxelized(extent / (config["resolution"] - 1)).fill().matrix.astype(bool)
    array = symmetric_voxels(array, config["symmetry"])
    np.save(output_path, array.astype(np.uint8))
    return output_path


def symmetric_voxels(array, symmetry):
    """Union over explicit point-group operations; never rescale individual axes."""
    from itertools import permutations

    import numpy as np
    array = np.asarray(array, dtype=bool)
    if array.ndim != 3 or not array.any():
        raise ValueError("Voxels are empty or are not a 3D array")
    n = max(array.shape)
    array = np.pad(array, [((n-s)//2, n-s-(n-s)//2) for s in array.shape])
    result = array.copy()
    for axis in range(3):
        result |= np.flip(result, axis=axis)
    if symmetry == "tetragonal":
        base = result.copy()
        for k in (1, 2, 3):
            result |= np.rot90(base, k, axes=(0, 1))
    elif symmetry == "cubic":
        base = result.copy()
        for axes in permutations(range(3)):
            result |= base.transpose(axes)
    elif symmetry != "orthorhombic":
        raise ValueError("Unsupported symmetry")
    return result


def homogenize(voxel_path, output_dir, config):
    """Use Python SI FEM by default, or explicitly select the CHFEM GPU adapter."""
    backend = config.get("solver_backend", "python")
    if backend == "python":
        from m3s.homogenization import homogenize as python_homogenize
        return python_homogenize(voxel_path, output_dir, config)
    if backend != "chfem":
        raise ValueError(f"Unknown homogenization solver: {backend}")
    folder = Path(output_dir)
    request_id = uuid4().hex
    request_path = folder / f"chfem_input_{request_id}.json"
    result_path = folder / f"chfem_output_{request_id}.json"
    request_path.write_text(json.dumps({"model_path": voxel_path, "Yang": config["young_gpa"], "nu": config["poisson"]}), encoding="utf-8")
    runner = Path(__file__).parent / "scripts" / "chfem_runner.py"
    env_name = os.getenv("CHFEM_CONDA_ENV", "freecad")
    command = (["conda", "run", "-n", env_name, "python"] if env_name else [sys.executable])
    command += [str(runner), "--input", str(request_path), "--output", str(result_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=int(os.getenv("CHFEM_CONDA_TIMEOUT", "3600")), env=env)
    if completed.returncode != 0:
        detail = json.loads(result_path.read_text(encoding="utf-8")).get("error", "") if result_path.is_file() else ""
        raise RuntimeError("Voxel homogenization failed: " + (detail or completed.stderr or completed.stdout)[-2000:])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "success":
        raise RuntimeError("Voxel homogenization failed: " + result.get("error", "No successful status returned"))
    return result["result"]["Ch"]


def engineering_constants(matrix):
    import numpy as np

    from m3s.elastic.elastic_constants import compute_elastic_constants
    array = np.asarray(matrix, dtype=float)
    if array.shape != (6, 6) or not np.isfinite(array).all():
        raise ValueError("Homogenization must return a finite 6x6 stiffness matrix")
    scale = float(np.max(np.abs(array)))
    if np.max(np.abs(array - array.T)) > 1e-5 * scale + 1e-9:
        raise ValueError("The homogenized stiffness matrix is not symmetric")
    if np.any(np.linalg.eigvalsh((array + array.T) / 2) <= 0):
        from m3s.doi_workflow import FIELDS
        return {"is_stable": False, **{field: None for fields in FIELDS.values() for field in fields}}
    result = compute_elastic_constants((array + array.T) / 2, n_points=20)
    # Normalize numpy scalars so state stays JSON-safe and bool matching stays strict.
    return json.loads(json.dumps(result, default=lambda value: value.tolist()))
