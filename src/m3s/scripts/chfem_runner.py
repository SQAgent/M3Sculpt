"""Run CHFEM homogenization in an isolated conda environment."""

import argparse
import json
import os
import sys
import traceback

import numpy as np


def _load_array(model_path: str) -> np.ndarray:
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".npy":
        array = np.load(model_path)
    else:
        from m3s.client.utils import load_Structure2Voxel

        array = load_Structure2Voxel(model_path, pitch=None, resolution=None)
    array = np.asarray(array)
    if (
        array.ndim != 3
        or not np.isfinite(array).all()
        or not np.isin(array, (0, 1, 255)).all()
        or not array.any()
    ):
        raise ValueError("CHFEM requires a nonempty finite 3D binary voxel array")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.max(initial=0) <= 1:
        array = array * 255
    # Keep the workflow's xyz convention. Swapping x/y here swaps E1/E2 and
    # G23/G31; an anisotropic CPU-FEM cross-check verifies the returned ordering.
    return array


def main() -> int:
    """Run CHFEM from a JSON parameter file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        params = json.load(f)

    try:
        import chfem

        array = _load_array(params["model_path"])
        if np.all(array == 255):
            # This CHFEM build rejects a domain with no void phase even for
            # elasticity. A homogeneous cell has the exact constituent tensor.
            young, poisson = (
                float(params.get("Yang", 1.0)),
                float(params.get("nu", 0.3)),
            )
            mu = young / (2 * (1 + poisson))
            lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
            ch = np.diag([2 * mu] * 3 + [mu] * 3)
            ch[:3, :3] += lam
        else:
            ch = chfem.compute_property(
                property="elasticity",
                array=array,
                mat_props={
                    0: (1e-9, 1e-9),
                    255: (float(params.get("Yang", 1.0)), float(params.get("nu", 0.3))),
                },
                voxel_size=1,
                solver="cg",
                direction="all",
                solver_tolerance=1e-8,
                solver_maxiter=10000,
                precondition=True,
                type_of_rhs=0,
                refinement=1,
                output_fields=None,
                nf_filepath=None,
            )
        if isinstance(ch, np.ndarray):
            ch = ch.tolist()
        result = {
            "result": {"Ch": ch},
            "status": "success",
            "homogeneous_cell": bool(np.all(array == 255)),
        }
    except Exception as exc:
        result = {
            "result": {},
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
