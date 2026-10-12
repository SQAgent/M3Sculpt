"""Run three local geometry/physics examples through the actual LangGraph pipeline.

Start from deterministic local voxel inputs; do not call literature retrieval,
Qwen image editing or Hunyuan3D. Those services are deliberately outside this
small numerical verification. Save all graph traces, variants and result rows.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from m3s import geometry
from m3s.engine import NODES, build_graph, run
from m3s.homogenization import compute_stiffness


def example_voxels(resolution):
    """Build a solid reference, closed-cell framework and simple cubic lattice."""
    solid = np.ones((resolution,) * 3, dtype=np.uint8)
    closed = solid.copy()
    width = max(2, resolution // 8)
    closed[width:-width, width:-width, width:-width] = 0
    lattice = np.zeros_like(solid)
    middle = resolution // 2
    half = max(1, resolution // 10)
    section = slice(middle - half, middle + half)
    lattice[:, section, section] = 1
    lattice[section, :, section] = 1
    lattice[section, section, :] = 1
    return {
        "solid_reference": solid,
        "closed_cell": closed,
        "simple_cubic_lattice": lattice,
    }


def main():
    """Verify the graph and emit a reproducible, limited-scope numerical report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--solver", choices=("python", "chfem"), default="python")
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Small example sweep, not the full production sweep",
    )
    args = parser.parse_args()
    if (
        not 8 <= args.resolution <= 256
        or args.resolution % 2
        or not 0 <= args.steps <= 32
    ):
        parser.error("resolution must be even in [8,256], steps in [0,32]")
    os.environ["M3S_HOMOGENIZATION_BACKEND"] = args.solver
    geometry.DEFAULT_SWEEP = {"erosion_steps": args.steps, "dilation_steps": args.steps}
    # A missed local example has no next paper; never invoke external retrieval.
    NODES["FIND_BY_TAGS"] = lambda state: {"doi_phase": "exhausted"}
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.mmd").write_text(
        build_graph().get_graph().draw_mermaid(), encoding="utf-8"
    )
    from PIL import Image

    summaries = []
    for name, voxels in example_voxels(args.resolution).items():
        folder = root / name
        folder.mkdir(exist_ok=True)
        initial = folder / "initial.npy"
        np.save(initial, voxels)
        image = folder / "source_slice.png"
        Image.fromarray(voxels[:, :, args.resolution // 2] * 255).save(image)
        family = "shear_modulus" if name == "simple_cubic_lattice" else "young_modulus"
        field = "G12" if family == "shear_modulus" else "E1"
        target = (
            0.04
            if family == "shear_modulus"
            else (1.0 if name == "solid_reference" else 0.42)
        )
        state = {
            "messages": [{"role": "user", "content": name}],
            "query": name,
            "evaluation_mode": "normal",
            "pipeline_stage": "SYMMETRIZE",
            "image_workdir": str(folder),
            "initial_voxel_path": str(initial),
            "current_model_path": "local_voxel_example",
            "processed_image_path": str(image),
            "current_source_key": name,
            "evaluation_paths": {name: str(image)},
            "evaluation_index": 0,
            "evaluation_rows": [],
            "result_origin": "computed",
            "doi_round": 1,
            "doi_phase": "check",
            "retrieved_doi": f"local-example/{name}",
            "doi_ranked": [f"local-example/{name}"],
            "doi_processed": [f"local-example/{name}"],
            "compute_config": {
                "young_gpa": 1.0,
                "poisson": 0.3,
                "resolution": args.resolution,
                "symmetry": "orthorhombic",
                "solver_backend": args.solver,
            },
            "doi_requirement": {
                "family": family,
                "direction": field,
                "target": target,
                "tolerance": 0.01,
            },
        }
        trace = []
        started = time.monotonic()
        run(state, lambda node, values: trace.append(node))
        rows = state["evaluation_rows"]
        assert len(rows) == len(state["voxel_candidates"])
        assert all(
            np.load(row["structure_files"][0]).shape == voxels.shape for row in rows
        )
        stable = [row for row in rows if row["elastic_constants"]["is_stable"]]
        assert stable, f"No stable candidates for {name}"
        best = min(
            stable, key=lambda row: abs(row["elastic_constants"][field] - target)
        )
        if name == "solid_reference":
            constants = best["elastic_constants"]
            assert abs(constants["E1"] - 1.0) < 1e-6
            assert abs(constants["v12"] - 0.3) < 1e-6
            assert abs(constants["G12"] - 1 / 2.6) < 1e-6
        summary = {
            "example": name,
            "resolution": args.resolution,
            "solver": args.solver,
            "sweep_steps": args.steps,
            "seconds": time.monotonic() - started,
            "candidates": len(rows),
            "stable_candidates": len(stable),
            "field": field,
            "target": target,
            "tolerance": 0.01,
            "nearest_value": best["elastic_constants"][field],
            "nearest_error": abs(best["elastic_constants"][field] - target),
            "matched": state["doi_history"][0]["matched"],
            "trace": trace,
            "validation": "passed",
        }
        (folder / "result.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
        summaries.append(summary)
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "trace"}
            ),
            flush=True,
        )
    # Independent CPU/GPU comparison uses an anisotropic cell so axis swaps show up.
    comparison = {"scope": "CPU Python FEM only"}
    if args.solver == "chfem":
        from m3s.computation import homogenize

        cell = np.ones((8, 8, 8), dtype=np.uint8)
        cell[2:6, 2:5, 2:4] = 0
        path = root / "cross_solver.npy"
        np.save(path, cell)
        python_matrix, _ = compute_stiffness(cell)
        gpu_matrix = np.asarray(
            homogenize(str(path), str(root), state["compute_config"])
        )
        error = float(np.max(np.abs(python_matrix - gpu_matrix)))
        assert error < 1e-5, f"CPU/GPU stiffness mismatch: {error} GPa"
        comparison = {
            "scope": "8³ anisotropic cell, Python FEM vs CHFEM",
            "max_abs_error_GPa": error,
        }
    report = {
        "validation": "passed",
        "examples": summaries,
        "cross_solver": comparison,
        "external_services_exercised": [],
        "scope": "Local voxel inputs, real LangGraph, real homogenization and engineering constants; no retrieval, image editing or reconstruction",
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
