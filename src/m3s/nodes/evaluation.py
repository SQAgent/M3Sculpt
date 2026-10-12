"""Interchangeable live/database acquisition followed by the same target check."""
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from m3s import computation, geometry
from m3s.doi_workflow import closest_row, source_paths
from m3s.messages import ai
from m3s.progress import report

STAGES = {"IMAGE_PROCESS", "RECONSTRUCT", "VOXELIZE", "SYMMETRIZE", "MORPHOLOGY", "HOMOGENIZE", "ENGINEERING_CONSTANTS", "LOAD_DATABASE", "CHECK_DOI_PROPERTY"}


def prepare_evaluation(state):
    mode = computation.mode_for(state)
    paths = list(dict.fromkeys(state.get("doi_subimages", [])))
    if mode == "test":
        return {"evaluation_rows": [], "evaluation_paths": source_paths(state["retrieved_doi"], paths),
                "pipeline_stage": "LOAD_DATABASE", "result_origin": "database"}
    config = computation.validate_config(state.get("material_config"))
    if state.get("frozen_compute_config") and config != state["frozen_compute_config"]:
        raise ValueError("Constituent parameters cannot change after computation preparation starts; start a new conversation.")
    sources = {f"live_{sha256(path.encode()).hexdigest()}": path for path in paths}
    return {"evaluation_rows": [], "evaluation_paths": sources, "evaluation_index": 0,
            "compute_config": config, "frozen_compute_config": dict(config), "result_origin": "computed",
            "pipeline_stage": "IMAGE_PROCESS" if paths else "CHECK_DOI_PROPERTY"}


def load_database(state):
    if computation.mode_for(state) != "test":
        raise RuntimeError("Normal mode must not substitute stored database properties for computation")
    import re

    from pymongo import MongoClient
    paths = state["evaluation_paths"]
    rows = []
    if paths:
        report("Load existing computation records", f"{len(paths)} structure images")
        uri = os.environ.get("MONGO_URL")
        if not uri:
            raise RuntimeError("Missing MONGO_URL")
        with MongoClient(uri, serverSelectionTimeoutMS=15000) as client:
            database = client[os.getenv("MONGO_PERFORMANCE_DATABASE", "MechanicalMetamaterialProjectDatabase")]
            # No target-based query: database rows replace computed engineering constants.
            query = {"design_space_id": os.getenv("MONGO_PERFORMANCE_DESIGN_SPACE", "DS3d_6"),
                     "$or": [{"structure_id": {"$regex": f"^{re.escape(key)}(?:_|$)"}} for key in paths]}
            rows = list(database["StructureProperties"].find(query, {"_id": 0, "structure_id": 1, "elastic_constants": 1, "properties.CH": 1}))
    return {"evaluation_rows": rows, "pipeline_stage": "CHECK_DOI_PROPERTY",
            "messages": [ai(f"Test mode: loaded {len(rows)} existing computation records for this DOI; skipped image processing and computation.") ]}


def _assert_normal(state):
    if computation.mode_for(state) != "normal":
        raise RuntimeError("Test mode must not run the live computation pipeline")


def process_image(state):
    _assert_normal(state)
    source_key, source = list(state["evaluation_paths"].items())[state["evaluation_index"]]
    report("Image processing", f"Image {state['evaluation_index'] + 1}/{len(state['evaluation_paths'])}")
    path = source if Path(source).is_file() else source.replace("/home/shangqing/sqdata/", "/database/")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Cannot read an image for the current DOI: {path}")
    root = Path(os.getenv("M3S_WORKDIR", str(Path(__file__).resolve().parents[1] / "log" / "doi_runs")))
    folder = root / uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    output = computation.process_image(path, str(folder / "processed.png"))
    return {"image_workdir": str(folder), "current_source_key": source_key, "processed_image_path": output,
            "pipeline_stage": "RECONSTRUCT", "messages": [ai(f"Image {state['evaluation_index']+1}: image processing complete; starting 3D reconstruction.") ]}


def reconstruct(state):
    _assert_normal(state)
    report("3D reconstruction", f"Image {state['evaluation_index'] + 1}/{len(state['evaluation_paths'])}")
    output = computation.reconstruct(state["processed_image_path"], str(Path(state["image_workdir"]) / "reconstruction"))
    return {"current_model_path": output, "pipeline_stage": "VOXELIZE",
            "messages": [ai("3D reconstruction complete; starting initial voxelization.") ]}


def voxelize(state):
    """Retain the unmodified reconstruction as a candidate in the paper workflow."""
    _assert_normal(state)
    output = geometry.voxelize_model(state["current_model_path"],
                                    Path(state["image_workdir"]) / "initial.npy",
                                    state["compute_config"]["resolution"])
    return {"initial_voxel_path": output, "pipeline_stage": "SYMMETRIZE"}


def symmetrize(state):
    """Generate up to eight candidates, one from each independently reflected octant."""
    import numpy as np

    _assert_normal(state)
    array = np.load(state["initial_voxel_path"], allow_pickle=False)
    candidates = []
    for octant, voxels in geometry.octant_reflections(array):
        path = Path(state["image_workdir"]) / f"sym{octant}.npy"
        np.save(path, voxels.astype(np.uint8))
        candidates.append({"octant": octant, "voxel_path": str(path)})
    return {"symmetry_candidates": candidates, "pipeline_stage": "MORPHOLOGY",
            "messages": [ai("Reflected all eight octants independently; generating erosion and dilation candidates.") ]}


def morphology(state):
    """Build the expanded set once, then evaluate each unique structure in order."""
    _assert_normal(state)
    candidates, manifest = geometry.expand_candidates(
        state["initial_voxel_path"], state["symmetry_candidates"],
        Path(state["image_workdir"]) / "candidates")
    return {"voxel_candidates": candidates, "candidate_index": 0,
            "candidate_manifest_path": manifest, "current_voxel_path": candidates[0]["voxel_path"],
            "pipeline_stage": "HOMOGENIZE",
            "messages": [ai(f"Generated {len(candidates)} unique nonempty candidates; homogenize each and select the nearest stable structure.") ]}


def homogenize(state):
    _assert_normal(state)
    report("Voxel homogenization", f"Resolution {state['compute_config'].get('resolution', '')}")
    report("Voxel homogenization", f"Candidate {state['candidate_index'] + 1}/{len(state['voxel_candidates'])}")
    matrix = computation.homogenize(state["current_voxel_path"], state["image_workdir"], state["compute_config"])
    return {"current_stiffness": matrix, "pipeline_stage": "ENGINEERING_CONSTANTS",
            "messages": [ai("Voxel homogenization complete; calculating engineering constants.") ]}


def engineering_constants(state):
    _assert_normal(state)
    constants = computation.engineering_constants(state["current_stiffness"])
    candidate = state["voxel_candidates"][state["candidate_index"]]
    row = {"structure_id": state["current_source_key"] + "_" + candidate["label"], "elastic_constants": constants,
           "properties": {"CH": state["current_stiffness"]}, "structure_files": [state["current_voxel_path"]],
           "compute_config": state["compute_config"], "processed_image_path": state["processed_image_path"],
           "model_path": state["current_model_path"], "geometry": candidate,
           "candidate_manifest_path": state["candidate_manifest_path"]}
    candidate_index = state["candidate_index"] + 1
    if candidate_index < len(state["voxel_candidates"]):
        return {"evaluation_rows": [*state["evaluation_rows"], row], "candidate_index": candidate_index,
                "current_voxel_path": state["voxel_candidates"][candidate_index]["voxel_path"],
                "pipeline_stage": "HOMOGENIZE"}
    index = state["evaluation_index"] + 1
    return {"evaluation_rows": [*state["evaluation_rows"], row], "evaluation_index": index,
            "pipeline_stage": "IMAGE_PROCESS" if index < len(state["evaluation_paths"]) else "CHECK_DOI_PROPERTY",
            "messages": [ai(f"Engineering constants calculated for all {candidate_index} candidates of image {index}.") ]}


def nearest_result(state):
    return closest_row(state.get("evaluation_rows", []), state.get("evaluation_paths", {}), state["doi_requirement"])


def route_evaluation(state):
    stage = state.get("pipeline_stage")
    if stage not in STAGES:
        raise ValueError(f"Unknown computation stage: {stage}")
    return stage
