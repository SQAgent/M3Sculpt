"""Exercise representative graph branches without changing deterministic image retrieval."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from m3s import computation, geometry
from m3s.engine import NODES, Engine, Session, build_graph, run
from m3s.nodes import doi_property_loop as loop
from m3s.nodes.parameters import material_parameters


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    image = tmp_path / "source.png"
    image.write_bytes(b"source-image-fixture")
    model = tmp_path / "model.stl"
    trimesh.creation.box(extents=(1, 0.5, 0.75)).export(model)
    monkeypatch.setenv("M3S_WORKDIR", str(tmp_path / "runs"))
    monkeypatch.setattr(computation, "process_image", lambda source, output: str(image))
    monkeypatch.setattr(computation, "reconstruct", lambda source, folder: str(model))
    monkeypatch.setattr(
        geometry, "DEFAULT_SWEEP", {"erosion_steps": 1, "dilation_steps": 1}
    )
    calls = []

    def solve(path, folder, config):
        array = np.load(path)
        calls.append((Path(path).name, array.shape))
        return (np.eye(6) * array.mean()).tolist()

    monkeypatch.setattr(computation, "homogenize", solve)

    def constants(matrix):
        value = matrix[0][0]
        return {"is_stable": True, "E1": value, "G12": value, "v12": 0.0}

    monkeypatch.setattr(computation, "engineering_constants", constants)
    state = {
        "messages": [{"role": "user", "content": "Honeycomb structures"}],
        "query": "Honeycomb structures",
        "evaluation_mode": "normal",
        "doi_requirement": {
            "family": "young_modulus",
            "direction": "E1",
            "target": 0.4,
            "tolerance": 1.0,
        },
        "doi_phase": "check",
        "retrieved_doi": "10.example/one",
        "doi_round": 1,
        "doi_processed": ["10.example/one"],
        "doi_ranked": ["10.example/one"],
        "doi_subimages": [str(image), str(image)],
    }
    return state, calls


def test_graph_has_real_langgraph_nodes():
    graph = build_graph()
    assert set(NODES) <= set(graph.nodes)
    assert {"VOXELIZE", "MORPHOLOGY", "EXTRACT_PARAMETERS"} <= set(graph.nodes)


def test_candidates_are_all_evaluated_before_closest_match(pipeline):
    state, calls = pipeline
    trace = []
    run(state, lambda name, values: trace.append(name))
    assert state["doi_phase"] == "matched"
    assert state["compute_config"]["resolution"] == 64
    assert len(calls) == len(state["voxel_candidates"]) == len(state["evaluation_rows"])
    assert len(calls) > 1
    assert all(shape == (64, 64, 64) for _, shape in calls)
    assert trace.count("IMAGE_PROCESS") == 1
    assert trace.count("HOMOGENIZE") == len(calls)
    assert state["doi_history"][0]["nearest_error"] == min(
        abs(row["elastic_constants"]["E1"] - 0.4) for row in state["evaluation_rows"]
    )
    manifest = json.loads(Path(state["candidate_manifest_path"]).read_text())
    assert manifest["boundary"] == "periodic"
    assert any(len(entry["aliases"]) > 1 for entry in manifest["candidates"])
    previous_count = len(state["messages"])
    state["messages"].append({"role": "user", "content": "1"})
    run(state)
    assert state["doi_phase"] == "selected"
    assert len(state["messages"]) == previous_count + 2


def test_miss_uses_next_doi_and_stops_when_candidates_exhausted(pipeline, monkeypatch):
    state, calls = pipeline
    state["doi_requirement"].update(target=2.0, tolerance=0.01)
    state["doi_ranked"].append("10.example/two")

    class Retriever:
        def close(self):
            pass

        def _get_tags_content(self, dois):
            return "Honeycomb"

        def _answer_with_ai(self, query, dois, info, tags):
            return "Relevant paper"

        def _get_subfigures(self, dois):
            return {doi: state["doi_subimages"] for doi in dois}

    monkeypatch.setattr(loop, "handler_for", lambda values: Retriever())
    monkeypatch.setattr(loop, "_rejection_reason", lambda *args: "")
    state["doi_paper_info"] = {"10.example/two": {"title": "Honeycomb structures"}}
    run(state)
    assert state["doi_phase"] == "exhausted"
    assert [entry["doi"] for entry in state["doi_history"]] == [
        "10.example/one",
        "10.example/two",
    ]
    assert all(not entry["matched"] for entry in state["doi_history"])
    assert len(calls) == 2 * len(state["voxel_candidates"])


def test_failure_is_visible_and_retries_the_same_candidate(pipeline, monkeypatch):
    state, calls = pipeline
    original = computation.homogenize
    monkeypatch.setattr(
        computation,
        "homogenize",
        lambda *args: (_ for _ in ()).throw(RuntimeError("solver unavailable")),
    )
    session = Session("test", "normal")
    session.state = state
    session.turn_id = "turn"
    Engine()._execute(session, "turn")
    assert session.status == "error"
    assert "solver unavailable" in session.error
    assert state["pipeline_stage"] == "HOMOGENIZE"
    assert state["candidate_index"] == 0
    assert not state.get("doi_history")
    assert not calls
    monkeypatch.setattr(computation, "homogenize", original)
    run(state)
    assert state["doi_phase"] == "matched"


def test_test_mode_does_not_compute(pipeline, monkeypatch):
    state, _ = pipeline
    state["evaluation_mode"] = "test"

    def load(values):
        return {
            "evaluation_paths": {"existing": values["doi_subimages"][0]},
            "evaluation_rows": [
                {
                    "structure_id": "existing",
                    "elastic_constants": {"is_stable": True, "E1": 0.4},
                }
            ],
            "pipeline_stage": "CHECK_DOI_PROPERTY",
        }

    monkeypatch.setitem(NODES, "LOAD_DATABASE", load)
    trace = []
    run(state, lambda name, values: trace.append(name))
    assert state["doi_phase"] == "matched"
    assert "LOAD_DATABASE" in trace
    assert "IMAGE_PROCESS" not in trace and "HOMOGENIZE" not in trace


def test_octants_remain_separate_and_preserve_selected_voxels():
    array = np.zeros((8, 8, 8), dtype=np.uint8)
    array[1, 2, 3] = 1
    array[7, 6, 5] = 1
    variants = dict(geometry.octant_reflections(array))
    assert len(variants) == 8
    assert variants["000"][1, 2, 3]
    assert variants["111"][7, 6, 5]
    assert not np.array_equal(variants["000"], variants["111"])
    for voxels in variants.values():
        assert all(np.array_equal(voxels, np.flip(voxels, axis)) for axis in range(3))


def test_morphology_is_six_connected_and_periodic():
    array = np.zeros((8, 8, 8), dtype=bool)
    array[0, 0, 0] = True
    expanded = geometry.morphology_step(array, "dilation")
    assert expanded.sum() == 7
    assert expanded[7, 0, 0] and not expanded[1, 1, 0]
    assert not geometry.morphology_step(array, "erosion").any()


def test_constituent_parameters_do_not_pollute_target():
    config, target = material_parameters(
        "constituent Young modulus 2000 MPa,constituent Poisson ratio 0.25;Young modulus 0.4 GPa,tolerance 0.01 GPa"
    )
    assert config["young_gpa"] == 2 and config["poisson"] == 0.25
    from m3s.doi_workflow import parse_requirement

    assert parse_requirement(target)["target"] == 0.4
    with pytest.raises(ValueError):
        material_parameters("constituent Young modulus 1,Young modulus 0.4 GPa")


def test_initial_voxelization_preserves_aspect_ratio(tmp_path):
    model = tmp_path / "box.stl"
    trimesh.creation.box(extents=(1, 0.5, 0.25)).export(model)
    output = geometry.voxelize_model(str(model), tmp_path / "initial.npy")
    array = np.load(output)
    assert array.shape == (64, 64, 64)
    spans = np.ptp(np.argwhere(array), axis=0) + 1
    assert spans[0] > spans[1] > spans[2]
