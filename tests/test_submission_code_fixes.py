"""Regressions for directional matching, screening failures and retry contracts."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from m3s.app.server import Handler
from m3s.computation import validate_config
from m3s.doi_workflow import closest_row, parse_requirement
from m3s.elastic.core import poisson_3d, poisson_4d, shear_3d, shear_4d
from m3s.elastic.directional import directional_response, global_extrema
from m3s.elastic.elastic_constants import compute_elastic_constants
from m3s.engine import NODES, Engine, SessionCapacityError, run
from m3s.homogenization import isotropic_stiffness
from m3s.nodes import doi_property_loop as loop
from m3s.nodes.evaluation import prepare_evaluation
from m3s.nodes.parameters import extract_parameters, material_parameters


def test_global_extrema_include_axes_and_preserve_rotation_invariance():
    compliance = np.diag(1 / np.array([.8, .3, .1, .025, .025, .025]))
    original = compute_elastic_constants(np.linalg.inv(compliance), n_points=20)
    assert original["Emax"] == pytest.approx(.8, abs=1e-9)
    rotation = Rotation.from_rotvec([.6, -.4, .3]).as_matrix()
    transform = np.empty((6, 6))
    for i, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
        stress = np.zeros((3, 3))
        stress[a, b] = stress[b, a] = 1
        local = rotation.T @ stress @ rotation
        transform[:, i] = local[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    rotated = global_extrema(transform.T @ compliance @ transform, n_points=20)
    for field in ("Emin", "Emax", "Gmin", "Gmax", "vmin", "vmax"):
        assert rotated[field] == pytest.approx(original[field], abs=1e-7)


def test_isotropic_extrema_and_transverse_directions():
    response = compute_elastic_constants(isotropic_stiffness(2, .25), n_points=20)
    for field in ("Emin", "Emax"):
        assert response[field] == pytest.approx(2)
    for field in ("Gmin", "Gmax"):
        assert response[field] == pytest.approx(.8)
    for field in ("vmin", "vmax"):
        assert response[field] == pytest.approx(.25)
        loading = [response[field + "_loading_dir"][axis] for axis in "xyz"]
        transverse = [response[field + "_deformation_dir"][axis] for axis in "xyz"]
        assert np.dot(loading, transverse) == pytest.approx(0, abs=1e-12)


def test_transverse_extrema_agree_with_dense_independent_angle_sweep():
    rng = np.random.default_rng(82)
    matrix = rng.normal(size=(6, 6))
    compliance = matrix.T @ matrix + np.eye(6)
    theta, phi = .83, 1.29
    chi = np.linspace(0, 2 * np.pi, 100001)
    thetas, phis = np.full_like(chi, theta), np.full_like(chi, phi)
    shear = shear_4d(compliance, thetas, phis, chi)
    poisson = poisson_4d(compliance, thetas, phis, chi)
    response = directional_response(compliance, theta, phi)
    assert response["Gmin"] == pytest.approx(shear.min(), abs=1e-8)
    assert response["Gmax"] == pytest.approx(shear.max(), abs=1e-8)
    assert response["vmin"] == pytest.approx(poisson.min(), abs=1e-8)
    assert response["vmax"] == pytest.approx(poisson.max(), abs=1e-8)
    for function, sweep, prefix in ((shear_3d, shear, "G"), (poisson_3d, poisson, "v")):
        low, average, high = function(compliance, np.array([theta]), np.array([phi]))
        assert low[0] == pytest.approx(response[prefix + "min"])
        assert high[0] == pytest.approx(response[prefix + "max"])
        assert average[0] == pytest.approx(sweep[:-1].mean(), abs=1e-8)


@pytest.mark.parametrize("token,field,unit", [
    ("E_z", "E3", "GPa"), ("ν_xz", "v13", ""), ("G_xy", "G12", "GPa"),
    ("G_xz", "G31", "GPa"), ("ν_yx", "v21", ""), ("E_max", "Emax", "GPa"),
    ("ν_{xz}", "v13", ""),
])
def test_cartesian_component_names(token, field, unit):
    assert parse_requirement(f"{token} 0.3 {unit}")["direction"] == field


@pytest.mark.parametrize("token", ["E_q", "E4", "G_zz", "ν_x", "E_xyz", "ν_{θ}", "Eθ"])
def test_unknown_component_is_not_silently_any(token):
    with pytest.raises(ValueError):
        parse_requirement(f"Young modulus {token} 0.3 GPa")


def test_invalid_screening_retries_then_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(loop, "chat_json", lambda *a, **kw: calls.append(kw) or {})
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        loop._rejection_reason({"query": "Honeycomb"}, None, {"d": {"title": "Structural research"}}, "d")
    assert len(calls) == 3
    assert all(call["retries"] == 1 for call in calls)
    replies = iter([{}, {"is_relevant": False, "reason": "Topic mismatch"}])
    monkeypatch.setattr(loop, "chat_json", lambda *a, **kw: next(replies))
    assert loop._rejection_reason({"query": "Honeycomb"}, None, {"d": {}}, "d") == "Topic mismatch"


def test_screening_budget_pauses_before_unscreened_paper_and_resumes(monkeypatch):
    inspected, images = [], []
    ranked = [f"d{i}" for i in range(21)]
    info = {doi: {"title": "Structural research"} for doi in ranked}

    class Retriever:
        def run(self, top_k, screen):
            screen(ranked, info)
            pytest.fail("Do not generate recommendations or retrieve images after the budget is exhausted")

        def close(self):
            pass

        def _get_tags_content(self, dois):
            return "Structure"

        def _answer_with_ai(self, *args):
            return "Relevant paper"

        def _get_subfigures(self, dois):
            images.extend(dois)
            return {dois[0]: []}

    monkeypatch.setattr(loop, "handler_for", lambda state: Retriever())

    def reject(state, handler, paper_info, doi):
        inspected.append(doi)
        return "Topic mismatch" if doi != "d20" else ""

    monkeypatch.setattr(loop, "_rejection_reason", reject)
    state = {"query": "Honeycomb", "messages": [{"role": "user", "content": "Honeycomb"}]}
    run(state)
    assert state["doi_phase"] == "screening_pending"
    assert len(inspected) == 20 and not images and not state.get("doi_processed")
    state["messages"].append({"role": "user", "content": "Continue"})
    run(state)
    assert state["retrieved_doi"] == "d20"
    assert inspected == ranked and images == ["d20"]


@pytest.mark.parametrize("constants", [
    {"is_stable": True, "E1": float("nan")},
    {"is_stable": True, "E1": float("inf")},
    {"is_stable": True, "E1": True}, {"is_stable": True},
    {"is_stable": "true", "E1": .3},
])
def test_corrupt_matching_data_is_an_error(constants):
    rows = [{"structure_id": "s", "elastic_constants": constants}]
    with pytest.raises(ValueError):
        closest_row(rows, {"s": "s.png"}, parse_requirement("E1 0.3 GPa"))
    assert closest_row(rows, {"unrelated": "x.png"}, parse_requirement("E1 0.3 GPa")) is None


def test_missing_matching_image_preserves_stage_for_retry(tmp_path):
    image = tmp_path / "missing.png"
    state = {"query": "Honeycomb", "messages": [], "evaluation_mode": "test",
             "pipeline_stage": "CHECK_DOI_PROPERTY", "doi_round": 1,
             "retrieved_doi": "d", "doi_requirement": parse_requirement("E1 0.3 GPa"),
             "evaluation_paths": {"s": str(image)},
             "evaluation_rows": [{"structure_id": "s", "elastic_constants": {"is_stable": True, "E1": .3}}]}
    with pytest.raises(FileNotFoundError):
        run(state)
    assert state["pipeline_stage"] == "CHECK_DOI_PROPERTY"
    assert not state.get("doi_history") and not state.get("matched_result")
    image.write_bytes(b"image-fixture")
    run(state)
    assert state["doi_phase"] == "matched"


@pytest.mark.parametrize("text", ["constituent Poisson ratio 0.3 GPa", "\u57fa\u4f53\u6cca\u677e\u6bd40.3GPa\u5e76\u67e5\u627e\u8702\u7a9d", "constituent Young modulus 1", "constituent Poisson ratio 0.5", "constituent Young modulus -1 GPa"])
def test_bad_constituent_input_prompts_without_advancing_graph(text):
    state = {"query": "Honeycomb", "messages": [{"role": "user", "content": text}],
             "pipeline_stage": "HOMOGENIZE"}
    run(state)
    assert state["parameters_valid"] is False
    assert state["pipeline_stage"] == "HOMOGENIZE"
    assert not state.get("material_config")


def test_freeze_before_first_result_blocks_changes_and_allows_retry(monkeypatch, tmp_path):
    config = validate_config()
    state = {"evaluation_mode": "normal", "material_config": config,
             "doi_subimages": [str(tmp_path / "s.png")], "retrieved_doi": "d"}
    state.update(prepare_evaluation(state))
    assert state["frozen_compute_config"] == config and not state["evaluation_rows"]
    state.update(query="Honeycomb", messages=[{"role": "user", "content": "constituent Young modulus 2 GPa"}])
    run(state)
    assert not state["parameters_valid"] and state["material_config"] == config
    assert state["compute_config"] == config and state["pipeline_stage"] == "IMAGE_PROCESS"
    state["messages"].append({"role": "user", "content": "Continue"})
    update = extract_parameters(state)
    assert update["parameters_valid"] and update["material_config"] == config
    corrected = material_parameters("constituent Young modulus 2000 MPa;E1 0.3 GPa")[0]
    assert corrected["young_gpa"] == 2


def test_lru_refreshes_access_and_never_evicts_running_sessions():
    engine = Engine(max_sessions=2)
    first, second = engine.create(), engine.create()
    assert engine.get(first.id) is first
    third = engine.create()
    assert engine.get(second.id) is None
    first.status = third.status = "running"
    with pytest.raises(SessionCapacityError):
        engine.create()
    assert engine.get(first.id) is first and engine.get(third.id) is third
    third.status = "success"
    fourth = engine.create()
    assert engine.get(first.id) is first and engine.get(fourth.id) is fourth
    with pytest.raises(KeyError):
        engine.submit(second, "Continue")


def test_running_session_survives_concurrent_creates(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked(state):
        entered.set()
        assert release.wait(5)
        return {"doi_phase": "exhausted"}

    monkeypatch.setitem(NODES, "FIND_BY_TAGS", blocked)
    engine = Engine(max_sessions=2)
    session = engine.create()
    original = engine._execute

    def execute(*args):
        try:
            original(*args)
        finally:
            finished.set()

    monkeypatch.setattr(engine, "_execute", execute)
    assert engine.submit(session, "Honeycomb")
    try:
        assert entered.wait(5)
        assert engine.submit(session, "Continue") is None
        for _ in range(5):
            engine.create()
        assert engine.get(session.id) is session
    finally:
        release.set()
        assert finished.wait(5)
    assert session.status == "success"


def test_http_reports_busy_capacity_and_returns_session_after_completion():
    engine = Engine(max_sessions=1)
    current = engine.create()
    current.status = "running"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.engine = engine
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/session"
    client = build_opener(ProxyHandler({}))

    def request():
        return Request(url, data=b"{}", headers={"Content-Type": "application/json"})

    try:
        with pytest.raises(HTTPError) as error:
            client.open(request(), timeout=3)
        assert error.value.code == 503
        assert "All sessions are running" in json.loads(error.value.read())["error"]
        with current.lock:
            current.status = "success"
        with client.open(request(), timeout=3) as response:
            assert response.status == 200
            assert json.load(response)["session_id"] != current.id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("symmetry", ["cubic", "tetragonal", "unknown"])
def test_unsupported_symmetry_fails_before_computation(symmetry):
    with pytest.raises(ValueError, match="only supports orthorhombic"):
        validate_config({"symmetry": symmetry})
