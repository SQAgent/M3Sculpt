"""Validate periodic FEM against analytical cells rather than stored model results."""

import numpy as np
import pytest

from m3s.computation import engineering_constants
from m3s.homogenization import compute_stiffness, hexahedron, isotropic_stiffness


def test_homogeneous_cell_recovers_constituent():
    matrix, audit = compute_stiffness(np.ones((8, 8, 8), dtype=np.uint8), 2.0, 0.25)
    constants = engineering_constants(matrix)
    assert constants["is_stable"]
    assert constants["E1"] == pytest.approx(2.0)
    assert constants["G12"] == pytest.approx(0.8)
    assert constants["v12"] == pytest.approx(0.25)
    assert audit["homogeneous_cell"]


def test_element_integrates_uniform_strain_energy():
    stiffness = isotropic_stiffness(1.0, 0.3)
    size = np.array([0.2, 0.3, 0.4])
    ke, fe = hexahedron(size, stiffness)
    assert np.allclose(ke, ke.T)
    translation = np.tile([1.0, 2.0, 3.0], 8)
    assert np.linalg.norm(ke @ translation) < 1e-14
    assert np.allclose(fe.sum(axis=0), 0, atol=1e-14)


def test_periodic_laminate_matches_analytical_in_plane_response():
    voxels = np.zeros((8, 8, 8), dtype=np.uint8)
    voxels[:4] = 1
    matrix, audit = compute_stiffness(voxels)
    # A slab normal to x has plane-stress stiffness in yz and no x load path.
    expected = 0.5 / (1 - 0.3**2)
    assert matrix[1, 1] == pytest.approx(expected, rel=1e-7)
    assert matrix[2, 2] == pytest.approx(expected, rel=1e-7)
    assert matrix[1, 2] == pytest.approx(0.3 * expected, rel=1e-7)
    assert matrix[3, 3] == pytest.approx(0.5 / 2.6, rel=1e-7)
    assert abs(matrix[0, 0]) < 1e-8
    assert max(audit["relative_residuals"]) < 2e-8


def test_closed_cell_is_stable_and_axis_permutation_is_consistent():
    voxels = np.ones((8, 8, 8), dtype=np.uint8)
    voxels[2:6, 2:5, 2:4] = 0
    matrix, _ = compute_stiffness(voxels)
    permuted, _ = compute_stiffness(voxels.transpose(1, 0, 2))
    order = [1, 0, 2, 4, 3, 5]
    assert np.all(np.linalg.eigvalsh(matrix) > 0)
    assert np.allclose(permuted, matrix[np.ix_(order, order)], rtol=1e-6, atol=1e-8)


def test_invalid_voxels_fail():
    with pytest.raises(ValueError):
        compute_stiffness(np.zeros((8, 8, 8)))
    with pytest.raises(ValueError):
        compute_stiffness(np.full((8, 8, 8), np.nan))


def test_three_dimensional_stability_does_not_accept_only_in_plane_stiffness():
    matrix = np.zeros((6, 6))
    matrix[np.ix_([0, 1, 5], [0, 1, 5])] = np.eye(3)
    assert engineering_constants(matrix)["is_stable"] is False


def test_small_solver_reciprocity_noise_is_accepted_but_large_asymmetry_fails():
    matrix = np.eye(6) * 0.04
    matrix[0, 4] = 3e-9
    assert engineering_constants(matrix)["is_stable"]
    matrix[0, 4] = 1e-3
    with pytest.raises(ValueError, match="not symmetric"):
        engineering_constants(matrix)
