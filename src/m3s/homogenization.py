"""Python periodic hexahedral FEM corresponding to the paper's SI section S8.

Use engineering shear strains and Voigt order xx, yy, zz, yz, xz, xy. Empty
voxels contribute no stiffness. Periodic node identification removes duplicate
boundary nodes; a reference node removes rigid translation. The energy formula
C_eff = volume_fraction*C_s - F.T*K^-1*F/volume is equivalent to the SI's
elementwise fluctuation-energy expression.
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

CORNERS = np.array(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
)


def isotropic_stiffness(young_gpa, poisson):
    """Return an isotropic constituent stiffness in GPa with engineering shear."""
    if (
        not np.isfinite(young_gpa)
        or young_gpa <= 0
        or not np.isfinite(poisson)
        or not -1 < poisson < 0.5
    ):
        raise ValueError("Invalid constituent parameters")
    mu = young_gpa / (2 * (1 + poisson))
    lam = young_gpa * poisson / ((1 + poisson) * (1 - 2 * poisson))
    matrix = np.diag([2 * mu] * 3 + [mu] * 3)
    matrix[:3, :3] += lam
    return matrix


def hexahedron(cell_size, stiffness):
    """Integrate an eight-node element using the SI's three-point Gauss quadrature."""
    signs = 2 * CORNERS - 1
    points = (-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5))
    weights = (5 / 9, 8 / 9, 5 / 9)
    ke, fe = np.zeros((24, 24)), np.zeros((24, 6))
    for indices in product(range(3), repeat=3):
        local = np.array([points[index] for index in indices])
        grad = np.empty((8, 3))
        for axis in range(3):
            other = [item for item in range(3) if item != axis]
            grad[:, axis] = (
                signs[:, axis]
                * np.prod(1 + signs[:, other] * local[other], axis=1)
                / (4 * cell_size[axis])
            )
        b = np.zeros((6, 24))
        for node, (dx, dy, dz) in enumerate(grad):
            b[:, 3 * node : 3 * node + 3] = [
                [dx, 0, 0],
                [0, dy, 0],
                [0, 0, dz],
                [0, dz, dy],
                [dz, 0, dx],
                [dy, dx, 0],
            ]
        weight = np.prod(cell_size) / 8 * np.prod([weights[index] for index in indices])
        ke += weight * b.T @ stiffness @ b
        fe += weight * b.T @ stiffness
    return ke, fe


def compute_stiffness(
    voxels, young_gpa=1.0, poisson=0.3, tolerance=1e-8, maxiter=10000
):
    """Solve six periodic cell problems and return the effective stiffness and audit."""
    voxels = np.asarray(voxels)
    if (
        voxels.ndim != 3
        or min(voxels.shape) < 2
        or not np.isin(voxels, (0, 1)).all()
        or not voxels.any()
    ):
        raise ValueError("Homogenization requires a nonempty 3D binary voxel array")
    solid = isotropic_stiffness(young_gpa, poisson)
    if voxels.all():
        # For a homogeneous periodic cell F is exactly zero, so C_eff = C_s.
        return solid, {
            "solver": "python",
            "homogeneous_cell": True,
            "iterations": [0] * 6,
            "relative_residuals": [0.0] * 6,
        }
    shape = np.array(voxels.shape)
    cell_size = 1 / shape
    ke, fe = hexahedron(cell_size, solid)
    coords = np.argwhere(voxels)
    periodic_nodes = (coords[:, None, :] + CORNERS[None, :, :]) % shape
    node_ids = np.ravel_multi_index(
        tuple(periodic_nodes.transpose(2, 0, 1)), voxels.shape
    )
    edof = (3 * node_ids[:, :, None] + np.arange(3)).reshape(-1, 24).astype(np.int32)
    # Compact to nodes touching a solid element, so void nodes are never solved.
    active, inverse = np.unique(edof, return_inverse=True)
    edof = inverse.reshape(edof.shape).astype(np.int32)
    ndof = len(active)
    rows = np.repeat(edof, 24, axis=1).ravel()
    columns = np.tile(edof, (1, 24)).ravel()
    data = np.broadcast_to(ke, (len(edof), 24, 24)).ravel()
    k = coo_matrix((data, (rows, columns)), shape=(ndof, ndof)).tocsr()
    del rows, columns, data
    f = np.zeros((ndof, 6))
    for local_dof in range(24):
        np.add.at(f, edof[:, local_dof], fe[local_dof])
    k = k[3:, 3:]
    rhs = f[3:]
    diagonal = k.diagonal()
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
        raise ValueError("The finite element matrix has invalid diagonal entries")
    preconditioner = diags(1 / diagonal)
    solutions = np.zeros_like(rhs)
    iterations, residuals = [], []
    for column in range(6):
        count = [0]

        def step(_solution):
            count[0] += 1

        norm = np.linalg.norm(rhs[:, column])
        if norm == 0:
            iterations.append(0)
            residuals.append(0.0)
            continue
        solution, info = cg(
            k,
            rhs[:, column],
            rtol=tolerance,
            atol=0.0,
            maxiter=maxiter,
            M=preconditioner,
            callback=step,
        )
        residual = float(np.linalg.norm(k @ solution - rhs[:, column]) / norm)
        if info != 0 or not np.isfinite(solution).all() or residual > 2 * tolerance:
            raise RuntimeError(
                f"Python homogenization did not converge: load {column}, info={info}, relative residual {residual:g}"
            )
        solutions[:, column] = solution
        iterations.append(count[0])
        residuals.append(residual)
    effective = float(voxels.mean()) * solid - rhs.T @ solutions
    effective = (effective + effective.T) / 2
    if not np.isfinite(effective).all():
        raise ValueError("Python homogenization returned nonfinite values")
    return effective, {
        "solver": "python",
        "homogeneous_cell": False,
        "iterations": iterations,
        "relative_residuals": residuals,
        "active_dofs": ndof,
        "solid_elements": len(edof),
    }


def homogenize(voxel_path, output_dir, config):
    """Run the SI-compatible solver and save the matrix with convergence evidence."""
    matrix, audit = compute_stiffness(
        np.load(voxel_path, allow_pickle=False), config["young_gpa"], config["poisson"]
    )
    path = Path(output_dir) / f"{Path(voxel_path).stem}_python_fem.json"
    path.write_text(
        json.dumps({"Ch": matrix.tolist(), "config": config, "audit": audit}, indent=2),
        encoding="utf-8",
    )
    return matrix.tolist()
