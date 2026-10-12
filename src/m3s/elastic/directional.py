"""Compute exact transverse extrema and refine loading directions numerically."""

import numpy as np
from scipy.optimize import minimize


def direction(theta, phi):
    """Return unit loading directions from broadcastable spherical angles."""
    theta, phi = np.broadcast_arrays(theta, phi)
    return np.stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1)


def directional_response(compliance, theta, phi):
    """Evaluate moduli and analytic transverse extrema in engineering Voigt order."""
    theta, phi = np.broadcast_arrays(theta, phi)
    n = direction(theta, phi)
    basis = np.stack((
        np.stack((np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)), axis=-1),
        np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1),
    ), axis=-2)
    x, y, z = np.moveaxis(n, -1, 0)
    normal_stress = np.stack((x*x, y*y, z*z, y*z, x*z, x*y), axis=-1)
    strain = np.einsum('ij,...j->...i', compliance, normal_stress)
    longitudinal = np.einsum('...i,...i->...', normal_stress, strain)
    if np.any(longitudinal <= 0) or not np.isfinite(longitudinal).all():
        raise ValueError("Invalid directional compliance")
    tensor = np.empty(theta.shape + (3, 3))
    tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2] = strain[..., 0], strain[..., 1], strain[..., 2]
    tensor[..., 1, 2] = tensor[..., 2, 1] = strain[..., 3] / 2
    tensor[..., 0, 2] = tensor[..., 2, 0] = strain[..., 4] / 2
    tensor[..., 0, 1] = tensor[..., 1, 0] = strain[..., 5] / 2
    poisson_form = -np.einsum('...ai,...ij,...bj->...ab', basis, tensor, basis) / longitudinal[..., None, None]
    poisson_values, poisson_vectors = np.linalg.eigh(poisson_form)
    tx, ty, tz = np.moveaxis(basis, -1, 0)
    shear_stress = np.stack((2*x[..., None]*tx, 2*y[..., None]*ty, 2*z[..., None]*tz,
                             y[..., None]*tz + z[..., None]*ty,
                             x[..., None]*tz + z[..., None]*tx,
                             x[..., None]*ty + y[..., None]*tx), axis=-1)
    shear_form = np.einsum('...ai,ij,...bj->...ab', shear_stress, compliance, shear_stress)
    shear_values = np.linalg.eigvalsh(shear_form)
    if np.any(shear_values <= 0) or not np.isfinite(shear_values).all():
        raise ValueError("Invalid transverse shear compliance")
    return {"E": 1 / longitudinal, "Gmin": 1 / shear_values[..., 1], "Gmax": 1 / shear_values[..., 0],
            "vmin": poisson_values[..., 0], "vmax": poisson_values[..., 1],
            "vmin_transverse": np.einsum('...a,...ai->...i', poisson_vectors[..., :, 0], basis),
            "vmax_transverse": np.einsum('...a,...ai->...i', poisson_vectors[..., :, 1], basis)}


def _refine(compliance, field, theta_grid, phi_grid, find_max):
    """Refine separated grid candidates while retaining their original values."""
    values = directional_response(compliance, theta_grid, phi_grid)[field].ravel()
    angles = np.column_stack((theta_grid.ravel(), phi_grid.ravel()))
    order = np.argsort(-values if find_max else values)
    best_angles = angles[order[0]].copy()
    sign = -1 if find_max else 1
    best_value = float(values[order[0]])
    scale = max(float(np.max(np.abs(values))), 1e-12)
    if np.ptp(values) <= 1e-12 * max(1.0, scale):
        return best_value, best_angles
    starts, vectors = [], []
    for index in order:
        vector = direction(*angles[index])
        if any(abs(np.dot(vector, previous)) > 0.98 for previous in vectors):
            continue
        starts.append(angles[index])
        vectors.append(vector)
        if len(starts) == 8:
            break

    def objective(angles):
        return sign * float(directional_response(compliance, *angles)[field]) / scale

    for start in starts:
        optimum = minimize(objective, start, method='L-BFGS-B',
                           bounds=((0, np.pi), (None, None)),
                           options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 250})
        if not optimum.success:
            optimum = minimize(objective, start, method='Powell',
                               options={"xtol": 1e-9, "ftol": 1e-12, "maxiter": 250})
        if not optimum.success or not np.isfinite(optimum.fun):
            raise RuntimeError(f"Directional extremum optimization did not converge: {field}, {optimum.message}")
        value = float(directional_response(compliance, *optimum.x)[field])
        if sign * value < sign * best_value:
            best_value, best_angles = value, optimum.x.copy()
    best_angles[1] %= 2 * np.pi
    # Powell is unconstrained; equivalent angles retain the same physical vector.
    n = direction(*best_angles)
    best_angles = np.array([np.arccos(np.clip(n[2], -1, 1)), np.arctan2(n[1], n[0]) % (2*np.pi)])
    return best_value, best_angles


def global_extrema(compliance, n_points=20):
    """Find extrema using an axis-inclusive seed grid and multi-start refinement."""
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 4:
        raise ValueError("The initial direction grid requires an integer of at least 4 points")
    theta = np.unique(np.r_[np.linspace(0, np.pi, n_points), np.pi / 2])
    phi = np.unique(np.r_[np.linspace(0, 2*np.pi, n_points, endpoint=False), np.arange(4) * np.pi / 2])
    tg, pg = np.meshgrid(theta, phi, indexing='ij')
    result = {}
    for name, field, maximize in (("Emin", "E", False), ("Emax", "E", True),
                                  ("Gmin", "Gmin", False), ("Gmax", "Gmax", True),
                                  ("vmin", "vmin", False), ("vmax", "vmax", True)):
        value, angles = _refine(compliance, field, tg, pg, maximize)
        n = direction(*angles)
        loading = {"theta": float(angles[0]), "phi": float(angles[1]), **dict(zip(("x", "y", "z"), map(float, n), strict=True))}
        result[name] = value
        if name.startswith('v'):
            transverse = directional_response(compliance, *angles)[name + '_transverse']
            result[name + '_loading_dir'] = loading
            result[name + '_deformation_dir'] = dict(zip(("x", "y", "z"), map(float, transverse), strict=True))
        else:
            result[name + '_direction'] = loading
    return result
