"""Python directional E, G and Poisson surfaces corresponding to ElasticPOST.

Engineering Voigt order: xx, yy, zz, yz, xz, xy; stiffness in GPa.
Transverse extrema use a 2x2 eigensystem instead of finite chi sampling.
"""

import numpy as np


def longitudinal(a):
    x, y, z = np.moveaxis(a, -1, 0)
    return np.stack([x * x, y * y, z * z, y * z, x * z, x * y], axis=-1)


def shear_vector(a, b):
    x, y, z = np.moveaxis(a, -1, 0)
    u, v, w = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            x * u,
            y * v,
            z * w,
            (y * w + z * v) / 2,
            (x * w + z * u) / 2,
            (x * v + y * u) / 2,
        ],
        axis=-1,
    )


def directional(C, theta, phi):
    """Return E, nu_min, nu_max, G_min, G_max at each loading direction."""
    C = np.asarray(C, float)
    C = (C + C.T) / 2
    if (
        C.shape != (6, 6)
        or not np.isfinite(C).all()
        or np.linalg.eigvalsh(C).min() <= 0
    ):
        raise ValueError("Stiffness must be finite and positive definite")
    S = np.linalg.inv(C)
    n = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=-1,
    )
    u = np.stack(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
        axis=-1,
    )
    v = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    q = longitudinal(n)

    def contract(a, b):
        return np.einsum("...i,ij,...j->...", a, S, b)

    invE = contract(q, q)
    # q(b) = cos² chi q(u) + sin² chi q(v) + 2 cos chi sin chi shear_vector(u,v).
    p = np.empty(theta.shape + (2, 2))
    p[..., 0, 0] = -contract(q, longitudinal(u)) / invE
    p[..., 1, 1] = -contract(q, longitudinal(v)) / invE
    p[..., 0, 1] = p[..., 1, 0] = -contract(q, shear_vector(u, v)) / invE
    pe = np.linalg.eigvalsh(p)
    r, t = shear_vector(n, u), shear_vector(n, v)
    g = np.empty_like(p)
    g[..., 0, 0] = 4 * contract(r, r)
    g[..., 1, 1] = 4 * contract(t, t)
    g[..., 0, 1] = g[..., 1, 0] = 4 * contract(r, t)
    ge = np.linalg.eigvalsh(g)
    return n, {
        "E": 1 / invE,
        "nu_min": pe[..., 0],
        "nu_max": pe[..., 1],
        "G_min": 1 / ge[..., 1],
        "G_max": 1 / ge[..., 0],
    }


def verify():
    """Check isotropic solution and independent sampled transverse contractions."""
    E, nu = 1.0, 0.3
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    np.fill_diagonal(C[:3, :3], lam + 2 * mu)
    C[3:, 3:] = np.eye(3) * mu
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 19), np.linspace(-np.pi, np.pi, 23))
    _, f = directional(C, theta, phi)
    for key, target in [
        ("E", E),
        ("nu_min", nu),
        ("nu_max", nu),
        ("G_min", mu),
        ("G_max", mu),
    ]:
        np.testing.assert_allclose(f[key], target, atol=1e-12)
    return "Isotropic E, nu and G checks passed."


if __name__ == "__main__":
    print(verify())
