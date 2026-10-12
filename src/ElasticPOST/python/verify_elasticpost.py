"""Physical regression checks against the existing sampled ElasticPOST port."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from elasticpost import directional, verify

from m3s.elastic.core import poisson_4d, shear_4d, young_3d

print(verify())
rng = np.random.default_rng(2)
A = rng.normal(size=(6, 6))
C = A.T @ A + np.eye(6)
t = rng.uniform(0, np.pi, 20)
p = rng.uniform(-np.pi, np.pi, 20)
_, f = directional(C, t, p)
S = np.linalg.inv(C)
np.testing.assert_allclose(f["E"], young_3d(S, t, p), rtol=1e-12)
chi = np.linspace(-np.pi, np.pi, 20001)
v = np.array([poisson_4d(S, t, p, c) for c in chi])
g = np.array([shear_4d(S, t, p, c) for c in chi])
for key, arr, fn in [
    ("nu_min", v, np.min),
    ("nu_max", v, np.max),
    ("G_min", g, np.min),
    ("G_max", g, np.max),
]:
    error = np.max(abs(f[key] - fn(arr, axis=0)))
    assert error < 1e-6, (key, error)
    print(key, "maximum absolute error:", error)
print("Independent sampled ElasticPOST core agreement passed.")
