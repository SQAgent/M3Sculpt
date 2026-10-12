"""
Calculate engineering constants and extrema from elastic stiffness matrices

Calculate principal engineering constants and directional extrema from stiffness matrix C.
"""

import numpy as np

from .core import (
    poisson_2dm,
    shear_2dm,
    stable_of_mechanical,
    young_2dm,
)
from .directional import global_extrema


def compute_elastic_constants(C, n_points=100):
    """
    Calculate engineering constants and extrema from an elastic stiffness matrix.

    Parameters
    ----------
    C : ndarray or list, shape (6, 6)
        Elastic stiffness matrix (GPa).
    n_points : int, optional
        Initial loading-direction grid points (default 100); include principal axes, then refine using multi-start continuous optimization.

    Returns
    -------
    result : dict
        Includes the following keys:
        - 'E1', 'E2', 'E3' : Principal Young modulus (GPa)
        - 'v12', 'v13', 'v23', 'v21', 'v31', 'v32' : Principal Poisson ratio
        - 'G12', 'G23', 'G31' : Principal shear modulus (GPa)
        - 'Emax', 'Emin' : Young modulus extrema (GPa)
        - 'Emax_direction', 'Emin_direction' : Extremum directions {'theta', 'phi', 'x', 'y', 'z'}
        - 'Gmax', 'Gmin' : Shear modulus extrema (GPa)
        - 'Gmax_direction', 'Gmin_direction' : Extremum directions {'theta', 'phi', 'x', 'y', 'z'}
        - 'vmax', 'vmin' : Poisson ratio extrema
        - 'vmax_loading_dir', 'vmin_loading_dir' : Loading direction {'theta', 'phi', 'x', 'y', 'z'}
        - 'vmax_deformation_dir', 'vmin_deformation_dir' : Transverse direction {'x', 'y', 'z'}
        - 'is_stable' : Mechanical stability

    Notes
    -----
    Poisson ratio convention: vij is the transverse strain ratio in direction i under loading in direction j.
    Loading direction is defined by (theta, phi); chi rotates the transverse direction around the loading axis.
    """
    C = np.asarray(C, dtype=np.float64)

    result = {}

    if C.shape != (6, 6) or not np.isfinite(C).all():
        raise ValueError("A finite 6x6 3D stiffness matrix is required")
    if not np.allclose(C, C.T, rtol=1e-5, atol=1e-9):
        raise ValueError("The stiffness matrix must be symmetric")
    C = (C + C.T) / 2
    is_stable = bool(np.all(np.linalg.eigvalsh(C) > 0))
    result['is_stable'] = is_stable

    if not is_stable:
        for key in ['E1', 'E2', 'E3', 'v12', 'v13', 'v23', 'v21', 'v31', 'v32',
                    'G12', 'G23', 'G31', 'Emax', 'Emin', 'Gmax', 'Gmin', 'vmax', 'vmin']:
            result[key] = None
        for key in ['Emax_direction', 'Emin_direction', 'Gmax_direction', 'Gmin_direction',
                    'vmax_loading_dir', 'vmin_loading_dir', 'vmax_deformation_dir', 'vmin_deformation_dir']:
            result[key] = None
        return result

    S = np.linalg.inv(C)

    result['E1'] = 1.0 / S[0, 0]
    result['E2'] = 1.0 / S[1, 1]
    result['E3'] = 1.0 / S[2, 2]

    result['v12'] = -S[0, 1] / S[1, 1]
    result['v13'] = -S[0, 2] / S[2, 2]
    result['v23'] = -S[1, 2] / S[2, 2]
    result['v21'] = -S[0, 1] / S[0, 0]
    result['v31'] = -S[0, 2] / S[0, 0]
    result['v32'] = -S[1, 2] / S[1, 1]

    result['G12'] = 1.0 / S[5, 5]
    result['G23'] = 1.0 / S[3, 3]
    result['G31'] = 1.0 / S[4, 4]

    result.update(global_extrema(S, n_points))
    return result


def compute_elastic_constants_2d(C, n_points=200):
    """
    Calculate engineering constants and extrema from a 2D 3x3 stiffness matrix.

    Parameters
    ----------
    C : ndarray, shape (3, 3)
        2D stiffness matrix [C11, C12, C16; C12, C22, C26; C16, C26, C66] (GPa).
    n_points : int
        Number of direction samples, default 200.

    Returns
    -------
    result : dict
        - 'E1', 'E2' : Principal Young modulus
        - 'v12', 'v21' : Principal Poisson ratio
        - 'G12' : Principal shear modulus
        - 'Emax', 'Emin' : Young modulus extrema and directions (phi, x, y)
        - 'Gmax', 'Gmin' : Shear modulus extrema and directions (phi, x, y)
        - 'vmax', 'vmin' : Poisson ratio extrema and directions (phi, x, y)
        - 'is_stable' : Mechanical stability
    """
    C = np.asarray(C, dtype=np.float64)
    result = {}

    is_stable = stable_of_mechanical(C)
    result['is_stable'] = is_stable

    if not is_stable:
        for key in ['E1', 'E2', 'v12', 'v21', 'G12',
                     'Emax', 'Emin', 'Gmax', 'Gmin', 'vmax', 'vmin',
                     'Emax_direction', 'Emin_direction',
                     'Gmax_direction', 'Gmin_direction',
                     'vmax_direction', 'vmin_direction']:
            result[key] = None
        return result

    S = np.linalg.inv(C)

    result['E1'] = 1.0 / S[0, 0]
    result['E2'] = 1.0 / S[1, 1]
    result['G12'] = 1.0 / S[2, 2]
    result['v12'] = -S[0, 1] / S[1, 1]
    result['v21'] = -S[0, 1] / S[0, 0]

    phi = np.linspace(0, 2 * np.pi, n_points)

    E = young_2dm(S, phi)
    imax = np.argmax(E)
    imin = np.argmin(E)
    result['Emax'] = float(E[imax])
    result['Emin'] = float(E[imin])
    result['Emax_direction'] = {
        'phi': float(phi[imax]), 'x': float(np.cos(phi[imax])), 'y': float(np.sin(phi[imax])),
    }
    result['Emin_direction'] = {
        'phi': float(phi[imin]), 'x': float(np.cos(phi[imin])), 'y': float(np.sin(phi[imin])),
    }

    G = shear_2dm(S, phi)
    imax = np.argmax(G)
    imin = np.argmin(G)
    result['Gmax'] = float(G[imax])
    result['Gmin'] = float(G[imin])
    result['Gmax_direction'] = {
        'phi': float(phi[imax]), 'x': float(np.cos(phi[imax])), 'y': float(np.sin(phi[imax])),
    }
    result['Gmin_direction'] = {
        'phi': float(phi[imin]), 'x': float(np.cos(phi[imin])), 'y': float(np.sin(phi[imin])),
    }

    v = poisson_2dm(S, phi)
    imax = np.argmax(v)
    imin = np.argmin(v)
    result['vmax'] = float(v[imax])
    result['vmin'] = float(v[imin])
    result['vmax_direction'] = {
        'phi': float(phi[imax]), 'x': float(np.cos(phi[imax])), 'y': float(np.sin(phi[imax])),
    }
    result['vmin_direction'] = {
        'phi': float(phi[imin]), 'x': float(np.cos(phi[imin])), 'y': float(np.sin(phi[imin])),
    }

    return result


def print_elastic_constants(result):
    """
    Print elastic constant results.

    Parameters
    ----------
    result : dict
        Result returned by compute_elastic_constants.
    """
    if result['is_stable'] is False:
        print("The material is mechanically unstable!")
        return

    print("=" * 60)
    print("Principal engineering constants")
    print("-" * 60)
    print(f"E1 = {result['E1']:.4f} GPa")
    print(f"E2 = {result['E2']:.4f} GPa")
    print(f"E3 = {result['E3']:.4f} GPa")
    print()
    print(f"v12 = {result['v12']:.4f}   v21 = {result['v21']:.4f}")
    print(f"v13 = {result['v13']:.4f}   v31 = {result['v31']:.4f}")
    print(f"v23 = {result['v23']:.4f}   v32 = {result['v32']:.4f}")
    print()
    print(f"G12 = {result['G12']:.4f} GPa")
    print(f"G23 = {result['G23']:.4f} GPa")
    print(f"G31 = {result['G31']:.4f} GPa")

    print("=" * 60)
    print("Directional extrema")
    print("-" * 60)

    print("\nYoung modulus:")
    print(f"  Emax = {result['Emax']:.4f} GPa")
    d = result['Emax_direction']
    print(f"    Direction: theta={d['theta']:.4f} rad, phi={d['phi']:.4f} rad")
    print(f"          (x, y, z) = ({d['x']:.4f}, {d['y']:.4f}, {d['z']:.4f})")
    print(f"  Emin = {result['Emin']:.4f} GPa")
    d = result['Emin_direction']
    print(f"    Direction: theta={d['theta']:.4f} rad, phi={d['phi']:.4f} rad")
    print(f"          (x, y, z) = ({d['x']:.4f}, {d['y']:.4f}, {d['z']:.4f})")

    print("\nShear modulus:")
    print(f"  Gmax = {result['Gmax']:.4f} GPa")
    d = result['Gmax_direction']
    print(f"    Direction: theta={d['theta']:.4f} rad, phi={d['phi']:.4f} rad")
    print(f"          (x, y, z) = ({d['x']:.4f}, {d['y']:.4f}, {d['z']:.4f})")
    print(f"  Gmin = {result['Gmin']:.4f} GPa")
    d = result['Gmin_direction']
    print(f"    Direction: theta={d['theta']:.4f} rad, phi={d['phi']:.4f} rad")
    print(f"          (x, y, z) = ({d['x']:.4f}, {d['y']:.4f}, {d['z']:.4f})")

    print("\nPoisson ratio:")
    print(f"  vmax = {result['vmax']:.4f}")
    ld = result['vmax_loading_dir']
    dd = result['vmax_deformation_dir']
    print(f"    Loading direction: theta={ld['theta']:.4f} rad, phi={ld['phi']:.4f} rad")
    print(f"              (x, y, z) = ({ld['x']:.4f}, {ld['y']:.4f}, {ld['z']:.4f})")
    print(f"    Transverse direction: (x, y, z) = ({dd['x']:.4f}, {dd['y']:.4f}, {dd['z']:.4f})")

    print(f"  vmin = {result['vmin']:.4f}")
    ld = result['vmin_loading_dir']
    dd = result['vmin_deformation_dir']
    print(f"    Loading direction: theta={ld['theta']:.4f} rad, phi={ld['phi']:.4f} rad")
    print(f"              (x, y, z) = ({ld['x']:.4f}, {ld['y']:.4f}, {ld['z']:.4f})")
    print(f"    Transverse direction: (x, y, z) = ({dd['x']:.4f}, {dd['y']:.4f}, {dd['z']:.4f})")
    print("=" * 60)
