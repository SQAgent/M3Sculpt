"""
Coordinate transformation functions for ElasticPOST
Converted from MATLAB to Python
"""

import numpy as np


def ang2direction(theta, phi):
    """
    Convert (theta, phi) angles to direction cosines (x, y, z).

    Parameters
    ----------
    theta : ndarray
        Polar angle (radians).
    phi : ndarray
        Azimuthal angle (radians).

    Returns
    -------
    x, y, z : ndarray
        Direction cosines.

    MATLAB: Ang2Direction.m
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def direction2ang(x, y, z):
    """
    Convert direction cosines (x, y, z) to (theta, phi) angles.

    Parameters
    ----------
    x, y, z : ndarray
        Direction cosines.

    Returns
    -------
    theta, phi : ndarray
        Polar and azimuthal angles (radians).

    MATLAB: Direction2Ang.m
    """
    theta = np.arccos(z / np.sqrt(x**2 + y**2 + z**2))

    denom = np.sqrt(x**2 + y**2)
    # Avoid division by zero
    phi = np.zeros_like(x)
    nonzero = denom > 0
    phi[nonzero] = np.arccos(x[nonzero] / denom[nonzero])
    phi[y < 0] = 2 * np.pi - phi[y < 0]
    phi[~nonzero] = 0.0

    return theta, phi


def d3_to_d2(Cij6):
    """
    Extract 2D compliance from 3D compliance matrix.
    Takes the top-left 3x3 plus C66 from a 6x6 matrix.

    Parameters
    ----------
    Cij6 : ndarray, shape (6, 6)
        3D compliance tensor.

    Returns
    -------
    Cij3 : ndarray, shape (3, 3)
        2D compliance tensor [C11, C12, C16; C12, C22, C26; C16, C26, C66]

    MATLAB: D3toD2.m
    """
    if Cij6.shape[0] == 6:
        Cij3 = Cij6[[0, 1, 5], :][:, [0, 1, 5]]
    else:
        Cij3 = Cij6
    return Cij3


def d2_to_d3(Cij3):
    """
    Convert 2D compliance (3x3) to 3D compliance (6x6) by padding with zeros.

    Parameters
    ----------
    Cij3 : ndarray, shape (3, 3)
        2D compliance tensor.

    Returns
    -------
    Cij6 : ndarray, shape (6, 6)
        3D compliance tensor with zeros for out-of-plane components.

    MATLAB: D2toD3.m
    """
    m = Cij3.shape[0]
    Cij6 = np.zeros((2 * m, 6))
    for i in range(m // 3):
        rows = [(i) * 6 + 0, (i) * 6 + 1, (i) * 6 + 5]
        cols = [(i) * 3 + 0, (i) * 3 + 1, (i) * 3 + 2]
        Cij6[np.ix_(rows, cols)] = Cij3[np.ix_(cols, [0, 1, 2])]
    return Cij6
