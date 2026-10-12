"""
Core computation functions for ElasticPOST
Converted from MATLAB to Python
"""

import numpy as np


def young_3d(S, theta, phi):
    """
    Calculate 3D Young's modulus directional distribution.

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor (inverse of stiffness tensor C).
    theta : ndarray
        Polar angle array (radians).
    phi : ndarray
        Azimuthal angle array (radians).

    Returns
    -------
    E : ndarray
        Young's modulus in GPa, same shape as theta/phi.

    MATLAB: Young_3D.m
    """
    ax = np.sin(theta) * np.cos(phi)
    ay = np.sin(theta) * np.sin(phi)
    az = np.cos(theta)

    E = np.zeros_like(ax, dtype=float)
    for i in range(6):
        if i == 0:
            Etmpi = ax * ax
        elif i == 1:
            Etmpi = ay * ay
        elif i == 2:
            Etmpi = az * az
        elif i == 3:
            Etmpi = ay * az
        elif i == 4:
            Etmpi = ax * az
        elif i == 5:
            Etmpi = ax * ay

        for j in range(6):
            if j == 0:
                Etmpij = S[i, j] * Etmpi * ax * ax
            elif j == 1:
                Etmpij = S[i, j] * Etmpi * ay * ay
            elif j == 2:
                Etmpij = S[i, j] * Etmpi * az * az
            elif j == 3:
                Etmpij = S[i, j] * Etmpi * ay * az
            elif j == 4:
                Etmpij = S[i, j] * Etmpi * ax * az
            elif j == 5:
                Etmpij = S[i, j] * Etmpi * ax * ay
            E = E + Etmpij

    return 1.0 / E


def young_2dm(S, phi):
    """
    Calculate 2D material Young's modulus directional distribution.

    Parameters
    ----------
    S : ndarray, shape (3, 3)
        2D compliance tensor.
    phi : ndarray
        Angle array (radians) in the 2D plane.

    Returns
    -------
    E : ndarray
        Young's modulus in GPa.

    MATLAB: Young_2DM.m
    """
    x = np.cos(phi)
    y = np.sin(phi)
    S11 = S[0, 0]
    S12 = S[0, 1]
    S13 = S[0, 2]
    S22 = S[1, 1]
    S23 = S[1, 2]
    S33 = S[2, 2]

    E = 1.0 / (S11 * x**4 + S22 * y**4 + (S33 + 2 * S12) * x**2 * y**2
               + 2 * (S13 * x**3 * y + S23 * x * y**3))
    return E


def bulk_3d(S, theta, phi):
    """
    Calculate 3D Bulk modulus directional distribution.

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray
        Polar angle array.
    phi : ndarray
        Azimuthal angle array.

    Returns
    -------
    B : ndarray
        Bulk modulus in GPa.

    MATLAB: Bulk_3D.m
    """
    ax = np.sin(theta) * np.cos(phi)
    ay = np.sin(theta) * np.sin(phi)
    az = np.cos(theta)

    B = np.zeros_like(ax, dtype=float)
    for i in range(6):
        for j in range(3):
            if i == 0:
                Btmp = S[i, j] * ax * ax
            elif i == 1:
                Btmp = S[i, j] * ay * ay
            elif i == 2:
                Btmp = S[i, j] * az * az
            elif i == 3:
                Btmp = S[i, j] * ay * az
            elif i == 4:
                Btmp = S[i, j] * ax * az
            elif i == 5:
                Btmp = S[i, j] * ax * ay
            B = B + Btmp

    return 1.0 / (3.0 * B)


def shear_4d(S, theta, phi, chi):
    """
    Calculate 3D Shear modulus at direction (theta, phi, chi).

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray
        Polar angle array.
    phi : ndarray
        Azimuthal angle array.
    chi : float or ndarray
        Rotation angle about the direction.

    Returns
    -------
    G : ndarray
        Shear modulus in GPa.

    MATLAB: Shear_4D.m
    """
    ax = np.sin(theta) * np.cos(phi)
    ay = np.sin(theta) * np.sin(phi)
    az = np.cos(theta)
    bx = np.cos(theta) * np.cos(phi) * np.cos(chi) - np.sin(phi) * np.sin(chi)
    by = np.cos(theta) * np.sin(phi) * np.cos(chi) + np.cos(phi) * np.sin(chi)
    bz = -np.sin(theta) * np.cos(chi)

    G = 0.0
    for i in range(6):
        if i == 0:
            Gtmpi = ax * bx
        elif i == 1:
            Gtmpi = ay * by
        elif i == 2:
            Gtmpi = az * bz
        elif i == 3:
            Gtmpi = 0.5 * (ay * bz + az * by)
        elif i == 4:
            Gtmpi = 0.5 * (ax * bz + az * bx)
        elif i == 5:
            Gtmpi = 0.5 * (ax * by + ay * bx)

        for j in range(6):
            if j == 0:
                Gtmpij = S[i, j] * Gtmpi * ax * bx
            elif j == 1:
                Gtmpij = S[i, j] * Gtmpi * ay * by
            elif j == 2:
                Gtmpij = S[i, j] * Gtmpi * az * bz
            elif j == 3:
                Gtmpij = 0.5 * S[i, j] * Gtmpi * (ay * bz + az * by)
            elif j == 4:
                Gtmpij = 0.5 * S[i, j] * Gtmpi * (ax * bz + az * bx)
            elif j == 5:
                Gtmpij = 0.5 * S[i, j] * Gtmpi * (ax * by + ay * bx)
            G = G + Gtmpij

    return 1.0 / G / 4.0


def shear_3d(S, theta, phi):
    """
    Calculate 3D Shear modulus min/average/max distribution.
    
    For each loading direction, diagonalize the transverse 2x2 form to obtain
    exact extrema and the continuous angular average.

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray, shape (m, n) or (n,)
        Polar angle array or meshgrid.
    phi : ndarray, shape (m, n) or (n,)
        Azimuthal angle array or meshgrid.

    Returns
    -------
    Gmin : ndarray
        Minimum shear modulus (easiest shear direction).
    Gave : ndarray
        Average shear modulus (average over all chi angles).
    Gmax : ndarray
        Maximum shear modulus (hardest shear direction).

    MATLAB: Shear_3D.m
    
    Physical meaning:
    - Gmin: For given tensile direction n(theta,phi), the minimum shear modulus
            among all possible transverse directions (controlled by chi).
    - Gmax: The maximum shear modulus among all transverse directions.
    - Gave: The average over all transverse directions (chi from -pi to pi).
    """
    from m3s.elastic.directional import directional_response

    response = directional_response(S, theta, phi)
    low, high = response["Gmin"], response["Gmax"]
    return low, np.sqrt(low * high), high


def shear_2dm(S, phi):
    """
    Calculate 2D material Shear modulus directional distribution.

    Parameters
    ----------
    S : ndarray, shape (3, 3)
        2D compliance tensor.
    phi : ndarray
        Angle array.

    Returns
    -------
    G : ndarray
        Shear modulus in GPa.

    MATLAB: Shear_2DM.m
    """
    x = np.cos(phi)
    y = np.sin(phi)
    S11 = S[0, 0]
    S12 = S[0, 1]
    S13 = S[0, 2]
    S22 = S[1, 1]
    S23 = S[1, 2]
    S33 = S[2, 2]

    # This calculation still requires verification
    # G0 = (S11 + S12 - 2 * S22 + S33) / 8.0
    # rG = 0.25 * (0.25 * (S33 + 2 * S12 - S11 - S22)**2
    #              + (S23 - S13)**2)**0.5
    # phiG = np.arctan2(2 * (S13 - S23), S33 + 2 * S12 - S11 - S22)
    # G = 0.25 / (G0 + rG * np.cos(4 * phi + phiG))

    term1 = (S11 + S22 - 2 * S12) * x**2 * y**2
    term2 = S33 * (x**2 - y**2)**2 / 4.0
    term3 = (S23 - S13) * x * y * (x**2 - y**2)
    G = 1.0 / (4.0 * (term1 + term2 + term3))

    return G


def poisson_4d(S, theta, phi, chi):
    """
    Calculate 3D Poisson's ratio at direction (theta, phi, chi).

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray
        Polar angle array.
    phi : ndarray
        Azimuthal angle array.
    chi : float or ndarray
        Rotation angle (controls transverse direction for lateral strain).

    Returns
    -------
    v : ndarray
        Poisson's ratio.

    MATLAB: Poisson_4D.m
    """
    ax = np.sin(theta) * np.cos(phi)
    ay = np.sin(theta) * np.sin(phi)
    az = np.cos(theta)
    bx = np.cos(theta) * np.cos(phi) * np.cos(chi) - np.sin(phi) * np.sin(chi)
    by = np.cos(theta) * np.sin(phi) * np.cos(chi) + np.cos(phi) * np.sin(chi)
    bz = -np.sin(theta) * np.cos(chi)

    v_f = 0.0
    v_d = 0.0
    for i in range(6):
        if i == 0:
            vtmpi = ax * ax
        elif i == 1:
            vtmpi = ay * ay
        elif i == 2:
            vtmpi = az * az
        elif i == 3:
            vtmpi = ay * az
        elif i == 4:
            vtmpi = ax * az
        elif i == 5:
            vtmpi = ax * ay

        for j in range(6):
            if j == 0:
                vtmpij_f = S[i, j] * vtmpi * bx * bx
                vtmpij_d = S[i, j] * vtmpi * ax * ax
            elif j == 1:
                vtmpij_f = S[i, j] * vtmpi * by * by
                vtmpij_d = S[i, j] * vtmpi * ay * ay
            elif j == 2:
                vtmpij_f = S[i, j] * vtmpi * bz * bz
                vtmpij_d = S[i, j] * vtmpi * az * az
            elif j == 3:
                vtmpij_f = S[i, j] * vtmpi * by * bz
                vtmpij_d = S[i, j] * vtmpi * ay * az
            elif j == 4:
                vtmpij_f = S[i, j] * vtmpi * bx * bz
                vtmpij_d = S[i, j] * vtmpi * ax * az
            elif j == 5:
                vtmpij_f = S[i, j] * vtmpi * bx * by
                vtmpij_d = S[i, j] * vtmpi * ax * ay
            v_f = v_f + vtmpij_f
            v_d = v_d + vtmpij_d

    return -v_f / v_d


def poisson_3d(S, theta, phi):
    """
    Calculate 3D Poisson's ratio min/average/max distribution.
    
    For each loading direction, diagonalize the transverse 2x2 form to obtain
    exact extrema and the continuous angular average.

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray, shape (m, n) or (n,)
        Polar angle array or meshgrid.
    phi : ndarray, shape (m, n) or (n,)
        Azimuthal angle array or meshgrid.

    Returns
    -------
    vmin : ndarray
        Minimum Poisson's ratio (smallest lateral contraction).
    vave : ndarray
        Average Poisson's ratio (average over all chi angles).
    vmax : ndarray
        Maximum Poisson's ratio (largest lateral contraction).

    MATLAB: Poisson_3D.m
    
    Physical meaning:
    - vmin: For given tensile direction n(theta,phi), the minimum Poisson's ratio
            among all possible transverse directions (controlled by chi).
    - vmax: The maximum Poisson's ratio among all transverse directions.
    - vave: The average over all transverse directions (chi from -pi to pi).
    """
    from m3s.elastic.directional import directional_response

    response = directional_response(S, theta, phi)
    low, high = response["vmin"], response["vmax"]
    return low, (low + high) / 2, high


def poisson_2dm(S, phi):
    """
    Calculate 2D material Poisson's ratio directional distribution.

    Parameters
    ----------
    S : ndarray, shape (3, 3)
        2D compliance tensor.
    phi : ndarray
        Angle array.

    Returns
    -------
    v : ndarray
        Poisson's ratio.

    MATLAB: Poisson_2DM.m
    """
    x = np.cos(phi)
    y = np.sin(phi)
    E = young_2dm(S, phi)
    S11 = S[0, 0]
    S12 = S[0, 1]
    S13 = S[0, 2]
    S22 = S[1, 1]
    S23 = S[1, 2]
    S33 = S[2, 2]

    # v0 = ((S11 + S22 - S33) / 2.0 + 3 * S12) / 4.0
    # rv = np.sqrt((S23 - S13)**2 + (S12 - (S11 + S22 - S33) / 2.0)**2) / 4.0
    # phiv = np.arctan2(S23 - S13, S12 - (S11 + S22 - S33) / 2.0)
    # v = -E * (v0 + rv * np.cos(4 * phi + phiv))

    # Expand the calculation
    term1 = (S11 + S22 - S33) * x**2 * y**2
    term2 = S12 * (x**4 + y**4)
    term3 = (S13 - S23) * (x * y**3 - y * x**3)
    v = E * (term1 + term2 + term3)
    return v


def hardness(E, B, flag=None):
    """
    Calculate material hardness from Young's modulus and Bulk modulus.

    Parameters
    ----------
    E : ndarray or float
        Young's modulus (GPa).
    B : ndarray or float
        Bulk modulus (GPa).
    flag : str, optional
        'c' for Chen 2011 model, 't' for Tian 2012 model.
        None for empirical E-B model.

    Returns
    -------
    H : ndarray or float
        Hardness in GPa.

    MATLAB: Hardness.m
    References:
        Tian 2012: http://doi.org/10.1016/j.ijrmhm.2012.02.021
    """
    if flag is not None:
        flag = flag.lower()
        G = 1.0 / (1.0 / E - 1.0 / (9.0 * B)) / 3.0
        k = G / B
        if flag == 'c':
            # Chen 2011
            H = 2.0 * (G * k**2)**0.585 - 3.0
        elif flag == 't':
            # Tian 2012
            H = 0.92 * k**1.137 * G**0.708
        else:
            raise ValueError("flag must be 'c', 't', or None")
    else:
        # Empirical E-B model
        H = 0.130548175274347 * E**2.2484942942017 * B**(-1.51675853808829)
    return H


def hardness_3d(S, theta, phi):
    """
    Calculate 3D hardness directional distribution.

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray
        Polar angle array.
    phi : ndarray
        Azimuthal angle array.

    Returns
    -------
    H : ndarray
        Hardness in GPa.

    MATLAB: Hardness_3D.m
    """
    B = bulk_3d(S, theta, phi)
    E = young_3d(S, theta, phi)
    return hardness(E, B)


def hardness_4d(S, theta, phi, chi):
    """
    Calculate 4D hardness distribution (with chi).

    Parameters
    ----------
    S : ndarray, shape (6, 6)
        Compliance tensor.
    theta : ndarray
        Polar angle array.
    phi : ndarray
        Azimuthal angle array.
    chi : float
        Rotation angle.

    Returns
    -------
    H : ndarray
        Hardness in GPa.

    MATLAB: Hardness_4D.m
    """
    B = bulk_3d(S, theta, phi)
    G = shear_4d(S, theta, phi, chi)
    return hardness(G, B)


def elastic_vrh3d(C):
    """
    Calculate Voigt-Reuss-Hill averages for polycrystalline elastic properties.

    Parameters
    ----------
    C : ndarray, shape (6, 6)
        Elastic stiffness tensor (GPa).

    Returns
    -------
    output : ndarray, shape (8, 6)
        Rows: B, G, E, H, Hc, Ht, v, B/G
        Columns: [V, R, H, Anisotropy, AU, AL]
        Returns zeros if mechanically unstable.

    MATLAB: ElasticVRH3D.m
    """
    flag = stable_of_mechanical(C)
    output = np.zeros((8, 6))
    if not flag:
        return output

    S = np.linalg.inv(C)

    Ca = (C[0, 0] + C[1, 1] + C[2, 2]) / 3.0
    Cb = (C[0, 1] + C[1, 2] + C[0, 2]) / 3.0
    Cc = (C[3, 3] + C[4, 4] + C[5, 5]) / 3.0
    Sa = (S[0, 0] + S[1, 1] + S[2, 2]) / 3.0
    Sb = (S[0, 1] + S[1, 2] + S[0, 2]) / 3.0
    Sc = (S[3, 3] + S[4, 4] + S[5, 5]) / 3.0

    # Voigt bounds
    BV = (Ca + 2 * Cb) / 3.0
    GV = (Ca - Cb + 3 * Cc) / 5.0

    # Reuss bounds
    BR = 1.0 / (3.0 * Sa + 6.0 * Sb)
    GR = 5.0 / (4.0 * Sa - 4.0 * Sb + 3.0 * Sc)

    # Hill averages
    BH = (BV + BR) / 2.0
    GH = (GV + GR) / 2.0

    # Young's modulus and Poisson's ratio from G and B
    EV = _young_from_GB(GV, BV)
    ER = _young_from_GB(GR, BR)
    EH = _young_from_GB(GH, BH)
    vV = _poisson_from_GB(GV, BV)
    vR = _poisson_from_GB(GR, BR)
    vH = _poisson_from_GB(GH, BH)

    # B/G ratio
    B2GV = BV / GV
    B2GR = BR / GR
    B2GH = BH / GH

    # Anisotropy percentages
    BA = abs((BV - BR) / (BV + BR))
    GA = abs((GV - GR) / (GV + GR))
    EA = abs((EV - ER) / (EV + ER))
    vA = abs((vV - vR) / (vV + vR))
    B2GA = abs((B2GV - B2GR) / (B2GV + B2GR))

    # Hardness (empirical model)
    HV = hardness(EV, BV)
    HR = hardness(ER, BR)
    HH = hardness(EH, BH)
    HA = (HV - HR) / (HV + HR)

    # Hardness Chen model
    HCV = hardness(EV, BV, 'c')
    HCR = hardness(ER, BR, 'c')
    HCH = hardness(EH, BH, 'c')
    HCA = (HCV - HCR) / (HCV + HCR)

    # Hardness Tian model
    HTV = hardness(EV, BV, 't')
    HTR = hardness(ER, BR, 't')
    HTH = hardness(EH, BH, 't')
    HTA = (HTV - HTR) / (HTV + HTR)

    # Universal anisotropy index
    AU = _anisotropy_U(BV, BR, GV, GR)
    # Log-Euclidean anisotropy index
    AL = _anisotropy_L(BV, BR, GV, GR)

    output = np.array([
        [BV, BR, BH, BA, AU, AL],
        [GV, GR, GH, GA, AU, AL],
        [EV, ER, EH, EA, AU, AL],
        [HV, HR, HH, HA, AU, AL],
        [HCV, HCR, HCH, HCA, AU, AL],
        [HTV, HTR, HTH, HTA, AU, AL],
        [vV, vR, vH, vA, AU, AL],
        [B2GV, B2GR, B2GH, B2GA, AU, AL],
    ])
    return output


def _young_from_GB(G, B):
    """E = 9BG / (3B + G)"""
    return 1.0 / (1.0 / (3.0 * G) + 1.0 / (9.0 * B))


def _poisson_from_GB(G, B):
    """v = 0.5 * (1 - 3G / (3B + G))"""
    return 0.5 * (1.0 - 3.0 * G / (3.0 * B + G))


def _anisotropy_U(BV, BR, GV, GR):
    """Universal anisotropy index A^U = 5*GV/GR + BV/BR - 6"""
    return 5.0 * GV / GR + BV / BR - 6.0


def _anisotropy_L(BV, BR, GV, GR):
    """Log-Euclidean anisotropy index A^L = sqrt(ln(BV/BR)^2 + 5*ln(GV/GR)^2)"""
    return np.sqrt((np.log(BV / BR))**2 + 5.0 * (np.log(GV / GR))**2)


def stable_of_mechanical(C):
    """
    Check mechanical stability of a structure.

    A structure is mechanically stable if all eigenvalues of the
    elastic stiffness matrix are positive.

    Parameters
    ----------
    C : ndarray, shape (6, 6) or (3, 3)
        Elastic stiffness tensor.

    Returns
    -------
    flag : bool
        True if mechanically stable, False otherwise.

    MATLAB: StableofMechanical.m
    """
    m, n = C.shape
    if m != n:
        print('This is not a effective CIJ, the row and column of CIJ must be EQUAL.')
        return False
    elif n == 6:
        C_tmp = C[2:5, 2:5]
        if np.allclose(C_tmp, 0):
            C = C[[0, 1, 5], :][:, [0, 1, 5]]
    elif n != 3:
        print('This is not a effective CIJ, The row and column must be 3 or 6')
        return False

    stable = np.linalg.eigvalsh(C)
    if np.all(stable > 0):
        return True
    else:
        return False
