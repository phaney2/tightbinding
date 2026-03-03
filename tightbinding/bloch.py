"""Bloch sums: H(k), S(k), and velocity operators.

Replaces get_H_v.m.  Computes the k-dependent Hamiltonian and its
derivatives by summing hopping matrices weighted by Bloch phases.

H(k) = sum_R  H_R .* exp(i * k . (R + atompos))
v_a(k) = dH/dk_a = i * (R_a + atompos_a) .* H_R .* exp(i*k*(R+atompos))
"""

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

from .types import System


def get_reciprocal_lattice(unitcell_vectors):
    """Compute reciprocal lattice vectors from real-space lattice vectors.

    Parameters
    ----------
    unitcell_vectors : (3, 3) array-like, rows are a1, a2, a3

    Returns
    -------
    b1, b2, b3 : each shape (3,)
    """
    a1, a2, a3 = unitcell_vectors
    vol = abs(np.dot(a1, np.cross(a2, a3)))
    b1 = 2 * np.pi * np.cross(a2, a3) / vol
    b2 = 2 * np.pi * np.cross(a3, a1) / vol
    b3 = 2 * np.pi * np.cross(a1, a2) / vol
    return b1, b2, b3


def diagonalize_hk(H, S, eigenvectors=False):
    """Diagonalize H(k), handling generalized eigenvalue problem if S != I.

    Parameters
    ----------
    H : (n, n) complex array — Hamiltonian (already Hermitianized by get_H_k)
    S : (n, n) complex array — overlap matrix
    eigenvectors : if True, return (eigenvalues, eigenvectors)

    Returns
    -------
    ek : sorted eigenvalues (real)
    psi : eigenvectors (columns), only if eigenvectors=True
    """
    dim = H.shape[0]
    S_is_identity = np.allclose(S, np.eye(dim), atol=1e-10)

    if eigenvectors:
        if S_is_identity:
            ek, psi = np.linalg.eigh(H)
        else:
            ek, psi = la.eigh(H, S)
        idx = np.argsort(ek)
        return ek[idx], psi[:, idx]
    else:
        if S_is_identity:
            ek = la.eigh(H, eigvals_only=True)
        else:
            ek = la.eigh(H, S, eigvals_only=True)
        return np.sort(ek)


def get_H_k(system: System, k: NDArray) -> tuple[NDArray, NDArray]:
    """Compute H(k) and S(k) via Bloch phase sums.

    Parameters
    ----------
    system : System with filled matrices and atompos
    k : k-vector, shape (3,)

    Returns
    -------
    H : Hamiltonian at k, shape (norbs, norbs), Hermitian
    S : Overlap at k, shape (norbs, norbs), Hermitian
    """
    dim = system.norbs
    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    ap = system.atompos
    kx, ky, kz = k

    for mat in system.matrices:
        R = mat.displacement
        # Phase matrix: element-wise phase including intra-cell positions
        phase = np.exp(1j * (
            kx * (R[0] + ap.x) +
            ky * (R[1] + ap.y) +
            kz * (R[2] + ap.z)
        ))
        H += mat.H * phase
        S += mat.S * phase

    # Hermitianize
    H = 0.5 * (H + H.conj().T)
    S = 0.5 * (S + S.conj().T)

    return H, S


def get_H_v(system: System, k: NDArray, order: int = 1
            ) -> tuple[NDArray, NDArray, dict[str, NDArray]]:
    """Compute H(k), S(k), and velocity operators.

    Parameters
    ----------
    system : System with filled matrices and atompos
    k : k-vector, shape (3,)
    order : 1 for first derivatives only, 2 for first + second derivatives

    Returns
    -------
    H : Hamiltonian at k
    S : Overlap at k
    v : dict of velocity operators
        order=1: {'x', 'y', 'z'}
        order=2: also includes {'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'}
    """
    dim = system.norbs
    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    labels_1 = ['x', 'y', 'z']
    v = {l: np.zeros((dim, dim), dtype=complex) for l in labels_1}

    if order >= 2:
        labels_2 = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        for l in labels_2:
            v[l] = np.zeros((dim, dim), dtype=complex)

    ap = system.atompos
    kx, ky, kz = k
    ap_arr = [ap.x, ap.y, ap.z]

    for mat in system.matrices:
        R = mat.displacement

        # Total position: R + atompos  for each Cartesian component
        Rtot = [R[alpha] + ap_arr[alpha] for alpha in range(3)]

        phase = np.exp(1j * (kx * Rtot[0] + ky * Rtot[1] + kz * Rtot[2]))

        Hphase = mat.H * phase
        Sphase = mat.S * phase

        H += Hphase
        S += Sphase

        # First derivatives: v_a = i * R_a * H * phase
        for alpha, label in enumerate(labels_1):
            v[label] += 1j * Rtot[alpha] * Hphase

        # Second derivatives: v_ab = (i)^2 * R_a * R_b * H * phase
        if order >= 2:
            for a in range(3):
                for b in range(3):
                    label = labels_1[a] + labels_1[b]
                    v[label] += (-1) * Rtot[a] * Rtot[b] * Hphase

    # Hermitianize
    H = 0.5 * (H + H.conj().T)
    S = 0.5 * (S + S.conj().T)

    for label in labels_1:
        v[label] = 0.5 * (v[label] + v[label].conj().T)

    if order >= 2:
        for a in range(3):
            for b in range(3):
                lab = labels_1[a] + labels_1[b]
                v[lab] = 0.5 * (v[lab] + v[lab].conj().T)

    return H, S, v
