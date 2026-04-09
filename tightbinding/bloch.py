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


def _prepare_bloch_arrays(system: System) -> None:
    """Pre-stack hopping matrices and displacements for vectorized Bloch sums.

    Caches the stacked arrays on the system object for reuse across k-points.
    """
    if hasattr(system, '_bloch_H_stack'):
        return

    nmat = len(system.matrices)
    dim = system.norbs

    H_stack = np.zeros((nmat, dim, dim), dtype=complex)
    S_stack = np.zeros((nmat, dim, dim), dtype=complex)
    R_stack = np.zeros((nmat, 3))

    for i, mat in enumerate(system.matrices):
        H_stack[i] = mat.H
        S_stack[i] = mat.S
        R_stack[i] = mat.displacement

    system._bloch_H_stack = H_stack
    system._bloch_S_stack = S_stack
    system._bloch_R_stack = R_stack


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
    _prepare_bloch_arrays(system)

    ap = system.atompos
    kx, ky, kz = k

    # Atompos phase (independent of R, computed once)
    phase_ap = np.exp(1j * (kx * ap.x + ky * ap.y + kz * ap.z))

    # Scalar Bloch phases: exp(i k·R) for each matrix
    scalar_phase = np.exp(1j * (system._bloch_R_stack @ k))  # (nmat,)

    # Weighted sums: sum_R X_R * exp(ik·R)
    H_bare = np.einsum('r,rij->ij', scalar_phase, system._bloch_H_stack)
    S_bare = np.einsum('r,rij->ij', scalar_phase, system._bloch_S_stack)

    # Apply atompos phase and Hermitianize
    H = phase_ap * H_bare
    S = phase_ap * S_bare
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
    _prepare_bloch_arrays(system)

    H_stack = system._bloch_H_stack
    S_stack = system._bloch_S_stack
    R_stack = system._bloch_R_stack

    ap = system.atompos
    kx, ky, kz = k
    ap_arr = [ap.x, ap.y, ap.z]

    # Atompos phase (independent of R, computed once)
    phase_ap = np.exp(1j * (kx * ap.x + ky * ap.y + kz * ap.z))

    # Scalar Bloch phases: exp(i k·R) for each matrix
    scalar_phase = np.exp(1j * (R_stack @ k))  # (nmat,)

    # Weighted H matrices: H_R * exp(ik·R), shape (nmat, dim, dim)
    weighted_H = H_stack * scalar_phase[:, None, None]

    # H_bare = sum_R H_R * exp(ik·R)
    H_bare = weighted_H.sum(axis=0)
    S_bare = (S_stack * scalar_phase[:, None, None]).sum(axis=0)

    # Apply atompos phase
    H = phase_ap * H_bare
    S = phase_ap * S_bare

    # Hermitianize
    H = 0.5 * (H + H.conj().T)
    S = 0.5 * (S + S.conj().T)

    # --- Velocity operators ---
    labels_1 = ['x', 'y', 'z']
    v = {}

    # R-weighted sums: sum_R R_alpha * H_R * exp(ik·R)
    H_R_weighted = [None, None, None]
    for alpha in range(3):
        H_R_weighted[alpha] = (R_stack[:, alpha, None, None] * weighted_H).sum(axis=0)

    # First derivatives: v_a = i * (R_a + ap_a) * H_R * phase
    #   = i * phase_ap * (H_R_weighted[a] + ap_a * H_bare)
    for alpha, label in enumerate(labels_1):
        v_raw = 1j * phase_ap * (H_R_weighted[alpha] + ap_arr[alpha] * H_bare)
        v[label] = 0.5 * (v_raw + v_raw.conj().T)

    # Second derivatives: v_ab = -(R_a+ap_a)(R_b+ap_b) * H_R * phase
    if order >= 2:
        # R_a * R_b weighted sums
        for a in range(3):
            for b in range(3):
                label = labels_1[a] + labels_1[b]
                H_RR = (R_stack[:, a, None, None] * R_stack[:, b, None, None]
                        * weighted_H).sum(axis=0)
                v_raw = -phase_ap * (
                    H_RR
                    + ap_arr[b] * H_R_weighted[a]
                    + ap_arr[a] * H_R_weighted[b]
                    + ap_arr[a] * ap_arr[b] * H_bare
                )
                v[label] = 0.5 * (v_raw + v_raw.conj().T)

    return H, S, v
