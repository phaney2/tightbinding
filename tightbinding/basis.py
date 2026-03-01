"""Orbital basis projectors.

The full local Hilbert space is 8-dimensional:
  [s_up, px_up, py_up, pz_up, s_down, px_down, py_down, pz_down]

Each basis type selects a subspace via a projection matrix P (8 x norb).
The active Hamiltonian block is obtained as  P^T @ H_full @ P.

Replaces getBasisVectors.m and get_norbs.m.
"""

import numpy as np
from numpy.typing import NDArray

_SQRT3_INV = 1.0 / np.sqrt(3.0)
_SQRT2_INV = 1.0 / np.sqrt(2.0)

# Each entry is a list of 8-element row vectors.  The projection matrix
# is formed by transposing the stack, giving shape (8, norb).
_BASIS_DEFS: dict[str, list[list[complex]]] = {
    's_u': [
        [1, 0, 0, 0, 0, 0, 0, 0],
    ],
    's_d': [
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
    's_ud': [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
    'pxpy_u': [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ],
    'pxpz_u': [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    'p_u': [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    'pxpy_d': [
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    'pxpy_ud': [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    'sj_ud': [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, _SQRT3_INV, 0, _SQRT3_INV, 1j*_SQRT3_INV, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, _SQRT3_INV, -1j*_SQRT3_INV, 0, 0, 0, 0, -_SQRT3_INV],
    ],
    'sj_u': [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, _SQRT2_INV, -1j*_SQRT2_INV, 0, 0, 0, 0, 0],
    ],
    'sj_d': [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, _SQRT2_INV, 1j*_SQRT2_INV, 0],
    ],
    'spxpy_u': [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ],
    'sp_u': [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    'sp_ud': [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
}

# Cache compiled projectors
_PROJECTOR_CACHE: dict[str, NDArray] = {}


def get_projector(basis: str) -> NDArray:
    """Return the (8, norb) projection matrix for the given basis type.

    Projects from the full 8-dim sp-spin space to the active subspace.
    """
    if basis not in _PROJECTOR_CACHE:
        if basis not in _BASIS_DEFS:
            raise ValueError(f"Unknown basis type: '{basis}'")
        rows = np.array(_BASIS_DEFS[basis], dtype=complex)
        # MATLAB: basis = transpose(basis) → columns become basis vectors
        # rows is (norb, 8), transpose gives (8, norb)
        _PROJECTOR_CACHE[basis] = rows.T.copy()
    return _PROJECTOR_CACHE[basis]


def get_norbs(basis: str) -> int:
    """Return the number of active orbitals for a basis type."""
    return get_projector(basis).shape[1]


def project_matrix(H_full: NDArray, proj_i: NDArray, proj_j: NDArray) -> NDArray:
    """Project H from full 8-dim space to active subspaces.

    H_active = proj_i^H @ H_full @ proj_j
    """
    return proj_i.conj().T @ H_full @ proj_j
