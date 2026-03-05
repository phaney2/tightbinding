"""Orbital basis projectors.

Two full-space dimensions are supported:

 8-dim (sp×spin): [s↑, px↑, py↑, pz↑, s↓, px↓, py↓, pz↓]
18-dim (spd×spin): [s↑, px↑, py↑, pz↑, dxy↑, dyz↑, dzx↑, dx²−y²↑, dz²↑,
                     s↓, px↓, py↓, pz↓, dxy↓, dyz↓, dzx↓, dx²−y²↓, dz²↓]

Each basis type selects a subspace via a projection matrix P (full_dim x norb).
The active Hamiltonian block is obtained as  P^T @ H_full @ P.

Replaces getBasisVectors.m and get_norbs.m.
"""

import numpy as np
from numpy.typing import NDArray

_SQRT3_INV = 1.0 / np.sqrt(3.0)
_SQRT2_INV = 1.0 / np.sqrt(2.0)

# ---- 8-dim (sp×spin) basis definitions ----
# Each entry is a list of 8-element row vectors.  The projection matrix
# is formed by transposing the stack, giving shape (8, norb).
_BASIS_DEFS_8: dict[str, list[list[complex]]] = {
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

# ---- 18-dim (spd×spin) basis definitions ----
# Order: [s↑, px↑, py↑, pz↑, dxy↑, dyz↑, dzx↑, dx²−y²↑, dz²↑,
#          s↓, px↓, py↓, pz↓, dxy↓, dyz↓, dzx↓, dx²−y²↓, dz²↓]
_Z18 = [0]*18

def _e18(i):
    """Unit vector of length 18 with a 1 at index i."""
    v = [0]*18
    v[i] = 1
    return v

_BASIS_DEFS_18: dict[str, list[list[complex]]] = {
    'spd_u': [_e18(i) for i in range(9)],       # 9 orbs: s,p,d spin-up
    'spd_ud': [_e18(i) for i in range(18)],      # 18 orbs: full spd×spin
}

# Combined lookup
_BASIS_DEFS: dict[str, list[list[complex]]] = {**_BASIS_DEFS_8, **_BASIS_DEFS_18}

# Set of basis names that use the 18-dim full space
_SPD_BASES = set(_BASIS_DEFS_18.keys())

# Cache compiled projectors
_PROJECTOR_CACHE: dict[str, NDArray] = {}


def get_full_dim(basis: str) -> int:
    """Return the full-space dimension for a basis type (8 or 18)."""
    if basis in _SPD_BASES:
        return 18
    if basis in _BASIS_DEFS_8:
        return 8
    raise ValueError(f"Unknown basis type: '{basis}'")


def is_spd_basis(basis: str) -> bool:
    """True if this basis uses the 18-dim spd×spin full space."""
    return basis in _SPD_BASES


def get_projector(basis: str) -> NDArray:
    """Return the (full_dim, norb) projection matrix for the given basis type.

    Projects from the full space (8-dim or 18-dim) to the active subspace.
    """
    if basis not in _PROJECTOR_CACHE:
        if basis not in _BASIS_DEFS:
            raise ValueError(f"Unknown basis type: '{basis}'")
        rows = np.array(_BASIS_DEFS[basis], dtype=complex)
        # rows is (norb, full_dim), transpose gives (full_dim, norb)
        _PROJECTOR_CACHE[basis] = rows.T.copy()
    return _PROJECTOR_CACHE[basis]


def get_norbs(basis: str) -> int:
    """Return the number of active orbitals for a basis type."""
    return get_projector(basis).shape[1]


def project_matrix(H_full: NDArray, proj_i: NDArray, proj_j: NDArray) -> NDArray:
    """Project H from full space to active subspaces.

    H_active = proj_i^H @ H_full @ proj_j
    """
    return proj_i.conj().T @ H_full @ proj_j
