"""Orbital basis projectors.

Full space is 18-dim (spd×spin):
  [s↑, px↑, py↑, pz↑, dxy↑, dyz↑, dzx↑, dx²−y²↑, dz²↑,
   s↓, px↓, py↓, pz↓, dxy↓, dyz↓, dzx↓, dx²−y²↓, dz²↓]

Each basis type selects a subspace via a projection matrix P (18 x norb).
The active Hamiltonian block is obtained as  P^T @ H_full @ P.

Replaces getBasisVectors.m and get_norbs.m.
"""

import numpy as np
from numpy.typing import NDArray

FULL_DIM = 18

_SQRT3_INV = 1.0 / np.sqrt(3.0)
_SQRT2_INV = 1.0 / np.sqrt(2.0)


def _e(i):
    """Unit vector of length 18 with a 1 at index i."""
    v = [0]*18
    v[i] = 1
    return v


# Index map for reference:
#  0=s↑  1=px↑  2=py↑  3=pz↑  4=dxy↑  5=dyz↑  6=dzx↑  7=dx²−y²↑  8=dz²↑
#  9=s↓ 10=px↓ 11=py↓ 12=pz↓ 13=dxy↓ 14=dyz↓ 15=dzx↓ 16=dx²−y²↓ 17=dz²↓


def _v(*components):
    """Build an 18-element vector from (index, value) pairs."""
    v = [0]*18
    for idx, val in components:
        v[idx] = val
    return v


_BASIS_DEFS: dict[str, list[list[complex]]] = {
    # ---- s-orbital subsets ----
    's_u': [_e(0)],
    's_d': [_e(9)],
    's_ud': [_e(0), _e(9)],

    # ---- p-orbital subsets ----
    'pxpy_u': [_e(1), _e(2)],
    'pxpz_u': [_e(1), _e(3)],
    'p_u': [_e(1), _e(2), _e(3)],
    'pxpy_d': [_e(10), _e(11)],
    'pxpy_ud': [_e(1), _e(2), _e(10), _e(11)],

    # ---- sp subsets ----
    'spxpy_u': [_e(0), _e(1), _e(2)],
    'sp_u': [_e(i) for i in (0, 1, 2, 3)],
    'sp_ud': [_e(i) for i in (0, 1, 2, 3, 9, 10, 11, 12)],

    # ---- j-basis (angular momentum eigenstates) ----
    'sj_u': [
        _e(0),
        _v((1, _SQRT2_INV), (2, -1j*_SQRT2_INV)),
    ],
    'sj_d': [
        _e(9),
        _v((10, _SQRT2_INV), (11, 1j*_SQRT2_INV)),
    ],
    'sj_ud': [
        _e(0),
        _v((3, _SQRT3_INV), (10, _SQRT3_INV), (11, 1j*_SQRT3_INV)),
        _e(9),
        _v((1, _SQRT3_INV), (2, -1j*_SQRT3_INV), (12, -_SQRT3_INV)),
    ],

    # ---- d-orbital subsets ----
    'd_u': [_e(i) for i in (4, 5, 6, 7, 8)],     # 5 orbs: dxy,dyz,dzx,dx2-y2,dz2 spin-up
    'd_z_even_u': [_e(i) for i in (8, 7, 4)],     # 3 orbs: dz2,dx2-y2,dxy spin-up

    # ---- pd subsets ----
    'pd_u': [_e(i) for i in (1, 2, 3, 8, 7, 4)],  # px,py,pz,dz2,dx2-y2,dxy

    # ---- full spd ----
    'spd_u': [_e(i) for i in range(9)],
    'spd_ud': [_e(i) for i in range(18)],
}

# Cache compiled projectors
_PROJECTOR_CACHE: dict[str, NDArray] = {}


def get_projector(basis: str) -> NDArray:
    """Return the (18, norb) projection matrix for the given basis type.

    Projects from the 18-dim full space to the active subspace.
    """
    if basis not in _PROJECTOR_CACHE:
        if basis not in _BASIS_DEFS:
            raise ValueError(f"Unknown basis type: '{basis}'")
        rows = np.array(_BASIS_DEFS[basis], dtype=complex)
        # rows is (norb, 18), transpose gives (18, norb)
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
