"""On-site Hamiltonian terms: orbital energies, exchange, spin-orbit coupling.

Supports two full-space dimensions:
  8-dim:  (s, px, py, pz) × (up, down)  — sp models
 18-dim:  (s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2) × (up, down)  — spd models

The result is then projected to the active basis via basis.project_matrix.

Replaces the on-site section of constructHamiltonian_periodic_toy.m.
"""

import numpy as np
from numpy.typing import NDArray

from .types import OnsiteParams


# =========================================================================
# 8-dim (sp×spin) on-site — unchanged from original
# =========================================================================

def build_onsite_8x8_from_params(params: OnsiteParams) -> NDArray:
    """Build the full (8, 8) on-site Hamiltonian from an OnsiteParams object."""
    return build_onsite_8x8(
        params.u_s, params.u_p,
        params.delta_s, params.delta_p,
        params.theta, params.phi,
        params.spinorbit,
    )


def build_onsite_8x8(u_s: float, u_p: float,
                     delta_s: float, delta_p: float,
                     theta: float, phi: float,
                     spinorbit: float) -> NDArray:
    """Build the full (8, 8) on-site Hamiltonian.

    Basis order: [s_up, px_up, py_up, pz_up, s_down, px_down, py_down, pz_down]

    Includes:
    1. Orbital energies (U): diagonal in orbital space, same for both spins
    2. Magnetic exchange (delta): Zeeman-like splitting with direction (theta, phi)
    3. Spin-orbit coupling: L.S in the p-orbital subspace
    """
    H = np.zeros((8, 8), dtype=complex)

    # --- Orbital energies ---
    H = _add_orbital_energies(H, u_s, u_p)

    # --- Magnetic exchange ---
    H = _add_exchange(H, delta_s, delta_p, theta, phi)

    # --- Spin-orbit coupling ---
    H = _add_soc(H, spinorbit)

    return H


def _add_orbital_energies(H: NDArray, u_s: float, u_p: float) -> NDArray:
    """Add diagonal orbital energies.  Same for both spins."""
    onsite = np.array([u_s, u_p, u_p, u_p], dtype=complex)
    H[0:4, 0:4] += np.diag(onsite)
    H[4:8, 4:8] += np.diag(onsite)
    return H


def _add_exchange(H: NDArray, delta_s: float, delta_p: float,
                  theta: float, phi: float) -> NDArray:
    """Add magnetic exchange splitting.

    Exchange Hamiltonian = delta/2 * (sigma_z*cos(theta) + sigma_x*sin(theta)*cos(phi)
                                      + sigma_y*sin(theta)*sin(phi))

    In the (up, down) block structure:
      uu block: +delta/2 * cos(theta)
      dd block: -delta/2 * cos(theta)
      ud block: +delta/2 * sin(theta) * exp(-i*phi)
      du block: +delta/2 * sin(theta) * exp(+i*phi)
    """
    delta_diag = np.array([delta_s, delta_p, delta_p, delta_p], dtype=complex)

    ct = np.cos(theta)
    st = np.sin(theta)
    ep = np.cos(phi) - 1j * np.sin(phi)   # exp(-i*phi)

    # uu block
    H[0:4, 0:4] += np.diag(delta_diag * 0.5 * ct)
    # dd block
    H[4:8, 4:8] += np.diag(-delta_diag * 0.5 * ct)
    # ud block
    H[0:4, 4:8] += np.diag(delta_diag * 0.5 * st * ep)
    # du block
    H[4:8, 0:4] += np.diag(delta_diag * 0.5 * st * ep.conj())

    return H


def _add_soc(H: NDArray, spinorbit: float) -> NDArray:
    """Add spin-orbit coupling L.S in the p-orbital subspace.

    Matches the hso matrix in constructHamiltonian_periodic_toy.m exactly.
    Basis order: [s_up, px_up, py_up, pz_up, s_down, px_down, py_down, pz_down]
    """
    if spinorbit == 0.0:
        return H

    lam = spinorbit

    # uu block (rows 0-3, cols 0-3): L_z terms
    H[1, 2] += lam * (-1j)
    H[2, 1] += lam * (1j)

    # ud block (rows 0-3, cols 4-7): L- terms
    H[1, 7] += lam * 1
    H[2, 7] += lam * (-1j)
    H[3, 5] += lam * (-1)
    H[3, 6] += lam * (1j)

    # du block (rows 4-7, cols 0-3): L+ terms
    H[5, 3] += lam * (-1)
    H[6, 3] += lam * (-1j)
    H[7, 1] += lam * 1
    H[7, 2] += lam * (1j)

    # dd block (rows 4-7, cols 4-7): -L_z terms
    H[5, 6] += lam * (1j)
    H[6, 5] += lam * (-1j)

    return H


# =========================================================================
# 18-dim (spd×spin) on-site
# =========================================================================
#
# Basis order (per spin): s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2
# Full 18-dim: [orb0..orb8 ↑, orb0..orb8 ↓]
#   indices 0-8 = spin-up, indices 9-17 = spin-down

def build_onsite_18x18_from_params(params: OnsiteParams) -> NDArray:
    """Build the full (18, 18) on-site Hamiltonian from an OnsiteParams object."""
    H = np.zeros((18, 18), dtype=complex)

    # --- Orbital energies ---
    orb_e = np.array([params.u_s,
                      params.u_p, params.u_p, params.u_p,
                      params.u_d, params.u_d, params.u_d, params.u_d, params.u_d],
                     dtype=complex)
    H[0:9, 0:9] += np.diag(orb_e)
    H[9:18, 9:18] += np.diag(orb_e)

    # --- Magnetic exchange ---
    delta_diag = np.array([params.delta_s,
                           params.delta_p, params.delta_p, params.delta_p,
                           params.delta_d, params.delta_d, params.delta_d,
                           params.delta_d, params.delta_d],
                          dtype=complex)

    ct = np.cos(params.theta)
    st = np.sin(params.theta)
    ep = np.cos(params.phi) - 1j * np.sin(params.phi)

    H[0:9, 0:9] += np.diag(delta_diag * 0.5 * ct)       # uu
    H[9:18, 9:18] += np.diag(-delta_diag * 0.5 * ct)     # dd
    H[0:9, 9:18] += np.diag(delta_diag * 0.5 * st * ep)  # ud
    H[9:18, 0:9] += np.diag(delta_diag * 0.5 * st * ep.conj())  # du

    # --- Spin-orbit coupling (p orbitals) ---
    if params.spinorbit != 0.0:
        _add_soc_p_18(H, params.spinorbit)

    # --- Spin-orbit coupling (d orbitals) ---
    if params.spinorbit_d != 0.0:
        _add_soc_d_18(H, params.spinorbit_d)

    return H


def _add_soc_p_18(H: NDArray, lam: float) -> NDArray:
    """Add p-orbital L·S SOC to 18×18 matrix.

    p orbitals are at indices 1-3 (up) and 10-12 (down).
    Same matrix elements as the 8×8 version, shifted to 18-dim indices.
    """
    # uu block: L_z in p-subspace
    H[1, 2] += lam * (-1j)   # px↑, py↑
    H[2, 1] += lam * (1j)

    # ud block: L- terms (p↑ → p↓)
    H[1, 12] += lam * 1       # px↑ → pz↓
    H[2, 12] += lam * (-1j)   # py↑ → pz↓
    H[3, 10] += lam * (-1)    # pz↑ → px↓
    H[3, 11] += lam * (1j)    # pz↑ → py↓

    # du block: L+ terms (p↓ → p↑)
    H[10, 3] += lam * (-1)    # px↓ → pz↑
    H[11, 3] += lam * (-1j)   # py↓ → pz↑
    H[12, 1] += lam * 1       # pz↓ → px↑
    H[12, 2] += lam * (1j)    # pz↓ → py↑

    # dd block: -L_z in p-subspace
    H[10, 11] += lam * (1j)
    H[11, 10] += lam * (-1j)

    return H


def _add_soc_d_18(H: NDArray, lam: float) -> NDArray:
    """Add d-orbital L·S SOC to 18×18 matrix.

    d orbitals at indices 4-8 (up) and 13-17 (down).
    Order: dxy(0), dyz(1), dzx(2), dx2-y2(3), dz2(4)  — local indices within d-block.

    L·S = Lz*Sz + (L+*S- + L-*S+)/2

    Matrix elements computed from complex-to-real harmonic transformation
    (Condon-Shortley convention). Eigenvalues: j=5/2 at +lam, j=3/2 at -3lam/2.
    """
    # d-block offsets: up = 4, down = 13
    u = 4   # start of d↑
    d = 13  # start of d↓
    s3 = np.sqrt(3.0)

    # Lz in real d-orbital basis {dxy, dyz, dzx, dx2-y2, dz2}:
    #   <dxy|Lz|dx2-y2> = +2i    (and h.c.)
    #   <dyz|Lz|dzx>    = +i     (and h.c.)
    Lz = np.zeros((5, 5), dtype=complex)
    Lz[0, 3] = 2j;    Lz[3, 0] = -2j
    Lz[1, 2] = 1j;    Lz[2, 1] = -1j

    # L- in real d-orbital basis (from Condon-Shortley transformation):
    Lm = np.zeros((5, 5), dtype=complex)
    Lm[0, 1] = 1;           Lm[1, 0] = -1
    Lm[0, 2] = -1j;         Lm[2, 0] = 1j
    Lm[1, 3] = -1j;         Lm[3, 1] = 1j
    Lm[1, 4] = -1j*s3;      Lm[4, 1] = 1j*s3
    Lm[2, 3] = -1;          Lm[3, 2] = 1
    Lm[2, 4] = s3;          Lm[4, 2] = -s3

    # uu block: lam/2 * Lz  (Sz = +1/2)
    H[u:u+5, u:u+5] += 0.5 * lam * Lz
    # dd block: -lam/2 * Lz  (Sz = -1/2)
    H[d:d+5, d:d+5] += -0.5 * lam * Lz
    # ud block: lam/2 * L-  (S+ flips down->up)
    H[u:u+5, d:d+5] += 0.5 * lam * Lm
    # du block: lam/2 * L+ = (L-)†
    H[d:d+5, u:u+5] += 0.5 * lam * Lm.conj().T

    return H
