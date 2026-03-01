"""On-site Hamiltonian terms: orbital energies, exchange, spin-orbit coupling.

All functions operate in the full 8-dimensional (s,px,py,pz)x(up,down) space.
The result is then projected to the active basis via basis.project_matrix.

Replaces the on-site section of constructHamiltonian_periodic_toy.m.
"""

import numpy as np
from numpy.typing import NDArray

from .types import OnsiteParams


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

    # The full 8x8 SOC matrix from the MATLAB code:
    #   hso = lam * [
    #     0   0   0   0    0   0   0    0
    #     0   0  -i   0    0   0   0    1
    #     0   i   0   0    0   0   0   -i
    #     0   0   0   0    0  -1   i    0
    #     0   0   0   0    0   0   0    0
    #     0   0   0  -1    0   0   i    0
    #     0   0   0  -i    0  -i   0    0
    #     0   1   i   0    0   0   0    0
    #   ]

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
