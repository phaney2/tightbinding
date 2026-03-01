"""Band structure calculation.

Computes energy eigenvalues along a k-space path and returns
data suitable for plotting.
"""

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la

from ..types import System, KPath
from ..bloch import get_H_k


def compute_band_structure(system: System, kpath: KPath) -> dict:
    """Compute band structure along a k-path.

    Parameters
    ----------
    system : fully constructed System
    kpath : k-path specification with high-symmetry points and labels

    Returns
    -------
    dict with keys:
      'k_distances' : (nk,) cumulative distance along path
      'energies' : (nk, nbands) sorted eigenvalues
      'tick_positions' : list of k-distance values at high-symmetry points
      'tick_labels' : corresponding labels
    """
    k_points, k_distances, tick_positions = _interpolate_kpath(
        kpath, system.unitcell_vectors
    )
    nk = len(k_points)
    nbands = system.norbs

    energies = np.zeros((nk, nbands))

    for ik, k in enumerate(k_points):
        energies[ik] = _diag_at_k(system, k)

    return {
        'k_distances': k_distances,
        'energies': energies,
        'tick_positions': tick_positions,
        'tick_labels': kpath.labels,
    }


def _diag_at_k(system: System, k: NDArray) -> NDArray:
    """Diagonalize H(k) at one k-point, return sorted eigenvalues."""
    H, S = get_H_k(system, k)

    # Check if S is identity (common case for TB_simple)
    S_is_identity = np.allclose(S, np.eye(S.shape[0]))

    if S_is_identity:
        eigvals = la.eigh(H, eigvals_only=True)
    else:
        eigvals = la.eigh(H, S, eigvals_only=True)

    return np.sort(eigvals.real)


def _interpolate_kpath(kpath: KPath, lattice_vectors: NDArray
                       ) -> tuple[list[NDArray], NDArray, list[float]]:
    """Generate k-points along the path between high-symmetry points.

    Parameters
    ----------
    kpath : KPath with points in fractional reciprocal coordinates
    lattice_vectors : (3, 3) real-space lattice vectors

    Returns
    -------
    k_points : list of k-vectors in Cartesian reciprocal coordinates
    k_distances : (nk,) cumulative distance along path
    tick_positions : k-distance values at each high-symmetry point
    """
    # Compute reciprocal lattice vectors
    a1, a2, a3 = lattice_vectors
    vol = abs(np.dot(a1, np.cross(a2, a3)))
    b1 = 2 * np.pi * np.cross(a2, a3) / vol
    b2 = 2 * np.pi * np.cross(a3, a1) / vol
    b3 = 2 * np.pi * np.cross(a1, a2) / vol

    # Convert fractional k-points to Cartesian
    pts_cart = []
    for pt_frac in kpath.points:
        pt_frac = np.asarray(pt_frac, dtype=float)
        k_cart = pt_frac[0] * b1 + pt_frac[1] * b2 + pt_frac[2] * b3
        pts_cart.append(k_cart)

    # Interpolate segments
    k_points = []
    k_dists = []
    tick_positions = [0.0]
    cumulative_dist = 0.0

    for seg in range(len(pts_cart) - 1):
        k_start = pts_cart[seg]
        k_end = pts_cart[seg + 1]
        seg_length = np.linalg.norm(k_end - k_start)
        npts = max(kpath.npoints_per_segment, 2)

        for ip in range(npts):
            if seg > 0 and ip == 0:
                continue  # avoid duplicate points at segment boundaries
            frac = ip / (npts - 1)
            k = k_start + frac * (k_end - k_start)
            k_points.append(k)
            if len(k_dists) == 0:
                k_dists.append(0.0)
            else:
                dk = np.linalg.norm(k - k_points[-2])
                k_dists.append(k_dists[-1] + dk)

        cumulative_dist = k_dists[-1]
        tick_positions.append(cumulative_dist)

    return k_points, np.array(k_dists), tick_positions


def plot_bands(result: dict, ax=None, **kwargs):
    """Plot band structure.

    Parameters
    ----------
    result : dict from compute_band_structure
    ax : matplotlib Axes (creates new figure if None)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()

    kd = result['k_distances']
    E = result['energies']

    ax.plot(kd, E, color='black', linewidth=0.8, **kwargs)

    # Add vertical lines and labels at high-symmetry points
    for pos in result['tick_positions']:
        ax.axvline(pos, color='gray', linewidth=0.5, linestyle='--')

    ax.set_xticks(result['tick_positions'])
    ax.set_xticklabels(result['tick_labels'])
    ax.set_ylabel('Energy')
    ax.set_xlim(kd[0], kd[-1])

    return ax
