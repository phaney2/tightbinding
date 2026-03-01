"""KD-tree based neighbor finding for periodic tight-binding systems.

Replaces the brute-force triple loop over lattice multiples. Uses
scipy.spatial.cKDTree to efficiently find all atom pairs within
hopping range across periodic images.
"""

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .types import NeighborEntry, HoppingMatrix


def find_neighbors(
    coords: NDArray,
    lattice_vectors: NDArray,
    hopping_range: float,
    norbs_total: int,
) -> tuple[list[NeighborEntry], list[HoppingMatrix]]:
    """Find all neighbor pairs within hopping_range using a KD-tree.

    Parameters
    ----------
    coords : (natoms, 3) Cartesian coordinates of unit-cell atoms
    lattice_vectors : (3, 3) lattice vectors, rows are a1, a2, a3
    hopping_range : cutoff distance
    norbs_total : total orbitals (for sizing H/S matrices)

    Returns
    -------
    neighbors : list of NeighborEntry
    matrices : list of HoppingMatrix (index 0 = on-site R=[0,0,0])
    """
    natoms = len(coords)
    a1, a2, a3 = lattice_vectors

    # Estimate required lattice multiples from hopping_range and cell geometry.
    # For each lattice direction, compute the perpendicular height of the cell
    # (distance between parallel faces) using the reciprocal lattice.
    vol = abs(np.dot(a1, np.cross(a2, a3)))
    if vol < 1e-14:
        # 2D or 1D system — use norms as fallback
        heights = np.array([
            np.linalg.norm(a1) if np.linalg.norm(a1) > 1e-14 else 1e10,
            np.linalg.norm(a2) if np.linalg.norm(a2) > 1e-14 else 1e10,
            np.linalg.norm(a3) if np.linalg.norm(a3) > 1e-14 else 1e10,
        ])
    else:
        # Perpendicular heights: h_i = vol / |a_j x a_k|
        cross_vecs = [np.cross(a2, a3), np.cross(a3, a1), np.cross(a1, a2)]
        heights = np.array([vol / np.linalg.norm(c) if np.linalg.norm(c) > 1e-14
                            else 1e10 for c in cross_vecs])

    # Number of multiples needed in each direction (+1 for safety)
    maxn = np.ceil(hopping_range / heights).astype(int) + 1

    # Handle degenerate directions (zero-length lattice vectors)
    for i in range(3):
        if np.linalg.norm(lattice_vectors[i]) < 1e-14:
            maxn[i] = 0

    # Generate all image positions
    n1_range = np.arange(-maxn[0], maxn[0] + 1)
    n2_range = np.arange(-maxn[1], maxn[1] + 1)
    n3_range = np.arange(-maxn[2], maxn[2] + 1)

    # All lattice translation vectors
    n1g, n2g, n3g = np.meshgrid(n1_range, n2_range, n3_range, indexing='ij')
    n_all = np.column_stack([n1g.ravel(), n2g.ravel(), n3g.ravel()])
    R_all = n_all @ lattice_vectors  # (num_translations, 3)

    # Build image positions: for each translation R, shift all atoms
    num_trans = len(R_all)
    # image_coords[t * natoms + j] = coords[j] + R_all[t]
    image_coords = np.repeat(coords, num_trans, axis=0).reshape(
        natoms, num_trans, 3) + R_all[None, :, :]
    image_coords = image_coords.reshape(-1, 3)  # (natoms * num_trans, 3)

    # Build KD-tree from image positions
    tree = cKDTree(image_coords)

    # Query: for each UC atom, find all images within hopping_range
    results = tree.query_ball_point(coords, hopping_range)

    # Map from displacement tuple -> matrix index
    # Index 0 is always the on-site (R=0) matrix
    disp_to_idx: dict[tuple, int] = {}
    matrices: list[HoppingMatrix] = []

    # Pre-register on-site
    matrices.append(HoppingMatrix(
        displacement=np.zeros(3),
        H=np.zeros((norbs_total, norbs_total), dtype=complex),
        S=np.eye(norbs_total, dtype=complex),
    ))
    disp_to_idx[(0, 0, 0)] = 0

    neighbors: list[NeighborEntry] = []

    for i in range(natoms):
        for flat_idx in results[i]:
            # Decode: flat_idx = j * num_trans + t
            j = flat_idx // num_trans
            t = flat_idx % num_trans
            n_tuple = tuple(n_all[t].tolist())

            # Skip self-interaction (same atom, zero displacement)
            if i == j and n_tuple == (0, 0, 0):
                continue

            # Displacement vector from atom i to image of atom j
            d = image_coords[flat_idx] - coords[i]
            dist = np.linalg.norm(d)

            if dist < 1e-12:
                continue

            direction = d / dist

            # Register displacement if new
            if n_tuple not in disp_to_idx:
                R_vec = R_all[t].copy()
                disp_to_idx[n_tuple] = len(matrices)
                matrices.append(HoppingMatrix(
                    displacement=R_vec,
                    H=np.zeros((norbs_total, norbs_total), dtype=complex),
                    S=np.zeros((norbs_total, norbs_total), dtype=complex),
                ))

            matrix_idx = disp_to_idx[n_tuple]

            neighbors.append(NeighborEntry(
                site_i=i,
                site_j=j,
                distance=dist,
                direction=direction,
                matrix_idx=matrix_idx,
            ))

    return neighbors, matrices
