"""Core data structures for the tight-binding code."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class NeighborEntry:
    """One neighbor pair in the unit cell."""
    site_i: int           # UC atom index
    site_j: int           # UC atom index of neighbor
    distance: float       # |r_i - (r_j + R)|
    direction: NDArray    # unit vector from i to j+R, shape (3,)
    matrix_idx: int       # index into system.matrices


@dataclass
class Atom:
    """One atom in the unit cell."""
    index: int
    coord: NDArray[np.floating]         # Cartesian coordinates, shape (3,)
    basis: str                          # e.g. 's_u', 'sp_ud', 'sj_ud'
    norb: int                           # number of active orbitals
    orb_slice: slice                    # slice into full Hamiltonian
    pair: int = 0                       # which HoppingMatrix this atom belongs to
    species: str = ''                   # atom type label

    # Hopping parameters (TB_simple)
    tss: float = 0.0
    tpp: float = 0.0
    tsp: float = 0.0
    tsp_rashba: float = 0.0
    tpp_rashba: float = 0.0
    hopping_range: float = 0.0

    # On-site parameters
    u_s: float = 0.0
    u_p: float = 0.0
    delta_s: float = 0.0
    delta_p: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    spinorbit: float = 0.0


@dataclass
class HoppingMatrix:
    """One H/S block for a specific lattice displacement R."""
    displacement: NDArray[np.floating]  # lattice vector R, shape (3,)
    H: NDArray[np.complexfloating]      # Hamiltonian block (norbs, norbs)
    S: NDArray[np.complexfloating]      # Overlap block (norbs, norbs)


@dataclass
class AtomPos:
    """Displacement matrices between orbital pairs.

    atompos.x[i,j] = -(r_i - r_j)_x  for orbital pair (i,j).
    Used in Bloch sum to include intra-cell position phases.
    """
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    z: NDArray[np.floating]


@dataclass
class System:
    """Complete tight-binding system — the central object passed to all calc engines."""
    atoms: list[Atom]
    matrices: list[HoppingMatrix]
    unitcell_vectors: NDArray[np.floating]   # shape (3, 3), rows are a1, a2, a3
    norbs: int
    atompos: AtomPos | None = None
    neighbors: list[NeighborEntry] | None = None


@dataclass
class KPath:
    """k-point path specification for band structure."""
    points: list[NDArray[np.floating]]
    labels: list[str]
    npoints_per_segment: int = 100
