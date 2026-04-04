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
    basis: str                          # e.g. 's_u', 'sp_ud', 'sj_ud', 'spd'
    norb: int                           # number of active orbitals
    orb_slice: slice                    # slice into full Hamiltonian
    species: str = ''                   # atom type label


@dataclass
class OnsiteParams:
    """On-site parameters for TB_simple (per species)."""
    u_s: float = 0.0
    u_p: float = 0.0
    u_d: float = 0.0
    u_pz: float | None = None       # pz on-site; defaults to u_p when None
    u_dz2: float | None = None      # dz² on-site; defaults to u_d when None
    u_dxz: float | None = None      # dxz,dyz on-site (Δ₁); defaults to u_d when None
    delta_s: float = 0.0
    delta_p: float = 0.0
    delta_d: float = 0.0
    theta: float = 0.0
    phi: float = 0.0
    spinorbit: float = 0.0
    spinorbit_d: float = 0.0


@dataclass
class HoppingParams:
    """Hopping parameters for TB_simple (per species)."""
    tss_sigma: float = 0.0
    tsp_sigma: float = 0.0
    tpp_sigma: float = 0.0
    tpp_pi: float = 0.0
    tsd_sigma: float = 0.0
    tpd_sigma: float = 0.0
    tpd_pi: float = 0.0
    tdd_sigma: float = 0.0
    tdd_pi: float = 0.0
    tdd_delta: float = 0.0
    tsp_rashba: float = 0.0
    tpp_rashba: float = 0.0
    tsd_rashba: float = 0.0
    tpd_rashba: float = 0.0
    tdd_rashba: float = 0.0


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
    onsite_params: dict[str, OnsiteParams] | None = None    # keyed by species
    hopping_params: dict[str, HoppingParams] | None = None  # keyed by species
    orbital_position: dict[str, NDArray] | None = None  # a^(W,a) matrices, keyed by 'x','y','z'
    hopping_anisotropy_direction: NDArray | None = None  # unit vector ê for strain axis
    hopping_anisotropy_factor: float = 0.0               # δ in traceless anisotropy


@dataclass
class KPath:
    """k-point path specification for band structure."""
    points: list[NDArray[np.floating]]
    labels: list[str]
    npoints_per_segment: int = 100
