"""Hamiltonian construction — fills H/S matrices.

Replaces constructHamiltonian_periodic_toy.m.  Iterates over the neighbor
table, computes hopping and on-site contributions, projects to the active
basis, and accumulates into the System's HoppingMatrix entries.
"""

from collections import defaultdict

import numpy as np

from .types import System
from .basis import get_projector, project_matrix
from .slater_koster import build_hopping_4x4, spin_double
from .onsite import build_onsite_8x8


def fill_hamiltonian(system: System) -> None:
    """Fill all H and S matrices in system.matrices in-place.

    Uses the neighbor table (system.neighbors) to iterate over pairs
    and accumulate hopping contributions into the correct HoppingMatrix.
    On-site contributions go into matrices[0] (R=0 block).
    """
    atoms = system.atoms
    neighbors = system.neighbors

    # Group neighbors by site_i
    nbrs_by_site = defaultdict(list)
    for nb in neighbors:
        nbrs_by_site[nb.site_i].append(nb)

    for i, atom_i in enumerate(atoms):
        proj_i = get_projector(atom_i.basis)
        si = atom_i.orb_slice

        # On-site term (diagonal block in R=0 matrix)
        H_onsite_full = build_onsite_8x8(
            atom_i.u_s, atom_i.u_p,
            atom_i.delta_s, atom_i.delta_p,
            atom_i.theta, atom_i.phi,
            atom_i.spinorbit,
        )
        H_onsite = project_matrix(H_onsite_full, proj_i, proj_i)
        system.matrices[0].H[si, si] += H_onsite

        # Hopping terms from neighbor list
        for nb in nbrs_by_site[i]:
            j = nb.site_j
            atom_j = atoms[j]
            proj_j = get_projector(atom_j.basis)
            sj = atom_j.orb_slice

            # Displacement vector from atom_i to atom_j + R
            d = nb.direction * nb.distance

            # Average hopping parameters
            tss = 0.5 * (atom_i.tss + atom_j.tss)
            tsp = 0.5 * (atom_i.tsp + atom_j.tsp)
            tpp = 0.5 * (atom_i.tpp + atom_j.tpp)
            tsp_rashba = 0.5 * (atom_i.tsp_rashba + atom_j.tsp_rashba)
            tpp_rashba = 0.5 * (atom_i.tpp_rashba + atom_j.tpp_rashba)

            # Build (4,4) orbital hopping, then double to (8,8) spin space
            H_orb = build_hopping_4x4(d, tss, tsp, tpp, 0.0,
                                      tsp_rashba, tpp_rashba)
            H_full = spin_double(H_orb)

            # Project to active basis
            H_contribution = project_matrix(H_full, proj_i, proj_j)

            system.matrices[nb.matrix_idx].H[si, sj] += H_contribution
