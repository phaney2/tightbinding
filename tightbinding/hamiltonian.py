"""Hamiltonian construction — fills H/S matrices.

Replaces constructHamiltonian_periodic_toy.m.  Iterates over the neighbor
table, computes hopping and on-site contributions, projects to the active
basis, and accumulates into the System's HoppingMatrix entries.
"""

from collections import defaultdict

import numpy as np

from .types import System, OnsiteParams, HoppingParams
from .basis import get_projector, project_matrix
from .slater_koster import build_hopping_4x4, spin_double
from .onsite import build_onsite_8x8_from_params


def _get_params(params_dict, species):
    """Look up params by species, falling back to '_default'."""
    if species in params_dict:
        return params_dict[species]
    return params_dict['_default']


def fill_hamiltonian(system: System) -> None:
    """Fill all H and S matrices in system.matrices in-place.

    Uses the neighbor table (system.neighbors) to iterate over pairs
    and accumulate hopping contributions into the correct HoppingMatrix.
    On-site contributions go into matrices[0] (R=0 block).
    """
    atoms = system.atoms
    neighbors = system.neighbors
    onsite_p = system.onsite_params
    hopping_p = system.hopping_params

    # Group neighbors by site_i
    nbrs_by_site = defaultdict(list)
    for nb in neighbors:
        nbrs_by_site[nb.site_i].append(nb)

    for i, atom_i in enumerate(atoms):
        proj_i = get_projector(atom_i.basis)
        si = atom_i.orb_slice

        # On-site term (diagonal block in R=0 matrix)
        osp = _get_params(onsite_p, atom_i.species)
        H_onsite_full = build_onsite_8x8_from_params(osp)
        H_onsite = project_matrix(H_onsite_full, proj_i, proj_i)
        system.matrices[0].H[si, si] += H_onsite

        # Hopping terms from neighbor list
        hp_i = _get_params(hopping_p, atom_i.species)

        for nb in nbrs_by_site[i]:
            j = nb.site_j
            atom_j = atoms[j]
            proj_j = get_projector(atom_j.basis)
            sj = atom_j.orb_slice

            # Direction vector for SK integrals: atom_i minus atom_j
            # (MATLAB convention: d points FROM neighbor TOWARD home atom)
            d = -nb.direction * nb.distance

            # Average hopping parameters between species
            hp_j = _get_params(hopping_p, atom_j.species)
            tss = 0.5 * (hp_i.tss + hp_j.tss)
            tsp = 0.5 * (hp_i.tsp + hp_j.tsp)
            tpp = 0.5 * (hp_i.tpp + hp_j.tpp)
            tsp_rashba = 0.5 * (hp_i.tsp_rashba + hp_j.tsp_rashba)
            tpp_rashba = 0.5 * (hp_i.tpp_rashba + hp_j.tpp_rashba)

            # Build (4,4) orbital hopping, then double to (8,8) spin space
            H_orb = build_hopping_4x4(d, tss, tsp, tpp, 0.0,
                                      tsp_rashba, tpp_rashba)
            H_full = spin_double(H_orb)

            # Project to active basis
            H_contribution = project_matrix(H_full, proj_i, proj_j)

            system.matrices[nb.matrix_idx].H[si, sj] += H_contribution
