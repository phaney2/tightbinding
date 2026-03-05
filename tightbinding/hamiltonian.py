"""Hamiltonian construction — fills H/S matrices.

Replaces constructHamiltonian_periodic_toy.m.  Iterates over the neighbor
table, computes hopping and on-site contributions, projects to the active
basis, and accumulates into the System's HoppingMatrix entries.

Supports two full-space dimensions:
  8-dim (sp×spin): uses build_hopping_4x4 + spin_double
 18-dim (spd×spin): uses build_hopping_9x9 + spin_double
"""

from collections import defaultdict

import numpy as np

from .types import System, OnsiteParams, HoppingParams
from .basis import get_projector, project_matrix, is_spd_basis
from .slater_koster import build_hopping_4x4, build_hopping_9x9, spin_double
from .onsite import build_onsite_8x8_from_params, build_onsite_18x18_from_params


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

    # Detect whether any atom uses spd basis
    use_spd = any(is_spd_basis(atom.basis) for atom in atoms)

    # Validate: don't mix sp and spd full-space dimensions
    if use_spd and any(not is_spd_basis(a.basis) for a in atoms):
        raise ValueError(
            "Cannot mix sp-type and spd-type basis types in the same system. "
            "All atoms must use the same full-space dimension."
        )

    # Group neighbors by site_i
    nbrs_by_site = defaultdict(list)
    for nb in neighbors:
        nbrs_by_site[nb.site_i].append(nb)

    for i, atom_i in enumerate(atoms):
        proj_i = get_projector(atom_i.basis)
        si = atom_i.orb_slice

        # On-site term (diagonal block in R=0 matrix)
        osp = _get_params(onsite_p, atom_i.species)
        if use_spd:
            H_onsite_full = build_onsite_18x18_from_params(osp)
        else:
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
            tss_s = 0.5 * (hp_i.tss_sigma + hp_j.tss_sigma)
            tsp_s = 0.5 * (hp_i.tsp_sigma + hp_j.tsp_sigma)
            tpp_s = 0.5 * (hp_i.tpp_sigma + hp_j.tpp_sigma)
            tpp_p = 0.5 * (hp_i.tpp_pi + hp_j.tpp_pi)
            tsp_rashba = 0.5 * (hp_i.tsp_rashba + hp_j.tsp_rashba)
            tpp_rashba = 0.5 * (hp_i.tpp_rashba + hp_j.tpp_rashba)

            if use_spd:
                # Average d-orbital hopping parameters
                tsd_s = 0.5 * (hp_i.tsd_sigma + hp_j.tsd_sigma)
                tpd_s = 0.5 * (hp_i.tpd_sigma + hp_j.tpd_sigma)
                tpd_p = 0.5 * (hp_i.tpd_pi + hp_j.tpd_pi)
                tdd_s = 0.5 * (hp_i.tdd_sigma + hp_j.tdd_sigma)
                tdd_p = 0.5 * (hp_i.tdd_pi + hp_j.tdd_pi)
                tdd_d = 0.5 * (hp_i.tdd_delta + hp_j.tdd_delta)
                tsd_r = 0.5 * (hp_i.tsd_rashba + hp_j.tsd_rashba)
                tpd_r = 0.5 * (hp_i.tpd_rashba + hp_j.tpd_rashba)
                tdd_r = 0.5 * (hp_i.tdd_rashba + hp_j.tdd_rashba)

                # Build (9,9) orbital hopping, then double to (18,18) spin space
                H_orb = build_hopping_9x9(
                    d, tss_s, tsp_s, tpp_s, tpp_p,
                    tsd_s, tpd_s, tpd_p,
                    tdd_s, tdd_p, tdd_d,
                    tsd_r, tpd_r, tdd_r,
                )
            else:
                # Build (4,4) orbital hopping, then double to (8,8) spin space
                H_orb = build_hopping_4x4(d, tss_s, tsp_s, tpp_s, tpp_p,
                                          tsp_rashba, tpp_rashba)

            H_full = spin_double(H_orb)

            # Project to active basis
            H_contribution = project_matrix(H_full, proj_i, proj_j)

            system.matrices[nb.matrix_idx].H[si, sj] += H_contribution
