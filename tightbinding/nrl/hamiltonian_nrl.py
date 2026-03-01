"""NRL tight-binding Hamiltonian builder.

Replaces constructHamiltonian_NRL_nospin.m and read_H_params.m.

Builds the 9-orbital-per-atom Hamiltonian (without spin) using
distance-dependent Slater-Koster parameters from NRL .dat files,
then optionally doubles for spin and adds SOC/exchange.
"""

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from ..types import Atom, HoppingMatrix, AtomPos, System, NeighborEntry
from ..slater_koster import build_hopping_9x9
from ..neighbors import find_neighbors
from .params import parse_dat_file, eval_hopping, cutoff_function


NORBS_PER_ATOM = 9  # s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2
RYD_TO_EV = 13.67


def build_nrl_system(species_list: list[str], coords: NDArray,
                     lattice_vectors: NDArray, hopping_range: float,
                     dat_file: str, vso: dict | None = None) -> System:
    """Build a complete NRL tight-binding System.

    Parameters
    ----------
    species_list : atom species (e.g., ['Pt'])
    coords : (natoms, 3) Cartesian coordinates
    lattice_vectors : (3, 3) lattice vectors, rows are a1, a2, a3
    hopping_range : cutoff distance for hopping
    dat_file : path to NRL .dat parameter file
    vso : dict with SOC parameters, e.g. {'Pt': {'p': 0.05, 'd': 0.5}}

    Returns
    -------
    System with spin-doubled Hamiltonian (18*natoms orbitals)
    """
    natoms = len(species_list)
    norbs_nospin = natoms * NORBS_PER_ATOM

    # Parse NRL parameters
    nrl_params = parse_dat_file(dat_file)

    # Create unit-cell atoms
    atoms = []
    for i in range(natoms):
        atom = Atom(
            index=i,
            coord=coords[i].copy(),
            basis='spd',
            norb=NORBS_PER_ATOM,
            orb_slice=slice(i * NORBS_PER_ATOM, (i + 1) * NORBS_PER_ATOM),
            species=species_list[i],
        )
        atoms.append(atom)

    # Find neighbors via KD-tree
    neighbors, matrices = find_neighbors(
        coords, lattice_vectors, hopping_range, norbs_nospin
    )

    # Fill Hamiltonian and overlap (no spin)
    _fill_nrl_hamiltonian(atoms, neighbors, matrices, nrl_params, norbs_nospin)

    # Convert from Rydberg to eV
    for mat in matrices:
        mat.H *= RYD_TO_EV

    # Spin-double: create 2*norbs system
    norbs_spin = 2 * norbs_nospin
    matrices_spin = _spin_double_matrices(matrices, norbs_nospin)

    # Add SOC and exchange to the on-site block (matrices_spin[0])
    if vso is not None:
        _add_soc_nrl(matrices_spin[0], atoms, vso, norbs_nospin)

    # Build atompos
    atompos = _build_atompos_nrl(atoms, norbs_nospin)

    # Update atoms with spin-doubled orbital slices
    atoms_spin = _spin_double_atoms(atoms)

    return System(
        atoms=atoms_spin,
        matrices=matrices_spin,
        unitcell_vectors=lattice_vectors,
        norbs=norbs_spin,
        atompos=atompos,
        neighbors=neighbors,
    )


def _fill_nrl_hamiltonian(atoms, neighbors, matrices, nrl_params, norbs_total):
    """Fill H and S matrices using NRL distance-dependent parameters."""
    t_params = nrl_params['hopping']
    s_params = nrl_params['overlap']
    onsite_params = nrl_params['onsite']
    lambda_ = nrl_params['lambda_']

    # Group neighbors by site_i for density calculation
    nbrs_by_site = defaultdict(list)
    for nb in neighbors:
        nbrs_by_site[nb.site_i].append(nb)

    natoms = len(atoms)

    for i in range(natoms):
        si = atoms[i].orb_slice

        # Compute effective electron density at this site
        rho = 0.0
        for nb in nbrs_by_site[i]:
            R = nb.distance
            fc = cutoff_function(R)
            rho += np.exp(-lambda_**2 * R) * fc

        # Hopping: iterate over neighbors
        for nb in nbrs_by_site[i]:
            j = nb.site_j
            R = nb.distance
            d = nb.direction * R  # displacement vector from i to j+R
            sj = atoms[j].orb_slice
            fc = cutoff_function(R)

            # Evaluate all hopping parameters at this distance
            hop_vals = {k: eval_hopping(v, R, fc) for k, v in t_params.items()}
            ovl_vals = {k: eval_hopping(v, R, fc) for k, v in s_params.items()}

            # Build 9x9 hopping and overlap blocks
            H_block = build_hopping_9x9(
                d, hop_vals['ss_s'], hop_vals['sp_s'],
                hop_vals['pp_s'], hop_vals['pp_p'],
                hop_vals['sd_s'], hop_vals['pd_s'], hop_vals['pd_p'],
                hop_vals['dd_s'], hop_vals['dd_p'], hop_vals['dd_d'],
            )
            S_block = build_hopping_9x9(
                d, ovl_vals['ss_s'], ovl_vals['sp_s'],
                ovl_vals['pp_s'], ovl_vals['pp_p'],
                ovl_vals['sd_s'], ovl_vals['pd_s'], ovl_vals['pd_p'],
                ovl_vals['dd_s'], ovl_vals['dd_p'], ovl_vals['dd_d'],
            )

            matrices[nb.matrix_idx].H[si, sj] += H_block
            matrices[nb.matrix_idx].S[si, sj] += S_block

        # On-site energies (density-dependent)
        e_s = (onsite_params['s'][0] + onsite_params['s'][1] * rho**(2/3) +
               onsite_params['s'][2] * rho**(4/3) + onsite_params['s'][3] * rho**2)
        e_p = (onsite_params['p'][0] + onsite_params['p'][1] * rho**(2/3) +
               onsite_params['p'][2] * rho**(4/3) + onsite_params['p'][3] * rho**2)
        e_d = (onsite_params['d'][0] + onsite_params['d'][1] * rho**(2/3) +
               onsite_params['d'][2] * rho**(4/3) + onsite_params['d'][3] * rho**2)

        s_start = i * NORBS_PER_ATOM
        matrices[0].H[s_start, s_start] += e_s
        for p_idx in range(s_start + 1, s_start + 4):
            matrices[0].H[p_idx, p_idx] += e_p
        for d_idx in range(s_start + 4, s_start + 9):
            matrices[0].H[d_idx, d_idx] += e_d


def _spin_double_matrices(matrices, norbs_nospin):
    """Double all matrices for spin: [H 0; 0 H], [S 0; 0 S]."""
    n = norbs_nospin
    n2 = 2 * n
    result = []
    for mat in matrices:
        H_spin = np.zeros((n2, n2), dtype=complex)
        S_spin = np.zeros((n2, n2), dtype=complex)
        H_spin[:n, :n] = mat.H
        H_spin[n:, n:] = mat.H
        S_spin[:n, :n] = mat.S
        S_spin[n:, n:] = mat.S
        result.append(HoppingMatrix(
            displacement=mat.displacement.copy(),
            H=H_spin,
            S=S_spin,
        ))
    return result


def _add_soc_nrl(onsite_mat: HoppingMatrix, atoms, vso: dict,
                 norbs_nospin: int):
    """Add spin-orbit coupling to the on-site Hamiltonian.

    Modifies onsite_mat.H in-place.  The spin structure is:
    [uu block | ud block]
    [du block | dd block]
    where each block is norbs_nospin x norbs_nospin.
    """
    n = norbs_nospin
    H = onsite_mat.H

    for site in range(len(atoms)):
        species = atoms[site].species
        if species not in vso:
            continue

        # p-orbital SOC
        tvso_p = vso[species].get('p', 0.0)
        if tvso_p != 0.0:
            px = NORBS_PER_ATOM * site + 1
            py = NORBS_PER_ATOM * site + 2
            pz = NORBS_PER_ATOM * site + 3

            # uu block
            H[px, py] += -1j * tvso_p
            H[py, px] += 1j * tvso_p
            # dd block
            H[n+px, n+py] += 1j * tvso_p
            H[n+py, n+px] += -1j * tvso_p
            # ud block
            H[px, n+pz] += tvso_p
            H[py, n+pz] += -1j * tvso_p
            H[pz, n+px] += -tvso_p
            H[pz, n+py] += 1j * tvso_p
            # du block
            H[n+px, pz] += -tvso_p
            H[n+py, pz] += -1j * tvso_p
            H[n+pz, px] += tvso_p
            H[n+pz, py] += 1j * tvso_p

        # d-orbital SOC
        tvso_d = vso[species].get('d', 0.0)
        if tvso_d != 0.0:
            dxy = NORBS_PER_ATOM * site + 4
            dyz = NORBS_PER_ATOM * site + 5
            dzx = NORBS_PER_ATOM * site + 6
            dxxyy = NORBS_PER_ATOM * site + 7
            dzz = NORBS_PER_ATOM * site + 8

            f = np.sqrt(3.0)

            # uu block
            H[dzx, dyz] += -1j * tvso_d
            H[dyz, dzx] += 1j * tvso_d
            H[dxxyy, dxy] += -2j * tvso_d
            H[dxy, dxxyy] += 2j * tvso_d

            # dd block
            H[n+dzx, n+dyz] += 1j * tvso_d
            H[n+dyz, n+dzx] += -1j * tvso_d
            H[n+dxxyy, n+dxy] += 2j * tvso_d
            H[n+dxy, n+dxxyy] += -2j * tvso_d

            # du block
            H[n+dzz, dzx] += f * tvso_d
            H[n+dzz, dyz] += 1j * f * tvso_d
            H[n+dzx, dzz] += -f * tvso_d
            H[n+dzx, dxxyy] += tvso_d
            H[n+dzx, dxy] += 1j * tvso_d
            H[n+dyz, dzz] += -1j * f * tvso_d
            H[n+dyz, dxxyy] += -1j * tvso_d
            H[n+dyz, dxy] += tvso_d
            H[n+dxxyy, dzx] += -tvso_d
            H[n+dxxyy, dyz] += 1j * tvso_d
            H[n+dxy, dzx] += -1j * tvso_d
            H[n+dxy, dyz] += -tvso_d

            # ud block
            H[dzz, n+dzx] += -f * tvso_d
            H[dzz, n+dyz] += 1j * f * tvso_d
            H[dzx, n+dzz] += f * tvso_d
            H[dzx, n+dxxyy] += -tvso_d
            H[dzx, n+dxy] += 1j * tvso_d
            H[dyz, n+dzz] += -1j * f * tvso_d
            H[dyz, n+dxxyy] += -1j * tvso_d
            H[dyz, n+dxy] += -tvso_d
            H[dxxyy, n+dzx] += tvso_d
            H[dxxyy, n+dyz] += 1j * tvso_d
            H[dxy, n+dzx] += -1j * tvso_d
            H[dxy, n+dyz] += tvso_d


def _build_atompos_nrl(atoms, norbs_nospin):
    """Build spin-doubled atompos displacement matrices."""
    natoms = len(atoms)
    n = norbs_nospin
    n2 = 2 * n

    xpos = np.zeros(n)
    ypos = np.zeros(n)
    zpos = np.zeros(n)

    for i in range(natoms):
        s = atoms[i].orb_slice
        xpos[s] = atoms[i].coord[0]
        ypos[s] = atoms[i].coord[1]
        zpos[s] = atoms[i].coord[2]

    # Build nospin atompos
    ax = -(xpos[:, None] - xpos[None, :])
    ay = -(ypos[:, None] - ypos[None, :])
    az = -(zpos[:, None] - zpos[None, :])

    # Double for spin
    ax2 = np.zeros((n2, n2))
    ay2 = np.zeros((n2, n2))
    az2 = np.zeros((n2, n2))
    ax2[:n, :n] = ax; ax2[n:, n:] = ax
    ay2[:n, :n] = ay; ay2[n:, n:] = ay
    az2[:n, :n] = az; az2[n:, n:] = az

    return AtomPos(x=ax2, y=ay2, z=az2)


def _spin_double_atoms(atoms):
    """Update atom orbital slices for spin-doubled system.

    Only unit-cell atoms — no image atoms to handle.
    """
    result = []
    for a in atoms:
        new_atom = Atom(
            index=a.index,
            coord=a.coord.copy(),
            basis=a.basis,
            norb=2 * a.norb,
            orb_slice=slice(a.orb_slice.start, a.orb_slice.stop),  # just uu block
            species=a.species,
        )
        result.append(new_atom)
    return result
