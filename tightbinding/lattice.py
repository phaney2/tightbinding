"""Supercell construction and neighbor finding.

Replaces make_lattice.m.  Builds the atom list from either explicit
positions or named lattice types, finds neighbors via KD-tree,
and constructs the atompos displacement matrices used in the Bloch sum.
"""

import numpy as np
from numpy.typing import NDArray

from .types import Atom, HoppingMatrix, AtomPos, System, OnsiteParams, HoppingParams
from .basis import get_norbs
from .neighbors import find_neighbors


def build_system(cfg: dict) -> System:
    """Build a System from a parsed YAML config.

    Supports two ways to specify atom positions:
    1. Explicit 'positions' list with species + coords
    2. Named 'lattice_type' (e.g., '2d_square') for convenience
    """
    sys_cfg = cfg['system']
    hop_cfg = cfg['hopping']
    onsite_cfg = cfg.get('onsite', {})

    basis = sys_cfg['basis']
    lattice_vectors = sys_cfg['lattice_vectors']
    hopping_range = hop_cfg['range']

    # Build unit cell atoms
    if 'positions' in sys_cfg:
        atoms, norbs_total = _create_atoms_from_positions(
            sys_cfg['positions'], basis, lattice_vectors,
            sys_cfg.get('coord_type', 'cartesian'),
        )
    else:
        lattice_type = sys_cfg['lattice_type']
        Nx = sys_cfg.get('Nx', 1)
        Ny = sys_cfg.get('Ny', 1)
        atoms, norbs_total = _create_unitcell_atoms(
            Nx, Ny, lattice_type, basis,
        )

    # Collect UC coordinates
    coords = np.array([a.coord for a in atoms])

    # Find neighbors via KD-tree
    neighbors, matrices = find_neighbors(
        coords, lattice_vectors, hopping_range, norbs_total
    )

    # Build atompos displacement matrices
    atompos = _build_atompos(atoms, norbs_total)

    # Build parameter dicts from config
    onsite_params = {'_default': OnsiteParams(
        u_s=onsite_cfg.get('u_s', 0.0),
        u_p=onsite_cfg.get('u_p', 0.0),
        u_d=onsite_cfg.get('u_d', 0.0),
        delta_s=onsite_cfg.get('delta_s', 0.0),
        delta_p=onsite_cfg.get('delta_p', 0.0),
        delta_d=onsite_cfg.get('delta_d', 0.0),
        theta=onsite_cfg.get('theta', 0.0),
        phi=onsite_cfg.get('phi', 0.0),
        spinorbit=onsite_cfg.get('spinorbit', 0.0),
        spinorbit_d=onsite_cfg.get('spinorbit_d', 0.0),
    )}
    hopping_params = {'_default': HoppingParams(
        tss_sigma=hop_cfg.get('tss_sigma', 0.0),
        tsp_sigma=hop_cfg.get('tsp_sigma', 0.0),
        tpp_sigma=hop_cfg.get('tpp_sigma', 0.0),
        tpp_pi=hop_cfg.get('tpp_pi', 0.0),
        tsd_sigma=hop_cfg.get('tsd_sigma', 0.0),
        tpd_sigma=hop_cfg.get('tpd_sigma', 0.0),
        tpd_pi=hop_cfg.get('tpd_pi', 0.0),
        tdd_sigma=hop_cfg.get('tdd_sigma', 0.0),
        tdd_pi=hop_cfg.get('tdd_pi', 0.0),
        tdd_delta=hop_cfg.get('tdd_delta', 0.0),
        tsp_rashba=hop_cfg.get('tsp_rashba', 0.0),
        tpp_rashba=hop_cfg.get('tpp_rashba', 0.0),
        tsd_rashba=hop_cfg.get('tsd_rashba', 0.0),
        tpd_rashba=hop_cfg.get('tpd_rashba', 0.0),
        tdd_rashba=hop_cfg.get('tdd_rashba', 0.0),
    )}

    system = System(
        atoms=atoms,
        matrices=matrices,
        unitcell_vectors=lattice_vectors,
        norbs=norbs_total,
        atompos=atompos,
        neighbors=neighbors,
        onsite_params=onsite_params,
        hopping_params=hopping_params,
    )
    return system


def _create_atoms_from_positions(positions, basis, lattice_vectors, coord_type):
    """Create atoms from an explicit list of positions.

    Each entry in positions is a dict with 'coord' (and optionally 'species').
    If coord_type == 'fractional', coordinates are converted to Cartesian.
    """
    atoms = []
    norbs_total = 0
    norb = get_norbs(basis)

    for pos in positions:
        coord = np.asarray(pos['coord'], dtype=float)
        if coord_type == 'fractional':
            coord = coord @ np.asarray(lattice_vectors, dtype=float)
        species = pos.get('species', '')

        idx = len(atoms)
        orb_start = norbs_total
        atom = Atom(
            index=idx,
            coord=coord,
            basis=basis,
            norb=norb,
            orb_slice=slice(orb_start, orb_start + norb),
            species=species,
        )
        atoms.append(atom)
        norbs_total += norb

    return atoms, norbs_total


def _create_unitcell_atoms(Nx, Ny, lattice_type, basis):
    """Create atoms in the unit cell from a named lattice type."""
    atoms = []
    norbs_total = 0
    norb = get_norbs(basis)

    if lattice_type == '2d_square':
        for c1 in range(Nx):
            for c2 in range(Ny):
                idx = len(atoms)
                orb_start = norbs_total
                atom = Atom(
                    index=idx,
                    coord=np.array([float(c1), float(c2), 0.0]),
                    basis=basis,
                    norb=norb,
                    orb_slice=slice(orb_start, orb_start + norb),
                )
                atoms.append(atom)
                norbs_total += norb
    else:
        raise ValueError(f"Unknown lattice_type: '{lattice_type}'")

    return atoms, norbs_total


def _build_atompos(uc_atoms: list[Atom], norbs_total: int) -> AtomPos:
    """Build atompos displacement matrices between orbital pairs.

    atompos.x[i,j] = -(r_i - r_j)_x for orbitals i, j in the unit cell.
    This matches the MATLAB convention used in get_H_v.m.
    """
    xpos = np.zeros(norbs_total)
    ypos = np.zeros(norbs_total)
    zpos = np.zeros(norbs_total)

    for atom in uc_atoms:
        s = atom.orb_slice
        xpos[s] = atom.coord[0]
        ypos[s] = atom.coord[1]
        zpos[s] = atom.coord[2]

    # atompos.x[i,j] = -(xpos[i] - xpos[j])
    ax = -(xpos[:, None] - xpos[None, :])
    ay = -(ypos[:, None] - ypos[None, :])
    az = -(zpos[:, None] - zpos[None, :])

    return AtomPos(x=ax, y=ay, z=az)
