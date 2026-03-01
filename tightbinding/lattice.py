"""Supercell construction and neighbor finding.

Replaces make_lattice.m.  Builds the atom list, identifies neighbor cells
within hopping range, initializes HoppingMatrix entries, and constructs the
atompos displacement matrices used in the Bloch sum.
"""

import numpy as np
from numpy.typing import NDArray

from .types import Atom, HoppingMatrix, AtomPos, System
from .basis import get_norbs
from .neighbors import find_neighbors


def build_system(cfg: dict) -> System:
    """Build a System from a parsed YAML config.

    This is the top-level system-building function for periodic TB_simple
    calculations.  It:
    1. Creates unit-cell atoms
    2. Finds neighbors via KD-tree
    3. Builds atompos displacement matrices
    """
    sys_cfg = cfg['system']
    hop_cfg = cfg['hopping']
    onsite_cfg = cfg.get('onsite', {})

    lattice_type = sys_cfg['lattice_type']
    basis = sys_cfg['basis']
    lattice_vectors = sys_cfg['lattice_vectors']

    Nx = sys_cfg.get('Nx', 1)
    Ny = sys_cfg.get('Ny', 1)

    hopping_range = hop_cfg['range']

    # Build unit cell atoms
    atoms, norbs_total = _create_unitcell_atoms(
        Nx, Ny, lattice_type, basis, hop_cfg, onsite_cfg, hopping_range
    )

    # Collect UC coordinates
    coords = np.array([a.coord for a in atoms])

    # Find neighbors via KD-tree
    neighbors, matrices = find_neighbors(
        coords, lattice_vectors, hopping_range, norbs_total
    )

    # Build atompos displacement matrices
    atompos = _build_atompos(atoms, norbs_total)

    system = System(
        atoms=atoms,
        matrices=matrices,
        unitcell_vectors=lattice_vectors,
        norbs=norbs_total,
        atompos=atompos,
        neighbors=neighbors,
    )
    return system


def _create_unitcell_atoms(Nx, Ny, lattice_type, basis, hop_cfg, onsite_cfg,
                           hopping_range):
    """Create atoms in the unit cell. Currently supports '2d_square'."""
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
                    pair=0,
                    tss=hop_cfg.get('tss', 0.0),
                    tpp=hop_cfg.get('tpp', 0.0),
                    tsp=hop_cfg.get('tsp', 0.0),
                    tsp_rashba=hop_cfg.get('tsp_rashba', 0.0),
                    tpp_rashba=hop_cfg.get('tpp_rashba', 0.0),
                    hopping_range=hopping_range,
                    u_s=onsite_cfg.get('u_s', 0.0),
                    u_p=onsite_cfg.get('u_p', 0.0),
                    delta_s=onsite_cfg.get('delta_s', 0.0),
                    delta_p=onsite_cfg.get('delta_p', 0.0),
                    theta=onsite_cfg.get('theta', 0.0),
                    phi=onsite_cfg.get('phi', 0.0),
                    spinorbit=onsite_cfg.get('spinorbit', 0.0),
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
