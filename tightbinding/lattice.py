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


def _make_onsite(d: dict) -> OnsiteParams:
    """Build OnsiteParams from a dict of values."""
    return OnsiteParams(
        u_s=d.get('u_s', 0.0),
        u_p=d.get('u_p', 0.0),
        u_d=d.get('u_d', 0.0),
        u_pz=d.get('u_pz', None),
        u_dz2=d.get('u_dz2', None),
        u_dxz=d.get('u_dxz', None),
        delta_s=d.get('delta_s', 0.0),
        delta_p=d.get('delta_p', 0.0),
        delta_d=d.get('delta_d', 0.0),
        theta=d.get('theta', 0.0),
        phi=d.get('phi', 0.0),
        spinorbit=d.get('spinorbit', 0.0),
        spinorbit_d=d.get('spinorbit_d', 0.0),
    )


def _build_onsite_params(onsite_cfg: dict) -> dict[str, OnsiteParams]:
    """Build onsite_params dict from config.

    Supports two formats:
      Flat (all species share same params):
        onsite:
          u_s: 1.0
          u_p: 0.5

      Per-species (sub-dicts keyed by species name):
        onsite:
          A:
            u_s: 0.5
          B:
            u_s: -0.5
    """
    if not onsite_cfg:
        return {'_default': OnsiteParams()}

    # Detect format: if any value is a dict, treat as per-species
    has_species = any(isinstance(v, dict) for v in onsite_cfg.values())

    if not has_species:
        return {'_default': _make_onsite(onsite_cfg)}

    params = {}
    for key, val in onsite_cfg.items():
        if isinstance(val, dict):
            params[key] = _make_onsite(val)
        # else: ignore stray scalar keys at top level when species dicts are present
    return params


def _make_hopping(d: dict) -> HoppingParams:
    """Build HoppingParams from a dict of values."""
    return HoppingParams(
        tss_sigma=d.get('tss_sigma', 0.0),
        tsp_sigma=d.get('tsp_sigma', 0.0),
        tpp_sigma=d.get('tpp_sigma', 0.0),
        tpp_pi=d.get('tpp_pi', 0.0),
        tsd_sigma=d.get('tsd_sigma', 0.0),
        tpd_sigma=d.get('tpd_sigma', 0.0),
        tpd_pi=d.get('tpd_pi', 0.0),
        tdd_sigma=d.get('tdd_sigma', 0.0),
        tdd_pi=d.get('tdd_pi', 0.0),
        tdd_delta=d.get('tdd_delta', 0.0),
        tsp_rashba=d.get('tsp_rashba', 0.0),
        tpp_rashba=d.get('tpp_rashba', 0.0),
        tsd_rashba=d.get('tsd_rashba', 0.0),
        tpd_rashba=d.get('tpd_rashba', 0.0),
        tdd_rashba=d.get('tdd_rashba', 0.0),
    )


def _build_hopping_params(hop_cfg: dict) -> dict[str, HoppingParams]:
    """Build hopping_params dict from config.

    Supports two formats:
      Flat (all species share same params):
        hopping:
          range: 3.5
          tpp_sigma: 0.5

      Per-species (sub-dicts keyed by species name):
        hopping:
          range: 3.5
          Mo:
            tdd_sigma: -0.933
          S:
            tpp_sigma: 0.696
    """
    # Separate control keys from parameter keys
    control_keys = {'range', 'anisotropy_factor', 'anisotropy_direction'}
    has_species = any(
        isinstance(v, dict) for k, v in hop_cfg.items() if k not in control_keys
    )

    if not has_species:
        return {'_default': _make_hopping(hop_cfg)}

    params = {}
    for key, val in hop_cfg.items():
        if key in control_keys:
            continue
        if isinstance(val, dict):
            params[key] = _make_hopping(val)
    return params


def build_system(cfg: dict) -> System:
    """Build a System from a parsed YAML config.

    Supports two ways to specify atom positions:
    1. Explicit 'positions' list with species + coords
    2. Named 'lattice_type' (e.g., '2d_square') for convenience
    """
    sys_cfg = cfg['system']
    hop_cfg = cfg['hopping']
    onsite_cfg = cfg.get('onsite', {})

    basis = sys_cfg.get('basis', None)
    lattice_vectors = sys_cfg['lattice_vectors']
    hopping_range = hop_cfg['range']

    # Build unit cell atoms
    if 'positions' in sys_cfg:
        atoms, norbs_total = _create_atoms_from_positions(
            sys_cfg['positions'], basis, lattice_vectors,
            sys_cfg.get('coord_type', 'cartesian'),
        )
    else:
        if basis is None:
            raise ValueError("system.basis is required when using lattice_type")
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
    onsite_params = _build_onsite_params(onsite_cfg)
    hopping_params = _build_hopping_params(hop_cfg)

    # Hopping anisotropy (traceless strain model)
    aniso_dir = None
    aniso_factor = hop_cfg.get('anisotropy_factor', 0.0)
    if aniso_factor != 0.0:
        raw_dir = hop_cfg.get('anisotropy_direction')
        if raw_dir is None:
            raise ValueError(
                "hopping.anisotropy_direction is required when "
                "anisotropy_factor is nonzero"
            )
        aniso_dir = np.array(raw_dir, dtype=float)
        norm = np.linalg.norm(aniso_dir)
        if norm < 1e-12:
            raise ValueError("hopping.anisotropy_direction must be nonzero")
        aniso_dir = aniso_dir / norm

    system = System(
        atoms=atoms,
        matrices=matrices,
        unitcell_vectors=lattice_vectors,
        norbs=norbs_total,
        atompos=atompos,
        neighbors=neighbors,
        onsite_params=onsite_params,
        hopping_params=hopping_params,
        hopping_anisotropy_direction=aniso_dir,
        hopping_anisotropy_factor=aniso_factor,
    )
    return system


def _create_atoms_from_positions(positions, default_basis, lattice_vectors, coord_type):
    """Create atoms from an explicit list of positions.

    Each entry in positions is a dict with 'coord' (and optionally 'species'
    and 'basis').  Per-atom 'basis' overrides the system-level default.
    If coord_type == 'fractional', coordinates are converted to Cartesian.
    """
    atoms = []
    norbs_total = 0

    for pos in positions:
        coord = np.asarray(pos['coord'], dtype=float)
        if coord_type == 'fractional':
            coord = coord @ np.asarray(lattice_vectors, dtype=float)
        species = pos.get('species', '')
        atom_basis = pos.get('basis', default_basis)
        if atom_basis is None:
            raise ValueError(
                f"No basis specified for atom at {coord}. "
                "Set system.basis or per-atom basis."
            )
        norb = get_norbs(atom_basis)

        idx = len(atoms)
        orb_start = norbs_total
        atom = Atom(
            index=idx,
            coord=coord,
            basis=atom_basis,
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
