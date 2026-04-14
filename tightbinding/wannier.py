"""Wannier90 _hr.dat and _tb.dat file parsers.

Reads a Wannier90 Hamiltonian in the _hr.dat or _tb.dat format and builds
a System object compatible with the tight-binding code's Bloch sum machinery.
The _tb.dat format additionally contains lattice vectors and position operator
matrix elements.
"""

import numpy as np
from numpy.typing import NDArray

from .types import Atom, HoppingMatrix, AtomPos, System


def read_hr(path: str) -> tuple[int, list[NDArray], list[int], list[NDArray]]:
    """Parse a Wannier90 _hr.dat file.

    Parameters
    ----------
    path : path to the *_hr.dat file

    Returns
    -------
    norbs : number of Wannier orbitals
    displacements : list of R-vectors in lattice coordinates, each shape (3,)
    degeneracies : degeneracy weight for each R-vector
    H_matrices : list of (norbs, norbs) complex Hamiltonian blocks
    """
    with open(path, 'r') as f:
        # Line 1: comment
        f.readline()
        # Line 2: number of Wannier functions
        norbs = int(f.readline().strip())
        # Line 3: number of R-vectors
        nrpts = int(f.readline().strip())

        # Degeneracy weights: 15 integers per line
        degeneracies = []
        while len(degeneracies) < nrpts:
            line = f.readline().split()
            degeneracies.extend(int(x) for x in line)

        # Hopping data: for each R-vector, norbs^2 lines
        # Format: R1 R2 R3 m n Re(H) Im(H)
        displacements = []
        H_matrices = []
        current_R = None
        H_re = np.zeros((norbs, norbs))
        H_im = np.zeros((norbs, norbs))

        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            r1, r2, r3 = int(parts[0]), int(parts[1]), int(parts[2])
            m, n = int(parts[3]) - 1, int(parts[4]) - 1  # 1-indexed -> 0-indexed
            re_h, im_h = float(parts[5]), float(parts[6])

            R = (r1, r2, r3)
            if current_R is None:
                current_R = R

            if R != current_R:
                # Save previous R-vector's matrix
                displacements.append(np.array(current_R, dtype=float))
                H_matrices.append(H_re + 1j * H_im)
                H_re = np.zeros((norbs, norbs))
                H_im = np.zeros((norbs, norbs))
                current_R = R

            H_re[m, n] = re_h
            H_im[m, n] = im_h

        # Don't forget the last block
        if current_R is not None:
            displacements.append(np.array(current_R, dtype=float))
            H_matrices.append(H_re + 1j * H_im)

    return norbs, displacements, degeneracies, H_matrices


def read_tb(path: str) -> tuple[NDArray, int, list[NDArray], list[int],
                                list[NDArray], list[list[NDArray]]]:
    """Parse a Wannier90 _tb.dat file.

    The _tb.dat format contains lattice vectors, the Hamiltonian blocks
    (same as _hr.dat), and the position operator matrix elements <0n|r|Rm>.

    Parameters
    ----------
    path : path to the *_tb.dat file

    Returns
    -------
    lattice_vectors : (3, 3) array, rows are a1, a2, a3
    norbs : number of Wannier orbitals
    displacements : list of R-vectors in lattice coordinates, each shape (3,)
    degeneracies : degeneracy weight for each R-vector
    H_matrices : list of (norbs, norbs) complex Hamiltonian blocks
    r_matrices : list (one per R-vector) of [r_x, r_y, r_z], each (norbs, norbs) complex
    """
    with open(path, 'r') as f:
        # Line 1: comment
        f.readline()
        # Lines 2-4: lattice vectors
        lattice_vectors = np.zeros((3, 3))
        for i in range(3):
            lattice_vectors[i] = [float(x) for x in f.readline().split()]
        # Number of Wannier functions
        norbs = int(f.readline().strip())
        # Number of R-vectors
        nrpts = int(f.readline().strip())

        # Degeneracy weights: 15 integers per line
        degeneracies = []
        while len(degeneracies) < nrpts:
            line = f.readline().split()
            degeneracies.extend(int(x) for x in line)

        # --- Hamiltonian blocks ---
        # Each block: blank line, R-vector line, then norbs^2 data lines
        displacements = []
        H_matrices = []
        for _ in range(nrpts):
            f.readline()  # blank line
            parts = f.readline().split()
            R = np.array([int(parts[0]), int(parts[1]), int(parts[2])], dtype=float)
            displacements.append(R)

            H = np.zeros((norbs, norbs), dtype=complex)
            for _ in range(norbs * norbs):
                parts = f.readline().split()
                m, n = int(parts[0]) - 1, int(parts[1]) - 1
                H[m, n] = float(parts[2]) + 1j * float(parts[3])
            H_matrices.append(H)

        # --- Position operator blocks ---
        # Same structure: blank line, R-vector, norbs^2 lines with 6 floats
        # (Re_x, Im_x, Re_y, Im_y, Re_z, Im_z)
        r_matrices = []
        for _ in range(nrpts):
            f.readline()  # blank line
            f.readline()  # R-vector line (same ordering as H blocks)

            r_x = np.zeros((norbs, norbs), dtype=complex)
            r_y = np.zeros((norbs, norbs), dtype=complex)
            r_z = np.zeros((norbs, norbs), dtype=complex)
            for _ in range(norbs * norbs):
                parts = f.readline().split()
                m, n = int(parts[0]) - 1, int(parts[1]) - 1
                r_x[m, n] = float(parts[2]) + 1j * float(parts[3])
                r_y[m, n] = float(parts[4]) + 1j * float(parts[5])
                r_z[m, n] = float(parts[6]) + 1j * float(parts[7])
            r_matrices.append([r_x, r_y, r_z])

    return lattice_vectors, norbs, displacements, degeneracies, H_matrices, r_matrices


def read_centres(path: str) -> NDArray:
    """Parse a Wannier90 *_centres.xyz file.

    Parameters
    ----------
    path : path to the *_centres.xyz file

    Returns
    -------
    centres : (nwann, 3) array of Wannier function centres in Cartesian Å.
              Only the 'X' (Wannier) entries are returned, not the atomic positions.
    """
    centres = []
    with open(path, 'r') as f:
        natoms = int(f.readline().strip())
        f.readline()  # comment line
        for _ in range(natoms):
            parts = f.readline().split()
            if parts[0] == 'X':
                centres.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(centres)


def build_system_from_hr(hr_path: str,
                         unitcell_vectors: NDArray,
                         centres_path: str = None,
                         ) -> System:
    """Build a System from a Wannier90 _hr.dat file.

    Parameters
    ----------
    hr_path : path to the *_hr.dat file
    unitcell_vectors : (3, 3) array, rows are a1, a2, a3
    centres_path : optional path to *_centres.xyz for Wannier function positions.
                   If provided, builds proper atompos for velocity operators.

    Returns
    -------
    System object ready for band structure and response calculations.
    The Hamiltonian blocks are divided by degeneracy weights and the
    displacement vectors are converted to Cartesian coordinates.
    """
    norbs, displacements, degeneracies, H_matrices = read_hr(hr_path)
    unitcell_vectors = np.asarray(unitcell_vectors, dtype=float)

    # Build HoppingMatrix list
    matrices = []
    for i, (R_latt, H) in enumerate(zip(displacements, H_matrices)):
        # Convert lattice coordinates to Cartesian: R_cart = n1*a1 + n2*a2 + n3*a3
        R_cart = R_latt[0] * unitcell_vectors[0] + \
                 R_latt[1] * unitcell_vectors[1] + \
                 R_latt[2] * unitcell_vectors[2]
        # Divide by degeneracy weight
        deg = degeneracies[i]
        H_scaled = H / deg
        # Wannier: no overlap (orthogonal basis)
        S = np.eye(norbs, dtype=complex) if np.allclose(R_latt, 0) else \
            np.zeros((norbs, norbs), dtype=complex)

        matrices.append(HoppingMatrix(
            displacement=R_cart,
            H=H_scaled,
            S=S,
        ))

    # Single dummy atom encompassing all Wannier orbitals
    atoms = [Atom(
        index=0,
        coord=np.zeros(3),
        basis='wannier',
        norb=norbs,
        orb_slice=slice(0, norbs),
        species='W',
    )]

    # Build atompos from Wannier centres if available
    if centres_path is not None:
        centres = read_centres(centres_path)
        if len(centres) != norbs:
            raise ValueError(
                f"Number of Wannier centres ({len(centres)}) does not match "
                f"number of orbitals ({norbs}) in _hr.dat"
            )
        # atompos.x[i,j] = -(r_i - r_j)_x, matching the TB code convention
        ax = -(centres[:, 0, None] - centres[None, :, 0])
        ay = -(centres[:, 1, None] - centres[None, :, 1])
        az = -(centres[:, 2, None] - centres[None, :, 2])
        atompos = AtomPos(x=ax, y=ay, z=az)
    else:
        z = np.zeros((norbs, norbs))
        atompos = AtomPos(x=z, y=z, z=z)

    return System(
        atoms=atoms,
        matrices=matrices,
        unitcell_vectors=unitcell_vectors,
        norbs=norbs,
        atompos=atompos,
    )


def build_system_from_tb(tb_path: str,
                         centres_path: str = None,
                         ) -> System:
    """Build a System from a Wannier90 _tb.dat file.

    The _tb.dat file contains lattice vectors (so they need not be specified
    in the YAML config) and position operator matrix elements (parsed and
    stored for later use in response calculations).

    Parameters
    ----------
    tb_path : path to the *_tb.dat file
    centres_path : optional path to *_centres.xyz for Wannier function positions.
                   If provided, builds proper atompos for velocity operators.

    Returns
    -------
    System object ready for band structure and response calculations.
    The Hamiltonian blocks are divided by degeneracy weights and the
    displacement vectors are converted to Cartesian coordinates.
    The position operator matrices are stored in system.wannier_r_matrices
    (list of [r_x, r_y, r_z] per R-vector, degeneracy-divided) and
    system.wannier_r_displacements (lattice-coordinate R-vectors).
    """
    lattice_vectors, norbs, displacements, degeneracies, H_matrices, r_matrices = \
        read_tb(tb_path)

    # Build HoppingMatrix list (identical logic to build_system_from_hr)
    matrices = []
    for i, (R_latt, H) in enumerate(zip(displacements, H_matrices)):
        R_cart = R_latt[0] * lattice_vectors[0] + \
                 R_latt[1] * lattice_vectors[1] + \
                 R_latt[2] * lattice_vectors[2]
        deg = degeneracies[i]
        H_scaled = H / deg
        S = np.eye(norbs, dtype=complex) if np.allclose(R_latt, 0) else \
            np.zeros((norbs, norbs), dtype=complex)

        matrices.append(HoppingMatrix(
            displacement=R_cart,
            H=H_scaled,
            S=S,
        ))

    # Divide position operator matrices by degeneracy weights
    r_matrices_scaled = []
    for i, r_xyz in enumerate(r_matrices):
        deg = degeneracies[i]
        r_matrices_scaled.append([r / deg for r in r_xyz])

    # Single dummy atom encompassing all Wannier orbitals
    atoms = [Atom(
        index=0,
        coord=np.zeros(3),
        basis='wannier',
        norb=norbs,
        orb_slice=slice(0, norbs),
        species='W',
    )]

    # Build atompos from Wannier centres if available
    if centres_path is not None:
        centres = read_centres(centres_path)
        if len(centres) != norbs:
            raise ValueError(
                f"Number of Wannier centres ({len(centres)}) does not match "
                f"number of orbitals ({norbs}) in _tb.dat"
            )
        ax = -(centres[:, 0, None] - centres[None, :, 0])
        ay = -(centres[:, 1, None] - centres[None, :, 1])
        az = -(centres[:, 2, None] - centres[None, :, 2])
        atompos = AtomPos(x=ax, y=ay, z=az)
    else:
        z = np.zeros((norbs, norbs))
        atompos = AtomPos(x=z, y=z, z=z)

    system = System(
        atoms=atoms,
        matrices=matrices,
        unitcell_vectors=lattice_vectors,
        norbs=norbs,
        atompos=atompos,
    )

    # Store position operator data for later use in response calculations
    system.wannier_r_matrices = r_matrices_scaled
    system.wannier_r_displacements = displacements

    # Store Wannier centres for position operator corrections (Eq. 20, arXiv:1804.04030)
    if centres_path is not None:
        system._wannier_centres = centres
    else:
        system._wannier_centres = np.zeros((norbs, 3))

    return system
