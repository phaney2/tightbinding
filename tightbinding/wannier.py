"""Wannier90 _hr.dat file parser.

Reads a Wannier90 Hamiltonian in the _hr.dat format and builds a System
object compatible with the tight-binding code's Bloch sum machinery.
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
