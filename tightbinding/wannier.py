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


def _symmetrize_position_blocks(displacements, r_matrices):
    """Impose r(-R) = r(R)^dagger on the position blocks, in place.

    Wannier90's ``write_tb`` evaluates the off-diagonal elements with the
    finite-difference formula (Eq. 44 of Wang, Yates, Souza & Vanderbilt,
    PRB 74, 195118), which does not preserve the Hermiticity of the Berry
    connection; ``postw90`` takes the Hermitian part before using it
    (``get_oper.F90``, ``get_AA_R``) and so must we.  Averaging r(R) with
    r(-R)^dagger is exactly Hermitianizing A^(W)(k) at every k.

    Returns the largest |r(R) - r(-R)^dagger| found, as a diagnostic.
    Raises if the R-set is not inversion-symmetric (Wannier90's Wigner-Seitz
    set always is).
    """
    index = {tuple(int(round(x)) for x in R): i for i, R in enumerate(displacements)}
    worst = 0.0
    done = set()
    for R, i in index.items():
        j = index.get(tuple(-x for x in R))
        if j is None:
            raise ValueError(
                f"_tb.dat position blocks: R-vector {R} has no -R partner, so "
                f"r(-R) = r(R)^dagger cannot be imposed"
            )
        if (j, i) in done:
            continue
        done.add((i, j))
        for a in range(3):
            ri, rj = r_matrices[i][a], r_matrices[j][a]
            worst = max(worst, float(np.abs(ri - rj.conj().T).max()))
            sym = 0.5 * (ri + rj.conj().T)
            r_matrices[i][a] = sym
            r_matrices[j][a] = sym.conj().T
    return worst


def build_system_from_tb(tb_path: str,
                         centres_path: str = None,
                         ) -> System:
    """Build a System from a Wannier90 _tb.dat file.

    The _tb.dat file contains lattice vectors (so they need not be specified
    in the YAML config) and position operator matrix elements, stored for the
    Wannier-gauge position correction (see calc/wannier_gauge.py).

    Parameters
    ----------
    tb_path : path to the *_tb.dat file
    centres_path : optional path to *_centres.xyz.  When given, the Wannier
                   centres come from it; otherwise from the band-diagonal of
                   the R=0 position block.  See "Gauge" below for why a
                   centres file is the safer choice.

    Returns
    -------
    System with ``matrices`` (degeneracy-divided), ``atompos`` built from the
    Wannier centres, ``wannier_r_matrices`` (list of [r_x, r_y, r_z] per
    R-vector, degeneracy-divided, Hermitian-paired, R=0 diagonal repaired),
    ``wannier_r_displacements`` (lattice-coordinate R-vectors) and
    ``_wannier_centres`` (the centres used, shape (norbs, 3)).

    Gauge
    -----
    ``atompos`` is always built from the Wannier centres, so `bloch.get_H_k`
    works in the atomic gauge exp(ik·(R + tau_m - tau_n)).  Two reasons:
    (i) with ``wannier_r`` off, r = -i v/w is then the point-like-orbital
    approximation, which is physically sensible, whereas in the lattice
    gauge (atompos = 0) it is missing the whole intra-cell term;
    (ii) `compute_A_W_k` subtracts the centres that atompos encodes, so the
    correction is consistent with H(k) either way — but the diagonal repair
    below is only meaningful if the centres are the true ones.

    Position-block conditioning
    ---------------------------
    Two known defects of Wannier90's ``_tb.dat`` position blocks are handled
    here, with a one-line diagnostic printed for each:

    * Off-diagonal elements are not Hermitian-paired (finite-difference
      formula); they are symmetrized, r(-R) <- r(R)^dagger averaged.
    * The band-diagonal R=0 elements <0n|r|0n> come from a Berry-phase
      log and can be wrapped by a lattice vector, or scrambled outright
      when a centre sits on the branch cut (e.g. an atom at c/2 with a
      single k-point along c).  They are compared with the centres and
      replaced by them where they differ.  The band-diagonal R != 0
      elements of an affected component come from the same log and are
      likely unreliable too; the diagnostic names the component so the
      user can avoid it.  A centres file avoids relying on that diagonal
      at all, which is why passing one is recommended.
    """
    from . import parallel

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

    # Position blocks: divide by degeneracy (same Fourier-sum origin as H,
    # same Wigner-Seitz bookkeeping), then Hermitian-pair them.
    r_matrices_scaled = []
    for i, r_xyz in enumerate(r_matrices):
        deg = degeneracies[i]
        r_matrices_scaled.append([r / deg for r in r_xyz])
    asym = _symmetrize_position_blocks(displacements, r_matrices_scaled)

    # Wannier centres: from the file if given, else the R=0 band-diagonal.
    r0_idx = next(i for i, R in enumerate(displacements) if np.allclose(R, 0))
    diag_centres = np.stack(
        [np.diag(r_matrices_scaled[r0_idx][a]).real for a in range(3)], axis=1)
    if centres_path is not None:
        centres = read_centres(centres_path)
        if len(centres) != norbs:
            raise ValueError(
                f"Number of Wannier centres ({len(centres)}) does not match "
                f"number of orbitals ({norbs}) in _tb.dat"
            )
        source = f"{centres_path}"
    else:
        centres = diag_centres.copy()
        source = "R=0 diagonal of the position blocks (no centres file given)"

    # Repair the R=0 band-diagonal against the centres.
    dev = np.abs(diag_centres - centres)
    bad = dev > 1e-4
    for a in range(3):
        blk = r_matrices_scaled[r0_idx][a]
        blk[np.arange(norbs), np.arange(norbs)] = centres[:, a]

    parallel.print_root(
        f"  [wannier_tb] {norbs} WFs, {len(displacements)} R-vectors; centres from "
        f"{source}; position blocks Hermitian-paired (max asymmetry "
        f"{asym:.2e} A)"
    )
    if bad.any():
        comps = ''.join(c for a, c in enumerate('xyz') if bad[:, a].any())
        wfs = [n + 1 for n in range(norbs) if bad[n].any()]
        parallel.print_root(
            f"  [wannier_tb] WARNING: the R=0 band-diagonal of the position "
            f"blocks disagrees with the centres for WF(s) {wfs} in "
            f"component(s) '{comps}' (max {dev.max():.3f} A) — Berry-phase "
            f"wrapping or branch-cut scramble in Wannier90's <0n|r|0n>.  "
            f"Replaced by the centres.  The band-diagonal R!=0 elements of "
            f"'{comps}' come from the same log and may be unreliable: avoid "
            f"the '{comps}' position operator from this file."
        )

    # Single dummy atom encompassing all Wannier orbitals
    atoms = [Atom(
        index=0,
        coord=np.zeros(3),
        basis='wannier',
        norb=norbs,
        orb_slice=slice(0, norbs),
        species='W',
    )]

    # atompos.x[i,j] = -(r_i - r_j)_x, matching the TB code convention:
    # atomic gauge, always.
    ax = -(centres[:, 0, None] - centres[None, :, 0])
    ay = -(centres[:, 1, None] - centres[None, :, 1])
    az = -(centres[:, 2, None] - centres[None, :, 2])
    atompos = AtomPos(x=ax, y=ay, z=az)

    system = System(
        atoms=atoms,
        matrices=matrices,
        unitcell_vectors=lattice_vectors,
        norbs=norbs,
        atompos=atompos,
    )

    # Store position operator data for the Wannier-gauge correction
    system.wannier_r_matrices = r_matrices_scaled
    system.wannier_r_displacements = displacements
    system._wannier_centres = centres

    return system
