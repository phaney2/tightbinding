"""DC field-induced change in the quantum geometric tensor delta Q^{ab}_n.

Implements the corrected projector formula (Section 3 of corrected notes):

  dQ^{ab}_n = Sum_{m!=n} [r^{c;a}_{nm} r^b_{mn} + r^a_{nm} r^{c;b}_{mn}] / w_{nm}
            + Sum_{m!=n} [r^c_{nm} r^b_{mn} D^a_{mn} + r^a_{nm} r^c_{mn} D^b_{mn}] / w^2_{nm}
            - Sum_{m!=n} Sum_{l!=n,m} [v^c_{nl} v^a_{lm} v^b_{mn} / (w^2_{nl} w_{lm} w_{nm})
                                      + v^a_{nm} v^c_{ln} v^b_{ml} / (w_{nm} w^2_{nl} w_{lm})]

where c is the DC field direction and all generalized derivatives r^{c;a}_{nm}
are evaluated via Sipe's sum rule.

The output is Sum_n f_n * dQ^{ab}_n, summed over bands with Fermi occupation
and accumulated over the k-grid.

Config keys (under cfg['calc']):
  components:      list of 2-char (a,b) pairs, e.g. ['xz', 'zx'], or 'all'
  field_direction: DC field direction c, e.g. 'x' or ['x', 'z']
  directions:      list of direction chars, only needed when components='all'
  nk, eflist, kT:  standard grid/Fermi parameters

Notation:
  w_{nm} = E_n - E_m
  r^a_{nm} = i v^a_{nm} / w_{nm}  (interband position, n != m)
  D^a_{mn} = v^a_{mm} - v^a_{nn}  (velocity difference)
  r^{a;b}_{nm} = generalized derivative via Sipe's sum rule
"""

import warnings

import numpy as np

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel


DEG_THR = 1e-5
_DIR = {'x': 0, 'y': 1, 'z': 2}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_delta_Q(system: System, cfg: dict) -> dict:
    """Compute DC field-induced change in the quantum geometric tensor.

    Parameters
    ----------
    system : System with filled Hamiltonian matrices
    cfg : full config dict; reads from cfg['calc']

    Returns
    -------
    dict with keys 'Q_tilde' (empty) and 'delta_Q'.
      delta_Q[a][b][c] -> array(nef,)  where (a,b) are metric indices,
                                        c is DC field direction
    """
    calc = cfg['calc']
    nk1, nk2 = calc['nk']
    eflist = np.asarray(calc['eflist'], dtype=float)
    kT = float(calc['kT'])
    nef = len(eflist)

    # Parse field direction(s)
    fd = calc.get('field_direction', calc.get('directions', ['x']))
    if isinstance(fd, str):
        field_dirs = [fd]
    else:
        field_dirs = list(fd)

    # Parse (a,b) components
    comp_cfg = calc.get('components', 'all')
    if comp_cfg == 'all':
        # Use 'directions' key to determine which (a,b) pairs
        dirs_cfg = calc.get('directions', field_dirs)
        if isinstance(dirs_cfg, str):
            dirs_cfg = [dirs_cfg]
        ab_pairs = [d1 + d2 for d1 in dirs_cfg for d2 in dirs_cfg]
    else:
        ab_pairs = list(comp_cfg)

    # Collect all unique direction chars needed
    dir_chars = sorted(set(
        c for ab in ab_pairs for c in ab
    ) | set(field_dirs))

    parallel.print_root(
        f"  Delta Q: components={ab_pairs}, field_direction={field_dirs}"
    )

    b1, b2, _b3 = get_reciprocal_lattice(system.unitcell_vectors)
    db1 = b1 / (nk1 - 1)
    db2 = b2 / (nk2 - 1) if nk2 > 1 else np.zeros(3)

    # Initialize output: delta_Q[a][b][c] -> array(nef,)
    delta_Q = {}
    for ab in ab_pairs:
        a, b = ab
        delta_Q.setdefault(a, {})
        delta_Q[a].setdefault(b, {})
        for c in field_dirs:
            delta_Q[a][b][c] = np.zeros(nef, dtype=complex)

    # Build k-point list
    k_list = []
    for kc1 in range(nk1):
        for kc2 in range(nk2):
            tk = -b1 / 2 - b2 / 2 + db1 * kc1 + db2 * kc2
            k_list.append(tk)

    total_jobs = len(k_list)
    norm = 1.0 / (nk1 * nk2)

    parallel.print_root(
        f"  Delta Q: {total_jobs} k-points on {parallel.size} rank(s)"
    )

    my_indices, my_klist = parallel.scatter_work(k_list)

    # Local accumulators
    local_dQ = {}
    for ab in ab_pairs:
        a, b = ab
        local_dQ.setdefault(a, {})
        local_dQ[a].setdefault(b, {})
        for c in field_dirs:
            local_dQ[a][b][c] = np.zeros(nef, dtype=complex)

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for i, tk in enumerate(my_klist):
        if parallel.is_root():
            total_local = len(my_klist)
            done = i + 1
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100 * done / total_local:.0f}%)")

        kpt = _process_kpoint(
            system, tk, dir_chars, ab_pairs, field_dirs,
            eflist, kT, nef,
        )

        for ab in ab_pairs:
            a, b = ab
            for c in field_dirs:
                local_dQ[a][b][c] += kpt[a][b][c] * norm

    # Reduce across all ranks
    for ab in ab_pairs:
        a, b = ab
        for c in field_dirs:
            delta_Q[a][b][c] = parallel.reduce_sum_complex_array(
                local_dQ[a][b][c]
            )

    return {'Q_tilde': {}, 'delta_Q': delta_Q}


# ---------------------------------------------------------------------------
# Per-k-point processing
# ---------------------------------------------------------------------------

def _process_kpoint(system, k, dir_chars, ab_pairs, field_dirs,
                    eflist, kT, nef):
    """Process a single k-point: diagonalize, build operators, assemble dQ."""

    # Diagonalize with 2nd-order derivatives (needed for Sipe sum rule)
    H, S, vtb = get_H_v(system, k, order=2)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    dim = len(ek)

    # Velocity matrices in eigenbasis: v^d_{nm}
    vmtx = {}
    for d in dir_chars:
        vmtx[d] = psi.conj().T @ vtb[d] @ psi

    # Second-derivative matrices: w^{d1 d2}_{nm}
    vvmtx = {}
    for d1 in dir_chars:
        for d2 in dir_chars:
            vvmtx[d1 + d2] = psi.conj().T @ vtb[d1 + d2] @ psi

    # Energy differences: de[n,m] = E_n - E_m = w_{nm}
    de = ek[:, None] - ek[None, :]
    nondeg = np.abs(de) >= DEG_THR
    de_safe = np.where(nondeg, de, 1.0)
    inv_de = np.where(nondeg, 1.0 / de_safe, 0.0)
    inv_de2 = inv_de ** 2

    # Diagonal velocities and Delta_code[d][n,m] = v^d_{nn} - v^d_{mm}
    # Note: PDF's D^a_{mn} = v^a_{mm} - v^a_{nn} = -Delta_code[a][n,m]
    vdiag = {d: np.diag(vmtx[d]).real.copy() for d in dir_chars}
    Delta = {d: vdiag[d][:, None] - vdiag[d][None, :] for d in dir_chars}

    # Position operator (chi convention): rmtx = -i v / w
    rmtx = {}
    for d in dir_chars:
        rmtx[d] = -1j * vmtx[d] * inv_de

    # Generalized derivative (chi convention): dk_rmtx[d1][d2]
    dk_rmtx = {}
    for d1 in dir_chars:
        dk_rmtx[d1] = {}
        for d2 in dir_chars:
            dk_rmtx[d1][d2] = _compute_dk_rmtx(
                vmtx[d1], vmtx[d2], vvmtx[d1 + d2],
                Delta[d1], Delta[d2], inv_de,
            )

    # Compute dQ^{ab}_n for requested (a,b) pairs and field directions
    dQ_band = {}
    for ab in ab_pairs:
        a, b = ab
        dQ_band.setdefault(a, {})
        dQ_band[a].setdefault(b, {})
        for c in field_dirs:
            dQ_band[a][b][c] = _assemble_delta_Q(
                vmtx, rmtx, dk_rmtx, Delta,
                inv_de, inv_de2, nondeg, a, b, c,
            )

    # Sum over bands with Fermi weight for each ef
    result = {}
    for ab in ab_pairs:
        a, b = ab
        result.setdefault(a, {})
        result[a].setdefault(b, {})
        for c in field_dirs:
            result[a][b][c] = np.zeros(nef, dtype=complex)

    for efc in range(nef):
        ef = eflist[efc]
        x = (ek - ef) / kT
        x_clip = np.clip(x, -500, 500)
        f = 1.0 / (1.0 + np.exp(x_clip))
        for ab in ab_pairs:
            a, b = ab
            for c in field_dirs:
                result[a][b][c][efc] = np.sum(f * dQ_band[a][b][c])

    return result


# ---------------------------------------------------------------------------
# Assembly of delta Q per band (corrected formula, all bands at once)
# ---------------------------------------------------------------------------

def _assemble_delta_Q(vmtx, rmtx, dk_rmtx, Delta,
                      inv_de, inv_de2, nondeg, a, b, c):
    r"""Compute dQ^{ab}_n(c) for all bands n simultaneously.

    Returns array of shape (dim,) with dQ for each band.

    Convention note: the chi code uses rmtx = -i*v/w and dk_rmtx is its
    covariant derivative.  The PDF uses r = +i*v/w = -rmtx.  In all three
    terms below, the signs from the r <-> rmtx conversion cancel in pairs.
    For Delta, PDF has D^a_{mn} = v^a_{mm} - v^a_{nn} = -Delta_code[a][n,m].
    """

    # --- Term 1: Sipe (dressed dipole) ---
    T_sipe = np.sum(
        nondeg * (dk_rmtx[c][a] * rmtx[b].T
                  + rmtx[a] * dk_rmtx[c][b].T) * inv_de,
        axis=1,
    )

    # --- Term 2: velocity difference ---
    T_delta = np.sum(
        nondeg * (rmtx[c] * rmtx[b].T * (-Delta[a])
                  + rmtx[a] * rmtx[c].T * (-Delta[b])) * inv_de2,
        axis=1,
    )

    # --- Term 3: three-band ---
    # Vectorized via matrix products.  Diagonal zeros of inv_de
    # automatically exclude l=n (inv_de2[n,n]=0) and l=m (inv_de[m,m]=0).

    # Part A: Sum v^c_{nl}/w^2_{nl} * v^a_{lm}/w_{lm} * v^b_{mn}/w_{nm}
    C2 = vmtx[c] * inv_de2
    A1 = vmtx[a] * inv_de
    part_A = np.sum(
        nondeg * (C2 @ A1) * vmtx[b].T * inv_de,
        axis=1,
    )

    # Part B: Sum v^a_{nm}/w_{nm} * v^c_{ln}/w^2_{nl} * v^b_{ml}/w_{lm}
    D = vmtx[c] * inv_de2
    B1 = vmtx[b].T * inv_de
    part_B = np.sum(
        nondeg * vmtx[a] * inv_de * (D.T @ B1),
        axis=1,
    )

    T_3band = -(part_A + part_B)

    return T_sipe + T_delta + T_3band


# ---------------------------------------------------------------------------
# Generalized derivative of position operator (Sipe sum rule)
# ---------------------------------------------------------------------------

def _compute_dk_rmtx(v_a, v_b, vv_ab, Delta_a, Delta_b, inv_de):
    """Compute generalized derivative of position operator via Sipe sum rule.

    Borrowed from nonlinear_optical.py.  Computes the covariant derivative
    of rmtx_chi = -i*v/de in the chi-code convention:

      dk_rmtx[n,m] = (i/de[n,m]) * {
          (v_a * Delta_b + v_b * Delta_a) * inv_de - vv_ab
          + Sum_{p!=n,m} (v_a[n,p]*v_b[p,m]/de[p,m] - v_b[n,p]*v_a[p,m]/de[n,p])
      }

    The PDF generalized derivative is r^{a;b}_PDF = -dk_rmtx[a][b].
    """
    diag_term = (v_a * Delta_b + v_b * Delta_a) * inv_de - vv_ab
    full_sum = v_a @ (v_b * inv_de) - (v_b * inv_de) @ v_a
    p_sum = full_sum - Delta_a * v_b * inv_de
    result = 1j * inv_de * (diag_term + p_sum)
    result = np.where(np.isfinite(result), result, 0.0)
    return result
