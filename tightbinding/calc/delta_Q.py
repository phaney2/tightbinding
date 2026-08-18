r"""DC field-induced change in the quantum geometric tensor delta Q^{ab}_n.

Implements Eq. 40 of revised_formula_sheet_eta.pdf with adiabatic iη:

  dQ^{ab}_n(η) = -Sum_{m!=n} r^{c;a}_{nm} v^b_{mn} / [(w_{nm}+iη) w_{nm}]
                 -Sum_{m!=n} r^c_{nm} D^a_{mn} v^b_{mn} / [(w_{nm}+iη)^2 w_{nm}]
                 -Sum_{m!=n} v^a_{nm} r^{c;b}_{mn} / [w_{nm} (w_{nm}-iη)]
                 -Sum_{m!=n} v^a_{nm} r^c_{mn} D^b_{mn} / [w_{nm} (w_{nm}-iη)^2]
                 -Sum_{m!=n} Sum_{l!=n,m} [ r^c_{nl} v^a_{lm} v^b_{mn}
                                            / ((w_{nl}+iη) w_{lm} w_{nm})
                                          + r^c_{ln} v^a_{nm} v^b_{ml}
                                            / ((w_{nl}-iη) w_{nm} w_{lm}) ]

The iη enters only the first-order state-mixing denominators (from the DC
perturbation).  Denominators from the unperturbed projector derivatives
(v/w terms) remain bare.  The +iη/−iη split preserves Hermiticity of δP_n.

Config keys (under cfg['calc']):
  components:      list of 2-char (a,b) pairs, e.g. ['xz', 'zx'], or 'all'
  field_direction: DC field direction c, e.g. 'x' or ['x', 'z']
  directions:      list of direction chars, only needed when components='all'
  nk, eflist, kT:  standard grid/Fermi parameters
  eta:             adiabatic broadening η (default 0.0)

Notation:
  w_{nm} = E_n - E_m
  r^a_{nm} = -i v^a_{nm} / w_{nm}  (interband position, n != m)
  D^a_{mn} = v^a_{mm} - v^a_{nn}  (velocity difference)
  r^{a;b}_{nm} = generalized derivative via Sipe's sum rule
"""

import warnings

import numpy as np

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel
from .nonlinear_optical import _compute_A_W_k


DEG_THR_DEFAULT = 1e-5
ETA_SOS_DEFAULT = 0.05
_DIR = {'x': 0, 'y': 1, 'z': 2}

# One-time deprecation warning bookkeeping.
_DEG_THR_WARNED = False

# Term-decomposition names.  The subspace formulation carries an extra
# 'T_mix' term from inner three-band sums restricted to the occupied
# manifold (see delta_Q_occ_derivation.tex, II_mix + III_mix).
_TERM_NAMES_BAND = ('T_Sipe_Delta', 'T_Sipe_d2H', 'T_Sipe_3band',
                    'T_Sipe_wannier_corr', 'T_Delta', 'T_3band')
_TERM_NAMES_SUBSPACE = _TERM_NAMES_BAND + ('T_mix',)


def _term_names(dQ_occupied_subspace):
    return (list(_TERM_NAMES_SUBSPACE) if dQ_occupied_subspace
            else list(_TERM_NAMES_BAND))


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
    eta = float(calc.get('eta', 0.0))

    # Near-degeneracy handling: Souza-style Lorentzian regularization
    #   inv_de = w_{nm} / (w_{nm}^2 + eta_sos^2)
    # replaces the old hard-cutoff deg_thr masking.  Bounded at w_{nm}→0
    # and smoothly collapses to 0 there.  Preserves sum-of-sub-terms
    # bookkeeping (no NaN/Inf cancellations).
    eta_sos = float(calc.get('eta_sos', ETA_SOS_DEFAULT))

    # Backward compat: accept deg_thr but ignore it (warn once).
    global _DEG_THR_WARNED
    if 'deg_thr' in calc and not _DEG_THR_WARNED:
        parallel.print_root(
            "  [delta_Q] NOTE: 'deg_thr' is deprecated; using Souza "
            f"regularization with eta_sos={eta_sos} instead."
        )
        _DEG_THR_WARNED = True
    deg_thr = 0.0  # not used anymore
    nef = len(eflist)

    # Whether to include the Wannier-gauge position-operator correction
    # (Eq. 22 + 36 of arXiv:1804.04030).  Default True; set False only
    # for diagnostic comparison against the bare TBA form.
    wannier_r = bool(calc.get('wannier_r', True))

    # Whether to compute the DC response of the occupied-subspace QGT
    # Q_occ^{ab} = Tr[P_occ (∂_a P_occ)(∂_b P_occ)], as derived in
    # delta_Q_occ_derivation.tex.  When True (default), the outer band
    # sums are restricted to (p∈occ, q∈unocc) pairs via a Pauli mask
    # f_p (1-f_q), and one additional 'T_mix' term appears.  When False,
    # the code reverts to the band-resolved Σ_n δQ_n form.
    dQ_occupied_subspace = bool(calc.get('dQ_occupied_subspace', True))

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
        f"  Delta Q: components={ab_pairs}, field_direction={field_dirs}, "
        f"eta={eta}, eta_sos={eta_sos}, wannier_r={wannier_r}, "
        f"dQ_occupied_subspace={dQ_occupied_subspace}"
    )

    b1, b2, _b3 = get_reciprocal_lattice(system.unitcell_vectors)
    # Periodic (endpoint-free) grid: db = b/nk with kc = 0..nk-1 tiles the BZ
    # exactly once, so the 1/(nk1*nk2) weight is an unbiased BZ average.
    # (The old db = b/(nk-1) sampled both zone edges — duplicated boundary
    # lines gave an O(1/nk) systematic error in all BZ integrals.)
    db1 = b1 / nk1
    db2 = b2 / nk2 if nk2 > 1 else np.zeros(3)

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

    # Term names depend on the formulation:
    #   band-resolved: 6 terms (default pre-2026-04-20 behavior).
    #   subspace:      6 terms + 'T_mix' (inner 3-band sum restricted to
    #                  the occupied manifold; see derivation).
    TERM_NAMES = _term_names(dQ_occupied_subspace)

    # Local accumulators
    local_dQ = {}
    local_dQ_terms = {}
    for ab in ab_pairs:
        a, b = ab
        local_dQ.setdefault(a, {})
        local_dQ[a].setdefault(b, {})
        local_dQ_terms.setdefault(a, {})
        local_dQ_terms[a].setdefault(b, {})
        for c in field_dirs:
            local_dQ[a][b][c] = np.zeros(nef, dtype=complex)
            local_dQ_terms[a][b][c] = {t: np.zeros(nef, dtype=complex)
                                       for t in TERM_NAMES}

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for i, tk in enumerate(my_klist):
        if parallel.is_root():
            total_local = len(my_klist)
            done = i + 1
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100 * done / total_local:.0f}%)")

        kpt, kpt_terms = _process_kpoint(
            system, tk, dir_chars, ab_pairs, field_dirs,
            eflist, kT, nef, eta, eta_sos=eta_sos,
            wannier_r=wannier_r,
            dQ_occupied_subspace=dQ_occupied_subspace,
        )

        for ab in ab_pairs:
            a, b = ab
            for c in field_dirs:
                local_dQ[a][b][c] += kpt[a][b][c] * norm
                for t in TERM_NAMES:
                    local_dQ_terms[a][b][c][t] += kpt_terms[a][b][c][t] * norm

    # Reduce across all ranks
    delta_Q_terms = {}
    for ab in ab_pairs:
        a, b = ab
        delta_Q_terms.setdefault(a, {})
        delta_Q_terms[a].setdefault(b, {})
        for c in field_dirs:
            delta_Q[a][b][c] = parallel.reduce_sum_complex_array(
                local_dQ[a][b][c]
            )
            delta_Q_terms[a][b][c] = {}
            for t in TERM_NAMES:
                delta_Q_terms[a][b][c][t] = parallel.reduce_sum_complex_array(
                    local_dQ_terms[a][b][c][t]
                )

    return {'Q_tilde': {}, 'delta_Q': delta_Q, 'delta_Q_terms': delta_Q_terms}


# ---------------------------------------------------------------------------
# Per-k-point processing
# ---------------------------------------------------------------------------

def _process_kpoint(system, k, dir_chars, ab_pairs, field_dirs,
                    eflist, kT, nef, eta, eta_sos=ETA_SOS_DEFAULT,
                    deg_thr=None, wannier_r=True,
                    dQ_occupied_subspace=True):
    """Process a single k-point: diagonalize, build operators, assemble dQ.

    Near-degeneracy handling uses Souza Lorentzian regularization:
      inv_de[n,m] = w_{nm} / (w_{nm}^2 + eta_sos^2)
    which is bounded at w_{nm} -> 0 and reduces to 1/w_{nm} for
    |w_{nm}| >> eta_sos.  The `deg_thr` argument is retained for
    backward compatibility but ignored.

    When ``wannier_r`` is True (default) and the system carries
    ``wannier_r_matrices``, the interband position operator ``rmtx`` and
    its generalized derivative ``dk_rmtx`` are augmented by the Wannier-
    gauge corrections (Eqs. 22 and 36 of arXiv:1804.04030).  The Eq. 36
    contribution is registered as a separate 'wannier_corr' entry of
    ``dk_rmtx_terms`` so that downstream code can decompose it.
    """
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

    # Souza-style Lorentzian regularization of the bare 1/w_{nm}.
    # Mask out only exact zeros (n == m self-pairs) — the regularization
    # keeps all other pairs finite.
    nondeg = np.abs(de) > 1e-12
    inv_de = np.where(nondeg, de / (de ** 2 + eta_sos ** 2), 0.0)

    # Broadened DC-perturbation denominators (Eq. 40 η, independent of
    # the eta_sos regularization of the bare 1/w_{nm}).
    # inv_de_p = 1/(w_{nm} + iη),  inv_de_m = 1/(w_{nm} - iη)
    inv_de_p = np.where(nondeg, 1.0 / (de + 1j * eta), 0.0)
    inv_de_m = np.where(nondeg, 1.0 / (de - 1j * eta), 0.0)

    # Diagonal velocities and Delta_code[d][n,m] = v^d_{nn} - v^d_{mm}
    # Note: PDF's D^a_{mn} = v^a_{mm} - v^a_{nn} = -Delta_code[a][n,m]
    vdiag = {d: np.diag(vmtx[d]).real.copy() for d in dir_chars}
    Delta = {d: vdiag[d][:, None] - vdiag[d][None, :] for d in dir_chars}

    # Position operator: rmtx = r = -i v / w  (bare, geometric quantity)
    rmtx = {}
    for d in dir_chars:
        rmtx[d] = -1j * vmtx[d] * inv_de

    # Generalized derivative: dk_rmtx[d1][d2] = r^{d1;d2}  (bare)
    # Also store sub-terms for term decomposition
    dk_rmtx = {}
    dk_rmtx_terms = {}
    zero_mat = np.zeros_like(vmtx[dir_chars[0]])
    for d1 in dir_chars:
        dk_rmtx[d1] = {}
        dk_rmtx_terms[d1] = {}
        for d2 in dir_chars:
            dk_rmtx[d1][d2], dk_rmtx_terms[d1][d2] = _compute_dk_rmtx(
                vmtx[d1], vmtx[d2], vvmtx[d1 + d2],
                Delta[d1], Delta[d2], inv_de,
                return_terms=True,
            )
            # Pre-populate so assembly is branch-free regardless of flag
            # or whether the system carries wannier_r_matrices.
            dk_rmtx_terms[d1][d2]['wannier_corr'] = zero_mat.copy()

    # --- Wannier-gauge position-operator corrections (arXiv:1804.04030) ---
    # Mirror the chi2 implementation in nonlinear_optical.py: the bare
    # r = -i v/w (TBA form) is corrected by the off-diagonal Berry
    # connection a^(H)_{nm} = A^(H)_{nm} (Eq. 22), and the bare Sipe
    # dk_rmtx acquires a three-part correction (Eq. 36).
    if wannier_r:
        A_W, dA_W = _compute_A_W_k(system, k, dir_chars)
    else:
        A_W, dA_W = None, None

    if A_W is not None:
        # Rotate Wannier-gauge connection into the Hamiltonian gauge.
        A_H = {d: psi.conj().T @ A_W[d] @ psi for d in dir_chars}

        # Off-diagonal part a_{nm} = A^(H)_{nm} for n != m.
        a_H = {}
        for d in dir_chars:
            a_H[d] = A_H[d].copy()
            np.fill_diagonal(a_H[d], 0.0)

        # Save pre-correction (TBA-only) rmtx for use inside the Eq. 36
        # commutator term, then apply Eq. 22 to the stored rmtx.
        r_bar = {d: rmtx[d].copy() for d in dir_chars}
        for d in dir_chars:
            rmtx[d] = rmtx[d] + a_H[d]

        # Rotate dA_W and pull out the diagonal Berry connection
        # xi_nn^d = A^(H)_{nn,d}.real.
        dA_H = {}
        for d1 in dir_chars:
            dA_H[d1] = {}
            for d2 in dir_chars:
                dA_H[d1][d2] = psi.conj().T @ dA_W[d1][d2] @ psi
        xi_diag = {d: np.diag(A_H[d]).real.copy() for d in dir_chars}

        # Eq. 36 three-part correction to the generalized derivative.
        for d1 in dir_chars:
            for d2 in dir_chars:
                # Part A: ordinary k-derivative of a^(H) (diagonal zeroed).
                corr = dA_H[d1][d2].copy()
                np.fill_diagonal(corr, 0.0)

                # Part B: connection ("covariant") commutator with r_bar.
                corr = corr + r_bar[d2] @ a_H[d1] - a_H[d1] @ r_bar[d2]

                # Part C: diagonal Berry-connection difference.
                xi_diff = xi_diag[d2][:, None] - xi_diag[d2][None, :]
                corr = corr + 1j * xi_diff * a_H[d1]

                np.fill_diagonal(corr, 0.0)

                dk_rmtx[d1][d2] = dk_rmtx[d1][d2] + corr
                dk_rmtx_terms[d1][d2]['wannier_corr'] = corr

    TERM_NAMES = _term_names(dQ_occupied_subspace)

    # Initialize result containers.
    result = {}
    result_terms = {}
    for ab in ab_pairs:
        a, b = ab
        result.setdefault(a, {})
        result[a].setdefault(b, {})
        result_terms.setdefault(a, {})
        result_terms[a].setdefault(b, {})
        for c in field_dirs:
            result[a][b][c] = np.zeros(nef, dtype=complex)
            result_terms[a][b][c] = {t: np.zeros(nef, dtype=complex)
                                     for t in TERM_NAMES}

    if not dQ_occupied_subspace:
        # Band-resolved path: dQ^{ab}_n(c) per band, summed with Fermi weight.
        dQ_band = {}
        dQ_band_terms = {}
        for ab in ab_pairs:
            a, b = ab
            dQ_band.setdefault(a, {})
            dQ_band[a].setdefault(b, {})
            dQ_band_terms.setdefault(a, {})
            dQ_band_terms[a].setdefault(b, {})
            for c in field_dirs:
                dQ_band[a][b][c], dQ_band_terms[a][b][c] = _assemble_delta_Q(
                    vmtx, rmtx, dk_rmtx, dk_rmtx_terms, Delta,
                    inv_de, inv_de_p, inv_de_m, nondeg, a, b, c,
                )

        for efc in range(nef):
            ef = eflist[efc]
            x_clip = np.clip((ek - ef) / kT, -500, 500)
            f = 1.0 / (1.0 + np.exp(x_clip))
            for ab in ab_pairs:
                a, b = ab
                for c in field_dirs:
                    result[a][b][c][efc] = np.sum(f * dQ_band[a][b][c])
                    for t in TERM_NAMES:
                        result_terms[a][b][c][t][efc] = np.sum(
                            f * dQ_band_terms[a][b][c][t]
                        )
    else:
        # Subspace path: Q_occ = Tr[P_occ ∂P_occ ∂P_occ] response, with
        # outer Pauli mask f_p(1-f_q) + one new 'T_mix' piece per ef.
        # We rebuild pair matrices per ef (cheap since nef is small and
        # the cost is dominated by a few O(N^3) matmuls).
        for efc in range(nef):
            ef = eflist[efc]
            x_clip = np.clip((ek - ef) / kT, -500, 500)
            f = 1.0 / (1.0 + np.exp(x_clip))
            for ab in ab_pairs:
                a, b = ab
                for c in field_dirs:
                    total_abc, terms_abc = _assemble_delta_Q_subspace(
                        vmtx, rmtx, dk_rmtx_terms, Delta,
                        inv_de, inv_de_p, inv_de_m, nondeg, a, b, c, f,
                    )
                    result[a][b][c][efc] = total_abc
                    for t in TERM_NAMES:
                        result_terms[a][b][c][t][efc] = terms_abc[t]

    return result, result_terms


# ---------------------------------------------------------------------------
# Assembly of delta Q per band (corrected formula, all bands at once)
# ---------------------------------------------------------------------------

def _compute_pair_matrices(vmtx, rmtx, dk_rmtx_terms, Delta,
                           inv_de, inv_de_p, inv_de_m, a, b, c):
    r"""Return the 6 (dim, dim) pair integrands that underlie Eq. 40.

    For each term T, pair[T][n,m] is the integrand BEFORE the outer
    contraction.  The band-resolved formula then reduces via
        T[n] = -Σ_m nondeg * pair[T][n,m]
    followed by `Σ_n f[n] T[n]`.  The subspace formula uses the same
    pair matrices with a different outer mask: f_p(1-f_q) on both axes.

    Sign / iη conventions match Eq. 40: Trace-II pieces carry +iη
    (inv_de_p), Trace-III pieces carry -iη (inv_de_m); all projector-
    derivative factors are bare (inv_de, Souza-regularized).
    """
    sipe_ca = dk_rmtx_terms[c][a]
    sipe_cb = dk_rmtx_terms[c][b]

    common_p = vmtx[b].T * inv_de_p * inv_de   # Trace II denominator
    common_m = vmtx[a] * inv_de * inv_de_m     # Trace III denominator

    pair = {}
    for sub, term in (('delta', 'T_Sipe_Delta'),
                      ('d2H', 'T_Sipe_d2H'),
                      ('3band', 'T_Sipe_3band'),
                      ('wannier_corr', 'T_Sipe_wannier_corr')):
        pair[term] = (sipe_ca[sub] * common_p
                      + common_m * sipe_cb[sub].T)

    # Velocity-difference (PDF D^a_{mn} = -Delta_code[a][n,m]).
    pair['T_Delta'] = (
        rmtx[c] * (-Delta[a]) * vmtx[b].T * inv_de_p**2 * inv_de
        + vmtx[a] * rmtx[c].T * (-Delta[b]) * inv_de * inv_de_m**2
    )

    # Explicit three-band (inner sum over all intermediate bands l).
    M_A = (rmtx[c] * inv_de_p) @ (vmtx[a] * inv_de)
    M_B = ((vmtx[b] * inv_de) @ (rmtx[c] * inv_de_p)).T
    pair['T_3band'] = (
        M_A * vmtx[b].T * inv_de
        + (vmtx[a] * inv_de) * M_B
    )

    return pair


def _assemble_delta_Q(vmtx, rmtx, dk_rmtx, dk_rmtx_terms, Delta,
                      inv_de, inv_de_p, inv_de_m, nondeg, a, b, c):
    r"""Compute dQ^{ab}_n(c) for all bands n simultaneously.

    Returns (total, terms_dict) where total is array(dim,) and terms_dict
    has 6 entries (T_Sipe_Delta, T_Sipe_d2H, T_Sipe_3band,
    T_Sipe_wannier_corr, T_Delta, T_3band).

    Implements Eq. 40 of revised_formula_sheet_eta.pdf.  The broadened
    denominators inv_de_p = 1/(w+iη) and inv_de_m = 1/(w-iη) enter only
    the DC perturbation factors; projector-derivative denominators (v/w)
    use bare inv_de.

    Code conventions: rmtx = r (PDF Eq. 6), dk_rmtx[c][a] = r^{c;a} (PDF),
    Delta_code[a][n,m] = v^a_{nn} - v^a_{mm} = -D^a_{mn} (PDF Eq. 2).
    """
    del dk_rmtx  # retained for signature compatibility; unused
    pair = _compute_pair_matrices(vmtx, rmtx, dk_rmtx_terms, Delta,
                                  inv_de, inv_de_p, inv_de_m, a, b, c)

    terms = {name: -np.sum(nondeg * M, axis=1) for name, M in pair.items()}
    total = sum(terms.values())
    return total, terms


def _assemble_delta_Q_subspace(vmtx, rmtx, dk_rmtx_terms, Delta,
                               inv_de, inv_de_p, inv_de_m, nondeg,
                               a, b, c, f):
    r"""Compute δQ^{ab}_occ(c) in the occupied-subspace formulation.

    Implements the expression derived in delta_Q_occ_derivation.tex:
    Trace I vanishes by block-off-diagonality; Traces II and III each
    split into 'direct' + 'dipole 3-band' + 'mix' pieces, with the
    outer (p,q) sum restricted to (occ, unocc) by construction.

    Finite-T Pauli mask used here is f_p(1-f_q) — the natural smooth
    generalization of the T=0 (p∈occ, q∈unocc) indicator and the form
    used throughout the chi^(2) code.  The derivation literally gives
    f_p(f_q - f_p) which differs by a self-Fermi smear f_p(1-f_p) that
    is peaked at the Fermi level and negligible when kT ≪ band gap; we
    use f_p(1-f_q) for consistency with the inter-band convention.

    Returns (total, terms_dict) with 7 scalar entries: the 6 standard
    terms (same structural pair integrands as the band-sum formula,
    just masked differently) plus 'T_mix' for the inner three-band sum
    whose intermediate band is restricted to the occupied manifold.
    """
    pair = _compute_pair_matrices(vmtx, rmtx, dk_rmtx_terms, Delta,
                                  inv_de, inv_de_p, inv_de_m, a, b, c)

    # T_mix depends on f via occupation weighting of the inner n-sum.
    pair['T_mix'] = _compute_T_mix_pair(
        vmtx, rmtx, inv_de, inv_de_p, inv_de_m, a, b, c, f
    )

    mask = f[:, None] * (1.0 - f[None, :])  # outer f_p(1-f_q)

    terms = {name: -np.sum(nondeg * mask * M) for name, M in pair.items()}
    total = sum(terms.values())
    return total, terms


def _compute_T_mix_pair(vmtx, rmtx, inv_de, inv_de_p, inv_de_m, a, b, c, f):
    r"""Pair integrand for the subspace-only T_mix term.

    T_mix = II_mix + III_mix per delta_Q_occ_derivation.tex, i.e., the
    three-band sums whose intermediate index n is restricted to the
    occupied manifold (f_n weight) and whose endpoints (p, q) are the
    outer occ/unocc pair.  Structurally distinct from T_3band, whose
    intermediate m ranges over all bands.

    Vectorized as four matrix products (M_rc @ N_va / N_vb and
    M_va/M_vb @ N_rc_m).  Diagonals of inv_de and of rmtx are zero, so
    n=p and n=q contributions are killed automatically — no explicit
    band masking is needed.

    iη conventions mirror the band-sum code: the 'adiabatic' c_{pn}
    coefficients carry +iη (inv_de_p), their complex conjugates
    c^*_{qn} carry -iη (inv_de_m), and the bare v/ω factors use
    Souza-regularized inv_de.
    """
    f_row = f[None, :]  # broadcast weight over inner index

    # c_{·,n}^{(+iη)}-like factor as an (outer, n) matrix, weighted by f_n.
    M_rc = rmtx[c] * inv_de_p.T * f_row      # elem = f_n r^c[·,n] / (ω_{n,·}+iη)
    # v^a_{·,n}/ω_{n,·} factor, f_n-weighted.
    M_va = vmtx[a] * inv_de.T * f_row        # elem = f_n v^a[·,n] / ω_{n,·}
    M_vb = vmtx[b] * inv_de.T * f_row        # elem = f_n v^b[·,n] / ω_{n,·}

    # Right-side partners (inner→end matrices, not weighted).
    N_va = vmtx[a] * inv_de                  # elem = v^a[n,·] / ω_{n,·}
    N_vb = vmtx[b] * inv_de                  # elem = v^b[n,·] / ω_{n,·}
    N_rc_m = rmtx[c] * inv_de_m              # elem = r^c[n,·] / (ω_{n,·}-iη)

    # II_mix[j,l] = -{(M_rc @ N_va)[j,l] + (M_va @ N_rc_m)[j,l]}
    #              * v^b[l,j]/ω_{lj}
    inner_II = M_rc @ N_va + M_va @ N_rc_m
    mix_II = -inner_II * vmtx[b].T * inv_de.T

    # III_mix[j,l] = + v^a[j,l]/ω_{jl}
    #               * {(M_rc @ N_vb)[l,j] + (M_vb @ N_rc_m)[l,j]}
    inner_III_lj = M_rc @ N_vb + M_vb @ N_rc_m  # indexed as (l, j)
    mix_III = vmtx[a] * inv_de * inner_III_lj.T

    return mix_II + mix_III


# ---------------------------------------------------------------------------
# Generalized derivative of position operator (Sipe sum rule)
# ---------------------------------------------------------------------------

def _compute_dk_rmtx(v_a, v_b, vv_ab, Delta_a, Delta_b, inv_de,
                     return_terms=False):
    """Compute generalized derivative of position operator via Sipe sum rule.

    Borrowed from nonlinear_optical.py.  Computes the covariant derivative
    r^{a;b}_{nm} of the position operator r^a_{nm} = -i*v^a_{nm}/w_{nm}:

      dk_rmtx[n,m] = (i/de[n,m]) * {
          (v_a * Delta_b + v_b * Delta_a) * inv_de - vv_ab
          + Sum_{p!=n,m} (v_a[n,p]*v_b[p,m]/de[p,m] - v_b[n,p]*v_a[p,m]/de[n,p])
      }

    dk_rmtx[a][b] = r^{a;b} in the PDF convention (no sign flip).

    If return_terms=True, returns (result, terms_dict) where terms_dict has:
      'delta': the Delta sub-term (velocity-difference diagonal contribution)
      'd2H':  the second k-derivative of H sub-term
      '3band': the 3-band sum sub-term
    """
    delta_part = (v_a * Delta_b + v_b * Delta_a) * inv_de
    d2H_part = -vv_ab
    full_sum = v_a @ (v_b * inv_de) - (v_b * inv_de) @ v_a
    p_sum = full_sum - Delta_a * v_b * inv_de

    result = 1j * inv_de * (delta_part + d2H_part + p_sum)
    result = np.where(np.isfinite(result), result, 0.0)

    if not return_terms:
        return result

    def _clean(x):
        return np.where(np.isfinite(x), x, 0.0)

    terms = {
        'delta': _clean(1j * inv_de * delta_part),
        'd2H':  _clean(1j * inv_de * d2H_part),
        '3band': _clean(1j * inv_de * p_sum),
    }
    return result, terms
