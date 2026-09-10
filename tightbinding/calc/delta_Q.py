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
  eta:             adiabatic broadening η (default 0.0; band/subspace paths only)
  formulation:     'thermal' (default) | 'subspace' | 'band'

Formulations:
  thermal   δQ^{ab}_T of the thermal density matrix ρ = f(H), valid for
            metals at finite temperature (delta_Q_metal_finite_T.pdf,
            Eq. eq:final).  Sharp occ/un masks are replaced by smooth
            weights W_pq = f_p f_pq², F_pq = f_pq/ω_pq; a new 'T_loop'
            triple sum appears (vanishes for T=0 insulators); the
            insulator 'dipole' and 'mix' three-band terms merge into a
            single T_3band.  All f' (Fermi-surface) terms cancel
            identically, so no ∂f/∂E enters the intrinsic result.
            The adiabatic iη is not used here — finite kT is the
            regulator.  The additive τ-linear RTA piece (Eq. eq:dQtau)
            is always computed too — it is O(N²), i.e. free next to the
            O(N³) intrinsic assembly — and reported separately as
            'delta_Q_tau', **per unit τ** (multiply by your relaxation
            time; extrinsic, diverges in the clean limit, never summed
            into delta_Q).  In insulators it is exponentially zero.
            A 'tau' config key is rejected so nobody expects the output
            to already contain a τ factor.
  subspace  T=0 occupied-projector formulation (delta_Q_occ_derivation),
            with the approximate finite-T Pauli mask f_p(1-f_q).
  band      band-resolved Σ_n f_n δQ_n (Eq. 40 of
            revised_formula_sheet_eta.pdf).
  Legacy key dQ_occupied_subspace: true/false maps to
  'subspace'/'band'; giving both keys is an error.

Sign convention: the metal note derives with H' = +E·r; this engine's
established output convention is H' = -E·r (see CLAUDE.md), so the
thermal and RTA assemblies carry an overall factor of -1 relative to the
note.  Verified: thermal == subspace on insulators at βE_gap >> 1.

Config keys read from cfg['system']:
  wannier_r:       Wannier-gauge position correction; see calc/wannier_gauge.py
                   (this key used to live under cfg['calc'] — it moved so that
                   chi^(2), delta_Q and quantum_metric share one switch)

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
from .wannier_gauge import (
    WANNIER_R_DEFAULT, apply_wannier_correction, compute_A_W_k,
    resolve_wannier_r,
)


DEG_THR_DEFAULT = 1e-5
ETA_SOS_DEFAULT = 0.05
# Below this |w_pq| the divided difference F_pq = f_pq/w_pq is replaced by
# f'(E) at the midpoint energy (its exact w -> 0 limit).
F_DEG_THR = 1e-7
_DIR = {'x': 0, 'y': 1, 'z': 2}

# One-time deprecation warning bookkeeping.
_DEG_THR_WARNED = False
_ETA_THERMAL_WARNED = False

# Term-decomposition names.  The subspace formulation carries an extra
# 'T_mix' term from inner three-band sums restricted to the occupied
# manifold (see delta_Q_occ_derivation.tex, II_mix + III_mix).  The
# thermal formulation merges the dipole/mix three-band split into a
# single T_3band and adds the 'T_loop' triple sum (which vanishes for
# T=0 insulators by occupation algebra).
_TERM_NAMES_BAND = ('T_Sipe_Delta', 'T_Sipe_d2H', 'T_Sipe_3band',
                    'T_Sipe_wannier_corr', 'T_Delta', 'T_3band')
_TERM_NAMES_SUBSPACE = _TERM_NAMES_BAND + ('T_mix',)
_TERM_NAMES_THERMAL = _TERM_NAMES_BAND + ('T_loop',)

_FORMULATIONS = ('thermal', 'subspace', 'band')


def _term_names(formulation):
    if formulation == 'thermal':
        return list(_TERM_NAMES_THERMAL)
    if formulation == 'subspace':
        return list(_TERM_NAMES_SUBSPACE)
    return list(_TERM_NAMES_BAND)


def _resolve_formulation(formulation, dQ_occupied_subspace):
    """Resolve the formulation name from the new and legacy config keys."""
    if formulation is not None and dQ_occupied_subspace is not None:
        raise ValueError(
            "delta_Q: give either 'formulation' or the legacy "
            "'dQ_occupied_subspace', not both"
        )
    if formulation is None:
        if dQ_occupied_subspace is None:
            return 'thermal'
        return 'subspace' if dQ_occupied_subspace else 'band'
    if formulation not in _FORMULATIONS:
        raise ValueError(
            f"delta_Q: unknown formulation '{formulation}'; "
            f"expected one of {_FORMULATIONS}"
        )
    return formulation


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
    # (Eq. 22 + 36 of arXiv:1804.04030).  Read from system.wannier_r by the
    # shared reader that chi^(2) and quantum_metric also use, so the three
    # engines cannot drift apart again.
    wannier_r = resolve_wannier_r(cfg, system, 'delta_Q')

    # Formulation selection: 'thermal' (default; metals at finite T,
    # delta_Q_metal_finite_T.pdf), 'subspace' (T=0 occupied projector,
    # delta_Q_occ_derivation.tex), or 'band' (band-resolved Eq. 40).
    # Legacy dQ_occupied_subspace: true/false maps to subspace/band.
    legacy_key = calc.get('dQ_occupied_subspace')
    if legacy_key is not None:
        legacy_key = bool(legacy_key)
    formulation = _resolve_formulation(calc.get('formulation'), legacy_key)

    # The extrinsic RTA transport piece is always computed on the thermal
    # path (O(N²), negligible next to the O(N³) intrinsic assembly) and
    # reported PER UNIT τ as 'delta_Q_tau'.  Reject a 'tau' key so nobody
    # mistakes the output for having a τ factor already applied.
    if 'tau' in calc:
        raise ValueError(
            "delta_Q: 'tau' is not a config key. The RTA piece is always "
            "computed with the thermal formulation and reported per unit "
            "tau as 'delta_Q_tau' — multiply by your relaxation time."
        )
    with_tau = (formulation == 'thermal')

    # The thermal path has no adiabatic iη — finite kT is the regulator.
    global _ETA_THERMAL_WARNED
    if formulation == 'thermal' and eta != 0.0 and not _ETA_THERMAL_WARNED:
        parallel.print_root(
            "  [delta_Q] NOTE: 'eta' is ignored by the thermal "
            "formulation (finite kT is the regulator)."
        )
        _ETA_THERMAL_WARNED = True

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
        f"formulation={formulation}"
        + (" (+ delta_Q_tau per unit tau)" if with_tau else "")
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
    #   band:     6 terms (band-resolved Eq. 40).
    #   subspace: 6 terms + 'T_mix' (inner 3-band sum restricted to the
    #             occupied manifold; see derivation).
    #   thermal:  6 terms + 'T_loop' (finite-T triple sum; the merged
    #             three-band sums live under 'T_3band').
    TERM_NAMES = _term_names(formulation)

    # Local accumulators
    local_dQ = {}
    local_dQ_terms = {}
    local_dQ_tau = {} if with_tau else None
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
        if local_dQ_tau is not None:
            local_dQ_tau.setdefault(a, {})
            local_dQ_tau[a].setdefault(b, {})
            for c in field_dirs:
                local_dQ_tau[a][b][c] = np.zeros(nef, dtype=complex)

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for i, tk in enumerate(my_klist):
        if parallel.is_root():
            total_local = len(my_klist)
            done = i + 1
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100 * done / total_local:.0f}%)")

        kpt, kpt_terms, kpt_tau = _process_kpoint(
            system, tk, dir_chars, ab_pairs, field_dirs,
            eflist, kT, nef, eta, eta_sos=eta_sos,
            wannier_r=wannier_r,
            formulation=formulation,
        )

        for ab in ab_pairs:
            a, b = ab
            for c in field_dirs:
                local_dQ[a][b][c] += kpt[a][b][c] * norm
                for t in TERM_NAMES:
                    local_dQ_terms[a][b][c][t] += kpt_terms[a][b][c][t] * norm
                if local_dQ_tau is not None:
                    local_dQ_tau[a][b][c] += kpt_tau[a][b][c] * norm

    # Reduce across all ranks
    delta_Q_terms = {}
    delta_Q_tau = {} if local_dQ_tau is not None else None
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
        if delta_Q_tau is not None:
            delta_Q_tau.setdefault(a, {})
            delta_Q_tau[a].setdefault(b, {})
            for c in field_dirs:
                delta_Q_tau[a][b][c] = parallel.reduce_sum_complex_array(
                    local_dQ_tau[a][b][c]
                )

    result = {'Q_tilde': {}, 'delta_Q': delta_Q,
              'delta_Q_terms': delta_Q_terms}
    if delta_Q_tau is not None:
        result['delta_Q_tau'] = delta_Q_tau
    return result


# ---------------------------------------------------------------------------
# Per-k-point processing
# ---------------------------------------------------------------------------

def _process_kpoint(system, k, dir_chars, ab_pairs, field_dirs,
                    eflist, kT, nef, eta, eta_sos=ETA_SOS_DEFAULT,
                    deg_thr=None, wannier_r=WANNIER_R_DEFAULT,
                    dQ_occupied_subspace=None, formulation=None):
    """Process a single k-point: diagonalize, build operators, assemble dQ.

    Returns (result, result_terms, result_tau); result_tau holds the RTA
    transport piece PER UNIT tau on the thermal path, None otherwise.

    Near-degeneracy handling uses Souza Lorentzian regularization:
      inv_de[n,m] = w_{nm} / (w_{nm}^2 + eta_sos^2)
    which is bounded at w_{nm} -> 0 and reduces to 1/w_{nm} for
    |w_{nm}| >> eta_sos.  The `deg_thr` argument is retained for
    backward compatibility but ignored.

    `formulation` selects the assembly ('thermal'/'subspace'/'band');
    the legacy boolean `dQ_occupied_subspace` maps to subspace/band.
    When neither is given the default is 'thermal'.

    When ``wannier_r`` is True and the system carries
    ``wannier_r_matrices``, the interband position operator ``rmtx`` and
    its generalized derivative ``dk_rmtx`` are augmented by the Wannier-
    gauge corrections (Eqs. 22 and 36 of arXiv:1804.04030).  The Eq. 36
    contribution is registered as a separate 'wannier_corr' entry of
    ``dk_rmtx_terms`` so that downstream code can decompose it.
    """
    formulation = _resolve_formulation(formulation, dQ_occupied_subspace)
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

    # --- Wannier-gauge position correction (calc/wannier_gauge.py) ---
    # Eq. 22 on r, and the covariant-derivative correction on r^{a;b}, which
    # is registered as the 'wannier_corr' Sipe sub-term so the sub-terms still
    # sum exactly to dk_rmtx.  Both are (None-guarded) no-ops when the flag is
    # off or the system carries no position matrices.  The bare vmtx is kept:
    # its only uses from here on are band-energy identities (Delta, the RTA
    # inverse-mass sum rule); every interband position factor in the
    # assemblies is built from rmtx.
    A_W, dA_W = compute_A_W_k(system, k, dir_chars, enabled=wannier_r)
    rmtx, corr, _ = apply_wannier_correction(A_W, dA_W, psi, rmtx, dir_chars)
    if corr is not None:
        for d1 in dir_chars:
            for d2 in dir_chars:
                dk_rmtx[d1][d2] = dk_rmtx[d1][d2] + corr[d1][d2]
                dk_rmtx_terms[d1][d2]['wannier_corr'] = corr[d1][d2]

    TERM_NAMES = _term_names(formulation)

    # Initialize result containers.
    result = {}
    result_terms = {}
    result_tau = {} if formulation == 'thermal' else None
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
        if result_tau is not None:
            result_tau.setdefault(a, {})
            result_tau[a].setdefault(b, {})
            for c in field_dirs:
                result_tau[a][b][c] = np.zeros(nef, dtype=complex)

    if formulation == 'band':
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
    elif formulation == 'subspace':
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

    else:
        # Thermal path: δQ^{ab}_T of ρ = f(H) (metal note Eq. eq:final),
        # ×(-1) for the engine's H' = -E·r convention.  All occupation
        # weights are per-ef; the operator build above is shared.
        for efc in range(nef):
            ef = eflist[efc]
            f, fp, fpp, fd, F = _thermal_tables(ek, de, ef, kT)
            for ab in ab_pairs:
                a, b = ab
                for c in field_dirs:
                    total_abc, terms_abc = _assemble_delta_Q_thermal(
                        rmtx, dk_rmtx_terms, Delta, f, fd, F, a, b, c,
                    )
                    result[a][b][c][efc] = total_abc
                    for t in TERM_NAMES:
                        result_terms[a][b][c][t][efc] = terms_abc[t]
                    result_tau[a][b][c][efc] = _assemble_delta_Q_rta(
                        vmtx, rmtx, vvmtx, vdiag, inv_de,
                        f, fp, fpp, fd, a, b, c,
                    )

    return result, result_terms, result_tau


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
    (inv_de_p), Trace-III pieces carry -iη (inv_de_m).  The projector-
    derivative factors v/ω are written as i r with the FULL interband r
    (rmtx, Wannier-corrected when the flag is on): for the bare part this
    is exactly v·inv_de (Souza-regularized), and the smooth a^(H) part is
    added unregularized, as in the thermal assembly.  `vmtx` is unused
    (kept in the signature); Delta carries the diagonal velocities.
    """
    del vmtx
    sipe_ca = dk_rmtx_terms[c][a]
    sipe_cb = dk_rmtx_terms[c][b]

    # i r^a = v^a inv_de for the bare operator; transposed partner uses the
    # antisymmetry of inv_de: v^b.T inv_de = -(v^b inv_de).T.
    V_a = 1j * rmtx[a]
    V_b = 1j * rmtx[b]
    Vt_b = -V_b.T

    common_p = Vt_b * inv_de_p                 # Trace II denominator
    common_m = V_a * inv_de_m                  # Trace III denominator

    pair = {}
    for sub, term in (('delta', 'T_Sipe_Delta'),
                      ('d2H', 'T_Sipe_d2H'),
                      ('3band', 'T_Sipe_3band'),
                      ('wannier_corr', 'T_Sipe_wannier_corr')):
        pair[term] = (sipe_ca[sub] * common_p
                      + common_m * sipe_cb[sub].T)

    # Velocity-difference (PDF D^a_{mn} = -Delta_code[a][n,m]).
    pair['T_Delta'] = (
        rmtx[c] * (-Delta[a]) * Vt_b * inv_de_p**2
        + V_a * rmtx[c].T * (-Delta[b]) * inv_de_m**2
    )

    # Explicit three-band (inner sum over all intermediate bands l).
    M_A = (rmtx[c] * inv_de_p) @ V_a
    M_B = (V_b @ (rmtx[c] * inv_de_p)).T
    pair['T_3band'] = (
        M_A * Vt_b
        + V_a * M_B
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
    the DC perturbation factors; projector-derivative factors (v/w) are
    i r with the full interband r (bare part: Souza inv_de).

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
    c^*_{qn} carry -iη (inv_de_m).  The v/ω factors are i r with the FULL
    interband r (see `_compute_pair_matrices`); `vmtx` is unused.
    """
    del vmtx
    f_row = f[None, :]  # broadcast weight over inner index

    # i r^a == v^a inv_de (bare part exactly); v^a inv_de.T == -(i r^a).
    R_a = 1j * rmtx[a]
    R_b = 1j * rmtx[b]

    # c_{·,n}^{(+iη)}-like factor as an (outer, n) matrix, weighted by f_n.
    M_rc = rmtx[c] * inv_de_p.T * f_row      # elem = f_n r^c[·,n] / (ω_{n,·}+iη)
    # v^a_{·,n}/ω_{n,·} factor, f_n-weighted.
    M_va = -R_a * f_row                      # elem = f_n v^a[·,n] / ω_{n,·}
    M_vb = -R_b * f_row                      # elem = f_n v^b[·,n] / ω_{n,·}

    # Right-side partners (inner→end matrices, not weighted).
    N_va = R_a                               # elem = v^a[n,·] / ω_{n,·}
    N_vb = R_b                               # elem = v^b[n,·] / ω_{n,·}
    N_rc_m = rmtx[c] * inv_de_m              # elem = r^c[n,·] / (ω_{n,·}-iη)

    # II_mix[j,l] = -{(M_rc @ N_va)[j,l] + (M_va @ N_rc_m)[j,l]}
    #              * v^b[l,j]/ω_{lj}
    inner_II = M_rc @ N_va + M_va @ N_rc_m
    mix_II = -inner_II * R_b.T

    # III_mix[j,l] = + v^a[j,l]/ω_{jl}
    #               * {(M_rc @ N_vb)[l,j] + (M_vb @ N_rc_m)[l,j]}
    inner_III_lj = M_rc @ N_vb + M_vb @ N_rc_m  # indexed as (l, j)
    mix_III = R_a * inner_III_lj.T

    return mix_II + mix_III


# ---------------------------------------------------------------------------
# Thermal (metallic, finite-T) formulation — delta_Q_metal_finite_T.pdf
# ---------------------------------------------------------------------------

def _thermal_tables(ek, de, ef, kT):
    r"""Occupation tables for the thermal formulation at one (k, ef).

    Returns (f, fp, fpp, fd, F):
      f    Fermi factors f(E_n)
      fp   f'(E_n) = -f(1-f)/kT
      fpp  f''(E_n) = f(1-f)(1-2f)/kT^2
      fd   fd[p,q] = f_p - f_q
      F    divided difference F[p,q] = (f_p - f_q)/w_pq — smooth and
           symmetric; replaced by its exact w -> 0 limit f'((E_p+E_q)/2)
           when |w_pq| < F_DEG_THR, so it is regular at degeneracies.
    """
    x = np.clip((ek - ef) / kT, -500, 500)
    f = 1.0 / (1.0 + np.exp(x))
    fp = -f * (1.0 - f) / kT
    fpp = f * (1.0 - f) * (1.0 - 2.0 * f) / kT ** 2

    fd = f[:, None] - f[None, :]

    xm = np.clip((0.5 * (ek[:, None] + ek[None, :]) - ef) / kT, -500, 500)
    fm = 1.0 / (1.0 + np.exp(xm))
    fpm = -fm * (1.0 - fm) / kT

    big = np.abs(de) > F_DEG_THR
    F = np.where(big, fd / np.where(big, de, 1.0), fpm)
    return f, fp, fpp, fd, F


def _assemble_delta_Q_thermal(rmtx, dk_rmtx_terms, Delta, f, fd, F, a, b, c):
    r"""Compute δQ^{ab}_T(c) for the thermal density matrix ρ = f(H).

    Implements Eq. eq:final of delta_Q_metal_finite_T.pdf with the
    regular-weight rewrite W_pq/w_pq = f_p f_pq F_pq and
    W_pq/w_pq^2 = f_p F_pq^2 (no bare 1/w in any occupation weight; the
    only 1/w factors live inside rmtx / dk_rmtx, Souza-regularized as
    everywhere else).  No occupation restrictions anywhere — the smooth
    weights do the Pauli blocking, and the zero diagonals of rmtx and
    F*rmtx[c] make the p != q / l != p,q restrictions automatic, so no
    explicit nondeg masking is needed.

    Term decomposition: the four T_Sipe_* names split the r^{c;a}/r^{c;b}
    bracket by dk_rmtx sub-term (as in the other formulations), T_Delta
    is the Δ bracket, T_3band the merged three-band sums (insulator
    limit: old T_3band + T_mix), T_loop the finite-T triple sum
    (vanishes at T=0 by occupation algebra).

    The note derives with H' = +E·r; the engine convention is H' = -E·r,
    hence the overall factor of -1 applied to every term.
    """
    mask1 = f[:, None] * fd          # f_p f_pq
    w_sipe = mask1 * F               # W_pq / w_pq
    w_delta = f[:, None] * F ** 2    # W_pq / w_pq^2

    note = {}

    # Line 1, Sipe bracket: i Σ (W/w) [r^a_pq r^{c;b}_qp - r^{c;a}_pq r^b_qp]
    for sub, name in (('delta', 'T_Sipe_Delta'),
                      ('d2H', 'T_Sipe_d2H'),
                      ('3band', 'T_Sipe_3band'),
                      ('wannier_corr', 'T_Sipe_wannier_corr')):
        s_ca = dk_rmtx_terms[c][a][sub]
        s_cb = dk_rmtx_terms[c][b][sub]
        note[name] = 1j * np.sum(
            w_sipe * (rmtx[a] * s_cb.T - s_ca * rmtx[b].T)
        )

    # Line 1, Δ bracket: i Σ (W/w²) [Δ^a r^c_pq r^b_qp - Δ^b r^a_pq r^c_qp]
    note['T_Delta'] = 1j * np.sum(
        w_delta * (Delta[a] * rmtx[c] * rmtx[b].T
                   - Delta[b] * rmtx[a] * rmtx[c].T)
    )

    # Line 2, loop term: -Σ_{pql distinct} F_pq f_ql f_lp r^c_pq r^a_ql r^b_lp
    Frc = F * rmtx[c]
    note['T_loop'] = -np.trace(Frc @ (fd * rmtx[a]) @ (fd * rmtx[b]))

    # Lines 3+4, merged three-band sums via commutators K_x = [r^x, F∘r^c]:
    #   -Σ f_p f_pq K_a[p,q] r^b_qp + Σ f_p f_pq r^a_pq K_b[q,p]
    K_a = rmtx[a] @ Frc - Frc @ rmtx[a]
    K_b = rmtx[b] @ Frc - Frc @ rmtx[b]
    note['T_3band'] = (-np.sum(mask1 * K_a * rmtx[b].T)
                       + np.sum(mask1 * rmtx[a] * K_b.T))

    # Engine sign convention (H' = -E·r) flips the note's H' = +E·r result.
    terms = {name: -val for name, val in note.items()}
    total = sum(terms.values())
    return total, terms


def _assemble_delta_Q_rta(vmtx, rmtx, vvmtx, vdiag, inv_de,
                          f, fp, fpp, fd, a, b, c):
    r"""RTA transport piece δQ^{ab}_τ / τ (note Eq. eq:dQtau, PER UNIT τ).

    The shifted-Fermi-sea response: δρ_pp = τ f'_p v^c_pp ≡ τ D_p
    inserted into the three traces; the exact linearity in τ is why the
    coefficient is reported and τ never enters the engine.  Purely
    two-band, O(N²); needs the inverse-mass tensor via the standard sum
    rule (note Eq. eq:mass) from the already-carried w^{ab} matrix
    elements.  Reported per unit field, additive to (but never summed
    into) the intrinsic result.  Carries the same overall -1
    engine-convention factor as the intrinsic thermal assembly.
    """
    D = fp * vdiag[c]

    # Inverse mass M^{xc}_p = w^{xc}_pp + 2 Re Σ_l v^x_pl v^c_lp / w_pl
    def mass(x):
        return (np.diag(vvmtx[x + c]).real
                + 2.0 * np.real(
                    np.sum(vmtx[x] * vmtx[c].T * inv_de, axis=1)))

    dDa = fpp * vdiag[a] * vdiag[c] + fp * mass(a)
    dDb = fpp * vdiag[b] * vdiag[c] + fp * mass(b)

    # Off-diagonal: Σ_{p≠q} [D_p f_pq² + 2 f_p f_pq (D_p - D_q)] r^a_pq r^b_qp
    Dd = D[:, None] - D[None, :]
    W_off = D[:, None] * fd ** 2 + 2.0 * f[:, None] * fd * Dd
    off = np.sum(W_off * rmtx[a] * rmtx[b].T)

    # Diagonal (Fermi-surface): Σ_p D_p f'² v^a v^b + f f' (v^b ∂aD + v^a ∂bD)
    diag = np.sum(D * fp ** 2 * vdiag[a] * vdiag[b]
                  + f * fp * (vdiag[b] * dDa + vdiag[a] * dDb))

    return -(off + diag)


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
