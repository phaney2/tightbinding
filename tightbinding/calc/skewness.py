"""Cumulants of the ground-state polarization: variance and skewness.

The many-body position X = Σ_i x_i of a band insulator has a distribution
whose cumulants per unit cell are BZ averages of projector traces
(arXiv:2409.16358 Eqs. 12-13, P = P_occ(k)):

    <X_a X_b>_c      =  V ∫_k  Re tr[ P (∂_a P)(∂_b P) ]          (the quantum metric)
    <X_a X_b X_c>_c  = -V ∫_k  Im tr[ P (∂_a P)(∂_b ∂_c P) ]

**Sign.**  The paper's Eq. 13 has no minus sign.  Evaluated in this code's
Bloch convention (exp(ik·(R + tau_m - tau_n)), the paper's own Fourier
convention) on the paper's fully dimerized Rice-Mele chain, where the
electron distribution is exactly Bernoulli on {0, 1/2} and the third
cumulant is d^3 q(1-q)(1-2q) = -m/(8 (4+m^2)^{3/2}), the literal Eq. 13 gives
+m/(8 (4+m^2)^{3/2}).  This engine reports the real-space cumulant, i.e.
``skewness = -(paper Eq. 13)``; `examples/test_skewness.py` pins that against
the Bernoulli value and the paper's closed forms.  Negative skewness means
the weight piles up at +x with the tail toward -x.

**Formula.**  Rather than expanding in r and the generalized derivative
r^{b;c} (which brings 1/w between two occupied bands, cancelling only
algebraically and hence spoiled by eta_sos), differentiate [H, P] = 0 in
the band basis, with f_pq = f_p - f_q and only occupied-unoccupied pairs:

    (∂_a P)_pq       = f_pq V^a_pq / (E_p - E_q)
    (∂_b ∂_c P)_pq   = [ f_pq W^{bc}_pq - ([V^b, ∂_c P] + [V^c, ∂_b P])_pq ] / (E_p - E_q)

There is no eta_sos, no adiabatic eta and no kT: the integrand is an
analytic periodic function for an insulator, so the uniform grid converges
exponentially.  O(N^3) per k-point.  Insulators only — the occupied count
must not change across the grid, and the engine raises if it does.

**Wannier input.**  ∂_a -> D_a = ∂_a - i[A^(W)_a, ·] throughout
(`wannier_gauge.covariant_H_derivatives`), the same correction every other
engine applies to r; the output is then independent of the orbital centres
the system was built with.  Two caveats, documented rather than hidden:
(i) with a truncated Wannier basis the connection has curvature F_bc, so
D_b D_c P != D_c D_b P and the BZ integral is not totally symmetric in
(a, b, c); the engine symmetrizes b <-> c and prints the antisymmetric
remainders; (ii) this is the skewness of the position operator projected
onto the Wannier manifold — the exact cumulant needs <0n|r_a r_b|Rm>
matrix elements a _tb.dat file does not carry.

Config::

    calc:
      type: skewness
      directions: [x, y]       # default
      nk: [200, 200]           # 2 or 3 entries
      eflist: [0.0]            # Fermi levels, sharp occupation E < ef

Output (arrays of shape (nef,), real): ``variance[a][b]``,
``skewness[a][b][c]`` (b <-> c symmetrized) and ``skewness_sym[a][b][c]``
(mean over all six index permutations).
"""

import itertools
import warnings

import numpy as np

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel
from .wannier_gauge import (
    compute_A_W_k, covariant_H_derivatives, resolve_wannier_r,
    system_has_wannier_r,
)


_IGNORED_KEYS = ('kT', 'eta', 'eta_sos', 'delta')


def compute_skewness(system: System, cfg: dict) -> dict:
    """BZ-averaged variance and skewness of the occupied-state polarization.

    Returns dict with ``variance[a][b]``, ``skewness[a][b][c]``,
    ``skewness_sym[a][b][c]`` -> real array(nef,), plus ``gap`` (min direct
    gap over the grid per ef) and ``n_occ``.
    """
    calc = cfg['calc']
    nk = [int(n) for n in calc['nk']]
    if len(nk) == 2:
        nk = nk + [1]
    if len(nk) != 3:
        raise ValueError("calc.nk must have 2 or 3 entries")
    eflist = np.asarray(calc['eflist'], dtype=float)
    nef = len(eflist)
    dir_chars = list(calc.get('directions', ['x', 'y']))

    for key in _IGNORED_KEYS:
        if key in calc:
            parallel.print_root(
                f"  [skewness] NOTE: calc.{key} is ignored — the cumulants "
                f"carry no broadening, regularization or temperature.")

    wannier_r = resolve_wannier_r(cfg, system, 'skewness')
    if wannier_r and system_has_wannier_r(system):
        parallel.print_root(
            "  [skewness] NOTE: Wannier input — this is the skewness of the "
            "position operator projected onto the Wannier manifold; the "
            "b<->c antisymmetric part below measures the connection's "
            "curvature.")

    b1, b2, b3 = get_reciprocal_lattice(system.unitcell_vectors)
    bvecs = [b1, b2, b3]
    # Periodic (endpoint-free) grid: db = b/nk, kc = 0..nk-1 tiles the BZ
    # exactly once, so the uniform weight is an unbiased BZ average.
    db = [bvecs[i] / nk[i] if nk[i] > 1 else np.zeros(3) for i in range(3)]
    k0 = -0.5 * (b1 + b2 + b3)

    k_list = []
    for kc1 in range(nk[0]):
        for kc2 in range(nk[1]):
            for kc3 in range(nk[2]):
                k_list.append(k0 + db[0] * kc1 + db[1] * kc2 + db[2] * kc3)
    total_jobs = len(k_list)
    norm = 1.0 / total_jobs

    parallel.print_root(
        f"  Skewness: {total_jobs} k-points on {parallel.size} rank(s), "
        f"directions {dir_chars}, {nef} Fermi level(s) (wannier_r={wannier_r})"
    )

    my_indices, my_klist = parallel.scatter_work(k_list)

    nd = len(dir_chars)
    local_var = np.zeros((nef, nd, nd), dtype=complex)
    local_T = np.zeros((nef, nd, nd, nd), dtype=complex)
    # Per-ef bookkeeping for the insulator check.
    nocc_min = np.full(nef, np.inf)
    nocc_max = np.full(nef, -np.inf)
    gap_min = np.full(nef, np.inf)

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for i, tk in enumerate(my_klist):
        if parallel.is_root():
            total_local = len(my_klist)
            done = i + 1
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100*done/total_local:.0f}%)")

        var_k, T_k, nocc_k, gap_k = _process_kpoint(
            system, tk, dir_chars, eflist, wannier_r)
        local_var += var_k * norm
        local_T += T_k * norm
        nocc_min = np.minimum(nocc_min, nocc_k)
        nocc_max = np.maximum(nocc_max, nocc_k)
        gap_min = np.minimum(gap_min, gap_k)

    var = parallel.reduce_sum_complex_array(local_var)
    T = parallel.reduce_sum_complex_array(local_T)
    nocc_min = np.array([-parallel.reduce_max(-x) for x in nocc_min])
    nocc_max = np.array([parallel.reduce_max(x) for x in nocc_max])
    gap_min = np.array([-parallel.reduce_max(-x) for x in gap_min])

    for efc, ef in enumerate(eflist):
        if nocc_min[efc] != nocc_max[efc]:
            raise ValueError(
                f"skewness: ef = {ef} is inside a band (occupied count varies "
                f"from {int(nocc_min[efc])} to {int(nocc_max[efc])} over the "
                f"grid).  The polarization cumulants exist only for an "
                f"insulator; choose ef in a gap.")
        if nocc_min[efc] == 0:
            raise ValueError(f"skewness: no occupied bands at ef = {ef}")

    # The raw T^{a;bc}: symmetrize b <-> c (exact for point-like orbitals,
    # an F_bc-sized choice for Wannier input), then also over all of (a,b,c)
    # (exact after BZ integration for point-like orbitals).
    T_bc = 0.5 * (T + np.swapaxes(T, 2, 3))
    T_sym = np.zeros_like(T)
    for perm in itertools.permutations(range(3)):
        T_sym += np.transpose(T, (0,) + tuple(p + 1 for p in perm))
    T_sym /= 6.0

    anti_bc = np.max(np.abs(T - T_bc)) if T.size else 0.0
    anti_abc = np.max(np.abs(T_bc - T_sym)) if T.size else 0.0
    scale = max(np.max(np.abs(T)), 1e-300)
    parallel.print_root(
        f"  Skewness: b<->c antisymmetric part {anti_bc:.3e} "
        f"(|T| scale {scale:.3e}); residual (a,b,c) asymmetry after "
        f"b<->c symmetrization {anti_abc:.3e}"
    )
    for efc, ef in enumerate(eflist):
        parallel.print_root(
            f"  Skewness: ef = {ef}: {int(nocc_min[efc])} occupied band(s), "
            f"min direct gap {gap_min[efc]:.4e}")

    variance = {}
    skewness = {}
    skewness_sym = {}
    for ia, a in enumerate(dir_chars):
        variance[a] = {}
        skewness[a] = {}
        skewness_sym[a] = {}
        for ib, b in enumerate(dir_chars):
            variance[a][b] = np.real(var[:, ia, ib]).copy()
            skewness[a][b] = {}
            skewness_sym[a][b] = {}
            for ic, c in enumerate(dir_chars):
                skewness[a][b][c] = -np.imag(T_bc[:, ia, ib, ic]).copy()
                skewness_sym[a][b][c] = -np.imag(T_sym[:, ia, ib, ic]).copy()

    return {'variance': variance, 'skewness': skewness,
            'skewness_sym': skewness_sym,
            'gap': gap_min, 'n_occ': nocc_min.astype(int)}


def _process_kpoint(system, k, dir_chars, eflist, wannier_r):
    """Per-k tr[P ∂_aP ∂_bP] and tr[P ∂_aP ∂_b∂_cP] for every ef.

    Returns (var, T, nocc, gap): var (nef, nd, nd), T (nef, nd, nd, nd)
    complex, nocc and gap (nef,).
    """
    nd = len(dir_chars)
    nef = len(eflist)

    H, S, vtb = get_H_v(system, k, order=2)
    dim = H.shape[0]
    if not np.allclose(S, np.eye(dim), atol=1e-12):
        raise NotImplementedError(
            "skewness: non-orthogonal basis (overlap S != 1) is not supported")
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)

    # Covariant derivatives of H (bare ∂H, ∂∂H when the correction is off or
    # inert), rotated to the band basis.
    A_W, dA_W = compute_A_W_k(system, k, dir_chars, enabled=wannier_r,
                              need_deriv=True)
    V, W = covariant_H_derivatives(A_W, dA_W, H, vtb, dir_chars)
    Vb = [psi.conj().T @ V[a] @ psi for a in dir_chars]
    Wb = [[psi.conj().T @ W[b][c] @ psi for c in dir_chars] for b in dir_chars]

    de = ek[:, None] - ek[None, :]

    var = np.zeros((nef, nd, nd), dtype=complex)
    T = np.zeros((nef, nd, nd, nd), dtype=complex)
    nocc = np.zeros(nef)
    gap = np.zeros(nef)

    for efc, ef in enumerate(eflist):
        f = (ek < ef).astype(float)
        n_occ = int(f.sum())
        nocc[efc] = n_occ
        gap[efc] = (ek[n_occ] - ek[n_occ - 1]) if 0 < n_occ < dim else np.inf
        if n_occ == 0 or n_occ == dim:
            continue

        fpq = f[:, None] - f[None, :]
        mask = fpq != 0                       # occupied-unoccupied pairs only
        inv = np.where(mask, 1.0 / np.where(mask, de, 1.0), 0.0)
        fin = fpq * inv                       # f_pq / (E_p - E_q)

        # (∂_a P)_pq = f_pq V^a_pq / (E_p - E_q)
        D = [fin * Vb[ia] for ia in range(nd)]

        # tr[P X Y] = Σ_n f_n (X Y)_nn = Σ_nm f_n X_nm Y_mn
        for ia in range(nd):
            for ib in range(nd):
                var[efc, ia, ib] = np.sum(f[:, None] * D[ia] * D[ib].T)

        for ib in range(nd):
            for ic in range(nd):
                C = (Vb[ib] @ D[ic] - D[ic] @ Vb[ib]
                     + Vb[ic] @ D[ib] - D[ib] @ Vb[ic])
                DD = fin * Wb[ib][ic] - inv * C     # (∂_b ∂_c P), occ-unocc block
                for ia in range(nd):
                    T[efc, ia, ib, ic] = np.sum(f[:, None] * D[ia] * DD.T)

    return var, T, nocc, gap
