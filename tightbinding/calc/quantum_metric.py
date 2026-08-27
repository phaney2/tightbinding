"""Quantum metric and its DC linear response.

Ported from compute_quantum_metric.m — computes the quantum metric tensor Q
and its intrinsic (dQ) and extrinsic (dQf) linear response over a 2D k-grid.

All three quantities are quadratic forms in the interband position operator,

    Q^{ab}   = Σ_nm  r^a_nm conj(r^b_nm) f_n (1-f_m)

so they take the Wannier-gauge correction ``r -> r + a^(H)`` (Eq. 22 of
arXiv:1804.04030) the same way ``delta_Q`` does, through the same switch —
see calc/wannier_gauge.py.  Before that switch existed this engine was pinned
at the bare TBA form while its own DC response ``delta_Q`` used the corrected
one, which put Q and dQ on opposite sides of the correction.
"""

import warnings

import numpy as np

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel
from .wannier_gauge import (
    WANNIER_R_DEFAULT, compute_A_W_k, offdiag_A_H, resolve_wannier_r,
)


DEG_THR = 1e-5


def compute_quantum_metric(system: System, cfg: dict) -> dict:
    """Compute quantum metric Q, intrinsic dQ, and extrinsic dQf.

    Parameters
    ----------
    system : System with filled Hamiltonian matrices
    cfg : full config dict; reads from cfg['calc']

    Returns
    -------
    dict with keys 'Q', 'dQ', 'dQf', each nested by direction labels.
    Q[d1][d2] -> array(nef,)
    dQ[d1][d2][d3] -> array(nef,)
    dQf[d1][d2][d3] -> array(nef,)
    """
    calc = cfg['calc']
    nk1, nk2 = calc['nk']
    eflist = np.asarray(calc['eflist'], dtype=float)
    kT = float(calc['kT'])
    eta = float(calc['eta'])
    delta = float(calc.get('delta', 0.001))
    nef = len(eflist)

    # Directions to compute (default: x and z for 2D)
    dir_chars = list(calc.get('metric_directions', ['x', 'z']))

    # Wannier-gauge position correction, shared with chi^(2) and delta_Q.
    wannier_r = resolve_wannier_r(cfg, system, 'quantum_metric')

    # Reciprocal lattice vectors
    b1, b2, _b3 = get_reciprocal_lattice(system.unitcell_vectors)

    # Periodic (endpoint-free) grid — see note in delta_Q.py.
    db1 = b1 / nk1
    db2 = b2 / nk2 if nk2 > 1 else np.zeros(3)

    dim = len(system.matrices[0].H)

    # Initialize output
    Q = {}
    dQ = {}
    dQf = {}
    for d1 in dir_chars:
        Q[d1] = {}
        dQ[d1] = {}
        dQf[d1] = {}
        for d2 in dir_chars:
            Q[d1][d2] = np.zeros(nef, dtype=complex)
            dQ[d1][d2] = {}
            dQf[d1][d2] = {}
            for d3 in dir_chars:
                dQ[d1][d2][d3] = np.zeros(nef, dtype=complex)
                dQf[d1][d2][d3] = np.zeros(nef, dtype=complex)

    # Build k-point list — full grid, matching MATLAB kc1=1:nk1, kc2=1:nk2
    k_list = []
    for kc1 in range(nk1):
        for kc2 in range(nk2):
            tk = -b1 / 2 - b2 / 2 + db1 * kc1 + db2 * kc2
            k_list.append(tk)

    total_jobs = len(k_list)
    norm = 1.0 / (nk1 * nk2)

    params = {
        'dim': dim,
        'dir_chars': dir_chars,
        'eflist': eflist,
        'kT': kT,
        'eta': eta,
        'delta': delta,
        'nef': nef,
        'wannier_r': wannier_r,
    }

    parallel.print_root(
        f"  Quantum metric: {total_jobs} k-points on {parallel.size} rank(s) "
        f"(wannier_r={wannier_r})"
    )

    # Scatter k-points across MPI ranks
    my_indices, my_klist = parallel.scatter_work(k_list)

    # Local accumulators
    local_Q = {}
    local_dQ = {}
    local_dQf = {}
    for d1 in dir_chars:
        local_Q[d1] = {}
        local_dQ[d1] = {}
        local_dQf[d1] = {}
        for d2 in dir_chars:
            local_Q[d1][d2] = np.zeros(nef, dtype=complex)
            local_dQ[d1][d2] = {}
            local_dQf[d1][d2] = {}
            for d3 in dir_chars:
                local_dQ[d1][d2][d3] = np.zeros(nef, dtype=complex)
                local_dQf[d1][d2][d3] = np.zeros(nef, dtype=complex)

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for i, tk in enumerate(my_klist):
        if parallel.is_root():
            total_local = len(my_klist)
            done = i + 1
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100*done/total_local:.0f}%)")

        kpt = _process_kpoint(system, tk, params)
        for d1 in dir_chars:
            for d2 in dir_chars:
                local_Q[d1][d2] += kpt['Q'][d1][d2] * norm
                for d3 in dir_chars:
                    local_dQ[d1][d2][d3] += kpt['dQ'][d1][d2][d3] * norm
                    local_dQf[d1][d2][d3] += kpt['dQf'][d1][d2][d3] * norm

    # Reduce across all ranks
    for d1 in dir_chars:
        for d2 in dir_chars:
            Q[d1][d2] = parallel.reduce_sum_complex_array(local_Q[d1][d2])
            for d3 in dir_chars:
                dQ[d1][d2][d3] = parallel.reduce_sum_complex_array(local_dQ[d1][d2][d3])
                dQf[d1][d2][d3] = parallel.reduce_sum_complex_array(local_dQf[d1][d2][d3])

    return {'Q': Q, 'dQ': dQ, 'dQf': dQf}


def _rr_sum(v1, v2, a1, a2, inv_de, inv_de2, weight):
    """Σ_nm  r^1_nm conj(r^2_nm) weight_nm,  for r = -i v / w + a^(H).

    The single place the metric's quadratic form is built — Q, dQ and dQf all
    come through here, differing only in `weight` and in which velocity
    matrices they pass.

    Written expanded rather than as ``r1 * conj(r2)``, and with the leading
    term multiplied in the original factor order, so that with the correction
    off (``a1 is None``) it reproduces the pre-correction engine **bit for
    bit**.  That is worth the slight awkwardness: dQ is a finite difference of
    two nearly equal sums, which amplifies a 1-ulp reassociation by
    |Q| / (delta |dQ|) — a factor of ~1e8 wherever dQ is small.

    Cross terms come from
        (-i v1/w + a1)(+i conj(v2)/w + conj(a2))
      = v1 conj(v2)/w^2 + (-i v1/w) conj(a2) + a1 conj(-i v2/w) + a1 conj(a2)
    """
    total = np.sum(v1 * np.conj(v2) * weight * inv_de2)
    if a1 is None:
        return total
    r1 = -1j * v1 * inv_de
    r2 = -1j * v2 * inv_de
    return total + np.sum(
        (r1 * np.conj(a2) + a1 * np.conj(r2) + a1 * np.conj(a2)) * weight
    )


def _process_kpoint(system, k, params):
    """Process a single k-point for quantum metric calculation.

    Q, dQ and dQf are all built from the interband position operator

        r^d = -i v^d / w   (+ a^(H)_d when the Wannier correction is on)

    as ``Σ r^{d1} conj(r^{d2}) × (occupation factor)`` — see `_rr`, which is
    the one place that product is formed.
    """
    dim = params['dim']
    dir_chars = params['dir_chars']
    eflist = params['eflist']
    kT = params['kT']
    eta = params['eta']
    delta = params['delta']
    nef = params['nef']
    wannier_r = params.get('wannier_r', WANNIER_R_DEFAULT)

    H, S, vtb = get_H_v(system, k)

    # Diagonalize
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    dim = len(ek)

    # Energy differences
    de_mtx = ek[:, None] - ek[None, :]
    # Set degenerate pairs to inf (kills their contribution)
    de_safe = np.where(np.abs(de_mtx) < DEG_THR, np.inf, de_mtx)

    # Non-degenerate mask
    nondeg = np.abs(de_mtx) >= DEG_THR

    # Safe 1/de and 1/de^2 — zero for degenerate pairs (and for n == m)
    inv_de = np.where(nondeg, 1.0 / de_safe, 0.0)
    inv_de2 = np.where(nondeg, 1.0 / de_mtx ** 2, 0.0)

    # Velocity in eigenbasis
    vmtx = {}
    for d in dir_chars:
        vmtx[d] = psi.conj().T @ vtb[d] @ psi

    # Perturbed eigenstates: psip = psi + psi * pert, psim = psi - psi * pert
    # pert = i*delta*vmtx / (de * (de + i*eta))  [zero for degenerate pairs]
    psip = {}
    psim = {}
    for d in dir_chars:
        denom = de_mtx * (de_mtx + 1j * eta)
        pert = np.where(nondeg, 1j * delta * vmtx[d] / denom, 0.0)
        psip[d] = psi + psi @ pert
        psim[d] = psi - psi @ pert

    # Perturbed velocity matrices: vmtxp[d1][d3] = psip[d3]' * vtb[d1] * psip[d3]
    vmtxp = {}
    vmtxm = {}
    for d1 in dir_chars:
        vmtxp[d1] = {}
        vmtxm[d1] = {}
        for d3 in dir_chars:
            vmtxp[d1][d3] = psip[d3].conj().T @ vtb[d1] @ psip[d3]
            vmtxm[d1][d3] = psim[d3].conj().T @ vtb[d1] @ psim[d3]

    # --- Position operator, with the Wannier-gauge correction if enabled ---
    # Only A^(W) is needed here, not its k-derivative: the metric involves r
    # itself, never the generalized derivative r^{a;b}.
    #
    # Masking follows chi^(2)/delta_Q: `nondeg` applies to the 1/w factor
    # only.  a^(H) is smooth across a degeneracy and is added unmasked (its
    # diagonal is already zeroed by `offdiag_A_H`).
    A_W, _ = compute_A_W_k(system, k, dir_chars, enabled=wannier_r,
                           need_deriv=False)
    a_H = offdiag_A_H(A_W, psi, dir_chars)

    # The perturbed connections use the same A^(W) rotated by the perturbed
    # states.  `pert` is anti-Hermitian, so psip is unitary to first order and
    # psip† A^(W) psip is a legitimate gauge rotation.  As in the pre-existing
    # finite-difference scheme, only the states are perturbed — 1/de is the
    # unperturbed one.
    a_Hp = {d3: offdiag_A_H(A_W, psip[d3], dir_chars) for d3 in dir_chars}
    a_Hm = {d3: offdiag_A_H(A_W, psim[d3], dir_chars) for d3 in dir_chars}

    def conn(a, d):
        """a^(H)_d out of a possibly-absent connection dict."""
        return None if a is None else a[d]

    # Loop over Fermi energies
    kpt_Q = {}
    kpt_dQ = {}
    kpt_dQf = {}
    for d1 in dir_chars:
        kpt_Q[d1] = {}
        kpt_dQ[d1] = {}
        kpt_dQf[d1] = {}
        for d2 in dir_chars:
            kpt_Q[d1][d2] = np.zeros(nef, dtype=complex)
            kpt_dQ[d1][d2] = {}
            kpt_dQf[d1][d2] = {}
            for d3 in dir_chars:
                kpt_dQ[d1][d2][d3] = np.zeros(nef, dtype=complex)
                kpt_dQf[d1][d2][d3] = np.zeros(nef, dtype=complex)

    for efc in range(nef):
        ef = eflist[efc]

        # Fermi function
        x = (ek - ef) / kT
        x_clip = np.clip(x, -500, 500)
        f = 1.0 / (1.0 + np.exp(x_clip))

        # df/dE
        de_f = -1.0 / kT * np.exp(x_clip) / (1.0 + np.exp(x_clip))**2
        de_f = np.where(np.isfinite(de_f), de_f, 0.0)

        # f_nm = f_n * (1 - f_m)
        fnm = f[:, None] * (1.0 - f[None, :])

        # dfnm for each direction: dfnm[d3] = vdf*(1-f') + f*(-vdf')
        dfnm = {}
        for d3 in dir_chars:
            vdf = de_f * np.diag(vmtx[d3])  # element-wise: de_f_n * v_nn
            dfnm[d3] = vdf[:, None] * (1.0 - f[None, :]) + f[:, None] * (-vdf[None, :])

        for d1 in dir_chars:
            for d2 in dir_chars:
                a1, a2 = conn(a_H, d1), conn(a_H, d2)

                # Quantum metric: sum_nm r[d1]_nm * conj(r[d2]_nm) * f_nm
                kpt_Q[d1][d2][efc] = _rr_sum(
                    vmtx[d1], vmtx[d2], a1, a2, inv_de, inv_de2, fnm)

                for d3 in dir_chars:
                    # Intrinsic: numerical finite difference
                    contp = _rr_sum(
                        vmtxp[d1][d3], vmtxp[d2][d3],
                        conn(a_Hp[d3], d1), conn(a_Hp[d3], d2),
                        inv_de, inv_de2, fnm)
                    contm = _rr_sum(
                        vmtxm[d1][d3], vmtxm[d2][d3],
                        conn(a_Hm[d3], d1), conn(a_Hm[d3], d2),
                        inv_de, inv_de2, fnm)
                    kpt_dQ[d1][d2][d3][efc] = (contp - contm) / (2 * delta)

                    # Extrinsic: Fermi surface
                    kpt_dQf[d1][d2][d3][efc] = _rr_sum(
                        vmtx[d1], vmtx[d2], a1, a2, inv_de, inv_de2, dfnm[d3])

    return {'Q': kpt_Q, 'dQ': kpt_dQ, 'dQf': kpt_dQf}
