"""Quantum metric and its DC linear response.

Ported from compute_quantum_metric.m — computes the quantum metric tensor Q
and its intrinsic (dQ) and extrinsic (dQf) linear response over a 2D k-grid.
"""

import warnings

import numpy as np

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel


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
    }

    parallel.print_root(
        f"  Quantum metric: {total_jobs} k-points on {parallel.size} rank(s)"
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


def _process_kpoint(system, k, params):
    """Process a single k-point for quantum metric calculation."""
    dim = params['dim']
    dir_chars = params['dir_chars']
    eflist = params['eflist']
    kT = params['kT']
    eta = params['eta']
    delta = params['delta']
    nef = params['nef']

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

        # Safe 1/de^2 — zero for degenerate pairs
        inv_de2 = np.where(nondeg, 1.0 / de_mtx**2, 0.0)

        for d1 in dir_chars:
            for d2 in dir_chars:
                # Quantum metric: sum_nm v[d1]_nm * conj(v[d2]_nm) * f_nm / de^2
                cont = np.sum(vmtx[d1] * np.conj(vmtx[d2]) * fnm * inv_de2)
                kpt_Q[d1][d2][efc] = cont

                for d3 in dir_chars:
                    # Intrinsic: numerical finite difference
                    contp = np.sum(
                        vmtxp[d1][d3] * np.conj(vmtxp[d2][d3]) * fnm * inv_de2
                    )
                    contm = np.sum(
                        vmtxm[d1][d3] * np.conj(vmtxm[d2][d3]) * fnm * inv_de2
                    )
                    kpt_dQ[d1][d2][d3][efc] = (contp - contm) / (2 * delta)

                    # Extrinsic: Fermi surface
                    cont_f = np.sum(
                        vmtx[d1] * np.conj(vmtx[d2]) * dfnm[d3] * inv_de2
                    )
                    kpt_dQf[d1][d2][d3][efc] = cont_f

    return {'Q': kpt_Q, 'dQ': kpt_dQ, 'dQf': kpt_dQf}
