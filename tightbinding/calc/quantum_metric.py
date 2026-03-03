"""Quantum metric and its DC linear response.

Ported from compute_quantum_metric.m — computes the quantum metric tensor Q
and its intrinsic (dQ) and extrinsic (dQf) linear response over a 2D k-grid.
"""

import os
import multiprocessing as mp

import numpy as np

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System


# Module-level globals for multiprocessing pool
_worker_system = None
_worker_params = None

DEG_THR = 1e-5


def _init_worker(system, params):
    global _worker_system, _worker_params
    _worker_system = system
    _worker_params = params
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def _worker_kpoint(tk):
    system = _worker_system
    p = _worker_params
    return _process_kpoint(system, tk, p)


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

    db1 = b1 / (nk1 - 1)
    db2 = b2 / (nk2 - 1) if nk2 > 1 else np.zeros(3)

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

    ncpu = os.cpu_count() or 1
    print(f"  Quantum metric: {total_jobs} k-points on {ncpu} cores")

    def _accumulate(kpt):
        for d1 in dir_chars:
            for d2 in dir_chars:
                Q[d1][d2] += kpt['Q'][d1][d2] * norm
                for d3 in dir_chars:
                    dQ[d1][d2][d3] += kpt['dQ'][d1][d2][d3] * norm
                    dQf[d1][d2][d3] += kpt['dQf'][d1][d2][d3] * norm

    if ncpu == 1:
        for i, tk in enumerate(k_list):
            if total_jobs >= 10 and (10 * (i + 1)) % total_jobs == 0:
                print(f"  k-point {i+1}/{total_jobs} ({100*(i+1)/total_jobs:.0f}%)")
            kpt = _process_kpoint(system, tk, params)
            _accumulate(kpt)
    else:
        with mp.Pool(
            processes=ncpu,
            initializer=_init_worker,
            initargs=(system, params),
        ) as pool:
            done = 0
            for kpt in pool.imap_unordered(
                _worker_kpoint, k_list,
                chunksize=max(1, total_jobs // (4 * ncpu)),
            ):
                done += 1
                if total_jobs >= 10 and (10 * done) % total_jobs == 0:
                    print(f"  k-point {done}/{total_jobs} ({100*done/total_jobs:.0f}%)")
                _accumulate(kpt)

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
