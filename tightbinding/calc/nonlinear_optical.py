"""Second-order nonlinear optical susceptibility χ^(2)_abc(ω₁, ω₂).

Ported from compute_nonlinear_optical_pll.m — computes frequency-dependent
nonlinear optical response by summing contributions over a 2D k-grid using
density-matrix perturbation theory (interband/intraband decomposition).
"""

import os
import multiprocessing as mp
from functools import partial

import numpy as np
from numpy.typing import NDArray

from ..bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System


# 14 chi component names matching MATLAB
CHI_NAMES = [
    'chi_ii',
    'chi_ee1', 'chi_ee2',
    'chi_ei1', 'chi_ei2',
    'chi_eit1', 'chi_eit2', 'chi_eit3',
    'chi_ie1', 'chi_ie2',
    'chi_e1', 'chi_e2',
    'chi_i1', 'chi_i2',
]

# Direction label → index
_DIR = {'x': 0, 'y': 1, 'z': 2}

# Module-level globals set by pool initializer
_worker_system = None
_worker_params = None


def _init_worker(system, params):
    """Initializer for pool workers — stores system and params as globals."""
    global _worker_system, _worker_params
    _worker_system = system
    _worker_params = params
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)


def _worker_kpoint(tk):
    """Worker function: process one k-point and return chi contributions."""
    system = _worker_system
    p = _worker_params

    kpt = _process_kpoint(
        system, tk, p['dim'], p['dir_chars'], p['directions'],
        p['omega1list'], p['omega2_val'], p['omega2_mtx'],
        p['eta_val'], p['eta_mtx'],
        p['eflist'], p['kT'], p['nef'], p['nomega'],
    )

    # Flatten to a simple dict of arrays for efficient return
    flat = {}
    for name in CHI_NAMES:
        for abc in p['directions']:
            flat[f'{name}_{abc}'] = kpt[name][abc]
    return flat


def compute_nonlinear_optical(system: System, cfg: dict) -> dict:
    """Compute χ^(2) nonlinear optical response.

    Parameters
    ----------
    system : System with filled Hamiltonian matrices
    cfg : full config dict; reads from cfg['calc']

    Returns
    -------
    dict with keys for each chi component, each a nested dict [a][b][c] → array(nef, nomega)
    """
    calc = cfg['calc']
    nk1, nk2 = calc['nk']
    omega1list = np.asarray(calc['omega1list'], dtype=float)
    omega2_val = float(calc.get('omega2', 0.0))
    eta_val = float(calc['eta'])
    eflist = np.asarray(calc['eflist'], dtype=float)
    kT = float(calc['kT'])
    directions = calc['directions']  # e.g. ['xzx', 'zxx']

    nomega = len(omega1list)
    nef = len(eflist)

    # Determine unique direction chars needed
    dir_chars = sorted(set(c for abc in directions for c in abc))

    # Initialize output arrays
    result = {}
    for name in CHI_NAMES:
        result[name] = {}
        for abc in directions:
            a, b, c = abc
            d = result[name]
            d.setdefault(a, {})
            d[a].setdefault(b, {})
            d[a][b][c] = np.zeros((nef, nomega), dtype=complex)

    # Reciprocal lattice vectors
    b1, b2, _b3 = get_reciprocal_lattice(system.unitcell_vectors)

    db1 = b1 / (nk1 - 1)
    db2 = b2 / (nk2 - 1) if nk2 > 1 else np.zeros(3)

    dim = len(system.matrices[0].H)

    # Precompute broadening/frequency matrices
    eta_mtx = eta_val * np.ones((dim, dim))
    omega2_mtx = omega2_val * np.ones((dim, dim))

    # Build list of k-points
    k_list = []
    for kc1 in range(1, nk1 - 1):
        for kc2 in range(nk2):
            tk = -b1/2 - b2/2 + db1 * kc1 + db2 * kc2
            kx = tk[0]
            if abs(kx + np.pi) < 1e-6 or abs(kx - np.pi) < 1e-6:
                continue
            k_list.append(tk)

    total_jobs = len(k_list)
    norm = 1.0 / (nk1 * nk2)

    # Shared params for workers
    params = {
        'dim': dim,
        'dir_chars': dir_chars,
        'directions': directions,
        'omega1list': omega1list,
        'omega2_val': omega2_val,
        'omega2_mtx': omega2_mtx,
        'eta_val': eta_val,
        'eta_mtx': eta_mtx,
        'eflist': eflist,
        'kT': kT,
        'nef': nef,
        'nomega': nomega,
    }

    ncpu = os.cpu_count() or 1
    print(f"  Nonlinear optical: {total_jobs} k-points on {ncpu} cores")

    def _accumulate(flat):
        for name in CHI_NAMES:
            for abc in directions:
                a, b, c = abc
                result[name][a][b][c] += flat[f'{name}_{abc}'] * norm

    if ncpu == 1:
        # Serial fallback
        for i, tk in enumerate(k_list):
            if total_jobs >= 10 and (10 * (i + 1)) % total_jobs == 0:
                print(f"  k-point {i+1}/{total_jobs} ({100*(i+1)/total_jobs:.0f}%)")
            kpt = _process_kpoint(
                system, tk, dim, dir_chars, directions,
                omega1list, omega2_val, omega2_mtx, eta_val, eta_mtx,
                eflist, kT, nef, nomega,
            )
            flat = {}
            for name in CHI_NAMES:
                for abc in directions:
                    flat[f'{name}_{abc}'] = kpt[name][abc]
            _accumulate(flat)
    else:
        with mp.Pool(
            processes=ncpu,
            initializer=_init_worker,
            initargs=(system, params),
        ) as pool:
            done = 0
            for flat in pool.imap_unordered(_worker_kpoint, k_list, chunksize=max(1, total_jobs // (4 * ncpu))):
                done += 1
                if total_jobs >= 10 and (10 * done) % total_jobs == 0:
                    print(f"  k-point {done}/{total_jobs} ({100*done/total_jobs:.0f}%)")
                _accumulate(flat)

    return result


def _process_kpoint(
    system, k, dim, dir_chars, directions,
    omega1list, omega2_val, omega2_mtx, eta_val, eta_mtx,
    eflist, kT, nef, nomega,
):
    """Process a single k-point: diagonalize, build operators, compute chi contributions."""

    H, S, vtb = get_H_v(system, k, order=2)

    # Diagonalize
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)

    # Energy difference matrix
    de_mtx = ek[:, None] - ek[None, :]  # de_mtx[n,m] = e_n - e_m

    # Safe inverse of de_mtx (set diagonal/degenerate to 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_de = np.where(np.abs(de_mtx) > 1e-12, 1.0 / de_mtx, 0.0)

    # Unique 2-char direction pairs needed
    dir_pairs = set()
    for abc in directions:
        a, b, c = abc
        dir_pairs.add(b + c)
        dir_pairs.add(c + b)
        dir_pairs.add(b + a)
        dir_pairs.add(c + a)
    for d1 in dir_chars:
        for d2 in dir_chars:
            dir_pairs.add(d1 + d2)

    # Transform to eigenbasis
    vmtx = {}
    for d in dir_chars:
        vmtx[d] = psi.conj().T @ vtb[d] @ psi

    vvmtx = {}
    for pair in dir_pairs:
        if pair in vtb:
            vvmtx[pair] = psi.conj().T @ vtb[pair] @ psi

    # Diagonal velocity (group velocity)
    Delta = {}
    for d in dir_chars:
        vd = np.diag(vmtx[d])
        Delta[d] = vd[:, None] - vd[None, :]  # Delta[d][n,m] = v_nn^d - v_mm^d

    # Position operator (interband): r_nm = -i * v_nm / (e_n - e_m)
    rmtx = {}
    for d in dir_chars:
        rmtx[d] = -1j * vmtx[d] * inv_de

    # Generalized derivative of position operator: dk_rmtx[d1][d2]
    dk_rmtx = {}
    for d1 in dir_chars:
        dk_rmtx[d1] = {}
        for d2 in dir_chars:
            pair = d1 + d2
            dk_rmtx[d1][d2] = _compute_dk_rmtx(
                vmtx[d1], vmtx[d2], vvmtx[pair], Delta[d1], Delta[d2],
                de_mtx, inv_de, dim,
            )

    # Now loop over ef and omega
    kpt = {}
    for name in CHI_NAMES:
        kpt[name] = {}
        for abc in directions:
            kpt[name][abc] = np.zeros((nef, nomega), dtype=complex)

    for efind, ef in enumerate(eflist):
        # Fermi function and derivatives
        x = (ek - ef) / kT
        # Clip to avoid overflow
        x_clip = np.clip(x, -500, 500)
        f = 1.0 / (1.0 + np.exp(x_clip))
        de_f = -1.0 / kT * np.exp(x_clip) / (1.0 + np.exp(x_clip))**2
        # Second derivative of f
        # de2_f = -2/kT^2 * csch((ef-ek)/kT)^3 * sinh((ef-ek)/(2*kT))^4
        x2 = (ef - ek) / kT  # note sign
        x2_clip = np.clip(x2, -500, 500)
        with np.errstate(divide='ignore', invalid='ignore'):
            de2_f = -2.0 / kT**2 * (1.0 / np.sinh(x2_clip))**3 * np.sinh(x2_clip / 2.0)**4

        # Zero out where |x| > 10 (far from Fermi level)
        far = np.abs(x) > 10
        de_f[far] = 0.0
        de2_f[far] = 0.0

        # f matrix
        f_mtx = f[:, None] - f[None, :]

        # dk_f and d2k_f
        dk_f = {}
        dk_f_mtx = {}
        d2k_f = {}
        for d in dir_chars:
            vdiag = np.diag(vmtx[d])
            dk_f[d] = vdiag * de_f
            dk_f_mtx[d] = dk_f[d][:, None] - dk_f[d][None, :]

            d2k_f[d] = {}
            for d2 in dir_chars:
                pair = d + d2
                vv_diag = np.diag(vvmtx[pair])
                v1_diag = np.diag(vmtx[d])
                v2_diag = np.diag(vmtx[d2])
                d2k_f[d][d2] = vv_diag * de_f + v1_diag * v2_diag * de2_f

        for eind, omega1_val in enumerate(omega1list):
            omega1_mtx = omega1_val * np.ones((dim, dim))

            for abc in directions:
                dir_a, dir_b, dir_c = abc

                # ---- intra-intra (chi_ii) ----
                rho_ii = (1.0 / (omega1_mtx + omega2_mtx - 2j * eta_mtx)) * (
                    np.diag(d2k_f[dir_b][dir_c]) / (omega2_val - 1j * eta_val) +
                    np.diag(d2k_f[dir_c][dir_b]) / (omega1_val - 1j * eta_val)
                )
                kpt['chi_ii'][abc][efind, eind] = np.trace(vmtx[dir_a] @ rho_ii)

                # ---- inter-inter (chi_ee1, chi_ee2) ----
                G1 = f_mtx * rmtx[dir_b] / (omega1_mtx - de_mtx - 1j * eta_mtx)
                rho_ee1 = G1 @ rmtx[dir_c] - rmtx[dir_c] @ G1
                rho_ee1 = rho_ee1 / (omega1_mtx + omega2_mtx - de_mtx - 2j * eta_mtx)

                G2 = f_mtx * rmtx[dir_c] / (omega2_mtx - de_mtx - 1j * eta_mtx)
                rho_ee2 = G2 @ rmtx[dir_b] - rmtx[dir_b] @ G2
                rho_ee2 = rho_ee2 / (omega1_mtx + omega2_mtx - de_mtx - 2j * eta_mtx)

                kpt['chi_ee1'][abc][efind, eind] = np.trace(vmtx[dir_a] @ rho_ee1)
                kpt['chi_ee2'][abc][efind, eind] = np.trace(vmtx[dir_a] @ rho_ee2)

                # ---- inter-intra (chi_ei) ----
                denom12 = 1.0 / (omega1_mtx + omega2_mtx - de_mtx - 2j * eta_mtx)

                t1 = denom12 * (f_mtx / (omega1_mtx - de_mtx - 1j * eta_mtx) * dk_rmtx[dir_b][dir_c])
                t2 = denom12 * (f_mtx / (omega2_mtx - de_mtx - 1j * eta_mtx) * dk_rmtx[dir_c][dir_b])
                t3 = denom12 * (rmtx[dir_b] * (dk_f_mtx[dir_c] / (omega1_mtx - de_mtx - 1j * eta_mtx)))
                t4 = denom12 * (rmtx[dir_c] * (dk_f_mtx[dir_b] / (omega2_mtx - de_mtx - 1j * eta_mtx)))
                t5 = -denom12 * (rmtx[dir_b] * f_mtx * Delta[dir_c] / (omega1_mtx - de_mtx - 1j * eta_mtx)**2)
                t6 = -denom12 * (rmtx[dir_c] * f_mtx * Delta[dir_b] / (omega2_mtx - de_mtx - 1j * eta_mtx)**2)

                va = vmtx[dir_a]
                kpt['chi_eit1'][abc][efind, eind] = np.trace(va @ (t1 + t2))
                kpt['chi_eit2'][abc][efind, eind] = np.trace(va @ (t3 + t4))
                kpt['chi_eit3'][abc][efind, eind] = np.trace(va @ (t5 + t6))
                kpt['chi_ei1'][abc][efind, eind] = np.trace(va @ (t1 + t3 + t5))
                kpt['chi_ei2'][abc][efind, eind] = np.trace(va @ (t2 + t4 + t6))

                # ---- intra-inter (chi_ie1, chi_ie2) ----
                rho_ie1 = denom12 * (dk_f_mtx[dir_b] * rmtx[dir_c] / (omega1_val - 1j * eta_val))
                rho_ie2 = denom12 * (dk_f_mtx[dir_c] * rmtx[dir_b] / (omega2_val - 1j * eta_val))

                kpt['chi_ie1'][abc][efind, eind] = np.trace(rho_ie1 @ va)
                kpt['chi_ie2'][abc][efind, eind] = np.trace(rho_ie2 @ va)

                # ---- first-order density matrix contributions ----
                rho1_e = rmtx[dir_c] * f_mtx / (omega2_mtx - de_mtx - 1j * eta_mtx)
                rho1_i = np.diag(dk_f[dir_c]) / (omega2_val - 1j * eta_val)

                kpt['chi_e1'][abc][efind, eind] = np.trace(rho1_e @ dk_rmtx[dir_b][dir_a])
                kpt['chi_i1'][abc][efind, eind] = np.trace(rho1_i @ dk_rmtx[dir_b][dir_a])

                rho2_e = rmtx[dir_b] * f_mtx / (omega1_mtx - de_mtx - 1j * eta_mtx)
                rho2_i = np.diag(dk_f[dir_b]) / (omega1_val - 1j * eta_val)

                kpt['chi_e2'][abc][efind, eind] = np.trace(rho2_e @ dk_rmtx[dir_c][dir_a])
                kpt['chi_i2'][abc][efind, eind] = np.trace(rho2_i @ dk_rmtx[dir_c][dir_a])

    return kpt


def _compute_dk_rmtx(v_a, v_b, vv_ab, Delta_a, Delta_b, de_mtx, inv_de, dim):
    """Compute generalized derivative of position operator.

    Vectorized version of the MATLAB triple loop (lines 183-199).

    dk_rmtx[n,m] = (i/de[n,m]) * {
        (v_a[n,m]*Delta_b[n,m] + v_b[n,m]*Delta_a[n,m]) / de[n,m] - vv_ab[n,m]
        + sum_{p!=n,m} (v_a[n,p]*v_b[p,m]/de[p,m] - v_b[n,p]*v_a[p,m]/de[n,p])
    }
    """
    # Non-sum terms (from n,m diagonal velocities)
    diag_term = (v_a * Delta_b + v_b * Delta_a) * inv_de - vv_ab

    # Full sum over all p:
    # Term1_p: v_a[n,p]*v_b[p,m]*inv_de[p,m]  → (v_a @ (v_b * inv_de))[n,m]
    # Term2_p: v_b[n,p]*v_a[p,m]*inv_de[n,p]  → ((v_b * inv_de) @ v_a)[n,m]
    full_sum = v_a @ (v_b * inv_de) - (v_b * inv_de) @ v_a

    # Subtract p=n and p=m (where inv_de diagonal = 0 kills some terms):
    # p=n in Term1: diag_va[n]*v_b[n,m]*inv_de[n,m]   (nonzero)
    # p=n in Term2: diag_vb[n]*v_a[n,m]*inv_de[n,n]=0  (diagonal inv_de=0)
    # p=m in Term1: v_a[n,m]*diag_vb[m]*inv_de[m,m]=0  (diagonal inv_de=0)
    # p=m in Term2: v_b[n,m]*diag_va[m]*inv_de[n,m]   (nonzero)
    # Net subtraction: (diag_va[n] - diag_va[m]) * v_b[n,m] * inv_de[n,m]
    #                = Delta_a[n,m] * v_b[n,m] * inv_de[n,m]
    p_sum = full_sum - Delta_a * v_b * inv_de

    # Combine: dk_rmtx = i/de * (diag_term + p_sum)
    result = 1j * inv_de * (diag_term + p_sum)

    # Clean up inf/nan from degenerate states
    result = np.where(np.isfinite(result), result, 0.0)

    return result
