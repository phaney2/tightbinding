"""Second-order nonlinear optical susceptibility χ^(2)_abc(ω₁, ω₂).

Ported from compute_nonlinear_optical_pll.m — computes frequency-dependent
nonlinear optical response by summing contributions over a 2D k-grid using
density-matrix perturbation theory (interband/intraband decomposition).
"""

import warnings

import numpy as np
from numpy.typing import NDArray

from ..bloch import get_H_v, get_H_k, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel
from .nonlinear_optical_fast import _process_kpoint_fast


# 14 chi component names matching MATLAB
CHI_NAMES = [
    'chi_ii',
    'chi_ee1', 'chi_ee2',
    'chi_ei1', 'chi_ei2',
    'chi_eit1', 'chi_eit2', 'chi_eit3',
    'chi_ie1', 'chi_ie2',
    'chi_e1', 'chi_e2',       # unphysical first-order density matrix terms
    'chi_i1', 'chi_i2',       # unphysical first-order density matrix terms
]

# 5-term breakdown of chi_ei1 and chi_ei2
CHI_EI_TERM_NAMES = [
    'chi_ei1_sipe_delta', 'chi_ei1_sipe_d2H', 'chi_ei1_sipe_3band',
    'chi_ei1_dk_f', 'chi_ei1_delta_r',
    'chi_ei2_sipe_delta', 'chi_ei2_sipe_d2H', 'chi_ei2_sipe_3band',
    'chi_ei2_dk_f', 'chi_ei2_delta_r',
]

# All chi names including sub-terms
CHI_ALL_NAMES = CHI_NAMES + CHI_EI_TERM_NAMES

# Physical terms that contribute to chi_total.
# chi_e1, chi_e2, chi_i1, chi_i2 are unphysical artefacts of the
# interband/intraband decomposition and must NOT be included in the total.
CHI_PHYSICAL = [
    'chi_ii',
    'chi_ee1', 'chi_ee2',
    'chi_ei1', 'chi_ei2',
    'chi_ie1', 'chi_ie2',
]

# Direction label → index
_DIR = {'x': 0, 'y': 1, 'z': 2}


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
    nk_cfg = list(calc['nk'])
    if len(nk_cfg) == 2:
        nk_cfg.append(1)
    nk1, nk2, nk3 = nk_cfg
    omega1list = np.asarray(calc['omega1list'], dtype=float)
    omega2_val = float(calc.get('omega2', 0.0))
    eta_val = float(calc['eta'])
    eflist = np.asarray(calc['eflist'], dtype=float)
    kT = float(calc['kT'])
    directions = calc['directions']  # e.g. ['xzx', 'zxx']
    method = calc.get('method', 'sos')  # 'sos' or 'projector'
    lam = float(calc.get('lam', 1e-4))  # finite-diff step for projector

    nomega = len(omega1list)
    nef = len(eflist)

    # Determine unique direction chars needed
    dir_chars = sorted(set(c for abc in directions for c in abc))

    # Initialize output arrays
    result = {}
    for name in CHI_ALL_NAMES:
        result[name] = {}
        for abc in directions:
            a, b, c = abc
            d = result[name]
            d.setdefault(a, {})
            d[a].setdefault(b, {})
            d[a][b][c] = np.zeros((nef, nomega), dtype=complex)

    # Reciprocal lattice vectors
    b1, b2, b3 = get_reciprocal_lattice(system.unitcell_vectors)

    db1 = b1 / (nk1 - 1) if nk1 > 1 else np.zeros(3)
    db2 = b2 / (nk2 - 1) if nk2 > 1 else np.zeros(3)
    db3 = b3 / (nk3 - 1) if nk3 > 1 else np.zeros(3)

    dim = len(system.matrices[0].H)

    # Precompute broadening/frequency matrices
    eta_mtx = eta_val * np.ones((dim, dim))
    omega2_mtx = omega2_val * np.ones((dim, dim))

    # Build list of k-points
    k_list = []
    if nk3 > 1:
        # 3D grid: uniform sampling over full BZ
        for kc1 in range(nk1):
            for kc2 in range(nk2):
                for kc3 in range(nk3):
                    tk = (-b1/2 - b2/2 - b3/2
                          + db1 * kc1 + db2 * kc2 + db3 * kc3)
                    k_list.append(tk)
    else:
        # 2D grid: original behavior with boundary exclusion
        for kc1 in range(1, nk1 - 1):
            for kc2 in range(nk2):
                tk = -b1/2 - b2/2 + db1 * kc1 + db2 * kc2
                kx = tk[0]
                if abs(kx + np.pi) < 1e-6 or abs(kx - np.pi) < 1e-6:
                    continue
                k_list.append(tk)

    total_jobs = len(k_list)
    norm = 1.0 / (nk1 * nk2 * nk3)

    method_label = f"method={method}" + (f", lam={lam:.0e}" if method == 'projector' else "")
    parallel.print_root(
        f"  Nonlinear optical: {total_jobs} k-points on {parallel.size} rank(s) ({method_label})"
    )

    # Scatter k-points across MPI ranks
    my_indices, my_klist = parallel.scatter_work(k_list)

    # Each rank accumulates into local result arrays
    local_result = {}
    for name in CHI_ALL_NAMES:
        for abc in directions:
            local_result[f'{name}_{abc}'] = np.zeros((nef, nomega), dtype=complex)

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    for i, tk in enumerate(my_klist):
        if parallel.is_root() and total_jobs >= 10:
            done = i + 1
            total_local = len(my_klist)
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100*done/total_local:.0f}%)")

        # SOS computation for all 14 components (vectorized over ef/omega)
        kpt = _process_kpoint_fast(
            system, tk, dim, dir_chars, directions,
            omega1list, omega2_val, omega2_mtx, eta_val, eta_mtx,
            eflist, kT, nef, nomega,
        )

        # Projector override for chi_e1/chi_e2
        if method == 'projector':
            chi_e_proj = _process_kpoint_projector_chi_e(
                system, tk, dir_chars, directions,
                omega1list, omega2_val, eta_val, eflist, kT, nef, nomega, lam,
            )
            for abc in directions:
                kpt['chi_e1'][abc] = chi_e_proj[abc]['chi_e1']
                kpt['chi_e2'][abc] = chi_e_proj[abc]['chi_e2']

        for name in CHI_ALL_NAMES:
            for abc in directions:
                local_result[f'{name}_{abc}'] += kpt[name][abc] * norm

    # Reduce across all ranks
    for name in CHI_ALL_NAMES:
        for abc in directions:
            a, b, c = abc
            result[name][a][b][c] = parallel.reduce_sum_complex_array(
                local_result[f'{name}_{abc}']
            )

    # Build chi_total from physical terms only
    result['chi_total'] = {}
    for abc in directions:
        a, b, c = abc
        result['chi_total'].setdefault(a, {})
        result['chi_total'][a].setdefault(b, {})
        total = np.zeros((nef, nomega), dtype=complex)
        for name in CHI_PHYSICAL:
            total += result[name][a][b][c]
        result['chi_total'][a][b][c] = total

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

    # Safe inverse of de_mtx (set near-degenerate pairs to 0).
    # Cutoff at eta: bands closer than the broadening are effectively degenerate
    # and their 1/de contributions would be unphysical artifacts.
    de_cutoff = eta_val
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_de = np.where(np.abs(de_mtx) > de_cutoff, 1.0 / de_mtx, 0.0)

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
    dk_rmtx_terms = {}
    for d1 in dir_chars:
        dk_rmtx[d1] = {}
        dk_rmtx_terms[d1] = {}
        for d2 in dir_chars:
            pair = d1 + d2
            dk_rmtx[d1][d2], dk_rmtx_terms[d1][d2] = _compute_dk_rmtx(
                vmtx[d1], vmtx[d2], vvmtx[pair], Delta[d1], Delta[d2],
                de_mtx, inv_de, dim,
                return_terms=True,
            )

    # Now loop over ef and omega
    kpt = {}
    for name in CHI_ALL_NAMES:
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

                # Sipe sub-term breakdown: split t1/t2 by dk_rmtx sub-terms
                sipe_bc = dk_rmtx_terms[dir_b][dir_c]
                sipe_cb = dk_rmtx_terms[dir_c][dir_b]
                d12_d1_scalar = denom12 * (f_mtx / (omega1_mtx - de_mtx - 1j * eta_mtx))
                d12_d2_scalar = denom12 * (f_mtx / (omega2_mtx - de_mtx - 1j * eta_mtx))
                for sub in ('delta', 'd2H', '3band'):
                    kpt[f'chi_ei1_sipe_{sub}'][abc][efind, eind] = np.trace(
                        va @ (d12_d1_scalar * sipe_bc[sub]))
                    kpt[f'chi_ei2_sipe_{sub}'][abc][efind, eind] = np.trace(
                        va @ (d12_d2_scalar * sipe_cb[sub]))
                kpt['chi_ei1_dk_f'][abc][efind, eind] = np.trace(va @ t3)
                kpt['chi_ei1_delta_r'][abc][efind, eind] = np.trace(va @ t5)
                kpt['chi_ei2_dk_f'][abc][efind, eind] = np.trace(va @ t4)
                kpt['chi_ei2_delta_r'][abc][efind, eind] = np.trace(va @ t6)

                # ---- intra-inter (chi_ie1, chi_ie2) ----
                rho_ie1 = denom12 * (dk_f_mtx[dir_b] * rmtx[dir_c] / (omega1_val - 1j * eta_val))
                rho_ie2 = denom12 * (dk_f_mtx[dir_c] * rmtx[dir_b] / (omega2_val - 1j * eta_val))

                kpt['chi_ie1'][abc][efind, eind] = np.trace(rho_ie1 @ va)
                kpt['chi_ie2'][abc][efind, eind] = np.trace(rho_ie2 @ va)

                # ---- first-order density matrix contributions ----
                # NOTE: chi_e1, chi_e2, chi_i1, chi_i2 are unphysical artefacts
                # of the interband/intraband decomposition. They are computed for
                # diagnostic purposes but excluded from chi_total.
                rho1_e = rmtx[dir_c] * f_mtx / (omega2_mtx - de_mtx - 1j * eta_mtx)
                rho1_i = np.diag(dk_f[dir_c]) / (omega2_val - 1j * eta_val)

                kpt['chi_e1'][abc][efind, eind] = np.trace(rho1_e @ dk_rmtx[dir_b][dir_a])
                kpt['chi_i1'][abc][efind, eind] = np.trace(rho1_i @ dk_rmtx[dir_b][dir_a])

                rho2_e = rmtx[dir_b] * f_mtx / (omega1_mtx - de_mtx - 1j * eta_mtx)
                rho2_i = np.diag(dk_f[dir_b]) / (omega1_val - 1j * eta_val)

                kpt['chi_e2'][abc][efind, eind] = np.trace(rho2_e @ dk_rmtx[dir_c][dir_a])
                kpt['chi_i2'][abc][efind, eind] = np.trace(rho2_i @ dk_rmtx[dir_c][dir_a])

    return kpt


def _compute_dk_rmtx(v_a, v_b, vv_ab, Delta_a, Delta_b, de_mtx, inv_de, dim,
                     return_terms=False):
    """Compute generalized derivative of position operator.

    Vectorized version of the MATLAB triple loop (lines 183-199).

    dk_rmtx[n,m] = (i/de[n,m]) * {
        (v_a[n,m]*Delta_b[n,m] + v_b[n,m]*Delta_a[n,m]) / de[n,m] - vv_ab[n,m]
        + sum_{p!=n,m} (v_a[n,p]*v_b[p,m]/de[p,m] - v_b[n,p]*v_a[p,m]/de[n,p])
    }

    If return_terms=True, returns (result, terms_dict) where terms_dict has:
      'delta': the Delta sub-term (velocity-difference diagonal contribution)
      'd2H':  the second k-derivative of H sub-term
      '3band': the 3-band sum sub-term
    """
    # Non-sum terms (from n,m diagonal velocities)
    delta_part = (v_a * Delta_b + v_b * Delta_a) * inv_de
    d2H_part = -vv_ab

    # Full sum over all p:
    full_sum = v_a @ (v_b * inv_de) - (v_b * inv_de) @ v_a
    p_sum = full_sum - Delta_a * v_b * inv_de

    # Combine: dk_rmtx = i/de * (diag_term + p_sum)
    result = 1j * inv_de * (delta_part + d2H_part + p_sum)

    # Clean up inf/nan from degenerate states
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


# ---------------------------------------------------------------------------
# Projector-based chi_e1/chi_e2 (arXiv:2412.03637)
# ---------------------------------------------------------------------------

def _match_bands(psi_ref, psi_shifted):
    """Match bands at shifted k to reference k via maximum overlap.

    Returns permutation: assignment[n] = index in psi_shifted for band n.
    """
    dim = psi_ref.shape[1]
    overlap = np.abs(psi_ref.conj().T @ psi_shifted) ** 2
    assignment = np.full(dim, -1, dtype=int)
    used = set()
    flat_idx = np.argsort(overlap.ravel())[::-1]
    for idx in flat_idx:
        n, m = divmod(idx, dim)
        if assignment[n] >= 0 or m in used:
            continue
        assignment[n] = m
        used.add(m)
        if len(used) == dim:
            break
    return assignment


def _diag_and_form_projectors(system, k_shift, psi_ref):
    """Diagonalize at shifted k, match bands, return list of projectors."""
    H, S = get_H_k(system, k_shift)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    dim = len(ek)
    assignment = _match_bands(psi_ref, psi)
    psi_matched = psi[:, assignment]
    return [np.outer(psi_matched[:, n], psi_matched[:, n].conj())
            for n in range(dim)]


def _compute_projector_derivs(system, k, dir_chars, lam=1e-4):
    """Compute band projectors and 1st/2nd derivatives via finite differences.

    Returns (ek, P, dP, d2P) where:
        ek: eigenvalues at k
        P[n]: projector for band n
        dP[d][n]: first derivative ∂_d P_n
        d2P[d1d2][n]: second derivative ∂_{d1}∂_{d2} P_n
    """
    dir_map = {'x': 0, 'y': 1, 'z': 2}

    H0, S0 = get_H_k(system, k)
    ek, psi0 = diagonalize_hk(H0, S0, eigenvectors=True)
    dim = len(ek)

    P = [np.outer(psi0[:, n], psi0[:, n].conj()) for n in range(dim)]

    e_vec = {}
    for d in dir_chars:
        e = np.zeros(3)
        e[dir_map[d]] = 1.0
        e_vec[d] = e

    # Projectors at k ± λ e_d
    P_plus = {}
    P_minus = {}
    for d in dir_chars:
        P_plus[d] = _diag_and_form_projectors(system, k + lam * e_vec[d], psi0)
        P_minus[d] = _diag_and_form_projectors(system, k - lam * e_vec[d], psi0)

    # Projectors for mixed second derivatives
    P_pp = {}
    P_pm = {}
    P_mp = {}
    P_mm = {}
    for d1 in dir_chars:
        for d2 in dir_chars:
            if d1 >= d2:
                continue
            pair = d1 + d2
            P_pp[pair] = _diag_and_form_projectors(
                system, k + lam * (e_vec[d1] + e_vec[d2]), psi0)
            P_pm[pair] = _diag_and_form_projectors(
                system, k + lam * (e_vec[d1] - e_vec[d2]), psi0)
            P_mp[pair] = _diag_and_form_projectors(
                system, k - lam * (e_vec[d1] - e_vec[d2]), psi0)
            P_mm[pair] = _diag_and_form_projectors(
                system, k - lam * (e_vec[d1] + e_vec[d2]), psi0)

    # First derivatives: central difference
    dP = {}
    for d in dir_chars:
        dP[d] = [(P_plus[d][n] - P_minus[d][n]) / (2 * lam) for n in range(dim)]

    # Second derivatives
    d2P = {}
    for d1 in dir_chars:
        for d2 in dir_chars:
            key = d1 + d2
            if d1 == d2:
                d2P[key] = [
                    (P_plus[d1][n] - 2 * P[n] + P_minus[d1][n]) / lam**2
                    for n in range(dim)
                ]
            elif d1 < d2:
                pair = d1 + d2
                d2P[key] = [
                    (P_pp[pair][n] - P_pm[pair][n]
                     - P_mp[pair][n] + P_mm[pair][n]) / (4 * lam**2)
                    for n in range(dim)
                ]
            else:
                pair = d2 + d1
                d2P[key] = [
                    (P_pp[pair][n] - P_mp[pair][n]
                     - P_pm[pair][n] + P_mm[pair][n]) / (4 * lam**2)
                    for n in range(dim)
                ]

    return ek, P, dP, d2P


def _compute_C_mn(P, dP, d2P, n, m, alpha, beta, gamma):
    """Compute C^{mn}_{α;βγ} from Eq. 31 of arXiv:2412.03637.

    C^{mn}_{α;βγ} = tr[P_n (∂_β P_m) ((∂_α ∂_γ P_n) + (∂_α P_m)(∂_γ P_n))]
    """
    ag_key = alpha + gamma
    term1 = P[n] @ dP[beta][m] @ d2P[ag_key][n]
    term2 = P[n] @ dP[beta][m] @ (dP[alpha][m] @ dP[gamma][n])
    return np.trace(term1 + term2)


def _process_kpoint_projector_chi_e(
    system, k, dir_chars, directions,
    omega1list, omega2_val, eta_val, eflist, kT, nef, nomega, lam,
):
    """Compute chi_e1/chi_e2 at a single k-point using projector method.

    Returns dict with chi_e1/chi_e2 for each direction triplet.
    """
    ek, P, dP, d2P = _compute_projector_derivs(system, k, dir_chars, lam)
    dim = len(ek)

    # Determine needed C^{mn} combinations
    needed_C = set()
    for abc in directions:
        a, b, c = abc
        needed_C.add((a, c, b))  # C^{mn}_{a;cb} for chi_e1
        needed_C.add((a, b, c))  # C^{mn}_{a;bc} for chi_e2

    C_cache = {}
    for (alpha, beta, gamma) in needed_C:
        key = (alpha, beta, gamma)
        C_cache[key] = np.zeros((dim, dim), dtype=complex)
        for n in range(dim):
            for m in range(dim):
                if n == m:
                    continue
                C_cache[key][n, m] = _compute_C_mn(
                    P, dP, d2P, n, m, alpha, beta, gamma
                )

    de_mtx = ek[:, None] - ek[None, :]

    chi_e = {}
    for abc in directions:
        chi_e[abc] = {
            'chi_e1': np.zeros((nef, nomega), dtype=complex),
            'chi_e2': np.zeros((nef, nomega), dtype=complex),
        }

    for efind, ef in enumerate(eflist):
        x = (ek - ef) / kT
        x_clip = np.clip(x, -500, 500)
        f_arr = 1.0 / (1.0 + np.exp(x_clip))
        f_mtx = f_arr[:, None] - f_arr[None, :]

        for eind, omega1_val in enumerate(omega1list):
            for abc in directions:
                a, b, c = abc

                # chi_e1: -Σ_{n≠m} C^{mn}_{a;cb} f_{nm} / (ω₂ - ε_{nm} - iη)
                C_acb = C_cache[(a, c, b)]
                denom1 = omega2_val - de_mtx - 1j * eta_val
                chi_e[abc]['chi_e1'][efind, eind] = -np.sum(C_acb * f_mtx / denom1)

                # chi_e2: -Σ_{n≠m} C^{mn}_{a;bc} f_{nm} / (ω₁ - ε_{nm} - iη)
                C_abc = C_cache[(a, b, c)]
                denom2 = omega1_val - de_mtx - 1j * eta_val
                chi_e[abc]['chi_e2'][efind, eind] = -np.sum(C_abc * f_mtx / denom2)

    return chi_e
