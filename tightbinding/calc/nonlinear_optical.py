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
from .freq_integral import FreqIntegralSpec
from .wannier_gauge import (
    WANNIER_R_DEFAULT, apply_wannier_correction, compute_A_W_k,
    resolve_wannier_r, system_has_wannier_r,
)

# Back-compat alias: older scratch scripts import `_compute_A_W_k` from this
# module.  The implementation moved to
# calc/wannier_gauge.py so that every engine gates the same kernel through the
# same switch; the 3-argument call form is unchanged.
_compute_A_W_k = compute_A_W_k


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

# 6-term breakdown of chi_ei1 and chi_ei2.
# 'wannier_corr' is the Wannier-gauge correction to dk_rmtx
# (arXiv:1804.04030 Eq. 36), separated out so the 6 sub-terms sum
# exactly to the direct chi_ei1/chi_ei2 values.
CHI_EI_TERM_NAMES = [
    'chi_ei1_sipe_delta', 'chi_ei1_sipe_d2H', 'chi_ei1_sipe_3band',
    'chi_ei1_sipe_wannier_corr',
    'chi_ei1_dk_f', 'chi_ei1_delta_r',
    'chi_ei2_sipe_delta', 'chi_ei2_sipe_d2H', 'chi_ei2_sipe_3band',
    'chi_ei2_sipe_wannier_corr',
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

    Two modes, selected by the config:

    * **sampled** (``calc.omega1list``) — χ^(2) at each listed photon energy.
      Output arrays are ``(nef, nomega)``.
    * **frequency-integrated** (``calc.freq_integral``) — the closed-form
      ``J = ∫ dω ω^(-p) χ^(2)(ω)`` over ``[omega_min, omega_max]``, evaluated
      from exact antiderivatives rather than quadrature. Output arrays are
      ``(nef, n_p)``, one column per requested power ``p``. Additional keys
      ``endpt_log_<name>`` (and ``endpt_pow<j>_<name>`` for p ≥ 2) carry the
      lower-endpoint divergence coefficients; see `freq_integral`.

    Parameters
    ----------
    system : System with filled Hamiltonian matrices
    cfg : full config dict; reads from cfg['calc']

    Returns
    -------
    dict with keys for each chi component, each a nested dict [a][b][c] → array
    """
    calc = cfg['calc']
    nk_cfg = list(calc['nk'])
    if len(nk_cfg) == 2:
        nk_cfg.append(1)
    nk1, nk2, nk3 = nk_cfg
    omega2_val = float(calc.get('omega2', 0.0))
    eta_val = float(calc['eta'])
    # Souza-style Lorentzian regularization of the bare 1/w_{nm}:
    #   inv_de = w_{nm} / (w_{nm}^2 + eta_sos^2)
    # Replaces the old hard cutoff `de_cutoff = eta`.  Default matches
    # the prior cutoff scale so existing scripts see similar behaviour
    # quantitatively while gaining smooth regularization properties.
    eta_sos = float(calc.get('eta_sos', eta_val))
    eflist = np.asarray(calc['eflist'], dtype=float)
    kT = float(calc['kT'])
    directions = calc['directions']  # e.g. ['xzx', 'zxx']
    method = calc.get('method', 'sos')  # 'sos' or 'projector'
    lam = float(calc.get('lam', 1e-4))  # finite-diff step for projector

    # Wannier-gauge position correction: one switch, read from system.wannier_r
    # by the shared reader that delta_Q and quantum_metric also use.
    wannier_r = resolve_wannier_r(cfg, system, 'nonlinear_optical')

    # --- sampled vs frequency-integrated omega axis ---
    fi_block = calc.get('freq_integral')
    freq_spec = None
    omega1list = None
    if fi_block is None:
        omega1list = np.asarray(calc['omega1list'], dtype=float)
        nomega = len(omega1list)
    else:
        if 'omega1list' in calc:
            raise ValueError(
                "calc: give either 'omega1list' (sampled chi^(2)) or "
                "'freq_integral' (analytic frequency integral), not both"
            )
        if method != 'sos':
            raise ValueError(
                f"calc.method='{method}' is not supported with freq_integral; "
                "the projector chi_e path is built on a sampled omega list"
            )
        gap = _estimate_direct_gap(system, nk_cfg, float(eflist[0]))
        freq_spec = FreqIntegralSpec.from_config(fi_block, eta_val, gap=gap)
        nomega = freq_spec.nchan

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

    # Periodic (endpoint-free) grid — see note in delta_Q.py.
    db1 = b1 / nk1 if nk1 > 1 else np.zeros(3)
    db2 = b2 / nk2 if nk2 > 1 else np.zeros(3)
    db3 = b3 / nk3 if nk3 > 1 else np.zeros(3)

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
        # 2D grid: full periodic grid (the old version excluded the
        # kc1 = 0, nk1-1 rows and kx = ±pi points, biasing BZ averages
        # by O(1/nk); near-degeneracies are now handled by the Souza
        # eta_sos regularization, so no exclusions are needed).
        for kc1 in range(nk1):
            for kc2 in range(nk2):
                tk = -b1/2 - b2/2 + db1 * kc1 + db2 * kc2
                k_list.append(tk)

    total_jobs = len(k_list)
    norm = 1.0 / (nk1 * nk2 * nk3)

    method_label = f"method={method}" + (f", lam={lam:.0e}" if method == 'projector' else "")
    parallel.print_root(
        f"  Nonlinear optical: {total_jobs} k-points on {parallel.size} rank(s) "
        f"({method_label}, eta={eta_val}, eta_sos={eta_sos}, "
        f"wannier_r={wannier_r})"
    )
    if method == 'projector' and wannier_r and system_has_wannier_r(system):
        parallel.print_root(
            "  NOTE: method='projector' rebuilds chi_e1/chi_e2 from H(k) "
            "projectors, which carry no Wannier-gauge position correction. "
            "Those two terms are effectively wannier_r=False regardless of the "
            "flag (both are unphysical and excluded from chi_total)."
        )
    if freq_spec is not None:
        parallel.print_root(f"  Analytic frequency integral: {freq_spec.describe()}")

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
            eflist, kT, nef, nomega, eta_sos=eta_sos,
            freq_integral=freq_spec, wannier_r=wannier_r,
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

    if freq_spec is not None:
        cond = parallel.reduce_max(freq_spec.max_cond)
        parallel.print_root(
            f"  Frequency integral: worst cancellation ratio {cond:.2e} "
            f"(~{max(0.0, np.log10(max(cond, 1.0))):.1f} decimal digits lost "
            f"to the near-coincident z12/z1 poles)"
        )
        if any(p <= 1 for p in freq_spec.p_list):
            parallel.print_root(
                "  Note: chi_e1 and chi_i1 are omega-independent, so their "
                "integral to omega_max=inf does not converge for p <= 1 and "
                "is returned as NaN. Both are unphysical and excluded from "
                "chi_total."
            )
        result = _split_freq_integral(result, freq_spec)

    return result


def _estimate_direct_gap(system, nk_cfg, ef, nk_max=16):
    """Minimum direct gap straddling `ef`, on a coarse grid.

    Only used to sanity-check omega_min against ``omega_min << E_gap``; a few
    hundred diagonalizations, run on rank 0 and broadcast. Returns ``inf`` if
    no k-point has states on both sides of `ef`.
    """
    if not parallel.is_root():
        return parallel.bcast(None)

    b1, b2, b3 = get_reciprocal_lattice(system.unitcell_vectors)
    n1, n2, n3 = (max(1, min(n, nk_max)) for n in nk_cfg)
    db1 = b1 / n1 if n1 > 1 else np.zeros(3)
    db2 = b2 / n2 if n2 > 1 else np.zeros(3)
    db3 = b3 / n3 if n3 > 1 else np.zeros(3)

    gap = np.inf
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                tk = (-b1 / 2 - b2 / 2 - b3 / 2
                      + db1 * i1 + db2 * i2 + db3 * i3)
                H, S = get_H_k(system, tk)
                ek = diagonalize_hk(H, S, eigenvectors=False)
                below = ek[ek <= ef]
                above = ek[ek > ef]
                if below.size and above.size:
                    gap = min(gap, float(above.min() - below.max()))
    return parallel.bcast(gap)


def _split_freq_integral(result, spec):
    """Split the channel axis into integral values plus endpoint diagnostics.

    Every chi name keeps its ``[a][b][c]`` nesting, with the second array axis
    reduced from `spec.nchan` channels to one column per requested power `p`.
    The lower-endpoint coefficients (note section 5.2) are emitted as extra
    top-level names, for the physical terms and their total only — those are
    the ones whose divergences are supposed to cancel against each other.
    """
    jidx = spec.channel_indices('J')

    def _remap(src, idx):
        out = {}
        for a, bd in src.items():
            out[a] = {}
            for b, cd in bd.items():
                out[a][b] = {}
                for c, arr in cd.items():
                    out[a][b][c] = arr[:, idx]
        return out

    out = {name: _remap(dirs, jidx) for name, dirs in result.items()}
    for suffix, kind, j in spec.diagnostic_names():
        idx = spec.channel_indices(kind, j)
        for name in CHI_PHYSICAL + ['chi_total']:
            out[f'{suffix}_{name}'] = _remap(result[name], idx)
    return out


def _process_kpoint(
    system, k, dim, dir_chars, directions,
    omega1list, omega2_val, omega2_mtx, eta_val, eta_mtx,
    eflist, kT, nef, nomega, eta_sos=None,
    wannier_r=WANNIER_R_DEFAULT,
):
    """Process a single k-point: diagonalize, build operators, compute chi contributions.

    Near-degeneracy handling uses Souza Lorentzian regularization
      inv_de = w_{nm} / (w_{nm}^2 + eta_sos^2)
    (bounded at w_{nm}->0).  If eta_sos is None, defaults to eta_val.

    `wannier_r` gates the Wannier-gauge position correction (see
    calc/wannier_gauge.py).  It is inert for systems without
    ``wannier_r_matrices``.  This is the unvectorized reference path and must
    track `_process_kpoint_fast`, which takes the same keyword.
    """
    if eta_sos is None:
        eta_sos = eta_val

    H, S, vtb = get_H_v(system, k, order=2)

    # Diagonalize
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)

    # Energy difference matrix
    de_mtx = ek[:, None] - ek[None, :]  # de_mtx[n,m] = e_n - e_m

    # Souza-style Lorentzian regularization of the bare 1/w_{nm}.
    # Mask out exact zeros (n == m self-pairs) only.
    nondeg = np.abs(de_mtx) > 1e-12
    inv_de = np.where(nondeg, de_mtx / (de_mtx ** 2 + eta_sos ** 2), 0.0)

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

    # --- Wannier-gauge position correction (calc/wannier_gauge.py) ---
    # Eq. 22 on r, the covariant-derivative correction on r^{a;b} (registered
    # as the 'wannier_corr' Sipe sub-term so the sub-terms still sum exactly
    # to dk_rmtx), and the physical velocity v = i[H, r] as the current vertex
    # used below.  The bare v has already gone into Delta and the Sipe sum
    # rule, where it belongs.  All no-ops when the flag is off or the system
    # carries no position matrices.
    A_W, dA_W = compute_A_W_k(system, k, dir_chars, enabled=wannier_r)
    rmtx, corr, vcur = apply_wannier_correction(
        A_W, dA_W, psi, rmtx, dir_chars, vmtx=vmtx, de_mtx=de_mtx)
    if corr is not None:
        for d1 in dir_chars:
            for d2 in dir_chars:
                dk_rmtx[d1][d2] = dk_rmtx[d1][d2] + corr[d1][d2]
                dk_rmtx_terms[d1][d2]['wannier_corr'] = corr[d1][d2]

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
                # Band curvature d_d d_d2 E_n = <n|d d H|n> + 2 Re sum_m v_nm v_mn / w_nm
                # (Hellmann-Feynman twice).  The diagonal of the second
                # derivative of H alone is neither the curvature nor gauge
                # invariant; the interband sum rule completes it (bare v, as
                # for every band-energy identity).
                curv = (np.diag(vvmtx[pair])
                        + 2.0 * np.real(np.sum(vmtx[d] * vmtx[d2].T * inv_de, axis=1)))
                v1_diag = np.diag(vmtx[d])
                v2_diag = np.diag(vmtx[d2])
                d2k_f[d][d2] = curv * de_f + v1_diag * v2_diag * de2_f

        for eind, omega1_val in enumerate(omega1list):
            omega1_mtx = omega1_val * np.ones((dim, dim))

            for abc in directions:
                dir_a, dir_b, dir_c = abc

                # ---- intra-intra (chi_ii) ----
                # Overall (-i)^2 = -1: each intraband (i d/dk) vertex of the
                # length-gauge coupling E.(r_e + i d/dk) carries a factor -i
                # relative to the legacy MATLAB expressions (2026-07 audit).
                rho_ii = -(1.0 / (omega1_mtx + omega2_mtx + 2j * eta_mtx)) * (
                    np.diag(d2k_f[dir_b][dir_c]) / (omega2_val + 1j * eta_val) +
                    np.diag(d2k_f[dir_c][dir_b]) / (omega1_val + 1j * eta_val)
                )
                kpt['chi_ii'][abc][efind, eind] = np.trace(vcur[dir_a] @ rho_ii)

                # ---- inter-inter (chi_ee1, chi_ee2) ----
                G1 = f_mtx * rmtx[dir_b] / (omega1_mtx - de_mtx + 1j * eta_mtx)
                rho_ee1 = G1 @ rmtx[dir_c] - rmtx[dir_c] @ G1
                rho_ee1 = rho_ee1 / (omega1_mtx + omega2_mtx - de_mtx + 2j * eta_mtx)

                G2 = f_mtx * rmtx[dir_c] / (omega2_mtx - de_mtx + 1j * eta_mtx)
                rho_ee2 = G2 @ rmtx[dir_b] - rmtx[dir_b] @ G2
                rho_ee2 = rho_ee2 / (omega1_mtx + omega2_mtx - de_mtx + 2j * eta_mtx)

                kpt['chi_ee1'][abc][efind, eind] = np.trace(vcur[dir_a] @ rho_ee1)
                kpt['chi_ee2'][abc][efind, eind] = np.trace(vcur[dir_a] @ rho_ee2)

                # ---- inter-intra (chi_ei) ----
                denom12 = 1.0 / (omega1_mtx + omega2_mtx - de_mtx + 2j * eta_mtx)

                # 2026-07 audit corrections (verified against exact pole-sum
                # sigma^{abc} and the delta_Q sum rule):
                #  (a) all chi_ei terms carry the factor -i of the intraband
                #      vertex i d/dk (dropped in the MATLAB original);
                #  (b) t5/t6: d/dk_c (1/(w - w_nm)) = +Delta^c/(w - w_nm)^2,
                #      so the old leading minus sign was wrong.
                t1 = -1j * denom12 * (f_mtx / (omega1_mtx - de_mtx + 1j * eta_mtx) * dk_rmtx[dir_b][dir_c])
                t2 = -1j * denom12 * (f_mtx / (omega2_mtx - de_mtx + 1j * eta_mtx) * dk_rmtx[dir_c][dir_b])
                t3 = -1j * denom12 * (rmtx[dir_b] * (dk_f_mtx[dir_c] / (omega1_mtx - de_mtx + 1j * eta_mtx)))
                t4 = -1j * denom12 * (rmtx[dir_c] * (dk_f_mtx[dir_b] / (omega2_mtx - de_mtx + 1j * eta_mtx)))
                t5 = -1j * denom12 * (rmtx[dir_b] * f_mtx * Delta[dir_c] / (omega1_mtx - de_mtx + 1j * eta_mtx)**2)
                t6 = -1j * denom12 * (rmtx[dir_c] * f_mtx * Delta[dir_b] / (omega2_mtx - de_mtx + 1j * eta_mtx)**2)

                va = vcur[dir_a]   # physical current vertex
                kpt['chi_eit1'][abc][efind, eind] = np.trace(va @ (t1 + t2))
                kpt['chi_eit2'][abc][efind, eind] = np.trace(va @ (t3 + t4))
                kpt['chi_eit3'][abc][efind, eind] = np.trace(va @ (t5 + t6))
                kpt['chi_ei1'][abc][efind, eind] = np.trace(va @ (t1 + t3 + t5))
                kpt['chi_ei2'][abc][efind, eind] = np.trace(va @ (t2 + t4 + t6))

                # Sipe sub-term breakdown: split t1/t2 by dk_rmtx sub-terms.
                # 'wannier_corr' only present when the system carries
                # wannier_r_matrices (i.e. position operator corrections
                # from arXiv:1804.04030).
                sipe_bc = dk_rmtx_terms[dir_b][dir_c]
                sipe_cb = dk_rmtx_terms[dir_c][dir_b]
                d12_d1_scalar = -1j * denom12 * (f_mtx / (omega1_mtx - de_mtx + 1j * eta_mtx))
                d12_d2_scalar = -1j * denom12 * (f_mtx / (omega2_mtx - de_mtx + 1j * eta_mtx))
                for sub in ('delta', 'd2H', '3band', 'wannier_corr'):
                    if sub in sipe_bc:
                        kpt[f'chi_ei1_sipe_{sub}'][abc][efind, eind] = np.trace(
                            va @ (d12_d1_scalar * sipe_bc[sub]))
                    if sub in sipe_cb:
                        kpt[f'chi_ei2_sipe_{sub}'][abc][efind, eind] = np.trace(
                            va @ (d12_d2_scalar * sipe_cb[sub]))
                kpt['chi_ei1_dk_f'][abc][efind, eind] = np.trace(va @ t3)
                kpt['chi_ei1_delta_r'][abc][efind, eind] = np.trace(va @ t5)
                kpt['chi_ei2_dk_f'][abc][efind, eind] = np.trace(va @ t4)
                kpt['chi_ei2_delta_r'][abc][efind, eind] = np.trace(va @ t6)

                # ---- intra-inter (chi_ie1, chi_ie2) ----
                # -i from the intraband first vertex (see audit note above).
                rho_ie1 = -1j * denom12 * (dk_f_mtx[dir_b] * rmtx[dir_c] / (omega1_val + 1j * eta_val))
                rho_ie2 = -1j * denom12 * (dk_f_mtx[dir_c] * rmtx[dir_b] / (omega2_val + 1j * eta_val))

                kpt['chi_ie1'][abc][efind, eind] = np.trace(rho_ie1 @ va)
                kpt['chi_ie2'][abc][efind, eind] = np.trace(rho_ie2 @ va)

                # ---- first-order density matrix contributions ----
                # NOTE: chi_e1, chi_e2, chi_i1, chi_i2 are unphysical artefacts
                # of the interband/intraband decomposition. They are computed for
                # diagnostic purposes but excluded from chi_total.
                rho1_e = rmtx[dir_c] * f_mtx / (omega2_mtx - de_mtx + 1j * eta_mtx)
                rho1_i = np.diag(dk_f[dir_c]) / (omega2_val + 1j * eta_val)

                kpt['chi_e1'][abc][efind, eind] = np.trace(rho1_e @ dk_rmtx[dir_b][dir_a])
                kpt['chi_i1'][abc][efind, eind] = np.trace(rho1_i @ dk_rmtx[dir_b][dir_a])

                rho2_e = rmtx[dir_b] * f_mtx / (omega1_mtx - de_mtx + 1j * eta_mtx)
                rho2_i = np.diag(dk_f[dir_b]) / (omega1_val + 1j * eta_val)

                kpt['chi_e2'][abc][efind, eind] = np.trace(rho2_e @ dk_rmtx[dir_c][dir_a])
                kpt['chi_i2'][abc][efind, eind] = np.trace(rho2_i @ dk_rmtx[dir_c][dir_a])

    return kpt


def _compute_dk_rmtx(v_a, v_b, vv_ab, Delta_a, Delta_b, de_mtx, inv_de, dim,
                     return_terms=False):
    """Compute generalized derivative of position operator (Sipe sum rule).

    dk_rmtx[n,m] = (i/de[n,m]) * {
        (v_a[n,m]*Delta_b[n,m] + v_b[n,m]*Delta_a[n,m]) / de[n,m] - vv_ab[n,m]
        + sum_{p!=n,m} (v_a[n,p]*v_b[p,m]/de[p,m] - v_b[n,p]*v_a[p,m]/de[n,p])
    }

    With Souza-regularized inv_de (bounded at w→0), no NaN/Inf appear
    and the three sub-terms sum exactly to the total dk_rmtx.

    If return_terms=True, returns (result, terms_dict) with:
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

    if not return_terms:
        # Combine: dk_rmtx = i·inv_de·(delta + d2H + p_sum)
        return 1j * inv_de * (delta_part + d2H_part + p_sum)

    # Return sub-terms as separate arrays so they sum exactly to dk_rmtx.
    terms = {
        'delta': 1j * inv_de * delta_part,
        'd2H':   1j * inv_de * d2H_part,
        '3band': 1j * inv_de * p_sum,
    }
    result = terms['delta'] + terms['d2H'] + terms['3band']
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
                denom1 = omega2_val - de_mtx + 1j * eta_val
                chi_e[abc]['chi_e1'][efind, eind] = -np.sum(C_acb * f_mtx / denom1)

                # chi_e2: -Σ_{n≠m} C^{mn}_{a;bc} f_{nm} / (ω₁ - ε_{nm} - iη)
                C_abc = C_cache[(a, b, c)]
                denom2 = omega1_val - de_mtx + 1j * eta_val
                chi_e[abc]['chi_e2'][efind, eind] = -np.sum(C_abc * f_mtx / denom2)

    return chi_e
