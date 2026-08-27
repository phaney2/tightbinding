"""Vectorized χ^(2) — eliminates ef/omega Python loops via broadcasting.

Drop-in replacement for _process_kpoint. All arrays carry (nef, dim, dim, nomega)
shape; einsum replaces trace-of-matmul patterns.

Notation:
  E = nef, W = nomega, D = dim
  indices: e=ef, w=omega, n,m,p = band

The trailing W axis is an *omega axis*, not necessarily a list of photon
energies: every phase-3 contraction is linear in the omega-dependent kernels,
so substituting analytically frequency-integrated kernels for the sampled ones
turns the same code into the frequency-integrated engine.  See
`_build_omega_kernels` and `calc/freq_integral.py`.
"""

import numpy as np

from .wannier_gauge import WANNIER_R_DEFAULT, compute_A_W_k


def _build_omega_kernels(de_mtx, omega1list, omega2_val, eta_val,
                         freq_integral=None):
    """Build every omega-dependent factor phase 3 consumes.

    All omega dependence of chi^(2) lives in denominator factors; matrix
    elements, Fermi factors and vertex prefactors are omega-independent.  This
    function is the single place those factors are formed, which is what lets
    the sampled and frequency-integrated modes share the term algebra.

    Parameters
    ----------
    freq_integral : FreqIntegralSpec or None
        None  -> sampled mode, W = len(omega1list), kernels evaluated at each
                 photon energy (the historical behaviour, unchanged).
        spec  -> integrated mode, W = spec.nchan, each kernel replaced by its
                 closed-form integral int dw w^(-p) (...) over
                 [omega_min, omega_max], plus the endpoint-coefficient
                 channels.

    Returns
    -------
    K : dict of arrays whose trailing axis is W
        'd1'       (D,D,W)   1/(w - de + i eta)
        'd12'      (D,D,W)   1/(w + w2 - de + 2i eta)
        'd12_d1'   (D,D,W)   product of the two above, same band pair
        'd12_d1sq' (D,D,W)   d12 * d1^2
        'd12_om1'  (D,D,W)   d12 * 1/(w + i eta)
        'om1'      (W,)      1/(w + i eta)
        'om12'     (W,)      1/(w + w2 + 2i eta)
        'om12_om1' (W,)      product of the two above
        'const'    (W,)      1 (sampled) / int dw w^(-p) (integrated)
        'ee1A'     (D,D,D,W) [m,n,p] = d12[m,n] * d1[m,p]
        'ee1B'     (D,D,D,W) [m,n,p] = d12[m,n] * d1[p,n]
    denom2 : (D,D)   1/(w2 - de + i eta), omega-independent
    denom2_sq : (D,D)
    s_om2 : complex  1/(w2 + i eta), omega-independent

    Notes
    -----
    * `denom2`/`s_om2` depend on the fixed omega2 only, so they stay outside
      the integral and simply multiply the integrated kernels.
    * The three-band contraction in chi_ee1 pairs poles at *different* band
      pairs, so its kernel genuinely needs three band indices; it does not
      factorize once integrated.  The (D,D,D,W) transient is ~19 MB at
      D = 18, W = 200 and is built once per k-point.
    """
    d_shape = de_mtx.shape

    if freq_integral is None:
        omega1 = np.asarray(omega1list, dtype=float)
        omega12 = omega1 + omega2_val
        d1 = 1.0 / (omega1[None, None, :] - de_mtx[:, :, None] + 1j * eta_val)
        d12 = 1.0 / (omega12[None, None, :] - de_mtx[:, :, None] + 2j * eta_val)
        om1 = 1.0 / (omega1 + 1j * eta_val)
        om12 = 1.0 / (omega12 + 2j * eta_val)
        K = {
            'd1': d1,
            'd12': d12,
            'd12_d1': d12 * d1,
            'd12_d1sq': d12 * d1 ** 2,
            'd12_om1': d12 * om1,
            'om1': om1,
            'om12': om12,
            'om12_om1': om12 * om1,
            'const': np.ones(len(omega1), dtype=complex),
            # [m,n,p,w] = d12[m,n,w] * d1[m,p,w]
            'ee1A': d12[:, :, None, :] * d1[:, None, :, :],
            # [m,n,p,w] = d12[m,n,w] * d1[p,n,w]
            'ee1B': d12[:, :, None, :] * d1.transpose(1, 0, 2)[None, :, :, :],
        }
    else:
        # Retarded poles: 1/(w - X + i c eta) = 1/(w - z) with z = X - i c eta,
        # so every z sits in the lower half plane for eta > 0.
        z1 = de_mtx - 1j * eta_val                        # (D,D)
        z12 = de_mtx - omega2_val - 2j * eta_val          # (D,D)
        zo1 = np.array(-1j * eta_val, dtype=complex)      # scalar
        zo12 = np.array(-omega2_val - 2j * eta_val, dtype=complex)
        fi = freq_integral
        z1_b = np.broadcast_to(zo1, d_shape)
        K = {
            'd1': fi.kernel([z1], [1]),
            'd12': fi.kernel([z12], [1]),
            'd12_d1': fi.kernel([z12, z1], [1, 1]),
            'd12_d1sq': fi.kernel([z12, z1], [1, 2]),
            'd12_om1': fi.kernel([z12, z1_b], [1, 1]),
            'om1': fi.kernel([zo1], [1]),
            'om12': fi.kernel([zo12], [1]),
            'om12_om1': fi.kernel([zo12, zo1], [1, 1]),
            'const': fi.kernel([], []),
            'ee1A': fi.kernel([z12[:, :, None], z1[:, None, :]], [1, 1]),
            'ee1B': fi.kernel([z12[:, :, None], z1.T[None, :, :]], [1, 1]),
        }

    denom2 = 1.0 / (omega2_val - de_mtx + 1j * eta_val)   # (D,D)
    denom2_sq = denom2 ** 2
    s_om2 = 1.0 / (omega2_val + 1j * eta_val)             # scalar
    return K, denom2, denom2_sq, s_om2


def _process_kpoint_fast(
    system, k, dim, dir_chars, directions,
    omega1list, omega2_val, omega2_mtx_unused, eta_val, eta_mtx_unused,
    eflist, kT, nef, nomega,
    # Pre-built k-only quantities (pass None to build internally)
    _k_data=None,
    # Souza regularization for 1/w_{nm}; defaults to eta_val when None.
    eta_sos=None,
    # FreqIntegralSpec: switches the omega axis from sampled to integrated.
    freq_integral=None,
    # Wannier-gauge position correction; see calc/wannier_gauge.py.
    wannier_r=WANNIER_R_DEFAULT,
):
    """Vectorized _process_kpoint: broadcasts over all ef and omega at once.

    Near-degeneracy handling uses Souza Lorentzian regularization:
      inv_de = w_{nm} / (w_{nm}^2 + eta_sos^2)
    bounded at w->0, smooth everywhere.  Ensures that the three Sipe
    sub-terms of dk_rmtx sum exactly to the total (no NaN cleanup).

    When `freq_integral` is a FreqIntegralSpec, `omega1list` is ignored and
    the length-W axis of every returned array indexes that spec's channels
    (analytic frequency integrals plus lower-endpoint coefficients) instead of
    photon energies.  Phase 3 is unchanged either way: it is linear in the
    omega-dependent kernels built by `_build_omega_kernels`.

    `wannier_r` gates the Wannier-gauge position correction (see
    calc/wannier_gauge.py); it is inert for systems without
    ``wannier_r_matrices``.  Note it applies only to the Phase-1 build: when
    `_k_data` is supplied the operators are taken as given and the flag has no
    effect, since whoever built that cache already decided the question.
    """
    from ..bloch import get_H_v, diagonalize_hk

    if eta_sos is None:
        eta_sos = eta_val

    # ================================================================
    # Phase 1: k-only setup (same as original)
    # ================================================================
    if _k_data is not None:
        ek, vmtx, vvmtx, rmtx, dk_rmtx, Delta, de_mtx, inv_de = _k_data
        dk_rmtx_terms = None  # not available from cached data
    else:
        H, S, vtb = get_H_v(system, k, order=2)
        ek, psi = diagonalize_hk(H, S, eigenvectors=True)

        de_mtx = ek[:, None] - ek[None, :]
        # Souza-style Lorentzian regularization of the bare 1/w_{nm}
        nondeg = np.abs(de_mtx) > 1e-12
        inv_de = np.where(nondeg, de_mtx / (de_mtx ** 2 + eta_sos ** 2), 0.0)

        dir_pairs = set()
        for abc in directions:
            a, b, c = abc
            dir_pairs.update([b+c, c+b, b+a, c+a])
        for d1 in dir_chars:
            for d2 in dir_chars:
                dir_pairs.add(d1 + d2)

        vmtx = {d: psi.conj().T @ vtb[d] @ psi for d in dir_chars}
        vvmtx = {}
        for pair in dir_pairs:
            if pair in vtb:
                vvmtx[pair] = psi.conj().T @ vtb[pair] @ psi

        Delta = {}
        rmtx = {}
        for d in dir_chars:
            vd = np.diag(vmtx[d])
            Delta[d] = vd[:, None] - vd[None, :]
            rmtx[d] = -1j * vmtx[d] * inv_de

        from . nonlinear_optical import _compute_dk_rmtx
        dk_rmtx = {}
        dk_rmtx_terms = {}
        for d1 in dir_chars:
            dk_rmtx[d1] = {}
            dk_rmtx_terms[d1] = {}
            for d2 in dir_chars:
                dk_rmtx[d1][d2], dk_rmtx_terms[d1][d2] = _compute_dk_rmtx(
                    vmtx[d1], vmtx[d2], vvmtx[d1+d2],
                    Delta[d1], Delta[d2], de_mtx, inv_de, dim,
                    return_terms=True)

        # --- Wannier position operator corrections (arXiv:1804.04030) ---
        A_W, dA_W = compute_A_W_k(system, k, dir_chars, enabled=wannier_r)
        if A_W is not None:
            A_H = {}
            for d in dir_chars:
                A_H[d] = psi.conj().T @ A_W[d] @ psi

            a_H = {}
            for d in dir_chars:
                a_H[d] = A_H[d].copy()
                np.fill_diagonal(a_H[d], 0.0)

            # Eq. 22: r_nm += a_nm; save internal-only r_bar
            r_bar = {}
            for d in dir_chars:
                r_bar[d] = rmtx[d].copy()
                rmtx[d] = rmtx[d] + a_H[d]

            dA_H = {}
            for d1 in dir_chars:
                dA_H[d1] = {}
                for d2 in dir_chars:
                    dA_H[d1][d2] = psi.conj().T @ dA_W[d1][d2] @ psi

            xi_diag = {}
            for d in dir_chars:
                xi_diag[d] = np.diag(A_H[d]).real.copy()

            # Eq. 36 corrections to dk_rmtx.  Registered also as a
            # separate sub-term 'wannier_corr' so the Sipe sub-terms
            # (delta, d2H, 3band, wannier_corr) sum exactly to
            # the Wannier-corrected dk_rmtx.
            for d1 in dir_chars:
                for d2 in dir_chars:
                    corr = dA_H[d1][d2].copy()
                    np.fill_diagonal(corr, 0.0)
                    corr += r_bar[d2] @ a_H[d1] - a_H[d1] @ r_bar[d2]
                    xi_diff = xi_diag[d2][:, None] - xi_diag[d2][None, :]
                    corr += 1j * xi_diff * a_H[d1]
                    np.fill_diagonal(corr, 0.0)
                    dk_rmtx[d1][d2] = dk_rmtx[d1][d2] + corr
                    if dk_rmtx_terms is not None and d1 in dk_rmtx_terms \
                            and d2 in dk_rmtx_terms[d1]:
                        dk_rmtx_terms[d1][d2]['wannier_corr'] = corr

    # ================================================================
    # Phase 2: Vectorized ef/omega computation
    # ================================================================
    # Shapes: E=nef, W=nomega, D=dim

    # --- Fermi functions for ALL ef values at once ---
    # x[e, n] = (ek[n] - ef[e]) / kT
    x = (ek[None, :] - eflist[:, None]) / kT  # (E, D)
    x_clip = np.clip(x, -500, 500)
    f_arr = 1.0 / (1.0 + np.exp(x_clip))  # (E, D)
    de_f = -1.0 / kT * np.exp(x_clip) / (1.0 + np.exp(x_clip))**2  # (E, D)

    x2 = -x  # (ef - ek) / kT
    x2_clip = np.clip(x2, -500, 500)
    with np.errstate(divide='ignore', invalid='ignore'):
        de2_f = -2.0 / kT**2 * (1.0 / np.sinh(x2_clip))**3 * np.sinh(x2_clip / 2.0)**4

    far = np.abs(x) > 10
    de_f[far] = 0.0
    de2_f[far] = 0.0
    de2_f = np.where(np.isfinite(de2_f), de2_f, 0.0)

    # f_mtx[e, n, m] = f[e,n] - f[e,m]
    f_mtx = f_arr[:, :, None] - f_arr[:, None, :]  # (E, D, D)

    # --- Omega-dependent kernels, for the whole omega axis at once ---
    # Sampled mode: one column per photon energy.  Integrated mode: one column
    # per freq_integral channel.  Phase 3 never touches omega directly.
    K, denom2, denom2_sq, s_om2 = _build_omega_kernels(
        de_mtx, omega1list, omega2_val, eta_val, freq_integral
    )
    denom1 = K['d1']    # (D, D, W)
    denom12 = K['d12']  # (D, D, W)

    # --- Per-direction precomputations ---
    # dk_f[e, d, n] = v_nn^d * de_f[e, n]
    vdiag = {d: np.diag(vmtx[d]) for d in dir_chars}  # (D,)
    dk_f_all = {d: vdiag[d][None, :] * de_f for d in dir_chars}  # (E, D)
    # dk_f_mtx[e, d, n, m] = dk_f[e,d,n] - dk_f[e,d,m]
    dk_f_mtx_all = {d: dk_f_all[d][:, :, None] - dk_f_all[d][:, None, :]
                    for d in dir_chars}  # (E, D, D)
    # d2k_f[e, d1, d2, n] = vv_diag * de_f + v1*v2*de2_f
    vv_diag = {}
    for d1 in dir_chars:
        for d2 in dir_chars:
            vv_diag[d1+d2] = np.diag(vvmtx[d1+d2])
    d2k_f_all = {}
    for d1 in dir_chars:
        for d2 in dir_chars:
            pair = d1 + d2
            d2k_f_all[pair] = (vv_diag[pair][None, :] * de_f +
                               vdiag[d1][None, :] * vdiag[d2][None, :] * de2_f)  # (E, D)

    # ================================================================
    # Phase 3: Compute all chi terms vectorized
    # ================================================================
    from .nonlinear_optical import CHI_ALL_NAMES

    kpt = {}
    for name in CHI_ALL_NAMES:
        kpt[name] = {}
        for abc in directions:
            kpt[name][abc] = np.zeros((nef, nomega), dtype=complex)

    for abc in directions:
        dir_a, dir_b, dir_c = abc
        va = vmtx[dir_a]  # (D, D)

        # ---- chi_ii: intra-intra ----
        # rho_ii is diagonal, depends on ef through d2k_f, omega through scalar denoms
        # chi_ii = sum_n va[n,n] * (d2k_f_bc[e,n] * s_om2 + d2k_f_cb[e,n] * s_om1[w]) * s_om12[w]
        va_diag = np.diag(va)  # (D,)
        d2k_bc = d2k_f_all[dir_b + dir_c]  # (E, D)
        d2k_cb = d2k_f_all[dir_c + dir_b]  # (E, D)
        # term1[e, w] = s_om12[w] * sum_n va[n,n] * (d2k_bc[e,n]*s_om2 + d2k_cb[e,n]*s_om1[w])
        t_bc = np.einsum('n,en->e', va_diag, d2k_bc)  # (E,)
        t_cb = np.einsum('n,en->e', va_diag, d2k_cb)  # (E,)
        # (-i)^2 = -1 from the two intraband i d/dk vertices (2026-07 audit).
        # s_om2 is omega-independent, so it multiplies the s_om12 kernel;
        # the s_om12*s_om1 product is a genuine two-pole kernel.
        kpt['chi_ii'][abc] = -(t_bc[:, None] * (s_om2 * K['om12'])[None, :]
                               + t_cb[:, None] * K['om12_om1'][None, :])

        # ---- chi_ee1: inter-inter ----
        # G1[e,n,m,w] = f_mtx[e,n,m] * rmtx_b[n,m] * denom1[n,m,w]
        # rho_ee1 = (G1 @ r_c - r_c @ G1) * denom12, traced against v_a:
        #   chi_ee1[e,w] = sum_{n,m,p} va[n,m] * denom12[m,n,w] * (
        #       Fr_b[e,m,p] * r_c[p,n] * denom1[m,p,w]      <- term A
        #     - r_c[m,p] * Fr_b[e,p,n] * denom1[p,n,w] )    <- term B
        Fr_b = f_mtx * rmtx[dir_b][None, :, :]  # (E, D, D)
        r_c = rmtx[dir_c]  # (D, D)
        r_b = rmtx[dir_b]  # (D, D)

        # This is the one term whose two omega-dependent factors sit at
        # *different* band pairs -- denom12 at (m,n), denom1 at (m,p) or (p,n).
        # The product therefore cannot be folded into a two-index kernel, and
        # once integrated it does not factorize at all, so both modes go
        # through the same (D,D,D,W) kernels.  D^3 W is small.
        chi_A = np.einsum('nm,pn,emp,mnpw->ew',
                          va, r_c, Fr_b, K['ee1A'], optimize=True)
        chi_B = np.einsum('nm,mp,epn,mnpw->ew',
                          va, r_c, Fr_b, K['ee1B'], optimize=True)

        kpt['chi_ee1'][abc] = chi_A - chi_B

        # va_d12[n,m,w] = va[n,m] * denom12[m,n,w]  -- note the index swap.
        # Used by chi_ee2, whose second omega factor (denom2) is a constant.
        va_d12 = va[:, :, None] * denom12.transpose(1, 0, 2)  # (D,D,W)

        # ---- chi_ee2: same structure, swap b<->c and use denom2 (omega2-dependent) ----
        Fr_c = f_mtx * rmtx[dir_c][None, :, :]  # (E, D, D)
        # G2 uses denom2 which is omega2-dependent (scalar omega2, so denom2 is (D,D) not (D,D,W))
        # chi_ee2[e,w] = sum_{n,m,p} va_d12[n,m,w] * (
        #     Fr_c[e,m,p] * r_b[p,n] * denom2[m,p]
        #   - r_b[m,p] * Fr_c[e,p,n] * denom2[p,n] )
        chi_A2 = np.einsum('pmw,emp,mp->ew',
                           np.einsum('pn,nmw->pmw', r_b, va_d12),
                           Fr_c, denom2)
        chi_B2 = np.einsum('nmw,epn,pn,mp->ew', va_d12, Fr_c, denom2, r_b)
        kpt['chi_ee2'][abc] = chi_A2 - chi_B2

        # ---- chi_ei terms ----
        dk_b = dk_rmtx[dir_b][dir_c]  # (D, D)
        dk_c = dk_rmtx[dir_c][dir_b]  # (D, D)
        Db = Delta[dir_b]  # (D, D)
        Dc = Delta[dir_c]  # (D, D)

        # Common denominator products.  The *_d1 pair multiplies two
        # omega-dependent factors and so comes from a two-pole kernel; the
        # *_d2 pair only rescales the d12 kernel by the omega-independent
        # denom2, which passes straight through the frequency integral.
        d12_d1 = K['d12_d1']  # (D, D, W)
        d12_d1sq = K['d12_d1sq']  # (D, D, W)
        d12_d2 = denom12 * denom2[:, :, None]  # (D, D, W)
        d12_d2sq = denom12 * denom2_sq[:, :, None]  # (D, D, W)

        # 2026-07 audit corrections (see nonlinear_optical._process_kpoint):
        #  (a) every chi_ei term carries the -i of the intraband vertex
        #      i d/dk in the length-gauge coupling E.(r_e + i d/dk);
        #  (b) t5/t6: d/dk_c (1/(w - w_nm)) = +Delta^c/(w - w_nm)^2, so the
        #      legacy leading minus sign is removed (net -1j prefactor).

        # t1: -i * denom12 * f_mtx * denom1 * dk_rmtx_bc  (Sipe derivative, ω₁)
        K_t1 = va.T * dk_b  # (D, D): K_t1[m,n] = va[n,m] * dk_b[m,n]
        kpt['chi_eit1'][abc] = -1j * np.einsum('mn,emn,mnw->ew', K_t1, f_mtx, d12_d1)

        # t2: same but with dk_c and denom2  (Sipe derivative, ω₂)
        K_t2 = va.T * dk_c
        kpt['chi_eit1'][abc] += -1j * np.einsum('mn,emn,mnw->ew', K_t2, f_mtx, d12_d2)

        # t3: -i * denom12 * rmtx_b * dk_f_mtx_c * denom1  (dk_f, ω₁)
        K_t3 = va.T * r_b  # (D, D)
        kpt['chi_eit2'][abc] = -1j * np.einsum('mn,emn,mnw->ew', K_t3, dk_f_mtx_all[dir_c], d12_d1)

        # t4: -i * denom12 * rmtx_c * dk_f_mtx_b * denom2  (dk_f, ω₂)
        K_t4 = va.T * r_c
        kpt['chi_eit2'][abc] += -1j * np.einsum('mn,emn,mnw->ew', K_t4, dk_f_mtx_all[dir_b], d12_d2)

        # t5: -i * denom12 * rmtx_b * f_mtx * Delta_c * denom1^2  (Delta*r, ω₁)
        K_t5 = va.T * r_b * Dc  # (D, D)
        kpt['chi_eit3'][abc] = -1j * np.einsum('mn,emn,mnw->ew', K_t5, f_mtx, d12_d1sq)

        # t6: -i * denom12 * rmtx_c * f_mtx * Delta_b * denom2^2  (Delta*r, ω₂)
        K_t6 = va.T * r_c * Db
        kpt['chi_eit3'][abc] += -1j * np.einsum('mn,emn,mnw->ew', K_t6, f_mtx, d12_d2sq)

        # Compose ei1 = t1+t3+t5, ei2 = t2+t4+t6
        ei1_sipe = -1j * np.einsum('mn,emn,mnw->ew', K_t1, f_mtx, d12_d1)
        ei1_dk_f = -1j * np.einsum('mn,emn,mnw->ew', K_t3, dk_f_mtx_all[dir_c], d12_d1)
        ei1_delta_r = -1j * np.einsum('mn,emn,mnw->ew', K_t5, f_mtx, d12_d1sq)
        kpt['chi_ei1'][abc] = ei1_sipe + ei1_dk_f + ei1_delta_r

        ei2_sipe = -1j * np.einsum('mn,emn,mnw->ew', K_t2, f_mtx, d12_d2)
        ei2_dk_f = -1j * np.einsum('mn,emn,mnw->ew', K_t4, dk_f_mtx_all[dir_b], d12_d2)
        ei2_delta_r = -1j * np.einsum('mn,emn,mnw->ew', K_t6, f_mtx, d12_d2sq)
        kpt['chi_ei2'][abc] = ei2_sipe + ei2_dk_f + ei2_delta_r

        # 5-term breakdown of chi_ei1 and chi_ei2
        # Split the Sipe derivative (t1/t2) into delta, d2H, 3band sub-terms
        kpt['chi_ei1_dk_f'][abc] = ei1_dk_f
        kpt['chi_ei1_delta_r'][abc] = ei1_delta_r
        kpt['chi_ei2_dk_f'][abc] = ei2_dk_f
        kpt['chi_ei2_delta_r'][abc] = ei2_delta_r

        if dk_rmtx_terms is not None:
            sipe_bc = dk_rmtx_terms[dir_b][dir_c]
            sipe_cb = dk_rmtx_terms[dir_c][dir_b]
            # 'wannier_corr' only present when the system carries
            # wannier position operators (arXiv:1804.04030 corrections).
            for sub in ('delta', 'd2H', '3band', 'wannier_corr'):
                if sub in sipe_bc:
                    K_sub1 = va.T * sipe_bc[sub]
                    kpt[f'chi_ei1_sipe_{sub}'][abc] = -1j * np.einsum(
                        'mn,emn,mnw->ew', K_sub1, f_mtx, d12_d1)
                if sub in sipe_cb:
                    K_sub2 = va.T * sipe_cb[sub]
                    kpt[f'chi_ei2_sipe_{sub}'][abc] = -1j * np.einsum(
                        'mn,emn,mnw->ew', K_sub2, f_mtx, d12_d2)
        else:
            # Fallback: full Sipe term without sub-decomposition
            kpt['chi_ei1_sipe_delta'][abc] = ei1_sipe
            kpt['chi_ei2_sipe_delta'][abc] = ei2_sipe

        # ---- chi_ie1, chi_ie2: intra-inter ----
        # rho_ie1[m,n] = denom12[m,n,w] * dk_f_mtx_b[e,m,n] * r_c[m,n] / (omega1 - i*eta)
        # chi_ie1 = trace(rho_ie1 @ va) = sum_{n,m} rho_ie1[n,m] * va[m,n]
        #         = sum_{n,m} va[m,n] * denom12[n,m,w] * dk_f_mtx_b[e,n,m] * r_c[n,m] * s_om1[w]
        # Wait: trace(rho @ va) = sum_n (rho @ va)[n,n] = sum_{n,m} rho[n,m]*va[m,n]
        K_ie1 = va * r_c.T  # va[m,n] * r_c[n,m] -> K_ie1[m,n]. But we want rho indexed [n,m].
        # Let me reindex: chi_ie1[e,w] = sum_{n,m} dk_f_mtx_b[e,n,m] * r_c[n,m] * va[m,n] * denom12[n,m,w] * s_om1[w]
        # -i from the intraband first vertex (2026-07 audit).
        K_ie1 = r_c * va.T  # (D,D): K_ie1[n,m] = r_c[n,m] * va[m,n]
        # denom12 * s_om1: two omega-dependent factors -> two-pole kernel.
        kpt['chi_ie1'][abc] = -1j * np.einsum(
            'nm,enm,nmw->ew', K_ie1, dk_f_mtx_all[dir_b], K['d12_om1'])

        # chi_ie2: uses dk_f_mtx_c, r_b, s_om2 (scalar)
        K_ie2 = r_b * va.T
        kpt['chi_ie2'][abc] = -1j * s_om2 * np.einsum('nm,enm,nmw->ew', K_ie2, dk_f_mtx_all[dir_c], denom12)

        # ---- chi_e1, chi_i1, chi_e2, chi_i2 ----
        # NOTE: these are unphysical artefacts of the interband/intraband
        # decomposition. Computed for diagnostics but excluded from chi_total.
        dk_ba = dk_rmtx[dir_b][dir_a]  # (D, D)
        dk_ca = dk_rmtx[dir_c][dir_a]  # (D, D)

        # chi_e1 = trace(rho1_e @ dk_ba) where rho1_e = r_c * f_mtx / (omega2 - de - i*eta)
        # = sum_{n,m} (r_c * f_mtx * denom2)[e,n,m] * dk_ba[m,n]  -- no omega dep (omega2 scalar)!
        # = sum_{n,m} r_c[n,m] * f_mtx[e,n,m] * denom2[n,m] * dk_ba[m,n]
        # chi_e1 and chi_i1 have NO omega dependence at all, so their kernel is
        # the bare weight int dw w^(-p).  That has no b -> inf limit for p <= 1,
        # and K['const'] is NaN there; both terms are unphysical and excluded
        # from chi_total, so nothing downstream is contaminated.
        K_e1 = r_c * denom2 * dk_ba.T  # (D, D)
        kpt['chi_e1'][abc] = (np.einsum('nm,enm->e', K_e1, f_mtx)[:, None]
                              * K['const'][None, :])

        # chi_i1 = trace(diag(dk_f_c / (om2 - i*eta)) @ dk_ba)
        # = sum_n dk_f_c[e,n] * s_om2 * dk_ba[n,n]
        dk_ba_diag = np.diag(dk_ba)  # (D,)
        kpt['chi_i1'][abc] = (s_om2 * np.einsum('en,n->e', dk_f_all[dir_c], dk_ba_diag)[:, None]
                              * K['const'][None, :])

        # chi_e2 = trace(rho2_e @ dk_ca) where rho2_e = r_b * f_mtx / (omega1 - de - i*eta)
        # = sum_{n,m} r_b[n,m] * f_mtx[e,n,m] * denom1[n,m,w] * dk_ca[m,n]
        K_e2 = r_b * dk_ca.T  # (D, D)
        kpt['chi_e2'][abc] = np.einsum('nm,enm,nmw->ew', K_e2, f_mtx, denom1)

        # chi_i2 = trace(diag(dk_f_b / (om1 - i*eta)) @ dk_ca)
        # = sum_n dk_f_b[e,n] * s_om1[w] * dk_ca[n,n]
        dk_ca_diag = np.diag(dk_ca)
        kpt['chi_i2'][abc] = (np.einsum('en,n->e', dk_f_all[dir_b], dk_ca_diag)[:, None]
                              * K['om1'][None, :])

    return kpt
