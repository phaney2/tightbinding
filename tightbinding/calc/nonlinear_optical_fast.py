"""Vectorized χ^(2) — eliminates ef/omega Python loops via broadcasting.

Drop-in replacement for _process_kpoint. All arrays carry (nef, dim, dim, nomega)
shape; einsum replaces trace-of-matmul patterns.

Notation:
  E = nef, W = nomega, D = dim
  indices: e=ef, w=omega, n,m,p = band
"""

import numpy as np


def _process_kpoint_fast(
    system, k, dim, dir_chars, directions,
    omega1list, omega2_val, omega2_mtx_unused, eta_val, eta_mtx_unused,
    eflist, kT, nef, nomega,
    # Pre-built k-only quantities (pass None to build internally)
    _k_data=None,
):
    """Vectorized _process_kpoint: broadcasts over all ef and omega at once."""
    from ..bloch import get_H_v, diagonalize_hk

    # ================================================================
    # Phase 1: k-only setup (same as original)
    # ================================================================
    if _k_data is not None:
        ek, vmtx, vvmtx, rmtx, dk_rmtx, Delta, de_mtx, inv_de = _k_data
    else:
        H, S, vtb = get_H_v(system, k, order=2)
        ek, psi = diagonalize_hk(H, S, eigenvectors=True)

        de_mtx = ek[:, None] - ek[None, :]
        de_cutoff = eta_val
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_de = np.where(np.abs(de_mtx) > de_cutoff, 1.0 / de_mtx, 0.0)

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
        for d1 in dir_chars:
            dk_rmtx[d1] = {}
            for d2 in dir_chars:
                dk_rmtx[d1][d2] = _compute_dk_rmtx(
                    vmtx[d1], vmtx[d2], vvmtx[d1+d2],
                    Delta[d1], Delta[d2], de_mtx, inv_de, dim)

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

    # --- Omega denominators for ALL omega values at once ---
    omega1 = omega1list  # (W,)
    omega12 = omega1list + omega2_val  # (W,)

    # denom1[n, m, w] = 1 / (omega1[w] - de[n,m] - i*eta)
    denom1 = 1.0 / (omega1[None, None, :] - de_mtx[:, :, None] - 1j * eta_val)  # (D, D, W)
    # denom2[n, m, w] = 1 / (omega2 - de[n,m] - i*eta)  — omega2 is scalar
    denom2 = 1.0 / (omega2_val - de_mtx - 1j * eta_val)  # (D, D)
    # denom12[n, m, w] = 1 / (omega12[w] - de[n,m] - 2i*eta)
    denom12 = 1.0 / (omega12[None, None, :] - de_mtx[:, :, None] - 2j * eta_val)  # (D, D, W)
    # scalar omega denoms
    s_om1 = 1.0 / (omega1 - 1j * eta_val)  # (W,)
    s_om2 = 1.0 / (omega2_val - 1j * eta_val)  # scalar
    s_om12 = 1.0 / (omega12 - 2j * eta_val)  # (W,)

    # denom1_sq[n, m, w] = 1 / (omega1[w] - de[n,m] - i*eta)^2
    denom1_sq = denom1 ** 2  # (D, D, W)
    # denom2_sq — omega2 is scalar
    denom2_sq = denom2 ** 2  # (D, D)

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
    from .nonlinear_optical import CHI_NAMES

    kpt = {}
    for name in CHI_NAMES:
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
        kpt['chi_ii'][abc] = s_om12[None, :] * (t_bc[:, None] * s_om2 + t_cb[:, None] * s_om1[None, :])

        # ---- chi_ee1: inter-inter ----
        # G1[e,n,m,w] = f_mtx[e,n,m] * rmtx_b[n,m] * denom1[n,m,w]
        Fr_b = f_mtx * rmtx[dir_b][None, :, :]  # (E, D, D)
        # G1 @ r_c: sum_p G1[e,n,p,w] * r_c[p,m] = sum_p Fr_b[e,n,p] * denom1[n,p,w] * r_c[p,m]
        # rho_ee1[e,n,m,w] = (sum_p Fr_b[e,n,p]*r_c[p,m]*denom1[n,p,w] - r_c[n,p]*Fr_b[e,p,m]*denom1[p,m,w]) * denom12[n,m,w]
        # chi_ee1[e,w] = sum_{n,m} va[n,m] * rho_ee1[e,m,n,w]
        r_c = rmtx[dir_c]  # (D, D)
        r_b = rmtx[dir_b]  # (D, D)
        # Use einsum: Fr_b[e,n,p] * denom1[n,p,w] * r_c[p,m] summed over p -> (E, D, D, W)
        # Then subtract the swapped term, multiply by denom12, trace with va
        # To keep memory manageable, compute trace directly:
        # chi_ee1[e,w] = sum_{n,m,p} va[n,m] * denom12[m,n,w] * (
        #     Fr_b[e,m,p] * r_c[p,n] * denom1[m,p,w]
        #   - r_c[m,p] * Fr_b[e,p,n] * denom1[p,n,w] )
        # Term A: sum_{n,m,p} va[n,m] * denom12[m,n,w] * Fr_b[e,m,p] * r_c[p,n] * denom1[m,p,w]
        # Term B: sum_{n,m,p} va[n,m] * denom12[m,n,w] * r_c[m,p] * Fr_b[e,p,n] * denom1[p,n,w]

        # For dim=9, we can afford to build (D,D,W) intermediates and contract with (E,D,D)
        # Term A: group (m,p) first
        #   A_mp[m,p,w] = sum_n va[n,m] * denom12[m,n,w] * r_c[p,n] = sum_n (va * r_c^T)[n,m,p???]
        # Actually, let's do it more simply. For D=9, the full 4-index tensor is tiny.

        # Build va_d12[n,m,w] = va[n,m] * denom12[m,n,w]  -- note index swap in denom12
        va_d12 = va[:, :, None] * denom12.transpose(1, 0, 2)  # va[n,m] * denom12[m,n,w] -> (D,D,W)

        # Term A: sum_{n,m,p} va_d12[n,m,w] * Fr_b[e,m,p] * r_c[p,n] * denom1[m,p,w]
        # = sum_{m,p} (sum_n va_d12[n,m,w] * r_c[p,n]) * Fr_b[e,m,p] * denom1[m,p,w]
        # Let Q_A[p,m,w] = sum_n va_d12[n,m,w] * r_c[p,n] = sum_n r_c[p,n] * va[n,m] * denom12[m,n,w]
        Q_A = np.einsum('pn,nmw->pmw', r_c, va_d12)  # (D, D, W)
        # chi_A[e,w] = sum_{m,p} Q_A[p,m,w] * Fr_b[e,m,p] * denom1[m,p,w]
        QA_d1 = Q_A * denom1.transpose(1, 0, 2)  # Q_A[p,m,w] * denom1[m,p,w] -- need denom1 as [p,m,w]
        # Wait, denom1[n,m,w] = 1/(om1[w] - de[n,m] - i*eta). We need denom1[m,p,w].
        # denom1 is already indexed as [first_band, second_band, w].
        # Q_A[p,m,w] needs denom1[m,p,w]
        QA_d1 = Q_A * denom1  # Q_A is (D_p, D_m, W), denom1 is (D_m?, D_p?, W)...

        # Let me be more careful with indices. denom1[i,j,w] = 1/(om1[w] - de[i,j]).
        # We need denom1 with first index=m, second=p: denom1[m,p,w]. That's just denom1[m,p,w] as-is.
        # But Q_A has shape (p, m, w). So we need denom1 transposed to (p, m, w) form? No.
        # Q_A[p,m,w] * denom1[m,p,w]: we need to match indices.
        # Use einsum for clarity:
        chi_A = np.einsum('pmw,emp,mpw->ew', Q_A, Fr_b, denom1)  # (E, W)

        # Term B: sum_{n,m,p} va_d12[n,m,w] * r_c[m,p] * Fr_b[e,p,n] * denom1[p,n,w]
        # = sum_{m,p} r_c[m,p] * (sum_n va_d12[n,m,w] * Fr_b[e,p,n] * denom1[p,n,w])
        # Let's compute: sum_n va_d12[n,m,w] * denom1[p,n,w] * Fr_b[e,p,n]
        # = sum_n va[n,m]*denom12[m,n,w] * denom1[p,n,w] * Fr_b[e,p,n]
        # Q_B[e,m,p,w] = sum_n va_d12[n,m,w] * denom1[p,n,w] * Fr_b[e,p,n]
        chi_B = np.einsum('nmw,pnw,epn,mp->ew', va_d12, denom1, Fr_b, r_c)

        kpt['chi_ee1'][abc] = chi_A - chi_B

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

        # t1: denom12 * f_mtx * denom1 * dk_rmtx_bc
        # trace(va @ t1)[e,w] = sum_{n,m} va[n,m] * denom12[m,n,w] * f_mtx[e,m,n] * denom1[m,n,w] * dk_b[m,n]
        K1 = dk_b * denom2  # (D,D) -- dk_rmtx_cb with denom2 for t2
        # t1: sum_{n,m} va[n,m] * f_mtx[e,m,n] * dk_b[m,n] * denom12[m,n,w] * denom1[m,n,w]
        W1_nm = va.T * dk_b  # va[n,m].T = va[m,n]... let me be careful
        # trace(va @ M) = sum_{n} (va @ M)[n,n] = sum_{n,m} va[n,m] * M[m,n]
        # So chi = sum_{n,m} va[n,m] * rho[m,n]
        # For t1: rho[m,n] = denom12[m,n,w] * f_mtx[m,n] * denom1[m,n,w] * dk_b[m,n]
        # (but f_mtx has ef dim)
        # Precompute the k-only part: K_t1[m,n] = va[n,m] * dk_b[m,n] (element-wise product with transposed va)
        K_t1 = va.T * dk_b  # (D, D): K_t1[m,n] = va[n,m] * dk_b[m,n]
        # chi_t1[e,w] = sum_{m,n} K_t1[m,n] * f_mtx[e,m,n] * denom12[m,n,w] * denom1[m,n,w]
        d12_d1 = denom12 * denom1  # (D, D, W)
        kpt['chi_eit1'][abc] = np.einsum('mn,emn,mnw->ew', K_t1, f_mtx, d12_d1)

        # t2: same but with dk_c and denom2
        K_t2 = va.T * dk_c
        d12_d2 = denom12 * denom2[:, :, None]  # (D, D, W)
        kpt['chi_eit1'][abc] += np.einsum('mn,emn,mnw->ew', K_t2, f_mtx, d12_d2)

        # t3: denom12 * rmtx_b * dk_f_mtx_c * denom1
        # rho[m,n] = denom12[m,n,w] * r_b[m,n] * dk_f_mtx_c[e,m,n] * denom1[m,n,w]
        K_t3 = va.T * r_b  # (D, D)
        kpt['chi_eit2'][abc] = np.einsum('mn,emn,mnw->ew', K_t3, dk_f_mtx_all[dir_c], d12_d1)

        # t4: denom12 * rmtx_c * dk_f_mtx_b * denom2
        K_t4 = va.T * r_c
        kpt['chi_eit2'][abc] += np.einsum('mn,emn,mnw->ew', K_t4, dk_f_mtx_all[dir_b], d12_d2)

        # t5: -denom12 * rmtx_b * f_mtx * Delta_c * denom1^2
        K_t5 = va.T * r_b * Dc  # (D, D)
        d12_d1sq = denom12 * denom1_sq  # (D, D, W)
        kpt['chi_eit3'][abc] = -np.einsum('mn,emn,mnw->ew', K_t5, f_mtx, d12_d1sq)

        # t6: -denom12 * rmtx_c * f_mtx * Delta_b * denom2^2
        K_t6 = va.T * r_c * Db
        d12_d2sq = denom12 * denom2_sq[:, :, None]  # (D, D, W)
        kpt['chi_eit3'][abc] -= np.einsum('mn,emn,mnw->ew', K_t6, f_mtx, d12_d2sq)

        # Compose ei1 = t1+t3+t5, ei2 = t2+t4+t6
        # We computed eit1 = t1+t2, eit2 = t3+t4, eit3 = t5+t6. Recompute ei1/ei2 directly.
        kpt['chi_ei1'][abc] = (
            np.einsum('mn,emn,mnw->ew', K_t1, f_mtx, d12_d1) +
            np.einsum('mn,emn,mnw->ew', K_t3, dk_f_mtx_all[dir_c], d12_d1) -
            np.einsum('mn,emn,mnw->ew', K_t5, f_mtx, d12_d1sq)
        )
        kpt['chi_ei2'][abc] = (
            np.einsum('mn,emn,mnw->ew', K_t2, f_mtx, d12_d2) +
            np.einsum('mn,emn,mnw->ew', K_t4, dk_f_mtx_all[dir_b], d12_d2) -
            np.einsum('mn,emn,mnw->ew', K_t6, f_mtx, d12_d2sq)
        )

        # ---- chi_ie1, chi_ie2: intra-inter ----
        # rho_ie1[m,n] = denom12[m,n,w] * dk_f_mtx_b[e,m,n] * r_c[m,n] / (omega1 - i*eta)
        # chi_ie1 = trace(rho_ie1 @ va) = sum_{n,m} rho_ie1[n,m] * va[m,n]
        #         = sum_{n,m} va[m,n] * denom12[n,m,w] * dk_f_mtx_b[e,n,m] * r_c[n,m] * s_om1[w]
        # Wait: trace(rho @ va) = sum_n (rho @ va)[n,n] = sum_{n,m} rho[n,m]*va[m,n]
        K_ie1 = va * r_c.T  # va[m,n] * r_c[n,m] -> K_ie1[m,n]. But we want rho indexed [n,m].
        # Let me reindex: chi_ie1[e,w] = sum_{n,m} dk_f_mtx_b[e,n,m] * r_c[n,m] * va[m,n] * denom12[n,m,w] * s_om1[w]
        K_ie1 = r_c * va.T  # (D,D): K_ie1[n,m] = r_c[n,m] * va[m,n]
        kpt['chi_ie1'][abc] = np.einsum('nm,enm,nmw,w->ew', K_ie1, dk_f_mtx_all[dir_b], denom12, s_om1)

        # chi_ie2: uses dk_f_mtx_c, r_b, s_om2 (scalar)
        K_ie2 = r_b * va.T
        kpt['chi_ie2'][abc] = s_om2 * np.einsum('nm,enm,nmw->ew', K_ie2, dk_f_mtx_all[dir_c], denom12)

        # ---- chi_e1, chi_i1, chi_e2, chi_i2 ----
        dk_ba = dk_rmtx[dir_b][dir_a]  # (D, D)
        dk_ca = dk_rmtx[dir_c][dir_a]  # (D, D)

        # chi_e1 = trace(rho1_e @ dk_ba) where rho1_e = r_c * f_mtx / (omega2 - de - i*eta)
        # = sum_{n,m} (r_c * f_mtx * denom2)[e,n,m] * dk_ba[m,n]  -- no omega dep (omega2 scalar)!
        # = sum_{n,m} r_c[n,m] * f_mtx[e,n,m] * denom2[n,m] * dk_ba[m,n]
        K_e1 = r_c * denom2 * dk_ba.T  # (D, D)
        kpt['chi_e1'][abc] = np.einsum('nm,enm->e', K_e1, f_mtx)[:, None].repeat(nomega, axis=1)

        # chi_i1 = trace(diag(dk_f_c / (om2 - i*eta)) @ dk_ba)
        # = sum_n dk_f_c[e,n] * s_om2 * dk_ba[n,n]
        dk_ba_diag = np.diag(dk_ba)  # (D,)
        kpt['chi_i1'][abc] = s_om2 * np.einsum('en,n->e', dk_f_all[dir_c], dk_ba_diag)[:, None].repeat(nomega, axis=1)

        # chi_e2 = trace(rho2_e @ dk_ca) where rho2_e = r_b * f_mtx / (omega1 - de - i*eta)
        # = sum_{n,m} r_b[n,m] * f_mtx[e,n,m] * denom1[n,m,w] * dk_ca[m,n]
        K_e2 = r_b * dk_ca.T  # (D, D)
        kpt['chi_e2'][abc] = np.einsum('nm,enm,nmw->ew', K_e2, f_mtx, denom1)

        # chi_i2 = trace(diag(dk_f_b / (om1 - i*eta)) @ dk_ca)
        # = sum_n dk_f_b[e,n] * s_om1[w] * dk_ca[n,n]
        dk_ca_diag = np.diag(dk_ca)
        kpt['chi_i2'][abc] = np.einsum('en,n->e', dk_f_all[dir_b], dk_ca_diag)[:, None] * s_om1[None, :]

    return kpt
