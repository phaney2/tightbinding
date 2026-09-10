"""Wannier-gauge position-operator correction: one switch, one kernel.

`bloch.get_H_k` builds H(k) with Bloch phases exp(ik·(R + tau_m - tau_n)),
tau being the orbital centres in ``system.atompos``.  For point-like orbitals
sitting at tau (every ``build_system`` model) the position operator in that
gauge is exactly r = -i v/w with no intra-cell term.  Wannier functions have
a finite spread and off-diagonal dipoles, and a ``_tb.dat`` file carries
them as <0n|r|Rm>; the part of r that r = -i v/w misses is the Wannier-gauge
Berry connection

    A^(W)_{nm,a}(k) = Σ_R exp(ik·(R + tau_m - tau_n)) <0n|r_a|Rm> - tau_{n,a} δ_nm

computed by `compute_A_W_k` with the SAME tau the Bloch phases use, and
applied by `apply_wannier_correction` to every operator an engine forms:

    r^a_nm      += a^a_nm                      a = offdiag(U† A^(W) U)     (Eq. 22)
    r^{a;b}     += covariant-derivative piece  (the full expression is in the
                                                function docstring)
    v^a_nm      += i w_nm a^a_nm               the physical current vertex

while everything that is a derivative of H(k) itself — the group velocity
Delta, the Sipe sum rule, the band curvature, the inverse mass — keeps the
bare v.  The rule that fixes which is which: **every engine output must be
independent of the gauge the system was built in** (its atompos).
`examples/test_wannier_gauge.py` enforces that exactly, by re-expressing
point-like models in other gauges (where A^(W) is then a known diagonal) and
demanding the atomic-gauge answer back, and it checks the k-dependent term
against finite differences on a real MoS2 connection.

Every engine that builds a position operator reads the same switch from the
same place through `resolve_wannier_r`, gates the same kernel through the
``enabled`` argument of `compute_A_W_k`, and applies it through
`apply_wannier_correction`.  That is deliberate: before this module existed
the three engines each did their own thing and drifted apart silently, and
before the 2026-09 fix each carried its own copy of the Eq. 36 algebra, all
wrong in the same way.  Add a new engine with a position operator, and it
must go through these functions.

Config::

    system:
      wannier_tb: mos2_tb.dat
      wannier_centres: mos2_centres.xyz   # recommended, see wannier.py
      wannier_r: true         # <- here, NOT under calc:  (default)

The correction is a **no-op** for systems built with ``build_system`` +
``fill_hamiltonian`` and for ``_hr.dat`` input: they carry no
``wannier_r_matrices`` and the flag is inert either way.
"""

import numpy as np

from .. import parallel


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

# The correction is physically required for Wannier input, so True is the
# default.  False keeps the point-like-orbital approximation r = -i v/w (in
# the atomic gauge, since `build_system_from_tb` always builds atompos from
# the centres); it is a diagnostic setting, not a physics preference.  It was
# False from 2026-08-27 to 2026-09-10 while BUG_wannier_r_correction.md was
# open; that file records the defect and the fix.
WANNIER_R_DEFAULT = True

_CONFIG_KEY = 'wannier_r'

# One-shot bookkeeping so an MPI run prints each notice once, not once per
# engine call.  Keyed by (engine, tag).
_NOTICED = set()


def _notice_once(engine, tag, msg):
    key = (engine, tag)
    if key in _NOTICED:
        return
    _NOTICED.add(key)
    parallel.print_root(msg)


def reset_notices():
    """Forget which notices have been printed (for tests driving many runs)."""
    _NOTICED.clear()


def system_has_wannier_r(system) -> bool:
    """True if `system` carries Wannier position matrices, i.e. the flag bites.

    Only ``build_system_from_tb`` sets these — a ``_hr.dat`` system has a
    Hamiltonian but no position operator, and a TB_simple system needs none.
    """
    return (system is not None
            and getattr(system, 'wannier_r_matrices', None) is not None)


def validate_wannier_r(cfg) -> None:
    """Check the placement and type of the ``wannier_r`` switch.

    Raises ValueError for the old ``calc.wannier_r`` location and for
    non-boolean values.  Called by `resolve_wannier_r` and by
    `config.load_config`, so a YAML typo fails at load time rather than
    halfway through a BZ sweep.
    """
    calc = cfg.get('calc') or {}
    if _CONFIG_KEY in calc:
        raise ValueError(
            "calc.wannier_r has moved to system.wannier_r — it describes the "
            "gauge of the system's position operator, not the calculation, "
            "and every engine that builds an r operator now reads it from "
            "there.  Move the key into the 'system:' block."
        )

    sys_cfg = cfg.get('system') or {}
    if _CONFIG_KEY not in sys_cfg:
        return
    value = sys_cfg[_CONFIG_KEY]
    if not isinstance(value, bool):
        raise ValueError(
            f"system.wannier_r must be a boolean (true/false), got "
            f"{value!r} of type {type(value).__name__}"
        )


def resolve_wannier_r(cfg, system, engine) -> bool:
    """Read ``system.wannier_r``, validate it, announce it, return it.

    Parameters
    ----------
    cfg : full config dict (not just the ``calc`` section)
    system : the System being computed on; decides whether the flag is inert
    engine : engine name, used in the printed messages

    On a system that carries position matrices, ``False`` prints a note
    saying the finite-spread part of r is being dropped — it is a diagnostic
    setting, not a physics preference.
    """
    validate_wannier_r(cfg)

    sys_cfg = cfg.get('system') or {}
    flag = bool(sys_cfg.get(_CONFIG_KEY, WANNIER_R_DEFAULT))

    if not system_has_wannier_r(system):
        _notice_once(
            engine, 'inert',
            f"  [{engine}] wannier_r={flag} (inert: this system carries no "
            f"Wannier position matrices, so r = -i v/w is exact)"
        )
    elif flag:
        _notice_once(
            engine, 'on',
            f"  [{engine}] wannier_r=True: Wannier-gauge position correction "
            f"enabled (finite-spread part of r from the _tb.dat position "
            f"blocks, applied to r, r^{{a;b}} and the current vertex)"
        )
    else:
        _notice_once(
            engine, 'off',
            f"  [{engine}] NOTE: wannier_r=False — the finite-spread part of "
            f"the position operator is DROPPED; r = -i v/w is the point-like-"
            f"orbital approximation (atomic gauge). Diagnostic setting, not a "
            f"physics preference."
        )

    return flag


def warn_unused_wannier_r(cfg, calc_type) -> None:
    """Note that ``system.wannier_r`` is set for an engine that ignores it.

    ``bands``, ``all_ek`` and ``jdos`` build no position operator, so the flag
    has nothing to act on.  Say so rather than dropping it silently.
    """
    sys_cfg = cfg.get('system') or {}
    if _CONFIG_KEY not in sys_cfg:
        return
    _notice_once(
        calc_type, 'unused',
        f"  [{calc_type}] NOTE: system.wannier_r is set but calc.type="
        f"'{calc_type}' uses no position operator; the flag is ignored."
    )


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

def wannier_centres_from_atompos(system):
    """Per-orbital centres tau_j implied by ``system.atompos``, shape (norbs, 3).

    ``atompos.x[i, j] = tau_j - tau_i`` is exactly the intra-cell offset that
    `bloch.get_H_k` puts in its Bloch phases, so row 0 is tau_j up to the
    common constant tau_0.  A constant shift of every centre changes A^(W) by
    a multiple of the identity, which drops out of everything built from it
    (only off-diagonal parts and diagonal *differences* are ever used), so the
    constant is immaterial.  Deriving tau from atompos, rather than from a
    stored centre list, is what guarantees the connection and H(k) are in the
    same gauge no matter how the system was built.
    """
    ap = system.atompos
    return np.stack([np.real(ap.x[0]), np.real(ap.y[0]), np.real(ap.z[0])],
                    axis=1)


def compute_A_W_k(system, k, dir_chars=None, enabled=True, need_deriv=True):
    """Fourier-interpolate the Wannier-gauge Berry connection and its k-derivative.

    `bloch.get_H_k` builds H(k) with Bloch phases exp(ik·(R + tau_m - tau_n)),
    tau being the intra-cell offsets in ``system.atompos``.  In that same gauge
    the Berry connection A^(W)_a = i <u^(W)_n | d_a u^(W)_m> is

        A^(W)_{nm,a}(k) = Σ_R exp(ik·(R + tau_m - tau_n)) <0n|r_a|Rm> - tau_{n,a} δ_nm

    Eq. 20 of arXiv:1804.04030 is the tau = 0 (lattice-gauge) case; the tau
    terms are the gauge transformation of A under the diagonal unitary
    exp(ik·tau), and the subtracted centres are the SAME tau the Bloch phases
    use — taken from atompos — so H(k) and A^(W)(k) are in one gauge by
    construction.  For point-like orbitals sitting at tau the sum is exactly
    tau_n δ_nm and A^(W) vanishes: that is the atomic-gauge TBA limit in which
    r = -i v/w is complete.  The correction is therefore gauge-covariant: the
    same physical system expressed with different atompos (including zero)
    gives the same physical r once A^(W) is added — `examples/test_wannier_gauge.py`
    checks exactly that.

    Returns (A_W, dA_W) where:
      A_W[d]       = A^(W)_d(k), shape (norbs, norbs)
      dA_W[d1][d2] = ∂_{d2} A^(W)_{d1}(k), shape (norbs, norbs)
                     (the subtracted centres are k-independent)

    Returns (None, None) when `enabled` is False, or when the system has no
    ``wannier_r_matrices`` (the point-like case, where the correction is
    identically zero).  Callers therefore need only the ``if A_W is not None:``
    guard they already have — the flag needs no separate branch.

    `need_deriv=False` skips ``dA_W`` (returned as None) for callers that only
    correct ``r`` and not its generalized derivative.

    The position blocks are used as stored on the system.  `wannier.build_system_from_tb`
    Hermitianizes them (r(-R) = r(R)^dagger) and repairs the R=0 diagonal
    against the centres before storing, so A^(W)(k) here is Hermitian.
    """
    if not enabled:
        return None, None
    if not system_has_wannier_r(system):
        return None, None

    lattice_vectors = system.unitcell_vectors   # (3, 3), rows = a1, a2, a3
    r_matrices = system.wannier_r_matrices      # list of [r_x, r_y, r_z] per R
    displacements = system.wannier_r_displacements  # list of R in lattice coords
    norbs = system.norbs
    ap = system.atompos
    kx, ky, kz = k
    ap_arr = [ap.x, ap.y, ap.z]

    labels = ['x', 'y', 'z']
    if dir_chars is None:
        dir_chars = labels

    # phase_ap[i,j] = exp(ik·(τ_j - τ_i)), the same factor get_H_k applies.
    phase_ap = np.exp(1j * (kx * ap.x + ky * ap.y + kz * ap.z))
    tau = wannier_centres_from_atompos(system)          # (norbs, 3)

    # Bare Bloch sums Σ_R exp(ik·R) r(R) and Σ_R i R_b exp(ik·R) r(R)
    bare_A = {d: np.zeros((norbs, norbs), dtype=complex) for d in labels}
    bare_dA = {d1: {d2: np.zeros((norbs, norbs), dtype=complex)
                    for d2 in labels} for d1 in labels}

    for r_idx, R_latt in enumerate(displacements):
        R_cart = np.asarray(R_latt, dtype=float) @ lattice_vectors
        scalar_phase = np.exp(1j * np.dot(k, R_cart))

        for a_idx, a_label in enumerate(labels):
            weighted = r_matrices[r_idx][a_idx] * scalar_phase
            bare_A[a_label] += weighted
            if need_deriv:
                for b_idx, b_label in enumerate(labels):
                    bare_dA[a_label][b_label] += 1j * R_cart[b_idx] * weighted

    # Apply the intra-cell phase and subtract the centres:
    #   A_W[a]     = phase_ap * bare_A[a] - diag(tau_a)
    #   dA_W[a][b] = phase_ap * (bare_dA[a][b] + i * ap_b * bare_A[a])
    A_W = {}
    dA_W = {} if need_deriv else None
    for a_idx, a_label in enumerate(labels):
        A_W[a_label] = phase_ap * bare_A[a_label] - np.diag(tau[:, a_idx])
        if not need_deriv:
            continue
        dA_W[a_label] = {}
        for b_idx, b_label in enumerate(labels):
            dA_W[a_label][b_label] = phase_ap * (
                bare_dA[a_label][b_label] + 1j * ap_arr[b_idx] * bare_A[a_label]
            )

    return A_W, dA_W


def offdiag_A_H(A_W, psi, dir_chars):
    """Rotate A^(W) into the Hamiltonian gauge and strip the diagonal.

    Returns ``{d: a^(H)_d}`` with ``a^(H) = U† A^(W) U`` and the diagonal
    zeroed — the off-diagonal external Berry connection ``a_nm`` that Eq. 22
    adds to the interband position operator.  Returns None if `A_W` is None.
    """
    if A_W is None:
        return None
    a_H = {}
    for d in dir_chars:
        mat = psi.conj().T @ A_W[d] @ psi
        np.fill_diagonal(mat, 0.0)
        a_H[d] = mat
    return a_H


def apply_wannier_correction(A_W, dA_W, psi, rmtx, dir_chars,
                             vmtx=None, de_mtx=None):
    """Correct the interband position operator and its generalized derivative.

    Inputs are the bare tight-binding operators in the eigenbasis `psi`:
    ``rmtx[a] = rbar^a = -i v^a / w`` (off-diagonal), plus the Wannier
    connection ``(A_W, dA_W)`` from `compute_A_W_k`.  Returns ``(r, corr, v)``:

        r^a          = rbar^a + a^a                     (Eq. 22)
        v^a_nm       = vbar^a_nm + i w_nm a^a_nm        (n != m; diagonal unchanged)
        corr^{a;b}   = U† (∂_b A^(W)_a) U
                       - i [a^a, rbar^b]
                       - i (ξ^b_nn - ξ^b_mm) (a^a + rbar^a)_nm
                       - i (ξ^a_nn - ξ^a_mm) rbar^b_nm

    with ``a = offdiag(U† A^(W) U)`` and ``ξ = diag(U† A^(W) U)``.  ``corr``
    is the piece that must be ADDED to the TB Sipe sum-rule r^{a;b}
    (`nonlinear_optical._compute_dk_rmtx`) so that the sum is the generalized
    derivative r^a_{nm;b} = ∂_b r^a_nm - i (ξ^b_nn - ξ^b_mm) r^a_nm of the
    corrected r, where ξ is the full diagonal Berry connection.

    Derivation.  In the Hamiltonian gauge r = Ā + iD with Ā = U† A^(W) U and
    D = U† ∂U.  For any M = U† X U the covariant derivative is
        M_{;b} = U† (∂_b X) U + [M, D^off_b] - i (Ā^b_nn - Ā^b_mm) M_nm ,
    the diagonal D_nn cancelling identically between ∂_b M and the ξ term.
    Applied to Ā_a and to iD_a with D^off = -i rbar, and with [Ā, ·] split
    into its off-diagonal (a) and diagonal (ξ) parts, everything carrying an Ā
    is collected above; the D-only remainder is the TB Sipe formula.  Each
    piece satisfies corr_nm^* = corr_mn, as the derivative of a Hermitian
    operator must.  The previous implementation lacked the factor i on the
    commutator (making it anti-Hermitian), had the ξ·a term with the opposite
    sign, and omitted both ξ·rbar terms — see BUG_wannier_r_correction.md.

    The third output is the *physical* velocity, v = i[H, r] with the
    corrected r, whose interband elements are i w_nm r_nm = vbar_nm + i w_nm
    a_nm; it is what an engine must use as its current vertex (and anywhere
    else an interband v_nm stands for i w_nm r_nm), while the bare vbar stays
    in the Sipe sum rule, the band-energy identities (Delta, inverse mass)
    and everything else that is a derivative of H(k).  It is returned only
    when `vmtx` and `de_mtx` (= E_n - E_m) are given, else None.  Every
    engine output must be independent of the gauge the system was built in
    (its atompos); `examples/test_wannier_gauge.py` checks that, and it is
    what fixes which operator goes where.

    Returns ``(rmtx, None, vmtx)`` unchanged when `A_W` is None, so call
    sites keep a single ``if corr is not None:`` guard.
    """
    if A_W is None:
        return rmtx, None, vmtx
    if dA_W is None:
        raise ValueError("apply_wannier_correction needs dA_W; call "
                         "compute_A_W_k with need_deriv=True")

    A_H = {d: psi.conj().T @ A_W[d] @ psi for d in dir_chars}
    a_H = {}
    xid = {}
    for d in dir_chars:
        mat = A_H[d].copy()
        np.fill_diagonal(mat, 0.0)
        a_H[d] = mat
        xi = np.diag(A_H[d]).real
        xid[d] = xi[:, None] - xi[None, :]

    r_new = {d: rmtx[d] + a_H[d] for d in dir_chars}

    corr = {}
    for a in dir_chars:
        corr[a] = {}
        for b in dir_chars:
            c = psi.conj().T @ dA_W[a][b] @ psi
            c = c + 1j * (rmtx[b] @ a_H[a] - a_H[a] @ rmtx[b])   # -i [a^a, rbar^b]
            c = c - 1j * xid[b] * (a_H[a] + rmtx[a])
            c = c - 1j * xid[a] * rmtx[b]
            np.fill_diagonal(c, 0.0)
            corr[a][b] = c

    v_new = None
    if vmtx is not None:
        if de_mtx is None:
            raise ValueError("apply_wannier_correction: de_mtx is required "
                             "together with vmtx")
        v_new = {d: vmtx[d] + 1j * de_mtx * a_H[d] for d in dir_chars}
    return r_new, corr, v_new
