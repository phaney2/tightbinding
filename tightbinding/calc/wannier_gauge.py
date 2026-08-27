"""Wannier-gauge position-operator correction: one switch, one kernel.

A Hamiltonian read from a Wannier90 ``_tb.dat`` file is **not** in the atomic
gauge, so the tight-binding form ``r = -i v / w`` is incomplete: the intra-cell
part of the position operator has to be added back from the position matrices
that ``_tb.dat`` carries alongside the Hamiltonian.  That correction is
``A^(W)(k)`` (Eq. 20 of arXiv:1804.04030), computed here by `compute_A_W_k`,
and applied by the engines as Eq. 22 (to ``r``) and Eq. 36 (to ``r^{a;b}``).

Every engine that builds a position operator reads the same switch from the
same place through `resolve_wannier_r`, and gates the same kernel through the
``enabled`` argument of `compute_A_W_k`.  That is deliberate: before this
module existed the three engines each did their own thing (χ^(2) hard-wired
ON, ``delta_Q`` switchable, ``quantum_metric`` hard-wired OFF) and drifted
apart silently.  Add a new engine with a position operator, and it must go
through these two functions.

Config::

    system:
      wannier_tb: mos2_tb.dat
      wannier_r: false        # <- here, NOT under calc:

The correction is a **no-op** for systems built with ``build_system`` +
``fill_hamiltonian``: `bloch.get_H_k` builds those in the atomic gauge, where
``r = -i v / w`` holds exactly with zero intra-cell term.  Such systems carry
no ``wannier_r_matrices`` and the flag is inert either way.
"""

import numpy as np

from .. import parallel


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

# TODO: restore to True when BUG_wannier_r_correction.md is closed.
#
# The correction is *physically required* for Wannier input, so True is the
# right long-run default.  It is False today only because `compute_A_W_k` is
# known-defective (it makes delta_Q complex when it must be real, and gives
# chi^(2) a sub-gap dissipative part that survives eta -> 0).  Neither setting
# is correct on a _tb.dat system right now, which is why both of them warn.
WANNIER_R_DEFAULT = False

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

    Both settings are noisy on a system that actually carries position
    matrices, because `BUG_wannier_r_correction.md` is open and neither
    setting gives a trustworthy answer there.  ``False`` is not a fix for that
    bug — it is a diagnostic that drops a physically required term.
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
            f"  [{engine}] WARNING: wannier_r=True — the Wannier-gauge "
            f"position correction is ENABLED and is known to be defective. "
            f"See BUG_wannier_r_correction.md; results on _tb.dat input are "
            f"contaminated."
        )
    else:
        _notice_once(
            engine, 'off',
            f"  [{engine}] WARNING: wannier_r=False — the Wannier-gauge "
            f"position correction is DISABLED. r = -i v/w is NOT the correct "
            f"position operator for _tb.dat input; this is a diagnostic "
            f"setting, not a physics preference. It is the default only while "
            f"BUG_wannier_r_correction.md is open."
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

def compute_A_W_k(system, k, dir_chars=None, enabled=True, need_deriv=True):
    """Fourier-interpolate the Wannier position operator and its k-derivative.

    Implements Eq. 20 of Ibañez-Azpiroz et al. (arXiv:1804.04030):
        A^(W)_{k,nm,a} = Σ_R exp(ik·(R+τ_m-τ_n)) <0n|r̂_a - τ_{m,a}|Rm>

    Returns (A_W, dA_W) where:
      A_W[d]       = A^(W)_d(k), shape (norbs, norbs)
      dA_W[d1][d2] = ∂_{d2} A^(W)_{d1}(k), shape (norbs, norbs)

    Returns (None, None) when `enabled` is False, or when the system has no
    ``wannier_r_matrices`` (the atomic-gauge case, where the correction is
    identically zero).  Callers therefore need only the ``if A_W is not None:``
    guard they already have — the flag needs no separate branch.

    `need_deriv=False` skips ``dA_W`` (returned as None) for callers that only
    correct ``r`` and not its generalized derivative.
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

    # phase_ap[i,j] = exp(ik·(τ_j - τ_i))
    phase_ap = np.exp(1j * (kx * ap.x + ky * ap.y + kz * ap.z))

    # Find R=0 index
    r0_idx = None
    for idx, R_latt in enumerate(displacements):
        if np.allclose(R_latt, 0):
            r0_idx = idx
            break

    # Accumulate bare Bloch sums (without phase_ap)
    bare_A = {d: np.zeros((norbs, norbs), dtype=complex) for d in labels}
    bare_dA = {d1: {d2: np.zeros((norbs, norbs), dtype=complex)
                    for d2 in labels} for d1 in labels}

    for r_idx, R_latt in enumerate(displacements):
        R_cart = R_latt @ lattice_vectors
        scalar_phase = np.exp(1j * np.dot(k, R_cart))

        for a_idx, a_label in enumerate(labels):
            r_a = r_matrices[r_idx][a_idx]
            # For R=0 diagonal: <0n|r̂_a - τ_{n,a}|0n> = 0 exactly.
            # Zero the diagonal rather than subtracting centres, to avoid
            # Berry-phase wrapping artifacts in Wannier90's position matrix.
            if r_idx == r0_idx:
                r_a = r_a.copy()
                np.fill_diagonal(r_a, 0.0)

            weighted = r_a * scalar_phase
            bare_A[a_label] += weighted

            if need_deriv:
                for b_idx, b_label in enumerate(labels):
                    bare_dA[a_label][b_label] += 1j * R_cart[b_idx] * weighted

    # Apply atompos phase:
    #   A_W[a] = phase_ap * bare_A[a]
    #   dA_W[a][b] = phase_ap * (bare_dA[a][b] + i * ap_b * bare_A[a])
    A_W = {}
    dA_W = {} if need_deriv else None
    for a_label in labels:
        A_W[a_label] = phase_ap * bare_A[a_label]
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
