"""Tests that `wannier_r` is one switch, reaching every engine the same way.

The Wannier-gauge position correction (calc/wannier_gauge.py) is a no-op for
systems built from YAML, so a test on a plain tight-binding model would pass
no matter how the gating were wired.  These tests therefore take the gapped
honeycomb and *decorate* it with synthetic ``wannier_r_matrices`` — built with
r(-R) = r(R)^dagger so A^(W)(k) is Hermitian, as a real ``_tb.dat`` gives —
which makes the correction bite and the flag observable.

Checks, for chi^(2) (fast and reference paths), delta_Q and quantum_metric:

1.  wannier_r=False on a decorated system is bit-identical to the same run on
    the undecorated system.  Off really means off.
2.  wannier_r=True differs measurably.  The gate is live, not stuck.
3.  chi^(2) slow and fast paths agree at *both* flag values (the failure mode
    of gating only one of them).
4.  Omitting the keyword gives WANNIER_R_DEFAULT.
5.  The config key reaches each engine from cfg['system']['wannier_r'], and
    the old cfg['calc']['wannier_r'] location raises.
6.  On an undecorated system True == False bitwise (the flag is inert).
7.  quantum_metric with the correction off reproduces the pre-change
    v*conj(v)/de^2 expression it was refactored out of.

Run:  python3 examples/test_wannier_r_flag.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore', category=RuntimeWarning)

from tightbinding.calc import wannier_gauge as wg           # noqa: E402
from tightbinding.calc.wannier_gauge import WANNIER_R_DEFAULT   # noqa: E402

FAILURES = []


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {label:<56s} {detail}")
    if not ok:
        FAILURES.append(label)
    return ok


def check_close(label, err, tol):
    return check(label, np.isfinite(err) and err <= tol,
                 f"{err:.3e}  (tol {tol:.0e})")


def maxdiff(a, b):
    """Max |a-b| over two nested dict/array structures with the same shape."""
    if isinstance(a, dict):
        return max((maxdiff(a[k], b[k]) for k in a), default=0.0)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def relnorm(a, b):
    """Max |a-b| relative to the scale of `a` (for 'these must differ')."""
    scale = _absmax(a)
    return maxdiff(a, b) / scale if scale > 0 else maxdiff(a, b)


def _absmax(a):
    if isinstance(a, dict):
        return max((_absmax(v) for v in a.values()), default=0.0)
    return float(np.max(np.abs(np.asarray(a))))


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def honeycomb_system(m=0.1, T=1.0):
    """Gapped honeycomb, s_u orbitals — the model used elsewhere in examples/."""
    from tightbinding.lattice import build_system
    from tightbinding.hamiltonian import fill_hamiltonian
    cfg = {
        'system': {
            'basis': 's_u',
            'lattice_vectors': [[0.8660254037844386, -1.5, 0.0],
                                [0.8660254037844386, 1.5, 0.0],
                                [0.0, 0.0, 20.0]],
            'positions': [
                {'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                {'species': 'B', 'coord': [0.0, 1.0, 0.0]},
            ],
        },
        'hopping': {'range': 1.1, 'tss_sigma': T},
        'onsite': {'A': {'u_s': m}, 'B': {'u_s': -m}},
    }
    system = build_system(cfg)
    fill_hamiltonian(system)
    return system


def decorate_wannier_r(system, seed=7):
    """Attach synthetic Wannier position matrices, as build_system_from_tb does.

    R-vectors come in +/- pairs with r(-R) = r(R)^dagger, and the R=0 block is
    Hermitian, so the Bloch sum A^(W)(k) = sum_R e^{ikR} r(R) is Hermitian at
    every k — the property a real _tb.dat position matrix has and the one
    `_compute_A_W_k` is supposed to preserve.
    """
    rng = np.random.default_rng(seed)
    n = system.norbs

    def rand_mat():
        return (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * 0.3

    pos = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
    displacements = [np.array((0, 0, 0))]
    mats = [[(lambda M: 0.5 * (M + M.conj().T))(rand_mat()) for _ in range(3)]]
    for R in pos:
        blocks = [rand_mat() for _ in range(3)]
        displacements.append(np.array(R))
        mats.append(blocks)
        displacements.append(-np.array(R))
        mats.append([b.conj().T for b in blocks])

    system.wannier_r_matrices = mats
    system.wannier_r_displacements = displacements
    return system


# ---------------------------------------------------------------------------
# Engine drivers — same k-point / grid, flag as the only variable
# ---------------------------------------------------------------------------

K_TEST = np.array([0.37, 0.11, 0.0])
DIRS = ['x', 'y']
TRIPLETS = ['yyy', 'yxx']


def run_chi2_fast(system, **kw):
    from tightbinding.calc.nonlinear_optical_fast import _process_kpoint_fast
    dim = len(system.matrices[0].H)
    return _process_kpoint_fast(
        system, K_TEST, dim, DIRS, TRIPLETS,
        np.array([0.3, 0.9]), 0.0, None, 1e-3, None,
        np.array([0.0]), 1e-3, 1, 2, eta_sos=1e-8, **kw)


def run_chi2_slow(system, **kw):
    from tightbinding.calc.nonlinear_optical import _process_kpoint
    dim = len(system.matrices[0].H)
    eta = 1e-3
    return _process_kpoint(
        system, K_TEST, dim, DIRS, TRIPLETS,
        np.array([0.3, 0.9]), 0.0, np.zeros((dim, dim)), eta,
        eta * np.ones((dim, dim)),
        np.array([0.0]), 1e-3, 1, 2, eta_sos=1e-8, **kw)


def run_dq(system, **kw):
    from tightbinding.calc.delta_Q import _process_kpoint
    res, _terms, _tau = _process_kpoint(
        system, K_TEST, DIRS, ['yy'], ['y'],
        np.array([0.0]), 1e-3, 1, 0.0, eta_sos=1e-8,
        dQ_occupied_subspace=True, **kw)
    return res


def run_qm(system, **kw):
    from tightbinding.calc.quantum_metric import _process_kpoint
    params = {'dim': len(system.matrices[0].H), 'dir_chars': DIRS,
              'eflist': np.array([0.0]), 'kT': 1e-3, 'eta': 1e-3,
              'delta': 1e-3, 'nef': 1}
    params.update(kw)
    return _process_kpoint(system, K_TEST, params)


ENGINES = [
    ('chi2_fast', run_chi2_fast),
    ('chi2_slow', run_chi2_slow),
    ('delta_Q  ', run_dq),
    ('quantum_metric', run_qm),
]


# ---------------------------------------------------------------------------
# 1 / 2 / 4 / 6: the gate itself
# ---------------------------------------------------------------------------

def test_gate():
    print("\n[1] wannier_r=False == undecorated system (off means off)")
    plain = honeycomb_system()
    wann = decorate_wannier_r(honeycomb_system())
    for name, run in ENGINES:
        err = maxdiff(run(plain, wannier_r=False), run(wann, wannier_r=False))
        check(f"{name}: off == no position matrices", err == 0.0,
              f"exact diff {err:.1e}")

    print("\n[2] wannier_r=True changes the answer (the gate is live)")
    for name, run in ENGINES:
        rel = relnorm(run(wann, wannier_r=False), run(wann, wannier_r=True))
        check(f"{name}: on != off", rel > 1e-6, f"rel {rel:.3e}")

    print(f"\n[4] omitting the keyword gives WANNIER_R_DEFAULT"
          f" = {WANNIER_R_DEFAULT}")
    for name, run in ENGINES:
        err = maxdiff(run(wann), run(wann, wannier_r=WANNIER_R_DEFAULT))
        other = maxdiff(run(wann), run(wann, wannier_r=not WANNIER_R_DEFAULT))
        check(f"{name}: default == {WANNIER_R_DEFAULT}",
              err == 0.0 and other > 0.0, f"diff {err:.1e} / {other:.1e}")

    print("\n[6] flag is inert on a system with no position matrices")
    for name, run in ENGINES:
        err = maxdiff(run(plain, wannier_r=False), run(plain, wannier_r=True))
        check(f"{name}: True == False when inert", err == 0.0,
              f"exact diff {err:.1e}")


# ---------------------------------------------------------------------------
# 3: the two chi^(2) paths must be gated identically
# ---------------------------------------------------------------------------

def test_slow_fast_agree():
    print("\n[3] chi^(2) reference path tracks the fast path at both settings")
    from tightbinding.calc.nonlinear_optical import CHI_ALL_NAMES
    wann = decorate_wannier_r(honeycomb_system())
    for flag in (False, True):
        fast = run_chi2_fast(wann, wannier_r=flag)
        slow = run_chi2_slow(wann, wannier_r=flag)
        worst = 0.0
        for nm in CHI_ALL_NAMES:
            for abc in TRIPLETS:
                a, b = np.asarray(fast[nm][abc]), np.asarray(slow[nm][abc])
                scale = max(np.max(np.abs(a)), 1e-30)
                worst = max(worst, np.max(np.abs(a - b)) / scale)
        check_close(f"slow vs fast, wannier_r={flag}", worst, 1e-10)


# ---------------------------------------------------------------------------
# 5: config plumbing
# ---------------------------------------------------------------------------

def _cfg(calc, wannier_r=None):
    cfg = {'system': {}, 'calc': calc}
    if wannier_r is not None:
        cfg['system']['wannier_r'] = wannier_r
    return cfg


def test_config_plumbing():
    print("\n[5] system.wannier_r reaches every engine; calc.wannier_r raises")
    from tightbinding.calc.nonlinear_optical import compute_nonlinear_optical
    from tightbinding.calc.delta_Q import compute_delta_Q
    from tightbinding.calc.quantum_metric import compute_quantum_metric

    wann = decorate_wannier_r(honeycomb_system())

    calcs = {
        'nonlinear_optical': (compute_nonlinear_optical, {
            'type': 'nonlinear_optical', 'nk': [4, 4],
            'omega1list': [0.3], 'eflist': np.array([0.0]), 'kT': 1e-3,
            'eta': 1e-3, 'eta_sos': 1e-8, 'directions': ['yyy']}),
        'delta_Q': (compute_delta_Q, {
            'type': 'delta_Q', 'nk': [4, 4], 'eflist': np.array([0.0]),
            'kT': 1e-3, 'eta': 0.0, 'eta_sos': 1e-8,
            'components': ['yy'], 'field_direction': 'y'}),
        'quantum_metric': (compute_quantum_metric, {
            'type': 'quantum_metric', 'nk': [4, 4],
            'eflist': np.array([0.0]), 'kT': 1e-3, 'eta': 1e-3,
            'metric_directions': ['x', 'y']}),
    }

    for name, (fn, calc) in calcs.items():
        wg.reset_notices()
        off = fn(wann, _cfg(dict(calc), wannier_r=False))
        wg.reset_notices()
        on = fn(wann, _cfg(dict(calc), wannier_r=True))
        wg.reset_notices()
        dflt = fn(wann, _cfg(dict(calc)))
        rel = relnorm(off, on)
        check(f"{name}: system.wannier_r switches the engine", rel > 1e-6,
              f"rel {rel:.3e}")
        ref = off if WANNIER_R_DEFAULT is False else on
        check(f"{name}: config default == {WANNIER_R_DEFAULT}",
              maxdiff(dflt, ref) == 0.0)

        try:
            bad = _cfg(dict(calc))
            bad['calc']['wannier_r'] = True
            fn(wann, bad)
            check(f"{name}: calc.wannier_r rejected", False, "no raise")
        except ValueError as exc:
            check(f"{name}: calc.wannier_r rejected", 'moved' in str(exc))
    wg.reset_notices()


# ---------------------------------------------------------------------------
# 7: the quantum_metric refactor preserves the old algebra
# ---------------------------------------------------------------------------

def test_quantum_metric_refactor():
    """Q, dQ, dQf with the correction off must equal the pre-refactor algebra.

    Reproduced inline rather than taken from a stored benchmark, because the
    example config `input_qm_test.yaml` happens to give machine-zero dQ/dQf
    and so would not exercise the finite-difference path at all.
    """
    print("\n[7] quantum_metric, correction off, vs the original v*conj(v)/de^2")
    from tightbinding.bloch import get_H_v, diagonalize_hk
    from tightbinding.calc.quantum_metric import DEG_THR

    system = honeycomb_system()
    kT, eta, delta = 1e-3, 1e-3, 1e-3
    got = run_qm(system, wannier_r=False)

    # --- the pre-refactor expressions, inline ---
    H, S, vtb = get_H_v(system, K_TEST)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    de = ek[:, None] - ek[None, :]
    nondeg = np.abs(de) >= DEG_THR
    inv_de2 = np.where(nondeg, 1.0 / de ** 2, 0.0)
    vmtx = {d: psi.conj().T @ vtb[d] @ psi for d in DIRS}

    psip, psim = {}, {}
    for d in DIRS:
        pert = np.where(nondeg, 1j * delta * vmtx[d] / (de * (de + 1j * eta)), 0.0)
        psip[d] = psi + psi @ pert
        psim[d] = psi - psi @ pert
    vp = {d1: {d3: psip[d3].conj().T @ vtb[d1] @ psip[d3] for d3 in DIRS}
          for d1 in DIRS}
    vm = {d1: {d3: psim[d3].conj().T @ vtb[d1] @ psim[d3] for d3 in DIRS}
          for d1 in DIRS}

    x = np.clip(ek / kT, -500, 500)
    f = 1.0 / (1.0 + np.exp(x))
    fnm = f[:, None] * (1.0 - f[None, :])
    de_f = -1.0 / kT * np.exp(x) / (1.0 + np.exp(x)) ** 2
    de_f = np.where(np.isfinite(de_f), de_f, 0.0)

    def relerr(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    worstQ = worstdQ = worstdQf = 0.0
    scale_dQ = 0.0
    for d1 in DIRS:
        for d2 in DIRS:
            ref = np.sum(vmtx[d1] * np.conj(vmtx[d2]) * fnm * inv_de2)
            worstQ = max(worstQ, relerr(got['Q'][d1][d2][0], ref))
            for d3 in DIRS:
                cp = np.sum(vp[d1][d3] * np.conj(vp[d2][d3]) * fnm * inv_de2)
                cm = np.sum(vm[d1][d3] * np.conj(vm[d2][d3]) * fnm * inv_de2)
                ref_dQ = (cp - cm) / (2 * delta)
                scale_dQ = max(scale_dQ, abs(ref_dQ))
                # dQ is (cp - cm) / 2delta with cp ~ cm ~ Q, so a 1-ulp
                # difference in the sums is amplified by |Q| / (delta |dQ|)
                # ~ 1e8 here.  Judge the agreement on the sums that were
                # actually differenced, not on their cancelled remainder.
                err = abs(got['dQ'][d1][d2][d3][0] - ref_dQ) * 2 * delta
                worstdQ = max(worstdQ, err / max(abs(cp), 1e-30))

                vdf = de_f * np.diag(vmtx[d3])
                dfnm = (vdf[:, None] * (1.0 - f[None, :])
                        + f[:, None] * (-vdf[None, :]))
                ref_f = np.sum(vmtx[d1] * np.conj(vmtx[d2]) * dfnm * inv_de2)
                worstdQf = max(worstdQf,
                               relerr(got['dQf'][d1][d2][d3][0], ref_f))

    check("dQ reference is not trivially zero", scale_dQ > 1e-6,
          f"max |dQ| {scale_dQ:.3e}")
    check_close("Q   matches the pre-refactor form", worstQ, 1e-13)
    check_close("dQ  matches (pre-cancellation scale)", worstdQ, 1e-13)
    check_close("dQf matches the pre-refactor form", worstdQf, 1e-13)


if __name__ == '__main__':
    test_gate()
    test_slow_fast_agree()
    test_config_plumbing()
    test_quantum_metric_refactor()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All wannier_r flag tests passed.")
