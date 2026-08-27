"""Tests for the analytic frequency-integrated chi^(2) path.

Four independent checks:

1.  `rational_integral` reproduces the reference values recorded in
    NOTE_analytic_frequency_integral.md section 4 (p = 1, s = 1..3, finite b).
2.  `rational_integral` matches a dense log-mesh trapezoid, for p = 0..4 and
    s = 1..3, on a range chosen so the mesh is trustworthy.  Adaptive
    quadrature is deliberately NOT used -- see note section 5.4.
3.  The full engine, run in integrated mode, matches a dense-mesh trapezoid
    over the sampled engine, at a single k-point of the gapped honeycomb.
4.  Sampled mode is unchanged by the kernel refactor: the phase-3 einsums are
    compared against the pre-refactor implementation checked out at HEAD.
    (Check 4 only runs when a baseline worktree path is supplied.)

Run:  python3 examples/test_freq_integral.py
"""

import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The engine suppresses these itself; the tests call the kernel directly.
warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_BASELINE = os.path.join(tempfile.gettempdir(), 'tb_head')

from tightbinding.calc.freq_integral import (          # noqa: E402
    FreqIntegralSpec, rational_integral,
)

FAILURES = []


def check(label, err, tol):
    ok = np.isfinite(err) and err <= tol
    print(f"  {'PASS' if ok else 'FAIL'} {label:<52s} {err:.3e}  (tol {tol:.0e})")
    if not ok:
        FAILURES.append(label)
    return ok


# ---------------------------------------------------------------------------
# 1. Reference values from the note
# ---------------------------------------------------------------------------

def test_note_reference():
    """NOTE section 4: w_nm = 0.25, eta = 0.025, [a,b] = [1e-4, 1e3], p = 1."""
    print("\n[1] Reference values from NOTE_analytic_frequency_integral.md s4")
    z = 0.25 - 0.025j
    ref = {
        1: -1.5147512413e+01,
        2: +6.8653223354e+01,
        3: -3.0080982623e+02,
    }
    for s, expected in ref.items():
        J, _ = rational_integral(1, [np.array(z)], [s], 1e-4, 1e3)
        got = float(np.imag(J))
        check(f"Im G(p=1, s={s}) = {got:+.10e}",
              abs(got - expected) / abs(expected), 1e-9)


# ---------------------------------------------------------------------------
# 2. Formula vs dense log-mesh trapezoid
# ---------------------------------------------------------------------------

def _trapz_log(p, poles, mults, a, b, n):
    """int_a^b dw w^-p f(w) on a uniform mesh in ln w.

    dw = w d(ln w), so the w^-1 of the weight is absorbed exactly and only the
    smooth part is discretized.
    """
    lw = np.linspace(np.log(a), np.log(b), n)
    w = np.exp(lw)
    g = w ** (1 - p)
    for z, s in zip(poles, mults):
        g = g / (w - z) ** s
    return np.trapezoid(g, lw)


def test_against_mesh():
    """p = 0..4, s = 1..3, three poles.  Benign range so the mesh is reliable.

    The mesh is the approximation here, not the formula, so each case is
    scored against the mesh's own convergence: |J - I_2n| must not exceed the
    mesh's self-difference |I_2n - I_n| by more than a small factor.  Judging
    against a fixed tolerance instead would just be measuring the trapezoid.
    """
    print("\n[2] rational_integral vs dense log-mesh trapezoid, [0.5, 20]")
    worst_ratio, worst_case, worst_rel = 0.0, '', 0.0
    for z in (0.25 - 0.025j, 2.363 - 0.01j, 6.005 - 0.001j):
        for p in range(0, 5):
            for s in (1, 2, 3):
                if p + s < 2:
                    continue
                J, _ = rational_integral(p, [np.array(z)], [s], 0.5, 20.0)
                i_n = _trapz_log(p, [z], [s], 0.5, 20.0, 1_000_001)
                i_2n = _trapz_log(p, [z], [s], 0.5, 20.0, 2_000_001)
                scale = max(abs(i_2n), 1e-300)
                rel = abs(J - i_2n) / scale
                mesh_err = abs(i_2n - i_n) / scale
                # Allowance: 10x the mesh's own convergence, floored at 1e-10
                # so that cases where the mesh is already near round-off are
                # not judged against its noise.
                ratio = rel / max(1e-10, 10 * mesh_err)
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_case = f"z={z}, p={p}, s={s}"
                    worst_rel = rel
    check(f"worst (formula-mesh)/allowance [{worst_case}, "
          f"rel={worst_rel:.1e}]", worst_ratio, 1.0)


def test_two_distinct_poles():
    """The z12/z1 pair: same real part, separated by exactly i*eta."""
    print("\n[3] Two distinct poles separated by i*eta (the generic case)")
    for eta in (0.025, 0.01, 1e-3):
        de = 2.363
        z12 = de - 2j * eta
        z1 = de - 1j * eta
        for mult in ([1, 1], [1, 2]):
            J, _, cond = rational_integral(
                1, [np.array(z12), np.array(z1)], mult, 0.5, 20.0,
                want_cond=True)
            ref = _trapz_log(1, [z12, z1], mult, 0.5, 20.0, 4_000_001)
            rel = abs(J - ref) / abs(ref)
            check(f"eta={eta:<7g} s={mult}  cond={float(cond):.1e}", rel, 1e-6)


def test_infinite_upper_limit():
    """b = inf must agree with a large finite b, and NaN when it must."""
    print("\n[4] b = inf")
    z = np.array(2.363 - 0.01j)
    for p, s in ((1, 1), (1, 2), (2, 1), (3, 3), (0, 2)):
        J_inf, _ = rational_integral(p, [z], [s], 0.5, np.inf)
        J_big, _ = rational_integral(p, [z], [s], 0.5, 1e12)
        check(f"p={p}, s={s}: inf vs b=1e12",
              abs(J_inf - J_big) / abs(J_inf), 1e-9)
    # Divergent cases: integrand decays no faster than 1/w
    for p, poles, mults in ((0, [z], [1]), (1, [], []), (0, [], [])):
        J, _ = rational_integral(p, poles, mults, 0.5, np.inf)
        ok = bool(np.all(np.isnan(J)))
        print(f"  {'PASS' if ok else 'FAIL'} "
              f"p={p}, {len(poles)} pole(s): NaN as expected")
        if not ok:
            FAILURES.append(f"divergent p={p} not NaN")


def test_endpoint_coefficients():
    """A_j must reproduce the measured d/d(ln a) and a-scaling of J."""
    print("\n[5] Lower-endpoint divergence coefficients")
    z = np.array(0.9 - 0.02j)
    for p in (1, 2, 3):
        a0 = 0.3
        J0, A = rational_integral(p, [z], [2], a0, np.inf)
        # J(a) = -A_1 ln a + sum_{j>=2} A_j a^(1-j)/(j-1) + regular(a)
        # Differentiate: dJ/da = -A_1/a - sum_{j>=2} A_j a^-j + d(regular)/da,
        # and d(regular)/da is just the integrand's non-singular part, so the
        # cleanest check is that J reconstructs from its own decomposition:
        h = 1e-6
        Jp, _ = rational_integral(p, [z], [2], a0 + h, np.inf)
        Jm, _ = rational_integral(p, [z], [2], a0 - h, np.inf)
        dJ = (Jp - Jm) / (2 * h)
        # dJ/da = -integrand(a)
        integrand = -1.0 / (a0 ** p * (a0 - z) ** 2)
        check(f"p={p}: dJ/da = -w^-p/(w-z)^2 at a",
              abs(dJ - integrand) / abs(integrand), 1e-7)
        # ln a coefficient: J(a) - J(a') should contain -A_1 ln(a/a')
        a1 = 0.3001
        J1, A1c = rational_integral(p, [z], [2], a1, np.inf)
        # Reconstruct the difference from the analytic pieces
        recon = (-A[0] * np.log(a1) + A[0] * np.log(a0))
        for j in range(2, p + 1):
            recon += A[j - 1] / (j - 1) * (a1 ** (1 - j) - a0 ** (1 - j))
        recon += -(np.log(a1 - z) - np.log(a0 - z)) * _b1(p, z, 2)
        recon += -_b2(p, z, 2) * ((a1 - z) ** -1 - (a0 - z) ** -1) / (-1)
        check(f"p={p}: J(a1)-J(a0) from coefficients",
              abs((J1 - J0) - recon) / abs(J1 - J0), 1e-9)


def _b1(p, z, s):
    from tightbinding.calc.freq_integral import partial_fractions
    _, B = partial_fractions(p, [np.array(z)], [s])
    return B[0][0]


def _b2(p, z, s):
    from tightbinding.calc.freq_integral import partial_fractions
    _, B = partial_fractions(p, [np.array(z)], [s])
    return B[0][1]


# ---------------------------------------------------------------------------
# 6. End-to-end: integrated engine vs mesh over the sampled engine
# ---------------------------------------------------------------------------

def _honeycomb_system(m=0.1, T=1.0):
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


def test_end_to_end():
    """Single k-point: integrated kernel vs log-mesh over the sampled kernel."""
    print("\n[6] Engine end-to-end at one k-point (gapped honeycomb)")
    from tightbinding.calc.nonlinear_optical_fast import _process_kpoint_fast
    from tightbinding.calc.nonlinear_optical import CHI_PHYSICAL

    system = _honeycomb_system(m=0.1)
    dim = len(system.matrices[0].H)
    directions = ['yyy', 'yxx']
    dir_chars = ['x', 'y']
    eflist = np.array([0.0])
    kT = 1e-4
    eta = 1e-3
    p = 1
    a, b = 0.05, 400.0          # finite b so the mesh can reach it
    k = np.array([0.37, 0.11, 0.0])

    spec = FreqIntegralSpec(p_list=[p], omega_min=a, omega_max=b,
                            diagnostics=False)
    got = _process_kpoint_fast(
        system, k, dim, dir_chars, directions,
        None, 0.0, None, eta, None, eflist, kT, 1, spec.nchan,
        eta_sos=1e-8, freq_integral=spec,
    )

    # Reference: sample chi on a log mesh and integrate d(ln w)
    nmesh = 400_001
    lw = np.linspace(np.log(a), np.log(b), nmesh)
    w = np.exp(lw)
    sampled = _process_kpoint_fast(
        system, k, dim, dir_chars, directions,
        w, 0.0, None, eta, None, eflist, kT, 1, nmesh,
        eta_sos=1e-8,
    )

    worst = 0.0
    for name in CHI_PHYSICAL:
        for abc in directions:
            ref = np.trapezoid(sampled[name][abc][0] * w ** (1 - p), lw)
            val = got[name][abc][0, 0]
            denom = max(abs(ref), 1e-30)
            worst = max(worst, abs(val - ref) / denom)
    check(f"worst rel. err over {len(CHI_PHYSICAL)} terms x 2 directions",
          worst, 2e-6)


def test_endpoint_diagnostic_identity():
    """The accumulated endpoint coefficients must equal chi at omega = 0.

    Two exact identities, both checked against the *sampled* engine:

      p = 1:  coefficient of ln(omega_min) in J  =  -chi(omega = 0)
      p = 2:  coefficient of omega_min^(-1) in J =  +chi(omega = 0)
              coefficient of ln(omega_min) in J  =  -chi'(omega = 0)

    because J = int_a domega w^-p chi(w) and chi is analytic at w = 0 for
    eta > 0.  "chi(0)" here is the rational function continued to w = 0 with
    the broadenings in place, which is exactly what the sampled engine returns
    for omega1list = [0.0], so the comparison is machine-precision.
    """
    print("\n[7] Endpoint diagnostics vs chi(omega=0) from the sampled engine")
    from tightbinding.calc.nonlinear_optical_fast import _process_kpoint_fast
    from tightbinding.calc.nonlinear_optical import CHI_PHYSICAL

    system = _honeycomb_system(m=0.1)
    dim = len(system.matrices[0].H)
    directions = ['yyy', 'yxx']
    dir_chars = ['x', 'y']
    eflist = np.array([0.0])
    eta = 1e-3
    k = np.array([0.37, 0.11, 0.0])
    base = (system, k, dim, dir_chars, directions)
    tail = dict(eta_sos=1e-8)

    spec = FreqIntegralSpec(p_list=[1, 2], omega_min=0.01, diagnostics=True)
    got = _process_kpoint_fast(*base, None, 0.0, None, eta, None,
                               eflist, 1e-5, 1, spec.nchan,
                               freq_integral=spec, **tail)

    # chi(0) and chi'(0) from the sampled engine.  chi varies on the scale of
    # eta, so the central difference needs h << eta: its truncation error is
    # (h/eta)^2/6, already 2e-5 at h = eta/100.
    h = 1e-7
    w = np.array([-h, 0.0, h])
    smp = _process_kpoint_fast(*base, w, 0.0, None, eta, None,
                               eflist, 1e-5, 1, 3, **tail)

    ilog = spec.channel_indices('log')      # one column per p
    ipow2 = spec.channel_indices('pow', 2)

    worst_v, worst_d = 0.0, 0.0
    for name in CHI_PHYSICAL:
        for abc in directions:
            row = smp[name][abc][0]
            chi0 = row[1]
            dchi0 = (row[2] - row[0]) / (2 * h)
            scale = max(abs(chi0), 1e-25)
            g = got[name][abc][0]
            worst_v = max(worst_v, abs(g[ilog[0]] - (-chi0)) / scale)
            worst_v = max(worst_v, abs(g[ipow2[1]] - chi0) / scale)
            # Normalize the derivative by its natural scale |chi(0)|/eta, not
            # by |chi'(0)| -- one term has an accidentally small chi'(0) and
            # dividing by it just measures the finite difference's round-off.
            worst_d = max(worst_d, abs(g[ilog[1]] - (-dchi0)) / (scale / eta))
    check("p=1 ln coeff = -chi(0); p=2 1/a coeff = +chi(0)", worst_v, 1e-11)
    # Truncation-limited: the reference is a central difference at h = eta/1e4,
    # so ~1e-9..1e-8 is the floor.  The chi(0) identity above is the exact one.
    check("p=2 ln coeff = -dchi/dw(0)  (finite-diff reference)", worst_d, 1e-7)


# ---------------------------------------------------------------------------
# 8. Sampled mode unchanged by the refactor
# ---------------------------------------------------------------------------

def test_sampled_regression(baseline_dir):
    """Compare phase 3 against the pre-refactor implementation."""
    print(f"\n[8] Sampled mode vs baseline at {baseline_dir}")
    if not os.path.isdir(baseline_dir):
        print("  SKIP (baseline worktree not found)")
        return
    import importlib

    system = _honeycomb_system(m=0.1)
    dim = len(system.matrices[0].H)
    directions = ['yyy', 'yxx', 'xyx']
    dir_chars = ['x', 'y']
    eflist = np.array([0.0, 0.05])
    omega = np.linspace(0.05, 3.0, 17)
    args = (system, np.array([0.37, 0.11, 0.0]), dim, dir_chars, directions,
            omega, 0.0, None, 0.02, None, eflist, 1e-3, 2, len(omega))

    from tightbinding.calc.nonlinear_optical_fast import _process_kpoint_fast
    new = _process_kpoint_fast(*args, eta_sos=0.02)

    saved = sys.path[:]
    saved_mods = {k: v for k, v in sys.modules.items()
                  if k == 'tightbinding' or k.startswith('tightbinding.')}
    try:
        for k in list(saved_mods):
            del sys.modules[k]
        sys.path.insert(0, baseline_dir)
        old_mod = importlib.import_module(
            'tightbinding.calc.nonlinear_optical_fast')
        old = old_mod._process_kpoint_fast(*args, eta_sos=0.02)
    finally:
        sys.path[:] = saved
        for k in list(sys.modules):
            if k == 'tightbinding' or k.startswith('tightbinding.'):
                del sys.modules[k]
        sys.modules.update(saved_mods)

    worst = 0.0
    worst_name = ''
    for name in new:
        for abc in directions:
            n, o = new[name][abc], old[name][abc]
            scale = max(np.max(np.abs(o)), 1e-30)
            err = np.max(np.abs(n - o)) / scale
            if err > worst:
                worst, worst_name = err, f'{name}.{abc}'
    check(f"worst rel. err over all 26 terms ({worst_name})", worst, 1e-12)


if __name__ == '__main__':
    test_note_reference()
    test_against_mesh()
    test_two_distinct_poles()
    test_infinite_upper_limit()
    test_endpoint_coefficients()
    test_end_to_end()
    test_endpoint_diagnostic_identity()
    test_sampled_regression(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASELINE)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL TESTS PASSED")
