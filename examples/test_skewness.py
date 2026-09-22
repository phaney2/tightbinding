"""Certificates for the polarization-cumulant engine (calc/skewness.py).

The engine returns the variance and the third cumulant (skewness) of the
ground-state polarization per unit cell, arXiv:2409.16358 Eqs. 12-13 with
``skewness = -(paper Eq. 13)`` so that it is the real-space cumulant.  Checks:

1.  Sign and normalization, real space.  The paper's Rice-Mele chain at
    delta = 1 is a set of decoupled dimers (sites at 0 and 1/2), so the
    electron distribution per cell is exactly Bernoulli and its cumulants are
    known in closed form: kappa2 = q(1-q)/4, kappa3 = q(1-q)(1-2q)/8 with
    q = (m + s)/(2s), s = sqrt(4 + m^2).  No band theory involved.
2.  The paper's closed forms for the Rice-Mele chain (elliptic integral) and
    the stub-SSH chain (three bands, non-trivial three-band terms), with the
    overall minus sign.
3.  Pointwise: the analytic tr[P d_aP d_bd_cP] against central differences of
    the projector P(k) = U_occ U_occ^dag itself (gauge-invariant, so no phase
    alignment), including a model with EXACT occupied-occupied degeneracies
    (spin-degenerate s_ud honeycomb), where the r/r^{b;c} expansion would
    need eta_sos and the projector form does not.
4.  Pointwise: the same trace re-expressed in the band operators r and the
    Sipe r^{b;c} of nonlinear_optical._compute_dk_rmtx plus the three-band
    terms, on a non-degenerate model.
5.  BZ-integrated: total symmetry in (a,b,c) (a total-derivative identity for
    point-like orbitals), the honeycomb's C3 relations, spin doubling, and
    variance == quantum_metric's Q on the same grid.
6.  Gauge covariance (the rule every engine obeys): the model re-expressed in
    another gauge (atompos = 0, or shifted centres) plus the Wannier
    correction reproduces the atomic-gauge per-k traces; the no-correction
    control fails.
7.  A synthetic k-dependent connection (random Hermitian-paired position
    blocks): covariant engine == band form with the corrected r, r^{b;c}
    pointwise, and the b<->c antisymmetric part (the connection's curvature)
    is nonzero there and zero for point-like orbitals.
8.  A metallic Fermi level raises.
9.  Optional: pass the directory holding mos2_tb.dat / mos2_centres.xyz as
    argv[1] for a real-data run (D3h relations, curvature asymmetry).

Run:  python3 examples/test_skewness.py [/path/to/MoS2_dir]
"""

import copy
import os
import sys
import warnings

import numpy as np
from scipy.special import ellipe

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings('ignore', category=RuntimeWarning)

from tightbinding.types import System, HoppingMatrix, AtomPos           # noqa: E402
from tightbinding.bloch import get_H_v, diagonalize_hk, get_reciprocal_lattice  # noqa: E402
from tightbinding.calc.nonlinear_optical import _compute_dk_rmtx        # noqa: E402
from tightbinding.calc.wannier_gauge import (                           # noqa: E402
    compute_A_W_k, apply_wannier_correction, covariant_H_derivatives,
    reset_notices)
from tightbinding.calc.skewness import compute_skewness, _process_kpoint  # noqa: E402
from tightbinding.calc.quantum_metric import compute_quantum_metric     # noqa: E402
from test_wannier_gauge import (                                        # noqa: E402
    honeycomb_system, orbital_centres, regauge,
    decorate_pointlike, decorate_random, kpoints)

FAILURES = []
DIRS = ['x', 'y']
G3 = None      # set in main: the gapped 3-atom model and a Fermi level in its gap
G3_EF = None
KFRACS = [(0.37, 0.11), (0.31, 0.29), (0.05, -0.23)]


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {label:<62s} {detail}")
    if not ok:
        FAILURES.append(label)
    return ok


def check_close(label, err, tol):
    return check(label, np.isfinite(err) and err <= tol, f"{err:.3e}  (tol {tol:.0e})")


def quiet(fn, *a, **kw):
    """Run an engine call with its progress prints suppressed."""
    import io
    import contextlib
    reset_notices()
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def stub_rice_mele(delta, m, Delta, L=20.0):
    """The paper's stub Rice-Mele chain: A, C at x = 0, B at x = 1/2, cell 1.

    H(k)_AB = (1+delta) e^{ik/2} + (1-delta) e^{-ik/2} in the code's Bloch
    convention, which is the paper's Eq. (S-Bloch).
    """
    n = 3
    H0 = np.array([[m, 1 + delta, Delta],
                   [1 + delta, -m, 0.0],
                   [Delta, 0.0, 0.0]], dtype=complex)
    Hm = np.zeros((n, n), dtype=complex)
    Hm[0, 1] = 1 - delta            # <0 A| H |R = -1, B>
    z = np.zeros((n, n), dtype=complex)
    mats = [HoppingMatrix(displacement=np.zeros(3), H=H0, S=np.eye(n, dtype=complex)),
            HoppingMatrix(displacement=np.array([-1.0, 0, 0]), H=Hm, S=z),
            HoppingMatrix(displacement=np.array([1.0, 0, 0]), H=Hm.conj().T, S=z)]
    c = np.array([[0.0, 0, 0], [0.5, 0, 0], [0.0, 0, 0]])
    ap = AtomPos(x=c[None, :, 0] - c[:, None, 0],
                 y=np.zeros((n, n)), z=np.zeros((n, n)))
    return System(atoms=[], matrices=mats, unitcell_vectors=np.diag([1.0, L, L]),
                  norbs=n, atompos=ap)


def honeycomb_ud(m=0.1, T=1.0):
    """Spin-degenerate honeycomb: every band exactly two-fold degenerate."""
    from tightbinding.lattice import build_system
    from tightbinding.hamiltonian import fill_hamiltonian
    cfg = {'system': {'basis': 's_ud',
                      'lattice_vectors': [[0.8660254037844386, -1.5, 0.0],
                                          [0.8660254037844386, 1.5, 0.0],
                                          [0.0, 0.0, 20.0]],
                      'positions': [{'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                                    {'species': 'B', 'coord': [0.0, 1.0, 0.0]}]},
           'hopping': {'range': 1.1, 'tss_sigma': T},
           'onsite': {'A': {'u_s': m}, 'B': {'u_s': -m}}}
    s = build_system(cfg)
    fill_hamiltonian(s)
    return s


def generic3_gapped():
    """`test_wannier_gauge.generic3_system` (three sp_u atoms at low-symmetry
    positions, Rashba) with hoppings x0.7 and A/C shifted by +/-4 so it is an
    insulator: 4 occupied, 8 unoccupied bands, global gap 0.69."""
    from tightbinding.lattice import build_system
    from tightbinding.hamiltonian import fill_hamiltonian
    t = 0.7
    cfg = {'system': {'basis': 'sp_u',
                      'lattice_vectors': [[2.0, 0.0, 0.0], [0.3, 1.8, 0.0], [0.0, 0.0, 20.0]],
                      'positions': [{'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                                    {'species': 'B', 'coord': [0.9, 0.4, 0.3]},
                                    {'species': 'C', 'coord': [1.3, 1.3, -0.2]}]},
           'hopping': {'range': 2.3, 'tss_sigma': -0.7 * t, 'tsp_sigma': 0.5 * t,
                       'tpp_sigma': 0.9 * t, 'tpp_pi': -0.2 * t, 'tsp_rashba': 0.15 * t},
           'onsite': {'A': {'u_s': 5.0, 'u_p': 3.5}, 'B': {'u_s': 0.4, 'u_p': 0.2},
                      'C': {'u_s': -4.8, 'u_p': -3.3}}}
    s = build_system(cfg)
    fill_hamiltonian(s)
    return s


def gap_ef(system, n_occ=1, nk=200):
    """A Fermi level in the gap above band `n_occ` (scan of a 1D grid)."""
    b1 = get_reciprocal_lattice(system.unitcell_vectors)[0]
    top, bot = -np.inf, np.inf
    for i in range(nk):
        H, S, _ = get_H_v(system, -b1 / 2 + b1 * i / nk)
        ek = diagonalize_hk(H, S)
        top, bot = max(top, ek[n_occ - 1]), min(bot, ek[n_occ])
    assert top < bot, "no gap"
    return 0.5 * (top + bot)


def gap_ef_2d(system, nk=40):
    """Fermi level in the widest global gap of a 2D model (grid scan)."""
    b1, b2, _ = get_reciprocal_lattice(system.unitcell_vectors)
    eks = []
    for i in range(nk):
        for j in range(nk):
            H, S, _ = get_H_v(system, -b1 / 2 - b2 / 2 + b1 * i / nk + b2 * j / nk)
            eks.append(diagonalize_hk(H, S))
    eks = np.array(eks)
    top, bot = eks.max(axis=0)[:-1], eks.min(axis=0)[1:]
    n = int(np.argmax(bot - top))
    assert bot[n] > top[n], "no global gap"
    return 0.5 * (top[n] + bot[n])


def run_1d(system, nk=200, ef=None):
    if ef is None:
        ef = gap_ef(system)
    cfg = {'system': {}, 'calc': {'nk': [nk, 1], 'eflist': [ef], 'directions': ['x']}}
    r = quiet(compute_skewness, system, cfg)
    return float(r['variance']['x']['x'][0]), float(r['skewness']['x']['x']['x'][0])


def run_2d(system, nk, ef, dirs=DIRS, wannier_r=None):
    cfg = {'system': {}, 'calc': {'nk': [nk, nk], 'eflist': [ef], 'directions': dirs}}
    if wannier_r is not None:
        cfg['system']['wannier_r'] = wannier_r
    return quiet(compute_skewness, system, cfg)


def per_k(system, k, ef, wannier_r, dirs=DIRS):
    """Engine per-k traces (var (nd,nd), T (nd,nd,nd)) for one ef."""
    var, T, _, _ = _process_kpoint(system, k, dirs, np.array([ef]), wannier_r)
    return var[0], T[0]


# ---------------------------------------------------------------------------
# [1] real-space Bernoulli pin, [2] the paper's closed forms
# ---------------------------------------------------------------------------

def test_dimer_and_closed_forms():
    print("\n[1] Decoupled dimers: cumulants of an exact Bernoulli distribution")
    worst = 0.0
    for m in (0.5, 1.0, -0.7, 2.0):
        s = np.sqrt(4 + m * m)
        q = (m + s) / (2 * s)                  # weight on site B (x = 1/2)
        k2 = q * (1 - q) / 4
        k3 = q * (1 - q) * (1 - 2 * q) / 8
        v, sk = run_1d(stub_rice_mele(1.0, m, 0.0), nk=16)
        worst = max(worst, abs(v - k2) / abs(k2), abs(sk - k3) / abs(k3))
    check_close("dimer: variance and skewness == Bernoulli kappa2, kappa3 (rel)", worst, 1e-12)
    _, sk = run_1d(stub_rice_mele(1.0, 1.0, 0.0), nk=16)
    check("dimer m > 0: weight piles up at +x, so skewness < 0", sk < 0, f"{sk:+.4e}")

    print("\n[2] Paper closed forms (skewness = -paper Eq. 13)")
    worst_v = worst_s = 0.0
    for delta, m in [(0.3, 0.8), (0.6, -0.4), (0.15, 1.5), (-0.5, 0.7)]:
        var_p = (1 + delta**2) / (8 * np.sqrt((4 + m**2) * (m**2 + 4 * delta**2)))
        skew_p = (m * delta / (4 * (4 + m**2) * np.pi * np.sqrt(m**2 + 4 * delta**2))
                  * ellipe(4 * (delta**2 - 1) / (m**2 + 4 * delta**2)))
        v, sk = run_1d(stub_rice_mele(delta, m, 0.0), nk=400)
        worst_v = max(worst_v, abs(v - var_p) / abs(var_p))
        worst_s = max(worst_s, abs(sk + skew_p) / abs(skew_p))
    check_close("Rice-Mele: variance == Var(delta, m, 0) (rel)", worst_v, 1e-10)
    check_close("Rice-Mele: skewness == -Skew(delta, m, 0) (rel)", worst_s, 1e-10)

    worst_v = worst_s = 0.0
    for delta, D in [(0.4, 0.9), (0.7, 1.6), (-0.3, 0.5)]:
        A = (4 + D**2) * (4 * delta**2 + D**2)
        var_p = (4 * D**2 + (1 + delta**2) * (D**4 + 4 * delta**2 * (2 + D**2))) / (4 * A**1.5)
        skew_p = -(delta * D**2 * (-12 - 12 * delta**6 + D**2 + 5 * D**4 + D**6
                                   + delta**4 * (28 + D**2)
                                   + delta**2 * (28 + 30 * D**2 + 5 * D**4))
                   / (4 * A**2.5))
        v, sk = run_1d(stub_rice_mele(delta, 0.0, D), nk=400)
        worst_v = max(worst_v, abs(v - var_p) / abs(var_p))
        worst_s = max(worst_s, abs(sk + skew_p) / abs(skew_p))
    check_close("stub-SSH (3 bands): variance == Var(delta, 0, Delta) (rel)", worst_v, 1e-10)
    check_close("stub-SSH (3 bands): skewness == -Skew(delta, 0, Delta) (rel)", worst_s, 1e-10)
    v, sk = run_1d(stub_rice_mele(1.0, 0.0, 1.3), nk=16)
    D = 1.3
    check_close("stub dimer: skewness == +Delta^2(2+Delta^2)/(4(4+Delta^2)^3) (rel)",
                abs(sk - D**2 * (2 + D**2) / (4 * (4 + D**2)**3)) / abs(sk), 1e-12)


# ---------------------------------------------------------------------------
# [3] finite differences of the projector, [4] band form
# ---------------------------------------------------------------------------

def projector(system, k, ef):
    H, S, _ = get_H_v(system, k)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    occ = psi[:, ek < ef]
    return occ @ occ.conj().T


def fd_traces(system, k, ef, h, dirs=DIRS):
    """tr[P d_aP d_bP], tr[P d_aP d_bd_cP] by central differences of P(k)."""
    e = {'x': np.array([1.0, 0, 0]), 'y': np.array([0, 1.0, 0]), 'z': np.array([0, 0, 1.0])}
    P0 = projector(system, k, ef)
    dP = {a: (projector(system, k + h * e[a], ef) - projector(system, k - h * e[a], ef)) / (2 * h)
          for a in dirs}
    ddP = {}
    for b in dirs:
        ddP[b] = {}
        for c in dirs:
            if b == c:
                ddP[b][c] = (projector(system, k + h * e[b], ef) - 2 * P0
                             + projector(system, k - h * e[b], ef)) / h**2
            else:
                ddP[b][c] = (projector(system, k + h * (e[b] + e[c]), ef)
                             - projector(system, k + h * (e[b] - e[c]), ef)
                             - projector(system, k - h * (e[b] - e[c]), ef)
                             + projector(system, k - h * (e[b] + e[c]), ef)) / (4 * h**2)
    nd = len(dirs)
    var = np.zeros((nd, nd), dtype=complex)
    T = np.zeros((nd, nd, nd), dtype=complex)
    for ia, a in enumerate(dirs):
        for ib, b in enumerate(dirs):
            var[ia, ib] = np.trace(P0 @ dP[a] @ dP[b])
            for ic, c in enumerate(dirs):
                T[ia, ib, ic] = np.trace(P0 @ dP[a] @ ddP[b][c])
    return var, T


def test_fd():
    print("\n[3] Analytic traces vs central differences of the projector P(k)")
    for name, system, ef in [('3-atom sp', G3, G3_EF),
                             ('honeycomb', honeycomb_system(), 0.0),
                             ('s_ud honeycomb (exact occ-occ degeneracy)', honeycomb_ud(), 0.0)]:
        errs = {}
        for h in (2e-3, 1e-3):
            worst = 0.0
            for k in kpoints(system, KFRACS):
                var, T = per_k(system, k, ef, False)
                var_fd, T_fd = fd_traces(system, k, ef, h)
                scale = max(np.max(np.abs(T)), np.max(np.abs(var)))
                worst = max(worst, np.max(np.abs(T - T_fd)) / scale,
                            np.max(np.abs(var - var_fd)) / scale)
            errs[h] = worst
        check_close(f"{name}: var, T == FD at h = 1e-3 (rel)", errs[1e-3], 1e-4)
        ratio = errs[2e-3] / errs[1e-3]
        check(f"{name}: FD error is O(h^2)", 3.0 < ratio < 5.0, f"ratio {ratio:.2f}")


def band_form(system, k, ef, wannier_r, dirs=DIRS, eta_sos=1e-10):
    """T^{a;bc} = sum r^a_nm r^{b;c}_mn + i sum r^a_nm (sum_{p occ} r^b_mp r^c_pn
                                                         - sum_{p unocc} r^c_mp r^b_pn)
    over n occ, m unocc — the r / r^{b;c} expansion of tr[P d_aP d_c d_bP].
    (r^{b;c} is the derivative of r^b along c, so this is the engine's T[a][c][b].)
    """
    n = system.norbs
    H, S, vtb = get_H_v(system, k, order=2)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    de = ek[:, None] - ek[None, :]
    inv_de = np.where(np.abs(de) > 1e-12, de / (de**2 + eta_sos**2), 0.0)
    vm = {d: psi.conj().T @ vtb[d] @ psi for d in dirs}
    Delta = {d: np.diag(vm[d])[:, None] - np.diag(vm[d])[None, :] for d in dirs}
    rbar = {d: -1j * vm[d] * inv_de for d in dirs}
    dkr = {a: {b: _compute_dk_rmtx(vm[a], vm[b], psi.conj().T @ vtb[a + b] @ psi,
                                   Delta[a], Delta[b], de, inv_de, n)
               for b in dirs} for a in dirs}
    A_W, dA_W = compute_A_W_k(system, k, dirs, enabled=wannier_r)
    r, corr, _ = apply_wannier_correction(A_W, dA_W, psi, rbar, dirs)
    if corr is not None:
        dkr = {a: {b: dkr[a][b] + corr[a][b] for b in dirs} for a in dirs}
    f = (ek < ef).astype(float)
    Po = np.diag(f)
    Pu = np.diag(1 - f)
    nd = len(dirs)
    T = np.zeros((nd, nd, nd), dtype=complex)
    for ia, a in enumerate(dirs):
        ra = Po @ r[a] @ Pu                     # n occ, m unocc
        for ib, b in enumerate(dirs):
            for ic, c in enumerate(dirs):
                three = r[b] @ Po @ r[c] - r[c] @ Pu @ r[b]
                M = dkr[b][c] + 1j * three       # (m, n) block used below
                T[ia, ib, ic] = np.sum(ra * M.T)
    return T


def test_band_form():
    print("\n[4] Projector form vs band form (r, Sipe r^{b;c}, three-band terms)")
    for name, system, ef in [('3-atom sp', G3, G3_EF),
                             ('honeycomb', honeycomb_system(), 0.0)]:
        worst = 0.0
        for k in kpoints(system, KFRACS):
            _, T = per_k(system, k, ef, False)
            Tb = band_form(system, k, ef, False)
            worst = max(worst, np.max(np.abs(np.swapaxes(T, 1, 2) - Tb)) / np.max(np.abs(T)))
        check_close(f"{name}: T^{{a;bc}} == band form (rel)", worst, 1e-9)


# ---------------------------------------------------------------------------
# [5] BZ-integrated identities
# ---------------------------------------------------------------------------

def test_bz():
    print("\n[5] BZ-integrated identities (nk = 60)")
    hc = honeycomb_system()
    r = run_2d(hc, 60, 0.0)
    sk = r['skewness']
    s = {a + b + c: float(sk[a][b][c][0]) for a in DIRS for b in DIRS for c in DIRS}
    scale = max(abs(v) for v in s.values())
    check_close("honeycomb: total symmetry xxy = xyx = yxx (rel)",
                max(abs(s['xxy'] - s['xyx']), abs(s['xxy'] - s['yxx'])) / scale, 1e-12)
    check_close("honeycomb: C3, yyy = -xxy (rel)", abs(s['yyy'] + s['xxy']) / scale, 1e-12)
    check_close("honeycomb: C3, xxx = xyy = 0 (rel)", max(abs(s['xxx']), abs(s['xyy'])) / scale, 1e-12)
    check("honeycomb: skewness is live", scale > 1e-3, f"|yyy| = {abs(s['yyy']):.4e}")
    var = r['variance']
    check_close("honeycomb: isotropic variance, xx = yy, xy = 0 (rel)",
                max(abs(var['x']['x'][0] - var['y']['y'][0]), abs(var['x']['y'][0]))
                / abs(var['x']['x'][0]), 1e-12)

    r3 = run_2d(G3, 60, G3_EF)
    sk3 = r3['skewness']
    s3 = {a + b + c: float(sk3[a][b][c][0]) for a in DIRS for b in DIRS for c in DIRS}
    scale3 = max(abs(v) for v in s3.values())
    check_close("3-atom sp: total symmetry xxy = xyx = yxx, xyy = yxy = yyx (rel)",
                max(abs(s3['xxy'] - s3['xyx']), abs(s3['xxy'] - s3['yxx']),
                    abs(s3['xyy'] - s3['yxy']), abs(s3['xyy'] - s3['yyx'])) / scale3, 1e-12)

    ud = run_2d(honeycomb_ud(), 60, 0.0)
    check_close("s_ud honeycomb: skewness == 2 x spinless (rel)",
                abs(ud['skewness']['y']['y']['y'][0] - 2 * s['yyy']) / abs(2 * s['yyy']), 1e-12)

    qm_cfg = {'system': {}, 'calc': {'nk': [60, 60], 'eflist': [0.0], 'kT': 1e-4,
                                     'eta': 0.0, 'metric_directions': DIRS}}
    qm = quiet(compute_quantum_metric, hc, qm_cfg)
    worst = max(abs(qm['Q'][a][b][0].real - var[a][b][0]) for a in DIRS for b in DIRS)
    check_close("honeycomb: variance == quantum_metric Q (abs)", worst, 1e-10)

    # nk convergence: analytic periodic integrand -> exponential (the error
    # squares when nk doubles).  m = 0.4 (gap 0.8) so nk = 60 is converged;
    # the m = 0.1 model (gap 0.2) needs nk ~ 240 for 1e-12.
    hc4 = honeycomb_system(m=0.4)
    v = [run_2d(hc4, nk, 0.0)['skewness']['y']['y']['y'][0] for nk in (30, 60, 120)]
    d30, d60 = abs(v[0] - v[2]), abs(v[1] - v[2])
    check("honeycomb m=0.4: nk 30 -> 60 -> 120 converges exponentially",
          d60 < 1e-11 and d60 < 1e-4 * d30, f"|30-120| = {d30:.2e}, |60-120| = {d60:.2e}")


# ---------------------------------------------------------------------------
# [6] gauge covariance, [7] synthetic k-dependent connection
# ---------------------------------------------------------------------------

def test_gauge():
    print("\n[6] Gauge covariance: re-gauged model + correction == atomic gauge")
    rng = np.random.default_rng(5)
    for name, system, ef in [('honeycomb', honeycomb_system(), 0.0),
                             ('3-atom sp', G3, G3_EF)]:
        tau = orbital_centres(system)
        variants = [np.zeros_like(tau), tau + rng.normal(size=tau.shape), tau]
        worst_on, worst_off = 0.0, np.inf
        for k in kpoints(system, KFRACS):
            varA, TA = per_k(system, k, ef, False)
            scale = np.max(np.abs(TA))
            for cen in variants:
                s = decorate_pointlike(regauge(system, cen), tau)
                var, T = per_k(s, k, ef, True)
                worst_on = max(worst_on, np.max(np.abs(T - TA)) / scale,
                               np.max(np.abs(var - varA)) / scale)
            s = decorate_pointlike(regauge(system, np.zeros_like(tau)), tau)
            _, T = per_k(s, k, ef, False)
            worst_off = min(worst_off, np.max(np.abs(T - TA)) / scale)
        check_close(f"{name}: 3 gauges x {len(KFRACS)} k, var and T (rel)", worst_on, 1e-12)
        check(f"{name}: control (no correction) fails as it must", worst_off > 1e-2,
              f"rel err {worst_off:.2e}")


def test_synthetic_connection():
    print("\n[7] Synthetic k-dependent connection: covariant engine == corrected band form")
    system = decorate_random(honeycomb_system(), seed=7, scale=0.3)
    worst, anti_w, anti_p = 0.0, 0.0, 0.0
    plain = honeycomb_system()
    for k in kpoints(system, KFRACS):
        _, T = per_k(system, k, 0.0, True)
        Tb = band_form(system, k, 0.0, True)
        scale = np.max(np.abs(T))
        worst = max(worst, np.max(np.abs(np.swapaxes(T, 1, 2) - Tb)) / scale)
        anti_w = max(anti_w, np.max(np.abs(T - np.swapaxes(T, 1, 2))) / scale)
        _, Tp = per_k(plain, k, 0.0, False)
        anti_p = max(anti_p, np.max(np.abs(Tp - np.swapaxes(Tp, 1, 2))) / np.max(np.abs(Tp)))
    check_close("T^{a;bc} == band form with corrected r, r^{b;c} (rel)", worst, 1e-9)
    check("b<->c antisymmetric part is live with curvature", anti_w > 1e-2, f"rel {anti_w:.2e}")
    check_close("b<->c antisymmetric part vanishes for point-like orbitals", anti_p, 1e-13)
    # the covariant kernel reduces to the bare one when the connection is absent
    H, S, vtb = get_H_v(plain, kpoints(plain, KFRACS)[0], order=2)
    V, W = covariant_H_derivatives(None, None, H, vtb, DIRS)
    check("covariant_H_derivatives(None) returns the bare derivatives",
          all(V[a] is vtb[a] for a in DIRS) and all(W[b][c] is vtb[b + c] for b in DIRS for c in DIRS))


# ---------------------------------------------------------------------------
# [8] metal, [9] real data
# ---------------------------------------------------------------------------

def test_metal():
    print("\n[8] A metallic Fermi level is refused")
    hc = honeycomb_system()
    try:
        run_2d(hc, 12, -1.5)      # inside the lower band
        check("ef inside a band raises", False)
    except ValueError as e:
        check("ef inside a band raises", 'inside a band' in str(e))


def test_real_tb(directory):
    print(f"\n[9] Real _tb.dat data from {directory}")
    from tightbinding.wannier import build_system_from_tb
    tb = os.path.join(directory, 'mos2_tb.dat')
    cen = os.path.join(directory, 'mos2_centres.xyz')
    if not os.path.exists(tb):
        print("  (mos2_tb.dat not found; skipping)")
        return
    sA = build_system_from_tb(tb, centres_path=cen if os.path.exists(cen) else None)
    # gauge covariance on the real connection
    sL = regauge(sA, np.zeros((sA.norbs, 3)))
    sL.wannier_r_matrices = sA.wannier_r_matrices
    sL.wannier_r_displacements = sA.wannier_r_displacements
    ef = 0.0
    worst = 0.0
    for k in kpoints(sA, KFRACS):
        _, TA = per_k(sA, k, ef, True)
        _, TL = per_k(sL, k, ef, True)
        worst = max(worst, np.max(np.abs(TA - TL)) / np.max(np.abs(TA)))
    check_close("atomic vs lattice gauge on the real connection (rel)", worst, 1e-10)
    for wr in (True, False):
        cfg = {'system': {'wannier_r': wr},
               'calc': {'nk': [36, 36], 'eflist': [ef], 'directions': DIRS}}
        reset_notices()
        r = compute_skewness(sA, cfg)
        sk = r['skewness']
        s = {a + b + c: float(sk[a][b][c][0]) for a in DIRS for b in DIRS for c in DIRS}
        print(f"  wannier_r={wr}: n_occ={r['n_occ'][0]}, gap={r['gap'][0]:.3f}, "
              f"var xx={r['variance']['x']['x'][0]:.5f} yy={r['variance']['y']['y'][0]:.5f}")
        print("   skewness " + "  ".join(f"{t}={s[t]:+.5f}" for t in
                                         ('yyy', 'xxy', 'xyx', 'yxx', 'xxx', 'xyy', 'yxy', 'yyx')))
        ssym = r['skewness_sym']
        print("   fully symmetrized yyy=%+.5f xxy=%+.5f" % (ssym['y']['y']['y'][0], ssym['x']['x']['y'][0]))


if __name__ == '__main__':
    G3 = generic3_gapped()
    G3_EF = gap_ef_2d(G3)
    test_dimer_and_closed_forms()
    test_fd()
    test_band_form()
    test_bz()
    test_gauge()
    test_synthetic_connection()
    test_metal()
    if len(sys.argv) > 1:
        test_real_tb(sys.argv[1])
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All skewness certificates passed.")
