"""Validation of the thermal (metallic, finite-T) delta_Q formulation.

Implements the numerical certificate of delta_Q_metal_finite_T.pdf
(Appendix A) against the engine in calc/delta_Q.py:

1.  Insulator limit: on the gapped honeycomb at beta*E_gap = 20 the
    thermal path reproduces the subspace path (ratio +1, validating the
    overall -1 engine-convention factor vs the note's H' = +E.r), and
    the legacy dQ_occupied_subspace key maps to the right formulations.
2.  Trace assembly: on a random complex 5-band metal (beta = 3, mu in
    the band manifold), the engine's collected formula (note eq:final)
    equals Tr I + II + III assembled independently from the building
    blocks delta_rho, d_a rho, d_a delta_rho — *including all f' pieces*
    (note eqs. drho, dka_rho, ddrho_offdiag/dF/ddrho_diag).  This is the
    note's "2e-16" check and confirms the f'-cancellation theorem holds
    in the implementation.
3.  Finite differences: the same quantity from central differences in
    the field (rho_tilde = f(H + E r_perp) by direct diagonalization)
    and in k of the gauge-invariant rho_tilde — no band-basis input.
4.  Metallic sanity: Re dQ^{ab} = Re dQ^{ba} (the metric part is
    symmetric); T_loop is alive in the metal and dead in the insulator;
    a BZ sweep of a metallic honeycomb is finite with default eta_sos.
5.  RTA piece (always computed on the thermal path, reported PER UNIT
    tau): engine eq:dQtau vs the three traces with D_hat inserted for
    delta_rho (analytic d_a D_hat, incl. the mass sum rule);
    exponentially dead in the gapped limit; time-reversal rule
    Re[dQ_tau(k) + dQ_tau(-k)] = 0 on a real-hopping metal while the
    individual values are nonzero.

Run:  python3 examples/test_dQ_thermal.py
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings('ignore', category=RuntimeWarning)

from tightbinding.types import System, HoppingMatrix, AtomPos     # noqa: E402
from tightbinding.bloch import get_H_v, diagonalize_hk            # noqa: E402
from tightbinding.calc.delta_Q import (                           # noqa: E402
    _process_kpoint, _compute_dk_rmtx, _thermal_tables,
    compute_delta_Q,
)
from test_wannier_r_flag import honeycomb_system                  # noqa: E402

FAILURES = []


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {label:<58s} {detail}")
    if not ok:
        FAILURES.append(label)
    return ok


def check_close(label, err, tol):
    return check(label, np.isfinite(err) and err <= tol,
                 f"{err:.3e}  (tol {tol:.0e})")


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def random_system(n=5, seed=11, nhop=6, scale=0.35, real=False):
    """The note's certificate model: H(k) = H0 + sum_m (T_m e^{ikR} + h.c.).

    Random complex hoppings on a cubic lattice (unit cell = identity,
    zero AtomPos), R in {-2..2}^3, H(-R) = H(R)^dagger.  `real=True`
    gives real hoppings, i.e. a time-reversal-symmetric model with
    H(-k) = H(k)*.
    """
    rng = np.random.default_rng(seed)
    zero = np.zeros((n, n))

    def rand_mat(s):
        m = rng.normal(size=(n, n)).astype(complex)
        if not real:
            m = m + 1j * rng.normal(size=(n, n))
        return m * s

    H0 = rand_mat(0.5)
    H0 = 0.5 * (H0 + H0.conj().T) + np.diag(np.linspace(-2.0, 2.0, n))
    mats = [HoppingMatrix(displacement=np.zeros(3), H=H0,
                          S=np.eye(n, dtype=complex))]
    for _ in range(nhop):
        R = rng.integers(-2, 3, size=3).astype(float)
        if not np.any(R):
            R[0] = 1.0
        T = rand_mat(scale)
        mats.append(HoppingMatrix(displacement=R, H=T,
                                  S=np.zeros((n, n), dtype=complex)))
        mats.append(HoppingMatrix(displacement=-R, H=T.conj().T,
                                  S=np.zeros((n, n), dtype=complex)))
    return System(atoms=[], matrices=mats, unitcell_vectors=np.eye(3),
                  norbs=n, atompos=AtomPos(x=zero, y=zero, z=zero))


K0 = np.array([0.31, -0.42, 0.17])


def fermi(e, mu, kT):
    x = np.clip((np.asarray(e) - mu) / kT, -500, 500)
    return 1.0 / (1.0 + np.exp(x))


# ---------------------------------------------------------------------------
# Independent building blocks (note Secs. 3-4), plain loops throughout
# ---------------------------------------------------------------------------

def band_operators(system, k, dirs):
    """ek, band-basis v, w matrices, bare 1/w, r = -i v/w."""
    H, S, vtb = get_H_v(system, k, order=2)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    n = len(ek)
    vm = {d: psi.conj().T @ vtb[d] @ psi for d in dirs}
    vv = {d1 + d2: psi.conj().T @ vtb[d1 + d2] @ psi
          for d1 in dirs for d2 in dirs}
    de = ek[:, None] - ek[None, :]
    off = ~np.eye(n, dtype=bool)
    inv_de = np.where(off, 1.0 / np.where(off, de, 1.0), 0.0)
    rm = {d: -1j * vm[d] * inv_de for d in dirs}
    return ek, psi, vm, vv, de, inv_de, rm


def note_traces(system, k, dirs, a, b, c, mu, kT):
    """Tr I + II + III of note eq:threetraces from the raw building blocks.

    Every f'-carrying piece (the band-diagonal parts of d_rho and
    d_delta_rho, and the f' part of dF) is kept — the cancellation
    theorem is *not* assumed.
    """
    ek, _psi, vm, vv, de, inv_de, rm = band_operators(system, k, dirs)
    n = len(ek)
    f = fermi(ek, mu, kT)
    fp = -f * (1.0 - f) / kT
    F = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            F[p, q] = ((f[p] - f[q]) / de[p, q] if p != q
                       else fp[p])
    vdiag = {d: np.diag(vm[d]).real for d in dirs}
    Delta = {d: vdiag[d][:, None] - vdiag[d][None, :] for d in dirs}

    # r^{c;x} via the corrected Sipe sum rule (validated machinery)
    dk_r = {x: _compute_dk_rmtx(vm[c], vm[x], vv[c + x],
                                Delta[c], Delta[x], inv_de)
            for x in dirs}

    # delta_rho (eq:drho) and d_a rho (eq:dka_rho)
    drho = F * rm[c]
    np.fill_diagonal(drho, 0.0)

    def d_rho(x):
        m = 1j * (f[:, None] - f[None, :]) * rm[x]
        np.fill_diagonal(m, fp * vdiag[x])
        return m

    # d_a delta_rho (eqs. ddrho_offdiag, dF, ddrho_diag)
    def d_drho(x):
        m = np.zeros((n, n), dtype=complex)
        for p in range(n):
            for q in range(n):
                if p == q:
                    s = 0.0
                    for l in range(n):
                        if l != p:
                            s += F[l, p] * np.imag(rm[x][p, l] * rm[c][l, p])
                    m[p, p] = 2.0 * s
                    continue
                dF = ((fp[p] * vdiag[x][p] - fp[q] * vdiag[x][q]) / de[p, q]
                      - (f[p] - f[q]) * Delta[x][p, q] / de[p, q] ** 2)
                s3 = 0.0
                for l in range(n):
                    if l in (p, q):
                        continue
                    s3 += (F[l, q] * rm[x][p, l] * rm[c][l, q]
                           - F[p, l] * rm[c][p, l] * rm[x][l, q])
                m[p, q] = dF * rm[c][p, q] + F[p, q] * dk_r[x][p, q] - 1j * s3
        return m

    rho = np.diag(f).astype(complex)
    tr1 = np.trace(drho @ d_rho(a) @ d_rho(b))
    tr2 = np.trace(rho @ d_drho(a) @ d_rho(b))
    tr3 = np.trace(rho @ d_rho(a) @ d_drho(b))
    return tr1 + tr2 + tr3


def rta_traces(system, k, dirs, a, b, c, mu, kT):
    """D_hat inserted for delta_rho in the three traces (note Sec. 8).

    Per unit tau, matching the engine's delta_Q_tau reporting.
    """
    ek, _psi, vm, vv, de, inv_de, rm = band_operators(system, k, dirs)
    n = len(ek)
    f = fermi(ek, mu, kT)
    fp = -f * (1.0 - f) / kT
    fpp = f * (1.0 - f) * (1.0 - 2.0 * f) / kT ** 2
    vdiag = {d: np.diag(vm[d]).real for d in dirs}

    D = fp * vdiag[c]
    Dhat = np.diag(D).astype(complex)

    def d_rho(x):
        m = 1j * (f[:, None] - f[None, :]) * rm[x]
        np.fill_diagonal(m, fp * vdiag[x])
        return m

    def d_Dhat(x):
        # eq:dD with the mass sum rule eq:mass
        m = 1j * (D[:, None] - D[None, :]) * rm[x]
        mass = (np.diag(vv[x + c]).real
                + 2.0 * np.real(np.sum(vm[x] * vm[c].T * inv_de, axis=1)))
        np.fill_diagonal(m, fpp * vdiag[x] * vdiag[c] + fp * mass)
        return m

    rho = np.diag(f).astype(complex)
    tr1 = np.trace(Dhat @ d_rho(a) @ d_rho(b))
    tr2 = np.trace(rho @ d_Dhat(a) @ d_rho(b))
    tr3 = np.trace(rho @ d_rho(a) @ d_Dhat(b))
    return tr1 + tr2 + tr3


# ---------------------------------------------------------------------------
# Finite-difference certificate (gauge-invariant operators only)
# ---------------------------------------------------------------------------

def rho_tilde_orb(system, k, Ec, c, mu, kT):
    """rho_tilde = f(H + Ec * r_perp) in the orbital basis (basis-free)."""
    H, S, vtb = get_H_v(system, k, order=1)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    n = len(ek)
    de = ek[:, None] - ek[None, :]
    off = ~np.eye(n, dtype=bool)
    inv_de = np.where(off, 1.0 / np.where(off, de, 1.0), 0.0)
    rm_c = -1j * (psi.conj().T @ vtb[c] @ psi) * inv_de
    r_perp = psi @ rm_c @ psi.conj().T
    Ht = H + Ec * r_perp
    Ht = 0.5 * (Ht + Ht.conj().T)
    ekt, psit = np.linalg.eigh(Ht)
    return psit @ np.diag(fermi(ekt, mu, kT)).astype(complex) @ psit.conj().T


def QT_fd(system, k, Ec, a, b, c, mu, kT, dk=1e-4):
    """Q_T^{ab} = Tr[rho_tilde d_a rho_tilde d_b rho_tilde], k-FD."""
    ea = np.zeros(3)
    ea['xyz'.index(a)] = 1.0
    eb = np.zeros(3)
    eb['xyz'.index(b)] = 1.0
    rho0 = rho_tilde_orb(system, k, Ec, c, mu, kT)
    da = (rho_tilde_orb(system, k + dk * ea, Ec, c, mu, kT)
          - rho_tilde_orb(system, k - dk * ea, Ec, c, mu, kT)) / (2 * dk)
    db = (rho_tilde_orb(system, k + dk * eb, Ec, c, mu, kT)
          - rho_tilde_orb(system, k - dk * eb, Ec, c, mu, kT)) / (2 * dk)
    return np.trace(rho0 @ da @ db)


# ---------------------------------------------------------------------------
# Engine driver
# ---------------------------------------------------------------------------

def engine(system, k, dirs, ab, c, mu, kT, formulation='thermal', **kw):
    res, terms, rtau = _process_kpoint(
        system, k, dirs, [ab], [c], np.array([mu]), kT, 1, 0.0,
        eta_sos=1e-8, formulation=formulation, **kw)
    a, b = ab
    out = res[a][b][c][0]
    out_terms = {t: v[0] for t, v in terms[a][b][c].items()}
    out_tau = rtau[a][b][c][0] if rtau is not None else None
    return out, out_terms, out_tau


# ---------------------------------------------------------------------------
# 1: insulator limit + config plumbing
# ---------------------------------------------------------------------------

def test_insulator_limit():
    print("\n[1] Insulator limit: thermal == subspace at beta*E_gap = 20")
    sysh = honeycomb_system(m=0.1, T=1.0)  # gap 2m = 0.2
    kT = 0.01
    k = np.array([0.37, 0.11, 0.0])

    worst = 0.0
    for ab, c in (('yy', 'y'), ('xy', 'x'), ('xx', 'y')):
        th, _, _ = engine(sysh, k, ['x', 'y'], ab, c, 0.0, kT)
        sub, _, _ = engine(sysh, k, ['x', 'y'], ab, c, 0.0, kT,
                           formulation='subspace')
        scale = max(abs(sub), 1e-30)
        worst = max(worst, abs(th - sub) / scale)
    check_close("thermal == subspace (rel, 3 channels)", worst, 1e-6)

    th, _, _ = engine(sysh, k, ['x', 'y'], 'yy', 'y', 0.0, kT)
    sub, _, _ = engine(sysh, k, ['x', 'y'], 'yy', 'y', 0.0, kT,
                       formulation='subspace')
    ratio = (th / sub).real
    check("sign convention: thermal/subspace ratio = +1",
          abs(ratio - 1.0) < 1e-6, f"ratio {ratio:+.8f}")

    # Legacy key mapping is exact
    for legacy, form in ((True, 'subspace'), (False, 'band')):
        v1, _, _ = engine(sysh, k, ['x', 'y'], 'yy', 'y', 0.0, kT,
                          formulation=None, dQ_occupied_subspace=legacy)
        v2, _, _ = engine(sysh, k, ['x', 'y'], 'yy', 'y', 0.0, kT,
                          formulation=form)
        check(f"legacy dQ_occupied_subspace={legacy} == '{form}'",
              v1 == v2)

    # Config validation
    cfg = {'system': {}, 'calc': {
        'type': 'delta_Q', 'nk': [2, 2], 'eflist': [0.0], 'kT': kT,
        'components': ['yy'], 'field_direction': 'y'}}
    try:
        bad = {'system': {}, 'calc': dict(cfg['calc'],
                                          formulation='thermal',
                                          dQ_occupied_subspace=True)}
        compute_delta_Q(sysh, bad)
        check("formulation + legacy key together raise", False, "no raise")
    except ValueError:
        check("formulation + legacy key together raise", True)
    try:
        bad = {'system': {}, 'calc': dict(cfg['calc'], tau=0.1)}
        compute_delta_Q(sysh, bad)
        check("calc.tau raises (output is per unit tau)", False, "no raise")
    except ValueError:
        check("calc.tau raises (output is per unit tau)", True)


# ---------------------------------------------------------------------------
# 2: trace assembly on a random 5-band metal
# ---------------------------------------------------------------------------

def test_trace_assembly():
    print("\n[2] eq:final == Tr I+II+III (random 5-band metal, beta = 3)")
    syst = random_system(seed=11)
    dirs = ['x', 'y', 'z']
    kT = 1.0 / 3.0
    ek = band_operators(syst, K0, dirs)[0]
    mu = 0.5 * (ek[2] + ek[3])  # fractional occupations across the manifold
    gap = np.min(np.abs(ek[:, None] - ek[None, :])[~np.eye(5, dtype=bool)])
    print(f"    E = {np.array2string(ek, precision=3)}, mu = {mu:.3f}, "
          f"min |w_pq| = {gap:.3e}")

    worst = 0.0
    for ab, c in (('xy', 'z'), ('xx', 'z'), ('yz', 'x'), ('zz', 'y')):
        a, b = ab
        eng, _, _ = engine(syst, K0, dirs, ab, c, mu, kT)
        note = note_traces(syst, K0, dirs, a, b, c, mu, kT)
        err = abs(eng - (-note)) / max(abs(note), 1e-30)
        worst = max(worst, err)
    check_close("engine == -(TrI+TrII+TrIII) (rel, 4 channels)", worst, 1e-9)


# ---------------------------------------------------------------------------
# 3: finite-difference certificate
# ---------------------------------------------------------------------------

def test_finite_difference():
    print("\n[3] eq:final == d/dE Tr[rho (d rho)(d rho)] by central FD")
    syst = random_system(seed=11)
    dirs = ['x', 'y', 'z']
    kT = 1.0 / 3.0
    ek = band_operators(syst, K0, dirs)[0]
    mu = 0.5 * (ek[2] + ek[3])
    Ec = 1e-4

    worst = 0.0
    for ab, c in (('xy', 'z'), ('yy', 'x')):
        a, b = ab
        eng, _, _ = engine(syst, K0, dirs, ab, c, mu, kT)
        fd = (QT_fd(syst, K0, +Ec, a, b, c, mu, kT)
              - QT_fd(syst, K0, -Ec, a, b, c, mu, kT)) / (2 * Ec)
        err = abs(eng - (-fd)) / max(abs(fd), 1e-30)
        worst = max(worst, err)
    check_close("engine == -FD certificate (rel, 2 channels)", worst, 1e-5)


# ---------------------------------------------------------------------------
# 4: metallic sanity
# ---------------------------------------------------------------------------

def test_metallic_sanity():
    print("\n[4] Metallic sanity")
    syst = random_system(seed=11)
    dirs = ['x', 'y', 'z']
    kT = 1.0 / 3.0
    ek = band_operators(syst, K0, dirs)[0]
    mu = 0.5 * (ek[2] + ek[3])

    xy, terms_xy, _ = engine(syst, K0, dirs, 'xy', 'z', mu, kT)
    yx, _, _ = engine(syst, K0, dirs, 'yx', 'z', mu, kT)
    check_close("Re dQ^{xy} == Re dQ^{yx} (metric symmetric)",
                abs(xy.real - yx.real) / max(abs(xy.real), 1e-30), 1e-12)
    check("T_loop is alive in the metal",
          abs(terms_xy['T_loop']) > 1e-8 * abs(xy),
          f"|T_loop| {abs(terms_xy['T_loop']):.3e}")

    sysh = honeycomb_system(m=0.1, T=1.0)
    _, terms_ins, _ = engine(sysh, np.array([0.37, 0.11, 0.0]),
                             ['x', 'y'], 'yy', 'y', 0.0, 0.01)
    check("T_loop is dead in the insulator",
          abs(terms_ins['T_loop']) < 1e-12,
          f"|T_loop| {abs(terms_ins['T_loop']):.3e}")

    # Metallic honeycomb BZ sweep with the *default* eta_sos: finite,
    # metric part symmetric after BZ average.  The xy/yx channels vanish
    # by symmetry here, so their difference is checked against the scale
    # of the surviving yy channel, not against themselves.
    cfg = {'system': {}, 'calc': {
        'type': 'delta_Q', 'formulation': 'thermal', 'nk': [8, 8],
        'eflist': [-1.0], 'kT': 0.1, 'components': ['yy', 'xy', 'yx'],
        'field_direction': 'y', 'directions': ['x', 'y']}}
    res = compute_delta_Q(sysh, cfg)
    dyy = res['delta_Q']['y']['y']['y'][0]
    dxy = res['delta_Q']['x']['y']['y'][0]
    dyx = res['delta_Q']['y']['x']['y'][0]
    check("metallic BZ sweep finite and alive (default eta_sos)",
          np.isfinite(dyy) and abs(dyy) > 1e-8,
          f"dQ_yy(y) = {dyy:+.3e}")
    check_close("BZ: Re dQ^{xy} == Re dQ^{yx} (vs yy scale)",
                abs(dxy.real - dyx.real) / max(abs(dyy), 1e-30), 1e-10)


# ---------------------------------------------------------------------------
# 5: RTA transport piece
# ---------------------------------------------------------------------------

def test_rta():
    print("\n[5] RTA piece eq:dQtau")
    syst = random_system(seed=11)
    dirs = ['x', 'y', 'z']
    kT = 1.0 / 3.0
    ek = band_operators(syst, K0, dirs)[0]
    mu = 0.5 * (ek[2] + ek[3])

    worst = 0.0
    for ab, c in (('xy', 'z'), ('yy', 'x')):
        a, b = ab
        _, _, eng_tau = engine(syst, K0, dirs, ab, c, mu, kT)
        note_tau = rta_traces(syst, K0, dirs, a, b, c, mu, kT)
        err = abs(eng_tau - (-note_tau)) / max(abs(note_tau), 1e-30)
        worst = max(worst, err)
    check_close("engine tau piece == -(three traces with D_hat)", worst,
                1e-9)

    # Gapped limit: exponentially dead
    sysh = honeycomb_system(m=0.1, T=1.0)
    _, _, tau_ins = engine(sysh, np.array([0.37, 0.11, 0.0]),
                           ['x', 'y'], 'yy', 'y', 0.0, 0.005)
    check("gapped limit: |dQ_tau| vanishes", abs(tau_ins) < 1e-30,
          f"{abs(tau_ins):.3e}")

    # Time-reversal rule on a real-hopping (TR-symmetric) metal:
    # pointwise Re[dQ_tau(k) + dQ_tau(-k)] = 0, individual values alive.
    strr = random_system(seed=5, real=True)
    ekr = band_operators(strr, K0, dirs)[0]
    mur = 0.5 * (ekr[2] + ekr[3])
    _, _, tp = engine(strr, K0, dirs, 'xy', 'z', mur, kT)
    _, _, tm = engine(strr, -K0, dirs, 'xy', 'z', mur, kT)
    scale = max(abs(tp.real), abs(tm.real))
    check("TR metal: Re dQ_tau(k) individually nonzero", scale > 1e-8,
          f"|Re| {scale:.3e}")
    check_close("TR metal: Re[dQ_tau(k) + dQ_tau(-k)] = 0",
                abs((tp + tm).real) / scale, 1e-10)


if __name__ == '__main__':
    test_insulator_limit()
    test_trace_assembly()
    test_finite_difference()
    test_metallic_sanity()
    test_rta()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All thermal delta_Q tests passed.")
