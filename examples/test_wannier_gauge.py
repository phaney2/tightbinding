"""Certificates for the Wannier-gauge position correction (calc/wannier_gauge.py).

The correction — a^(H) added to the interband r (Eq. 22 of arXiv:1804.04030)
and the covariant-derivative piece added to r^{a;b} — is checked here against
exact statements, not reference numbers:

1.  Gauge covariance, kernel level.  A multi-atom TB_simple model has
    point-like orbitals, so in the atomic gauge (atompos from the atom
    coordinates) r = -i v/w and the Sipe sum rule are exact with no
    correction.  Re-express the SAME model with a different atompos — zero
    (the lattice gauge) or the true centres plus random per-orbital offsets —
    and decorate it with the point-like position blocks <0n|r|0m> = tau_n d_nm.
    Then A^(W) = diag(tau - tau') exactly and the corrected r and r^{a;b} must
    reproduce the atomic-gauge ones, compared through phase-invariant products
    so eigenvector phases drop out.  This pins the commutator's factor of i,
    the sign and completeness of the diagonal-connection terms, and the
    centre subtraction, all of which were wrong before
    (BUG_wannier_r_correction.md).  The no-correction control must FAIL.

2.  Gauge covariance, engine level.  The same comparison on the per-k outputs
    of chi^(2) (fast and reference paths), delta_Q (thermal, subspace, band,
    plus the RTA piece) and quantum_metric (Q, dQf), so every call site is
    exercised.  This is what decides where the bare velocity stays (Sipe sum
    rule, Delta, band curvature, inverse mass) and where the corrected
    r / i w r goes (current vertex, projector derivatives, the metric).
    quantum_metric's finite-difference dQ is a bare-velocity heuristic and is
    reported, not asserted.  The Sipe sub-term split (Delta/d2H/3band/
    wannier_corr) is gauge-dependent by construction; only totals are compared.

3.  Finite-difference certificate with a k-DEPENDENT connection.  1 and 2
    cannot see the U^dag (d_b A^(W)_a) U term, because A^(W) is k-independent
    there.  Decorate the honeycomb with synthetic Hermitian-paired position
    blocks (as a real _tb.dat gives) and check the analytic r^{a;b} against
    phase-aligned central differences of r, using the definition
    r^a_{;b} = d_b r^a - i (xi^b_nn - xi^b_mm) r^a.

4.  Hermiticity: A^(W)(k), the corrected r, and corr^{a;b}_nm^* = corr^{a;b}_mn.

5.  Optional, on a real _tb.dat (pass the directory holding mos2_tb.dat and
    mos2_centres.xyz as argv[1]): loader diagnostics, then 1 and 3 on the
    real connection.  The BZ-integrated checks of BUG_wannier_r_correction.md
    (delta_Q real, sub-gap dissipation -> 0) are separate runs, not here.

Run:  python3 examples/test_wannier_gauge.py [/path/to/MoS2_dir]
"""

import copy
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore', category=RuntimeWarning)

from tightbinding.types import AtomPos                                  # noqa: E402
from tightbinding.bloch import get_H_v, diagonalize_hk, get_reciprocal_lattice  # noqa: E402
from tightbinding.calc.nonlinear_optical import _compute_dk_rmtx        # noqa: E402
from tightbinding.calc.wannier_gauge import (                           # noqa: E402
    compute_A_W_k, apply_wannier_correction, reset_notices)

FAILURES = []
DIRS3 = ['x', 'y', 'z']
DIRS = ['x', 'y']
KFRACS = [(0.37, 0.11), (0.31, 0.29), (0.05, -0.23)]


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'} {label:<58s} {detail}")
    if not ok:
        FAILURES.append(label)
    return ok


def check_close(label, err, tol):
    return check(label, np.isfinite(err) and err <= tol, f"{err:.3e}  (tol {tol:.0e})")


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def honeycomb_system(m=0.1, T=1.0):
    from tightbinding.lattice import build_system
    from tightbinding.hamiltonian import fill_hamiltonian
    cfg = {'system': {'basis': 's_u',
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


def generic3_system():
    """Three sp_u atoms at low-symmetry positions, complex hoppings via Rashba."""
    from tightbinding.lattice import build_system
    from tightbinding.hamiltonian import fill_hamiltonian
    cfg = {'system': {'basis': 'sp_u',
                      'lattice_vectors': [[2.0, 0.0, 0.0], [0.3, 1.8, 0.0], [0.0, 0.0, 20.0]],
                      'positions': [{'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                                    {'species': 'B', 'coord': [0.9, 0.4, 0.3]},
                                    {'species': 'C', 'coord': [1.3, 1.3, -0.2]}]},
           'hopping': {'range': 2.3, 'tss_sigma': -0.7, 'tsp_sigma': 0.5,
                       'tpp_sigma': 0.9, 'tpp_pi': -0.2, 'tsp_rashba': 0.15},
           'onsite': {'A': {'u_s': 1.0, 'u_p': -0.5}, 'B': {'u_s': 0.4, 'u_p': 0.2},
                      'C': {'u_s': -0.8, 'u_p': 0.7}}}
    s = build_system(cfg)
    fill_hamiltonian(s)
    return s


def orbital_centres(system):
    tau = np.zeros((system.norbs, 3))
    for atom in system.atoms:
        tau[atom.orb_slice] = atom.coord
    return tau


def regauge(system, centres):
    """The same physical system with Bloch phases built from `centres` (norbs, 3)."""
    s = copy.deepcopy(system)
    c = np.asarray(centres, dtype=float)
    s.atompos = AtomPos(x=-(c[:, 0, None] - c[None, :, 0]),
                        y=-(c[:, 1, None] - c[None, :, 1]),
                        z=-(c[:, 2, None] - c[None, :, 2]))
    for attr in ('_bloch_H_stack', '_bloch_S_stack', '_bloch_R_stack'):
        if hasattr(s, attr):
            delattr(s, attr)
    return s


def decorate_pointlike(system, tau):
    """Position blocks of point-like orbitals at `tau`: <0n|r|0m> = tau_n d_nm only."""
    system.wannier_r_matrices = [[np.diag(tau[:, i]).astype(complex) for i in range(3)]]
    system.wannier_r_displacements = [np.zeros(3)]
    return system


def decorate_random(system, seed=7, scale=0.3):
    """Synthetic Hermitian-paired position blocks on R = 0, +/-a1, +/-a2, +/-(a1+a2)."""
    rng = np.random.default_rng(seed)
    n = system.norbs

    def rand_mat():
        return (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * scale

    displacements = [np.zeros(3)]
    mats = [[0.5 * (M + M.conj().T) for M in (rand_mat() for _ in range(3))]]
    for R in [(1, 0, 0), (0, 1, 0), (1, 1, 0)]:
        blocks = [rand_mat() for _ in range(3)]
        displacements.append(np.array(R, dtype=float))
        mats.append(blocks)
        displacements.append(-np.array(R, dtype=float))
        mats.append([b.conj().T for b in blocks])
    system.wannier_r_matrices = mats
    system.wannier_r_displacements = displacements
    return system


def kpoints(system, fracs=KFRACS):
    b1, b2, b3 = get_reciprocal_lattice(system.unitcell_vectors)
    return [f[0] * b1 + f[1] * b2 for f in fracs]


# ---------------------------------------------------------------------------
# Operators at a k-point, through the real kernel
# ---------------------------------------------------------------------------

def operators(system, k, wannier_r, eta_sos=1e-8, psi_ref=None):
    """(ek, psi, r, r^{a;b}, corr, A_W); phases aligned to psi_ref if given."""
    n = system.norbs
    H, S, vtb = get_H_v(system, k, order=2)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    if psi_ref is not None:
        ov = np.sum(psi_ref.conj() * psi, axis=0)
        psi = psi * np.exp(-1j * np.angle(ov))[None, :]
    de = ek[:, None] - ek[None, :]
    inv_de = np.where(np.abs(de) > 1e-12, de / (de ** 2 + eta_sos ** 2), 0.0)
    vm = {d: psi.conj().T @ vtb[d] @ psi for d in DIRS3}
    Delta = {d: np.diag(vm[d])[:, None] - np.diag(vm[d])[None, :] for d in DIRS3}
    rbar = {d: -1j * vm[d] * inv_de for d in DIRS3}
    dkr = {a: {b: _compute_dk_rmtx(vm[a], vm[b], psi.conj().T @ vtb[a + b] @ psi,
                                   Delta[a], Delta[b], de, inv_de, n)
               for b in DIRS3} for a in DIRS3}
    A_W, dA_W = compute_A_W_k(system, k, DIRS3, enabled=wannier_r)
    r, corr, _ = apply_wannier_correction(A_W, dA_W, psi, rbar, DIRS3)
    if corr is not None:
        dkr = {a: {b: dkr[a][b] + corr[a][b] for b in DIRS3} for a in DIRS3}
    return ek, psi, r, dkr, corr, A_W


def invariants(r, dkr):
    """Phase-invariant products: |r^a_nm|, r^a_nm r^b_mn, r^a_nm r^{b;c}_mn."""
    out = {}
    for a in DIRS3:
        out['abs_' + a] = np.abs(r[a])
        for b in DIRS3:
            out[f'rr_{a}{b}'] = r[a] * r[b].T
            for c in DIRS3:
                out[f'rdk_{a}{b}{c}'] = r[a] * dkr[b][c].T
    return out


def compare_invariants(ref, other):
    """(rel err in r products, rel err in r r^{b;c} products)."""
    e_r = max(np.abs(other[f'rr_{a}{b}'] - ref[f'rr_{a}{b}']).max()
              for a in DIRS3 for b in DIRS3)
    s_r = max(np.abs(ref[f'rr_{a}{b}']).max() for a in DIRS3 for b in DIRS3)
    e_dk = max(np.abs(other[f'rdk_{a}{b}{c}'] - ref[f'rdk_{a}{b}{c}']).max()
               for a in DIRS3 for b in DIRS3 for c in DIRS3)
    s_dk = max(np.abs(ref[f'rdk_{a}{b}{c}']).max() for a in DIRS3 for b in DIRS3 for c in DIRS3)
    return e_r / s_r, e_dk / s_dk


# ---------------------------------------------------------------------------
# [1] kernel-level gauge covariance
# ---------------------------------------------------------------------------

def test_kernel_covariance():
    print("\n[1] Kernel-level gauge covariance: re-gauged model + correction == atomic gauge")
    rng = np.random.default_rng(3)
    for name, system in [('honeycomb', honeycomb_system()), ('3-atom sp', generic3_system())]:
        tau = orbital_centres(system)
        variants = [('lattice gauge (atompos = 0)', np.zeros_like(tau)),
                    ('shifted atompos (tau + random)', tau + rng.normal(size=tau.shape)),
                    ('atomic atompos (tau)', tau)]
        worst_on = 0.0
        worst_off = np.inf
        for k in kpoints(system):
            _, _, rA, dkA, _, _ = operators(system, k, False)
            ref = invariants(rA, dkA)
            for _label, cen in variants:
                s = decorate_pointlike(regauge(system, cen), tau)
                _, _, r, dk, _, _ = operators(s, k, True)
                worst_on = max(worst_on, *compare_invariants(ref, invariants(r, dk)))
            # control: lattice gauge with the correction OFF must be wrong
            s = decorate_pointlike(regauge(system, np.zeros_like(tau)), tau)
            _, _, r, dk, _, _ = operators(s, k, False)
            worst_off = min(worst_off, max(compare_invariants(ref, invariants(r, dk))))
        check_close(f"{name}: 3 gauges x {len(KFRACS)} k, r and r^{{a;b}} (rel)", worst_on, 1e-11)
        check(f"{name}: control (no correction) fails as it must", worst_off > 1e-2,
              f"rel err {worst_off:.2e}")


# ---------------------------------------------------------------------------
# [2] engine-level gauge covariance
# ---------------------------------------------------------------------------

TRIPLETS = ['yyy', 'yxx', 'xxy', 'xyx', 'xxx', 'yyx']


def run_chi2_fast(system, k, wannier_r, ef):
    from tightbinding.calc.nonlinear_optical_fast import _process_kpoint_fast
    dim = system.norbs
    return _process_kpoint_fast(
        system, k, dim, DIRS, TRIPLETS, np.array([0.3, 0.9]), 0.0, None, 1e-3, None,
        np.array(ef), 0.05, len(ef), 2, eta_sos=1e-8, wannier_r=wannier_r)


def run_chi2_slow(system, k, wannier_r, ef):
    from tightbinding.calc.nonlinear_optical import _process_kpoint
    dim = system.norbs
    eta = 1e-3
    return _process_kpoint(
        system, k, dim, DIRS, TRIPLETS, np.array([0.3, 0.9]), 0.0, np.zeros((dim, dim)),
        eta, eta * np.ones((dim, dim)), np.array(ef), 0.05, len(ef), 2,
        eta_sos=1e-8, wannier_r=wannier_r)


def run_dq(system, k, wannier_r, ef, formulation):
    from tightbinding.calc.delta_Q import _process_kpoint
    res, _terms, tau = _process_kpoint(
        system, k, DIRS, ['xx', 'xy', 'yy'], ['x', 'y'], np.array(ef), 0.05,
        len(ef), 0.05, eta_sos=1e-8, wannier_r=wannier_r, formulation=formulation)
    return {'dQ': res, 'dQ_tau': tau if tau is not None else {}}


def run_qm(system, k, wannier_r, ef):
    from tightbinding.calc.quantum_metric import _process_kpoint
    params = {'dim': system.norbs, 'dir_chars': DIRS, 'eflist': np.array(ef),
              'kT': 0.05, 'eta': 1e-3, 'delta': 1e-3, 'nef': len(ef),
              'wannier_r': wannier_r}
    return _process_kpoint(system, k, params)


ENGINES = [
    # The Sipe sub-term split (chi_ei*_sipe_delta/d2H/3band/wannier_corr) is
    # gauge-dependent by construction — only its sum is; compare the 14 chi
    # terms and the covariant dk_f / delta_r pieces.
    ('chi2 fast', lambda s, k, w, ef: {n: v for n, v in run_chi2_fast(s, k, w, ef).items()
                                       if '_sipe_' not in n}),
    ('chi2 reference', lambda s, k, w, ef: {n: v for n, v in run_chi2_slow(s, k, w, ef).items()
                                            if '_sipe_' not in n}),
    ('delta_Q thermal', lambda s, k, w, ef: run_dq(s, k, w, ef, 'thermal')),
    ('delta_Q subspace', lambda s, k, w, ef: run_dq(s, k, w, ef, 'subspace')),
    ('delta_Q band', lambda s, k, w, ef: run_dq(s, k, w, ef, 'band')),
    ('quantum_metric Q, dQf', lambda s, k, w, ef: {q: v for q, v in run_qm(s, k, w, ef).items()
                                                   if q != 'dQ'}),
]


def _flat(d, prefix=''):
    if isinstance(d, dict):
        out = {}
        for key, val in d.items():
            out.update(_flat(val, f"{prefix}/{key}"))
        return out
    return {prefix: np.asarray(d)}


def rel_diff(a, b):
    fa, fb = _flat(a), _flat(b)
    scale = max(np.abs(v).max() for v in fa.values())
    worst = max(np.abs(fa[key] - fb[key]).max() for key in fa)
    return worst / scale


def test_engine_covariance():
    print("\n[2] Engine-level gauge covariance (per k-point, lattice gauge + correction == atomic)")
    for name, system, ef in [('honeycomb', honeycomb_system(), [0.0, 1.0]),
                             ('3-atom sp', generic3_system(), [0.0])]:
        tau = orbital_centres(system)
        lat = decorate_pointlike(regauge(system, np.zeros_like(tau)), tau)
        for ename, run in ENGINES:
            worst_on, worst_off = 0.0, np.inf
            for k in kpoints(system):
                reset_notices()
                ref = run(system, k, False, ef)
                worst_on = max(worst_on, rel_diff(ref, run(lat, k, True, ef)))
                worst_off = min(worst_off, rel_diff(ref, run(lat, k, False, ef)))
            check_close(f"{name}: {ename} (rel, all outputs)", worst_on, 1e-9)
            check(f"{name}: {ename} control (no correction) differs", worst_off > 1e-3,
                  f"rel {worst_off:.2e}")
        # quantum_metric's finite-difference dQ perturbs the states with the
        # bare-velocity vertex and cannot be made covariant without the
        # k-derivatives that delta_Q carries; report it, do not assert on it.
        k = kpoints(system)[0]
        dq = rel_diff(run_qm(system, k, False, ef)['dQ'], run_qm(lat, k, True, ef)['dQ'])
        print(f"  info {name}: quantum_metric dQ (FD heuristic) gauge dependence: rel {dq:.2e}")


# ---------------------------------------------------------------------------
# [3] finite-difference certificate with a k-dependent A^(W)
# ---------------------------------------------------------------------------

def fd_certificate(system, k0, h, eta_sos=1e-10):
    """max rel err of analytic r^{a;c} vs phase-aligned central differences,
    and max |corr| / |r^{a;c}| so the caller knows the correction was live."""
    _, psi0, r0, dk0, corr0, A_W0 = operators(system, k0, True, eta_sos)
    A_H0 = {d: psi0.conj().T @ A_W0[d] @ psi0 for d in DIRS3}
    xi0 = {d: np.diag(A_H0[d]).real for d in DIRS3}
    worst, live = 0.0, 0.0
    for cidx, c in enumerate(DIRS):
        e = np.zeros(3)
        e[cidx] = h
        _, _, rp, _, _, _ = operators(system, k0 + e, True, eta_sos, psi_ref=psi0)
        _, _, rm, _, _, _ = operators(system, k0 - e, True, eta_sos, psi_ref=psi0)
        for a in DIRS:
            fd = (rp[a] - rm[a]) / (2 * h) - 1j * (xi0[c][:, None] - xi0[c][None, :]) * r0[a]
            np.fill_diagonal(fd, 0.0)
            scale = np.abs(dk0[a][c]).max()
            worst = max(worst, np.abs(fd - dk0[a][c]).max() / scale)
            live = max(live, np.abs(corr0[a][c]).max() / scale)
    return worst, live


def test_fd_certificate():
    print("\n[3] Finite-difference certificate of r^{a;b} with a k-dependent A^(W)")
    system = decorate_random(honeycomb_system())
    worst, live = 0.0, 0.0
    for k in kpoints(system, [(0.31, 0.29), (0.2, 0.05)]):
        w, l = fd_certificate(system, k, 1e-4)
        worst, live = max(worst, w), max(live, l)
    check_close("decorated honeycomb: analytic == FD (rel, h=1e-4)", worst, 1e-6)
    check("decorated honeycomb: correction is a live fraction of r^{a;b}", live > 0.1,
          f"|corr|/|r^{{a;b}}| = {live:.2f}")


# ---------------------------------------------------------------------------
# [4] Hermiticity
# ---------------------------------------------------------------------------

def hermiticity(system, ks, eta_sos=1e-8):
    worst = 0.0
    for k in ks:
        _, _, r, _, corr, A_W = operators(system, k, True, eta_sos)
        worst = max(worst, *(np.abs(A_W[d] - A_W[d].conj().T).max() for d in DIRS3))
        worst = max(worst, *(np.abs(r[d] - r[d].conj().T).max() for d in DIRS3))
        worst = max(worst, *(np.abs(corr[a][b] - corr[a][b].conj().T).max()
                             for a in DIRS3 for b in DIRS3))
    return worst


def test_hermiticity():
    print("\n[4] Hermiticity of A^(W), r and corr^{a;b}")
    system = decorate_random(honeycomb_system())
    check_close("decorated honeycomb: max anti-Hermitian part", hermiticity(system, kpoints(system)), 1e-12)
    system = decorate_random(generic3_system(), seed=11)
    check_close("decorated 3-atom sp: max anti-Hermitian part", hermiticity(system, kpoints(system)), 1e-12)


# ---------------------------------------------------------------------------
# [5] optional: a real _tb.dat
# ---------------------------------------------------------------------------

def test_real_tb(directory):
    print(f"\n[5] Real _tb.dat in {directory}")
    from tightbinding.wannier import build_system_from_tb
    tb = os.path.join(directory, 'mos2_tb.dat')
    cen = os.path.join(directory, 'mos2_centres.xyz')
    if not os.path.exists(tb):
        print("  (mos2_tb.dat not found; skipping)")
        return
    sA = build_system_from_tb(tb, centres_path=cen if os.path.exists(cen) else None)
    ks = [k for k in kpoints(sA, [(0.31, 0.29), (0.05, -0.23), (0.2, 0.05)])]
    # Wannier90 writes the off-diagonal blocks un-Hermitianized; the loader fixes that.
    check_close("A^(W), r, corr Hermitian on the real data", hermiticity(sA, ks, 0.025), 1e-12)
    # gauge covariance: same data, Bloch phases from zero centres
    sL = regauge(sA, np.zeros((sA.norbs, 3)))
    sL.wannier_r_matrices = sA.wannier_r_matrices
    sL.wannier_r_displacements = sA.wannier_r_displacements
    worst = 0.0
    for k in ks:
        _, _, rA, dkA, _, _ = operators(sA, k, True)
        _, _, rL, dkL, _, _ = operators(sL, k, True)
        worst = max(worst, *compare_invariants(invariants(rA, dkA), invariants(rL, dkL)))
    check_close("atomic vs lattice gauge on the real connection (rel)", worst, 1e-11)
    worst, live = 0.0, 0.0
    for k in ks:
        w, l = fd_certificate(sA, k, 2e-5)
        worst, live = max(worst, w), max(live, l)
    check_close("real connection: analytic r^{a;b} == FD (rel, h=2e-5)", worst, 1e-5)
    check("real connection: correction is live", live > 0.01, f"|corr|/|r^{{a;b}}| = {live:.2f}")


if __name__ == '__main__':
    test_kernel_covariance()
    test_engine_covariance()
    test_fd_certificate()
    test_hermiticity()
    if len(sys.argv) > 1:
        test_real_tb(sys.argv[1])
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All Wannier-gauge certificates passed.")
