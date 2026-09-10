"""Sum-rule check:  int_0^inf dw (1/w) Re chi^{abc}(w; w, 0)  =  pi * dQ^{ab}(c).

Left side: the analytic frequency-integrated chi^(2) (calc.freq_integral,
p = 1, omega_max = inf), in the code's convention where chi_total is the
second-order conductivity, so Re is the dissipative part.  Right side: the
thermal delta_Q engine on the same k-grid with the same kT and eta_sos, so
both sides use identical operators.  All eight in-plane components are
compared, and several index assignments are tested so the pairing of
(output a, omega-field b, DC-field c) with dQ^{ab}(c) is checked rather than
assumed (the yyy component alone cannot tell them apart).

What was established with this script (2026-09-10):

* The pairing is the plain one, J^{abc} = pi dQ^{ab}(c): on the anisotropic
  honeycomb (C3 broken by 30%) every independent component obeys it and the
  alternative pairings fail by 15-20%.
* At finite eta the ratio J/(pi dQ) is 1 + O(eta ln(1/eta)), from the
  1/(w + i eta)-type poles under the 1/w weight: 1.0142 at eta = 1e-3 on the
  honeycomb (gap 0.2), 1.0013 on MoS2 (gap 1.76); the residual fits
  A eta ln(1/eta) + B eta with no constant term, so the eta -> 0 limit is 1
  to better than 1e-4.  Quote the sum rule with the eta it was run at, or
  extrapolate.
* With the Wannier-gauge correction on, the rule holds component-wise only
  for a = c.  For a != c the deviation is antisymmetric under a <-> c
  (MoS2: xxy -2.2%, yxx +2.3%; their sum obeys the rule), independent of
  nk, eta and eta_sos, and it is NOT symmetry breaking or time reversal:
  it is the non-Abelian curvature F^{ac} = d_a A_c - d_c A_a - i[A_a, A_c]
  of the Wannier connection, i.e. the projected position components of a
  truncated Wannier basis do not commute.  The models 'honeycomb_wflat'
  (F = 0 exactly) and 'honeycomb_wcurv' (same size, F != 0) demonstrate
  it: the first obeys the rule in all components, the second shows the
  antisymmetric violation.  The (a <-> c)-symmetrized relation holds.

Run (serial or under mpiexec):

    python3 examples/check_sumrule_chi2_dQ.py honeycomb --nk 240 --eta 1e-3
    python3 examples/check_sumrule_chi2_dQ.py honeycomb_wcurv --nk 240
    mpiexec -np 12 python3 examples/check_sumrule_chi2_dQ.py mos2 \\
        --mos2-dir /path/with/mos2_tb.dat --nk 200 --eta 1e-3 --eta-sos 0.025

Models: honeycomb (gapped, D3h), honeycomb_aniso (uniaxial strain, C3
broken), honeycomb_wflat / honeycomb_wcurv (synthetic Wannier connection,
flat / curved), honeycomb_wreal / honeycomb_wcplx (random Hermitian-paired
position blocks, real = TRS kept / complex = TRS broken), mos2 (needs
mos2_tb.dat and mos2_centres.xyz in --mos2-dir).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tightbinding import parallel                                    # noqa: E402
from tightbinding.calc.nonlinear_optical import compute_nonlinear_optical  # noqa: E402
from tightbinding.calc.delta_Q import compute_delta_Q                # noqa: E402

DIRS = [a + b + c for a in 'xy' for b in 'xy' for c in 'xy']


def build_honeycomb(variant):
    from tightbinding.lattice import build_system
    from tightbinding.hamiltonian import fill_hamiltonian
    cfg = {'system': {'basis': 's_u',
                      'lattice_vectors': [[0.8660254037844386, -1.5, 0.0],
                                          [0.8660254037844386, 1.5, 0.0],
                                          [0.0, 0.0, 20.0]],
                      'positions': [{'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                                    {'species': 'B', 'coord': [0.0, 1.0, 0.0]}]},
           'hopping': {'range': 1.1, 'tss_sigma': 1.0},
           'onsite': {'A': {'u_s': 0.1}, 'B': {'u_s': -0.1}}}
    if variant == 'aniso':      # uniaxial strain along x: C3 broken, M_x kept
        cfg['hopping'].update({'anisotropy_factor': 0.3, 'anisotropy_direction': [1, 0, 0]})
    system = build_system(cfg)
    fill_hamiltonian(system)
    n = system.norbs
    tau = np.zeros((n, 3))
    for atom in system.atoms:
        tau[atom.orb_slice] = atom.coord

    if variant in ('wflat', 'wcurv'):
        # A^W_b(k) = eps * sum_i R_{i,b} M_i cos(k.R_i): real, TRS-preserving.
        # flat:   M_1 = M_2               -> F = dA - dA - i[A, A] = 0 exactly
        # curved: M_1, M_2 non-commuting  -> F = -i eps^2 cos cos (R1 ^ R2) [M1, M2]
        eps = 0.15
        M1 = np.diag([1.0, -1.0])
        M2 = M1.copy() if variant == 'wflat' else np.array([[0.0, 1.0], [1.0, 0.0]])
        disp = [np.zeros(3)]
        mats = [[np.diag(tau[:, a]).astype(complex) for a in range(3)]]
        for Rl, M in [((1, 0, 0), M1), ((0, 1, 0), M2)]:
            Rc = np.array(Rl, float) @ system.unitcell_vectors
            blocks = [0.5 * eps * Rc[a] * M.astype(complex) for a in range(3)]
            disp.append(np.array(Rl, float))
            mats.append(blocks)
            disp.append(-np.array(Rl, float))
            mats.append([b.conj().T for b in blocks])
        system.wannier_r_matrices, system.wannier_r_displacements = mats, disp
    elif variant in ('wreal', 'wcplx'):
        rng = np.random.default_rng(7)
        scale = 0.1

        def rand_mat():
            M = rng.normal(size=(n, n)) * scale
            if variant == 'wcplx':
                M = M + 1j * rng.normal(size=(n, n)) * scale
            return M

        disp = [np.zeros(3)]
        mats = [[0.5 * (M + M.conj().T) for M in (rand_mat() for _ in range(3))]]
        for R in [(1, 0, 0), (0, 1, 0), (1, 1, 0)]:
            blocks = [rand_mat() for _ in range(3)]
            disp.append(np.array(R, float))
            mats.append(blocks)
            disp.append(-np.array(R, float))
            mats.append([b.conj().T for b in blocks])
        system.wannier_r_matrices, system.wannier_r_displacements = mats, disp
    return system, 0.0, 1e-5


def build_mos2(directory):
    from tightbinding.wannier import build_system_from_tb
    tb = os.path.join(directory, 'mos2_tb.dat')
    cen = os.path.join(directory, 'mos2_centres.xyz')
    system = build_system_from_tb(tb, centres_path=cen if os.path.exists(cen) else None)
    return system, -0.030102, 0.025      # midgap Ef of that file, kT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('model', choices=['honeycomb', 'honeycomb_aniso', 'honeycomb_wflat',
                                      'honeycomb_wcurv', 'honeycomb_wreal', 'honeycomb_wcplx',
                                      'mos2'])
    ap.add_argument('--nk', type=int, default=240)
    ap.add_argument('--eta', type=float, default=1e-3)
    ap.add_argument('--omega-min', type=float, default=None, help='default 10*eta')
    ap.add_argument('--eta-sos', type=float, default=None,
                    help='default 1e-8 for the honeycombs, 0.025 for mos2')
    ap.add_argument('--wannier-r', type=int, default=1, choices=[0, 1])
    ap.add_argument('--mos2-dir', default='.')
    args = ap.parse_args()

    if args.model == 'mos2':
        system, ef, kT = build_mos2(args.mos2_dir)
        eta_sos = 0.025 if args.eta_sos is None else args.eta_sos
    else:
        system, ef, kT = build_honeycomb(args.model.split('_', 1)[1] if '_' in args.model else '')
        eta_sos = 1e-8 if args.eta_sos is None else args.eta_sos
    omega_min = 10 * args.eta if args.omega_min is None else args.omega_min
    wannier_r = bool(args.wannier_r)
    nk = args.nk

    cfg_chi = {'system': {'wannier_r': wannier_r},
               'calc': {'type': 'nonlinear_optical', 'nk': [nk, nk],
                        'freq_integral': {'p': [1], 'omega_min': omega_min,
                                          'omega_max': float('inf')},
                        'omega2': 0.0, 'eta': args.eta, 'eta_sos': eta_sos,
                        'eflist': [ef], 'kT': kT, 'directions': DIRS}}
    res_chi = compute_nonlinear_optical(system, cfg_chi)

    cfg_dq = {'system': {'wannier_r': wannier_r},
              'calc': {'type': 'delta_Q', 'formulation': 'thermal',
                       'components': ['xx', 'xy', 'yx', 'yy'], 'field_direction': ['x', 'y'],
                       'nk': [nk, nk], 'eflist': [ef], 'kT': kT,
                       'eta_sos': eta_sos, 'eta': 0.0}}
    res_dq = compute_delta_Q(system, cfg_dq)

    if not parallel.is_root():
        return

    print(f"\n##### {args.model}: nk={nk} eta={args.eta} omega_min={omega_min} "
          f"eta_sos={eta_sos} wannier_r={wannier_r} kT={kT} ef={ef}")
    print(f"{'abc':>4} {'Re J':>14} {'Im J':>14} {'Re logcoef':>12} {'pi*dQ':>14} {'ReJ/(pi dQ)':>12}")
    J, D = {}, {}
    for abc in DIRS:
        a, b, c = abc
        Jv = res_chi['chi_total'][a][b][c][0, 0]
        L = res_chi['endpt_log_chi_total'][a][b][c][0, 0]
        dq = res_dq['delta_Q'][a][b][c][0]
        J[abc], D[abc] = Jv.real, np.pi * dq.real
        ratio = Jv.real / D[abc] if abs(D[abc]) > 1e-300 else float('nan')
        print(f"{abc:>4} {Jv.real:+14.6e} {Jv.imag:+14.6e} {L.real:+12.3e} "
              f"{D[abc]:+14.6e} {ratio:+12.6f}")

    scale = max(abs(v) for v in D.values())
    hyps = {
        'J^abc = pi dQ^ab(c)': lambda a, b, c: D[a + b + c],
        'J^abc = pi dQ^cb(a)': lambda a, b, c: D[c + b + a],
        'J^abc = pi dQ^bc(a)': lambda a, b, c: D[b + c + a],
        'J^abc = pi dQ^ac(b)': lambda a, b, c: D[a + c + b],
        '(J^abc + J^cba) = pi (dQ^ab(c) + dQ^cb(a))':
            lambda a, b, c: (D[a + b + c] + D[c + b + a]) * J[a + b + c] / (J[a + b + c] + J[c + b + a])
            if abs(J[a + b + c] + J[c + b + a]) > 1e-12 else float('nan'),
    }
    print("  index-assignment test, J/(pi dQ) per hypothesis (components > 1% of the largest):")
    for name, f in hyps.items():
        row = []
        for abc in DIRS:
            a, b, c = abc
            ref = f(a, b, c)
            if abs(D[abc]) > 1e-2 * scale and np.isfinite(ref) and abs(ref) > 1e-2 * scale:
                row.append(f"{abc}:{J[abc] / ref:+.4f}")
        print(f"     {name:<44s} " + "  ".join(row))


if __name__ == '__main__':
    main()
