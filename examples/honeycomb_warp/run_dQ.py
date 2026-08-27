"""DC-field-induced quantum metric of gapped graphene vs. the warping note.

Runs delta_Q on the honeycomb s-orbital model for a sweep of (m, nk) and
compares the full-BZ integral against Eq. 15 of tmd_warping_note_corrected.pdf.

Mapping (a = NN distance, T = tss_sigma):
    v = 3Ta/2,  lam = -3Ta^2/8,  m = (u_s^A - u_s^B)/2,  lam/v = -a/4

Eq. 15 per valley, K -> infinity:   int d2k dg^yy(Ey) = -pi*lam/(2*v*m)
With lam/v = -a/4 this collapses to  +pi*a/(8m), independent of T.
Both valleys are valley-even and add:  T^yyy = pi*a/(4m).

The code returns the BZ *average*; multiply by the BZ area to get int d2k.

Usage:
    mpiexec --use-hwthread-cpus -np 8 python3 examples/honeycomb_warp/run_dQ.py
"""
import sys

import numpy as np

sys.path.insert(0, '.')

from tightbinding.lattice import build_system
from tightbinding.hamiltonian import fill_hamiltonian
from tightbinding.calc.delta_Q import compute_delta_Q
from tightbinding import parallel

A_NN = 1.0
T_HOP = 1.0
SQ3 = np.sqrt(3.0)

CELL_AREA = 1.5 * SQ3 * A_NN**2          # 3*sqrt(3)/2 a^2
BZ_AREA = (2 * np.pi) ** 2 / CELL_AREA

AB_PAIRS = ['xx', 'xy', 'yx', 'yy']
FIELD_DIRS = ['x', 'y']

# T^{abc} = int d2k dg^{ab}(E_c).  Channels with an odd number of x indices
# must vanish by the T*M_x selection rule (Eq. 12).
CHANNELS = [(a + b + c, (a + b + c).count('x') % 2 == 1)
            for a in 'xy' for b in 'xy' for c in 'xy']


def make_cfg(m, nk):
    return {
        'system': {
            'basis': 's_u',
            'lattice_vectors': [
                [0.5 * SQ3 * A_NN, -1.5 * A_NN, 0.0],
                [0.5 * SQ3 * A_NN,  1.5 * A_NN, 0.0],
                [0.0, 0.0, 20.0],
            ],
            'positions': [
                {'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                {'species': 'B', 'coord': [0.0, A_NN, 0.0]},
            ],
        },
        'hopping': {'range': 1.1 * A_NN, 'tss_sigma': T_HOP},
        'onsite': {'A': {'u_s': m}, 'B': {'u_s': -m}},
        'calc': {
            'type': 'delta_Q',
            'components': AB_PAIRS,
            'field_direction': FIELD_DIRS,
            'nk': [nk, nk],
            'eflist': [0.0],          # mid-gap
            'kT': m / 50.0,           # kT << gap
            'eta': 0.0,               # note has no broadening
            'eta_sos': 1.0e-8,        # bands never degenerate (gap = 2m)
        },
    }


def run(m, nk):
    cfg = make_cfg(m, nk)
    if parallel.is_root():
        system = build_system(cfg)
        fill_hamiltonian(system)
    else:
        system = None
    system = parallel.bcast(system)

    res = compute_delta_Q(system, cfg)['delta_Q']
    # BZ average -> integral over d2k
    return {a + b + c: BZ_AREA * res[a][b][c][0].real
            for a in 'xy' for b in 'xy' for c in 'xy'}


def main():
    m_list = [float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1
                                 else ['0.1'])]
    nk_list = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2
                                else ['201', '401', '601'])]

    parallel.print_root(f"BZ area = {BZ_AREA:.6f}   cell area = {CELL_AREA:.6f}")

    for m in m_list:
        pred = np.pi * A_NN / (4.0 * m)     # both valleys, Eq. 15
        parallel.print_root(
            f"\n=== m = {m}  (gap {2*m})   v = {1.5*T_HOP*A_NN}, "
            f"lam = {-0.375*T_HOP*A_NN**2} ===")
        parallel.print_root(
            f"    Eq. 15 prediction  T^yyy = pi*a/(4m) = {pred:+.6f}")
        header = "  {:>5}".format('nk') + ''.join(
            f"{name:>13}" for name, _ in CHANNELS) + f"{'yyy/pred':>11}"
        parallel.print_root(header)
        for nk in nk_list:
            t = run(m, nk)
            row = f"  {nk:>5}" + ''.join(
                f"{t[name]:>13.6f}" for name, _ in CHANNELS)
            row += f"{t['yyy']/pred:>11.4f}"
            parallel.print_root(row)


if __name__ == '__main__':
    main()
