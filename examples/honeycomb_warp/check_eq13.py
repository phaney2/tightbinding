"""Pointwise check of the delta_Q engine against Eqs. 7/13 of the warping note.

At each k the note's exact two-band identity reads (per unit field)

  dQ^{ab}(E_c) = -(2i/w~^2)[Q^{cb} D^a - Q^{ac} D^b]
                 - (1/w~^2)[w^{ca}_vc r^b_cv + r^a_vc w^{cb}_cv]        (Eq. 7)
  dg^yy(E_y)   = -(2/w~^2) Re[r^y_vc w^yy_cv]                           (Eq. 13)
  dg^yy(E_y)  ~= -lam*v*m / (2 eps0^4),  eps0 = sqrt(m^2 + v^2 k^2)     (Eq. 14)

Compares all three against tightbinding's _process_kpoint at the same k.
"""
import sys

import numpy as np

sys.path.insert(0, '.')

from tightbinding.lattice import build_system
from tightbinding.hamiltonian import fill_hamiltonian
from tightbinding.bloch import get_H_v, get_reciprocal_lattice, diagonalize_hk
from tightbinding.calc.delta_Q import _process_kpoint

A_NN, T_HOP, M = 1.0, 1.0, 0.1
SQ3 = np.sqrt(3.0)
V_KP, LAM_KP = 1.5 * T_HOP * A_NN, -0.375 * T_HOP * A_NN**2

cfg = {
    'system': {
        'basis': 's_u',
        'lattice_vectors': [[0.5*SQ3, -1.5, 0.0], [0.5*SQ3, 1.5, 0.0],
                            [0.0, 0.0, 20.0]],
        'positions': [{'species': 'A', 'coord': [0.0, 0.0, 0.0]},
                      {'species': 'B', 'coord': [0.0, A_NN, 0.0]}],
    },
    'hopping': {'range': 1.1, 'tss_sigma': T_HOP},
    'onsite': {'A': {'u_s': M}, 'B': {'u_s': -M}},
}
system = build_system(cfg)
fill_hamiltonian(system)

b1, b2, _ = get_reciprocal_lattice(cfg['system']['lattice_vectors'])
K = (b1 + b2) / 3.0


def note_dQ(k, a, b, c):
    """Eq. 7 of the note, evaluated directly from the lattice H(k)."""
    H, S, vtb = get_H_v(system, k, order=2)
    ek, psi = diagonalize_hk(H, S, eigenvectors=True)
    vm = {d: psi.conj().T @ vtb[d] @ psi for d in 'xy'}
    wm = {d1 + d2: psi.conj().T @ vtb[d1 + d2] @ psi
          for d1 in 'xy' for d2 in 'xy'}
    iv, ic = 0, 1                              # valence, conduction
    wt = ek[ic] - ek[iv]                       # w~ = w_cv = 2 eps

    def r(d, n, mm):                           # r^d_nm = -i v^d_nm / w_nm
        return -1j * vm[d][n, mm] / (ek[n] - ek[mm])

    def Q(i, j):
        return r(i, iv, ic) * r(j, ic, iv)

    def D(d):                                  # Delta^a = v^a_cc - v^a_vv
        return (vm[d][ic, ic] - vm[d][iv, iv]).real

    band = -2j / wt**2 * (Q(c, b) * D(a) - Q(a, c) * D(b))
    warp = -1.0 / wt**2 * (wm[c + a][iv, ic] * r(b, ic, iv)
                           + r(a, iv, ic) * wm[c + b][ic, iv])
    return band + warp, -2.0 / wt**2 * (r('y', iv, ic)
                                        * wm['yy'][ic, iv]).real


def code_dQ(k, a, b, c):
    res, _, _tau = _process_kpoint(
        system, k, ['x', 'y'], [a + b], [c],
        eflist=np.array([0.0]), kT=M / 50.0, nef=1, eta=0.0,
        eta_sos=1e-8, wannier_r=True, dQ_occupied_subspace=True,
    )
    return res[a][b][c][0]


print(f"v = {V_KP}, lam = {LAM_KP}, m = {M}    lam*v*m = {LAM_KP*V_KP*M:+.6f}")
print()
print(f"{'|q|':>7} {'chan':>5} {'code dQ':>13} {'Eq.7 dQ':>13} "
      f"{'code/Eq7':>10} | {'Eq.13 dg^yy':>13} {'Eq.14 dg^yy':>13}")
for mag in (0.01, 0.03, 0.1, 0.3):
    q = np.array([mag * 0.6, mag * 0.8, 0.0])
    k = K + q
    eps0 = np.sqrt(M**2 + V_KP**2 * mag**2)
    eq14 = -LAM_KP * V_KP * M / (2 * eps0**4)
    for (a, b, c) in (('y', 'y', 'y'), ('x', 'x', 'y'), ('x', 'y', 'x')):
        cd = code_dQ(k, a, b, c)
        nt, eq13 = note_dQ(k, a, b, c)
        extra = (f" | {eq13:>13.6f} {eq14:>13.6f}" if (a, b, c) == ('y','y','y')
                 else "")
        print(f"{mag:>7.2f} {a+b+c:>5} {cd.real:>13.6f} {nt.real:>13.6f} "
              f"{(cd/nt).real:>10.6f}{extra}")
