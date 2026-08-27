"""Verify that the lattice H(k) reduces to Eq. 1 of the warping note near K.

Compares the lattice off-diagonal element H_12(K + q) against the k.p form
    H_12 = v * conj(z) + lam * z^2,     z = tau*qx + i*qy
with v = 3*T*a/2, lam = -3*T*a^2/8, up to the single global gauge phase
e^{i*gamma} (a sublattice phase rotation, which is unobservable).
"""
import numpy as np

from tightbinding.config import load_config
from tightbinding.lattice import build_system
from tightbinding.hamiltonian import fill_hamiltonian
from tightbinding.bloch import get_H_k, get_reciprocal_lattice

cfg = load_config('examples/honeycomb_warp/bands.yaml')
system = build_system(cfg)
fill_hamiltonian(system)

a = 1.0
T = cfg['hopping']['tss_sigma']
m = 0.5 * (cfg['onsite']['A']['u_s'] - cfg['onsite']['B']['u_s'])
v = 1.5 * T * a
lam = -0.375 * T * a**2

b1, b2, _ = get_reciprocal_lattice(cfg['system']['lattice_vectors'])
Kpt = {+1: (b1 + b2) / 3.0, -1: -(b1 + b2) / 3.0}

print(f"params: a={a}  T={T}  m={m}   =>  v={v}  lam={lam}  lam/v={lam/v}")
print(f"K = {Kpt[+1]}   |K| = {np.linalg.norm(Kpt[+1]):.6f}  "
      f"(4pi/(3sqrt3) = {4*np.pi/(3*np.sqrt(3)):.6f})")
print()

rng = np.random.default_rng(0)
for tau in (+1, -1):
    # Calibrate the global gauge phase on one small-q point, then test others.
    q0 = np.array([1e-4, 0.0, 0.0])
    H0, _ = get_H_k(system, Kpt[tau] + q0)
    z0 = tau * q0[0] + 1j * q0[1]
    gamma = np.angle(H0[0, 1] / (v * np.conj(z0)))

    print(f"tau = {tau:+d}   gauge phase gamma = {gamma:+.6f} rad")
    print(f"  {'|q|':>8}  {'H12 (lattice)':>28}  {'H12 (k.p, Eq.1)':>28}  {'rel err':>10}")
    for mag in (0.002, 0.01, 0.05, 0.1, 0.2):
        errs = []
        for _ in range(6):
            ang = rng.uniform(0, 2 * np.pi)
            q = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0])
            H, _ = get_H_k(system, Kpt[tau] + q)
            lat = H[0, 1] * np.exp(-1j * gamma)
            z = tau * q[0] + 1j * q[1]
            kp = v * np.conj(z) + lam * z**2
            errs.append((lat, kp, abs(lat - kp) / abs(kp)))
        lat, kp, _ = errs[0]
        worst = max(e[2] for e in errs)
        print(f"  {mag:8.3f}  {lat.real:+13.8f}{lat.imag:+13.8f}j  "
              f"{kp.real:+13.8f}{kp.imag:+13.8f}j  {worst:10.2e}")
    # diagonal should be exactly +-m, k-independent
    H, _ = get_H_k(system, Kpt[tau] + np.array([0.3, -0.2, 0.0]))
    print(f"  diagonal at large q: {H[0,0].real:+.10f}, {H[1,1].real:+.10f} "
          f"(expect {m:+.10f}, {-m:+.10f})")
    print()
