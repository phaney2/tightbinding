"""Figure out the exact d-orbital mapping between Python and MATLAB."""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tightbinding.io.xyz import read_xyz
from tightbinding.nrl.hamiltonian_nrl import build_nrl_system
from tightbinding.bloch import get_H_k

xyz_file = r'C:\Users\haney\master_response\Pt.xyz'
dat_file = r'C:\Users\haney\master_response\pt.dat'

species, sysdata, lattice_vectors = read_xyz(xyz_file)
coords = sysdata[:, 0:3]

system = build_nrl_system(species, coords, lattice_vectors, 24.0,
                          dat_file, vso={'Pt': {'p': 0.0, 'd': 0.0}})

n = system.norbs // 2  # 9

# k = (0, 0.2, 0) Cartesian
k_cart = np.array([0.0, 0.2, 0.0])
H_k, S_k = get_H_k(system, k_cart)
H_py = H_k[:n, :n]

# Python ordering: [s=0, px=1, py=2, pz=3, dxy=4, dyz=5, dzx=6, dx2y2=7, dz2=8]

print("Python H(k) at k=(0, 0.2, 0) - full 9x9 no-spin block:")
print("Real part:")
for i in range(n):
    row = [f"{H_py[i,j].real:8.4f}" for j in range(n)]
    print(" ".join(row))

print("\nImag part:")
for i in range(n):
    row = [f"{H_py[i,j].imag:8.4f}" for j in range(n)]
    print(" ".join(row))

print("\nPython orbital labels: [s, px, py, pz, dxy, dyz, dzx, dx2y2, dz2]")
print("Python diag:", [f"{H_py[i,i].real:.4f}" for i in range(n)])

# Key off-diagonal elements that identify orbital coupling:
print("\nKey off-diagonal elements (Python):")
for i in range(n):
    for j in range(i+1, n):
        val = H_py[i, j]
        if abs(val) > 0.01:
            labels = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2y2', 'dz2']
            print(f"  H[{labels[i]},{labels[j]}] = {val:.4f}")

print("\n\nMATLAB H(k) at k=(0, 0.2, 0):")
print("MATLAB orbital labels: [s, px, py, pz, d4, d5, d6, d7, d8]")
# MATLAB diagonal
m_diag = [-5.6856, 25.3254, 20.8628, 25.3254, 0.5063, 0.5063, -0.5119, 1.6346, 1.8359]
print("MATLAB diag:", [f"{v:.4f}" for v in m_diag])

print("\nKey off-diagonal elements (MATLAB):")
print("  H[s, py] = -9.2418i")
print("  H[px, d4] = 2.2986i")
print("  H[pz, d5] = 2.2986i")
print("  H[s, d7] = -0.6304")
print("  H[s, d8] = -0.3640")
print("  H[py, d7] = -1.3673i")
print("  H[py, d8] = -0.7894i")
print("  H[d7, d8] = -0.1744")

# Now match:
# Python s-py = H[0,2] imag
print(f"\n--- Matching ---")
print(f"s-py:  Python={H_py[0,2]:.4f}, MATLAB=-9.2418i")
print(f"s-dxy: Python={H_py[0,4]:.4f}")
print(f"s-dyz: Python={H_py[0,5]:.4f}")
print(f"s-dzx: Python={H_py[0,6]:.4f}")
print(f"s-dx2y2: Python={H_py[0,7]:.4f}")
print(f"s-dz2: Python={H_py[0,8]:.4f}")
print(f"px-dxy: Python={H_py[1,4]:.4f}")
print(f"px-dyz: Python={H_py[1,5]:.4f}")
print(f"px-dzx: Python={H_py[1,6]:.4f}")
print(f"pz-dxy: Python={H_py[3,4]:.4f}")
print(f"pz-dyz: Python={H_py[3,5]:.4f}")
print(f"pz-dzx: Python={H_py[3,6]:.4f}")
