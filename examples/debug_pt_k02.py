"""Compare H/S at k=(0, 0.2, 0) Cartesian against MATLAB benchmark."""

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

norbs_nospin = system.norbs // 2  # 9

# MATLAB benchmark at k=(0, 0.2, 0) in Cartesian (1/Bohr)
k_cart = np.array([0.0, 0.2, 0.0])
H_k, S_k = get_H_k(system, k_cart)

# Extract no-spin block
H_ns = H_k[:norbs_nospin, :norbs_nospin]
S_ns = S_k[:norbs_nospin, :norbs_nospin]

# MATLAB d-orbital order: [s, px, py, pz, dz2, dxz, dyz, dx2-y2, dxy]
# Python d-orbital order: [s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2]
# Reorder: Python [4,5,6,7,8] = [dxy, dyz, dzx, dx2-y2, dz2]
#          MATLAB [4,5,6,7,8] = [dz2, dxz, dyz, dx2-y2, dxy]
# Python->MATLAB: [0,1,2,3, 8, 6, 5, 7, 4]
reorder = [0, 1, 2, 3, 8, 6, 5, 7, 4]
H_reord = H_ns[np.ix_(reorder, reorder)]
S_reord = S_ns[np.ix_(reorder, reorder)]

# MATLAB benchmark
H_matlab = np.array([
    [-5.6856+0j, 0+0j, 0-9.2418j, 0-0j, 0+0j, -0+0j, -0+0j, -0.6304+0j, -0.3640+0j],
    [0-0j, 25.3254+0j, 0+0j, 0+0j, 0+2.2986j, 0-0j, 0+0j, 0-0j, 0-0j],
    [0+9.2418j, 0+0j, 20.8628+0j, -0+0j, 0-0j, 0+0j, 0-0j, 0-1.3673j, 0-0.7894j],
    [0+0j, 0+0j, -0+0j, 25.3254+0j, 0-0j, 0+2.2986j, 0-0j, 0+0j, 0+0j],
    [0+0j, 0-2.2986j, 0+0j, 0+0j, 0.5063+0j, 0+0j, -0+0j, 0+0j, 0+0j],
    [-0+0j, 0+0j, 0-0j, 0-2.2986j, 0+0j, 0.5063+0j, -0+0j, 0+0j, -0+0j],
    [-0+0j, 0-0j, 0+0j, 0+0j, -0+0j, -0+0j, -0.5119+0j, 0+0j, 0+0j],
    [-0.6304+0j, 0+0j, 0+1.3673j, 0-0j, 0+0j, 0+0j, 0+0j, 1.6346+0j, -0.1744+0j],
    [-0.3640+0j, 0+0j, 0+0.7894j, 0-0j, 0+0j, -0+0j, 0+0j, -0.1744+0j, 1.8359+0j],
])

S_matlab = np.array([
    [1.3326+0j, 0-0j, 0+0.0425j, 0+0j, -0+0j, 0+0j, -0+0j, 0.1307+0j, 0.0754+0j],
    [0+0j, 1.4775+0j, -0+0j, -0+0j, 0-0.0844j, 0+0j, 0-0j, 0-0j, 0+0j],
    [0-0.0425j, -0+0j, 1.5047+0j, 0+0j, 0+0j, 0-0j, 0+0j, 0+0.0496j, 0+0.0287j],
    [0-0j, -0+0j, 0+0j, 1.4775+0j, 0+0j, 0-0.0844j, 0-0j, 0+0j, 0+0j],
    [-0+0j, 0+0.0844j, 0-0j, 0-0j, 0.9870+0j, 0+0j, -0+0j, 0+0j, -0+0j],
    [0+0j, 0-0j, 0+0j, 0+0.0844j, 0+0j, 0.9870+0j, 0+0j, -0+0j, -0+0j],
    [-0+0j, 0+0j, 0-0j, 0+0j, -0+0j, 0+0j, 1.0433+0j, 0+0j, -0+0j],
    [0.1307+0j, 0+0j, 0-0.0496j, 0-0j, 0+0j, -0+0j, 0+0j, 0.9226+0j, 0.0125+0j],
    [0.0754+0j, 0-0j, 0-0.0287j, 0-0j, -0+0j, -0+0j, -0+0j, 0.0125+0j, 0.9081+0j],
])

print("=== H(k) comparison at k=(0, 0.2, 0) ===")
print("\nPython H (reordered to MATLAB basis) - real part:")
print(np.array2string(H_reord.real, precision=4, suppress_small=True))
print("\nMATLAB H - real part:")
print(np.array2string(H_matlab.real, precision=4, suppress_small=True))
print("\nMax absolute difference H:")
print(f"  {np.max(np.abs(H_reord - H_matlab)):.6f}")

print("\n=== S(k) comparison ===")
print("\nPython S (reordered) - real part:")
print(np.array2string(S_reord.real, precision=4, suppress_small=True))
print("\nMax absolute difference S:")
print(f"  {np.max(np.abs(S_reord - S_matlab)):.6f}")

# Diagonal comparison
print("\nH diagonal comparison (real):")
print(f"  Python:  {np.diag(H_reord).real}")
print(f"  MATLAB:  {np.diag(H_matlab).real}")
print(f"  Diff:    {np.diag(H_reord).real - np.diag(H_matlab).real}")

print("\nS diagonal comparison (real):")
print(f"  Python:  {np.diag(S_reord).real}")
print(f"  MATLAB:  {np.diag(S_matlab).real}")
print(f"  Diff:    {np.diag(S_reord).real - np.diag(S_matlab).real}")
