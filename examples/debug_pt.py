"""Debug Pt band structure: compare H/S at specific k-points against MATLAB benchmark."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tightbinding.io.xyz import read_xyz
from tightbinding.nrl.hamiltonian_nrl import build_nrl_system
from tightbinding.bloch import get_H_k

# Paths
xyz_file = r'C:\Users\haney\master_response\Pt.xyz'
dat_file = r'C:\Users\haney\master_response\pt.dat'

# Read xyz
species, sysdata, lattice_vectors = read_xyz(xyz_file)
coords = sysdata[:, 0:3]

print(f"Lattice vectors (Bohr):\n{lattice_vectors}")
print(f"Coords: {coords}")

# Build system with large hopping range
hopping_range = 24.0
vso = {'Pt': {'p': 0.0, 'd': 0.0}}

system = build_nrl_system(species, coords, lattice_vectors, hopping_range,
                          dat_file, vso=vso)

print(f"\nNumber of orbitals (with spin): {system.norbs}")
print(f"Number of hopping matrices: {len(system.matrices)}")

norbs_nospin = system.norbs // 2  # 9

# ---- Test at Gamma (k=0,0,0) ----
print("\n" + "="*60)
print("Gamma point (k = 0, 0, 0)")
print("="*60)

H_gamma, S_gamma = get_H_k(system, np.array([0.0, 0.0, 0.0]))

# Extract the no-spin (upper-left) block
H_ns = H_gamma[:norbs_nospin, :norbs_nospin]
S_ns = S_gamma[:norbs_nospin, :norbs_nospin]

print("\nPython H(Gamma) [9x9 no-spin block]:")
print(np.array2string(H_ns.real, precision=4, suppress_small=True))

print("\nPython S(Gamma) [9x9 no-spin block]:")
print(np.array2string(S_ns.real, precision=4, suppress_small=True))

# MATLAB benchmark values at Gamma
H_matlab_gamma = np.diag([-9.5128, 27.4629, 27.4629, 27.4629,
                           0.0062, 0.0062, 0.0062, 1.8583, 1.8583])
S_matlab_gamma = np.diag([1.3528, 1.5965, 1.5965, 1.5965,
                           1.0123, 1.0123, 1.0123, 0.9050, 0.9050])

print("\nMATLAB H(Gamma) diagonal:")
print(np.diag(H_matlab_gamma))

print("\nPython H(Gamma) diagonal:")
print(np.diag(H_ns).real)

print("\nDifference H diagonal (Python - MATLAB):")
print(np.diag(H_ns).real - np.diag(H_matlab_gamma))

print("\nMATLAB S(Gamma) diagonal:")
print(np.diag(S_matlab_gamma))

print("\nPython S(Gamma) diagonal:")
print(np.diag(S_ns).real)

print("\nDifference S diagonal (Python - MATLAB):")
print(np.diag(S_ns).real - np.diag(S_matlab_gamma))

# Note: MATLAB orbital ordering is [s, px, py, pz, dz2, dxz, dyz, dx2-y2, dxy]
# Python orbital ordering is [s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2]
# Map: MATLAB d-indices [4,5,6,7,8] = [dz2, dxz, dyz, dx2-y2, dxy]
# Python d-indices [4,5,6,7,8] = [dxy, dyz, dzx, dx2-y2, dz2]
# So reordering Python to MATLAB: [0,1,2,3, 8, 6, 5, 7, 4]

reorder = [0, 1, 2, 3, 8, 6, 5, 7, 4]  # Python -> MATLAB ordering
H_reordered = H_ns[np.ix_(reorder, reorder)]
S_reordered = S_ns[np.ix_(reorder, reorder)]

print("\n" + "="*60)
print("After reordering d-orbitals to match MATLAB [s,px,py,pz,dz2,dxz,dyz,dx2-y2,dxy]")
print("="*60)

print("\nReordered H(Gamma) diagonal:")
print(np.diag(H_reordered).real)

print("\nMATLAB H(Gamma) diagonal:")
print(np.diag(H_matlab_gamma))

print("\nDifference (reordered Python - MATLAB):")
print(np.diag(H_reordered).real - np.diag(H_matlab_gamma))

print("\nReordered S(Gamma) diagonal:")
print(np.diag(S_reordered).real)

print("\nMATLAB S(Gamma) diagonal:")
print(np.diag(S_matlab_gamma))

print("\nDifference S (reordered Python - MATLAB):")
print(np.diag(S_reordered).real - np.diag(S_matlab_gamma))

# ---- Test at k=(0, 0.2, 0) ----
# Need to figure out k-point units. MATLAB uses: kset = 2*pi*[frac] / (a0/2)
# For FCC Pt, a0 = 3.92/.529/2 Bohr (half lattice constant?)
# Let's check what a0 is
a_lat = 3.92 / 0.529  # full lattice constant in Bohr
print(f"\nFCC lattice constant: {a_lat:.4f} Bohr")
a0_matlab = a_lat / 2  # MATLAB uses a0 = 3.92/.529/2
print(f"MATLAB a0 = a/2 = {a0_matlab:.4f} Bohr")

# MATLAB: kset = 2*pi*[0, 0.2, 0] / (a0/2) = 2*pi*[0, 0.2, 0] / (a_lat/4)
# Wait, let me re-read: kset = kset / a0/2, which is kset / (a0/2) = kset * 2/a0
# So k_cart = 2*pi*[0, 0.2, 0] * 2/a0 = 2*pi*[0, 0.2, 0] * 2/(a_lat/2) = 2*pi*[0, 0.2, 0] * 4/a_lat

# Actually, MATLAB says: a0 = 3.92 / .529 / 2
# Then kset = kset / a0 / 2
# Is that (kset / a0) / 2 or kset / (a0 / 2)?
# In MATLAB, / is left-to-right, so kset / a0 / 2 = (kset / a0) / 2
# So k_cart = 2*pi*[0, 0.2, 0] / a0 / 2 = 2*pi*[0, 0.2, 0] / (2*a0)

# But our Bloch sum uses: exp(i * k_frac . R_cart) where k is in fractional reciprocal coords
# and R is in Cartesian... no, let me check our code.

print("\n" + "="*60)
print("Checking Bloch sum convention")
print("="*60)

# Our get_H_k uses: phase = exp(i * k . displacement)
# where k is whatever we pass in, and displacement is Cartesian
# So we need to pass Cartesian k.

# For the benchmark k=(0,0.2,0), MATLAB computes:
# k_cart = 2*pi*[0,0.2,0] / a0 / 2 where a0 = 3.92/0.529/2
a0 = 3.92 / 0.529 / 2
k_matlab = 2 * np.pi * np.array([0, 0.2, 0]) / a0 / 2
print(f"MATLAB k_cart for (0,0.2,0): {k_matlab}")

# But wait - our run_pt_bands.py uses fractional k-points and converts via reciprocal vectors
# Let me check what our code actually does with them

# For now, let's just test with Gamma which is unambiguous
# And separately figure out the k convention

# Check eigenvalues at Gamma
from scipy.linalg import eigh
eigvals_gamma = eigh(H_gamma, S_gamma, eigvals_only=True)
print(f"\nEigenvalues at Gamma (all, spin doubled):")
print(np.sort(eigvals_gamma)[:18])

# With reordered matrices
H_reord_full = np.zeros_like(H_gamma)
S_reord_full = np.zeros_like(S_gamma)
# Reorder both spin blocks
reorder_full = reorder + [r + norbs_nospin for r in reorder]
H_reord_full = H_gamma[np.ix_(reorder_full, reorder_full)]
S_reord_full = S_gamma[np.ix_(reorder_full, reorder_full)]
eigvals_reord = eigh(H_reord_full, S_reord_full, eigvals_only=True)
print(f"\nEigenvalues at Gamma (reordered, should be same):")
print(np.sort(eigvals_reord)[:18])
