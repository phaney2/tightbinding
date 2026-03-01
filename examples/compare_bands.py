"""Side-by-side comparison of Python vs MATLAB Pt band structure."""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tightbinding.io.xyz import read_xyz
from tightbinding.nrl.hamiltonian_nrl import build_nrl_system
from tightbinding.types import KPath
from tightbinding.calc.bands import compute_band_structure, plot_bands

xyz_file = r'C:\Users\haney\master_response\Pt.xyz'
dat_file = r'C:\Users\haney\master_response\pt.dat'

species, sysdata, lattice_vectors = read_xyz(xyz_file)
coords = sysdata[:, 0:3]

system = build_nrl_system(species, coords, lattice_vectors, 24.0,
                          dat_file, vso={'Pt': {'p': 0.0, 'd': 0.0}})

# Match MATLAB k-path: Gamma -> X -> W -> L -> Gamma (no K)
kpath = KPath(
    points=[
        np.array([0.0, 0.0, 0.0]),         # Gamma
        np.array([0.5, 0.5, 0.0]),         # X
        np.array([0.75, 0.25, 0.25]),      # W
        np.array([0.5, 0.0, 0.5]),         # L
        np.array([0.0, 0.0, 0.0]),         # Gamma
    ],
    labels=['$\\Gamma$', 'X', 'W', 'L', '$\\Gamma$'],
    npoints_per_segment=100,
)

print("Computing band structure...")
result = compute_band_structure(system, kpath)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Python result
ax = axes[0]
plot_bands(result, ax=ax)
ax.set_ylabel('Energy (eV)')
ax.set_title('Python (NRL-TB, no SOC)')
ax.set_ylim(-10, 10)

# MATLAB benchmark
ax2 = axes[1]
matlab_img = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'NRL_bandStruct_MATLAB', 'bandstruct.png'))
ax2.imshow(matlab_img)
ax2.set_title('MATLAB Benchmark')
ax2.axis('off')

plt.tight_layout()
plt.savefig('pt_bands_comparison.png', dpi=150)
print("Saved to pt_bands_comparison.png")

# Also save just the Python result with matching y-range
fig2, ax3 = plt.subplots(figsize=(8, 6))
plot_bands(result, ax=ax3)
ax3.set_ylabel('Energy (eV)')
ax3.set_title('Pt Band Structure (Python NRL-TB, no SOC)')
ax3.set_ylim(-10, 10)
plt.tight_layout()
plt.savefig('pt_bands_matched.png', dpi=150)
print("Saved to pt_bands_matched.png")

# Print eigenvalues at high-symmetry points for quantitative comparison
print("\nEigenvalues at high-symmetry points (eV):")
from tightbinding.bloch import get_H_k
from scipy.linalg import eigh

# Reciprocal lattice vectors
a1, a2, a3 = lattice_vectors
vol = abs(np.dot(a1, np.cross(a2, a3)))
b1 = 2 * np.pi * np.cross(a2, a3) / vol
b2 = 2 * np.pi * np.cross(a3, a1) / vol
b3 = 2 * np.pi * np.cross(a1, a2) / vol

sym_points = {
    'Gamma': np.array([0.0, 0.0, 0.0]),
    'X': 0.5*b1 + 0.5*b2,
    'W': 0.75*b1 + 0.25*b2 + 0.25*b3,
    'L': 0.5*b1 + 0.5*b3,
}

for name, k in sym_points.items():
    H, S = get_H_k(system, k)
    eigvals = np.sort(eigh(H, S, eigvals_only=True))
    # Only show unique (non-spin-doubled)
    unique = eigvals[::2]  # every other (spin degenerate)
    print(f"  {name}: {np.array2string(unique, precision=2, suppress_small=True)}")
