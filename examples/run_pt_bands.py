"""Run Pt band structure using NRL tight-binding parameters."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tightbinding.io.xyz import read_xyz
from tightbinding.nrl.hamiltonian_nrl import build_nrl_system
from tightbinding.types import KPath
from tightbinding.calc.bands import compute_band_structure, plot_bands

# Paths
xyz_file = r'C:\Users\haney\master_response\Pt.xyz'
dat_file = r'C:\Users\haney\master_response\pt.dat'

# Read xyz
species, sysdata, lattice_vectors = read_xyz(xyz_file)
coords = sysdata[:, 0:3]

print(f"Species: {species}")
print(f"Coords: {coords}")
print(f"Lattice vectors:\n{lattice_vectors}")

# Build system
hopping_range = 16.0
vso = {'Pt': {'p': 0.05, 'd': 0.5}}  # SOC on (Ryd -> eV)

system = build_nrl_system(species, coords, lattice_vectors, hopping_range,
                          dat_file, vso=vso)

print(f"\nNumber of orbitals (with spin): {system.norbs}")
print(f"Number of hopping matrices: {len(system.matrices)}")
print(f"Number of atoms (with images): {len(system.atoms)}")

# Check H is Hermitian at Gamma
from tightbinding.bloch import get_H_k
H0, S0 = get_H_k(system, np.array([0, 0, 0]))
print(f"\nH(Gamma) Hermitian: {np.allclose(H0, H0.conj().T)}")
print(f"S(Gamma) Hermitian: {np.allclose(S0, S0.conj().T)}")
print(f"S(Gamma) positive definite: {np.all(np.linalg.eigvalsh(S0) > 0)}")

# FCC Pt k-path: Gamma -> X -> W -> L -> Gamma -> K
# Fractional coords in the PRIMITIVE reciprocal lattice basis (not conventional cubic!)
# Computed by inverting B = [b1 b2 b3] and applying to standard Cartesian k-points.
kpath = KPath(
    points=[
        np.array([0.0, 0.0, 0.0]),         # Gamma
        np.array([0.5, 0.5, 0.0]),         # X = (2pi/a)(0,1,0)
        np.array([0.75, 0.25, 0.25]),      # W = (2pi/a)(1/2,1,0)
        np.array([0.5, 0.0, 0.5]),         # L = (2pi/a)(1/2,1/2,1/2)
        np.array([0.0, 0.0, 0.0]),         # Gamma
        np.array([0.75, 0.0, 0.375]),      # K = (2pi/a)(3/4,3/4,0)
    ],
    labels=['$\\Gamma$', 'X', 'W', 'L', '$\\Gamma$', 'K'],
    npoints_per_segment=150,
)

print("\nComputing band structure...")
result = compute_band_structure(system, kpath)

print(f"Energy range: [{result['energies'].min():.2f}, {result['energies'].max():.2f}] eV")

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
plot_bands(result, ax=ax)
ax.set_ylabel('Energy (eV)')
ax.set_title('Pt Band Structure (NRL-TB, with SOC)')
ax.set_ylim(-10, 10)
plt.tight_layout()
plt.savefig('pt_bands.png', dpi=150)
print("Saved to pt_bands.png")
