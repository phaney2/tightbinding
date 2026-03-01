"""Debug k-path: verify that our fractional k-points produce the correct Cartesian k-points."""

import numpy as np

# FCC lattice vectors (primitive cell) for Pt
lattice_vectors = np.array([
    [3.7051, 3.7051, 0.0],
    [-3.7051, 3.7051, 0.0],
    [3.7051, 0.0, 3.7051],
])

a1, a2, a3 = lattice_vectors
vol = abs(np.dot(a1, np.cross(a2, a3)))
b1 = 2 * np.pi * np.cross(a2, a3) / vol
b2 = 2 * np.pi * np.cross(a3, a1) / vol
b3 = 2 * np.pi * np.cross(a1, a2) / vol

a = 7.4102  # full FCC lattice constant
print(f"a = {a:.4f} Bohr")
print(f"2*pi/a = {2*np.pi/a:.4f}")
print(f"\nb1 = {b1}")
print(f"b2 = {b2}")
print(f"b3 = {b3}")

# Expected Cartesian high-symmetry points (standard FCC):
# Gamma: (0, 0, 0)
# X: (2*pi/a) * (0, 1, 0)  (or permutations)
# W: (2*pi/a) * (1/2, 1, 0)
# L: (2*pi/a) * (1/2, 1/2, 1/2)
# K: (2*pi/a) * (3/4, 3/4, 0)

print("\n=== Expected Cartesian k-points ===")
expected = {
    'Gamma': np.array([0, 0, 0]) * 2*np.pi/a,
    'X': np.array([0, 1, 0]) * 2*np.pi/a,
    'W': np.array([0.5, 1, 0]) * 2*np.pi/a,
    'L': np.array([0.5, 0.5, 0.5]) * 2*np.pi/a,
    'K': np.array([0.75, 0.75, 0]) * 2*np.pi/a,
}
for name, k in expected.items():
    print(f"  {name}: {k}")

# What our WRONG fractional coordinates produce:
print("\n=== WRONG fractional coords (current code) ===")
wrong_frac = {
    'Gamma': np.array([0, 0, 0]),
    'X': np.array([0, 1, 0]),
    'W': np.array([0.5, 1, 0]),
    'L': np.array([0.5, 0.5, 0.5]),
    'K': np.array([0.75, 0.75, 0]),
}
for name, kf in wrong_frac.items():
    k_cart = kf[0]*b1 + kf[1]*b2 + kf[2]*b3
    print(f"  {name}: frac={kf} -> cart={k_cart}")

# Correct fractional coordinates for primitive FCC reciprocal lattice:
# Solve: k_cart = c1*b1 + c2*b2 + c3*b3
# b1 = (2pi/a)[1, 1, -1], b2 = (2pi/a)[-1, 1, 1], b3 = (2pi/a)[1, -1, 1]
# For X = (2pi/a)(0, 1, 0): c1=1/2, c2=1/2, c3=0
# For W = (2pi/a)(1/2, 1, 0): solve -> c1=3/4, c2=1/2, c3=1/4
# For L = (2pi/a)(1/2, 1/2, 1/2): c1=1/2, c2=1/2, c3=1/2
# For K = (2pi/a)(3/4, 3/4, 0): c1=3/4, c2=3/8, c3=3/8

# Let me solve this properly using the inverse of the reciprocal lattice matrix
B = np.array([b1, b2, b3]).T  # columns are reciprocal vectors
B_inv = np.linalg.inv(B)

print("\n=== CORRECT fractional coords ===")
for name, k_cart in expected.items():
    kf = B_inv @ k_cart
    k_check = kf[0]*b1 + kf[1]*b2 + kf[2]*b3
    print(f"  {name}: cart={k_cart} -> frac={kf}  (check: {k_check})")
