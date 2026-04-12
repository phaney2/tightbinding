# Tight-Binding Code — Technical Reference

## Package Structure

```
tightbinding/
├── __init__.py
├── __main__.py              # CLI entry: python -m tightbinding <config.yaml>
├── main.py                  # Pipeline orchestration + I/O helpers
├── config.py                # YAML parsing and validation
├── types.py                 # Core dataclasses
├── lattice.py               # System builder (atoms + neighbors → System)
├── neighbors.py             # KD-tree neighbor search
├── hamiltonian.py           # TB_simple Hamiltonian filler
├── bloch.py                 # Bloch sums: H(k), v(k), reciprocal lattice
├── basis.py                 # Orbital basis projectors (unified 18-dim → active subspace)
├── onsite.py                # On-site Hamiltonian (8×8 sp-spin + 18×18 spd-spin)
├── slater_koster.py         # Slater-Koster hopping rules (sp + spd)
│
├── calc/
│   ├── bands.py             # Band structure along k-path
│   ├── all_ek.py            # Full BZ eigenvalues + DOS
│   ├── nonlinear_optical.py # Second-order nonlinear optical response χ^(2)
│   ├── quantum_metric.py    # Quantum metric tensor + linear response
│   └── delta_Q.py           # DC field-induced quantum geometric tensor change
│
├── wannier.py                   # Wannier90 _hr.dat / _tb.dat parsers + System builders
│
├── nrl/
│   ├── params.py            # NRL .dat parameter file parser
│   └── hamiltonian_nrl.py   # NRL System builder (density-dependent, with overlap)
│
└── io/
    └── xyz.py               # XYZ lattice file parser
```

---

## Pipeline Overview

```
YAML config
    │
    ▼
load_config()          [config.py]
    │
    ▼
build_system()         [lattice.py]
    │  ├─ create atoms from positions or lattice_type
    │  ├─ find_neighbors() → neighbor table + HoppingMatrix list
    │  ├─ build AtomPos displacement matrices
    │  └─ assemble OnsiteParams / HoppingParams
    │
    ▼
fill_hamiltonian()     [hamiltonian.py]
    │  ├─ on-site: build_onsite_18x18 → project → matrices[0].H
    │  └─ hopping: build_hopping_9x9 → spin_double → project → matrices[idx].H
    │
    ▼
_dispatch()            [main.py]
    │
    ├─ band_structure  → compute_band_structure()  [calc/bands.py]
    ├─ all_ek          → compute_all_ek()          [calc/all_ek.py]
    ├─ nonlinear_optical → compute_nonlinear_optical() [calc/nonlinear_optical.py]
    └─ quantum_metric  → compute_quantum_metric()  [calc/quantum_metric.py]
```

The NRL path bypasses `lattice.py` and `hamiltonian.py`, instead using
`nrl/hamiltonian_nrl.py:build_nrl_system()` to construct the System directly.

The Wannier path similarly bypasses `lattice.py` and `hamiltonian.py`, using
`wannier.py:build_system_from_hr()` or `build_system_from_tb()` to construct
the System from Wannier90 output files.

---

## Core Data Structures (`types.py`)

### `System`

The central object passed to all calculation engines.

```python
@dataclass
class System:
    atoms: list[Atom]                        # unit cell atoms
    matrices: list[HoppingMatrix]            # H/S blocks indexed by displacement
    unitcell_vectors: NDArray                 # (3,3), rows = a1, a2, a3
    norbs: int                               # total orbital count
    atompos: AtomPos | None                  # intra-cell phase matrices
    neighbors: list[NeighborEntry] | None    # neighbor table
    onsite_params: dict[str, OnsiteParams] | None   # keyed by species
    hopping_params: dict[str, HoppingParams] | None # keyed by species
    orbital_position: dict[str, NDArray] | None     # a^(W,a) matrices
    hopping_anisotropy_direction: NDArray | None     # strain axis unit vector
    hopping_anisotropy_factor: float                 # δ for traceless anisotropy
```

### `HoppingMatrix`

One H/S block for a specific lattice translation R.

```python
@dataclass
class HoppingMatrix:
    displacement: NDArray    # lattice vector R, shape (3,)
    H: NDArray               # (norbs, norbs) complex Hamiltonian block
    S: NDArray               # (norbs, norbs) complex overlap block
```

- `matrices[0]` is always the on-site block (R = [0,0,0]) with `S = I`.
- Off-site blocks have `S = 0` for TB_simple; the NRL path fills S with overlap integrals.

### `Atom`

```python
@dataclass
class Atom:
    index: int               # position in atom list
    coord: NDArray           # Cartesian position (3,)
    basis: str               # e.g. 'sp_u'
    norb: int                # number of active orbitals
    orb_slice: slice         # slice into H/S matrices
    species: str = ''        # atom type (for multi-species)
```

### `NeighborEntry`

```python
@dataclass
class NeighborEntry:
    site_i: int              # source UC atom
    site_j: int              # target UC atom
    distance: float          # |r_i - (r_j + R)|
    direction: NDArray       # unit vector from i toward j+R, shape (3,)
    matrix_idx: int          # index into system.matrices
```

### `AtomPos`

Intra-cell displacement matrices for Bloch phase factors.

```python
@dataclass
class AtomPos:
    x: NDArray    # shape (norbs, norbs): -(r_i - r_j)_x for orbital pair (i,j)
    y: NDArray
    z: NDArray
```

These enter the Bloch sum as `exp(i * k · (R + atompos))`.

### `KPath`

```python
@dataclass
class KPath:
    points: list[NDArray]        # high-symmetry k-points (fractional reciprocal)
    labels: list[str]            # labels for each point
    npoints_per_segment: int     # interpolation density
```

---

## Module Reference

### `config.py`

#### `load_config(path: str) -> dict`

Reads and validates a YAML configuration file. Performs array conversions
(lists → numpy arrays) for lattice vectors, k-points, energy lists, etc.
Raises `ValueError` for missing required keys.

**Validation rules:**
- `system.lattice_vectors` → `np.ndarray` shape (3,3)
- `system.positions[*].coord` → `np.ndarray` shape (3,)
- `calc.nk` → Python list
- `calc.eflist`, `calc.omega1list` → `np.ndarray`
- `calc.kpath.points.*` → `np.ndarray` shape (3,)

---

### `lattice.py`

#### `build_system(cfg: dict) -> System`

Constructs a complete `System` from a validated config dict.

**Pipeline:**
1. Parse atom positions — either from explicit `positions` list or `lattice_type` shorthand
   - Each atom can specify its own `basis` and `species` (per-atom basis overrides system default)
2. Call `find_neighbors()` → `NeighborEntry` list + `HoppingMatrix` list
3. Build `AtomPos` displacement matrices from atom coordinates
4. Build `OnsiteParams` dict (per-species or flat) via `_build_onsite_params()`
5. Build `HoppingParams` dict (per-species or flat) via `_build_hopping_params()`
6. Parse hopping anisotropy (if configured)
7. Return assembled `System`

For `coord_type: 'fractional'`, coordinates are converted via `r = f @ lattice_vectors`.

**Per-species parameter resolution:** Both onsite and hopping configs support two formats:
- Flat: all species share the same parameters
- Per-species: sub-dicts keyed by species name (e.g., `Mo:`, `S:`)

In `fill_hamiltonian`, inter-species hopping parameters are averaged: `t_ij = (t_i + t_j) / 2`.

---

### `neighbors.py`

#### `find_neighbors(coords, lattice_vectors, hopping_range, norbs_total) -> (neighbors, matrices)`

Finds all atom pairs within `hopping_range` across periodic images.

**Algorithm:**
1. Compute the perpendicular height of the unit cell along each lattice direction
2. Determine the number of periodic images needed: `maxn[i] = ceil(range / height_i) + 1`
3. Generate all translation vectors R within the image shell
4. Build a `scipy.spatial.cKDTree` from all image atom positions
5. Query neighbors for each unit-cell atom
6. Register unique displacement vectors as new `HoppingMatrix` entries
7. Pre-register `R = [0,0,0]` as `matrices[0]` with `S = identity`

**Returns:** `(list[NeighborEntry], list[HoppingMatrix])`

---

### `hamiltonian.py`

#### `fill_hamiltonian(system: System) -> None`

Fills all H and S blocks in `system.matrices` in-place.

Always works in the unified 18-dim (spd×spin) full space. Each atom's basis type selects a subspace via projection.

**For each atom i:**
1. Build the 18×18 on-site Hamiltonian via `build_onsite_18x18_from_params()`
   - Supports per-orbital crystal field energies (u_pz, u_dz2, u_dxz)
2. Project to active basis: `H_active = P_i^H @ H_18x18 @ P_i`
3. Add to `matrices[0].H[si, si]` (on-site block)

**For each neighbor pair (i → j):**
1. Compute direction vector `d = -nb.direction * nb.distance` (MATLAB sign convention)
2. Average hopping parameters between species i and j
3. Build 9×9 orbital hopping block via `build_hopping_9x9(d, ...)`
4. Spin-double to 18×18
5. Apply hopping anisotropy factor if enabled
6. Project to active basis: `H_active = P_i^H @ H_18x18 @ P_j`
7. Accumulate into `matrices[nb.matrix_idx].H[si, sj]`

**Important:** The direction convention `d = -nb.direction * nb.distance` points from the neighbor back toward the home atom. This matches the MATLAB code and is critical for correct Slater-Koster signs.

---

### `bloch.py`

#### `get_reciprocal_lattice(unitcell_vectors) -> (b1, b2, b3)`

Computes reciprocal lattice vectors using the standard formula:

```
b1 = 2π (a2 × a3) / V
b2 = 2π (a3 × a1) / V
b3 = 2π (a1 × a2) / V
V  = |a1 · (a2 × a3)|
```

#### `diagonalize_hk(H, S, eigenvectors=False) -> ek [, psi]`

Diagonalizes the Bloch Hamiltonian H(k), handling the generalized eigenvalue
problem H·ψ = E·S·ψ when S ≠ I.

- Auto-detects if S is the identity matrix (`np.allclose(S, I)`)
- Uses `np.linalg.eigh(H)` for standard eigenvalue problems
- Uses `scipy.linalg.eigh(H, S)` for generalized problems
- Returns sorted eigenvalues (and eigenvectors if requested)

#### `get_H_k(system, k) -> (H, S)`

Computes H(k) and S(k) via Bloch phase sums:

```
H(k) = Σ_R H_R · exp(i · k · (R + atompos))
S(k) = Σ_R S_R · exp(i · k · (R + atompos))
```

Both matrices are Hermitianized: `H = (H + H†) / 2`.

#### `get_H_v(system, k, order=1) -> (H, S, v)`

Computes H(k), S(k), and velocity operators v_a = dH/dk_a:

```
v_a(k) = Σ_R i·(R_a + atompos_a) · H_R · exp(i · k · (R + atompos))
```

- `order=1`: `v` has keys `'x'`, `'y'`, `'z'`
- `order=2`: additionally has `'xx'`, `'xy'`, ..., `'zz'` (second derivatives)

All returned matrices are Hermitianized.

---

### `basis.py`

Manages projection between the unified 18-dimensional full space and the active orbital subspace.

Full space (FULL_DIM = 18):
```
[s↑, px↑, py↑, pz↑, dxy↑, dyz↑, dzx↑, dx²-y²↑, dz²↑,
 s↓, px↓, py↓, pz↓, dxy↓, dyz↓, dzx↓, dx²-y²↓, dz²↓]
```

All basis types — including pure-s, pure-p, sp, d-only, pd, spd, and angular momentum (j-basis) — are defined as subsets of this single 18-dim space. There is no separate 8-dim code path.

**Available basis types:**

| Basis | Orbitals | Dim | Description |
|-------|----------|-----|-------------|
| `s_u` / `s_d` / `s_ud` | s | 1/1/2 | s orbital(s) |
| `pxpy_u` / `pxpz_u` / `p_u` | p subsets | 2/2/3 | p orbital subsets |
| `pxpy_d` / `pxpy_ud` | px,py | 2/4 | in-plane p |
| `spxpy_u` / `sp_u` / `sp_ud` | s+p | 3/4/8 | s+p combinations |
| `sj_u` / `sj_d` / `sj_ud` | s+j | 2/2/4 | angular momentum eigenstates |
| `d_u` | dxy,dyz,dzx,dx²-y²,dz² | 5 | all d orbitals, spin-up |
| `d_z_even_u` | dz²,dx²-y²,dxy | 3 | z-even d orbitals, spin-up |
| `pd_u` | px,py,pz,dz²,dx²-y²,dxy | 6 | p + z-even d, spin-up |
| `spd_u` / `spd_ud` | s+p+d | 9/18 | full spd |

#### `get_projector(basis: str) -> NDArray`

Returns the (18, norb) projection matrix P for the named basis. Results are cached.

#### `get_norbs(basis: str) -> int`

Returns the number of active orbitals for a basis type.

#### `project_matrix(H_full, proj_i, proj_j) -> NDArray`

Projects an 18×18 matrix to the active subspace: `H_active = P_i^H @ H_full @ P_j`.

---

### `onsite.py`

Provides two on-site Hamiltonian builders:

#### `build_onsite_8x8(u_s, u_p, delta_s, delta_p, theta, phi, spinorbit) -> NDArray`

Constructs the 8×8 (sp×spin) on-site Hamiltonian. Used by NRL path and legacy sp-only models.

#### `build_onsite_18x18_from_params(params: OnsiteParams) -> NDArray`

Constructs the 18×18 (spd×spin) on-site Hamiltonian. This is the primary builder used by `fill_hamiltonian`. Supports per-orbital crystal field splitting:
- `u_pz`: pz on-site energy (defaults to `u_p` when None)
- `u_dz2`: dz² on-site energy (defaults to `u_d` when None)
- `u_dxz`: dxz,dyz on-site energy (defaults to `u_d` when None)

Three contributions:

**1. Orbital energies** — diagonal:
```
H_orb = diag([u_s, u_p, u_p, u_pz, u_d, u_dxz, u_dxz, u_d, u_dz2]) ⊗ I_spin
```

**2. Exchange splitting** — Zeeman-like term along magnetization direction (θ, φ):
```
H_ex = (δ/2) · [σ_z·cos(θ) + σ_x·sin(θ)·cos(φ) + σ_y·sin(θ)·sin(φ)]
```
Applied separately for s, p, and d channels (with delta_s, delta_p, delta_d).

**3. Spin-orbit coupling** — L·S in both p and d subspaces:
- p-orbital SOC (parameter `spinorbit`): standard L·S with coupling constant λ
- d-orbital SOC (parameter `spinorbit_d`): L·S in real spherical harmonic basis (Condon-Shortley convention)

---

### `slater_koster.py`

#### TB_simple functions (sp space)

| Function | Returns | Description |
|----------|---------|-------------|
| `ss_hopping(tss)` | (1,1) | s-s sigma hopping |
| `sp_hopping(d, tsp, tsp_rashba)` | (1,3), (3,1) | s-p with optional Rashba |
| `pp_hopping(d, tpp_sigma, tpp_pi, tpp_rashba)` | (3,3) | p-p with direction cosines |
| `build_hopping_4x4(d, ...)` | (4,4) | Full sp hopping block |
| `spin_double(H)` | (2n,2n) | Block-diagonal `[[H,0],[0,H]]` |

**Direction cosines:** For a bond direction `d = (l, m, n)` (unit vector):
- `pp_sigma: H[a,b] = l_a·l_b·(tσ - tπ) + δ_ab·tπ`
- `sp_sigma: H[0,a] = l_a·tsp` (odd parity: ps has opposite sign)

#### NRL functions (spd space)

Orbital order: `s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2`

| Function | Returns | Description |
|----------|---------|-------------|
| `sd_hopping(d, tvsd_s)` | (1,5), (5,1) | s-d sigma (even parity) |
| `pd_hopping(d, tvpd_s, tvpd_p)` | (3,5), (5,3) | p-d sigma+pi (odd parity) |
| `dd_hopping(d, tvdd_s, tvdd_p, tvdd_d)` | (5,5) | d-d σ+π+δ |
| `build_hopping_9x9(d, ...)` | (9,9) | Full spd hopping block |

---

### `main.py`

#### `main(config_path: str) -> dict`

Top-level entry point. Runs the full pipeline:
`load_config → build_system → fill_hamiltonian → _dispatch → return result`

#### `_dispatch(system, cfg, calc_type)`

Routes to calculation engines based on `calc.type`:

| `calc_type` | Engine | Module |
|-------------|--------|--------|
| `band_structure` | `compute_band_structure` | `calc.bands` |
| `all_ek` | `compute_all_ek` | `calc.all_ek` |
| `nonlinear_optical` | `compute_nonlinear_optical` | `calc.nonlinear_optical` |
| `quantum_metric` | `compute_quantum_metric` | `calc.quantum_metric` |
| `kubo` | Not yet implemented | — |

#### I/O Helpers

**`_save_npz(output_file, flat, cfg)`** — Saves a flat dict + embedded config JSON to `.npz`.

**`_load_npz(path)`** — Loads `.npz` and extracts config. Returns `(data, cfg)`.

**`load_nonlinear_optical(path)`** — Returns `(result_dict, config_dict)`.
Result structure: `result[chi_name][a][b][c]` → `array(nef, nomega)`.

**`load_quantum_metric(path)`** — Returns `(result_dict, config_dict)`.
Result keys: `'Q'`, `'dQ'`, `'dQf'`, nested by direction labels.

**`load_all_ek(path)`** — Returns `(result_dict, config_dict)`.
Result keys: `'ndim'`, `'ekset'`, `'k_grid'`, `'dos_energies'`, `'dos_values'`,
`'band_summary'`, `'band_gap'`.

---

### `wannier.py`

Parses Wannier90 output files and builds `System` objects directly (bypassing
`lattice.py` and `hamiltonian.py`).

#### `read_hr(path) -> (norbs, displacements, degeneracies, H_matrices)`

Parses a `*_hr.dat` file. Returns the number of Wannier orbitals, R-vectors
(lattice coordinates), degeneracy weights, and complex Hamiltonian blocks.

#### `read_tb(path) -> (lattice_vectors, norbs, displacements, degeneracies, H_matrices, r_matrices)`

Parses a `*_tb.dat` file. Returns everything from `read_hr` plus lattice
vectors (3x3 array) and position operator matrices. Each entry in `r_matrices`
is a list `[r_x, r_y, r_z]` of `(norbs, norbs)` complex arrays containing
`<0n|r_a|Rm>`.

**`_tb.dat` format:**
```
Line 1:       comment (date)
Lines 2-4:    lattice vectors a1, a2, a3 (3 floats each)
Line 5:       norbs (number of Wannier functions)
Line 6:       nrpts (number of R-vectors)
Next lines:   degeneracy weights (15 per line)
H blocks:     for each R: blank line, R-vector, norbs² lines of (m, n, Re H, Im H)
r blocks:     for each R: blank line, R-vector, norbs² lines of (m, n, Re rx, Im rx, Re ry, Im ry, Re rz, Im rz)
```

#### `read_centres(path) -> NDArray`

Parses a `*_centres.xyz` file. Returns `(nwann, 3)` array of Wannier function
centres in Cartesian coordinates.

#### `build_system_from_hr(hr_path, unitcell_vectors, centres_path=None) -> System`

Builds a `System` from a `_hr.dat` file. Lattice vectors must be provided
externally (in the YAML config). Hamiltonian blocks are divided by degeneracy
weights and displacement vectors are converted to Cartesian.

#### `build_system_from_tb(tb_path, centres_path=None) -> System`

Builds a `System` from a `_tb.dat` file. Lattice vectors are read from the
file. In addition to the Hamiltonian, parses and stores the position operator
matrices as `system.wannier_r_matrices` (list of `[r_x, r_y, r_z]` per
R-vector, degeneracy-divided) and `system.wannier_r_displacements`
(lattice-coordinate R-vectors).

**Common to both builders:**
- All Wannier orbitals are grouped under a single dummy `Atom` with `basis='wannier'`
- Overlap matrices are identity (on-site) or zero (off-site) — orthogonal Wannier basis
- `AtomPos` is built from Wannier centres if a `_centres.xyz` file is provided

---

## Calculation Engine Details

### `calc/bands.py`

#### `compute_band_structure(system, kpath) -> dict`

1. Convert fractional k-path points to Cartesian via reciprocal lattice
2. Interpolate between high-symmetry points (`npoints_per_segment` per segment)
3. Diagonalize at each k-point
4. Return `{'k_distances', 'energies', 'tick_positions', 'tick_labels'}`

#### `plot_bands(result, ax=None) -> Axes`

Plots E(k) with vertical lines and labels at high-symmetry points.

---

### `calc/all_ek.py`

#### `detect_dimensionality(system) -> (ndim, active_indices)`

Determines 1D/2D/3D by checking which lattice directions carry nonzero hopping.
For each `HoppingMatrix`, computes the fractional displacement `f = A^{-1} · R`
and marks direction `i` as active if `|f_i| > 0.1`.

Falls back to lattice vector norms if no off-site hoppings are found.

#### `compute_all_ek(system, cfg) -> dict`

1. Auto-detect dimensionality
2. Build uniform k-grid centered on BZ: `tk = -b/2 + db·kc` for each active direction
3. Diagonalize at all k-points (parallelized with `multiprocessing.Pool`)
4. Reshape eigenvalues to grid shape
5. Compute DOS histogram
6. Compute per-band summary (min, max, bandwidth)
7. Detect band gap from `eflist` if provided

#### `plot_all_ek(result, save_path=None)`

- **1D**: side-by-side E(k) line plot + DOS bar chart
- **2D**: side-by-side 3D surface plot + DOS bar chart
- **3D**: DOS bar chart only
- Annotates VBM/CBM lines if band gap is detected

---

### `calc/nonlinear_optical.py`

#### `compute_nonlinear_optical(system, cfg) -> dict`

Computes χ^(2)_abc(ω1, ω2) using density-matrix perturbation theory.

**At each k-point:**
1. `get_H_v(order=2)` → H, S, first and second velocity operators
2. Diagonalize → eigenvalues `ek`, eigenvectors `ψ`
3. Build velocity matrix in eigenbasis: `v_nm = ψ†·vtb·ψ`
4. Build position operator: `r_nm = -i·v_nm / (E_n - E_m)` for n ≠ m
5. Build generalized derivative `dk_r` via `_compute_dk_rmtx`
6. Compute 14 chi components for each (ef, ω1) pair

**14 chi components:**
`chi_ii`, `chi_ee1`, `chi_ee2`, `chi_ei1`, `chi_ei2`,
`chi_eit1`, `chi_eit2`, `chi_eit3`, `chi_ie1`, `chi_ie2`,
`chi_e1`, `chi_e2`, `chi_i1`, `chi_i2`

Each component has shape `(nef, nomega)`.

**k-grid:** BZ-centered: `tk = -b1/2 - b2/2 + db1·kc1 + db2·kc2`

---

### `calc/quantum_metric.py`

#### `compute_quantum_metric(system, cfg) -> dict`

**At each k-point:**
1. `get_H_v(order=1)` → H, S, velocity operators
2. Diagonalize → `ek`, `ψ`
3. Velocity in eigenbasis: `v_nm = ψ†·vtb·ψ`
4. Build perturbed eigenstates via finite-difference:
   ```
   pert = i·δ·v_nm / [ΔE_nm · (ΔE_nm + i·η)]    (zero for degenerate pairs)
   ψ± = ψ ± ψ·pert
   ```
5. Compute perturbed velocity matrices: `v±[d1][d3] = ψ±[d3]†·vtb[d1]·ψ±[d3]`
6. Loop over Fermi energies:
   - **Q[d1][d2]** = Σ_nm v[d1]_nm · conj(v[d2]_nm) · f_nm / ΔE²_nm
   - **dQ[d1][d2][d3]** = (Q+ - Q-) / (2δ)  — intrinsic, via numerical derivative
   - **dQf[d1][d2][d3]** = Σ_nm v[d1]_nm · conj(v[d2]_nm) · df_nm[d3] / ΔE²_nm
     — extrinsic (Fermi surface)

**Degeneracy handling:** Pairs with `|ΔE| < 1e-5` are masked out (zero contribution).

---

## NRL Path (`nrl/`)

### `nrl/params.py`

#### `parse_dat_file(path) -> dict`

Parses NRL `.dat` parameter files (73 values):
- 1 value: `lambda_` (electron density decay)
- 12 values: on-site polynomial coefficients `[a, b, c, d]` for s, p, d
- 30 values: hopping integrals `[e, f, g]` for 10 bond types (ss_σ, sp_σ, pp_σ, pp_π, sd_σ, pd_σ, pd_π, dd_σ, dd_π, dd_δ)
- 30 values: overlap integrals (same structure)

#### `eval_hopping(params, R, fc) -> float`

```
V(R) = (e + f·R) · exp(-g²·R) · fc
```

#### `cutoff_function(R, Rc=14.0, Lc=0.5) -> float`

Smooth Fermi cutoff: `fc = 1 / (1 + exp((R - Rc) / Lc))`

### `nrl/hamiltonian_nrl.py`

#### `build_nrl_system(...) -> System`

Complete NRL pipeline:
1. Parse `.dat` parameter file
2. Create atoms with `basis='spd'`, 9 orbitals each
3. Find neighbors via KD-tree
4. Fill H and S using density-dependent on-site + Slater-Koster hopping
5. Convert Rydberg → eV (multiply by 13.67)
6. Spin-double all matrices (9×9 → 18×18)
7. Add spin-orbit coupling (p-SOC and d-SOC)
8. Build spin-doubled AtomPos
9. Return System with `norbs = 18 × natoms`

**Density-dependent on-site energies:**
```
ρ_i = Σ_{j≠i} exp(-λ²·R_ij) · fc(R_ij)
e_orb = a + b·ρ^(2/3) + c·ρ^(4/3) + d·ρ²
```

---

## Key Design Decisions

1. **System as central object**: All calculation engines receive the same `System` dataclass. The Hamiltonian is stored as real-space blocks `H_R` indexed by lattice displacement, enabling efficient Bloch sums.

2. **Projection-based basis**: The full 18-dim (spd×spin) space is always used internally. Basis selection works by projecting down to the active subspace, making it easy to add new basis types without changing the Hamiltonian construction. Different atoms in the same unit cell can use different basis types (e.g., `d_u` on a transition metal, `p_u` on a chalcogen).

3. **KD-tree neighbor finding**: Replaces the O(N³) brute-force triple loop with O(N log N) spatial queries via `scipy.spatial.cKDTree`.

4. **Shared Bloch machinery**: All calc engines use the same `get_H_k` / `get_H_v` functions. Reciprocal lattice computation and eigenvalue solving are centralized in `bloch.py`.

5. **Multiprocessing parallelism**: k-point parallelism via `multiprocessing.Pool` with a module-level global initializer pattern. Each worker stores a reference to the `System` to avoid pickling overhead.

6. **Embedded config in output**: Every `.npz` file stores the full config as JSON, ensuring reproducibility.
