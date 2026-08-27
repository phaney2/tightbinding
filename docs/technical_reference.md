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
│   ├── nonlinear_optical_fast.py # Vectorized χ^(2) kernel (the one actually called)
│   ├── freq_integral.py     # Closed-form ∫dω ω^(-p)(...) for integrated χ^(2)
│   ├── wannier_gauge.py     # Wannier-gauge position correction: the switch + the kernel
│   ├── quantum_metric.py    # Quantum metric tensor + linear response
│   ├── delta_Q.py           # DC field-induced quantum geometric tensor change
│   └── jdos.py              # Joint density of states
│
├── parallel.py                  # MPI wrapper (mpi4py, serial fallback)
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
    │                      └─ sampled ω list, or analytic ∫dω ω^(-p) via
    │                         [calc/freq_integral.py]
    ├─ quantum_metric  → compute_quantum_metric()  [calc/quantum_metric.py]
    ├─ delta_Q         → compute_delta_Q()         [calc/delta_Q.py]
    └─ jdos            → compute_jdos()            [calc/jdos.py]
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

Computes χ^(2)_abc(ω1, ω2) using density-matrix perturbation theory. The
per-k-point work is done by `nonlinear_optical_fast._process_kpoint_fast`;
`_process_kpoint` in this module is the unvectorized reference and must track it.

**At each k-point:**
1. `get_H_v(order=2)` → H, S, first and second velocity operators
2. Diagonalize → eigenvalues `ek`, eigenvectors `ψ`
3. Build velocity matrix in eigenbasis: `v_nm = ψ†·vtb·ψ`
4. Build position operator: `r_nm = -i·v_nm / (E_n - E_m)` for n ≠ m
5. Build generalized derivative `dk_r` via `_compute_dk_rmtx`
6. Apply the Wannier-gauge correction to both (Eqs. 22 and 36) when
   `system.wannier_r` is on — see `calc/wannier_gauge.py`
7. Compute 14 chi components for each (ef, ω1) pair

**14 chi components:**
`chi_ii`, `chi_ee1`, `chi_ee2`, `chi_ei1`, `chi_ei2`,
`chi_eit1`, `chi_eit2`, `chi_eit3`, `chi_ie1`, `chi_ie2`,
`chi_e1`, `chi_e2`, `chi_i1`, `chi_i2`

Each component has shape `(nef, nomega)`.

**k-grid:** BZ-centered: `tk = -b1/2 - b2/2 + db1·kc1 + db2·kc2`

#### Frequency-integrated mode

Supplying `calc.freq_integral` instead of `calc.omega1list` switches the ω axis
from *photon energies* to *analytic-integral channels*, returning

```
J = ∫_{omega_min}^{omega_max} dω ω^(-p) χ^(2)(ω)
```

The whole feature rests on one structural fact: **ω enters every χ term only
through denominator factors.** Matrix elements, Fermi factors and vertex
prefactors are ω-independent, so each term is
`(ω-independent prefactor) × (rational function of ω)` and the ω-integral can be
done per term with the prefactor pulled out.

Additional helpers in this module:

- **`_estimate_direct_gap(system, nk_cfg, ef, nk_max=16)`** — minimum direct gap
  straddling `ef` on a coarse grid (≤16³ points), computed on rank 0 and
  broadcast. Used only for the `omega_min << E_gap` warning; returns `inf` when
  no k-point has states on both sides of `ef`.
- **`_split_freq_integral(result, spec)`** — after the MPI reduce, splits the
  channel axis into the `J` columns (which keep the original chi names) and the
  endpoint-diagnostic columns, emitted as extra top-level names
  `endpt_log_<name>` / `endpt_pow<j>_<name>` for `CHI_PHYSICAL + ['chi_total']`.
  These names contain no `.`, so the existing `.npz` flat-key save/load works
  unchanged.

Rejected configurations: `omega1list` together with `freq_integral`;
`method='projector'` with `freq_integral` (the projector χ_e path assumes a
sampled ω list).

---

### `calc/nonlinear_optical_fast.py`

#### `_process_kpoint_fast(..., freq_integral=None, wannier_r=WANNIER_R_DEFAULT)`

Vectorized per-k-point kernel. All arrays carry a trailing length-`W` axis.
That axis is an **ω axis, not necessarily a list of photon energies**: every
phase-3 contraction is linear in the ω-dependent kernels, so substituting
integrated kernels for sampled ones turns the same code into the integrated
engine. `freq_integral=None` gives the historical sampled behaviour.

`wannier_r` gates the Wannier-gauge correction, but only in the Phase-1
operator build: when the caller supplies `_k_data`, the operators are taken as
given and the flag has no effect.

#### `_build_omega_kernels(de_mtx, omega1list, omega2_val, eta_val, freq_integral)`

The single place ω-dependent factors are formed — the reason the two modes
cannot drift. Returns `(K, denom2, denom2_sq, s_om2)`.

| kernel | shape | factor |
|---|---|---|
| `d1` | (D,D,W) | `1/(ω - Δe + iη)` |
| `d12` | (D,D,W) | `1/(ω + ω₂ - Δe + 2iη)` |
| `d12_d1` | (D,D,W) | product of the two above, same band pair |
| `d12_d1sq` | (D,D,W) | `d12 · d1²` |
| `d12_om1` | (D,D,W) | `d12 · 1/(ω + iη)` |
| `om1`, `om12`, `om12_om1` | (W,) | band-independent scalar denominators |
| `const` | (W,) | `1` (sampled) / `∫dω ω^(-p)` (integrated) |
| `ee1A` | (D,D,D,W) | `[m,n,p] = d12[m,n]·d1[m,p]` |
| `ee1B` | (D,D,D,W) | `[m,n,p] = d12[m,n]·d1[p,n]` |

Two subtleties this table encodes:

- **`denom1` and `denom12` carry different broadening** (`iη` vs `2iη`), so
  their poles sit at different points and `d12_d1` is *not* `1/(ω-z)²`. It needs
  a genuine two-pole kernel; a single-pole formula is insufficient.
- **`denom2` and `s_om2` depend only on the fixed ω₂**, so they pass straight
  through the integral as constants and multiply the integrated kernels
  (`d12_d2 = denom2 · d12`). Only factors that actually depend on ω need a
  kernel.

`chi_ee1` is the one term whose two ω-factors sit at *different band pairs*, so
its kernel needs three band indices and does not factorize once integrated.
Both modes therefore use the (D,D,D,W) form; that reassociates its
floating-point sum (~8e-15 vs. the pre-refactor code — every other term is
bit-identical).

`chi_e1` and `chi_i1` have no ω dependence at all, so their kernel is `const`.

---

### `calc/freq_integral.py`

#### `rational_integral(p, poles, mults, a, b=inf, want_cond=False)`

```
∫_a^b dω / ( ω^p ∏_i (ω - z_i)^(s_i) )
```

for integer `p >= 0` and arbitrary multiplicities, vectorized over arrays of
poles. Returns `(J, A[, cond])`, where `A[j-1]` is the partial-fraction
coefficient `A_j` and `cond` is the cancellation ratio.

Preconditions, all guaranteed by the calling kernels when `eta > 0`:

- **`Im z_i < 0` for every pole.** The χ denominators are retarded, so
  `1/(ω - ω_nm + iη) = 1/(ω - z)` with `z = ω_nm - iη`. This is what makes the
  log branch unambiguous: for real `ω > 0` both `a - z` and `b - z` have
  positive imaginary part, so their principal arguments lie in `(0, π)`.
  Logs are always evaluated as **differences of principal logs**, never as
  `log((b-z)/(a-z))`, which can cross the cut.
- **Poles distinct.** Pairs that occur share a real part but differ by `iη`.
  Genuine repeated poles are passed as a multiplicity, not as two entries.
- **`a > 0`.**

`b = inf` is handled by dropping all upper-endpoint terms, which is exact:
whenever `p + Σs_i >= 2` the simple-pole residues sum to zero, so the log terms
cancel and the antiderivative vanishes at infinity. `J = -F(a)`, with no
large-`b` cancellation. When `p + Σs_i < 2` the limit does not exist and `J` is
returned as NaN (`A` is still filled, since it describes the `a` endpoint).

#### `partial_fractions(p, poles, mults) -> (A, B)`

Coefficients of

```
1/(ω^p ∏(ω-z_i)^s_i) = Σ_j A_j/ω^j + Σ_i Σ_l B_{i,l}/(ω-z_i)^l
```

obtained by Taylor-expanding the *complementary* factors about each pole
(`_series_inv_power` + `_series_mul`). Numerically identical to the closed-form
generalized-binomial expressions, but with no special case for negative
binomial arguments — the direct formula needs `C(-1, 0)` in a legitimate case,
which `math.comb` rejects.

#### `class FreqIntegralSpec`

Config parsing, validation and channel bookkeeping. Channels per requested `p`:

| channel | meaning |
|---|---|
| `('J', 0)` | the integral |
| `('log', 1)` | coefficient of `ln(omega_min)` in J, i.e. `-A_1` (0 when p=0) |
| `('pow', j)` | coefficient of `omega_min^(1-j)`, i.e. `A_j/(j-1)`, j = 2..pmax |

- `from_config(block, eta, gap=None)` applies the guards
  (`OMEGA_MIN_ETA_ERROR = 5`, `OMEGA_MIN_ETA_WARN = 10`, `OMEGA_MIN_GAP_WARN = 0.1`);
  `omega_min` defaults to `10*eta`, `eta <= 0` is rejected.
- `kernel(poles, mults)` returns the integrated kernel, shape
  `broadcast(poles) + (nchan,)`, one `rational_integral` call per power.
- `max_cond` accumulates the worst cancellation ratio seen; the driver
  MPI-max-reduces it and prints it.

**Conditioning.** At ω₂ = 0 the `z12`/`z1` pair is separated by exactly `iη`,
i.e. nearly coincident on the scale over which the integrand varies. The
`1/(z₁ - z₂)` factors in the partial fractions cancel against each other,
costing roughly `log10(|z|/η)` digits per unit of excess multiplicity — about 8
digits for the `s = 3` kernel at η = 1e-3. Measured end-to-end accuracy there is
~1e-8.

**Tests:** `examples/test_freq_integral.py` (8 groups, including a
sampled-mode regression against a `git worktree` baseline passed as `argv[1]`).

---

### `calc/wannier_gauge.py`

The single switch and the single kernel for the Wannier-gauge position
correction. Every engine that builds an `r` operator — `nonlinear_optical`,
`nonlinear_optical_fast`, `delta_Q`, `quantum_metric` — goes through both, so
they cannot drift apart. A new engine with a position operator must do the same.

#### `compute_A_W_k(system, k, dir_chars=None, enabled=True, need_deriv=True)`

Fourier-interpolates the Wannier position matrices (Eq. 20 of
arXiv:1804.04030):

```
A^(W)_{k,nm,a} = Σ_R exp(ik·(R + τ_m - τ_n)) <0n|r̂_a - τ_{m,a}|Rm>
```

Returns `(A_W, dA_W)` with `A_W[d]` and `dA_W[d1][d2] = ∂_{d2} A^(W)_{d1}`, or
`(None, None)` when `enabled=False` or the system has no `wannier_r_matrices`.
Both cases returning `None` is what lets every call site keep a single
`if A_W is not None:` guard with no separate branch for the flag.
`need_deriv=False` skips `dA_W` for callers that only correct `r` itself.

This function was `nonlinear_optical._compute_A_W_k` before the switch was
unified; that name survives there as an alias.

#### `resolve_wannier_r(cfg, system, engine) -> bool`

Reads `cfg['system']['wannier_r']`, defaulting to `WANNIER_R_DEFAULT`. Raises
on a non-boolean value and on the old `cfg['calc']['wannier_r']` location.
Prints one notice per engine per run: *inert* when the system carries no
position matrices, and a warning naming `BUG_wannier_r_correction.md` at
**either** setting when it does — with that bug open, neither value gives a
trustworthy answer on `_tb.dat` input.

`WANNIER_R_DEFAULT` is currently `False`, which is *not* the physically correct
value; it is the diagnostic setting, chosen while the correction is known
defective. Flip it back to `True` when the bug closes.

Also exports `system_has_wannier_r(system)`, `validate_wannier_r(cfg)` (called
from `config.load_config` so YAML errors surface at load time),
`warn_unused_wannier_r(cfg, calc_type)` (called from `main._dispatch` for
engines with no position operator), `offdiag_A_H(A_W, psi, dir_chars)`, and
`reset_notices()` for tests that drive many runs in one process.

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
6. Wannier-gauge connection `a^(H) = offdiag(ψ† A^(W) ψ)`, and the same rotated
   by `ψ±` for the perturbed quantities — skipped when `system.wannier_r` is
   off or the system carries no position matrices
7. Loop over Fermi energies, everything through `_rr_sum`:
   - **Q[d1][d2]** = Σ_nm r[d1]_nm · conj(r[d2]_nm) · f_nm
   - **dQ[d1][d2][d3]** = (Q+ - Q-) / (2δ)  — intrinsic, via numerical derivative
   - **dQf[d1][d2][d3]** = Σ_nm r[d1]_nm · conj(r[d2]_nm) · df_nm[d3]
     — extrinsic (Fermi surface)

#### `_rr_sum(v1, v2, a1, a2, inv_de, inv_de2, weight)`

The one place the quadratic form is built, so Q, dQ and dQf cannot use
different position operators. With `r = -i v/ω + a^(H)` it expands to

```
v1 conj(v2)/ω² + (-i v1/ω) conj(a2) + a1 conj(-i v2/ω) + a1 conj(a2)
```

deliberately *not* written as `r1 * conj(r2)`: keeping the leading term in the
original factor order makes the `a = None` case reproduce the pre-correction
engine bit for bit. That is not cosmetic — dQ is a finite difference of two
nearly equal sums, so a 1-ulp reassociation is amplified by `|Q| / (δ|dQ|)`,
a factor of ~1e8 wherever dQ is small.

**Position operator:** `r = -i v/ω` plus the Wannier-gauge correction when
`system.wannier_r` is on; see `calc/wannier_gauge.py`. Masking matches
χ^(2)/`delta_Q`: the degeneracy mask applies to the `1/ω` factor only, and
`a^(H)` — smooth across a degeneracy — is added unmasked.

**Degeneracy handling:** Pairs with `|ΔE| < 1e-5` are masked out (zero
contribution). Note this is a hard cutoff, unlike the Souza `eta_sos`
regularization used by `nonlinear_optical.py` and `delta_Q.py` — an
inconsistency that predates the shared position operator.

---

### `calc/delta_Q.py`

#### `compute_delta_Q(system, cfg) -> dict`

DC field-induced change in the quantum geometric tensor, δQ^{ab} for a static
field along `c`. Implements Eq. 40 of `revised_formula_sheet_eta.pdf`.

> ⚠️ Shares `compute_A_W_k` with `nonlinear_optical.py` and
> `quantum_metric.py`, and that function is currently broken for Wannier input
> — see `BUG_wannier_r_correction.md`. TB_simple systems are unaffected:
> `bloch.py` builds H(k) in the atomic gauge, where the tight-binding position
> operator is exactly `r = -i v/ω` with no intra-cell correction, so
> `compute_A_W_k` correctly returns `None`.

**Returns** `{'Q_tilde': {}, 'delta_Q': ..., 'delta_Q_terms': ...}` with
`delta_Q[a][b][c] -> array(nef,)`. `Q_tilde` is always empty.

**Broadening — three distinct parameters, easily confused:**

| symbol | where it enters |
|---|---|
| `eta` | the DC-perturbation denominators only: `1/(ω_nm ± iη)` |
| `eta_sos` | Souza regularization of the *bare* `1/ω_nm`: `ω/(ω² + η_sos²)` |
| — | projector-derivative factors (`v/ω`) stay bare apart from `eta_sos` |

The `+iη`/`−iη` split across Trace II and Trace III is what preserves
Hermiticity of δP_n. `eta_sos` defaults to `0.05`, large enough to matter in
low-energy models. `deg_thr` is accepted for backward compatibility, ignored,
and warned about once.

**Two formulations**, selected by `dQ_occupied_subspace` (default `True`):

- **Subspace** (`_assemble_delta_Q_subspace`) — responds
  `Q_occ = Tr[P_occ ∂ₐP_occ ∂_b P_occ]`. The outer (p,q) sum is masked by
  `f_p (1 - f_q)`, and an extra `T_mix` term appears from inner three-band sums
  whose intermediate index is restricted to the occupied manifold. Note the
  derivation literally gives `f_p (f_q - f_p)`; the code uses `f_p (1 - f_q)`
  for consistency with the interband convention elsewhere, differing by a
  self-smear term negligible for `kT ≪ gap`.
- **Band-resolved** (`_assemble_delta_Q`) — `Σ_n f_n δQ^{ab}_n`, 6 terms, no
  `T_mix`.

Both share `_compute_pair_matrices`, which returns the pair integrands before
the outer contraction; only the outer mask differs. That is what keeps the two
paths from drifting.

**Term names:** `T_Sipe_Delta`, `T_Sipe_d2H`, `T_Sipe_3band`,
`T_Sipe_wannier_corr`, `T_Delta`, `T_3band` (+ `T_mix` in the subspace path).
The four `T_Sipe_*` pieces sum exactly to the full Sipe generalized derivative —
`wannier_corr` is pre-populated with zeros so the bookkeeping holds whether or
not the system carries Wannier position matrices.

**k-grid:** 2D only — `nk1, nk2 = calc['nk']` unpacks exactly two entries.
Periodic spacing `db = b/nk`.

**Sign convention:** the implemented Eq. 40 corresponds to `H' = -E·r`; notes
written with `H' = +E·r` differ by an overall minus, verified pointwise.

**Serialization caveat:** `_save_delta_Q` writes the term decomposition under
5-part keys `delta_Q_terms.<a>.<b>.<c>.<term>`, but `load_delta_Q` only handles
3- and 4-part keys, so those entries are silently dropped on reload. Read them
with `np.load` directly.

---

### `calc/jdos.py`

#### `compute_jdos(system, cfg) -> dict`

```
D(ω) = (1/N_k) Σ_k Σ_{n≠m} [f(E_m) - f(E_n)] · L(E_n - E_m - ω, η)
L(x, η) = (1/π) η / (x² + η²)
```

**At each k-point:** `get_H_k` → `diagonalize_hk` (eigenvectors not needed) →
optional band restriction → build `de[n,m]` and `f_mn[n,m]` → broadcast the
Lorentzian over ω as a `(nomega, N, N)` array and sum the two band axes.

Dimensionality comes from `all_ek.detect_dimensionality`; `nk` is broadcast from
a scalar or padded/truncated to `ndim` entries.

**Returns** a flat dict: `omega`, `jdos`, `ef`, `eta`, `kT`, `nk`, `ndim`.

Notes for anyone extending this:

- Only `eflist[0]` is used — the engine is single-Fermi-level by construction.
- `bands` indexes into the *sorted* eigenvalue array at each k, so a fixed index
  list does not follow a band through a crossing.
- The reduction goes through `reduce_sum_complex_array` on a cast-to-complex
  copy, then takes `.real`, because `parallel.py` has no real-array sum reducer
  wired in here.
- There is no `load_jdos` in `main.py`; `_save_jdos` exists but the read path is
  plain `np.load`.
- **k-grid:** uses `db = b/(nk-1)` (endpoint-inclusive), mirroring `all_ek` and
  *unlike* the periodic `db = b/nk` used by the response engines. Both zone
  edges are sampled while the `1/N_k` weight is unchanged, an O(1/nk) bias in
  the absolute normalization. See the k-grid convention note in `CLAUDE.md`.

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

5. **MPI parallelism**: k-point parallelism via `parallel.py`, a thin `mpi4py` wrapper that scatters k-points round-robin and all-reduces the per-rank accumulators. When `mpi4py` is absent the same code path runs serially, so engines need no branching.

6. **Embedded config in output**: Every `.npz` file stores the full config as JSON, ensuring reproducibility.

7. **One place per ω-dependence**: In χ^(2), ω enters only through denominator factors. `_build_omega_kernels` is the sole place those are formed, with sampled and analytically-integrated backends behind one interface. The term algebra downstream is shared verbatim, so the two modes cannot drift — the alternative, a parallel copy of the ~15 term expressions, is the drift risk already flagged between `nonlinear_optical.py` and `nonlinear_optical_fast.py`.
