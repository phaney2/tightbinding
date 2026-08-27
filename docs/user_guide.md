# Tight-Binding Code — User Guide

## Installation

The package requires Python 3.10+ and the following dependencies:

```
numpy
scipy
matplotlib
pyyaml
```

No installation step is needed — run directly from the repository root.

## Running a Calculation

```bash
python -m tightbinding <config.yaml>
```

The program reads a YAML configuration file, builds the tight-binding system,
and dispatches to the requested calculation engine. Results are printed to
stdout and optionally saved to `.npz` files and/or PNG plots.

---

## Configuration File Format

Every config file has four top-level sections:

```yaml
system:    # Crystal structure and orbital basis
hopping:   # Hopping parameters
onsite:    # On-site energies (optional for some setups)
calc:      # Calculation type and parameters
output:    # Output file names (optional)
```

### `system` — Crystal Structure

#### Lattice vectors (required)

```yaml
system:
  lattice_vectors:
    - [1, 0, 0]       # a1
    - [0, 0, 1]       # a2
    - [0, 10, 0]      # a3 (large = vacuum for 2D)
```

Rows are the three lattice vectors a1, a2, a3 in Cartesian coordinates (Angstroms or any consistent unit). For 2D systems, set one lattice vector much larger than the hopping range to create a vacuum direction.

#### Orbital basis (required)

```yaml
system:
  basis: sp_u
```

All basis types project from a unified 18-dim (spd×spin) full space. Available types:

| Basis | Orbitals | Dim | Description |
|-------|----------|-----|-------------|
| `s_u` | s↑ | 1 | s orbital, spin-up |
| `s_d` | s↓ | 1 | s orbital, spin-down |
| `s_ud` | s↑, s↓ | 2 | s orbital, both spins |
| `p_u` | px↑, py↑, pz↑ | 3 | All p orbitals, spin-up |
| `pxpy_u` | px↑, py↑ | 2 | In-plane p, spin-up |
| `pxpz_u` | px↑, pz↑ | 2 | p_x + p_z, spin-up |
| `pxpy_d` | px↓, py↓ | 2 | In-plane p, spin-down |
| `pxpy_ud` | px↑↓, py↑↓ | 4 | In-plane p, both spins |
| `spxpy_u` | s↑, px↑, py↑ | 3 | s + in-plane p, spin-up |
| `sp_u` | s↑, p↑ | 4 | s + p orbitals, spin-up |
| `sp_ud` | s↑↓, p↑↓ | 8 | Full s + p, both spins |
| `sj_u` | s + j=1/2↑ | 2 | j-basis, up-like |
| `sj_d` | s + j=1/2↓ | 2 | j-basis, down-like |
| `sj_ud` | s + j=3/2 | 4 | Total angular momentum basis |
| `d_u` | dxy,dyz,dzx,dx²-y²,dz² ↑ | 5 | All d orbitals, spin-up |
| `d_z_even_u` | dz²,dx²-y²,dxy ↑ | 3 | z-even d orbitals, spin-up |
| `pd_u` | px,py,pz,dz²,dx²-y²,dxy ↑ | 6 | p + z-even d, spin-up |
| `spd_u` | s,p,d ↑ | 9 | Full spd, spin-up |
| `spd_ud` | s,p,d ↑↓ | 18 | Full spd, both spins |

#### Atom positions

**Option 1 — Lattice type shorthand:**

```yaml
system:
  lattice_type: 2d_square
  Nx: 1
  Ny: 1
```

Currently only `2d_square` is implemented. `Nx` and `Ny` set the number of atoms per unit cell along each direction (usually 1×1 for a single-atom cell).

**Option 2 — Explicit positions (multi-atom cells):**

```yaml
system:
  positions:
    - {coord: [0, 0, 0], species: 'A'}
    - {coord: [0.5, 0.5, 0], species: 'B'}
  coord_type: fractional   # or 'cartesian' (default)
```

When `coord_type: fractional`, coordinates are interpreted as fractions of the lattice vectors: `r = f1*a1 + f2*a2 + f3*a3`.

**Per-atom basis:** Each atom can override the system-level `basis` by specifying its own:

```yaml
system:
  positions:
    - species: Mo
      basis: d_u           # 5 d-orbitals on Mo
      coord: [0, 0, 0]
    - species: S
      basis: p_u           # 3 p-orbitals on S
      coord: [0, 1.84, 1.59]
```

When per-atom `basis` is used, the system-level `basis` key is optional (it serves as a fallback for atoms that don't specify one).

### `hopping` — Hopping Parameters

```yaml
hopping:
  range: 1.5           # cutoff distance for neighbor search
  tss_sigma: -0.5      # s-s sigma
  tsp_sigma: 0.707     # s-p sigma
  tpp_sigma: 1.0       # p-p sigma
  tpp_pi: 0.0          # p-p pi
  tsd_sigma: 0.0       # s-d sigma
  tpd_sigma: 0.0       # p-d sigma
  tpd_pi: 0.0          # p-d pi
  tdd_sigma: 0.0       # d-d sigma
  tdd_pi: 0.0          # d-d pi
  tdd_delta: 0.0       # d-d delta
  tsp_rashba: 0.0      # Rashba corrections (optional, one per channel)
```

The `range` parameter controls which neighbors are included. All atom pairs within this distance are connected by hopping. Direction-dependent hopping is computed via Slater-Koster rules using the bond direction cosines.

**Per-species hopping** for multi-atom cells with different atom types:

```yaml
hopping:
  range: 3.20
  Mo:
    tpd_sigma: -2.619
    tdd_sigma: -0.933
  S:
    tpd_sigma: -2.619
    tpp_sigma: 0.696
```

Inter-species hopping parameters are averaged: `t_ij = (t_i + t_j) / 2`.

### `onsite` — On-Site Energies

```yaml
onsite:
  u_s: 10.0           # s orbital energy
  u_p: 0.0            # p orbital energy (px, py)
  u_pz: null           # pz orbital energy (defaults to u_p if null/omitted)
  u_d: 0.0            # d orbital energy (dxy, dx²-y²)
  u_dz2: null          # dz² energy (defaults to u_d if null/omitted)
  u_dxz: null          # dxz,dyz energy (defaults to u_d if null/omitted)
  delta_s: 0.0        # s exchange splitting
  delta_p: 0.0        # p exchange splitting
  delta_d: 0.0        # d exchange splitting
  theta: 0            # magnetization polar angle (radians)
  phi: 0              # magnetization azimuthal angle (radians)
  spinorbit: 0.0      # p-orbital spin-orbit coupling constant
  spinorbit_d: 0.0    # d-orbital spin-orbit coupling constant
```

All on-site parameters are optional and default to 0. The per-orbital overrides (`u_pz`, `u_dz2`, `u_dxz`) allow crystal field splitting without breaking the default structure — if omitted or null, they inherit from the parent orbital energy (`u_p` or `u_d`).

For multi-species systems, provide per-species blocks:

```yaml
onsite:
  Mo:
    u_d: -2.529        # Δ₂: dx²-y², dxy
    u_dz2: -1.016      # Δ₀: dz²
    u_dxz: 0.0         # Δ₁: dxz, dyz
  S:
    u_p: -0.780        # Δ_p: px, py
    u_pz: -7.740       # Δ_z: pz
```

### Wannier90 Input

Instead of building a tight-binding model from scratch, you can import a Hamiltonian from Wannier90 output files. This bypasses the `hopping`, `onsite`, basis, and positions sections entirely.

**Option 1 — `_hr.dat` file** (Hamiltonian only; lattice vectors required in config):

```yaml
system:
  wannier_hr: path/to/MoS2_hr.dat
  lattice_vectors:
    - [2.439, 0.0, 0.0]
    - [-1.219, 2.112, 0.0]
    - [0.0, 0.0, 13.0]
  wannier_centres: path/to/MoS2_centres.xyz   # optional

calc:
  type: band_structure
  ...
```

**Option 2 — `_tb.dat` file** (Hamiltonian + lattice vectors + position operator):

```yaml
system:
  wannier_tb: path/to/MoS2_tb.dat
  wannier_centres: path/to/MoS2_centres.xyz   # optional

calc:
  type: band_structure
  ...
```

The `_tb.dat` format is produced by Wannier90 with `write_tb = .true.` in the `.win` file. It contains:
- Lattice vectors (read automatically, no need to specify `lattice_vectors` in the config)
- Hamiltonian matrix elements (same as `_hr.dat`)
- Position operator matrix elements `<0n|r|Rm>` (parsed and stored for later use in response calculations)

**Notes:**
- When using `wannier_tb`, the `lattice_vectors` key is optional — if omitted, the lattice vectors are read from the file. If provided, the config value is ignored and the file's lattice vectors are used.
- The `wannier_centres` option provides Wannier function centre positions for building the intra-cell displacement matrices needed by velocity operators. Without it, all Wannier functions are treated as located at the origin.

---

## Calculation Engines

### Band Structure (`band_structure`)

Computes energy eigenvalues along a k-space path through high-symmetry points.

```yaml
calc:
  type: band_structure
  kpath:
    points:
      Gamma: [0, 0, 0]
      X: [0.5, 0, 0]
      M: [0.5, 0, 0.5]
    path: [Gamma, X, M, Gamma]
    npoints: 100          # points per segment (optional, default 100)
  no_plot: false          # set true to suppress PNG output (optional)

output:
  file: my_bands          # plot saved as my_bands_bands.png
```

**k-path points** are given in **fractional reciprocal coordinates**: `k = f1*b1 + f2*b2 + f3*b3`, where b1, b2, b3 are the reciprocal lattice vectors.

**Output:**
- PNG plot of band structure (unless `no_plot: true`)
- Returns dict with `k_distances`, `energies`, `tick_positions`, `tick_labels`

### Full BZ Eigenvalues + DOS (`all_ek`)

Computes all energy eigenvalues on a uniform k-grid spanning the full Brillouin zone. Auto-detects system dimensionality (1D, 2D, or 3D) from the hopping structure.

```yaml
calc:
  type: all_ek
  nk: [60, 60]            # k-grid resolution (1, 2, or 3 values)
  eflist: [2.0]           # Fermi energy for band gap detection (optional)
  dos_bins: 200           # number of DOS histogram bins (optional, default 200)
  compute_berry_curvature: true  # compute k-resolved Berry curvature (optional, default false)
  kT: 0.01               # smearing temperature for occupations (optional, default 0.05)
  outputfile: results/out # saves to results/out.npz
```

**Berry curvature:** When `compute_berry_curvature` is enabled, the Berry curvature of the occupied bands is computed at each k-point using the Kubo formula:

Omega_z(k) = -2 sum_{n occ, m unocc} Im[vx_nm * vy_mn] / (em - en)^2

This requires `eflist` to determine band occupations. The result includes `berry_curvature` (array on the k-grid) and `k_list` (Cartesian k-point coordinates). The BZ integral of the Berry curvature gives the Chern number (zero for time-reversal-invariant systems).

**Output:**
- PNG plot: energy surfaces (2D) or E(k) (1D) + density of states
- `.npz` file with eigenvalues, DOS, band summary (+ Berry curvature if enabled)
- Printed summary: band edges, bandwidths, band gap, van Hove singularities

**Reloading saved results:**

```python
from tightbinding.main import load_all_ek
result, cfg = load_all_ek('results/out')
# result['ekset']        — eigenvalue array
# result['dos_energies'] — DOS bin centers
# result['dos_values']   — DOS values
# result['band_summary'] — per-band min/max/bandwidth
# result['band_gap']     — band gap (or None)
```

### Nonlinear Optical Response (`nonlinear_optical`)

Computes the second-order nonlinear optical susceptibility χ^(2)_abc(ω1, ω2) using density-matrix perturbation theory with full interband/intraband decomposition.

```yaml
calc:
  type: nonlinear_optical
  nk: [60, 60]            # k-grid resolution
  omega1list: [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]   # photon energies (eV)
  omega2: 0.0             # second photon energy (optional, default 0 = DC)
  eta: 0.075              # broadening parameter (eV)
  eflist: [5.0]           # Fermi energies to compute
  kT: 1.0e-7              # temperature (eV), small = zero-T limit
  directions: ['xzx', 'zxx']   # direction triplets abc for χ_abc
  outputfile: results/chi2      # saves to results/chi2.npz
```

Each direction string `'abc'` specifies the three Cartesian indices of χ^(2). Common choices:
- `'xzx'`, `'zxx'` — relevant for 2D systems with broken z-mirror symmetry

**Output:**
- `.npz` file with all 14 chi components for each direction triplet
- Result structure: `result[chi_name][a][b][c]` → complex array of shape `(nef, nomega)`

**Reloading:**

```python
from tightbinding.main import load_nonlinear_optical
result, cfg = load_nonlinear_optical('results/chi2')
# result['chi_ii']['x']['z']['x']  — array(nef, nomega)
```

### Frequency-Integrated χ^(2) (`nonlinear_optical` + `freq_integral`)

The same engine can return the **frequency-integrated** response

```
J = ∫ dω  ω^(-p)  χ^(2)(ω)
```

computed from closed-form antiderivatives rather than by quadrature over an ω
mesh. Replace `omega1list` with a `freq_integral` block — supplying both is an
error:

```yaml
calc:
  type: nonlinear_optical
  nk: [120, 120]
  freq_integral:
    p: [1, 2]           # ω^(-p) weight; scalar or list, one output column each
    omega_min: 0.01     # optional — defaults to 10*eta
    omega_max: .inf     # optional — default
    diagnostics: true   # optional — default
  omega2: 0.0
  eta: 1.0e-3
  eta_sos: 1.0e-8
  eflist: [0.0]
  kT: 1.0e-5
  directions: ['yyy', 'yxx']
  outputfile: results/chi2_int
```

Use this instead of integrating `omega1list` numerically: matching the closed
form by quadrature needs ~10⁶ ω-points per k-point, and adaptive integrators do
*worse*, not better, because the sharpest terms have an integrand peaking at
~1/η³ that integrates to O(1).

**Choosing `omega_min` — the window is two-sided:**

```
~10·eta  <<  omega_min  <<  E_gap
```

The lower bound is not cosmetic. Several vertices contribute poles at `ω = -iη`
and `-2iη`, sitting right on top of the `ω^(-p)` endpoint. Below `ω ~ η` the
integrand changes character, so an `omega_min` inside that crossover makes `J` a
function of the η/`omega_min` interplay rather than of the physics. The code
enforces this:

| condition | behaviour |
|---|---|
| `omega_min` omitted | set to `10*eta` |
| `omega_min < 5*eta` | **hard error** |
| `omega_min < 10*eta` | warning |
| `omega_min > 0.1*E_gap` | warning (gap estimated on a coarse pre-pass) |
| `eta = 0` | **hard error** |

Both bounds together require `eta << E_gap/10`. A model with gap 0.2 eV and
`eta = 0.025` has **no valid window at all** — reduce `eta` rather than
`omega_min`.

**What converges at `omega_max: .inf`:**

- `p >= 2` — every term.
- `p = 1` — everything except `chi_e1` and `chi_i1`, which are ω-independent and
  come back as **NaN**. Both are unphysical and already excluded from
  `chi_total`, so nothing else is affected.
- `p = 0` — several terms diverge individually; set a finite `omega_max`.

**Output:**
- Same `result[name][a][b][c]` nesting and same `.npz` save/load, but arrays are
  shape `(nef, n_p)` — one column per requested power, in the order given.
- Reported in the **same convention as the sampled path**, so `Im[J]` is the
  reactive part and `-Re[J]` the dissipative one.
- Extra keys `endpt_log_<name>` and (for p ≥ 2) `endpt_pow<j>_<name>`, for the
  seven physical terms and `chi_total` — see below.

```python
from tightbinding.main import load_nonlinear_optical
result, cfg = load_nonlinear_optical('results/chi2_int')
J = result['chi_total']['y']['y']['y']       # (nef, n_p); column 0 is p=1
lncoef = result['endpt_log_chi_total']['y']['y']['y']
```

**`omega_min` dependence — read the diagnostic, don't assume it cancels.**
Each term diverges as `omega_min → 0`, and the `endpt_*` keys hold the
coefficients of that divergence, BZ-accumulated. Two exact identities apply:

```
p = 1:  coefficient of ln(omega_min)   = -chi(omega=0)
p = 2:  coefficient of omega_min^(-1)  = +chi(omega=0)
        coefficient of ln(omega_min)   = -dchi/domega (0)
```

So **at p = 1 the integral is genuinely log-divergent as `omega_min → 0`**,
with coefficient `-χ^(2)(0)`; the individual-term divergences only partly
cancel. Always quote `J` together with the `omega_min` it was computed at. If
the diagnostic comes back at round-off, `J` is `omega_min`-independent and you
can say so.

**Accuracy.** The run prints a worst-case cancellation ratio; its base-10 log is
roughly the number of digits lost to the nearly-coincident poles that the
broadening creates. At `eta = 1e-3` this is ~8 digits in the worst kernel, with
measured end-to-end accuracy ~1e-8 — still two orders better than what a fine ω
mesh achieves. Larger `eta` is better conditioned.

### Quantum Metric (`quantum_metric`)

Computes the quantum metric tensor Q and its DC linear response (intrinsic dQ and extrinsic dQf) over the Brillouin zone.

```yaml
calc:
  type: quantum_metric
  nk: [60, 60]             # k-grid resolution
  eflist: [2.0]            # Fermi energies
  kT: 0.1                  # temperature (eV)
  eta: 0.1                 # broadening (eV)
  delta: 0.001             # finite-difference step (optional, default 0.001)
  metric_directions: [x, z]   # directions to compute (optional, default [x, z])
  outputfile: results/qm      # saves to results/qm.npz
```

**Output:**
- `.npz` file with Q, dQ, dQf tensors
- Q[d1][d2] → complex array of shape `(nef,)` — metric tensor
- dQ[d1][d2][d3] → complex array of shape `(nef,)` — intrinsic response
- dQf[d1][d2][d3] → complex array of shape `(nef,)` — extrinsic (Fermi surface) response

**Reloading:**

```python
from tightbinding.main import load_quantum_metric
result, cfg = load_quantum_metric('results/qm')
# result['Q']['x']['z']       — array(nef,)
# result['dQ']['x']['z']['x'] — array(nef,)
```

### DC Field-Induced Quantum Geometry (`delta_Q`)

Computes δQ^{ab}, the change in the quantum geometric tensor produced by a
static electric field along direction `c`, with adiabatic iη broadening.

> ⚠️ **Read the blocking-bug notice in `CLAUDE.md` before running this on
> Wannier (`_tb.dat`) input.** `delta_Q` shares the Wannier-gauge position
> correction with χ^(2), and that function is currently broken — it makes
> `delta_Q` complex when it must be real. Systems built from YAML (TB_simple)
> are unaffected, because the atomic gauge needs no such correction.

```yaml
calc:
  type: delta_Q
  nk: [60, 60]              # 2D grid — exactly two entries required
  eflist: [2.0]             # Fermi energies
  kT: 0.1                   # temperature (eV)
  eta: 0.1                  # adiabatic broadening (optional, default 0.0)
  eta_sos: 0.05             # Souza regularization of 1/w_nm (optional)
  components: [xz, zx]      # (a,b) metric index pairs, or 'all'
  field_direction: x        # DC field direction c — string or list
  directions: [x, z]        # only needed when components: 'all'
  outputfile: results/dQ
```

Additional switches, both defaulting to `true`:

- `wannier_r` — apply the Wannier-gauge position correction when the system
  carries one. Set `false` only for diagnostic comparison against the bare
  tight-binding form; it is *not* a fix for the bug above.
- `dQ_occupied_subspace` — respond the occupied-subspace tensor
  `Q_occ = Tr[P_occ ∂ₐP_occ ∂_b P_occ]`, restricting the outer band sum to
  occupied→unoccupied pairs. Setting `false` reverts to the band-resolved
  `Σ_n f_n δQ_n` form and drops the extra `T_mix` term.

`deg_thr` is accepted but ignored (it was replaced by the `eta_sos`
regularization); the code warns once if you pass it.

**Output:**
- `result['delta_Q'][a][b][c]` → complex array of shape `(nef,)`
- `result['delta_Q_terms'][a][b][c][term]` → same shape, per-term breakdown.
  Terms: `T_Sipe_Delta`, `T_Sipe_d2H`, `T_Sipe_3band`, `T_Sipe_wannier_corr`,
  `T_Delta`, `T_3band`, plus `T_mix` when `dQ_occupied_subspace` is on.
- `result['Q_tilde']` is present but always empty.

**Reloading:**

```python
from tightbinding.main import load_delta_Q
result, cfg = load_delta_Q('results/dQ')
# result['delta_Q']['x']['z']['x'] — array(nef,)
```

Note: `load_delta_Q` returns only `Q_tilde` and `delta_Q`. The per-term
decomposition *is* written to the `.npz` under keys
`delta_Q_terms.<a>.<b>.<c>.<term>`, but the loader drops them — read those
directly with `np.load` if you need them.

**Sign convention.** The implemented formula corresponds to `H' = -E·r`. Notes
written with `H' = +E·r` give the opposite overall sign; the two agree to
machine precision in magnitude and structure at every k and every channel.

**Gotcha for small-gap models.** `eta_sos` defaults to `0.05`, which is
comparable to or larger than the gap in low-energy models — set it to ~`1e-8`
whenever bands are never degenerate. Set `eta: 0.0` to compare against
unbroadened analytic results, and keep `kT` well below the gap.

### Joint Density of States (`jdos`)

Computes the Lorentzian-broadened joint density of states

```
D(ω) = (1/N_k) Σ_k Σ_{n,m} [f(E_m) - f(E_n)] · L(E_n - E_m - ω, η)
```

with `L` a unit-area Lorentzian standing in for the energy-conserving
δ-function. The Fermi weight restricts the positive-ω part to
occupied→unoccupied transitions as T → 0.

```yaml
calc:
  type: jdos
  nk: [80, 80]              # per active lattice direction; a single int works too
  eflist: [0.0]             # only the FIRST entry is used
  kT: 0.025                 # optional, default 0.025
  eta: 0.05                 # Lorentzian width (optional, default 0.05)
  omega_range: [0.0, 5.0]   # optional, default [0, 5]
  nomega: 200               # optional, default 200
  # omegalist: [...]        # alternative: explicit ω values, overrides the above
  bands: all                # optional; or a list of band indices to include
  outputfile: results/jdos
```

Dimensionality is auto-detected, and `nk` is padded or truncated to match, so a
single integer gives an isotropic grid.

**Output** — a flat dict (and the same keys in the `.npz`):

| key | contents |
|---|---|
| `omega` | array(nomega,), eV |
| `jdos` | array(nomega,), states/eV/unit cell |
| `ef`, `eta`, `kT` | the scalars actually used |
| `nk` | grid shape tuple |
| `ndim` | 1, 2 or 3 |

There is no `load_jdos` helper; read the `.npz` directly:

```python
import numpy as np, json
d = np.load('results/jdos.npz', allow_pickle=True)
omega, jdos = d['omega'], d['jdos']
cfg = json.loads(str(d['_config_json']))
```

**Caveat on the k-grid.** `jdos` uses the endpoint-inclusive spacing
`db = b/(nk-1)`, matching `all_ek` rather than the periodic `db = b/nk` used by
`nonlinear_optical`, `quantum_metric` and `delta_Q`. That samples both zone
edges while keeping the `1/N_k` weight, an O(1/nk) bias in the normalization.
Relative lineshapes are unaffected; treat absolute JDOS magnitudes with care and
converge in `nk`.

---

## NRL Tight-Binding (Advanced)

For first-principles-quality tight-binding with density-dependent on-site energies and overlap matrices, use the NRL (Naval Research Laboratory) path. This bypasses the simple TB pipeline and builds the system directly from NRL `.dat` parameter files.

```python
from tightbinding.nrl.hamiltonian_nrl import build_nrl_system

system = build_nrl_system(
    species_list=['Pt'] * natoms,
    coords=atom_positions,           # (natoms, 3) in Bohr
    lattice_vectors=lattice_vecs,    # (3, 3) in Bohr
    hopping_range=16.0,              # Bohr
    dat_file='pt.dat',               # NRL parameter file
    vso={'Pt': {'p': 0.05, 'd': 0.5}},  # spin-orbit coupling (Ry)
)
```

The NRL path:
- Uses 9 orbitals per atom (s, p, d) with spin doubling → 18 orbitals/atom
- Computes density-dependent on-site energies from the local atomic environment
- Includes overlap matrices (generalized eigenvalue problem)
- Supports spin-orbit coupling for p and d orbitals
- Works in Rydberg units internally, converts to eV at the end

The returned `System` object is compatible with all calculation engines.

---

## Parallelization

k-point parallelism goes through MPI (`tightbinding/parallel.py`). When `mpi4py`
is installed, k-points are scattered round-robin across ranks and the results
are all-reduced; without it, the same code path runs serially. Progress is
printed at 10% intervals by rank 0.

```bash
python3 -m tightbinding input.yaml              # serial
mpiexec -np 8 python3 -m tightbinding input.yaml  # 8 ranks
```

The MPI launcher name and its accepted flags vary by machine.

---

## Output Files

All `.npz` output files embed the full configuration as a JSON string under the key `_config_json`. This means you can always recover the input parameters that produced a given result:

```python
import numpy as np, json
data = np.load('results/out.npz', allow_pickle=True)
cfg = json.loads(str(data['_config_json']))
```

---

## Tips

- **Convergence**: Always check convergence with respect to `nk`. Start with a coarse grid (e.g., 20×20) for quick tests, then increase (60×60, 100×100) for production.
- **Broadening**: The `eta` parameter controls Lorentzian broadening. Smaller values give sharper spectral features but require finer k-grids. A good starting point is `eta = 0.05–0.1 eV`.
- **Temperature**: Set `kT` very small (e.g., `1e-7`) for zero-temperature Fermi functions, or use finite `kT` (e.g., `0.1 eV`) to smooth the Fermi surface.
- **Dimensionality**: The code auto-detects 1D/2D/3D from the hopping structure. For a 2D system, set the non-periodic lattice vector large (e.g., `[0, 10, 0]`) and ensure `hopping.range` is smaller than that vector's length.
- **First look**: Use `all_ek` as your first calculation on any new system. The energy surfaces, DOS, and band summary give an immediate overview of the electronic structure.
