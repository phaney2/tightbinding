# Tight-Binding Python Code

## Project Overview
Tight-binding electronic structure code ported from MATLAB. Computes band structures, nonlinear optical response, quantum metric, and density of states for periodic systems (1D/2D/3D).

## Package Structure
```
tightbinding/
  main.py          — entry point: YAML config -> build_system -> fill_hamiltonian -> calc engine
  config.py        — YAML config loader
  types.py         — dataclasses: System, Atom, HoppingMatrix, KPath, OnsiteParams, HoppingParams
  lattice.py       — build_system(): creates atoms, unit cell, neighbor table
  neighbors.py     — KD-tree neighbor finding (scipy cKDTree)
  hamiltonian.py   — fill_hamiltonian(): populates HoppingMatrix.H via Slater-Koster
  slater_koster.py — SK hopping integrals (sp 4x4 and spd 9x9)
  onsite.py        — on-site energies, exchange, SOC (8-dim sp-spin + 18-dim spd-spin)
  basis.py         — orbital basis projectors: all types project from unified 18-dim full space
  bloch.py         — get_H_k(), get_H_v(): Bloch sums + velocity operators
  parallel.py      — MPI wrapper (mpi4py with serial fallback)
  calc/
    bands.py              — band structure along k-path
    all_ek.py             — full BZ eigenvalues + DOS
    nonlinear_optical.py  — chi^(2) nonlinear optical susceptibility
    quantum_metric.py     — quantum metric tensor + linear response
    delta_Q.py            — DC field-induced change in quantum geometric tensor
  nrl/
    params.py          — NRL parameter file parser
    hamiltonian_nrl.py — NRL Hamiltonian builder (density-dependent on-site, SOC)
  io/
    xyz.py — XYZ file reader
```

## Key Architecture
- `System` dataclass is the central object: atoms, matrices (list of HoppingMatrix), neighbors, unitcell_vectors, onsite_params, hopping_params
- `bloch.py` iterates `system.matrices` with Bloch phases — do not modify its interface lightly
- **Unified 18-dim full space**: all basis types project from a single 18-dim (spd×spin) space. The hamiltonian always builds 9×9 orbital blocks (via `build_hopping_9x9`) and 18×18 on-site blocks (via `build_onsite_18x18_from_params`), then projects to the active subspace. There is no separate 8-dim code path.
- Two model types: TB_simple (projected from 18-dim) and NRL (spd, 9×9 direct with overlap)
- **Per-atom basis**: each atom in a multi-atom unit cell can have its own basis type (e.g., Mo with `d_u`, S with `p_u`)
- **Per-species parameters**: onsite and hopping parameters can be specified per species; inter-species hoppings are averaged
- Parallelization via `parallel.py`: MPI (mpi4py) when available, serial fallback otherwise

## Running
```bash
# Serial or single-node
python3 -m tightbinding examples/input_qm_test.yaml

# MPI parallel (use --use-hwthread-cpus on this machine)
mpiexec --use-hwthread-cpus -np 8 python3 -m tightbinding input.yaml

# Run a standalone script with MPI
mpiexec --use-hwthread-cpus -np 8 python3 run_MoS2_dQ.py
```
Note: `python` is not available on this system; always use `python3`. The `--use-hwthread-cpus` flag is needed to avoid "not enough slots" errors with mpiexec.

## Implementing Formulas from PDFs
Common workflow: user provides a PDF with derivations, Claude reads the equations and implements them in the calc engine. Key practices:
- Read the PDF carefully; identify the target equation number and all variable definitions
- Map PDF notation to code variables (e.g., ω_{nm} → `de`, r^a_{nm} → `rmtx[a]`, Δ^a_{mn} → `-Delta[a]`)
- Check sign conventions against existing validated code (e.g., `_compute_dk_rmtx` is validated via chi^(2))
- After implementation: test at a single k-point first (check real/imaginary parts, TRS), then run BZ-integrated convergence studies at increasing nk

## Example Configs
- `examples/input_bands_test.yaml` — band structure
- `examples/input_nonlinear_test.yaml` — chi^(2) (2D square sp_u, validated against MATLAB)
- `examples/input_qm_test.yaml` — quantum metric (validated against MATLAB)
- `examples/input_all_ek_test.yaml` — full BZ eigenvalues + DOS
- `examples/input_delta_Q_test.yaml` — delta Q (2D square sp_u with Rashba)
- `examples/input_MoS2_bands.yaml` — MoS2 monolayer 11-band model (Mo d_u + S p_u, Cappelluti params)

## Validated Benchmarks
- Pt NRL band structure: exact match with MATLAB (hopping_range=16.0, SOC on)
- TB_simple 1D chain: verified
- Nonlinear optical chi^(2): all 14 components match MATLAB to ~1e-13 relative error
- Quantum metric Q, dQ, dQf: match MATLAB to ~1e-12

## Delta Q Engine
DC field-induced change in the quantum geometric tensor, `delta_Q.py`. Implements Eq. 40 of `revised_formula_sheet_eta.pdf` with adiabatic iη broadening and three terms:
- **T_Sipe**: dressed-dipole term using generalized derivatives r^{c;a} via Sipe sum rule
- **T_Delta**: velocity-difference term
- **T_3band**: three-band virtual transition term (fully vectorized via matrix products)

The iη broadening enters only the DC perturbation denominators (ω_{nm} → ω_{nm} ± iη), preserving Hermiticity of δP_n. Projector-derivative denominators (v/ω) remain bare.

Output is Sum_n f_n * dQ^{ab}_n accumulated over the k-grid. Config:
```yaml
calc:
  type: delta_Q
  components: [xz, zx]     # explicit (a,b) pairs, or 'all'
  field_direction: x        # DC field direction c (string or list)
  directions: [x, z]        # only needed when components='all'
  nk: [60, 60]
  eflist: [2.0]
  kT: 0.1
  eta: 0.1                  # adiabatic broadening (default 0.0)
```
Uses `_compute_dk_rmtx` (Sipe sum rule) borrowed from `nonlinear_optical.py`. Sign convention: code and PDF both use r = -i*v/w; dk_rmtx[c][a] = r^{c;a} (no sign flip). Reference: `revised_formula_sheet_eta.pdf` (Eq. 40).

## MATLAB Source Reference
Original MATLAB code is in `C:\Users\haney\master_response\` for comparison when porting.

## Conventions
- Input format: YAML
- All numpy arrays; no sparse matrices currently
- SK direction convention: `d = -nb.direction * nb.distance` in hamiltonian.py
- k-grid offset: `tk = -b1/2 - b2/2 + db1*kc1 + db2*kc2` (BZ centering)
- Hopping param naming: `tXY_channel` (e.g., `tss_sigma`, `tsp_sigma`, `tpp_sigma`, `tpp_pi`, `tsd_sigma`, `tpd_sigma`, `tpd_pi`, `tdd_sigma`, `tdd_pi`, `tdd_delta`)
- When validating new features, always compare against MATLAB output numerically

## Hopping Anisotropy (Uniaxial Strain Model)
Traceless phenomenological model for uniaxial strain, implemented in `hamiltonian.py`. All inter-atomic hopping amplitudes t_ij are multiplied by a direction-dependent factor:
- **2D**: 1 + δ·(2cos²φ − 1)  [= 1 + δ·cos(2φ)]
- **3D**: 1 + δ·P₂(cosθ)  [= 1 + δ·(3cos²θ − 1)/2]

where θ/φ is the angle between the bond direction R̂_ij and the strain axis ê. Traceless form ensures the angular average is zero — anisotropy without bandwidth renormalization. Dimensionality auto-detected from lattice vectors (`_detect_ndim`).

Applied uniformly to all SK channels (σ, π, δ) — a known simplification. On-site energies are not modified.

Config (in `hopping` section):
```yaml
hopping:
  anisotropy_factor: 0.1          # δ (strength; positive = enhanced hopping along ê)
  anisotropy_direction: [1, 0, 0] # strain axis ê (auto-normalized to unit vector)
```

Storage: `System.hopping_anisotropy_direction` (unit vector, None when δ=0) and `System.hopping_anisotropy_factor` (float). Parsed in `lattice.py`, applied in `hamiltonian.py` per neighbor pair after `spin_double` and before basis projection.
