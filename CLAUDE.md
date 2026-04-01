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
  onsite.py        — on-site energies, exchange, SOC in 8-dim sp-spin space
  basis.py         — 14 basis types via projection matrices
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
- Two model types: TB_simple (sp, 4x4->8x8 projected) and NRL (spd, 9x9 direct)
- Parallelization via `parallel.py`: MPI (mpi4py) when available, serial fallback otherwise

## Running
```bash
# Serial or single-node
python -m tightbinding examples/input_qm_test.yaml

# MPI parallel
mpiexec -np 4 python -m tightbinding input.yaml
```

## Example Configs
- `examples/input_bands_test.yaml` — band structure
- `examples/input_nonlinear_test.yaml` — chi^(2) (2D square sp_u, validated against MATLAB)
- `examples/input_qm_test.yaml` — quantum metric (validated against MATLAB)
- `examples/input_all_ek_test.yaml` — full BZ eigenvalues + DOS
- `examples/input_delta_Q_test.yaml` — delta Q (2D square sp_u with Rashba)

## Validated Benchmarks
- Pt NRL band structure: exact match with MATLAB (hopping_range=16.0, SOC on)
- TB_simple 1D chain: verified
- Nonlinear optical chi^(2): all 14 components match MATLAB to ~1e-13 relative error
- Quantum metric Q, dQ, dQf: match MATLAB to ~1e-12

## Delta Q Engine
DC field-induced change in the quantum geometric tensor, `delta_Q.py`. Implements the corrected projector formula with three terms:
- **T_Sipe**: dressed-dipole term using generalized derivatives r^{c;a} via Sipe sum rule
- **T_Delta**: velocity-difference term
- **T_3band**: three-band virtual transition term (fully vectorized via matrix products)

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
```
Uses `_compute_dk_rmtx` (Sipe sum rule) borrowed from `nonlinear_optical.py`. Sign convention: code uses chi-convention rmtx = -i*v/w internally; PDF convention r = +i*v/w = -rmtx. Products of two r's have signs cancel. Reference: `corrected_projector_delta_Q_summary_typeset.pdf` (Section 3).

## MATLAB Source Reference
Original MATLAB code is in `C:\Users\haney\master_response\` for comparison when porting.

## Conventions
- Input format: YAML
- All numpy arrays; no sparse matrices currently
- SK direction convention: `d = -nb.direction * nb.distance` in hamiltonian.py
- k-grid offset: `tk = -b1/2 - b2/2 + db1*kc1 + db2*kc2` (BZ centering)
- Hopping param naming: `tXY_channel` (e.g., `tss_sigma`, `tsp_sigma`, `tpp_sigma`, `tpp_pi`, `tsd_sigma`, `tpd_sigma`, `tpd_pi`, `tdd_sigma`, `tdd_pi`, `tdd_delta`)
- When validating new features, always compare against MATLAB output numerically
