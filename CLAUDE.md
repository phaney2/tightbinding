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

## Validated Benchmarks
- Pt NRL band structure: exact match with MATLAB (hopping_range=16.0, SOC on)
- TB_simple 1D chain: verified
- Nonlinear optical chi^(2): all 14 components match MATLAB to ~1e-13 relative error
- Quantum metric Q, dQ, dQf: match MATLAB to ~1e-12

## MATLAB Source Reference
Original MATLAB code is in `C:\Users\haney\master_response\` for comparison when porting.

## Conventions
- Input format: YAML
- All numpy arrays; no sparse matrices currently
- SK direction convention: `d = -nb.direction * nb.distance` in hamiltonian.py
- k-grid offset: `tk = -b1/2 - b2/2 + db1*kc1 + db2*kc2` (BZ centering)
- Hopping param naming: `tXY_channel` (e.g., `tss_sigma`, `tsp_sigma`, `tpp_sigma`, `tpp_pi`, `tsd_sigma`, `tpd_sigma`, `tpd_pi`, `tdd_sigma`, `tdd_pi`, `tdd_delta`)
- When validating new features, always compare against MATLAB output numerically
