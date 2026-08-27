# Tight-Binding Python Code

## ⚠ OPEN BLOCKING BUG — read before running production calculations

`_compute_A_W_k` (the Wannier-gauge position correction,
`calc/nonlinear_optical.py:213-289`) is **broken**. It makes `delta_Q`
complex when it must be real, and gives chi^(2) a non-vanishing
dissipative part below the band gap that survives eta -> 0. Both engines
share the one function, so **chi^(2) and delta_Q results from Wannier
(`_tb.dat`) input are both affected** — including any chi^(2)-vs-delta_Q
comparison, where it contaminates both sides at once.

Full diagnosis, the specific things to check, and the test criteria for a
fix are in **`BUG_wannier_r_correction.md`**. Do not generate production
data from Wannier input until this is resolved. Do not "fix" it by
setting `wannier_r=False` — that is physically wrong, see the note there.

## Working Style: Docs First
Read this file (and any PDF/notes the user points to) BEFORE opening source files.
This CLAUDE.md documents the package structure, config schema, conventions, and
validated benchmarks — most questions can be answered from it directly. Only read
code when the docs are genuinely insufficient (e.g. verifying an exact numerical
convention or debugging). Don't grep the codebase to answer questions the docs
already cover; don't preface an answer with a code survey the user didn't ask for.
If a doc turns out to be wrong or missing something, fix/extend this file.

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
  wannier.py       — Wannier90 _hr.dat / _tb.dat parsers + System builders
  basis.py         — orbital basis projectors: all types project from unified 18-dim full space
  bloch.py         — get_H_k(), get_H_v(): Bloch sums + velocity operators
  parallel.py      — MPI wrapper (mpi4py with serial fallback)
  calc/
    bands.py              — band structure along k-path
    all_ek.py             — full BZ eigenvalues + DOS
    nonlinear_optical.py  — chi^(2) nonlinear optical susceptibility
    nonlinear_optical_fast.py — vectorized chi^(2) path (must track nonlinear_optical.py)
    freq_integral.py      — closed-form int dw w^-p (...) for the frequency-integrated chi^(2)
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

# MPI parallel
mpiexec -np 8 python3 -m tightbinding input.yaml

# Run a standalone script with MPI
mpiexec -np 8 python3 run_MoS2_dQ.py
```
Standalone scripts need `sys.path.insert(0, '.')` or `PYTHONPATH=.` to import
`tightbinding`. The MPI launcher, its accepted flags, and the interpreter name vary by
machine — see "Machine-specific notes" below.

## Machine-specific Notes Go Elsewhere
**This file is tracked in git and is shared across machines. Do not put machine-specific
facts in it** — interpreter paths, MPI implementation and its flags, absolute paths to
external notes/PDFs, scheduler details, or anything containing a home directory. Those
thrash on every push from a different machine.

Instead:
- **Facts about the machine** (which Python, which MPI, shell quirks) → `~/.claude/CLAUDE.md`,
  which Claude Code loads for every project and which lives outside all repos.
- **Facts about this checkout on this machine** (where the dQ notes PDF or the MATLAB
  reference source lives) → `CLAUDE.local.md` in the repo root, which is gitignored.
  `CLAUDE.local.md.example` is the tracked template; copy it on a new machine.

Refer to external documents by *filename* here (e.g. `tmd_warping_note_corrected.pdf`)
and let `CLAUDE.local.md` say where they live.

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
- `examples/input_jdos_test.yaml` — joint density of states (2D square sp_u)
- `examples/input_nonlinear_freq_integral.yaml` — analytic frequency-integrated chi^(2)
- `examples/input_MoS2_bands.yaml` — MoS2 monolayer 11-band model (Mo d_u + S p_u, Cappelluti params)

## Validated Benchmarks
- Pt NRL band structure: exact match with MATLAB (hopping_range=16.0, SOC on)
- TB_simple 1D chain: verified
- Nonlinear optical chi^(2): validated against an **analytic pole-sum**, not MATLAB — see
  "chi^(2) no longer matches MATLAB" below. `chi_total` equals an independently derived exact
  sigma^{abc}(w;w,0) at ratio +1.0000 for all 8 components on a synthetic random 4-band System
  (rel. err 1e-8); slow and fast paths agree to 5e-16. MoS2 D3h-forbidden components (xxx, xyy,
  yxy, yyx) are machine-zero (1e-13) and chi_yyy = -chi_xxy = -chi_xyx = -chi_yxx exactly.
- Quantum metric Q, dQ, dQf: match MATLAB to ~1e-12 (predates the periodic-grid change; the
  pointwise agreement is unaffected, but BZ-integrated totals shift by O(1/nk))
- Gapped graphene / trigonal warping (`examples/honeycomb_warp/`): reproduces the warped-valley
  model of `tmd_warping_note_corrected.pdf` (Eq. 1) and its delta_Q predictions — see below

## chi^(2) No Longer Matches MATLAB (deliberately)
As of `8b5a079`, the chi^(2) engine **diverges from the MATLAB original by design**. The port
reproduced MATLAB to 1e-13, which established that MATLAB carries the same three bugs:

1. **Missing intraband vertex factor of i.** The length-gauge coupling is E·(r_e + i d/dk), but
   the derivative vertex omitted the i. `chi_ei*`/`chi_ie*` were off by (-i) and `chi_ii` by -1;
   `chi_ee` (no derivative vertex) was exact. An inconsistent i between term families rotates Re
   into Im differently per family, which is what broke the chi^(2) ↔ static-field quantum
   geometry relation.
2. **Triple-pole sign (t5/t6).** d/dk_c [1/(w - w_nm)] = +Delta^c/(w - w_nm)^2; the code had a
   leading minus.
3. **Advanced instead of retarded broadening.** Denominators used (w - w_nm - i eta), giving
   negative absorption (Re sigma^(1)_xx < 0 at the interband peak). Now +i eta throughout the
   response denominators. The Souza `eta_sos` regularization of the bare 1/w_nm is untouched.

**Do not "fix" a MATLAB mismatch in chi^(2) by reverting to MATLAB.** Validate against the
analytic pole-sum instead. Term-family ratios against the exact result (constant +i / -i / +1
per family, uniform in frequency and direction) were the diagnostic that found this.

**Consequence:** chi^(2) data generated before `8b5a079` is invalid and must be regenerated, and
term-decomposition conclusions (chi_ei vs chi_ee balance) re-derived.

## Gapped Graphene = Warped Valley Model
`examples/honeycomb_warp/` implements Eq. 1 of `tmd_warping_note_corrected.pdf`
as a honeycomb lattice of `s_u` orbitals with staggered on-site energy. NN-only hopping
reproduces the k·p Hamiltonian exactly to O(k^2) at both valleys:

    v = 3*T*a/2,   lambda = -3*T*a^2/8,   m = (u_s^A - u_s^B)/2
    (a = NN distance, T = tss_sigma; lambda/v = -a/4 is LOCKED by NN geometry)

**Orientation matters:** the note's T*M_x symmetry requires M_x to swap valleys, i.e. Gamma-K
along x-hat. Achieve this by putting the A-B bond along **y**-hat. The naive bond-along-x setup
is rotated 90 deg and swaps which response chain survives.

Because lambda/v is fixed, the dimensionless warping parameter is lambda*m/v^2 = m/(6T) — so
**small gap relative to hopping** is what puts you in the note's perturbative regime.
Eq. 15 then collapses to a hopping-independent prediction, both valleys added:

    T^yyy = int d2k delta_g^yy(E_y) = pi*a/(4m)

Note `compute_delta_Q` returns the BZ *average*; multiply by BZ area = (2pi)^2 / (3*sqrt(3)/2 a^2).

Scripts: `check_kp.py` (verifies the k·p mapping against lattice H(k)), `check_eq13.py`
(pointwise check vs. note Eqs. 7/13/14), `run_dQ.py` (BZ sweep; `mpiexec -np 8`).
Results at nk=801: x-odd channels vanish to ~1e-16; chain yyy = -yxx = -xxy = -xyx exact;
|T^yyy| = 15.714 vs. 15.708 predicted at m=0.05 (0.04%), 7.900 vs. 7.854 at m=0.1 (0.6%),
with the residual shrinking as m/T -> 0 as expected.

⚠️ **Those numbers predate the periodic-grid fix (`4381228`) and have not been re-run.** They
were produced with `db = b/(nk-1)`, which double-counts both zone edges — an O(1/nk) bias. At
nk=801 the shift should be ~0.1%, i.e. comparable to the m=0.05 residual quoted above, so the
agreement is expected to hold but the digits will move. Re-run `run_dQ.py` before citing them.

## Frequency-Integrated chi^(2) (analytic)

`calc.type: nonlinear_optical` has a second mode that returns

    J = int_{omega_min}^{omega_max}  domega  omega^(-p)  chi^(2)(omega)

from **closed-form antiderivatives**, not quadrature. Selected by replacing
`omega1list` with a `freq_integral` block (giving both is an error):

```yaml
calc:
  type: nonlinear_optical
  freq_integral:
    p: [1, 2]          # scalar or list; one output column per power
    omega_min: 0.01    # optional, defaults to 10*eta
    omega_max: .inf    # optional, default
    diagnostics: true  # optional, default
  eta: 1.0e-3
```

Output arrays are `(nef, n_p)` instead of `(nef, nomega)`; the nesting
`result[name][a][b][c]` and the `.npz` save/load are unchanged. Reported in the
**code's chi convention**, same as the sampled path — so `Im[J]` is reactive and
`-Re[J]` dissipative (see "Output convention carries a factor of i" below).

**Why it exists:** matching the closed form by quadrature costs ~1e6 omega points
per k-point. The s=3 kernels have an integrand peaking at ~1/eta^3 that integrates
to O(1), i.e. ~1e7 cancellation between the lobes of the peak, and below ~4 mesh
points per resonance width the error jumps to 1e5 relative rather than degrading
gracefully. Adaptive quadrature is *worse*, not better — `scipy.integrate.quad`
reached 2.2e+06 relative error on those cases even with `points=` at the pole.

### The omega_min window is two-sided and often empty

    ~10 * eta  <<  omega_min  <<  E_gap

The lower bound is not cosmetic. `s_om1 = 1/(omega + i eta)` and
`s_om12 = 1/(omega + omega2 + 2i eta)` put poles at `-i eta` and `-2i eta`, right
on top of the `omega^(-p)` endpoint. The integrand changes character at
`omega ~ eta` — above it those terms go as `omega^(-p-1)`, below it they flatten
to `omega^(-p)/(i eta)` — so an `omega_min` inside the crossover makes J a
function of the eta/a interplay rather than of the physics. Enforced:
**hard error below 5*eta**, warning below 10*eta, warning above 0.1*E_gap
(the gap is estimated on a coarse pre-pass grid). `omega_min` defaults to 10*eta.

Both bounds together need `eta << E_gap/10`. The gapped honeycomb at
gap = 0.2, eta = 0.025 has **no valid window at all** — that is why the eta = 1e-3
scale is the working one for these models, not the eta = 0.025 of the earlier
exercise. `eta = 0` is rejected outright.

### Convergence at the upper limit

`b = inf` is numerically *better* than a large finite b: the simple-pole residues
sum to zero whenever `p + sum(s_i) >= 2`, so the antiderivative vanishes at
infinity identically and `J = -F(a)` with no large-b cancellation. But that
condition also decides what converges:

- **p >= 2** — every term converges.
- **p = 1** — everything converges except `chi_e1` and `chi_i1`, which are
  omega-independent (`sum(s_i) = 0`). They are returned as **NaN** with a
  printed note; both are unphysical and already excluded from `chi_total`.
- **p = 0** — the single-pole terms log-diverge individually. Needs a finite
  `omega_max`.

### Lower-endpoint diagnostics

Each term carries a divergence as `omega_min -> 0`. The engine BZ-accumulates the
coefficients alongside J and emits them for the physical terms and their total:
`endpt_log_<name>` (coefficient of `ln(omega_min)`) and `endpt_pow<j>_<name>`
(coefficient of `omega_min^(1-j)`, p >= 2), same `[a][b][c]` nesting, shape
`(nef, n_p)`.

Two exact identities make these self-checking, both verified to machine precision:

    p = 1:  coefficient of ln(omega_min)      = -chi^(2)(omega = 0)
    p = 2:  coefficient of omega_min^(-1)     = +chi^(2)(omega = 0)
            coefficient of ln(omega_min)      = -d chi^(2)/d omega (0)

**Consequence: at p = 1 the integral is genuinely log-divergent as
`omega_min -> 0`, with coefficient `-chi^(2)(0)`.** It is not omega_min-independent
and no amount of summing makes it so. On the honeycomb at m = 0.1, the
`chi_ii`/`chi_ie`/`chi_ee` endpoint coefficients cancel to ~1e-15 and the whole
residue sits in `chi_ei1 + chi_ei2`, equal to the static response. Quote J only
together with the `omega_min` it was computed at.

### Conditioning

The pole pairs are separated by exactly `i eta` (at omega2 = 0), so they are nearly
coincident on the scale over which the integrand varies. The partial-fraction
`1/(z_1 - z_2)` factors cancel against each other, costing roughly
`log10(|z|/eta)` digits per unit of excess multiplicity. The engine prints the
worst cancellation ratio at the end of a run; ~4e7 (≈8 digits lost in the worst
kernel) at eta = 1e-3 on the honeycomb. Measured end-to-end accuracy there is
still ~1e-8, i.e. two orders better than the 1e-6 quadrature ceiling.

### Implementation

`freq_integral.py` exposes `rational_integral(p, poles, mults, a, b)`, a general
routine over a list of (pole, multiplicity) pairs — not the hard-coded p=1,
s=1..3 forms. Coefficients come from Taylor expansions of the other factors about
each pole, which is equivalent to the closed-form generalized binomial but needs
no special case for negative binomial arguments. Logs are evaluated as
differences of principal logs, never as `log((b-z)/(a-z))` (that can cross the
cut); this is safe because every pole has `Im z < 0`.

`_build_omega_kernels` in `nonlinear_optical_fast.py` is the single place the
omega-dependent factors are formed, with sampled and integrated backends. Phase 3
is shared verbatim, so the two modes cannot drift. `chi_ee1` is the one term whose
two omega factors sit at *different* band pairs; it uses a (D,D,D,W) kernel in both
modes, which reassociates its floating-point sum (8e-15 vs the pre-refactor code —
every other term is bit-identical).

Tests: `examples/test_freq_integral.py` — the note's reference values, formula vs
dense log-mesh, the near-coincident pole pair at eta down to 1e-3, `b = inf`,
endpoint coefficients, end-to-end vs a mesh over the sampled engine, the
`chi(0)` identities, and a sampled-mode regression against a `git worktree`
baseline (pass the baseline path as argv[1]). Example config:
`examples/input_nonlinear_freq_integral.yaml`, which also reproduces the D3h
invariant `chi_yyy = -chi_yxx = -chi_xxy = -chi_xyx` to 1e-12 on the integrated
output, with the forbidden components at 1e-11 relative.

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

**Overall sign vs. the +E·r convention.** Eq. 40 as implemented corresponds to H' = -E·r. Notes written with H' = +E·r (e.g. `tmd_warping_note_corrected.pdf`, Eq. 7) give the *opposite* overall sign. Verified pointwise: `delta_Q` output = -1.000000 x (that note's Eq. 7) at every k and every (a,b,c) channel. Magnitude and all structure agree to machine precision — only the global sign differs.

**Gotchas for small-gap models.** `eta_sos` defaults to 0.05, which is comparable to (or larger than) the gap in low-energy models — set it to ~1e-8 whenever bands are never degenerate. Also set `eta: 0.0` to compare against unbroadened analytic results, and `kT` well below the gap.

**Position operator gauge.** `bloch.py` builds H(k) in the *atomic gauge*, exp(ik·(R + tau_j - tau_i)) via `atompos`. In that gauge the TBA position operator is exactly r = -i*v/w with zero intra-cell Wannier correction, so `_compute_A_W_k` correctly returns None for TB_simple systems even in multi-atom cells. k·p expansions of `get_H_k` output should therefore be done in the atomic gauge too.

## MATLAB Source Reference
The original MATLAB code (`master_response`) is the reference when porting and validating.
Its location is machine-specific — see `CLAUDE.local.md`.

**It is not infallible.** chi^(2) has deliberately diverged from it (see "chi^(2) No Longer
Matches MATLAB"). Treat MATLAB as the reference for *porting* — establishing that a new Python
engine reproduces the intended algorithm — but not as ground truth for *physics*. Where an
analytic limit, sum rule, or symmetry argument is available, that wins.

## Conventions
- Input format: YAML
- All numpy arrays; no sparse matrices currently
- SK direction convention: `d = -nb.direction * nb.distance` in hamiltonian.py
- k-grid offset: `tk = -b1/2 - b2/2 + db1*kc1 + db2*kc2` (BZ centering), `kc = 0..nk-1`
- **k-grid spacing: `db = b/nk`, NOT `b/(nk-1)`.** The periodic (endpoint-free) grid tiles the
  BZ exactly once, so the uniform `1/(nk1*nk2*nk3)` weight is an unbiased BZ average. Using
  `b/(nk-1)` samples both zone edges — the boundary lines are duplicated while the
  normalization is unchanged, an O(1/nk) systematic error in every integrated quantity.
  Applies to `delta_Q.py`, `quantum_metric.py`, `nonlinear_optical.py`, `nonlinear_optical_fast.py`
  (fixed in `4381228`). **`all_ek.py` still uses `b/(nk-1)`** and was not touched by that commit
  — whether that is intentional or an oversight is unresolved. Its DOS is a BZ integral and so
  is subject to the same edge-double-counting bias; check before trusting normalized DOS values.
- Never exclude k-points from a grid to dodge degeneracies — it breaks the crystal point-group
  symmetry of the sampling. chi^(2) used to drop the `kc1 = 0`/`nk1-1` rows and `kx = ±pi`
  points, which made the D3h-forbidden MoS2 components come out at up to ~16% of the allowed
  ones. The Souza `eta_sos` regularization bounds the near-degenerate 1/w_nm factors instead.
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
