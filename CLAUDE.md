# Tight-Binding Python Code

## Wannier-Gauge Correction — RESOLVED 2026-09-10

The Wannier-gauge position correction (`calc/wannier_gauge.py`) was broken
from its introduction until 2026-09-10: it made `delta_Q` complex and gave
chi^(2) an O(1) dissipative part below the gap. `BUG_wannier_r_correction.md`
now opens with the resolution (five separate defects, what was done, test
numbers) and keeps the original report as the record. `system.wannier_r`
defaults to `true` again. **chi^(2), delta_Q and quantum_metric data from
`_tb.dat` input generated before that date must be regenerated**, and so
must every *metallic* `chi_ii` from any input (defect 5 there). The
enforcing test is `examples/test_wannier_gauge.py`; see "Wannier-Gauge
Position Correction" below for the rules it encodes.

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
    wannier_gauge.py      — Wannier-gauge position correction: the one switch + the one kernel
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
- `examples/input_delta_Q_metal.yaml` — delta Q of a *metal* (thermal formulation + RTA piece)
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
- Wannier-gauge position correction: an exact **gauge-covariance certificate**
  (`examples/test_wannier_gauge.py`, 41 checks). A point-like multi-atom model re-expressed
  in another gauge (atompos = 0, or true centres + random offsets) plus the correction must
  reproduce its atomic-gauge answer: it does, to 1e-13 for r and r^{a;b} and 1e-14 for every
  engine's per-k output (chi^(2) fast/reference, delta_Q thermal/subspace/band + RTA,
  quantum_metric Q/dQf); the no-correction control fails by 1-100%. The k-dependent
  U†(∂A^W)U term is checked against finite differences on the real MoS2 connection at 1e-7
  (h² convergence). On MoS2 with the correction on: delta_Q real to 1e-9 (thermal), thermal
  == subspace to 1e-6, sub-gap Re chi^(2) ∝ eta (ratio 0.1275 over an 8x ladder vs 0.125).
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

4. **`chi_ii` band curvature (fixed 2026-09-10).** The second k-derivative of the occupation
   used the diagonal of ∂²H as ∂_b∂_c E_n, omitting the interband sum rule
   2 Re Σ_m v_nm v_mn / w_nm — which is also what makes the curvature gauge invariant (found by
   the gauge-covariance certificate). No effect on insulators, where chi_ii = 0 below the gap;
   every *metallic* chi_ii before the fix is wrong. Inherited from MATLAB.

**Do not "fix" a MATLAB mismatch in chi^(2) by reverting to MATLAB.** Validate against the
analytic pole-sum instead. Term-family ratios against the exact result (constant +i / -i / +1
per family, uniform in frequency and direction) were the diagnostic that found this.

**Consequence:** chi^(2) data generated before `8b5a079` is invalid and must be regenerated, and
term-decomposition conclusions (chi_ei vs chi_ee balance) re-derived.
Metallic chi_ii data generated before 2026-09-10 is invalid as well (item 4).

**Output convention.** `chi_total` equals the second-order *conductivity* sigma^{abc}(w; w, 0):
**Im is the reactive part and Re the dissipative part** (for an insulator below the gap,
Re → 0 linearly in eta and Im → a constant). Test 2 of the original bug report was written
the other way round; do not repeat that.

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

### The chi^(2) <-> delta_Q sum rule (checked 2026-09-10)

    int_0^inf dw (1/w) Re chi^{abc}(w; w, 0)  =  pi * dQ^{ab}(c)

with Re the dissipative part (chi_total is the conductivity) and dQ from
`delta_Q` (thermal) on the same grid, same kT and eta_sos. Driver:
`examples/check_sumrule_chi2_dQ.py` (all eight in-plane components, several
index pairings, synthetic-connection models; `mpiexec` for MoS2). Findings:

- **The pairing is the plain one.** Output a, omega-field b, DC-field c go with
  dQ^{ab}(c). Established on the anisotropic honeycomb (C3 broken by 30%), where
  every independent component obeys it and the alternative pairings fail by 15-20%.
  yyy alone cannot distinguish pairings.
- **Finite eta costs O(eta ln(1/eta)).** J/(pi dQ) = 1.0142, 1.0066, 1.0029, 1.0011 on
  the honeycomb at eta = 1e-3, 3e-4, 1e-4, 3e-5 (gap 0.2); 1.00133, 1.00074, 1.00041
  on MoS2 yyy at eta = 1e-3, 5e-4, 2.5e-4 (gap 1.76). Both fit A eta ln(1/eta) + B eta
  with **no constant term**, so the eta -> 0 limit is 1 to better than 1e-4. The source
  is the 1/(w + i eta)-type poles under the 1/w weight (the same crossover as the
  `omega_min` window). Quote the sum rule with its eta, or extrapolate; the "7e-6" in
  `NOTE_wannier_r_flag_consistency.md` is not a finite-eta number. Independent of nk
  once converged (nk 120/240/480 identical to six digits on the honeycomb).
- **With the Wannier correction on, the rule holds component-wise only for a = c.**
  MoS2 (nk 200 and 300, any eta, eta_sos 0.025 or 1e-3): yyy, xyx, xxx at the eta
  offset; xxy at 0.9786 and yxx at 1.0243, i.e. a deviation antisymmetric under a <-> c
  whose sum obeys the rule. Not symmetry breaking of the Hamiltonian (correction off,
  all eight components obey it on the same unsymmetrized file), not time reversal (the
  in-plane position blocks are real to 6e-9; a real synthetic decoration with exact TRS
  shows the same violation). It is the **non-Abelian curvature of the Wannier
  connection**, F^{ac} = d_a A_c - d_c A_a - i[A_a, A_c]: a synthetic connection with
  F = 0 exactly (`honeycomb_wflat`) obeys the rule in all components, one of the same
  size with F != 0 (`honeycomb_wcurv`) reproduces the antisymmetric violation. The
  projected position components of a truncated Wannier basis do not commute, and the
  sum-rule derivation exchanges r^{a;c} for r^{c;a}, which costs an F^{ac} term. On MoS2
  |F^{xy}| is 0.4-2.8 A^2 in the Hamiltonian gauge and the effect is 2.3%. The
  (a <-> c)-symmetrized relation holds at the eta offset for every pair. For point-like
  orbitals F = 0 identically, which is why the honeycomb never showed it. Decided
  2026-09-10: no symmetrized redefinition, no explicit derivation of the F term.

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
DC field-induced change in the quantum geometric tensor, `delta_Q.py`. Three formulations,
selected by `calc.formulation`:

- **`thermal` (default)** — δQ^{ab}_T of the thermal density matrix ρ = f(H), valid for
  **metals at finite temperature**. Reference: `delta_Q_metal_finite_T.pdf` (Eq. eq:final;
  location in `CLAUDE.local.md`). Sharp occ/un masks are replaced by smooth weights
  W_pq = f_p f_pq², F_pq = f_pq/ω_pq (divided difference, → f'(E) at degeneracies, guarded
  below |ω| = 1e-7). All f' (Fermi-surface) terms cancel identically — proven in the note,
  verified in the test — so the intrinsic response is purely interband even in a metal.
  New `T_loop` term (triple sum, vanishes for T=0 insulators); the insulator dipole/mix
  three-band split merges into a single `T_3band`. **No adiabatic iη** on this path —
  finite kT is the regulator (`eta` is ignored with a note). Weights are assembled as
  W/ω = f_p·f_pq·F and W/ω² = f_p·F², so no occupation weight carries a bare 1/ω; the
  only 1/ω lives in r and r^{c;a} (Souza `eta_sos`, as everywhere).
  Converge nk together with kT: the Fermi-surface structure sharpens as kT → 0.
- **`subspace`** — T=0 occupied-projector formulation (`delta_Q_occ_derivation`), Pauli
  mask f_p(1-f_q), extra `T_mix` term. Reproduced by `thermal` to exponential accuracy on
  insulators at βE_gap ≫ 1 (verified: 2e-16 at βE_gap = 20).
- **`band`** — band-resolved Σ_n f_n δQ^{ab}_n, Eq. 40 of `revised_formula_sheet_eta.pdf`,
  with adiabatic iη in the DC perturbation denominators only.

Legacy key `dQ_occupied_subspace: true/false` still maps to `subspace`/`band`; giving both
keys is an error.

**RTA transport piece — always computed on the thermal path, PER UNIT τ.** The extrinsic
τ-linear Fermi-surface response δQ^{ab}_τ (shifted Fermi sea δρ = τ f' v^c, note
Eq. eq:dQtau) is exactly linear in τ, so the engine reports the *coefficient*
(`delta_Q_tau` = δQ_τ/τ; multiply by your relaxation time in post). It is O(N²) —
free next to the O(N³) intrinsic assembly — so there is no switch; in insulators it
comes out exponentially zero, which doubles as a sanity check. Never summed into
`delta_Q` (it diverges in the clean limit; report the two pieces separately as in the
nonlinear-Hall literature). A `calc.tau` key **raises**, so nobody mistakes the output
for having a τ factor applied. In a TR-symmetric metal the BZ-integrated *metric* part
of δQ_τ vanishes (Berry-curvature-dipole channel only).

Config:
```yaml
calc:
  type: delta_Q
  formulation: thermal      # default | subspace | band
  components: [xz, zx]      # explicit (a,b) pairs, or 'all'
  field_direction: x        # DC field direction c (string or list)
  directions: [x, z]        # only needed when components='all'
  nk: [60, 60]
  eflist: [2.0]
  kT: 0.1
  eta: 0.1                  # adiabatic broadening (band/subspace only)
  save_kresolved: false     # also keep the per-k integrand (default false)
```
Uses `_compute_dk_rmtx` (Sipe sum rule) borrowed from `nonlinear_optical.py`. Sign
convention: code and PDFs use r = -i*v/w; dk_rmtx[c][a] = r^{c;a} (no sign flip).

**k-resolved output (`save_kresolved`).** Off by default. On, the result dict and the
`.npz` gain `delta_Q_k[a][b][c]` of shape `(nk1*nk2, nef)` — the **unweighted** per-k
integrand, so `delta_Q_k.mean(axis=0) == delta_Q` (verified to 6e-19 on
`input_delta_Q_test.yaml`, where the per-k values are O(1) and the average is a 1e-17
symmetry zero) — plus `kpoints` `(nk1*nk2, 3)` Cartesian and `nk_grid` `[nk1, nk2]`.
The k axis is C-ordered (kc1 outer, kc2 inner), so `dQk[:, ief].reshape(nk1, nk2)` is
the map and `kpoints.reshape(nk1, nk2, 3)` its axes. Only the total is stored, not the
term decomposition or the RTA piece. MPI-safe: `parallel.gather_array` undoes the
round-robin scatter, and 4 ranks reproduce serial bit-for-bit. The gather puts a full
copy on **every** rank — the engine prints the size and warns above 512 MiB. A
non-boolean value raises. With the flag off, output is unchanged (verified: the
`delta_Q`/`delta_Q_tau` arrays and the npz key set are identical).

`main.load_delta_Q` also reads back `delta_Q_terms` now; its 5-part keys were written
but silently dropped by the loader before.

**Wannier input.** All three formulations use the same corrected position operator: the
`subspace`/`band` pair integrands and `T_mix` are written in terms of the full interband
r (their bare part is exactly v·inv_de), and `thermal` was always written in r. With the
correction on, thermal == subspace to 1e-6 on MoS2 (they disagreed by 10% before 2026-09-10
because subspace used the bare v/w). The `T_Sipe_*` split — including `T_Sipe_wannier_corr`
— is gauge-dependent by construction; only the sum is physical. On MoS2 the correction is a
27% effect on dQ_yyy (+0.090 on -0.336).

**Thermal-path sign.** `delta_Q_metal_finite_T.pdf` derives with H' = +E·r; the engine's
established convention is H' = -E·r, so the thermal and RTA assemblies carry an overall
factor of **-1** relative to that note (`_assemble_delta_Q_thermal`, `_assemble_delta_Q_rta`).
Verified: thermal/subspace ratio = +1.000000 on the gapped honeycomb.

**Validation** (`examples/test_dQ_thermal.py`, 19 checks): trace assembly
Tr I+II+III == collected formula at 2e-15 on a random 5-band metal (incl. all f' pieces,
so the cancellation theorem is exercised, not assumed); field+k finite-difference
certificate at 4e-7; RTA vs operator traces at 2e-16, TR selection rule to machine
precision; insulator limit == subspace at 2e-16.

**Overall sign vs. the +E·r convention.** Eq. 40 as implemented corresponds to H' = -E·r. Notes written with H' = +E·r (e.g. `tmd_warping_note_corrected.pdf`, Eq. 7) give the *opposite* overall sign. Verified pointwise: `delta_Q` output = -1.000000 x (that note's Eq. 7) at every k and every (a,b,c) channel. Magnitude and all structure agree to machine precision — only the global sign differs.

**Gotchas for small-gap models.** `eta_sos` defaults to 0.05, which is comparable to (or larger than) the gap in low-energy models — set it to ~1e-8 whenever bands are never degenerate. Also set `eta: 0.0` to compare against unbroadened analytic results, and `kT` well below the gap.

**Position operator gauge.** `bloch.py` builds H(k) in the *atomic gauge*, exp(ik·(R + tau_j - tau_i)) via `atompos`, for every input: atom coordinates for TB_simple, Wannier centres for `_tb.dat` (always, since 2026-09-10). For point-like orbitals the TBA position operator in that gauge is exactly r = -i*v/w with zero intra-cell term, so `compute_A_W_k` correctly returns None for TB_simple systems even in multi-atom cells; for Wannier functions the finite-spread part is what the correction adds (see "Wannier-Gauge Position Correction" below). k·p expansions of `get_H_k` output should therefore be done in the atomic gauge too.

## Wannier-Gauge Position Correction (`system.wannier_r`)

One switch, one kernel, one application function, in `calc/wannier_gauge.py`. Every engine
that builds a position operator reads the flag through `resolve_wannier_r(cfg, system,
engine)`, builds the connection through `compute_A_W_k(system, k, dir_chars, enabled=...)`,
and applies it through `apply_wannier_correction(A_W, dA_W, psi, rmtx, dir_chars, vmtx=,
de_mtx=)`. **Add an engine with an `r` operator and it must go through all three.**

```yaml
system:
  wannier_tb: mos2_tb.dat
  wannier_centres: mos2_centres.xyz   # recommended — see the loader note
  wannier_r: true         # <- here, NOT under calc:  (default)
```

**What the kernel computes.** In the gauge `bloch.py` uses (Bloch phases with the centres
tau from `atompos`), the Wannier-gauge Berry connection is

    A^W_{nm,a}(k) = Σ_R exp(ik·(R + tau_m - tau_n)) <0n|r_a|Rm>  -  tau_{n,a} δ_nm

Eq. 20 of arXiv:1804.04030 is the tau = 0 case. The subtracted centres are *derived from
`atompos`*, so H(k) and A^W(k) are in one gauge by construction. For point-like orbitals at
tau the sum is exactly tau_n δ_nm and A^W = 0.

**What gets corrected, and what must not be.** With Ā = U†A^W U, a = offdiag(Ā),
ξ = diag(Ā), rbar = -i v/w:

| operator | corrected form | used by |
|---|---|---|
| interband r | rbar + a (Eq. 22) | every engine |
| generalized derivative r^{a;b} | TB Sipe sum rule + U†(∂_b A^W_a)U − i[a_a, rbar_b] − i(ξ^b_nn−ξ^b_mm)(a^a+rbar^a) − i(ξ^a_nn−ξ^a_mm) rbar^b | chi^(2), delta_Q |
| current vertex v (interband) | vbar_nm + i w_nm a_nm  (v = i[H, r] with the full r) | chi^(2) |
| projector-derivative factors v/w | i r (full) | delta_Q subspace/band |
| perturbation vertex | full r | quantum_metric |
| Delta = v_nn − v_mm, band curvature, inverse mass, the Sipe sum rule's own v's | **bare** v | all |

The rule that decides the table: **every engine output must be independent of the gauge the
system was built in**. `examples/test_wannier_gauge.py` enforces exactly that (see Validated
Benchmarks). The previous implementation had the commutator without its factor of i, the
ξ·a term with the wrong sign, the ξ·rbar terms missing, and corrected r but not v — the
subject of `BUG_wannier_r_correction.md`.

**The loader (`wannier.build_system_from_tb`)** conditions the `_tb.dat` position blocks
before storing them, printing one diagnostic line (two if it had to repair something):
- symmetrizes r(R) ← ½[r(R) + r(−R)†] (Wannier90 writes the finite-difference Eq. 44 of
  Wang et al. raw; `postw90` Hermitianizes too);
- always builds `atompos` from the Wannier centres — the `_centres.xyz` file if given, else
  the band-diagonal of the R=0 block — so `_tb.dat` input runs in the atomic gauge;
- compares that diagonal with the centres and replaces it where they differ. The file's
  <0n|r|0n> comes from a Berry-phase log and can be wrapped by a lattice vector or scrambled
  outright when a centre sits on the branch cut (the MoS2 file: Mo at z = c/2 with one
  k-point along c, so its z entries are garbage while x, y are fine). The band-diagonal
  R≠0 elements of such a component come from the same log: **do not use the z position
  operator from that file.** Pass the centres file; it is not optional in practice.

**Default is `True`.** `False` is the point-like-orbital approximation r = -i v/w in the
atomic gauge, a diagnostic; both settings print a one-line note. TB_simple and `_hr.dat`
input carry no position matrices, so the flag is inert there (the banner says so).

**Errors, not silent drops.** `calc.wannier_r` raises (it lived there before the switch was
unified); a non-boolean value raises. `config.load_config` runs the same validator.

**Known gaps, documented rather than fixed:**
- `quantum_metric`'s finite-difference `dQ` perturbs the states with the full r but is a
  bare-velocity heuristic overall and is **not gauge-covariant** with the correction on (17%
  on the test models); `Q` and `dQf` are. The engine prints a note. Use `calc.type: delta_Q`.
- The Souza `eta_sos` regularization acts on the bare -i v/w only, so results are
  gauge-covariant only to O(eta_sos²/w²) × (intra-cell term) near degeneracies — one more
  reason the atomic gauge is now the only gauge `_tb.dat` input runs in.
- `calc.method: projector` rebuilds `chi_e1`/`chi_e2` from H(k) projectors, which carry no
  correction; both are unphysical and excluded from `chi_total`. The engine prints a note.
- `_process_kpoint_fast(_k_data=...)` bypasses the Phase-1 operator build, so the flag has
  no effect on that path.

**Regression status.** TB_simple input never enters the correction; `input_qm_test.yaml`,
`input_nonlinear_test.yaml` and `input_delta_Q_test.yaml` are **bit-identical** to the
pre-fix code. `_tb.dat` results change (that is the point); `wannier_r: false` results on
`_tb.dat` input *without* a centres file change too, because they were in the lattice
gauge before. Tests: `examples/test_wannier_gauge.py` (the certificates; pass the MoS2
directory as argv[1] to include the real-data section) and
`examples/test_wannier_r_flag.py` (31 checks on the switch itself).

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
