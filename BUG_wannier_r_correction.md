# BUG: Wannier-gauge position correction `_compute_A_W_k`

**Status:** RESOLVED 2026-09-10. Found 2026-08-25 while validating the
chi^(2) <-> static-field quantum-geometry relation on MoS2. The original
report is kept below, unchanged, as the record; the resolution comes first.

---

## Resolution (2026-09-10)

### What was actually wrong

Five separate defects, not one. The original report's checklist item 3
("the derivative `dA_W`") was fine; items 1, 2 and 4 each turned out to be
real but not in the way guessed.

1. **The Eq. 36 assembly at the call sites was wrong** (not the kernel).
   `nonlinear_optical.py`, `nonlinear_optical_fast.py` and `delta_Q.py`
   each carried the same copy-pasted block. The covariant derivative of the
   corrected r = Abar + iD is, with a = offdiag(Abar), xi = diag(Abar),
   rbar = -i v/w:

       corr^{a;b} = U^dag(d_b A^W_a)U - i[a_a, rbar_b]
                    - i(xi^b_nn - xi^b_mm)(a^a + rbar^a)_nm
                    - i(xi^a_nn - xi^a_mm) rbar^b_nm

   The code had the commutator WITHOUT the factor i (anti-Hermitian, and on
   MoS2 five times larger than its Hermitian part — this is the complex
   delta_Q), the xi.a term with the opposite sign, and both xi.rbar terms
   missing. Proven by an exact gauge-covariance certificate (below), not by
   comparison to a reference number.
2. **Only r was corrected; the velocity was not.** The physical current
   vertex is v = i[H, r] with the full r, i.e. v_nm = vbar_nm + i w_nm a_nm.
   chi^(2) used the bare v as its output vertex, the `subspace`/`band`
   delta_Q formulations used bare v/w as projector-derivative factors, and
   quantum_metric perturbed its states with the bare v. Only the `thermal`
   delta_Q path (written in terms of r throughout) was consistent — which is
   why `thermal` and `subspace` disagreed by 10% on MoS2 with the correction
   on, and agree to 1e-6 now.
3. **The `_tb.dat` position blocks are not Hermitian-paired.** Wannier90's
   `write_tb` writes Eq. 44 of Wang et al. raw; `postw90` takes the
   Hermitian part before use (`get_oper.F90`, with a comment saying why).
   On `mos2_tb.dat` |r(R) - r(-R)^dag| reaches 1.3e-2 A, so A^W(k) was
   non-Hermitian at 4e-2 A. The loader now symmetrizes.
4. **The R=0 diagonal handling assumed atompos == centres.** Zeroing that
   diagonal is right only in the atomic gauge with atompos built from the
   true centres. Without a centres file the Bloch sum ran in the lattice
   gauge and the required tau_n d_nm term was dropped — an error of
   offdiag(U^dag tau U) ~ 0.5 A in-plane on MoS2, the size of the genuine
   correction. Worse, the file's R=0 diagonal itself is unreliable in z for
   this MoS2 (top-S entries wrapped by one lattice vector; the five Mo-like
   entries scrambled by the Berry-phase branch cut, because Mo sits at
   z = c/2 with one k-point along c). The loader now always builds atompos
   from the centres (file if given, else the R=0 diagonal), compares the
   two, repairs the diagonal, and warns naming the affected component.
5. **`chi_ii` used the diagonal of d^2H as the band curvature.** Found by
   the same certificate, unrelated to Wannier input: d_b d_c E_n needs the
   interband sum rule 2 Re sum_m v_nm v_mn / w_nm, which is also what makes
   it gauge invariant. Invisible in insulators (chi_ii = 0 below the gap);
   wrong for every metallic chi_ii computed before, in any input format.
   Inherited from the MATLAB original.

### What was done

- `calc/wannier_gauge.py`: `compute_A_W_k` subtracts the centres implied
  by `atompos` (so the connection and H(k) are in one gauge by
  construction, whatever that gauge is); new `apply_wannier_correction`
  returns the corrected r, the r^{a;b} piece AND the corrected velocity,
  and is the only place the algebra lives. `WANNIER_R_DEFAULT = True`.
- `wannier.py`: `build_system_from_tb` Hermitian-pairs the position blocks,
  repairs the R=0 diagonal against the centres, always builds atompos from
  the centres, prints one diagnostic line (two if the diagonal was bad).
- The three chi^(2)/delta_Q call sites call the helper; chi^(2) uses the
  corrected velocity as its current vertex; `delta_Q`'s `subspace`/`band`
  pair integrands are written in terms of the full r; `quantum_metric`
  perturbs with the full r; `chi_ii` uses the sum-rule curvature.
- `examples/test_wannier_gauge.py` (new, 41 checks). The key idea: a
  multi-atom TB_simple model has point-like orbitals, so its atomic-gauge
  r and r^{a;b} are exact with no correction. Re-express the SAME model
  with a different atompos (zero, or the true centres plus random offsets):
  A^W is then a known diagonal, and the corrected operators — and every
  engine's per-k output — must reproduce the atomic-gauge answer. They do,
  to 1e-13 at the operator level and 1e-14 at the engine level; the
  no-correction control fails by 1% to 100%. A finite-difference certificate
  with a k-dependent A^W (synthetic, and the real MoS2 connection) checks
  the U^dag(dA^W)U term at 1e-7, converging as h^2.

### Test results (MoS2, `mos2_tb.dat` + `mos2_centres.xyz`)

**Test 1 — delta_Q real.** nk=200, Ef=-0.030102, kT=eta=eta_sos=0.025,
`subspace` (the report's setup; HEAD baseline reproduces the report bit
for bit):

| | dQ_yyy | dQ_xxy | Im/Re (yyy) |
|---|---|---|---|
| baseline, wannier_r=True | -0.533203 -0.199584j | +0.542321 +0.203721j | 0.374 |
| fixed, wannier_r=True, subspace | -0.335564 +0.000000j | +0.333092 -0.000000j | 4e-18 |
| fixed, wannier_r=True, thermal | -0.335563 -0.000000j | +0.333091 +0.000000j | 3e-18 |
| fixed, wannier_r=False (point-like) | -0.357861 | +0.317245 | 3e-18 |

PASS. In the default `thermal` formulation every one of the 8 components is
real to <= 1e-9 relative (the 1e-9 is on xyx = 0.3269; the two components
that are ~1e-3 of the others sit at 8e-6). `subspace` keeps a 2e-4 relative
imaginary part on xyx/yxx that is also there with the correction OFF: an
eta = 0.025 artefact of the T=0 formulation, not this bug. The correction
is +0.090 on a total of -0.336 (T_Sipe_wannier_corr), i.e. a 27% effect —
it is not small, as the report anticipated. The D3h chain
yyy = -xxy = -xyx = -yxx holds to 2.6% (thermal), inside the 3.5% floor of
this unsymmetrized Hamiltonian; `subspace` was at 12% before the vertex fix
and is at 2.6% now.

**Test 2 — sub-gap dissipation.** nk=300, all 27 abc, eta ladder
0.05 -> 0.00625, S = sum_abc chi_total. **The criterion in the original
report used the wrong component.** `chi_total` is the second-order
conductivity (it matches sigma^{abc}(w; w, 0) at ratio +1), so as CLAUDE.md
says under "Frequency-Integrated chi^(2)": Im is the reactive part and Re
the dissipative one. Read that way:

| omega | Re S at eta=0.05 -> 0.00625 (ratio; pure broadening 0.125) | Im S (ratio) | Re/Im at 0.00625 | 2eta/w + eta/(gap-w) |
|---|---|---|---|---|
| 0.05 | 1.068 -> 0.136 (0.1275) | -0.548 (1.004) | 0.25 | 0.25 |
| 0.30 | 1.109 -> 0.141 (0.1275) | -3.337 (1.004) | 0.042 | 0.046 |
| 0.60 | 1.247 -> 0.159 (0.1274) | -6.996 (1.004) | 0.023 | 0.026 |
| 1.00 | 1.662 -> 0.211 (0.1272) | -13.14 (1.006) | 0.016 | 0.021 |

PASS: the dissipative part scales linearly in eta at every omega and is
quantitatively the eta in the 1/(w + i eta), 1/(w + 2i eta) and interband
denominators; the reactive part converges to an eta-independent value.
(The `2eta/w` piece is the omega ~ eta crossover discussed in the
frequency-integral section; at w = 0.05 and eta = 0.00625 it is 0.25 and
dominates.) The old code's failure, restated in the right component, is in
the table under "Baseline comparison" below.

**Test 3 — regressions.** `test_wannier_r_flag.py` (31), `test_dQ_thermal.py`
(19), `test_freq_integral.py` incl. the sampled-mode comparison against a
HEAD worktree (0.0), `honeycomb_warp/check_eq13.py` (ratio -1.000000
unchanged), and the three benchmark configs bit-identical to HEAD.
TB_simple input never enters the correction, so nothing changes there
except metallic `chi_ii` (item 5) — the benchmark chi^(2) config sits in a
gap and is unchanged.

**Test 4 — `chi_ee1`.** Its reactive (Im) part varies 12% across the
ladder at w = 0.05 and 0.6% at w = 1.0; that is the 2eta/w crossover
(2eta/w = 2 at the top of the ladder), not a defect. No separate item.

### What changed for users

- `system.wannier_r` defaults to **true** again. `false` is the point-like-
  orbital approximation, in the atomic gauge; both settings print a note.
- `_tb.dat` input is now always in the atomic gauge, so `wannier_r: false`
  results on `_tb.dat` input WITHOUT a centres file change (they were in
  the lattice gauge before, missing the intra-cell term). With a centres
  file they are unchanged.
- Pass `wannier_centres` for `_tb.dat` input: the file's own R=0 diagonal
  can be wrapped or scrambled (it is, in z, for this MoS2), and the loader
  tells you when it had to repair it. Do not use the z position operator
  from this MoS2 file.
- `delta_Q` `subspace`/`band` and `quantum_metric` results with the
  correction on change (item 2). `thermal` changes only through item 1.
- `quantum_metric`'s finite-difference `dQ` is a bare-velocity heuristic and
  is NOT gauge-covariant with the correction on (17% on the test models);
  `Q` and `dQf` are. The engine prints a note; use `calc.type: delta_Q`.
- The Sipe sub-term split (`T_Sipe_*`, `chi_ei*_sipe_*`) is gauge-dependent
  by construction — only the sums are physical. Do not compare sub-terms
  across gauges or read physics into `T_Sipe_wannier_corr` alone.
- The Souza `eta_sos` regularization acts on the bare -i v/w only, so it is
  not gauge-covariant near a degeneracy (O(eta_sos^2/w^2) x the intra-cell
  term); the atomic gauge, where the bare part is the physical point-like
  operator, is the right place to regularize, which is another reason it is
  now the only gauge `_tb.dat` input runs in.
- Metallic `chi_ii` values from before this fix are wrong (item 5).

### Baseline comparison

Same ladder endpoints at nk=100, HEAD baseline vs fixed code, dissipative
part Re S = sum_abc Re chi_total (the reactive Im S alongside):

| omega | baseline Re S, eta 0.05 -> 0.00625 (ratio) | baseline Im S (ratio) | fixed Re S (ratio) | fixed Im S (ratio) |
|---|---|---|---|---|
| 0.05 | -2.730 -> -3.360 (1.23) | -2.807 -> -3.398 (1.21) | 1.071 -> 0.138 (0.129) | -0.5458 -> -0.5479 (1.004) |
| 0.30 | -2.176 -> -2.847 (1.31) | -3.541 -> -4.104 (1.16) | 1.112 -> 0.143 (0.129) | -3.3235 -> -3.3367 (1.004) |
| 0.60 | -1.540 -> -2.307 (1.50) | -4.580 -> -5.139 (1.12) | 1.251 -> 0.161 (0.128) | -6.9653 -> -6.9961 (1.004) |
| 1.00 | -0.428 -> -1.473 (3.44) | -6.557 -> -7.238 (1.10) | 1.666 -> 0.214 (0.128) | -13.065 -> -13.140 (1.006) |

The old code had an O(1) dissipative response below the gap that *grew* as
eta -> 0, and a reactive part that did not converge either. The fixed code's
nk=100 and nk=300 values agree to three digits, so these are converged.

---

## Original report (2026-08-25), unchanged

**Location:** `tightbinding/calc/wannier_gauge.py` (`compute_A_W_k`),
added in commit `1054879` ("Add Wannier-gauge r-correction and
occupied-subspace dQ formulation") as
`nonlinear_optical._compute_A_W_k`, and moved to its own module when the
`wannier_r` switch was unified. `nonlinear_optical._compute_A_W_k` is
still an alias, so existing scripts and the snippets below keep working.

**Blast radius:** one function, four call sites, three engines.

| file | note |
|---|---|
| `calc/wannier_gauge.py` | definition (`compute_A_W_k`) |
| `calc/nonlinear_optical.py` | slow/reference chi^(2) path |
| `calc/nonlinear_optical_fast.py` | vectorized chi^(2) path — **this is the one that actually executes** |
| `calc/delta_Q.py` | per k-point |
| `calc/quantum_metric.py` | per k-point, Eq. 22 only (no `dA_W`) |

It surfaces as `chi_ei1/2_sipe_wannier_corr` in chi^(2) and as
`T_Sipe_wannier_corr` in delta_Q. Because every engine calls the one
definition, **all three carry the same defect**, and both sides of any
chi^(2)-vs-delta_Q comparison are contaminated simultaneously.

**Since 2026-08-27 the default is `system.wannier_r: false`** — the
correction is off unless asked for, so new results do not silently
inherit the defect. That is a holding position, not a fix: `false` drops
a physically required term (see "Why the obvious quick fixes are wrong").
Restore `WANNIER_R_DEFAULT = True` in `calc/wannier_gauge.py` when this
bug closes. Reproducing any of the numbers below now requires
`system.wannier_r: true` explicitly.

---

## Diagnosis

The correction injects a spurious contribution that survives eta -> 0.
Two independent symptoms, one cause.

### Symptom 1 — delta_Q becomes complex (cleanest signal)

delta_Q is a DC-field-induced change in the quantum metric: a real
observable. It must be real. It isn't.

MoS2, nk=200, eta = eta_sos = 0.025, kT = 0.025, Ef = midgap = -0.030102,
`dQ_occupied_subspace=True`:

| `wannier_r` | dQ_yyy | dQ_xxy |
|---|---|---|
| `True` (default) | -0.533203 **-0.199584j** | +0.542321 **+0.203721j** |
| `False` | -0.357861 **+0.000000j** | +0.317245 **-0.000000j** |

With the correction disabled the result is **exactly real to printed
precision**. With it enabled the imaginary part is 37% of the real part.
The correction also shifts the real part by 33%, so this is not a small
perturbation on an otherwise-correct number.

### Symptom 2 — chi^(2) has non-vanishing sub-gap dissipation

Below the band gap an insulator has no real transitions, so the
dissipative response must vanish as eta -> 0. For denominators
`1/(w - w_nm +/- i*eta)^p` with `|w - w_nm| >> eta`, the imaginary part
is suppressed by `~eta/|w - w_nm|` and must scale linearly in eta.

Test: `S = sum_{abc} chi_total^{abc}` over all 27 abc, MoS2, nk=300,
gap = 1.7611 eV, eta ladder 0.05 -> 0.00625 (8x).

At w = 0.05 eV (1.71 eV below the gap), eta = 0.00625:

- predicted `|Im/Re| ~ eta/(gap-w) = 0.0037`
- observed `|Im/Re| = 1.007` — ~270x too large
- Im changes by only 13% over an 8x eta reduction (a ratio of 1.16 where
  pure broadening gives 0.125), i.e. **the imaginary part survives eta -> 0**

Localizing by sub-term (27-index sum, Im at eta=0.00625, ratio over the
8x eta ladder):

| sub-term | chi_ei1 | ratio | chi_ei2 | ratio |
|---|---|---|---|---|
| `sipe_delta` | -3.31e-11 | **0.126** | -1.84e-11 | **0.126** |
| `sipe_d2H` | -1.4181e-02 | 1.000 | -7.0905e-03 | 1.000 |
| `sipe_3band` | +9.1953e-03 | 1.162 | +4.5968e-03 | 1.163 |
| **`sipe_wannier_corr`** | **-1.5935e+00** | 1.164 | **-1.5862e+00** | 1.165 |
| `dk_f` | 0 | — | 0 | — |
| `delta_r` | +4.6168e-03 | 1.044 | +2.3556e-04 | 1.446 |
| parent | -1.5938e+00 | | -1.5884e+00 | |

`sipe_wannier_corr` is **99.98%** of chi_ei1's sub-gap Im and **99.9%** of
chi_ei2's. Subtracting only that term drops `|Im/Re|` at w=0.05 from
1.007 to 0.0372 — about 96% of the discrepancy.

**Internal control:** `sipe_delta` scales at 0.126 against a predicted
0.125. A correctly-behaving term reproduces the analytic expectation to
1%, which establishes that the test methodology is sound and the anomaly
is real rather than an artefact of the diagnostic.

### Why the obvious quick fixes are wrong

- **Do not treat `wannier_r=False` as the fix.** The Wannier gauge genuinely
  needs this correction — a `_tb.dat` Hamiltonian is not in the atomic
  gauge where `r = -i v/w` holds with zero intra-cell term (see the
  "Position operator gauge" note in `CLAUDE.md`, which applies to
  TB_simple, *not* to Wannier input). Disabling it gives a clean-looking
  but physically wrong answer.
- **It is not a relative sign between `chi_ei1` and `chi_ei2`.** Their
  `sipe_wannier_corr` contributions are nearly equal (-1.5935 vs
  -1.5862 at w=0.05), so flipping the relative sign cancels them against
  each other and looks like a 45x improvement at low w. But the residual
  is still eta-independent (ratio 0.92-0.99), and the cancellation
  degrades to nothing by w ~ 0.6 eV. Each ordering is individually wrong.

### Secondary, smaller issue

After `sipe_wannier_corr` is removed, the residual sub-gap Im is still
eta-independent (ratio ~0.91) and is then dominated by `chi_ee1`
(-7.1e-02 of the -6.9e-02 total at w=0.05). `chi_ee2` also shows erratic
eta scaling. This is roughly 4% of the original discrepancy — a separate,
smaller problem. Confirm it after the main fix rather than chasing it now.

`chi_ii`, `chi_ie1`, `chi_ie2` are **exactly zero** below the gap at every
eta. Those terms are clean.

---

## What needs to be fixed

Review `compute_A_W_k` (`calc/wannier_gauge.py`) against
arXiv:1804.04030 Eq. 22 + Eq. 36. The output is used as a Wannier-gauge
correction to `dk_rmtx = r^{b;c}`, so an error in its phase, Hermiticity,
or the sign/placement of the intra-cell displacement will propagate into
every response built on the generalized derivative.

Specific things to check:

1. **Hermiticity of the returned `A_W` and `dA_W`.** The Berry connection
   matrix in the Wannier gauge must be Hermitian at each k. A
   non-Hermitian `A_W` is the most direct way to get a real observable
   coming out complex.
2. **Phase convention of the Bloch sum.** `bloch.py` builds `H(k)` in the
   *atomic* gauge, `exp(ik.(R + tau_j - tau_i))`. If `compute_A_W_k`
   builds its sum in the lattice/periodic gauge `exp(ik.R)`, the two are
   inconsistent and the correction is applied in the wrong gauge. Check
   which convention `wannier_r_displacements` is expressed in and whether
   the intra-cell `tau` offsets are handled the same way as in `get_H_k`.
3. **The derivative `dA_W`.** Verify it is the k-derivative of the same
   object actually being used as `A_W`, with a consistent factor of i.
4. **Whether the correction should be subtracted or added** in
   `_compute_dk_rmtx`, and whether the same sign is correct for both
   `chi_ei1` and `chi_ei2` orderings.

Fix in `calc/wannier_gauge.py` only — `nonlinear_optical.py`,
`nonlinear_optical_fast.py`, `delta_Q.py` and `quantum_metric.py` all
import the single definition, so the fix propagates. But note the *fast*
path is what executes in practice; verify against it.

When the fix lands, flip `WANNIER_R_DEFAULT` back to `True` in that same
file and re-run `examples/test_wannier_r_flag.py` (its default-value
checks read the constant, so they follow automatically).

---

## How to test the fix

Run these in order. Test 1 is the primary gate.

### Test 1 (primary) — delta_Q must be real

Cheapest and least ambiguous: one number, unambiguous correct answer,
~20 s at nk=200 on 16 ranks.

```python
cfg = {'system': {'wannier_r': True}, 'calc': {
    'type': 'delta_Q', 'nk': [200, 200], 'eflist': [-0.030102],
    'kT': 0.025, 'eta': 0.025, 'eta_sos': 0.025,
    'components': ['xx', 'xy', 'yx', 'yy'], 'field_direction': ['x', 'y'],
    'dQ_occupied_subspace': True}}
```

(`wannier_r` under `calc` now raises — it moved to `system` when the
switch was unified across engines.)

**PASS:** `Im(dQ_yyy) / |Re(dQ_yyy)| < 1e-10` with `wannier_r=True`.
Currently 0.374.

The `wannier_r=False` result (dQ_yyy = -0.357861) is *not* the target —
the corrected `wannier_r=True` value should differ from it, since the
correction is physically required. What must hold is that it is real.

### Test 2 — chi^(2) sub-gap dissipation must vanish as eta -> 0

MoS2, nk=300, all 27 abc, eta ladder [0.05, 0.025, 0.0125, 0.00625],
omega well below the 1.7611 eV gap. Form `S = sum_{abc} chi_total^{abc}`.

**PASS, both conditions:**
- `|Im S / Re S| ~ eta/(gap - omega)` to within a factor of ~2
  (at w=0.05, eta=0.00625 that is 0.0037; currently 1.007)
- Im S scales linearly in eta: ratio over an 8x eta reduction should be
  ~0.125, matching what `sipe_delta` already achieves (0.126).
  Currently 1.16.

`Re S` should retain smooth reactive omega-dependence — it already does,
and must not be destroyed by the fix.

### Test 3 (regression) — existing benchmarks must still hold

- `chi_total` vs the **analytic pole-sum** on the synthetic random 4-band
  System: ratio +1.0000, rel. err ~1e-8. See "Validated Benchmarks" in
  `CLAUDE.md`. This is the sharpest existing check and must not regress.
- Slow and fast chi^(2) paths agree to ~5e-16.
- MoS2 D3h relations: forbidden components (xxx, xyy, yxy, yyx) small,
  and `chi_yyy = -chi_xxy = -chi_xyx = -chi_yxx`. Note the MoS2 Wannier
  Hamiltonian is **not** symmetrized — measured D3h violation floor is
  1.5% (forbidden/allowed) and 3.5% (chain spread), independent of nk
  from 100 to 800. Do not chase residuals below that floor.

### Test 4 — then re-check the secondary `chi_ee1` issue

Re-run the Test 2 decomposition per sub-term. If `chi_ee1`'s Im still
fails to scale with eta after the main fix, open a separate item.

---

## Reproduction data

Scripts and stored results are in the dQ/chi2 TMD project under
`rerun_2026-08/MoS2/` (see `CLAUDE.local.md` for where that lives):

- `chi2_sum27.npz` — 27 directions x 27 stored quantities
  (`CHI_ALL_NAMES` + `chi_total`, including all 12 `chi_ei` sub-terms)
  x 28 omega x 4 eta. Everything needed to re-test any recombination of
  terms with no recomputation.
- `run_sum27_full.py` — regenerates the above (~17 min, 16 ranks).
- `dissect_sum27.py` — per-term eta-scaling tables.
- `check_dissipative.py` — sub-gap eta-scaling for a chosen direction set.
- `mos2_setup.py` — shared system build; recomputes Ef from the
  eigenvalues at K rather than hard-coding it.

MoS2 reference values used above: VBM -0.910647, CBM +0.850443,
gap 1.7611, Ef(midgap) -0.030102 eV.
