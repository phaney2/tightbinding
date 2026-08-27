# BUG: Wannier-gauge position correction `_compute_A_W_k`

**Status:** open, blocking. Found 2026-08-25 while validating the
chi^(2) <-> static-field quantum-geometry relation on MoS2.

**Location:** `tightbinding/calc/nonlinear_optical.py:213-289`
(`_compute_A_W_k`), added in commit `1054879` ("Add Wannier-gauge
r-correction and occupied-subspace dQ formulation").

**Blast radius:** one function, three call sites, two engines.

| file | line | note |
|---|---|---|
| `calc/nonlinear_optical.py` | 213 | definition |
| `calc/nonlinear_optical.py` | 365 | slow/reference chi^(2) path |
| `calc/nonlinear_optical_fast.py` | 71, 84 | vectorized chi^(2) path — **this is the one that actually executes** |
| `calc/delta_Q.py` | 39, 330 | imports from `nonlinear_optical`, calls per k-point |

It surfaces as `chi_ei1/2_sipe_wannier_corr` in chi^(2) and as
`T_Sipe_wannier_corr` in delta_Q. Because `delta_Q.py` imports the
function rather than reimplementing it, **both engines carry the same
defect**, and both sides of any chi^(2)-vs-delta_Q comparison are
contaminated simultaneously.

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

- **Do not just set `wannier_r=False`.** The Wannier gauge genuinely
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

Review `_compute_A_W_k` (`nonlinear_optical.py:213-289`) against
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
   *atomic* gauge, `exp(ik.(R + tau_j - tau_i))`. If `_compute_A_W_k`
   builds its sum in the lattice/periodic gauge `exp(ik.R)`, the two are
   inconsistent and the correction is applied in the wrong gauge. Check
   which convention `wannier_r_displacements` is expressed in and whether
   the intra-cell `tau` offsets are handled the same way as in `get_H_k`.
3. **The derivative `dA_W`.** Verify it is the k-derivative of the same
   object actually being used as `A_W`, with a consistent factor of i.
4. **Whether the correction should be subtracted or added** in
   `_compute_dk_rmtx`, and whether the same sign is correct for both
   `chi_ei1` and `chi_ei2` orderings.

Fix in `nonlinear_optical.py` only — `delta_Q.py` and
`nonlinear_optical_fast.py` both import the single definition, so the fix
propagates. But note the *fast* path is what executes in practice; verify
against it.

---

## How to test the fix

Run these in order. Test 1 is the primary gate.

### Test 1 (primary) — delta_Q must be real

Cheapest and least ambiguous: one number, unambiguous correct answer,
~20 s at nk=200 on 16 ranks.

```python
cfg = {'system': {}, 'calc': {
    'type': 'delta_Q', 'nk': [200, 200], 'eflist': [-0.030102],
    'kT': 0.025, 'eta': 0.025, 'eta_sos': 0.025,
    'components': ['xx', 'xy', 'yx', 'yy'], 'field_direction': ['x', 'y'],
    'dQ_occupied_subspace': True, 'wannier_r': True}}
```

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
