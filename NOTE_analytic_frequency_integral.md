# NOTE: analytic frequency-integrated chi^(2)

**Status: IMPLEMENTED (2026-08-25).** See the "Frequency-Integrated chi^(2)"
section of `CLAUDE.md` for the user-facing description, and
`tightbinding/calc/freq_integral.py` for the code. This note is kept as the
derivation and as the record of the numerical exercise behind it.

**Raised:** 2026-08-25, after a numerical exercise on the gapped-honeycomb toy model.

## What changed relative to this note

The open design decisions were settled as:

* Config: a `calc.freq_integral` sub-block replacing `omega1list`, inside the
  existing `calc.type: nonlinear_optical` (not a new calc type). `p` may be a
  list. Output keeps the `[a][b][c]` nesting with the omega axis reduced to one
  column per power.
* Output reported in the **code's chi convention** (§5.1), documented rather
  than silently rotated by -i.
* `omega_max` defaults to infinity; `chi_e1`/`chi_i1` are NaN at p <= 1.
* §3.1 was implemented as a general (pole, multiplicity) routine as §3.3 asks,
  but via Taylor expansion of the complementary factors rather than the
  binomial formula — same coefficients, no negative-`n` special case.
* Phase 3 of `nonlinear_optical_fast.py` was refactored so the sampled and
  integrated modes share one implementation of the term algebra.

Three findings that revise this note:

1. **§5.2 is only half right.** The lower-endpoint divergences do *not* fully
   cancel. On the honeycomb the `chi_ii`/`chi_ie`/`chi_ee` coefficients cancel
   to ~1e-15, but the `chi_ei` pair survives, and the residue is exactly
   `-chi^(2)(0)`. At p = 1 the integral is genuinely log-divergent as
   `omega_min -> 0`. The engine now accumulates these coefficients so this is
   readable off any run.
2. **The `a = 1e-4`, `eta = 0.025` configuration of §4 is not usable.** `a`
   must stay several eta *above* zero, not below it, or the poles at `-i eta`
   and `-2i eta` are straddled. Combined with `omega_min << E_gap` the window
   needs `eta << E_gap/10`; the §4 setup has no valid window. The §4 reference
   numbers are retained as unit tests of the bare `G` routine, which is
   well-defined for any `a > 0`, but not as a production configuration.
3. **`b = inf` is easier than finite `b`**, not harder — the residues sum to
   zero so `J = -F(a)` exactly, with no large-b cancellation. See §3.3 below
   for what still had to be general.

---

## 1. What is wanted

`calc.type: nonlinear_optical` currently returns chi^(2) sampled on a list of
photon energies (`omega1list`). We want an *additional* mode that returns the
**frequency-integrated** response

```
    J = integral  dω  ω^(-p)  chi^(2)(ω)
```

computed from **closed-form antiderivatives**, not from quadrature over
`omega1list`.

Two requirements:

1. Use the analytic form of the integral (the integrand is a rational
   function of ω, so this is elementary — see §3).
2. Support **arbitrary integer power `p`** in the `ω^(-p)` weight. The
   exercise that motivated this used `p = 1`; `p = 0, 2, 3, ...` must be
   reachable. Note `p >= 2` changes the endpoint behaviour qualitatively
   (§5.2).

The payoff is accuracy and cost: the quadrature route needs a very fine ω mesh
(§4) and still only reaches ~1e-6 in the hard cases, whereas the closed form is
exact up to round-off and costs one evaluation per (k, band pair, term).

---

## 2. Where the ω-dependence lives in the code

This is the structural fact that makes the feature straightforward:

> **In every chi term, ω enters *only* through the denominator factors.
> All matrix elements, Fermi factors and prefactors are ω-independent.**

So each term is `(ω-independent prefactor) x (rational function of ω)`, and the
ω integral can be done analytically per term with the prefactor pulled out.

`tightbinding/calc/nonlinear_optical_fast.py`, in `_process_kpoint_fast`,
builds all the ω-dependent objects in one block (~lines 153-171):

| symbol       | definition                                | ω-dependent? |
|--------------|-------------------------------------------|--------------|
| `denom1`     | `1/(ω - Δe + i·eta)`                      | yes          |
| `denom12`    | `1/(ω + ω2 - Δe + 2i·eta)`                | yes (**2i·eta**) |
| `denom2`     | `1/(ω2 - Δe + i·eta)`                     | **no** (ω2 is a scalar) |
| `denom1_sq`  | `denom1**2`                               | yes          |
| `denom2_sq`  | `denom2**2`                               | no           |
| `s_om1`      | `1/(ω + i·eta)`                           | yes          |
| `s_om2`      | `1/(ω2 + i·eta)`                          | no           |
| `s_om12`     | `1/(ω + ω2 + 2i·eta)`                     | yes          |

and the composite products actually consumed by the terms:

```
d12_d1    = denom12 * denom1            d12_d2    = denom12 * denom2
d12_d1sq  = denom12 * denom1_sq         d12_d2sq  = denom12 * denom2_sq
```

Two things to notice, both of which matter for the implementation:

* **`denom1` and `denom12` do not share the same broadening** — `i·eta` vs
  `2i·eta`. Their poles are at *different* points in the complex plane, so a
  product like `d12_d1` is **not** `1/(ω-z)^2`; it is
  `1/[(ω-z_a)(ω-z_b)]` with `z_a != z_b`. A single-pole formula is not enough
  (§3.3).
* `chi_ee1` (~line 260-269) contracts `denom12[m,n,w]` against
  `denom1[m,p,w]` over an internal band index, i.e. the two poles sit at
  *different band-pair energies* as well. Same conclusion.

Per-term entry points, for reference:
`chi_ii` ~L217, `chi_ee1` ~L269, `chi_eit1/2/3` ~L301-325, `chi_ei1/2` ~L329,
`chi_ie1/2` ~L371-375, `chi_e1/e2/i1/i2` ~L383-399. The term-name lists and
`CHI_PHYSICAL` (what actually sums into `chi_total`) live in
`tightbinding/calc/nonlinear_optical.py` L20-54.

---

## 3. The analytic integrals

### 3.1 Single pole, general p and s — the general case

With `z = ω_nm - i·eta` (note the sign: `ω - ω_nm + i·eta = ω - z`, so `z` sits
in the **lower** half plane and `ω - z` never vanishes for real ω):

```
    G(p, s; a, b) = integral_a^b  dω  /  ( ω^p (ω - z)^s )
```

Partial fractions:

```
    1/(ω^p (ω-z)^s)  =  sum_{j=1..p} A_j / ω^j  +  sum_{i=1..s} B_i / (ω-z)^i

    A_j = (-1)^s     * C(s+p-j-1, p-j) * z^(j-s-p)
    B_i = (-1)^(s-i) * C(p+s-i-1, s-i) * z^(i-p-s)
```

**`C` must be the *generalized* binomial** `C(n,k) = n(n-1)...(n-k+1)/k!`, not
`math.comb`. The first argument goes negative in a legitimate case — `B_i` at
`p = 0, i = s` needs `C(-1, 0) = 1` — and `math.comb` raises on negative `n`:

```python
from math import factorial
def binom(n, k):                       # valid for negative n
    r = 1.0
    for t in range(k):
        r *= (n - t)
    return r / factorial(k)
```

Antiderivatives:

```
    integral A_1/ω dω      = A_1 ln ω
    integral A_j/ω^j dω    = A_j ω^(1-j)/(1-j)        j >= 2
    integral B_1/(ω-z) dω  = B_1 ln(ω-z)
    integral B_i/(ω-z)^i dω= B_i (ω-z)^(1-i)/(1-i)    i >= 2
```

**Branch care:** evaluate the log term as `ln(b-z) - ln(a-z)`, a *difference of
principal logs*, never as `ln((b-z)/(a-z))`. Because `Im z = -eta < 0`, both
`b-z` and `a-z` have positive imaginary part, so both principal arguments lie
in `(0, pi)` and the difference is unambiguous. Forming the ratio first can
cross the cut.

Since `ω^(-p)` is real, `integral ω^(-p) Im[chi] dω = Im[ integral ω^(-p) chi dω ]` —
integrate the complex function and take the imaginary part at the end.

### 3.2 p = 1, s = 1,2,3 written out (this is what was verified)

`L(ω) = ln(ω - z)`:

```
 s=1:  (1/z) [ L(b)-L(a) - ln(b/a) ]
 s=2:  (1/z^2) ln(b/a) - (1/z^2)[L(b)-L(a)] - (1/z)[1/(b-z) - 1/(a-z)]
 s=3: -(1/z^3) ln(b/a) + (1/z^3)[L(b)-L(a)] + (1/z^2)[1/(b-z) - 1/(a-z)]
                                            - (1/(2z))[1/(b-z)^2 - 1/(a-z)^2]
```

These agree with the general formula in §3.1 (checked term by term).

### 3.2b The general formula has been verified

Two independent checks, both passed:

* **Algebraic identity.** For `p = 0..5`, `s = 1..5`, four different poles,
  400 random real ω each: the partial-fraction expansion reproduces
  `1/(ω^p (ω-z)^s)` to a worst relative error of **7.0e-16**.
* **Integral.** `G(p,s)` vs adaptive quadrature on the benign range
  `[0.5, 20]` (deliberately away from the `ω^(-p)` endpoint blow-up, so the
  quadrature is trustworthy — see §5.4), `p = 0..4`, `s = 1..3`, two poles:
  agreement to **1e-15..1e-16**, with a few `s = 3` cases at 1e-11 that are
  quadrature error rather than formula error.

So §3.1 can be implemented as written. Note the sanity trap: verifying it by
quadrature over a range that *includes* small ω appears to show gross
disagreement for `p >= 2`, but that is the integrator failing on the
`1/ω^p` endpoint, not the formula.

### 3.3 What is still needed: distinct poles

As flagged in §2, real terms need

```
    integral  dω  /  ( ω^p  (ω-z_1)^(s_1) (ω-z_2)^(s_2) ... )
```

with **distinct** `z_i` (from the `i·eta` vs `2i·eta` mismatch, and from
different band pairs in the three-band contractions). Still elementary — the
partial-fraction expansion over distinct poles plus the `ω^(-p)` block — but the
implementation should be written as a **general routine** taking a list of
`(pole, multiplicity)` pairs plus `p`, rather than hard-coding §3.2.

Highest multiplicity currently needed is `s = 3` (from `d12_d1sq`:
`denom12 * denom1^2`). A degenerate-pole guard is needed for the case
`z_1 -> z_2` (it can happen if a future call path uses the same eta in both
factors, or if `Δe` coincides between band pairs) — the distinct-pole partial
fractions blow up there and must fall back to the repeated-pole form.

---

## 4. Numerical facts established (use as regression targets)

Everything below was measured on the gapped honeycomb (s orbital/site, T = 1 eV,
staggered on-site ±0.125 eV, gap 0.25 eV), for p = 1.

**Reference values.** `ω_nm = 0.25`, `eta = 0.025`, range `[a,b] = [1e-4, 1e3]`:

```
    Im G(p=1, s=1) = -1.5147512413e+01
    Im G(p=1, s=2) = +6.8653223354e+01
    Im G(p=1, s=3) = -3.0080982623e+02
```

**Quadrature comparison.** Using the substitution
`integral (1/ω) g dω = integral g d(ln ω)` (exact for the weight, so the only
error is the trapezoid error on `g`), over five interband energies the model
actually has (0.25 band edge, 2.016 M van Hove, 2.363 generic, 5.006, 6.005
Gamma max) x eta in {0.025, 0.01, 0.001} x s in {1,2,3}:

| trapezoid points | worst relative error |
|---|---|
| 1e4 | 9.6e+06 |
| 1e5 | 7.3e+05 |
| 1e6 | 1.1e-06 |

At the production `eta = 0.025` it reaches 1e-14..1e-15. A realistic combined
chi (all 15 terms, complex prefactors) hits 4.4e-15 at 1e6 points.

**The controlling parameter** is not the point count but the number of mesh
points inside the resonance width, `m = n*eta/(ω_nm * ln(b/a))` for a log mesh:

| s | m for 1e-6 | m for 1e-10 | m for 1e-13 |
|---|---|---|---|
| 1 | 2.6 | 3.9 | 8.3 |
| 2 | 3.9 | 5.7 | 17.9 |
| 3 | 5.7 | 6.6 | 447 |

Below `m ~ 4` the error does not degrade gracefully — it jumps to 1e5 relative.
The `s = 3` terms bind: their integrand peaks at ~`1/eta^3` but integrates to
O(1), i.e. ~1e7 cancellation between the lobes of the peak.

**This is the argument for the feature:** matching the closed form by quadrature
costs ~1e6 ω points per k-point.

---

## 5. Gotchas

### 5.1 Output convention carries a factor of i

Established by an eta -> 0 study (six decades, two sub-gap frequencies, k-grid
converged to 8e-10). Below the gap:

* `Re[chi]` from this code vanishes **linearly in eta** (log-log slope +1.0000)
* `Im[chi]` from this code is **eta-independent** (plateau stable to 9 sig figs)

A gapped insulator cannot dissipate below its gap, so the surviving part is the
reactive one. Writing `phi = -i * chi_code`:

```
    Re_reactive[phi]    =  Im[chi_code]
    Im_dissipative[phi] = -Re[chi_code]
```

Source-level cause: every `chi_ei` term carries an explicit `-1j` prefactor
(nonlinear_optical_fast.py ~L301-330) and `chi_ee` is built from `v_a r_b r_c`
with `r = -i v * inv_de`. **Whoever implements this should decide whether the
integrated output is reported in the code's convention or in `phi`'s, and
document it** — it is currently an easy trap when reading Re/Im plots.

### 5.2 The lower endpoint diverges term by term

`Im[1/(ω-z)^s]` tends to a nonzero constant as `ω -> 0`, so with the `ω^(-p)`
weight **each individual term diverges at `a -> 0`**:

* `p = 1`: logarithmic, `-A_1 ln a`
* `p >= 2`: power law, `a^(1-j)` for `j` up to `p` — much worse

These divergences must cancel against the matrix-element prefactors once the
terms are summed. Two consequences:

* Keeping `a` finite and identical across all terms makes the comparison
  exact and unambiguous; that is how the validation above was done.
* If an `a -> 0` limit is wanted, the cancellation should be done
  **analytically on the coefficients** (sum the `A_j` across terms first, check
  they cancel, then integrate) rather than numerically. Doing it numerically
  is where a Kramers-Kronig cross-check went wrong during the exercise.

### 5.3 Poles near ω = 0

`s_om1 = 1/(ω + i·eta)` and `s_om12 = 1/(ω + ω2 + 2i·eta)` put poles at
`ω = -i·eta` and `-2i·eta`, i.e. essentially *on top of* the `ω^(-p)` endpoint
singularity. With `ω2 = 0` these appear in `chi_ii`, `chi_ie1/2`, `chi_i1/2`.

For a **gapped insulator with Ef in the gap and kT << gap they are harmless**:
they multiply Fermi-surface factors (`dk_f ~ df/dE`, `d2k_f`) which vanish
identically. Verified numerically — the ratio (largest physical sub-term)/(total)
stayed at 0.7 all the way down to `eta = 2.5e-8`, with no `1/eta` blow-up.
**For metals or doped systems this will not hold** and the ω≈0 region needs real
care. Worth an explicit guard or at least a documented restriction.

Also: `eta = 0` exactly must not be allowed on this path — `denom2` and `s_om2`
become `1/(i*0)`, and multiplied by the vanishing Fermi factors that is
`0 * inf = NaN`.

### 5.4 Do not trust adaptive quadrature as the cross-check

`scipy.integrate.quad` reached **2.2e+06 relative error** on the sharp `s = 3`
cases, even with `points=` placed at the pole and `epsabs=1e-14`. It misses the
cancelling lobes. If a numerical cross-check is wanted in the test suite, use a
dense uniform log mesh with `m` from the table in §4, not a black-box adaptive
integrator.

### 5.5 One thing that did not work

An attempt to beat the uniform mesh with a graded one (extra points spaced
logarithmically in `|ω - ω_nm|`) **saturated at ~5e-3** regardless of point
count, and got *worse* with a denser background mesh. Unioning an asymmetric
log background into the `s = 3` peak appears to break the ~1e7 cancellation
between its lobes. Not diagnosed further. Mentioned so nobody re-treads it
assuming it is easy.

---

## 6. Reference material

Scripts and data from the exercise (a separate project directory, read-only
from here):

```
C:\Users\phaney\wrk\dQ\rerun_2026-08\honeycomb\
    honeycomb_setup.py             model builder, D3h symmetry notes
    run_freq_integral_check.py     analytic F_s + numeric comparison  <-- start here
    plot_freq_integral_check.py    mesh-requirement scaling
    freq_integral_check.npz        the comparison table
    run_eta_zero.py                the eta -> 0 study behind §5.1
    chi2_eta_zero.npz              its ladder, all sub-terms per eta
    run_single_k_integral.py       per-k kernel called directly, no BZ sum
    run_kk_sumrule.py              the KK attempt; see §5.2 for why it misfired
```

`run_single_k_integral.py` shows how to call `_process_kpoint_fast` standalone
for one k-point with no BZ sum or normalisation — useful for unit tests of the
analytic integrator against the existing sampled path.

Symmetry cross-check available for free on this model: D3h forces
`chi_yyy = -chi_yxx = -chi_xxy = -chi_xyx`, all other in-plane triplets zero,
and every triplet containing z identically zero. Holds to 4e-12 in the current
code, so it is a good invariant for the integrated path to reproduce.

### Caveat on k-space

The reactive/dissipative split exists only **after** pairing `k` with `-k`:
at a single k, `Re[chi(k)]` is odd in k (cancels in the BZ sum) and `Im[chi(k)]`
is even. A single bare k-point is therefore not a meaningful test of §5.1,
though it is fine for testing the ω integral itself.
