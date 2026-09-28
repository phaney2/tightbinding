# The three-piece decomposition of delta_Q

**2026-09-26.** Splitting `calc.type: delta_Q` into the three pieces of Eq. 1
(metric part Eq. 17) of `delta_Q_multiband_expansion.pdf`, and measuring their
BZ-integrated relative contributions.

Contents: [the decomposition](#1-the-decomposition) ·
[what maps to what](#2-the-engines-partition-is-already-this-one) ·
[a correction to the note](#3-a-slip-in-the-note) ·
[certificate](#4-certificate) ·
[results](#5-results) ·
[MoS2 Wannier, correction on and off](#54-mos2-wannier-11-band-7-occupied) ·
[Liu three-band TMDs](#55-liu-three-band-tmd-models-24-models) ·
[caveats](#6-two-caveats-on-the-split) ·
[reproducing](#7-reproducing-all-of-this)

---

## 1. The decomposition

With p in occ, q in un, w_pq = E_p − E_q < 0 and the field factor absorbed,
the note writes dQ^ab = F^ab + conj(F^ba) with

| | piece | what it is |
|---|---|---|
| **(i)** | `i S_pq r^a_pq r^{c;b}_qp / w_pq` | **connection** — the shift-current matrix element carrying a 1/w weight |
| **(ii)** | `i S_pq d_b(1/w_pq) r^a_pq r^c_qp` | **gap gradient** — in the metric part this is (gap gradient) × (two-state Berry curvature), i.e. the Berry-curvature dipole |
| **(iii)** | `S_pq (r^c_qp/w_pq) [ S_{q'!=q} r^a_pq' r^b_q'q − S_{p'!=p} r^a_p'q r^b_pp' ]` | **three-band** — closed loops of position matrix elements; identically zero for two bands |

Piece (ii) is usually displayed after an integration by parts that moves the
derivative onto `Im(r^a r^c)` and makes the dipole form manifest. That transfer
is **not** done numerically: only the BZ integral is wanted and it is the same
either way.

Throughout, `Re` is the metric part `dg^ab` (Eq. 17) and `Im` the curvature
part, `dOmega^ab = −2 Im dQ^ab` (Eq. 18). The engine uses `H' = −E·r`, so every
number below is **−1 × the note's sign**; no ratio is affected.

---

## 2. The engine's partition is already this one

It is the same split. Only the names mislead:

| note piece | engine terms |
|---|---|
| (i) connection | `T_Sipe_Delta + T_Sipe_d2H + T_Sipe_3band + T_Sipe_wannier_corr` |
| (ii) gap gradient | `T_Delta` |
| (iii) three-band | `T_3band` + `T_mix` (subspace) / `+ T_loop` (thermal) |

Two traps:

* **`T_Sipe_3band` belongs to (i), not (iii).** It is the three-band sum
  *inside* the Sipe generalized derivative `r^{c;b}`, which the note treats as
  one atomic object. The genuine three-band loop is split across two or three
  other term names according to the occupation of the intermediate band.
* **`T_Delta` is (ii)** because `Delta^b_pq = v^b_pp − v^b_qq` is exactly
  `d_b w_pq`, so `d_b(1/w_pq) = −Delta^b_pq / w_pq^2`.

Exposed as `result['delta_Q_groups'][a][b][c][group]`, saved to and reloaded
from the `.npz`. `calc.delta_Q.group_terms()` rebuilds it from `delta_Q_terms`
for a file written before 2026-09-25.

### All three pieces are separately gauge invariant

This is the reason to group at all, and it holds in both senses of the word.

* **Band phase gauge**, `|u_n> -> exp(i phi_n)|u_n>`.  Every piece is a closed
  loop of covariant objects — `r^a_pq r^c_qp` for (ii), the three-element loop
  for (iii), and for (i) the generalized derivative `r^{c;b}` transforms exactly
  like `r^c` (that is precisely what its `-i(xi_nn - xi_mm)` term is for).  So
  every piece is invariant term by term in (p,q), not merely after summation.
  The note says this only of (iii); it is true of all three.  **Verified to
  9e-17.**
* **The atompos / Bloch-phase convention** — whether `bloch.py` builds the
  Bloch sum with `R` or with `R + tau_j - tau_i`.  With the Wannier correction
  on, each piece is again invariant: **6e-15** on a point-like three-atom model
  re-expressed in the lattice gauge and in a randomly shifted gauge.  With the
  correction off the same re-gauging moves the pieces by **81%**.

The contrast that matters is with the raw term names.  The individual `T_*`
terms are covariant under band phases too, but they are **not** invariant under
the convention: `T_Sipe_wannier_corr` moves by **33%** between gauges while the
(i) group it belongs to does not move at all — `wannier_corr` and `3band`
shuffle into each other and only their sum, the covariant `r^{c;b}`, is fixed.
That is what CLAUDE.md means by "the `T_Sipe_*` split is gauge-dependent; only
the sum is physical", and it makes the three groups the **coarsest partition of
`delta_Q_terms` that is physically meaningful**.  Checks: section [5] of
`examples/test_dQ_term_groups.py`.

On the real MoS2 file the same test is stark (nk = 120, eta_sos = 1e-4):

```
MoS2 dQ_yyy                  corr        total     (i) conn  (iii) 3band
atomic gauge (centres)       ON      -0.335574     0.061006    -0.396580
lattice gauge (atompos=0)    ON      -0.335574     0.061006    -0.396580
atomic gauge (centres)       OFF     -0.357864    -0.159271    -0.198593
lattice gauge (atompos=0)    OFF     -1.457000    -1.509266     0.052266
```

Correction on: identical to six digits across a gauge change.  Correction off:
a **factor of four**.

So `wannier_r: false` on `_tb.dat` input is not gauge invariant, and is the
wrong operator.  It is not *ill-defined*, though, only because the loader pins
the convention: since 2026-09-10 `atompos` is always built from the Wannier
centres, so the flag means one specific thing — replace each Wannier function
by a point at its centre.  That is a definite approximation and a sensible
one (the centres are the right leading-order placement, hence only 6.2% error
on the total); `atompos = 0` is a *bad* placement, and the factor of four is
the size of the ambiguity, not of the approximation.  Quote the corrected
result; use the uncorrected one only as the diagnostic it is, a measure of how
much the finite Wannier spread matters.

Three qualifications:

* **For TB_simple / point-like orbitals the flag is inert and nothing is
  wrong.**  In the atomic gauge `r = -i v/w` is exact there and
  `compute_A_W_k` returns None.  All of this is specific to `_tb.dat`.
* **"With the correction" means right within the Wannier manifold, not
  exact.**  It is the position operator projected onto a truncated basis; its
  non-Abelian curvature F != 0 is what produces the 2.3% a != c violation of
  the chi2 <-> dQ sum rule, and `_tb.dat` carries no <0n|r_a r_b|Rm> at all.
* **Covariance with the correction on is exact only to
  O(eta_sos^2/w^2) x (intra-cell term)** near degeneracies, since the Souza
  regularization acts on the bare `-i v/w` only.  Here it held to six digits.

> **Gauge invariance is not operator independence.**  Toggling `system.wannier_r`
> is *not* a gauge transformation — it changes the position operator itself, by
> dropping the finite-spread intra-cell part.  So there is no tension between
> "each piece is gauge invariant" and the large swing in section 5.4: the pieces
> are invariant under relabelling the basis, and sensitive to which operator you
> put in them.  The control above is exactly this distinction — re-gauging with
> the correction on changes nothing (6e-15), re-gauging with it off changes
> everything (81%), because only then is the operator itself different in each
> gauge.

---

## 3. A slip in the note

The note's text line

> `dQ^ab = F^ab + F^ba`, "using Tr[(D_aT_c)^dag R_b] = Tr[R_b^dag D_a T_c]"

is loose: `Tr[(D_aT_c)^dag R_b]` is the **conjugate** of `F^ba`, not `F^ba`.
The correct statement is `dQ^ab = F^ab + conj(F^ba)`, which is what the note's
own Eq. 18 for `dOmega` requires (`Im F^ab − Im F^ba`) and what the engine
computes. Eq. 17 for the metric part is unaffected either way, since
`Re conj(F^ba) = Re F^ba`. Verified: with the conjugate the engine matches
Eq. 1 in **both** real and imaginary parts to 3e-16; without it, only the real
part matches.

---

## 4. Certificate

`examples/test_dQ_term_groups.py` — 25 checks. Eq. 1 is evaluated with explicit
python loops over occ/un band indices and compared against the engine's masked
matrix algebra, for both the `subspace` and `thermal` assemblies.

```
Certificate for the 3-piece regrouping of the delta_Q term decomposition.

[1] Group bookkeeping
  PASS no term appears in two groups                              
  PASS every 'band' term is grouped                               
  PASS every 'subspace' term is grouped                           
  PASS every 'thermal' term is grouped                            
  PASS no group names a nonexistent term                          

[2] Two-band insulator: piece (iii) must vanish identically
  PASS two-band: three_band piece is exactly zero                 max |(iii)| 0.0e+00
  PASS two-band subspace: groups == Eq. 1 pieces                  3.759e-16  (tol 1e-13)
  PASS two-band thermal:  groups == Eq. 1 pieces                  4.568e-16  (tol 1e-13)
  PASS two-band: groups sum to the engine total                   0.000e+00  (tol 1e-14)
  PASS Wannier Sipe sub-term is alive on the decorated model      max |wannier_corr| 5.10e-01
  PASS two-band + Wannier: groups == Eq. 1 pieces                 6.275e-17  (tol 1e-13)

[3] Four-band insulator, 2 occupied: all three pieces alive
  PASS connection piece is nonzero                                max |connection| 4.644e-01  (scale 4.644e-01)
  PASS gap_gradient piece is nonzero                              max |gap_gradient| 5.715e-02  (scale 4.644e-01)
  PASS three_band piece is nonzero                                max |three_band| 2.163e-01  (scale 4.644e-01)
  PASS 4-band subspace: groups == Eq. 1 pieces                    1.793e-16  (tol 1e-12)
  PASS 4-band thermal:  groups == Eq. 1 pieces                    2.988e-16  (tol 1e-12)
  PASS 4-band subspace: total == Eq. 1 total                      1.336e-16  (tol 1e-12)
  PASS 4-band thermal:  total == Eq. 1 total                      2.155e-16  (tol 1e-12)
  PASS 4-band: groups sum to the engine total                     2.988e-17  (tol 1e-13)

[4] compute_delta_Q output and .npz round trip
  PASS result carries delta_Q_groups                              
  PASS BZ: groups sum to delta_Q (rel)                            4.776e-18  (tol 1e-14)
  PASS delta_Q_groups round-trips through .npz                    0.000e+00  (tol 0e+00)
  PASS group_terms() reproduces it from delta_Q_terms             0.000e+00  (tol 0e+00)

All delta_Q term-group tests passed.
```

The bookkeeping checks are the guard that matters for the future: every term
name of every formulation must land in exactly one group, so a term added later
cannot silently vanish from the breakdown — the test fails instead.

---

## 5. Results

`share` = piece / total (can exceed 100% when pieces cancel);
`weight` = |piece| / sum |piece|, the cancellation-free fraction. Components
whose total is a symmetry zero are marked and excluded from the summary line.

### 5.1 Gapped honeycomb — 2 bands, 1 occupied, C3

`model=honeycomb formulation=thermal nk=240 ef=0.0 kT=0.002 eta_sos=1e-08 wannier_r=on`

```
====================================================================================================
Re dQ^{abc}   -- the metric part dg^{ab} (note Eq. 17)
====================================================================================================
  abc         total |      (i)   connection       |     (ii)  gap gradient      |      (iii) three-band       
                    |        value   share  weight|        value   share  weight|        value   share  weight
----------------------------------------------------------------------------------------------------
  xxx  -2.53684e-18 |  -2.53684e-18       -       - |             0       -       - |             0       -       - |   <- symmetry zero
  xxy      0.521171 |      0.521171  100.0%  100.0% |  -4.63171e-15   -0.0%    0.0% |             0    0.0%    0.0% |
  xyx      0.521171 |      0.521171  100.0%  100.0% |   2.38004e-15    0.0%    0.0% |             0    0.0%    0.0% |
  xyy   6.93889e-16 |   5.28882e-16       -       - |   5.82867e-16       -       - |             0       -       - |   <- symmetry zero
  yxx      0.521171 |      0.521171  100.0%  100.0% |   2.27422e-15    0.0%    0.0% |             0    0.0%    0.0% |
  yxy   8.88178e-16 |   5.51285e-16       -       - |   5.96745e-16       -       - |             0       -       - |   <- symmetry zero
  yyx  -1.77636e-15 |  -1.14664e-15       -       - |  -1.13798e-15       -       - |             0       -       - |   <- symmetry zero
  yyy     -0.521171 |     -0.521171  100.0%  100.0% |             0   -0.0%    0.0% |             0   -0.0%    0.0% |
----------------------------------------------------------------------------------------------------
  summed |piece| over the 4 nonzero components:  (i) 100.0%   (ii)   0.0%   (iii)   0.0%
```

Piece (iii) is **identically zero** — there is no third band to loop through.
Piece (ii) integrates to 9e-15 on a total of 0.52: zero, by the C3 argument in
[section 6.1](#61-piece-ii-dies-on-symmetry-not-on-smallness).

### 5.2 The same under 30% uniaxial strain along x — C3 broken, M_x kept

`model=honeycomb_aniso formulation=thermal nk=240 ef=0.0 kT=0.002 eta_sos=1e-08 wannier_r=on`

```
====================================================================================================
Re dQ^{abc}   -- the metric part dg^{ab} (note Eq. 17)
====================================================================================================
  abc         total |      (i)   connection       |     (ii)  gap gradient      |      (iii) three-band       
                    |        value   share  weight|        value   share  weight|        value   share  weight
----------------------------------------------------------------------------------------------------
  xxx  -2.19007e-18 |  -2.19007e-18       -       - |             0       -       - |             0       -       - |   <- symmetry zero
  xxy      0.566085 |       0.67722  119.6%   85.9% |     -0.111135  -19.6%   14.1% |             0    0.0%    0.0% |
  xyx       0.67722 |      0.621653   91.8%   91.8% |     0.0555677    8.2%    8.2% |             0    0.0%    0.0% |
  xyy  -1.17961e-15 |  -2.05639e-16       -       - |  -2.28983e-16       -       - |             0       -       - |   <- symmetry zero
  yxx       0.67722 |      0.621653   91.8%   91.8% |     0.0555677    8.2%    8.2% |             0    0.0%    0.0% |
  yxy  -1.11022e-15 |  -2.52377e-16       -       - |  -2.22045e-16       -       - |             0       -       - |   <- symmetry zero
  yyx   2.22045e-15 |   4.79285e-16       -       - |   4.02456e-16       -       - |             0       -       - |   <- symmetry zero
  yyy     -0.345397 |     -0.345397  100.0%  100.0% |             0   -0.0%    0.0% |             0   -0.0%    0.0% |
----------------------------------------------------------------------------------------------------
  summed |piece| over the 4 nonzero components:  (i)  91.1%   (ii)   8.9%   (iii)   0.0%
```

Breaking C3 brings piece (ii) to life (8.9% overall, −20% of `xxy`). It still
vanishes exactly for `yyy` and `xxx`, where a = c. (iii) is still identically
zero — still two bands.

### 5.3 Generic 4-band insulator, 2 occupied, no point group

`model=random4 formulation=thermal nk=240 ef=3.0 kT=0.02 eta_sos=1e-08 wannier_r=on`

```
====================================================================================================
Re dQ^{abc}   -- the metric part dg^{ab} (note Eq. 17)
====================================================================================================
  abc         total |      (i)   connection       |     (ii)  gap gradient      |      (iii) three-band       
                    |        value   share  weight|        value   share  weight|        value   share  weight
----------------------------------------------------------------------------------------------------
  xxx     0.0685336 |     0.0894886  130.6%   81.0% |             0    0.0%    0.0% |     -0.020955  -30.6%   19.0% |
  xxy    -0.0570357 |    -0.0579385  101.6%   98.4% |  -1.95753e-05    0.0%    0.0% |   0.000922416   -1.6%    1.6% |
  xyx    -0.0604129 |    -0.0659216  109.1%   92.3% |   9.78763e-06   -0.0%    0.0% |    0.00549893   -9.1%    7.7% |
  xyy     0.0478122 |     0.0428826   89.7%   89.7% |    0.00288506    6.0%    6.0% |    0.00204452    4.3%    4.3% |
  yxx    -0.0604129 |    -0.0659216  109.1%   92.3% |   9.78763e-06   -0.0%    0.0% |    0.00549893   -9.1%    7.7% |
  yxy     0.0478122 |     0.0428826   89.7%   89.7% |    0.00288506    6.0%    6.0% |    0.00204452    4.3%    4.3% |
  yyx     0.0453006 |     0.0494955  109.3%   87.1% |   -0.00577011  -12.7%   10.2% |    0.00157524    3.5%    2.8% |
  yyy    -0.0412886 |    -0.0390253   94.5%   94.5% |             0   -0.0%    0.0% |   -0.00226328    5.5%    5.5% |
----------------------------------------------------------------------------------------------------
  summed |piece| over the 8 nonzero components:  (i)  89.6%   (ii)   2.3%   (iii)   8.1%
```

This is the shape of the note's own numerical check, and it agrees with the
note's "(iii) is 10–40% of dQ^ab at generic k": BZ-integrated, (iii) is 8.1% of
the metric part and 34.8% of the curvature part.

### 5.4 MoS2 Wannier (11-band, 7 occupied)

`mos2_tb.dat` + `mos2_centres.xyz`, thermal, ef = −0.030102 (midgap),
kT = 0.025, eta = 0, nk = 200, eta_sos = 1e-4. In-plane only — the z position
blocks of this file are unreliable.

#### With the Wannier-gauge position correction (`system.wannier_r: true`, the default)

`model=mos2 formulation=thermal nk=200 ef=-0.030102 kT=0.025 eta_sos=0.0001 wannier_r=on`

```
====================================================================================================
Re dQ^{abc}   -- the metric part dg^{ab} (note Eq. 17)
====================================================================================================
  abc         total |      (i)   connection       |     (ii)  gap gradient      |      (iii) three-band       
                    |        value   share  weight|        value   share  weight|        value   share  weight
----------------------------------------------------------------------------------------------------
  xxx    0.00398355 |    0.00625633  157.1%   73.4% |             0    0.0%    0.0% |   -0.00227279  -57.1%   26.6% |
  xxy      0.333097 |     -0.051215  -15.4%   11.7% |  -0.000326122   -0.1%    0.1% |      0.384638  115.5%   88.2% |
  xyx      0.326918 |    -0.0614125  -18.8%   13.7% |   0.000163061    0.0%    0.0% |      0.388167  118.7%   86.3% |
  xyy   0.000362344 |   6.59481e-05   18.2%   18.2% |   0.000241068   66.5%   66.5% |   5.53283e-05   15.3%   15.3% |
  yxx      0.326918 |    -0.0614125  -18.8%   13.7% |   0.000163061    0.0%    0.0% |      0.388167  118.7%   86.3% |
  yxy   0.000362344 |   6.59481e-05   18.2%   18.2% |   0.000241068   66.5%   66.5% |   5.53283e-05   15.3%   15.3% |
  yyx    0.00331476 |    0.00379981  114.6%   88.7% |  -0.000482136  -14.5%   11.3% |  -2.92117e-06   -0.1%    0.1% |
  yyy     -0.335574 |     0.0609971  -18.2%   13.3% |             0   -0.0%    0.0% |     -0.396571  118.2%   86.7% |
----------------------------------------------------------------------------------------------------
  summed |piece| over the 8 nonzero components:  (i)  13.6%   (ii)   0.1%   (iii)  86.3%
```

#### Without it (`wannier_r: false`) — the point-like approximation r = −i v/w

`model=mos2 formulation=thermal nk=200 ef=-0.030102 kT=0.025 eta_sos=0.0001 wannier_r=off`

```
====================================================================================================
Re dQ^{abc}   -- the metric part dg^{ab} (note Eq. 17)
====================================================================================================
  abc         total |      (i)   connection       |     (ii)  gap gradient      |      (iii) three-band       
                    |        value   share  weight|        value   share  weight|        value   share  weight
----------------------------------------------------------------------------------------------------
  xxx   1.64014e-05 |   -0.00102433-6245.4%   49.6% |             0    0.0%    0.0% |    0.00104073 6345.4%   50.4% |
  xxy      0.317238 |      0.164671   51.9%   51.9% |    0.00374081    1.2%    1.2% |      0.148826   46.9%   46.9% |
  xyx      0.319576 |      0.173807   54.4%   53.8% |    -0.0018704   -0.6%    0.6% |       0.14764   46.2%   45.7% |
  xyy  -8.11017e-06 |  -0.000287138 3540.5%   50.7% |   5.42475e-06  -66.9%    1.0% |   0.000273603-3373.6%   48.3% |
  yxx      0.319576 |      0.173807   54.4%   53.8% |    -0.0018704   -0.6%    0.6% |       0.14764   46.2%   45.7% |
  yxy  -8.11017e-06 |  -0.000287138 3540.5%   50.7% |   5.42475e-06  -66.9%    1.0% |   0.000273603-3373.6%   48.3% |
  yyx  -1.70113e-05 |  -0.000688602 4047.9%   49.8% |  -1.08495e-05   63.8%    0.8% |    0.00068244-4011.7%   49.4% |
  yyy     -0.357864 |     -0.159594   44.6%   44.6% |             0   -0.0%    0.0% |      -0.19827   55.4%   55.4% |
----------------------------------------------------------------------------------------------------
  summed |piece| over the 8 nonzero components:  (i)  50.8%   (ii)   0.6%   (iii)  48.6%
```

#### Reading these two

| | correction **on** | correction **off** |
|---|---|---|
| dQ_yyy (total) | **−0.335574** | −0.357864 |
| (i) connection, yyy | +0.060997 (−18.2%) | −0.159594 (+44.6%) |
| (ii) gap gradient, yyy | 0 | 0 |
| (iii) three-band, yyy | **−0.396571 (+118.2%)** | −0.198270 (+55.4%) |
| weights (i)/(ii)/(iii), all 8 components | **13.6 / 0.1 / 86.3 %** | 50.8 / 0.6 / 48.6 % |
| C3 chain spread, (yyy, −yxx, −xxy, −xyx) | 2.6% | 12.4% |
| largest M_x-odd component | 1.2% of \|yyy\| (`xxx`) | 4.8e-5 of \|yyy\| (`yyx`) |

* **The three-band piece dominates MoS2** — 86% of the weight, and +118% of
  `dQ_yyy`, partly cancelled by a connection piece that has *flipped sign*
  relative to the uncorrected calculation. This is stronger than the note's
  "10–40% at generic k", and it is the concrete version of its remark that in
  MoS2 with the d-band manifold this is the piece a two-band model misses.
* **The correction changes the split qualitatively, not just the total.** The
  total moves by 6.2% (−0.3579 → −0.3356); the (i)/(iii) partition goes from
  roughly 50/50 to 14/86 and (i) changes sign. Any claim about which piece
  dominates MoS2 is a statement about the corrected operator.
* **The symmetry artefacts are the file's** — the same ones the skewness work
  found. Correction off, M_x holds to 1e-5 but the C3 chain is off by 12%;
  correction on, C3 tightens to 2.6% but ~1% M_x breaking appears (`xxx` =
  0.0040 and `yyx` = 0.0033, i.e. 1.2% and 1.0% of |yyy|) from the off-diagonal
  position blocks. The MoS2 Wannier centres are not on the atomic sites, so
  they respect M_x but not C3.
  A symmetry-adapted Wannierization would fix both.
* **Stable in nk**: the (iii) weight is 86.3 / 86.4 / 86.4 % at nk = 200 / 400 /
  600. Unlike the Liu models, this file has no unoccupied-band crossing in the
  way.

#### Why the split moves so much for a small correction

The swing from 50/50 to 14/86 is not the correction being large.  It is small
and uniform — BZ-RMS of the added `a` against the bare `rbar = -i v/w`:

```
block          rms rbar    rms a   |a|/|rbar|
occ-unocc        0.3626   0.0611      0.17
occ-occ          0.9070   0.0327      0.04
unocc-unocc      1.7679   0.0569      0.03
```

17% at most.  What amplifies it is the cancellation structure of section 6.2.
Notice which `rbar` elements are large: the **intra-manifold** ones (1.77 A
between unoccupied bands, 0.91 A within the occupied manifold) against 0.36 A
across the gap — the fundamental gap is 1.76 eV but bands within each manifold
are close, so `v/w` is big there.  Those are precisely the elements that appear
explicitly in piece (iii) and implicitly, through the Sipe three-band sum, in
piece (i).  They cancel between the two; the total never carries them.

So the pieces are large pointwise and the total is not:

```
MoS2 dQ^yyy, per-k integrand      (i) conn   (iii) 3band    total
corr ON  BZ mean                    0.0555      -0.3910    -0.3356
corr ON  pointwise rms              3.0058       2.5966     0.7561
corr OFF BZ mean                   -0.1707      -0.1872    -0.3579
corr OFF pointwise rms              3.2792       2.8188     0.7887
ON-OFF   BZ mean                    0.2261      -0.2038     0.0223
ON-OFF   pointwise rms              0.5756       0.5581     0.1707
```

Piece (i) has pointwise rms 3.0 and BZ average 0.056 — a **54x cancellation**,
against 2.3x for the total.  The correction changes each piece pointwise by
rms 0.58, which is 19% of 3.0 (exactly what the 17% block ratio predicts) but
**ten times** the surviving BZ average of (i).  A sign flip is the expected
outcome, not a surprising one.  The total's pointwise change is a comparable
22% of its own rms, but it only cancels 2.3x, so it moves by 6%.

This is specific to MoS2's manifold structure, not generic:

```
dQ^yyy: how much does each piece cancel under the BZ integral?
model                  (i) mean (iii) mean  tot mean |  (i) amp (iii) amp  tot amp
honeycomb  1occ/1un     -0.5212     0.0000   -0.5212 |     13.4       0.0     13.4
Liu MoS2   1occ/2un      0.0909    -0.0078    0.0831 |      1.8      15.1      1.5
random4    2occ/2un     -0.0390    -0.0023   -0.0413 |      3.0      19.7      2.7
MoS2 tb    7occ/4un      0.0555    -0.3910   -0.3356 |     54.2       6.6      2.3

amp = pointwise rms of the integrand / |its BZ average|
```

Everywhere else piece (i) and the total cancel by similar factors.  MoS2 is the
only case where (i) cancels 24x harder than the total, because it is the only
one with large manifolds on **both** sides and small intra-manifold gaps.

**What to take from it.**  Both splits are numerically converged and correct
for their respective operators; the sensitivity is physical, not numerical.
The robust statement is the one with the correct operator: (iii) = -0.397
against a total of -0.336, so the three-band loops carry the response.  The
correction-off split is not a meaningful comparison — it is a different
operator, and the piece it moves most is the one with the least robust value.
Corollary worth remembering: **piece (i) on MoS2 will also be sensitive to the
Wannierization itself** (disentanglement window, spread minimisation), not just
to this switch.  Quote it as "the three-band piece dominates", not as a number
to two digits.

#### eta_sos sensitivity (dQ_yyy, nk = 200)

The **total** barely moves; the **split** does, because (i) and (iii) cancel.

```
eta_sos     total        (i) conn     (iii) 3band    weights (i)/(iii)
0.05      -0.335528      +0.031582     -0.367111        8.5 / 91.4 %
0.025     -0.335563      +0.047319     -0.382882       11.5 / 88.4 %
1e-2      -0.335573      +0.056611     -0.392183       12.6 / 87.4 %
1e-3      -0.335574      +0.060708     -0.396282       13.3 / 86.7 %
1e-4      -0.335574      +0.060997     -0.396571       13.6 / 86.3 %
1e-5      -0.335574      +0.061000     -0.396573       13.6 / 86.3 %
```

Converged below 1e-4. **The 0.025 used in the chi2 ↔ dQ sum-rule work is not
small enough to quote a split from.** `thermal` and `subspace` give identical
splits.

### 5.5 Liu three-band TMD models (24 models)

`examples/liu_tmd/`, Liu et al. PRB 88, 085433. Three metal d orbitals (d_z2,
d_xy, d_x2−y2) on the metal site, so `atompos = 0`, `r = −i v/w` is exact and
there is no Wannier correction and no connection curvature. One occupied band:
the **smallest system in which piece (iii) exists at all** (the q' sum has one
term, the p' sum over occupied intermediates is empty). Lengths in units of a,
so dQ is in a³. ef at midgap, kT = 0.01, eta_sos = 1e-6.

```
material model xc         ef     dQ_yyy   (i) conn  (ii) grad (iii) 3band    w(i)  w(ii)    w3(480)    w3(960)  C3 spread     x-odd
-----------------------------------------------------------------------------------------------------------------------------------
MoS2     NN    GGA     0.770   -0.00725    0.00347    0.00000    -0.01072   24.5%   0.0%      75.5%      75.5%    9.9e-15   8.5e-16
MoS2     NN    LDA     0.971   -0.00836    0.00277    0.00000    -0.01113   19.9%   0.0%      80.1%      80.1%    3.5e-15   6.4e-16
MoS2     TNN   GGA     0.767    0.08310    0.08950    0.00000    -0.00641   93.3%   0.0%       6.8%       6.7%    2.8e-15   1.6e-16
MoS2     TNN   LDA     0.972    0.07929    0.08832    0.00000    -0.00902   90.7%   0.0%       9.3%       9.3%    1.4e-15   2.1e-16
WS2      NN    GGA     0.845   -0.01465   -0.00533    0.00000    -0.00932   36.4%   0.0%      63.7%      63.6%    1.8e-15   6.1e-16
WS2      NN    LDA     1.081   -0.01503   -0.00661    0.00000    -0.00842   44.0%   0.0%      56.0%      56.0%    6.9e-15   3.1e-16
WS2      TNN   GGA     0.846    0.07635    0.08779    0.00000    -0.01144   88.5%   0.0%      11.6%      11.5%    3.8e-15   9.7e-17
WS2      TNN   LDA     1.080    0.07616    0.08887    0.00000    -0.01271   87.5%   0.0%      12.6%      12.5%    2.2e-15   2.0e-16
MoSe2    NN    GGA     0.765   -0.00207    0.01050    0.00000    -0.01257   45.5%   0.0%      54.5%      54.5%    1.2e-14   2.5e-15
MoSe2    NN    LDA     0.857   -0.00349    0.00996    0.00000    -0.01345   42.5%   0.0%      57.5%      57.5%    1.1e-14   2.3e-15
MoSe2    TNN   GGA     0.767    0.10180    0.11133    0.00000    -0.00952   92.1%   0.0%       7.9%       7.9%    3.5e-15   2.1e-16
MoSe2    TNN   LDA     0.858    0.10920    0.12223    0.00000    -0.01303   90.4%   0.0%       9.9%       9.6%    1.3e-15   8.0e-17
WSe2     NN    GGA     0.794   -0.01154   -0.00019    0.00000    -0.01135    1.6%   0.0%      98.3%      98.4%    6.6e-15   5.7e-16
WSe2     NN    LDA     0.984   -0.01224   -0.00078    0.00000    -0.01146    6.4%   0.0%      93.7%      93.6%    3.4e-15   6.8e-16
WSe2     TNN   GGA     0.794    0.09694    0.11084    0.00000    -0.01390   88.9%   0.0%      11.3%      11.1%    3.1e-15   5.9e-17
WSe2     TNN   LDA     0.943    0.10459    0.11781    0.00000    -0.01323   89.9%   0.0%      10.4%      10.1%    4.6e-15   1.1e-16
MoTe2    NN    GGA     0.577    0.00586    0.01834    0.00000    -0.01248   59.5%   0.0%      40.5%      40.5%    1.1e-14   2.1e-15
MoTe2    NN    LDA     0.609    0.00366    0.01654    0.00000    -0.01288   56.2%   0.0%      43.8%      43.8%    1.6e-14   3.8e-15
MoTe2    TNN   GGA     0.577   -0.02516    0.00833    0.00000    -0.03349   19.9%   0.0%      80.1%      80.1%    3.7e-15   1.3e-15
MoTe2    TNN   LDA     0.607    0.11446    0.12630    0.00000    -0.01184   91.4%   0.0%       8.6%       8.6%    1.8e-15   2.7e-16
WTe2     NN    GGA     0.598   -0.00699    0.00224    0.00000    -0.00923   19.5%   0.0%      80.5%      80.5%    1.3e-14   9.5e-16
WTe2     NN    LDA     0.630   -0.00848    0.00046    0.00000    -0.00894    4.9%   0.0%      95.1%      95.1%    3.7e-15   1.2e-15
WTe2     TNN   GGA     0.599    0.13631    0.14133    0.00000    -0.00502   96.6%   0.0%       3.6%       3.4%    1.4e-15   1.5e-16
WTe2     TNN   LDA     0.633    0.14719    0.14916    0.00000    -0.00197   98.7%   0.0%       1.5%       1.3%    5.7e-16   9.3e-17
-----------------------------------------------------------------------------------------------------------------------------------
dQ in a^3.  Pieces are the yyy component at nk = 960; the weights w are
sum|piece| / sum sum|piece| over all 8 components.
w3(nk) is the three-band weight at each grid — its spread is the
conical-crossing sampling error of section 6.2, not a bug.
C3 spread = spread of (yyy, -yxx, -xxy, -xyx) / their mean;
x-odd = largest M_x-forbidden component / |yyy|.
```

* **(ii) is exactly 0.00000 in all 24** — to the last digit, not merely small.
* **NN and TNN are qualitatively different objects for dQ.** TNN gives
  dQ_yyy = +0.076 … +0.147, connection-dominated (three-band 1.3–12.5%). NN
  gives −0.015 … +0.006 — an order of magnitude smaller, usually of the
  *opposite sign*, and a near-cancellation dominated by the three-band piece
  (40–98%). WSe2 NN GGA is the extreme: (i) = −0.00019 against
  (iii) = −0.01135. **Do not use the NN parameterization for dQ.**
* **MoTe2 TNN is functional-unstable**: GGA −0.025 with (iii) at 80%, LDA
  +0.114 with (iii) at 8.6%, where every other material agrees GGA/LDA to ~10%.
  Treat it as unconverged physics, not as a number.
* **These models are exactly symmetric**, unlike the MoS2 `_tb.dat` file: the C3
  chain yyy = −yxx = −xxy = −xyx closes to 1e-14…1e-16 and the M_x-odd
  components are 1e-15…1e-16 of \|yyy\|. The symmetry breaking seen on the
  Wannier file really is the file's.

#### Why NN and TNN differ so much

They are not nested. NN keeps the on-site terms and the first shell only (6
nearest metal neighbours, |R| = a; 8 parameters, Liu Table II, Eqs. 4–10). TNN
adds the second (|R| = √3·a) and third (|R| = 2a) shells, 6 neighbours each (19
parameters, Table IV, Eqs. 13–21). Setting r = u = 0 in TNN gives the NN *form*,
but the two are **independent fits** and the 8 shared parameters are refitted:

```
MoS2 GGA   param   NN fit   TNN fit
           e1       1.046    0.683
           t0      -0.184   -0.146
           t1      +0.401   -0.114     <- sign flip
           t2       0.507    0.506     <- the only survivor
           t11      0.218    0.085     <- x0.4
           t12      0.338    0.162     <- x0.5
           t22      0.057    0.073
```

and the outer shells are not small: in the TNN fit u11 = 0.266 eV exceeds the
1NN t11 = 0.085 eV. Both fits reproduce the band edges — at Gamma and K the two
agree to ~3 meV — and part company elsewhere: at M the valence band differs by
0.12 eV and the upper conduction band by 0.83 eV, and the valence bandwidth is
0.510 eV (NN) vs 0.864 eV (TNN).

Anything that is a BZ integral weighted toward the regions NN gets wrong will
differ qualitatively, and this is not peculiar to dQ. Same models, same grid:

```
MoS2 GGA     variance_xx   skewness_yyy
NN               0.17240       -0.01644
TNN              0.27752       +0.11919    <- sign flip here too
```

The paper's Table S1 lists both and the engine reproduces both, which is a
statement about the engine, not about the two models agreeing with each other.
**For any BZ-integrated geometric quantity, TNN is the one to use.**

---

## 6. Two caveats on the split

### 6.1 Piece (ii) dies on symmetry, not on smallness

Two independent mechanisms, which is why it keeps coming out as a machine zero:

* **Pointwise, whenever a = c.** `Im(r^a_pq r^c_qp) → Im|r^a_pq|² = 0`. This
  kills `yyy` and `xxx` in every model above, exactly.
* **After BZ integration, for any C3 crystal.** Piece (ii) is
  `d_b(scalar) × (two-state Berry curvature)`; in 2D the curvature is a
  pseudoscalar, so the object is a pseudovector in the free index b, and C3
  admits no invariant vector or pseudovector. Hence ~0 on the C3 honeycomb,
  0.1% on MoS2, and exactly 0 in all 24 Liu models.

Break C3 and it revives: 8.9% under 30% uniaxial strain, 2.3% on a model with no
point group at all.

### 6.2 The pieces carry intra-manifold denominators that dQ does not

This is the one genuinely surprising structural fact, and it is a property of
the *decomposition*, not of the physics.

`dQ` itself only ever sees occupied–unoccupied gaps. Both `(d_a P)_pq` and
`(delta P)_qp` carry `f_pq`, so they vanish unless one index is occupied and the
other is not. That is by construction and it holds.

The note's **residual blocks** (its Eq. 10) do not share that property. They are
the two ways of stepping off the pair (q,p) by one interband hop:

```
  sum_{q' != q}  r^a_pq'  r^b_q'q        q, q'  both UNOCCUPIED
- sum_{p' != p}  r^a_p'q  r^b_pp'        p, p'  both OCCUPIED
```

so piece (iii) carries `1/w` between two unoccupied bands and `1/w` between two
occupied bands, and piece (i) carries the equal and opposite ones inside the
Sipe sum rule, whose intermediate index runs over **all** bands. They cancel
pointwise, and the total is smooth.

**Consequence: where two bands touch inside a manifold, each piece diverges as
1/|dk| and the total does not.** Measured on the Liu models, whose two
unoccupied bands cross conically at six points at |k| ≈ 2.12 (they are also
degenerate at Gamma, but quadratically, and that one is harmless):

```
distance d from the crossing, MoS2 TNN GGA, dQ^yyy at that k
     d        E2-E1      (i) conn     (iii) 3band        total
  1.0e-01   9.0094e-02  -5.97682e-02   1.99623e-02   -3.98059e-02
  1.0e-02   9.0662e-03  -6.00167e-02   1.90232e-02   -4.09935e-02
  1.0e-04   9.0668e-05  -6.00193e-02   1.90137e-02   -4.10056e-02
```

Along that particular direction the residue happens to vanish; along a generic
direction `|piece| × (E2−E1) = 0.0124` exactly — a clean 1/|dk| cusp with a sign
that changes around the crossing. Scanning the whole grid, `max |(iii)|` grows
3.54 → 5.06 as nk goes 300 → 600, while the mean stays at −0.0064.

The cusp is **absolutely integrable in 2D** (the radial Jacobian cancels it), so
each piece does have a well-defined BZ integral. But a uniform grid converges
only O(h), with a coefficient set by where the grid falls relative to the
crossing — so the split scatters while the total is converged to 7 digits:

```
nk      120      240      300      480      600      960     1200     1440     1680
(iii) -.00781  -.00675  -.00632  -.00649  -.00645  -.00641  -.00636  -.00720  -.00647

odd nk (grid misses Gamma):   241  -.00569     481  -.00654     961  -.00644

total   0.0830961 at every single one of them
```

nk = 1440 is a **reproducible** outlier, not noise — the signature of an O(h)
grid-offset error rather than random scatter. So quote the split as a band:
(iii) = 6.7 ± 0.4 % for MoS2 TNN GGA. The *weights* are much steadier than the
values — w3(480) and w3(960) in the table above agree to ≤ 0.3 percentage points
everywhere.

The note never met this: its own four-band check had "minimum band separation
0.079" on a 25×25 grid. **Check an nk ladder before quoting a split**, and
expect non-monotone outliers rather than noise.

---

## 7. Reproducing all of this

```bash
# certificate (25 checks)
python3 examples/test_dQ_term_groups.py

# single model, full 8-component tables
python3 examples/dQ_term_groups.py honeycomb       --nk 240
python3 examples/dQ_term_groups.py honeycomb_aniso --nk 240
python3 examples/dQ_term_groups.py random4         --nk 240
python3 examples/dQ_term_groups.py liu --material MoS2 --liu-model TNN --functional GGA

mpiexec -np 22 python3 examples/dQ_term_groups.py mos2 \
    --mos2-dir /path/with/mos2_tb.dat --nk 200 --eta-sos 1e-4 --wannier-r on
mpiexec -np 22 python3 examples/dQ_term_groups.py mos2 \
    --mos2-dir /path/with/mos2_tb.dat --nk 200 --eta-sos 1e-4 --wannier-r off

# all 24 Liu models, plus the nk study behind section 6.2
mpiexec -np 22 python3 examples/liu_tmd/dQ_table.py --nk 480,960 --eta-sos 1e-6
mpiexec -np 22 python3 examples/liu_tmd/dQ_table.py --converge
```

In your own analysis:

```python
from tightbinding.main import load_delta_Q
res, cfg = load_delta_Q('results/dQ')
g = res['delta_Q_groups']['y']['y']['y']       # connection / gap_gradient / three_band

# older file without the key:
from tightbinding.calc.delta_Q import group_terms
g = group_terms(res['delta_Q_terms']['y']['y']['y'])
```

### Files

| path | what |
|---|---|
| `tightbinding/calc/delta_Q.py` | `NOTE_GROUPS`, `group_terms()`, `delta_Q_groups` in the result |
| `tightbinding/main.py` | saves / reloads `delta_Q_groups` |
| `examples/test_dQ_term_groups.py` | the certificate |
| `examples/dQ_term_groups.py` | single-model driver, all the models above |
| `examples/liu_tmd/dQ_table.py` | the 24-model sweep and the `--converge` study |
| `examples/liu_tmd/RESULTS_dQ.md` | the Liu table on its own |
| `delta_Q_multiband_expansion.pdf` | the note (repo root) |
