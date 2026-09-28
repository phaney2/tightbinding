# Session summary, 2026-09-10: Wannier-gauge fix, sum rule, cluster recipe

Two commits landed on `master` (pushed to `origin`):

- `aab6d5a` Fix the Wannier-gauge position correction; make every engine gauge-covariant
- `047db3c` Add the chi^(2) <-> delta_Q sum-rule check and record its findings

The detailed records are `BUG_wannier_r_correction.md` (resolution at the top,
original report kept below), the "Wannier-Gauge Position Correction" and
"The chi^(2) <-> delta_Q sum rule" sections of `CLAUDE.md`, and the docs.
This file is the one-page version plus the cluster recipe worked out at the end.

## 1. The bug (closed)

`BUG_wannier_r_correction.md` reported a complex `delta_Q` and an
eta-independent sub-gap "dissipation" in chi^(2) on MoS2 `_tb.dat` input.
Five separate defects were found, none of them in the Fourier kernel the
report pointed at:

1. The Eq. 36 covariant-derivative algebra, copy-pasted into three call sites,
   lacked the factor i on the commutator (anti-Hermitian, the complex delta_Q),
   had the wrong sign on the xi.a term, and omitted both xi.rbar terms.
   Now one function, `apply_wannier_correction`, derivation in its docstring.
2. Only r was corrected, never the velocity. The current vertex is
   v = i[H, r] with the full r; chi^(2), the subspace/band delta_Q
   integrands and quantum_metric's perturbation now use the corrected operator.
   thermal and subspace delta_Q agree to 1e-6 on MoS2 (10% apart before).
3. Wannier90's `_tb.dat` position blocks are not Hermitian-paired; the loader
   now symmetrizes them, as postw90 does.
4. Zeroing the R=0 position diagonal assumed atompos was built from the
   centres; without a centres file the Bloch sum ran in the lattice gauge and
   the intra-cell term was dropped. `_tb.dat` input now always runs in the
   atomic gauge, the kernel subtracts the centres atompos encodes, and the
   loader checks/repairs the file's R=0 diagonal (its z entries are scrambled
   for this MoS2: Mo sits at z = c/2 on the Berry-phase branch cut).
5. `chi_ii` used diag(d^2H) as the band curvature, missing the interband sum
   rule. Insulators are unaffected; every metallic chi_ii before the fix is wrong.

The tool that found 2 and 5, and now guards all of it: an exact
gauge-covariance certificate (`examples/test_wannier_gauge.py`, 41 checks).
A point-like multi-atom model re-expressed in another gauge plus the
correction must reproduce its atomic-gauge answer, at the operator level
(1e-13) and for every engine's per-k output (1e-14); the k-dependent term is
checked against finite differences on the real MoS2 connection (1e-7).

Results on MoS2 with the correction on: dQ_yyy = -0.3356, real to 1e-9 in
every component (was -0.533 - 0.200j); below the gap Re chi^(2) (dissipative
in this code's conductivity convention) scales linearly in eta (ratio 0.1275
over an 8x ladder vs 0.125) where it grew as eta -> 0 before. The correction
is a 27% effect on dQ_yyy. `WANNIER_R_DEFAULT` is True again. TB_simple
benchmarks are bit-identical; all regression suites pass.

Two things learned the hard way, now in `CLAUDE.md`:

- `chi_total` is the second-order *conductivity*: **Im is reactive, Re is
  dissipative.** Test 2 of the original bug report had it backwards.
- `quantum_metric`'s finite-difference `dQ` is a bare-velocity heuristic and
  is not gauge-covariant with the correction on; `Q` and `dQf` are. Use
  `calc.type: delta_Q`.

## 2. The sum rule  int dw (1/w) Re chi^{abc}(w; w, 0) = pi dQ^{ab}(c)

Driver: `examples/check_sumrule_chi2_dQ.py`. Findings:

- The index pairing is the plain one (output a, omega-field b, DC-field c
  with dQ^{ab}(c)); alternatives fail by 15-20% on the anisotropic honeycomb.
- Finite eta costs O(eta ln(1/eta)) with no constant term. MoS2 yyy:
  1.00133 / 1.00074 / 1.00041 at eta = 1e-3 / 5e-4 / 2.5e-4, extrapolating to
  1 to better than 1e-4; 1.016 at eta = 0.025. The old "7e-6" was an
  eta -> 0 number.
- With the Wannier correction on, the rule holds component-wise only for
  a = c (yyy, xyx, xxx). For a != c the deviation is antisymmetric under
  a <-> c (MoS2: xxy 0.9786, yxx 1.0243, their sum at the eta offset). It is
  neither Hamiltonian symmetry breaking nor time reversal; it is the
  non-Abelian curvature F^{ac} of the Wannier connection (proven with a flat
  vs a curved synthetic connection of equal size; MoS2 has |F^{xy}| of
  0.4-2.8 A^2). Decision: record it, no symmetrized redefinition.

**For the paper: the yyy sum rule is exact with the correction on.**

## 3. Reference values, MoS2 (mos2_tb.dat + centres, wannier_r on)

Ef = -0.030102 (midgap; VBM -0.910647, CBM +0.850443, gap 1.7611),
kT = 0.025, eta_sos = 0.025, omega from 10 eta to infinity. J = int dw (1/w)
chi_total_yyy(w), k-converged to the sixth digit at nk >= 100:

| eta (eV) | Re J_yyy | Im J_yyy |
|---|---|---|
| 0.05 | -1.08074 | -0.074691 |
| 0.025 | -1.07104 | -0.037689 |
| 0.01 | -1.06299 | -0.015112 |
| 1e-3 | -1.05561 | -0.0015119 |
| 5e-4 | -1.05498 | -0.00075595 |
| 2.5e-4 | -1.05463 | -0.00037798 |

pi dQ_yyy = -1.05420 (dQ_yyy = -0.335563, thermal). Im J = -1.512 eta.

Cluster run by the user (nk = 600x600, eta = 0.025, eta_sos = 1e-8, Ef = 0,
1900 points on [0.25, 12] eV): Re J = -1.07106027, Im J = +0.13412227.
The analytic value on the same interval is -1.071060 + 0.134122j, i.e. the
quadrature is exact to six digits in both parts. The Im part differs from the
table above only because of the tail above 12 eV: the reactive part decays
as 1/w and the tail contributes -0.172 to Im J, versus 2e-5 to Re J.

## 4. Recipe for reproducing the analytic integral from a sampled chi(w)

Calibrated on MoS2 by comparing trapezoid quadrature of the sampled path
against the analytic value on the same grid and eta.

- **Energy range: 10 eta to 12 eV**, same limits on the analytic side (or
  omega_max = inf there; the Re tail beyond 12 eV is 0.002%). Below 12 eV is
  not enough: 8 eV is 3.6% off, 4 eV a factor 2, because the 1.8-4 eV region
  (-2.35) cancels against 4-10 eV (+1.1).
- **Energy mesh: Delta_w <= eta/4**, trapezoid of Re chi / w. Error is 0.3% at
  eta/3 and 6% at 0.6 eta, then it blows up (the triple-pole terms peak at
  1/eta^3 and integrate to O(1)). Simpson is worse. About 1900 points at
  eta = 0.025, 4800 at eta = 0.01, uniform in w.
- **k-mesh: nk ~ 14 eV / eta** for the frequency-resolved curve (600 at
  eta = 0.025, 1400 at 0.01). Pointwise, nk = 300 is already within 0.16% of
  nk = 600 at eta = 0.025 (nk = 150 within 1.7%). The integral itself is
  k-converged at nk = 100.
- **Cost, single core, one direction:** ~0.035 ms per (k-point, w-point) plus
  ~10 ms per k-point; ~0.1 GB RAM per 1000 w-points per rank. Run one
  direction per job (four directions at 4096 points cost 6x per k-point).
  eta = 0.025 / nk 600 / 1900 points: ~8 core-hours. eta = 0.01 / nk 1400 /
  4800 points: ~100 core-hours. `MKL_NUM_THREADS=1`, one rank per core.
- Same kT and eta_sos on both sides; eta_sos (0.025 vs 1e-8) and Ef (0 vs
  midgap) move Re J at the 3e-5 level only.

## 5. Loose ends

- Set the git identity on this machine (`git config --global user.name/email`);
  the two commits were made with per-commit flags. `origin` now uses SSH.
- The local branch `fix/wannier-gauge-correction` is merged and can be deleted.
- Do not use the z position operator from this MoS2 file (scrambled centres).
- This file is untracked; commit it or delete it as you prefer.
