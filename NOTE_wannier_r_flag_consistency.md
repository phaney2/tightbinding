# NOTE: `wannier_r` across calculation engines — RESOLVED

**Status:** fixed 2026-08-27. Requested the same day; this file is now the
record of what was decided and why, not an open item.
**Related:** `BUG_wannier_r_correction.md` (resolved 2026-09-10 — the
correction itself was defective in five ways; that file has the account).
This note was always about the *switch*, not the physics. Two of its
statements are superseded and marked below.

---

## 1. What was wrong

Three engines used a position operator and all three behaved differently:

| engine | position operator | Wannier correction | switch |
|---|---|---|---|
| `nonlinear_optical` (+ `_fast`) | Eq. 22 corrected `r` | always applied | none — hard-wired ON |
| `delta_Q` | Eq. 22 corrected `r` | applied by default | `calc.wannier_r`, default `True` |
| `quantum_metric` | bare TBA `r = -i v / w` | never applied | none — hard-wired OFF |
| `jdos`, `all_ek`, `bands` | none | n/a | n/a |

Two consequences: chi^(2) could not be run with the suspect term removed
while `delta_Q` could (backwards, for debugging a bug both share), and
`Q` and its own DC response `dQ` sat on opposite sides of the correction.

## 2. What was done

**One switch, one kernel, in a new module `calc/wannier_gauge.py`.**

- `resolve_wannier_r(cfg, system, engine)` — the only reader of the flag.
- `compute_A_W_k(system, k, dir_chars, enabled=..., need_deriv=...)` — the
  only kernel, moved here from `nonlinear_optical.py` (which keeps
  `_compute_A_W_k` as an alias). It returns `(None, None)` both when
  disabled and when the system has no position matrices, so every call
  site keeps its existing `if A_W is not None:` guard and needs no branch
  for the flag.
- `validate_wannier_r`, `warn_unused_wannier_r`, `system_has_wannier_r`,
  `offdiag_A_H`, `reset_notices`.

Any future engine with an `r` operator must go through those two
functions. That is the durable part of the fix; the rest is plumbing.

### Design decisions (the three the original note left open)

**Flag location: `system.wannier_r`.** It describes the gauge of the
system's position operator, not the calculation, and the position
matrices it acts on arrive with `wannier_tb`. `calc.wannier_r` now
**raises** with a message saying where it went — a hard error rather than
a silent ignore or a soft alias, since silently dropping a stale key is
exactly the failure this whole exercise was about. A non-boolean value
raises too. `config.load_config` runs the same validator, so YAML
mistakes fail at load rather than mid-sweep.

**Default: `False`, temporarily.** *(Superseded 2026-09-10: the default is
`True` again, both settings print a one-line note, and `False` is the
point-like-orbital approximation in the atomic gauge.)* While the bug was
open, `False` was the diagnostic setting, chosen so new results did not
silently inherit the defect. TB_simple models are unaffected bitwise.

**`quantum_metric`: the correction was implemented, not refused.** `Q`,
`dQ` and `dQf` are all quadratic forms in `r`, now built through a single
`_rr_sum` helper so they cannot use different position operators. The
perturbed connection for the finite-difference `dQ` is `A^(W)` rotated by
the perturbed states, `psip† A^(W) psip`; `pert` is anti-Hermitian, so
`psip` is unitary to first order and that rotation is legitimate. Only
the states are perturbed — `1/de` stays unperturbed, matching the
pre-existing scheme. Masking follows chi^(2)/`delta_Q`: the degeneracy
mask applies to the `1/w` factor only, and `a^(H)`, smooth across a
degeneracy, is added unmasked.

*(Added 2026-09-10: the perturbation vertex `pert` now uses the full
`r = -i v/w + a^(H)` too. Even so the finite-difference `dQ` is not
gauge-covariant with the correction on — `Q` and `dQf` are — because
rotating the bare velocity by the perturbed states and dividing by the
unperturbed `1/de` is a heuristic; see `examples/test_wannier_gauge.py`
and the technical reference. The engine prints a note.)*

`_rr_sum` is written expanded rather than as `r1 * conj(r2)`, keeping the
leading term in its original factor order. That is not fussiness: `dQ` is
a finite difference of two nearly equal sums, so a 1-ulp reassociation is
amplified by `|Q| / (delta |dQ|)` — about 1e8 wherever `dQ` is small. The
expanded form makes the correction-off path bit-exact.

## 3. Known gaps, documented rather than fixed

- **`calc.method: projector`** rebuilds `chi_e1`/`chi_e2` from `H(k)`
  projectors, which carry no Wannier correction, so those two terms are
  effectively `wannier_r=False` whatever the flag says. The engine prints
  a note. Both are unphysical and excluded from `chi_total`.
- **`_process_kpoint_fast(_k_data=...)`** skips the Phase-1 operator
  build entirely, so the flag has no effect on that path. Documented in
  the docstring.
- **`quantum_metric` still uses a hard `DEG_THR = 1e-5` cutoff** where
  chi^(2) and `delta_Q` use Souza `eta_sos` regularization. Unrelated to
  this change and left alone.
- **The Eq. 36 algebra was still copy-pasted into three call sites** by
  this change; it was wrong in all three and is now one function,
  `apply_wannier_correction` (2026-09-10).

## 4. Verification

`examples/test_wannier_r_flag.py`, 31 checks. Because no `_tb.dat` file
exists on this machine, the test decorates the gapped honeycomb with
synthetic `wannier_r_matrices` built with `r(-R) = r(R)^dagger`, so
`A^(W)(k)` is Hermitian as a real `_tb.dat` gives and the correction
actually bites. It checks, for chi^(2) fast, chi^(2) reference, `delta_Q`
and `quantum_metric`: off == undecorated (bitwise); on != off; slow ==
fast at *both* settings; the keyword default is `WANNIER_R_DEFAULT`;
`system.wannier_r` reaches each engine from the config; `calc.wannier_r`
raises; the flag is inert bitwise on a system with no position matrices;
and `Q`/`dQ`/`dQf` with the correction off reproduce the pre-refactor
algebra (bitwise, checked inline — the example config
`input_qm_test.yaml` gives machine-zero `dQ` and would not have exercised
the finite-difference path at all).

Regressions, all passing and unchanged: `test_freq_integral.py` including
its `git worktree` baseline comparison at HEAD, `test_delta_Q_projector.py`,
`honeycomb_warp/check_eq13.py`. The three benchmark configs
(`input_qm_test`, `input_nonlinear_test`, `input_delta_Q_test`) are
**bit-identical** before and after.

`examples/test_wannier_gauge.py` (2026-09-10) adds the physics
certificates: gauge covariance at the operator and engine level, a
finite-difference check with a k-dependent connection, Hermiticity, and an
optional real-`_tb.dat` section.

## 5. What is *not* affected

For a model built with `build_system` + `fill_hamiltonian`, or from
`_hr.dat`, there are no `wannier_r_matrices` and the correction is a no-op
regardless of the flag — the run banner now says so explicitly instead of
leaving it ambiguous. The chi^(2) / `delta_Q` results in the honeycomb
directories, including the `int dw (1/w) Re chi_yyy = pi * dQ^yyy` sum
rule verified there to 7e-6, are untouched by this and by the open bug.
The issue bites only Wannier `_tb.dat`-derived systems, i.e. the TMD set.
