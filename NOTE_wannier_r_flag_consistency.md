# NOTE: `wannier_r` is not consistent across calculation engines

**Status:** documented, not fixed. Requested 2026-08-27.
**Related:** `BUG_wannier_r_correction.md` (open — the correction itself is
believed defective). This note is about the *switch*, not the physics.

**Design decisions (flag name, where it lives in the config, default value)
are deliberately left open** — the requester will settle those when
implementing.

---

## 1. The actual state

Three engines use a position operator, and all three behave differently:

| engine | position operator | Wannier correction | switch |
|---|---|---|---|
| `nonlinear_optical` (+ `_fast`) | Eq. 22 corrected `r` | **always applied** | **none — hard-wired ON** |
| `delta_Q` | Eq. 22 corrected `r` | applied by default | `calc.wannier_r`, default `True` |
| `quantum_metric` | bare TBA `r = -i v / w` | **never applied** | **none — hard-wired OFF** |
| `jdos`, `all_ek`, `bands` | none | n/a | n/a |

> **Note on a common misremembering:** it is chi^(2) that *lacks* the flag and
> `delta_Q` that *has* it, not the other way round. `grep -c 'wannier_r'`
> gives 0 for `nonlinear_optical.py`, `nonlinear_optical_fast.py` and
> `quantum_metric.py`, and 6 for `delta_Q.py`.

Two separate problems follow:

1. **You cannot turn the correction off in chi^(2) at all.** Since
   `BUG_wannier_r_correction.md` is open and both engines share the defect
   (`delta_Q` imports `_compute_A_W_k` from `nonlinear_optical` rather than
   reimplementing it), there is currently no way to run the chi^(2) side of a
   chi^(2)-vs-`delta_Q` comparison with the suspect term removed — while the
   `delta_Q` side can be. That asymmetry is exactly backwards for debugging.

2. **`quantum_metric` and `delta_Q` disagree with each other.** `Q` is
   computed with the bare TBA position operator and `delta_Q` — its own DC
   field response — with the corrected one. Whatever the right answer is, `Q`
   and `dQ` should not be on opposite sides of it. `quantum_metric` is
   effectively pinned at `wannier_r=False` with no way to say so.

---

## 2. Where the gating would go

### `delta_Q.py` — the existing pattern to copy

```
107    wannier_r = bool(calc.get('wannier_r', True))     # read from config
142    ...printed in the run banner...
212    wannier_r=wannier_r,                              # passed to _process_kpoint
246    def _process_kpoint(..., wannier_r=True, ...)     # signature
329-332  if wannier_r: A_W, dA_W = _compute_A_W_k(...)   # else A_W, dA_W = None, None
334    if A_W is not None:                               # the apply block
```

The `if A_W is not None:` guard already exists in every engine (it is how a
non-Wannier system is handled — see §4), so gating is only a matter of forcing
`A_W = None` earlier. No changes are needed inside the apply blocks.

### `nonlinear_optical_fast.py` — the path that actually executes

```
119    def _process_kpoint_fast(...)      # needs a new keyword
184    from .nonlinear_optical import _compute_dk_rmtx, _compute_A_W_k
197    A_W, dA_W = _compute_A_W_k(system, k, dir_chars)     # <-- gate here
198    if A_W is not None:
```

Threading required: `compute_nonlinear_optical` reads `calc` and calls
`_process_kpoint_fast` — the flag has to be read alongside `eta_sos` and
passed down the same way. Note `_process_kpoint_fast` is also called directly
by `examples/test_freq_integral.py` and by scratch scripts, so give the new
keyword a default rather than making it positional.

### `nonlinear_optical.py` — the slow/reference path

```
401    def _process_kpoint(...)           # same treatment
476    A_W, dA_W = _compute_A_W_k(system, k, dir_chars)     # <-- gate here
477    if A_W is not None:
```

Easy to miss because it is dead code in normal runs (`_process_kpoint_fast`
supersedes it), but it is the reference implementation and will silently
disagree with the fast path if only one is gated.

### `quantum_metric.py` — nothing to gate yet

`_process_kpoint` (line 142) builds the metric directly from
`vmtx * conj(vmtx) * inv_de2` (lines ~229-249) and never imports
`_compute_A_W_k`. Making the flag *consistent* here means either
(a) accepting a documented `wannier_r=False`-only engine and rejecting
`wannier_r: true` in its config with a clear error, or
(b) adding the correction so `Q` and `dQ` agree. (b) is the larger job and is
a physics decision, not a plumbing one.

---

## 3. Interaction with the open bug

`BUG_wannier_r_correction.md` warns: *"Do not just set `wannier_r=False`"* —
the Wannier gauge genuinely matters, and `False` is not the correct answer,
only a diagnostic. That advice should survive into whatever the unified flag
becomes: it is a debugging switch, not a physics preference. Consider making
the non-default setting noisy (a warning naming the bug note) rather than
silent.

A single shared reader would also stop the three engines drifting apart again;
right now the only thing tying them together is that they all call the same
`_compute_A_W_k`.

---

## 4. What is *not* affected

`_compute_A_W_k` returns `(None, None)` when the system has no
`wannier_r_matrices` attribute (`nonlinear_optical.py:324`, the
`hasattr` guard). That attribute is only set on systems built from Wannier90
input (`wannier_tb` / `wannier_hr` with centres).

**For a model built with `build_system` + `fill_hamiltonian` the correction is
a no-op regardless of the flag**, so the whole issue is invisible to the pure
tight-binding toy models. Verified for the gapped honeycomb used in
`examples/honeycomb_warp` and in `C:\Users\phaney\wrk\dQ\rerun_2026-08\honeycomb`:

```python
s, _ = build_honeycomb()
hasattr(s, 'wannier_r_matrices')            # False
_compute_A_W_k(s, k, ['x', 'y'])            # (None, None)
```

So the chi^(2) / `delta_Q` results in that directory — including the
`int dw (1/w) Re chi_yyy = pi * dQ^yyy` sum rule verified there to 7e-6 — are
untouched by this and by the open bug. The issue bites only Wannier90-derived
systems, i.e. the TMD set.
