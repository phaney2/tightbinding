# Case study: what the AI actually did

**Setup for the audience.** I returned to a tight-binding code I hadn't opened in
5–6 months and a set of analytical notes I hadn't read in about as long. The task:
*"Realize this analytical model (a trigonally-warped gapped Dirac valley) as a lattice
model in my code, and check whether the code's numerics reproduce the note's closed-form
predictions."* One session, start to finished PDF.

Every number below is reproducible from scripts committed in `examples/honeycomb_warp/`.

---

## 1. It did physics, not just programming

- **Read a 4-page derivation-heavy PDF** and extracted the one equation that mattered
  (Eq. 1), the conventions attached to it (ℏ=e=1, H′=+E·r, r = −iv/ω), the symmetry
  argument, and the two numerical predictions worth testing.
- **Did the analytical mapping by hand** — expanded the nearest-neighbour honeycomb
  hopping sum about K using complex-variable algebra, showed
  S(K+q) = −(3a/2)q̄ + (3a²/8)q², and matched it term-by-term to the note's
  Hamiltonian: **v = 3Ta/2, λ = −3Ta²/8, m = (u_sᴬ − u_sᴮ)/2**.
  This is textbook-adjacent physics, but the work is in specializing it to *this note's*
  exact sign and axis conventions — which is precisely where errors live.
- **Confirmed the model is exact, not approximate** — the honeycomb diagonal is exactly
  ±m with no k-dependence and no σ₀ term, so it matches Eq. 1 term for term rather than
  "approximately near the gap."

## 2. It found a constraint I hadn't asked about, and worked around it

- **Noticed λ/v = −a/4 is locked** by the lattice geometry. You cannot tune the
  trigonal warping independently of the Fermi velocity in a NN honeycomb model.
  I did not ask this question; it changes how the study has to be designed.
- **Evaluated the obvious fix and rejected it with a reason.** Third-neighbour hopping
  would relax the constraint, but a distance cutoff can't reach 3NN (at 2a) without also
  admitting NNN (at √3a ≈ 1.73a), and NNN introduces σ₀ and k²σ_z terms that are absent
  from the target model. So: don't do that.
- **Reformulated the experiment instead.** Identified that the physically meaningful
  quantity is the *dimensionless* warping parameter λm/v² = m/(6T), so the note's
  perturbative regime is reached by shrinking the gap relative to the hopping. That
  turned a dead end into the actual control variable.
- **Derived a sharper test than the one in the note.** Because λ/v is locked, the note's
  Eq. 15 collapses to **T^yyy = πa/(4m), independent of hopping strength** — a
  parameter-free prediction. Better test than the original.

## 3. It caught a setup subtlety that would have produced a plausible wrong answer

- **Orientation of the lattice is not free.** The note's selection rule rests on the
  antiunitary symmetry 𝒯M_x. Since 𝒯 swaps valleys, M_x must too — which forces Γ–K to
  lie along x̂, i.e. the A–B bond must point along **ŷ**, not x̂.
- The naive "bond along x̂" setup is rotated 90° and **swaps which response channel
  survives**. It would have returned a nonzero x-odd chain and a vanishing yyy — which
  looks like a physics result, not a bug. This is the failure mode that eats weeks.

## 4. It set numerical parameters from physics, not from defaults

- **Caught a poisoned default.** The engine's `eta_sos` regularizer defaults to 0.05,
  which is *comparable to the 0.1 gap* in this low-energy model — it would have
  corrupted the result silently. Set it to 1e-8 after reasoning that the two bands are
  never degenerate here.
- Set `eta = 0` to match the note's unbroadened identity, and `kT = m/50` for a clean
  T→0 occupation.
- **Estimated the required k-grid from the physics** (integrand peaked on the scale
  m/v, BZ of linear size 4π/3a) rather than guessing, and estimated the wall-clock cost
  before launching.

## 5. It designed a layered verification strategy

Rather than run one end-to-end number and compare:

| Layer | What it isolates |
|---|---|
| Lattice H(k) vs. the k·p model | Is the model mapping right? |
| Pointwise δQ vs. note Eq. 7 | Is the engine right, with zero integration error? |
| Note Eq. 13 vs. Eq. 7 | Is the note's own reduction right? |
| Note Eq. 14 vs. Eq. 13 | Is the leading-order-in-λ form right? |
| Full BZ vs. Eq. 15 | The actual target |

**Why this matters:** the run produced *two* discrepancies at once — a sign flip and a
0.6% magnitude offset. Had only the last layer been run, they'd have been entangled and
nearly undiagnosable. The layering separated them cleanly.

- **Checked error *scaling*, not just error size.** The k·p fit error goes as q² in
  relative terms — which is the *signature* of a correct O(q²) mapping with a cubic
  correction. A wrong λ would give the same error magnitude at one q but a different
  scaling. Checking one point would not have distinguished them.
- **Used symmetry as a prediction-independent check.** Four channels must vanish by the
  𝒯M_x rule; they came out at ~1e-16. The C₃ chain yyy = −yxx = −xxy = −xyx held exactly.
  Neither of these depends on any analytic value being right.
- **Convergence in two variables** — k-grid (numerical) and mass (physical) — to
  *distinguish* discretization error from genuine lattice corrections beyond the
  continuum model. Residual shrank as m/T → 0, identifying it as physical.

## 6. It ran a discrepancy to ground instead of hand-waving it

The first result was 0.3% right in magnitude and **sign-flipped**.

- Derived term-by-term that the code's implemented formula equals −1 × the note's Eq. 7.
- Then **didn't trust its own algebra** — verified empirically, pointwise, and found the
  ratio to be exactly −1.000000 at every k-point and in every channel tested.
- **Diagnosed the root cause:** the code's formula sheet uses H′ = −E·r; the note states
  H′ = +E·r.
- **Correctly classified it as a convention mismatch, not a bug.** A less careful
  analysis reports "code disagrees with theory," which is both wrong and expensive.

## 7. It fixed the tooling and the documentation as it went

- `CLAUDE.md` (the project's own instruction file) said to run MPI with
  `mpiexec --use-hwthread-cpus`. That **failed** — this machine has Microsoft MPI, not
  OpenMPI. It identified the actual MPI implementation and corrected the instruction,
  plus two other environment gotchas it hit (PYTHONPATH for standalone scripts, stdout
  buffering under MS-MPI).
- **I corrected its workflow mid-session** ("read the docs before diving into source"),
  told it to write that rule down, and it did — so the correction is *durable*, not
  conversational. Next session starts with it.
- **Wrote its findings back into the project docs**: the parameter mapping, the
  orientation requirement, the sign convention, the `eta_sos` trap, and the benchmark
  numbers. The next person — or the next session — starts from these rather than
  rediscovering them.

## 8. Deliverables

Four runnable artifacts, a LaTeX source, and a compiled 5-page PDF write-up — produced
in the same session as the work, not afterwards.

---

## What it did *not* do (worth saying out loud)

Credibility depends on this part.

- **I supplied the question, the notes, and the codebase.** It had no idea this was
  worth doing.
- **I caught its workflow inefficiency.** Its first instinct was to grep source code to
  answer a question the documentation already answered. It needed telling.
- **It cannot tell me the note itself is correct.** It verified *consistency* between
  note and code, which is a different and weaker claim than *truth*.
- **The residual is an inference.** It attributed the leftover 0.3–0.8% to
  O(λ³) + lattice corrections based on how the residual scales with m. That reasoning is
  sound but it is my job, not its job, to accept it.
- **It chose its own tests.** A verification suite designed by the same agent doing the
  work is not adversarial review. The symmetry checks are the strongest evidence here
  precisely because they don't depend on any predicted value.

---

## Boiling down — suggested single slide

> **From dormant code + stale notes to a verified benchmark, in one session**
>
> - Derived the analytical mapping by hand (honeycomb NN → warped Dirac valley)
> - Found a constraint I hadn't asked about (λ/v locked by geometry) — and redesigned
>   the study around it
> - Caught an orientation subtlety that would have silently produced the *wrong channel*
> - Overrode a default regularizer that was larger than the band gap
> - Built a 5-layer verification chain; symmetry checks passed to 1e-16
> - Found a sign flip, traced it to H′ = −E·r vs +E·r — **convention, not bug**
> - Fixed the project's own (wrong) MPI instructions, and wrote all of it back into the
>   docs for next time
>
> *Result: matches the analytic prediction to 0.04%. Human still supplies the question,
> the judgement, and the skepticism.*

**Suggested closing line:** the interesting capability isn't that it wrote the code —
it's that it *knew what to check, and didn't believe its own first answer*.
