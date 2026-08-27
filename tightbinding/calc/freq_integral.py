"""Closed-form frequency integrals of the chi^(2) denominator kernels.

Every term of the second-order response factorizes as

    (omega-independent prefactor) x (rational function of omega)

-- all matrix elements, Fermi factors and vertex prefactors are omega-independent
-- so the frequency-integrated response

    J = int_a^b  domega  omega^(-p)  chi^(2)(omega)

can be evaluated exactly, term by term, with the prefactor pulled outside the
integral.  This module supplies the integral of the rational part,

    G(p; {(z_i, s_i)}; a, b)
        = int_a^b domega / ( omega^p prod_i (omega - z_i)^(s_i) )

for arbitrary integer p >= 0 and pole multiplicities s_i >= 1, vectorized over
arrays of poles.  See NOTE_analytic_frequency_integral.md for the derivation
and for the numerical evidence that motivated replacing quadrature with this.

Conventions and restrictions
----------------------------
* **All poles must have Im z_i < 0.**  The chi^(2) denominators are retarded,
  e.g. 1/(w - w_nm + i eta) = 1/(w - z) with z = w_nm - i eta, so this holds
  for eta > 0 and fails only at eta = 0 -- which the caller must reject.
  It is what makes the log branch unambiguous: for real w > 0 both (a - z_i)
  and (b - z_i) have positive imaginary part, so their principal arguments lie
  in (0, pi) and a difference of principal logs is safe.  Never form the ratio
  log((b-z)/(a-z)) instead; that can cross the cut.

* **Poles must be distinct.**  With eta > 0 they always are: the pairs that
  actually occur share a real part but differ by i eta, because denom1 carries
  i eta while denom12 carries 2 i eta.  Genuine repeated poles (denom1^2) are
  passed as a multiplicity, not as two entries.

* **a > 0, and a must stay several eta above zero.**  Two separate reasons:
  individual terms diverge as a -> 0 (the divergences cancel only after the
  terms are summed against their matrix elements), and the near-origin poles
  at -i eta and -2 i eta sit right on top of the omega^(-p) endpoint.  The
  integrand changes character at w ~ eta, so an `a` inside that crossover
  makes the answer a function of the eta/a interplay rather than the physics.
  `FreqIntegralSpec.from_config` enforces this.

* **b = inf is allowed when p + sum_i s_i >= 2**, which makes the integrand
  decay as omega^-2 or faster.  In that case the antiderivative vanishes at
  infinity identically -- the simple-pole residues sum to zero, so the
  log terms cancel -- and J = -F(a) with no large-b cancellation at all.
  When p + sum_i s_i < 2 the b -> inf limit does not exist and J is NaN.

Conditioning
------------
The pole pairs that occur are separated by exactly i eta (at omega2 = 0), so
they are nearly coincident on the scale over which the integrand varies
(~ the interband energy).  The partial-fraction coefficients carry
1/(z_1 - z_2) factors that cancel against each other, costing roughly
log10(|z|/eta) digits per unit of excess multiplicity.  `rational_integral`
can report the cancellation ratio directly (`want_cond=True`); the number of
decimal digits lost is about its base-10 log.
"""

import warnings

import numpy as np


def _series_inv_power(c, s, order):
    """Taylor coefficients of (u + c)^(-s) in u about u = 0.

    Returns a list of ``order + 1`` entries; entry t is the coefficient of
    u^t, namely (-1)^t C(s+t-1, t) c^(-s-t), built by the ratio recursion

        coeff_t / coeff_{t-1} = -(s + t - 1) / (t c)

    which avoids the generalized binomial entirely.  `c` may be an array.
    """
    c = np.asarray(c, dtype=complex)
    if np.any(c == 0):
        raise ValueError(
            "degenerate poles: expansion point coincides with another pole "
            "(this cannot happen for eta > 0; check that eta was not set to 0)"
        )
    coeffs = [c ** (-s)]
    for t in range(1, order + 1):
        coeffs.append(coeffs[-1] * (-(s + t - 1) / (t * c)))
    return coeffs


def _series_mul(A, B, order):
    """Truncated Cauchy product of two coefficient lists, kept to `order`."""
    out = [0.0] * (order + 1)
    for i, ai in enumerate(A[:order + 1]):
        for j, bj in enumerate(B[:order + 1 - i]):
            out[i + j] = out[i + j] + ai * bj
    return out


def partial_fractions(p, poles, mults):
    """Partial-fraction decomposition of 1 / (w^p prod_i (w - z_i)^(s_i)).

        1/(w^p prod_i (w-z_i)^s_i)
            = sum_{j=1..p} A_j / w^j
            + sum_i sum_{l=1..s_i} B_{i,l} / (w - z_i)^l

    Both coefficient families are read off Taylor expansions of the *other*
    factors about the pole in question, which is numerically identical to the
    closed-form binomial expressions but needs no special-casing of negative
    binomial arguments.

    Returns
    -------
    A : list of length p; ``A[j-1]`` is A_j.
    B : list of length len(poles); ``B[i][l-1]`` is B_{i,l}.
    """
    # A_j: expand g0(w) = prod_i (w - z_i)^(-s_i) about w = 0.
    # f = w^-p g0(w) = sum_t c_t w^(t-p), so the coefficient of 1/w^j is c_{p-j}.
    A = []
    if p > 0:
        # Length p: indices 0..p-1 are needed even if no factor multiplies in
        # (the poles list may be empty).
        g0 = [1.0] + [0.0] * (p - 1)
        for z, s in zip(poles, mults):
            g0 = _series_mul(g0, _series_inv_power(-z, s, p - 1), p - 1)
        A = [g0[p - j] for j in range(1, p + 1)]

    # B_{i,l}: expand g_i(u) = (u + z_i)^-p prod_{j!=i} (u + z_i - z_j)^(-s_j)
    # about u = w - z_i = 0.  f = (w-z_i)^(-s_i) g_i(u) = sum_t d_t u^(t-s_i),
    # so the coefficient of 1/(w-z_i)^l is d_{s_i - l}.
    B = []
    for i, (zi, si) in enumerate(zip(poles, mults)):
        gi = [1.0] + [0.0] * (si - 1)   # indices 0..si-1
        if p > 0:
            gi = _series_mul(gi, _series_inv_power(zi, p, si - 1), si - 1)
        for j, (zj, sj) in enumerate(zip(poles, mults)):
            if j == i:
                continue
            gi = _series_mul(gi, _series_inv_power(zi - zj, sj, si - 1), si - 1)
        B.append([gi[si - l] for l in range(1, si + 1)])

    return A, B


def rational_integral(p, poles, mults, a, b=np.inf, want_cond=False):
    """Closed-form ``int_a^b dw / (w^p prod_i (w - z_i)^(s_i))``.

    Parameters
    ----------
    p : int >= 0
        Power of the ``w^(-p)`` weight.
    poles : sequence of complex arrays (broadcastable), all with Im z < 0
        Pole locations.  May be empty, giving ``int_a^b w^(-p) dw``.
    mults : sequence of int >= 1
        Multiplicity of each pole.  Same length as `poles`.
    a : float > 0
        Lower limit.
    b : float, default inf
        Upper limit.
    want_cond : bool
        Also return the cancellation ratio sum|terms| / |J|, whose base-10
        log is roughly the number of significant digits lost.

    Returns
    -------
    J : complex ndarray, broadcast shape of `poles`
        The integral.  NaN where ``b`` is infinite and ``p + sum(mults) < 2``,
        i.e. where the integrand decays no faster than 1/w and the limit does
        not exist.
    A : list of complex ndarrays, length p
        ``A[j-1]`` is the partial-fraction coefficient A_j.  The piece of `J`
        proportional to ``ln a`` is ``-A_1``, and the piece proportional to
        ``a^(1-j)`` is ``A_j/(j-1)`` -- the lower-endpoint divergences that
        must cancel once terms are summed (note section 5.2).
    cond : ndarray, only if `want_cond`
    """
    p = int(p)
    poles = [np.asarray(z, dtype=complex) for z in poles]
    mults = [int(s) for s in mults]
    if len(poles) != len(mults):
        raise ValueError("poles and mults must have the same length")
    if p < 0:
        raise ValueError(f"p must be a non-negative integer, got {p}")
    if a <= 0:
        raise ValueError(f"lower limit must be positive, got a = {a}")

    for z in poles:
        if np.any(z.imag >= 0):
            raise ValueError(
                "all poles must lie in the lower half plane (Im z < 0); "
                "got Im z >= 0, which means eta <= 0 or a non-retarded "
                "denominator reached this routine"
            )

    shape = np.broadcast_shapes(*(z.shape for z in poles)) if poles else ()
    degree = p + sum(mults)
    b_inf = not np.isfinite(b)

    A, B = partial_fractions(p, poles, mults)

    if b_inf and degree < 2:
        # Integrand ~ w^-degree with degree <= 1: no b -> inf limit.  A is
        # still meaningful (it describes the a-endpoint), so return it.
        J = np.full(shape, np.nan, dtype=complex)
        A = [np.broadcast_to(np.asarray(x, dtype=complex), shape) for x in A]
        return (J, A, np.full(shape, np.inf)) if want_cond else (J, A)

    J = np.zeros(shape, dtype=complex)
    mag = np.zeros(shape, dtype=float)

    def _add(term):
        nonlocal J, mag
        J = J + term
        mag = mag + np.abs(term)

    # --- log block ---------------------------------------------------------
    # Evaluated as a difference of principal logs at the two endpoints; for
    # b = inf every b-side log term drops, because A_1 + sum_i B_{i,1} = 0
    # whenever degree >= 2 and the residual log(1 - z/b) -> 0.
    if p > 0:
        _add(-A[0] * np.log(a))
        if not b_inf:
            _add(A[0] * np.log(b))
    for i, zi in enumerate(poles):
        _add(-B[i][0] * np.log(a - zi))
        if not b_inf:
            _add(B[i][0] * np.log(b - zi))

    # --- power block -------------------------------------------------------
    for j in range(2, p + 1):
        _add(-A[j - 1] * a ** (1 - j) / (1 - j))
        if not b_inf:
            _add(A[j - 1] * b ** (1 - j) / (1 - j))
    for i, (zi, si) in enumerate(zip(poles, mults)):
        for l in range(2, si + 1):
            _add(-B[i][l - 1] * (a - zi) ** (1 - l) / (1 - l))
            if not b_inf:
                _add(B[i][l - 1] * (b - zi) ** (1 - l) / (1 - l))

    J = np.broadcast_to(J, shape).copy()
    A = [np.broadcast_to(np.asarray(x, dtype=complex), shape) for x in A]

    if want_cond:
        with np.errstate(divide='ignore', invalid='ignore'):
            cond = np.where(np.abs(J) > 0, mag / np.abs(J), np.inf)
        return J, A, np.broadcast_to(cond, shape)
    return J, A


# ---------------------------------------------------------------------------
# Configuration / channel bookkeeping
# ---------------------------------------------------------------------------

# Guard multiples on omega_min relative to eta.  Below ERROR the near-origin
# poles at -i eta / -2 i eta are straddled by the lower endpoint and the
# result stops meaning anything; between ERROR and WARN it is marginal.
OMEGA_MIN_ETA_ERROR = 5.0
OMEGA_MIN_ETA_WARN = 10.0
OMEGA_MIN_GAP_WARN = 0.1


class FreqIntegralSpec:
    """Request for one or more analytic frequency integrals.

    The engine carries a single "omega axis" of length `nchan`.  In sampled
    mode that axis indexes photon energies; here it indexes output channels,
    so that every phase-3 contraction -- which is linear in the omega-dependent
    kernel -- serves both modes unchanged.

    Channels, per requested power p:
      ('J',   0)  the integral itself
      ('log', 1)  coefficient of ln(omega_min) in J, i.e. -A_1  (0 when p = 0)
      ('pow', j)  coefficient of omega_min^(1-j) in J, i.e. A_j/(j-1),
                  for j = 2 .. pmax  (0 when j > p)
    The 'log'/'pow' channels are the lower-endpoint divergences of note
    section 5.2; summed over terms with their matrix elements they must cancel,
    and that is exactly what the accumulated diagnostic reports.
    """

    def __init__(self, p_list, omega_min, omega_max=np.inf, diagnostics=True):
        self.p_list = [int(p) for p in p_list]
        if not self.p_list:
            raise ValueError("freq_integral: 'p' must name at least one power")
        if any(p < 0 for p in self.p_list):
            raise ValueError(f"freq_integral: p must be >= 0, got {self.p_list}")
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.diagnostics = bool(diagnostics)
        self.pmax = max(self.p_list)

        self.channels = []
        for ip, p in enumerate(self.p_list):
            self.channels.append((ip, 'J', 0))
            if self.diagnostics:
                self.channels.append((ip, 'log', 1))
                for j in range(2, self.pmax + 1):
                    self.channels.append((ip, 'pow', j))
        self.nchan = len(self.channels)
        self.max_cond = 0.0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, block, eta, gap=None):
        """Build from the ``calc.freq_integral`` YAML block.

        `omega_min` defaults to ``OMEGA_MIN_ETA_WARN * eta`` so the safe
        choice is the default.  `gap` (the minimum direct gap, if known) is
        used only for the upper-side warning.
        """
        if not isinstance(block, dict):
            raise ValueError("calc.freq_integral must be a mapping")
        unknown = set(block) - {'p', 'omega_min', 'omega_max', 'diagnostics'}
        if unknown:
            raise ValueError(
                f"calc.freq_integral: unknown key(s) {sorted(unknown)}; "
                "expected p, omega_min, omega_max, diagnostics"
            )
        if eta <= 0:
            raise ValueError(
                "calc.eta must be > 0 for the analytic frequency integral: "
                "at eta = 0 the omega2 denominators become 1/(i*0) and the "
                "vanishing Fermi factors turn that into 0*inf = NaN"
            )

        p_raw = block.get('p', 1)
        p_list = [p_raw] if np.isscalar(p_raw) else list(p_raw)

        omega_max = block.get('omega_max', np.inf)
        if omega_max is None or (isinstance(omega_max, str)
                                 and omega_max.lower() in ('inf', 'infinity')):
            omega_max = np.inf
        omega_max = float(omega_max)

        omega_min = block.get('omega_min')
        if omega_min is None:
            omega_min = OMEGA_MIN_ETA_WARN * eta
        omega_min = float(omega_min)

        if omega_min <= 0:
            raise ValueError(
                f"calc.freq_integral.omega_min must be > 0, got {omega_min}"
            )
        if omega_min < OMEGA_MIN_ETA_ERROR * eta:
            raise ValueError(
                f"calc.freq_integral.omega_min = {omega_min:g} is only "
                f"{omega_min / eta:.2g} x eta (eta = {eta:g}).  The poles at "
                f"-i*eta and -2i*eta sit on top of the omega^(-p) endpoint, so "
                f"the integral stops being a property of the physics below "
                f"~{OMEGA_MIN_ETA_ERROR:g} x eta.  Use omega_min >= "
                f"{OMEGA_MIN_ETA_WARN * eta:g}, or reduce eta."
            )
        if omega_min >= omega_max:
            raise ValueError(
                f"calc.freq_integral: omega_min ({omega_min:g}) must be below "
                f"omega_max ({omega_max:g})"
            )
        if omega_min < OMEGA_MIN_ETA_WARN * eta:
            warnings.warn(
                f"freq_integral: omega_min = {omega_min:g} is "
                f"{omega_min / eta:.2g} x eta; {OMEGA_MIN_ETA_WARN:g} x eta = "
                f"{OMEGA_MIN_ETA_WARN * eta:g} or more is recommended.",
                stacklevel=2,
            )
        if gap is not None and np.isfinite(gap) and gap > 0:
            if omega_min > OMEGA_MIN_GAP_WARN * gap:
                warnings.warn(
                    f"freq_integral: omega_min = {omega_min:g} is not well "
                    f"below the minimum direct gap ({gap:g}); the integral is "
                    f"meant to start at omega << E_gap.  The window is "
                    f"{OMEGA_MIN_ETA_WARN:g}*eta << omega_min << E_gap, which "
                    f"needs eta << {gap / OMEGA_MIN_ETA_WARN:g}.",
                    stacklevel=2,
                )

        return cls(p_list, omega_min, omega_max,
                   diagnostics=bool(block.get('diagnostics', True)))

    # -- kernel construction ----------------------------------------------

    def kernel(self, poles, mults):
        """Integrated kernel array, shape ``broadcast(poles) + (nchan,)``.

        One `rational_integral` call per requested power; the J and endpoint
        channels are filled from the same partial-fraction data.
        """
        shape = (np.broadcast_shapes(*(np.shape(z) for z in poles))
                 if poles else ())
        out = np.zeros(shape + (self.nchan,), dtype=complex)

        per_p = {}
        for ip, p in enumerate(self.p_list):
            per_p[ip] = rational_integral(
                p, poles, mults, self.omega_min, self.omega_max, want_cond=True
            )
            cond = per_p[ip][2]
            finite = cond[np.isfinite(cond)] if np.ndim(cond) else (
                np.array([cond]) if np.isfinite(cond) else np.array([]))
            if finite.size:
                self.max_cond = max(self.max_cond, float(np.max(finite)))

        for ci, (ip, kind, j) in enumerate(self.channels):
            J, A, _ = per_p[ip]
            if kind == 'J':
                out[..., ci] = J
            elif kind == 'log':
                # coefficient of ln(omega_min) in J; absent (0) when p = 0
                out[..., ci] = -A[0] if len(A) >= 1 else 0.0
            else:  # 'pow'
                out[..., ci] = A[j - 1] / (j - 1) if len(A) >= j else 0.0
        return out

    # -- output bookkeeping ------------------------------------------------

    def channel_indices(self, kind, j=None):
        """Column indices of a channel family, in p_list order."""
        return [ci for ci, (ip, k, jj) in enumerate(self.channels)
                if k == kind and (j is None or jj == j)]

    def diagnostic_names(self):
        """(suffix, kind, j) for each endpoint-diagnostic family."""
        if not self.diagnostics:
            return []
        out = [('endpt_log', 'log', 1)]
        for j in range(2, self.pmax + 1):
            out.append((f'endpt_pow{j}', 'pow', j))
        return out

    def describe(self):
        hi = 'inf' if not np.isfinite(self.omega_max) else f'{self.omega_max:g}'
        return (f"p={self.p_list}, omega range [{self.omega_min:g}, {hi}], "
                f"{self.nchan} channel(s)")
