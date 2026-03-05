"""Slater-Koster direction-cosine hopping rules.

Provides hopping matrices for:
- TB_simple: 4-orbital (s, px, py, pz) space with Rashba, doubled to 8-dim
- TB_NRL: 9-orbital (s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2) space

Replaces get_sp_hopping.m, get_pp_hopping.m, get_sd_hopping.m,
get_pd_hopping.m, get_dd_hopping.m, and the "2" variants.
"""

import numpy as np
from numpy.typing import NDArray


def ss_hopping(tss: float) -> NDArray:
    """s-s hopping.  Returns (1,1) matrix."""
    return np.array([[tss]], dtype=complex)


def sp_hopping(d: NDArray, tsp: float, tsp_rashba: float = 0.0
               ) -> tuple[NDArray, NDArray]:
    """s-p hopping.  Returns (sp, ps) matrices of shape (1,3) and (3,1).

    Parameters
    ----------
    d : displacement vector from atom1 to atom2, shape (3,)
    tsp : sigma hopping integral
    tsp_rashba : Rashba-type correction

    Returns
    -------
    sp : shape (1, 3) — <s|H|px>, <s|H|py>, <s|H|pz>
    ps : shape (3, 1) — conjugate terms with parity sign flip
    """
    R = np.linalg.norm(d)
    l, m, n = d / R

    sp = np.zeros((1, 3), dtype=complex)
    sp[0, 0] = l * tsp
    sp[0, 1] = m * tsp
    sp[0, 2] = n * tsp + tsp_rashba

    ps = np.zeros((3, 1), dtype=complex)
    ps[0, 0] = -l * tsp
    ps[1, 0] = -m * tsp
    ps[2, 0] = -n * tsp + tsp_rashba

    return sp, ps


def pp_hopping(d: NDArray, tpp_sigma: float, tpp_pi: float = 0.0,
               tpp_rashba: float = 0.0) -> NDArray:
    """p-p hopping.  Returns (3, 3) matrix.

    Parameters
    ----------
    d : displacement vector from atom1 to atom2
    tpp_sigma : sigma hopping integral
    tpp_pi : pi hopping integral
    tpp_rashba : Rashba-type correction
    """
    R = np.linalg.norm(d)
    l, m, n = d / R
    ts, tp = tpp_sigma, tpp_pi
    tr = tpp_rashba

    H = np.zeros((3, 3), dtype=complex)

    H[0, 0] = l**2 * ts + (1 - l**2) * tp
    H[0, 1] = l * m * (ts - tp)
    H[0, 2] = l * n * (ts - tp) + l * tr

    H[1, 0] = m * l * (ts - tp)
    H[1, 1] = m**2 * ts + (1 - m**2) * tp
    H[1, 2] = m * n * (ts - tp) + m * tr

    H[2, 0] = n * l * (ts - tp) - l * tr
    H[2, 1] = n * m * (ts - tp) - m * tr
    H[2, 2] = n**2 * ts + (1 - n**2) * tp

    return H


def build_hopping_4x4(d: NDArray, tss: float, tsp: float, tpp_sigma: float,
                      tpp_pi: float = 0.0, tsp_rashba: float = 0.0,
                      tpp_rashba: float = 0.0) -> NDArray:
    """Build the full (4, 4) hopping matrix in the (s, px, py, pz) basis.

    This combines ss, sp, and pp hopping into one orbital-space matrix.
    """
    H = np.zeros((4, 4), dtype=complex)

    H[0:1, 0:1] = ss_hopping(tss)
    sp_block, ps_block = sp_hopping(d, tsp, tsp_rashba)
    H[0:1, 1:4] = sp_block
    H[1:4, 0:1] = ps_block
    H[1:4, 1:4] = pp_hopping(d, tpp_sigma, tpp_pi, tpp_rashba)

    return H


def spin_double(H_orbital: NDArray) -> NDArray:
    """Expand orbital-only hopping to spin space.

    For non-magnetic hopping: block diagonal with identical spin-up/down blocks.
    H_spin = [[H_orb, 0], [0, H_orb]]

    Input shape: (n, n).  Output shape: (2n, 2n).
    """
    n = H_orbital.shape[0]
    H_spin = np.zeros((2 * n, 2 * n), dtype=complex)
    H_spin[:n, :n] = H_orbital
    H_spin[n:, n:] = H_orbital
    return H_spin


# =============================================================================
# NRL 9-orbital functions: (s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2)
# These operate on sub-blocks and are assembled by build_hopping_9x9.
# =============================================================================

def sd_hopping(d: NDArray, tvsd_s: float, tsd_rashba: float = 0.0
               ) -> tuple[NDArray, NDArray]:
    """s-d hopping.  Returns (sd, ds) blocks of shape (1,5) and (5,1).

    d-orbital order: dxy, dyz, dzx, dx2-y2, dz2
    Standard s-d coupling is even under parity, so sd and ds use the same sign.

    Rashba correction couples s to the z-odd d-orbitals (dyz, dzx) with
    direction cosines perpendicular to each odd orbital's plane:
      s-dyz: weighted by l (perp to yz)
      s-dzx: weighted by m (perp to xz)
    The Rashba part is antisymmetric: sd_R = -ds_R^T, ensuring H[i,j](d)=H[j,i](-d).
    """
    R = np.linalg.norm(d)
    l, m, n = d / R
    s3 = np.sqrt(3.0)

    sd = np.zeros((1, 5), dtype=complex)
    sd[0, 0] = s3 * l * m * tvsd_s             # dxy
    sd[0, 1] = s3 * m * n * tvsd_s             # dyz
    sd[0, 2] = s3 * n * l * tvsd_s             # dzx
    sd[0, 3] = s3 / 2 * (l**2 - m**2) * tvsd_s  # dx2-y2
    sd[0, 4] = (n**2 - 0.5 * (l**2 + m**2)) * tvsd_s  # dz2

    ds = sd.T.copy()  # standard part: same sign (both even parity)

    # Rashba: couples s to z-odd d-orbitals, antisymmetric under orbital swap
    sd[0, 1] += l * tsd_rashba      # s -> dyz
    sd[0, 2] += m * tsd_rashba      # s -> dzx
    ds[1, 0] -= l * tsd_rashba      # dyz -> s (opposite sign)
    ds[2, 0] -= m * tsd_rashba      # dzx -> s (opposite sign)

    return sd, ds


def pd_hopping(d: NDArray, tvpd_s: float, tvpd_p: float,
               tpd_rashba: float = 0.0) -> tuple[NDArray, NDArray]:
    """p-d hopping.  Returns (pd, dp) blocks of shape (3,5) and (5,3).

    p-orbital order: px, py, pz
    d-orbital order: dxy, dyz, dzx, dx2-y2, dz2

    Rashba correction couples all p-orbitals to the z-odd d-orbitals (dyz, dzx)
    with direction cosines perpendicular to each odd orbital's plane:
      p-dyz: weighted by l (perp to yz)
      p-dzx: weighted by m (perp to xz)
    The dp conjugate uses flipped direction cosines, giving automatic antisymmetry.
    """
    R = np.linalg.norm(d)
    l, m, n = d / R
    s3 = np.sqrt(3.0)
    ts, tp = tvpd_s, tvpd_p

    pd = np.zeros((3, 5), dtype=complex)

    # px row
    pd[0, 0] = s3 * l**2 * m * ts + m * (1 - 2*l**2) * tp
    pd[0, 1] = s3 * l * m * n * ts - 2 * l * m * n * tp
    pd[0, 2] = s3 * l**2 * n * ts + n * (1 - 2*l**2) * tp
    pd[0, 3] = s3/2 * l * (l**2 - m**2) * ts + l * (1 - l**2 + m**2) * tp
    pd[0, 4] = l * (n**2 - 0.5*(l**2 + m**2)) * ts - s3 * l * n**2 * tp

    # py row
    pd[1, 0] = s3 * m**2 * l * ts + l * (1 - 2*m**2) * tp
    pd[1, 1] = s3 * m**2 * n * ts + n * (1 - 2*m**2) * tp
    pd[1, 2] = s3 * m * n * l * ts - 2 * m * n * l * tp
    pd[1, 3] = s3/2 * m * (l**2 - m**2) * ts - m * (1 + l**2 - m**2) * tp
    pd[1, 4] = m * (n**2 - 0.5*(l**2 + m**2)) * ts - s3 * m * n**2 * tp

    # pz row
    pd[2, 0] = s3 * n * l * m * ts - 2 * n * l * m * tp
    pd[2, 1] = s3 * n**2 * m * ts + m * (1 - 2*n**2) * tp
    pd[2, 2] = s3 * n**2 * l * ts + l * (1 - 2*n**2) * tp
    pd[2, 3] = s3/2 * n * (l**2 - m**2) * ts - n * (l**2 - m**2) * tp
    pd[2, 4] = n * (n**2 - 0.5*(l**2 + m**2)) * ts + s3 * n * (l**2 + m**2) * tp

    # Conjugate: d-p.  Flip direction cosines (odd parity of p orbitals).
    dp = np.zeros((5, 3), dtype=complex)
    l2, m2, n2 = -l, -m, -n

    # dxy row
    dp[0, 0] = s3 * l2**2 * m2 * ts + m2 * (1 - 2*l2**2) * tp
    dp[0, 1] = s3 * m2**2 * l2 * ts + l2 * (1 - 2*m2**2) * tp
    dp[0, 2] = s3 * n2 * l2 * m2 * ts - 2 * n2 * l2 * m2 * tp

    # dyz row
    dp[1, 0] = s3 * l2 * m2 * n2 * ts - 2 * l2 * m2 * n2 * tp
    dp[1, 1] = s3 * m2**2 * n2 * ts + n2 * (1 - 2*m2**2) * tp
    dp[1, 2] = s3 * n2**2 * m2 * ts + m2 * (1 - 2*n2**2) * tp

    # dzx row
    dp[2, 0] = s3 * l2**2 * n2 * ts + n2 * (1 - 2*l2**2) * tp
    dp[2, 1] = s3 * m2 * n2 * l2 * ts - 2 * m2 * n2 * l2 * tp
    dp[2, 2] = s3 * n2**2 * l2 * ts + l2 * (1 - 2*n2**2) * tp

    # dx2-y2 row
    dp[3, 0] = s3/2 * l2 * (l2**2 - m2**2) * ts + l2 * (1 - l2**2 + m2**2) * tp
    dp[3, 1] = s3/2 * m2 * (l2**2 - m2**2) * ts - m2 * (1 + l2**2 - m2**2) * tp
    dp[3, 2] = s3/2 * n2 * (l2**2 - m2**2) * ts - n2 * (l2**2 - m2**2) * tp

    # dz2 row
    dp[4, 0] = l2 * (n2**2 - 0.5*(l2**2 + m2**2)) * ts - s3 * l2 * n2**2 * tp
    dp[4, 1] = m2 * (n2**2 - 0.5*(l2**2 + m2**2)) * ts - s3 * m2 * n2**2 * tp
    dp[4, 2] = n2 * (n2**2 - 0.5*(l2**2 + m2**2)) * ts + s3 * n2 * (l2**2 + m2**2) * tp

    # Rashba: couples all p to z-odd d (dyz=col1, dzx=col2)
    tr = tpd_rashba
    pd[0, 1] += l * tr       # px -> dyz
    pd[0, 2] += m * tr       # px -> dzx
    pd[1, 1] += l * tr       # py -> dyz
    pd[1, 2] += m * tr       # py -> dzx
    pd[2, 1] += l * tr       # pz -> dyz
    pd[2, 2] += m * tr       # pz -> dzx
    # dp conjugate: same formula with flipped cosines gives antisymmetric sign
    dp[1, 0] += l2 * tr      # dyz -> px
    dp[2, 0] += m2 * tr      # dzx -> px
    dp[1, 1] += l2 * tr      # dyz -> py
    dp[2, 1] += m2 * tr      # dzx -> py
    dp[1, 2] += l2 * tr      # dyz -> pz
    dp[2, 2] += m2 * tr      # dzx -> pz

    return pd, dp


def dd_hopping(d: NDArray, tvdd_s: float, tvdd_p: float, tvdd_d: float,
               tdd_rashba: float = 0.0) -> NDArray:
    """d-d hopping.  Returns (5, 5) matrix.

    d-orbital order: dxy, dyz, dzx, dx2-y2, dz2

    Rashba correction couples z-even d-orbitals {dxy(0), dx2-y2(3), dz2(4)}
    to z-odd d-orbitals {dyz(1), dzx(2)} with direction cosines perpendicular
    to each odd orbital's plane:
      even-dyz: weighted by l (perp to yz)
      even-dzx: weighted by m (perp to xz)
    Antisymmetric under orbital swap: H[even,odd] = -H[odd,even].
    """
    R = np.linalg.norm(d)
    l, m, n = d / R
    s3 = np.sqrt(3.0)
    ts, tp, td = tvdd_s, tvdd_p, tvdd_d

    H = np.zeros((5, 5), dtype=complex)

    # xy,xy
    H[0, 0] = 3*l**2*m**2*ts + (l**2+m**2-4*l**2*m**2)*tp + (n**2+l**2*m**2)*td
    # xy,yz
    H[0, 1] = 3*l*m**2*n*ts + l*n*(1-4*m**2)*tp + l*n*(m**2-1)*td
    # xy,zx
    H[0, 2] = 3*l**2*m*n*ts + m*n*(1-4*l**2)*tp + m*n*(l**2-1)*td
    # xy,x2-y2
    H[0, 3] = 1.5*l*m*(l**2-m**2)*ts + 2*l*m*(m**2-l**2)*tp + 0.5*l*m*(l**2-m**2)*td
    # xy,z2
    H[0, 4] = s3*l*m*(n**2-0.5*(l**2+m**2))*ts - 2*s3*l*m*n**2*tp + 0.5*s3*l*m*(1+n**2)*td

    # yz,xy
    H[1, 0] = H[0, 1]
    # yz,yz
    H[1, 1] = 3*m**2*n**2*ts + (m**2+n**2-4*m**2*n**2)*tp + (l**2+m**2*n**2)*td
    # yz,zx
    H[1, 2] = 3*l*n**2*m*ts + l*m*(1-4*n**2)*tp + l*m*(n**2-1)*td
    # yz,x2-y2
    H[1, 3] = 1.5*m*n*(l**2-m**2)*ts - m*n*(1+2*(l**2-m**2))*tp + m*n*(1+0.5*(l**2-m**2))*td
    # yz,z2
    H[1, 4] = s3*m*n*(n**2-0.5*(l**2+m**2))*ts + s3*m*n*(l**2+m**2-n**2)*tp - 0.5*s3*m*n*(l**2+m**2)*td

    # zx,xy
    H[2, 0] = H[0, 2]
    # zx,yz
    H[2, 1] = H[1, 2]
    # zx,zx
    H[2, 2] = 3*n**2*l**2*ts + (n**2+l**2-4*n**2*l**2)*tp + (m**2+n**2*l**2)*td
    # zx,x2-y2
    H[2, 3] = 1.5*n*l*(l**2-m**2)*ts + n*l*(1-2*(l**2-m**2))*tp - n*l*(1-0.5*(l**2-m**2))*td
    # zx,z2
    H[2, 4] = s3*l*n*(n**2-0.5*(l**2+m**2))*ts + s3*l*n*(l**2+m**2-n**2)*tp - 0.5*s3*l*n*(l**2+m**2)*td

    # x2-y2,xy
    H[3, 0] = H[0, 3]
    # x2-y2,yz
    H[3, 1] = H[1, 3]
    # x2-y2,zx
    H[3, 2] = H[2, 3]
    # x2-y2,x2-y2
    H[3, 3] = 0.75*(l**2-m**2)**2*ts + (l**2+m**2-(l**2-m**2)**2)*tp + (n**2+0.25*(l**2-m**2)**2)*td
    # x2-y2,z2
    H[3, 4] = s3*(0.5*(l**2-m**2)*(n**2-0.5*(l**2+m**2))*ts + n**2*(m**2-l**2)*tp + 0.25*(1+n**2)*(l**2-m**2)*td)

    # z2,xy
    H[4, 0] = H[0, 4]
    # z2,yz
    H[4, 1] = H[1, 4]
    # z2,zx
    H[4, 2] = H[2, 4]
    # z2,x2-y2
    H[4, 3] = H[3, 4]
    # z2,z2
    H[4, 4] = (n**2-0.5*(l**2+m**2))**2*ts + 3*n**2*(l**2+m**2)*tp + 0.75*(l**2+m**2)**2*td

    # Rashba: couples z-even to z-odd d-orbitals, antisymmetric
    tr = tdd_rashba
    # dxy(0) <-> dyz(1), dzx(2)
    H[0, 1] += l * tr;    H[1, 0] -= l * tr
    H[0, 2] += m * tr;    H[2, 0] -= m * tr
    # dx2-y2(3) <-> dyz(1), dzx(2)
    H[3, 1] += l * tr;    H[1, 3] -= l * tr
    H[3, 2] += m * tr;    H[2, 3] -= m * tr
    # dz2(4) <-> dyz(1), dzx(2)
    H[4, 1] += l * tr;    H[1, 4] -= l * tr
    H[4, 2] += m * tr;    H[2, 4] -= m * tr

    return H


def build_hopping_9x9(d: NDArray, tvss_s: float, tvsp_s: float,
                      tvpp_s: float, tvpp_p: float,
                      tvsd_s: float, tvpd_s: float, tvpd_p: float,
                      tvdd_s: float, tvdd_p: float, tvdd_d: float,
                      tsd_rashba: float = 0.0, tpd_rashba: float = 0.0,
                      tdd_rashba: float = 0.0) -> NDArray:
    """Build the full (9, 9) hopping matrix in the NRL orbital basis.

    Orbital order: s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2
    No spin.

    Optional Rashba parameters break z-mirror symmetry by coupling
    z-even and z-odd orbitals:
      tsd_rashba: s <-> {dyz, dzx}
      tpd_rashba: {px, py, pz} <-> {dyz, dzx}
      tdd_rashba: {dxy, dx2-y2, dz2} <-> {dyz, dzx}
    """
    H = np.zeros((9, 9), dtype=complex)

    # s-s (1x1)
    H[0, 0] = tvss_s

    # s-p (1x3) and p-s (3x1) — using the NRL convention (no Rashba)
    R = np.linalg.norm(d)
    l, m, n = d / R

    H[0, 1] = l * tvsp_s
    H[0, 2] = m * tvsp_s
    H[0, 3] = n * tvsp_s
    # conjugate: flip sign (odd parity)
    H[1, 0] = -l * tvsp_s
    H[2, 0] = -m * tvsp_s
    H[3, 0] = -n * tvsp_s

    # p-p (3x3)
    H[1:4, 1:4] = pp_hopping(d, tvpp_s, tvpp_p, 0.0)

    # s-d (1x5) and d-s (5x1)
    sd_block, ds_block = sd_hopping(d, tvsd_s, tsd_rashba)
    H[0:1, 4:9] = sd_block
    H[4:9, 0:1] = ds_block

    # p-d (3x5) and d-p (5x3)
    pd_block, dp_block = pd_hopping(d, tvpd_s, tvpd_p, tpd_rashba)
    H[1:4, 4:9] = pd_block
    H[4:9, 1:4] = dp_block

    # d-d (5x5)
    H[4:9, 4:9] = dd_hopping(d, tvdd_s, tvdd_p, tvdd_d, tdd_rashba)

    return H
