"""Joint density of states (JDOS) with Lorentzian broadening.

Computes

    D(omega) = (1/N_k) Σ_k Σ_{n,m}  [f(E_m) - f(E_n)] * L(E_n - E_m - omega, eta)

where L(x, eta) = (1/pi) * eta / (x**2 + eta**2) is a unit-area
Lorentzian that replaces the energy-conserving δ-function.  The
Fermi-weight prefactor (f_m - f_n) restricts the positive-omega part
to occupied→unoccupied transitions in the T → 0 limit.

This is a single-particle electronic-structure diagnostic; it shares
the k-grid machinery with all_ek.py (DOS) and stands in the same
directory.

Config keys (under cfg['calc']):
    nk          : list of ints (1-3 entries) OR single int (isotropic);
                  k-grid per active lattice direction
    eflist      : single Fermi level [eV]
    kT          : thermal smearing of Fermi function [eV] (default 0.025)
    eta         : Lorentzian broadening [eV] (default 0.05)
    omegalist   : explicit list of omega values [eV], OR
    omega_range : [omin, omax] [eV] with
    nomega      : number of ω points (default 200)
    bands       : 'all' (default) or list of band indices to include

Returns
-------
{
    'omega'      : array(nomega,) [eV]
    'jdos'       : array(nomega,) [states / eV / unit cell]
    'ef'         : Fermi level used
    'eta'        : Lorentzian width used
    'kT'         : thermal smearing used
    'nk'         : grid shape (tuple)
    'ndim'       : dimensionality (1, 2, or 3)
}
"""

import numpy as np

from ..bloch import get_H_k, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel
from .all_ek import detect_dimensionality


def _fermi(ek, ef, kT):
    x = np.clip((ek - ef) / kT, -500, 500)
    return 1.0 / (1.0 + np.exp(x))


def _build_k_grid(system, nk_list, ndim, active):
    """Generate k-point list (mirror of all_ek's pattern)."""
    b_all = list(get_reciprocal_lattice(system.unitcell_vectors))
    b_active = [b_all[i] for i in active]
    db = []
    for dim_i, nk in enumerate(nk_list):
        if nk > 1:
            db.append(b_active[dim_i] / (nk - 1))
        else:
            db.append(np.zeros(3))

    k_list = []
    if ndim == 1:
        for kc1 in range(nk_list[0]):
            k_list.append(-b_active[0] / 2 + db[0] * kc1)
    elif ndim == 2:
        for kc1 in range(nk_list[0]):
            for kc2 in range(nk_list[1]):
                k_list.append(
                    -b_active[0] / 2 - b_active[1] / 2
                    + db[0] * kc1 + db[1] * kc2
                )
    elif ndim == 3:
        for kc1 in range(nk_list[0]):
            for kc2 in range(nk_list[1]):
                for kc3 in range(nk_list[2]):
                    k_list.append(
                        -b_active[0] / 2 - b_active[1] / 2 - b_active[2] / 2
                        + db[0] * kc1 + db[1] * kc2 + db[2] * kc3
                    )
    return k_list


def compute_jdos(system: System, cfg: dict) -> dict:
    """Compute joint density of states.  See module docstring."""
    calc = cfg['calc']

    # --- dimensionality + k-grid ---
    ndim, active = detect_dimensionality(system)
    nk_cfg = calc['nk']
    if isinstance(nk_cfg, int):
        nk_list = [nk_cfg] * ndim
    else:
        nk_list = list(nk_cfg)
        while len(nk_list) < ndim:
            nk_list.append(1)
        nk_list = nk_list[:ndim]

    k_list = _build_k_grid(system, nk_list, ndim, active)
    total_k = len(k_list)

    # --- scalar params ---
    ef = float(np.asarray(calc['eflist']).flat[0])
    kT = float(calc.get('kT', 0.025))
    eta = float(calc.get('eta', 0.05))

    # --- omega grid ---
    if 'omegalist' in calc:
        omega = np.asarray(calc['omegalist'], dtype=float)
    else:
        omin, omax = calc.get('omega_range', [0.0, 5.0])
        nomega = int(calc.get('nomega', 200))
        omega = np.linspace(float(omin), float(omax), nomega)
    nomega = len(omega)

    # --- optional band restriction ---
    bands_cfg = calc.get('bands', 'all')

    parallel.print_root(
        f"  JDOS: {total_k} k-points on {parallel.size} rank(s), "
        f"nomega={nomega}, eta={eta}, ef={ef}"
    )

    my_indices, my_klist = parallel.scatter_work(k_list)
    local_jdos = np.zeros(nomega, dtype=float)
    inv_pi = 1.0 / np.pi

    for i, tk in enumerate(my_klist):
        if parallel.is_root():
            total_local = len(my_klist)
            done = i + 1
            if total_local >= 10 and (10 * done) % total_local == 0:
                print(f"  k-point {done}/{total_local} on rank 0 "
                      f"({100 * done / total_local:.0f}%)")

        H, S = get_H_k(system, tk)
        ek = diagonalize_hk(H, S)  # eigenvectors not needed

        if bands_cfg != 'all':
            ek = ek[list(bands_cfg)]

        f = _fermi(ek, ef, kT)

        # de[n,m] = E_n - E_m
        de = ek[:, None] - ek[None, :]
        # f_mn[n,m] = f[m] - f[n]
        f_mn = f[None, :] - f[:, None]
        # Mask n == m (de=0, f_mn=0 anyway, but cleaner)
        mask = ~np.eye(len(ek), dtype=bool)

        # Vectorize over omega: shape (nomega, N, N)
        # L(de - omega, eta) = (1/pi) * eta / ((de - omega)^2 + eta^2)
        # Memory-efficient: accumulate one omega at a time? At 11 bands,
        # (nomega=200, 11, 11) is tiny (~24k floats). Just do full array.
        arg = de[None, :, :] - omega[:, None, None]
        lor = inv_pi * eta / (arg ** 2 + eta ** 2)
        integrand = (f_mn * mask)[None, :, :] * lor
        local_jdos += integrand.sum(axis=(1, 2))

    local_jdos /= total_k  # per-cell normalization

    jdos = parallel.reduce_sum_complex_array(
        local_jdos.astype(complex)
    ).real

    return {
        'omega': omega,
        'jdos': jdos,
        'ef': ef,
        'eta': eta,
        'kT': kT,
        'nk': tuple(nk_list),
        'ndim': ndim,
    }
