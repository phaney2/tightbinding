"""NRL tight-binding parameter file parser.

Parses the .dat files from the NRL tight-binding database.
Format reference: http://ftp.aip.org/epaps/phys_rev_b/E-PRBMD-54-4519-215kb/

File structure:
  Line 1: lambda (for electron density calculation)
  Lines 2-13: on-site energy parameters (4 each for s, p, d)
  Lines 14-43: hopping parameters (3 each for 10 types: ss_s, sp_s, pp_s, pp_p,
               sd_s, pd_s, pd_p, dd_s, dd_p, dd_d)
  Lines 44-73: overlap parameters (same order as hopping)
"""

import numpy as np


# Hopping/overlap types in file order
_HOP_TYPES = ['ss_s', 'sp_s', 'pp_s', 'pp_p', 'sd_s', 'pd_s', 'pd_p',
              'dd_s', 'dd_p', 'dd_d']


def parse_dat_file(path: str) -> dict:
    """Parse an NRL .dat parameter file.

    Returns
    -------
    dict with keys:
      'lambda_' : float, electron density decay parameter
      'onsite' : dict with 's', 'p', 'd' keys, each a (4,) array [a, b, c, d]
      'hopping' : dict with keys like 'ss_s', each a (3,) array [e, f, g]
      'overlap' : dict with same keys, each a (3,) array [e, f, g]
    """
    values = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            values.append(float(parts[0]))

    idx = 0

    # Line 1: lambda
    lambda_ = values[idx]; idx += 1

    # Lines 2-13: on-site energies (4 params each for s, p, d)
    onsite = {}
    for orb in ['s', 'p', 'd']:
        onsite[orb] = np.array(values[idx:idx+4])
        idx += 4

    # Lines 14-43: hopping parameters (3 params each for 10 types)
    hopping = {}
    for hop_type in _HOP_TYPES:
        hopping[hop_type] = np.array(values[idx:idx+3])
        idx += 3

    # Lines 44-73: overlap parameters (3 params each for 10 types)
    overlap = {}
    for hop_type in _HOP_TYPES:
        overlap[hop_type] = np.array(values[idx:idx+3])
        idx += 3

    return {
        'lambda_': lambda_,
        'onsite': onsite,
        'hopping': hopping,
        'overlap': overlap,
    }


def eval_hopping(params: np.ndarray, R: float, fc: float) -> float:
    """Evaluate distance-dependent hopping/overlap parameter.

    V(R) = (e + f*R) * exp(-g^2 * R) * fc

    Parameters
    ----------
    params : (3,) array [e, f, g]
    R : interatomic distance
    fc : cutoff function value
    """
    e, f, g = params
    return (e + f * R) * np.exp(-g**2 * R) * fc


def cutoff_function(R: float, Rc: float = 14.0, Lc: float = 0.5) -> float:
    """Smooth cutoff function.

    fc = 1 / (1 + exp((R - Rc) / Lc))
    """
    return 1.0 / (1.0 + np.exp((R - Rc) / Lc))
