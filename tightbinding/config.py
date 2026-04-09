"""YAML configuration parsing."""

import numpy as np
import yaml


def load_config(path: str) -> dict:
    """Read YAML config file and return a validated parameter dict.

    The returned dict has top-level keys: 'system', 'hopping', 'onsite',
    'calc', 'output'.  Lattice vectors and eflist are converted to numpy arrays.
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    _validate(cfg)
    _convert_arrays(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    """Check that required sections and keys exist."""
    for section in ('system', 'calc'):
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")

    # hopping section is required unless using Wannier input
    is_wannier = 'wannier_hr' in cfg.get('system', {})
    if not is_wannier and 'hopping' not in cfg:
        raise ValueError("Missing required config section: 'hopping'")

    sys = cfg['system']
    if 'lattice_vectors' not in sys:
        raise ValueError("Missing system.lattice_vectors")

    # Wannier input bypasses lattice_type/positions/basis requirements
    if 'wannier_hr' not in sys:
        # Must have either lattice_type or positions
        if 'lattice_type' not in sys and 'positions' not in sys:
            raise ValueError(
                "system must have either 'lattice_type' or 'positions'"
            )

        # system.basis is required unless per-atom basis is given via positions
        if 'basis' not in sys and 'positions' not in sys:
            raise ValueError("Missing system.basis")

    if 'type' not in cfg['calc']:
        raise ValueError("Missing calc.type")


def _convert_arrays(cfg: dict) -> None:
    """Convert list-of-lists to numpy arrays where appropriate."""
    sys = cfg['system']
    lv = sys['lattice_vectors']
    sys['lattice_vectors'] = np.array(lv, dtype=float)

    # Convert position coords to numpy arrays
    if 'positions' in sys:
        for pos in sys['positions']:
            pos['coord'] = np.array(pos['coord'], dtype=float)

    calc = cfg['calc']
    if 'nk' in calc:
        calc['nk'] = list(calc['nk'])
    if 'ef' in calc:
        calc['ef'] = np.array(calc['ef'], dtype=float)
    if 'eflist' in calc:
        calc['eflist'] = np.array(calc['eflist'], dtype=float)
    if 'omega1list' in calc:
        calc['omega1list'] = np.array(calc['omega1list'], dtype=float)

    # kpath points
    if 'kpath' in calc:
        kp = calc['kpath']
        if 'points' in kp:
            for name, coords in kp['points'].items():
                kp['points'][name] = np.array(coords, dtype=float)
