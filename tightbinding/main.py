"""Main entry point for the tight-binding code.

Usage:
    python -m tightbinding input.yaml
"""

import json
import sys

import numpy as np

from .config import load_config
from .types import KPath
from .lattice import build_system
from .hamiltonian import fill_hamiltonian


def main(config_path: str) -> dict:
    """Run a tight-binding calculation from a YAML config file.

    Returns the result dict from the calculation engine.
    """
    cfg = load_config(config_path)

    # Build system
    system = build_system(cfg)

    # Fill Hamiltonian
    fill_hamiltonian(system)

    # Dispatch to calculation engine
    calc_type = cfg['calc']['type']
    result = _dispatch(system, cfg, calc_type)

    return result


def _dispatch(system, cfg, calc_type):
    """Dispatch to the appropriate calculation engine."""

    if calc_type == 'band_structure':
        from .calc.bands import compute_band_structure, plot_bands

        kpath = _build_kpath(cfg)
        result = compute_band_structure(system, kpath)

        # Plot if not suppressed
        if not cfg['calc'].get('no_plot', False):
            ax = plot_bands(result)
            import matplotlib.pyplot as plt
            title = cfg.get('output', {}).get('file', 'bands')
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(f"{title}_bands.png", dpi=150)
            print(f"Band structure saved to {title}_bands.png")

        return result

    elif calc_type == 'kubo':
        raise NotImplementedError("Kubo linear response not yet implemented")

    elif calc_type == 'quantum_metric':
        raise NotImplementedError("Quantum metric not yet implemented")

    elif calc_type == 'nonlinear_optical':
        from .calc.nonlinear_optical import compute_nonlinear_optical
        result = compute_nonlinear_optical(system, cfg)
        _save_nonlinear_optical(result, cfg)
        return result

    else:
        raise ValueError(f"Unknown calc_type: '{calc_type}'")


def _save_nonlinear_optical(result, cfg):
    """Save nonlinear optical results + config to .npz."""
    output_file = cfg.get('calc', {}).get('outputfile')
    if not output_file:
        return

    flat = {}
    for chi_name, dirs in result.items():
        for a, bd in dirs.items():
            for b, cd in bd.items():
                for c, arr in cd.items():
                    flat[f"{chi_name}.{a}.{b}.{c}"] = arr

    # Embed the full config as JSON string
    def _make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    cfg_serializable = json.loads(json.dumps(cfg, default=_make_serializable))
    flat['_config_json'] = np.array(json.dumps(cfg_serializable))

    np.savez(output_file, **flat)
    print(f"Results saved to {output_file}.npz")


def load_nonlinear_optical(path):
    """Load nonlinear optical results from .npz.

    Returns (result_dict, config_dict).
    result_dict has structure result[chi_name][a][b][c] → array(nef, nomega).
    """
    if not path.endswith('.npz'):
        path = path + '.npz'
    data = np.load(path, allow_pickle=True)

    cfg = json.loads(str(data['_config_json']))

    result = {}
    for key in data.files:
        if key.startswith('_'):
            continue
        parts = key.split('.')
        chi_name, a, b, c = parts
        result.setdefault(chi_name, {})
        result[chi_name].setdefault(a, {})
        result[chi_name][a].setdefault(b, {})
        result[chi_name][a][b][c] = data[key]

    return result, cfg


def _build_kpath(cfg) -> KPath:
    """Build a KPath from the calc section of the config."""
    calc = cfg['calc']

    if 'kpath' not in calc:
        raise ValueError("calc.kpath is required for band_structure calculations")

    kp = calc['kpath']
    path_labels = kp['path']
    points_dict = kp['points']
    npoints = kp.get('npoints', 100)

    points = [np.asarray(points_dict[label], dtype=float) for label in path_labels]

    return KPath(
        points=points,
        labels=path_labels,
        npoints_per_segment=npoints,
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m tightbinding <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
