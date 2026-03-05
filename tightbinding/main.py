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
from . import parallel


def _make_serializable(obj):
    """Convert numpy types to JSON-compatible Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _save_npz(output_file, flat, cfg):
    """Save flat dict + embedded config to .npz."""
    if not output_file:
        return
    cfg_serializable = json.loads(json.dumps(cfg, default=_make_serializable))
    flat['_config_json'] = np.array(json.dumps(cfg_serializable))
    np.savez(output_file, **flat)
    print(f"  Results saved to {output_file}.npz")


def _load_npz(path):
    """Load .npz and extract config. Returns (data, cfg)."""
    if not path.endswith('.npz'):
        path = path + '.npz'
    data = np.load(path, allow_pickle=True)
    cfg = json.loads(str(data['_config_json']))
    return data, cfg


def main(config_path: str) -> dict:
    """Run a tight-binding calculation from a YAML config file.

    Returns the result dict from the calculation engine.
    """
    cfg = load_config(config_path)

    # Build system on rank 0, broadcast to all ranks
    if parallel.is_root():
        system = build_system(cfg)
        fill_hamiltonian(system)
    else:
        system = None
    system = parallel.bcast(system)

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

        # Plot if not suppressed (rank 0 only)
        if parallel.is_root() and not cfg['calc'].get('no_plot', False):
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
        from .calc.quantum_metric import compute_quantum_metric
        result = compute_quantum_metric(system, cfg)
        if parallel.is_root():
            _save_quantum_metric(result, cfg)
        return result

    elif calc_type == 'nonlinear_optical':
        from .calc.nonlinear_optical import compute_nonlinear_optical
        result = compute_nonlinear_optical(system, cfg)
        if parallel.is_root():
            _save_nonlinear_optical(result, cfg)
        return result

    elif calc_type == 'all_ek':
        from .calc.all_ek import compute_all_ek, plot_all_ek
        result = compute_all_ek(system, cfg)
        if parallel.is_root():
            _save_all_ek(result, cfg)
            output_file = cfg.get('calc', {}).get('outputfile', 'all_ek')
            plot_all_ek(result, save_path=f"{output_file}.png")
        return result

    else:
        raise ValueError(f"Unknown calc_type: '{calc_type}'")


def _save_nonlinear_optical(result, cfg):
    """Save nonlinear optical results + config to .npz."""
    output_file = cfg.get('calc', {}).get('outputfile')
    flat = {}
    for chi_name, dirs in result.items():
        for a, bd in dirs.items():
            for b, cd in bd.items():
                for c, arr in cd.items():
                    flat[f"{chi_name}.{a}.{b}.{c}"] = arr
    _save_npz(output_file, flat, cfg)


def load_nonlinear_optical(path):
    """Load nonlinear optical results from .npz.

    Returns (result_dict, config_dict).
    result_dict has structure result[chi_name][a][b][c] → array(nef, nomega).
    """
    data, cfg = _load_npz(path)
    result = {}
    for key in data.files:
        if key.startswith('_'):
            continue
        chi_name, a, b, c = key.split('.')
        result.setdefault(chi_name, {})
        result[chi_name].setdefault(a, {})
        result[chi_name][a].setdefault(b, {})
        result[chi_name][a][b][c] = data[key]
    return result, cfg


def _save_quantum_metric(result, cfg):
    """Save quantum metric results + config to .npz."""
    output_file = cfg.get('calc', {}).get('outputfile')
    flat = {}
    for qty_name in ('Q', 'dQ', 'dQf'):
        qty = result[qty_name]
        for d1, v1 in qty.items():
            for d2, v2 in v1.items():
                if isinstance(v2, np.ndarray):
                    flat[f"{qty_name}.{d1}.{d2}"] = v2
                else:
                    for d3, arr in v2.items():
                        flat[f"{qty_name}.{d1}.{d2}.{d3}"] = arr
    _save_npz(output_file, flat, cfg)


def load_quantum_metric(path):
    """Load quantum metric results from .npz.

    Returns (result_dict, config_dict).
    result_dict has keys 'Q', 'dQ', 'dQf'.
    """
    data, cfg = _load_npz(path)
    result = {'Q': {}, 'dQ': {}, 'dQf': {}}
    for key in data.files:
        if key.startswith('_'):
            continue
        parts = key.split('.')
        qty_name = parts[0]
        if len(parts) == 3:
            _, d1, d2 = parts
            result[qty_name].setdefault(d1, {})
            result[qty_name][d1][d2] = data[key]
        elif len(parts) == 4:
            _, d1, d2, d3 = parts
            result[qty_name].setdefault(d1, {})
            result[qty_name][d1].setdefault(d2, {})
            result[qty_name][d1][d2][d3] = data[key]
    return result, cfg


def _save_all_ek(result, cfg):
    """Save all_ek results + config to .npz."""
    output_file = cfg.get('calc', {}).get('outputfile')
    flat = {
        'ekset': result['ekset'],
        'dos_energies': result['dos_energies'],
        'dos_values': result['dos_values'],
        'ndim': np.array(result['ndim']),
    }
    for k, v in result['k_grid'].items():
        flat[f'k_grid.{k}'] = np.asarray(v)
    flat['_band_summary'] = np.array(json.dumps(result['band_summary']))
    if result['band_gap'] is not None:
        flat['band_gap'] = np.array(result['band_gap'])
    _save_npz(output_file, flat, cfg)


def load_all_ek(path):
    """Load all_ek results from .npz. Returns (result_dict, config_dict)."""
    data, cfg = _load_npz(path)
    band_summary = json.loads(str(data['_band_summary']))
    k_grid = {}
    for key in data.files:
        if key.startswith('k_grid.'):
            k_grid[key.split('.', 1)[1]] = data[key]
    result = {
        'ndim': int(data['ndim']),
        'ekset': data['ekset'],
        'k_grid': k_grid,
        'dos_energies': data['dos_energies'],
        'dos_values': data['dos_values'],
        'band_summary': band_summary,
        'band_gap': float(data['band_gap']) if 'band_gap' in data else None,
    }
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
