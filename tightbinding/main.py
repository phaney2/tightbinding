"""Main entry point for the tight-binding code.

Usage:
    python -m tightbinding input.yaml
"""

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
        raise NotImplementedError("Nonlinear optical not yet implemented")

    else:
        raise ValueError(f"Unknown calc_type: '{calc_type}'")


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
