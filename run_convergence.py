"""K-convergence test for nonlinear optical response.

Two cases: ef=2 (in-band) and ef=5 (in-gap).
eta = kT = 0.1. Doubling nk until chi changes < 5%.
"""

import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\haney\tightbinding')

import numpy as np
from tightbinding.config import load_config
from tightbinding.lattice import build_system
from tightbinding.hamiltonian import fill_hamiltonian
from tightbinding.calc.nonlinear_optical import compute_nonlinear_optical, CHI_NAMES
from tightbinding.main import _save_nonlinear_optical

def get_nonzero_chi(result, directions):
    """Extract all nonzero chi values as a flat dict name.abc -> complex array."""
    vals = {}
    for name in CHI_NAMES:
        for abc in directions:
            a, b, c = abc
            arr = result[name][a][b][c].flatten()
            if np.max(np.abs(arr)) > 1e-15:
                vals[f'{name}.{abc}'] = arr
    return vals

def max_relative_change(prev, curr):
    """Max relative change across all nonzero components."""
    max_change = 0.0
    for key in curr:
        if key not in prev:
            continue
        denom = max(np.max(np.abs(prev[key])), np.max(np.abs(curr[key])))
        if denom > 1e-15:
            change = np.max(np.abs(curr[key] - prev[key])) / denom
            max_change = max(max_change, change)
    return max_change

def run_convergence(ef_val, label):
    print(f"\n{'='*70}")
    print(f"CONVERGENCE TEST: ef={ef_val} ({label})")
    print(f"eta=kT=0.1, directions=[xzx, zxx]")
    print(f"{'='*70}")

    nk_values = [6, 12, 24, 48, 96, 192]
    directions = ['xzx', 'zxx']

    prev_vals = None
    history = []

    for nk in nk_values:
        cfg = load_config(r'C:\Users\haney\tightbinding\examples\input_nonlinear_test.yaml')
        cfg['calc']['nk'] = [nk, nk]
        cfg['calc']['eflist'] = [ef_val]
        cfg['calc']['eta'] = 0.1
        cfg['calc']['kT'] = 0.1
        cfg['calc']['omega1list'] = list(np.linspace(1.5, 2.5, 10))
        cfg['calc']['outputfile'] = f'results/convergence_{label}_nk{nk}'

        system = build_system(cfg)
        fill_hamiltonian(system)

        t0 = time.time()
        result = compute_nonlinear_optical(system, cfg)
        elapsed = time.time() - t0
        _save_nonlinear_optical(result, cfg)

        curr_vals = get_nonzero_chi(result, directions)
        n_nonzero = len(curr_vals)

        if prev_vals is not None:
            change = max_relative_change(prev_vals, curr_vals)
            converged = change < 0.05
            history.append((nk, elapsed, n_nonzero, change, converged))
            print(f"\n  nk={nk:4d}: {elapsed:6.1f}s, {n_nonzero} nonzero components, "
                  f"max_change={change:.4f} ({change*100:.1f}%) {'** CONVERGED **' if converged else ''}")

            # Print a few representative values
            for key in sorted(curr_vals.keys())[:4]:
                if key in prev_vals:
                    print(f"    {key}: {curr_vals[key][0]:.6e}  (prev: {prev_vals[key][0]:.6e})")

            if converged:
                print(f"\n  Converged at nk={nk} for ef={ef_val} ({label})")
                prev_vals = curr_vals
                break
        else:
            history.append((nk, elapsed, n_nonzero, None, False))
            print(f"\n  nk={nk:4d}: {elapsed:6.1f}s, {n_nonzero} nonzero components (baseline)")
            for key in sorted(curr_vals.keys())[:4]:
                print(f"    {key}: {curr_vals[key][0]:.6e}")

        prev_vals = curr_vals

    print(f"\n  Summary for ef={ef_val} ({label}):")
    print(f"  {'nk':>6} {'time':>8} {'#nonzero':>8} {'max_change':>12} {'converged':>10}")
    print(f"  {'-'*50}")
    for nk, t, nn, ch, conv in history:
        ch_str = f"{ch:.4f}" if ch is not None else "---"
        print(f"  {nk:>6} {t:>7.1f}s {nn:>8} {ch_str:>12} {'YES' if conv else '':>10}")

    return history


if __name__ == '__main__':
    h1 = run_convergence(2.0, 'in_band')
    h2 = run_convergence(5.0, 'in_gap')

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
