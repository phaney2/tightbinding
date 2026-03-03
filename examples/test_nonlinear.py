"""Validate nonlinear optical response against MATLAB reference (data2.mat)."""

import sys
import os
import numpy as np
import scipy.io

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tightbinding.main import main

CHI_NAMES = [
    'chi_ii',
    'chi_ee1', 'chi_ee2',
    'chi_ei1', 'chi_ei2',
    'chi_eit1', 'chi_eit2', 'chi_eit3',
    'chi_ie1', 'chi_ie2',
    'chi_e1', 'chi_e2',
    'chi_i1', 'chi_i2',
]

DIRECTIONS = ['xzx', 'zxx']


def load_matlab_ref(mat_path):
    """Load reference data from MATLAB .mat file."""
    d = scipy.io.loadmat(mat_path, squeeze_me=True)
    results = d['data']['results'].item()
    ref = {}
    for name in CHI_NAMES:
        ref[name] = {}
        chi = results[name].item()
        for abc in DIRECTIONS:
            a, b, c = abc
            val = chi[a].item()[b].item()[c].item()
            ref[name][abc] = np.atleast_2d(val)
    return ref


def compare(result, ref, tol=1e-6):
    """Compare Python result to MATLAB reference."""
    max_err = 0.0
    all_pass = True
    for name in CHI_NAMES:
        for abc in DIRECTIONS:
            a, b, c = abc
            py = result[name][a][b][c]
            ml = ref[name][abc]
            # Relative error where |ml| > 0
            mask = np.abs(ml) > 1e-30
            if not np.any(mask):
                continue
            rel_err = np.max(np.abs(py[mask] - ml[mask]) / np.abs(ml[mask]))
            max_err = max(max_err, rel_err)
            if rel_err > tol:
                print(f"  FAIL {name}.{abc}: max rel err = {rel_err:.2e}")
                print(f"    Python: {py[0,:3]}")
                print(f"    MATLAB: {ml[0,:3]}")
                all_pass = False
            else:
                print(f"  PASS {name}.{abc}: max rel err = {rel_err:.2e}")
    return all_pass, max_err


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'input_nonlinear_test.yaml')
    mat_path = os.path.join(os.path.expanduser('~'), 'data2.mat')

    if not os.path.exists(mat_path):
        print(f"Reference file not found: {mat_path}")
        sys.exit(1)

    print("Running nonlinear optical calculation...")
    result = main(config_path)

    print("\nLoading MATLAB reference...")
    ref = load_matlab_ref(mat_path)

    print("\nComparing results:")
    passed, max_err = compare(result, ref)

    print(f"\nOverall max relative error: {max_err:.2e}")
    if passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
