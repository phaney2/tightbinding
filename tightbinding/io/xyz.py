"""XYZ file parser.

Replicates the format used by get_lattice.m:

  Line 1: natoms
  Line 2: header/comment
  Lines 3..natoms+2:  type  x  y  z  U  delta  theta  phi
  Lines natoms+3..natoms+5:  a1 = [...]   a2 = [...]   a3 = [...]

U and delta can be scalar or bracket-enclosed lists like [val1,val2].
"""

import re
import numpy as np
from numpy.typing import NDArray


def read_xyz(path: str) -> tuple[list[str], NDArray, NDArray, NDArray]:
    """Parse an XYZ lattice file.

    Parameters
    ----------
    path : path to the .xyz file

    Returns
    -------
    species : list of atom type strings
    sysdata : (natoms, 7) array — [x, y, z, U, delta, theta, phi]
              U and delta are stored as the first element if given as lists
    lattice_vectors : (3, 3) array, rows are a1, a2, a3
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    natoms = int(lines[0].strip())
    # Line 1 is header/comment, skip it

    species = []
    sysdata = np.zeros((natoms, 7))

    for i in range(natoms):
        line = lines[2 + i].strip()
        parts = line.split()

        species.append(parts[0])

        # Parse the 7 values: x, y, z, U, delta, theta, phi
        col = 0
        for j in range(1, len(parts)):
            if col >= 7:
                break
            val_str = parts[j]

            # Handle bracket-enclosed lists like [0,0] or [.6,.6]
            if val_str.startswith('['):
                # Might span multiple parts if there are spaces
                bracket_str = val_str
                while ']' not in bracket_str and j + 1 < len(parts):
                    j += 1
                    bracket_str += parts[j]
                # Extract first value from bracket list
                inner = bracket_str.strip('[]')
                vals = [float(v) for v in inner.split(',')]
                sysdata[i, col] = vals[0]
            else:
                sysdata[i, col] = float(val_str)
            col += 1

    # Parse lattice vectors
    lattice_vectors = np.zeros((3, 3))
    for k in range(3):
        line = lines[2 + natoms + k].strip()
        # Format: "a1 = [x y z]" or "a1 = [x, y, z]" — may use MATLAB syntax
        match = re.search(r'=\s*\[(.+)\]', line)
        if match:
            vec_str = match.group(1)
            # Replace MATLAB expressions: e.g., 3^.5 -> 3**0.5
            vec_str = vec_str.replace('^', '**')
            # Replace semicolons and commas with spaces
            vec_str = vec_str.replace(',', ' ').replace(';', ' ')
            # Evaluate each component (handles expressions like -3^.5/2)
            components = _parse_vector_components(vec_str)
            lattice_vectors[k] = components

    return species, sysdata, lattice_vectors


def _parse_vector_components(vec_str: str) -> NDArray:
    """Parse a vector string that may contain math expressions.

    Handles things like: "-3**.5/2 3/2 0"
    """
    # Split carefully: we need to handle negative numbers and expressions
    # Use a simple approach: evaluate the whole thing as a Python expression
    # by wrapping in brackets
    try:
        result = eval(f"[{vec_str}]")
        return np.array(result, dtype=float)
    except Exception:
        # Fallback: split on whitespace and evaluate individually
        parts = vec_str.split()
        return np.array([eval(p) for p in parts], dtype=float)
