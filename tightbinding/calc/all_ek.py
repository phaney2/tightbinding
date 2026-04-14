"""Full BZ eigenvalue computation + density of states.

Computes all energy eigenvalues on a uniform k-grid spanning the Brillouin zone.
Auto-detects system dimensionality (1D/2D/3D) and produces:
  - eigenvalue array over the grid
  - density of states histogram
  - band summary (edges, gap, van Hove singularities)
"""

import numpy as np

from ..bloch import get_H_k, get_H_v, get_reciprocal_lattice, diagonalize_hk
from ..types import System
from .. import parallel


def detect_dimensionality(system):
    """Detect system dimensionality from hopping displacements.

    A lattice direction is active if any hopping matrix has a nonzero
    fractional displacement along it. This correctly handles 2D systems
    with a large artificial 3rd lattice vector.

    Returns (ndim, active_indices) where active_indices lists which
    of the 3 lattice vectors are periodic.
    """
    a1, a2, a3 = system.unitcell_vectors
    A = np.array([a1, a2, a3]).T  # columns are lattice vectors
    Ainv = np.linalg.pinv(A)

    has_hopping = [False, False, False]
    for m in system.matrices:
        frac = Ainv @ m.displacement
        for i in range(3):
            if abs(frac[i]) > 0.1:
                has_hopping[i] = True

    active = [i for i in range(3) if has_hopping[i]]
    # Fallback: if no off-site hoppings, use lattice vector norms
    if not active:
        norms = [np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)]
        active = [i for i, n in enumerate(norms) if n > 1e-10]

    return len(active), active


def compute_all_ek(system: System, cfg: dict) -> dict:
    """Compute all eigenvalues over the full BZ.

    Parameters
    ----------
    system : System with filled Hamiltonian
    cfg : config dict; reads from cfg['calc']

    Returns
    -------
    dict with 'ndim', 'ekset', 'k_grid', 'dos_energies', 'dos_values',
    'band_summary', 'band_gap'
    """
    calc = cfg['calc']
    nk_cfg = list(calc['nk'])

    ndim, active = detect_dimensionality(system)
    print(f"  Detected {ndim}D system (active lattice vectors: {active})")

    # Pad nk to match active dimensions
    while len(nk_cfg) < ndim:
        nk_cfg.append(1)
    nk_list = nk_cfg[:ndim]  # trim to ndim

    # Reciprocal lattice vectors
    b_all = list(get_reciprocal_lattice(system.unitcell_vectors))
    b_active = [b_all[i] for i in active]

    # Build k-grid
    db = []
    for dim_i, nk in enumerate(nk_list):
        if nk > 1:
            db.append(b_active[dim_i] / (nk - 1))
        else:
            db.append(np.zeros(3))

    # Generate all k-points
    ranges = [range(nk) for nk in nk_list]
    k_list = []
    grid_indices = []

    if ndim == 1:
        for kc1 in ranges[0]:
            tk = -b_active[0] / 2 + db[0] * kc1
            k_list.append(tk)
            grid_indices.append((kc1,))
    elif ndim == 2:
        for kc1 in ranges[0]:
            for kc2 in ranges[1]:
                tk = -b_active[0] / 2 - b_active[1] / 2 + db[0] * kc1 + db[1] * kc2
                k_list.append(tk)
                grid_indices.append((kc1, kc2))
    elif ndim == 3:
        for kc1 in ranges[0]:
            for kc2 in ranges[1]:
                for kc3 in ranges[2]:
                    tk = (-b_active[0] / 2 - b_active[1] / 2 - b_active[2] / 2
                          + db[0] * kc1 + db[1] * kc2 + db[2] * kc3)
                    k_list.append(tk)
                    grid_indices.append((kc1, kc2, kc3))

    total_jobs = len(k_list)
    nbands = system.norbs

    parallel.print_root(
        f"  All eigenvalues: {total_jobs} k-points, {nbands} bands, "
        f"{parallel.size} rank(s)"
    )

    do_berry = bool(calc.get('compute_berry_curvature', False))
    ef_berry = None
    if do_berry:
        eflist = calc.get('eflist', None)
        if eflist is not None:
            ef_berry = float(np.asarray(eflist).flat[0])
        else:
            raise ValueError("compute_berry_curvature requires eflist")
        kT_berry = float(calc.get('kT', 0.05))
        parallel.print_root(f"  Berry curvature: ef={ef_berry}, kT={kT_berry}")

    # Scatter k-points across MPI ranks
    my_indices, my_klist = parallel.scatter_work(k_list)

    # Each rank diagonalizes its subset
    if not do_berry:
        local_ek = np.array([diagonalize_hk(*get_H_k(system, tk)) for tk in my_klist])
        local_omega = None
    else:
        local_ek = np.zeros((len(my_klist), nbands))
        local_omega = np.zeros(len(my_klist))
        for i_k, tk in enumerate(my_klist):
            H, S, v = get_H_v(system, tk, order=1)
            ek, psi = diagonalize_hk(H, S, eigenvectors=True)
            local_ek[i_k] = ek
            # Velocity matrix elements in eigenbasis
            vx = psi.conj().T @ v['x'] @ psi
            vy = psi.conj().T @ v['y'] @ psi
            # Occupation: Fermi-Dirac
            fn = 1.0 / (np.exp((ek - ef_berry) / kT_berry) + 1.0)
            # Berry curvature: Omega_z = -2 sum_{n,m} f_nm Im[vx_nm vy_mn] / (em-en)^2
            de = ek[None, :] - ek[:, None]  # de[n,m] = em - en
            mask = np.abs(de) > 1e-10
            np.fill_diagonal(mask, False)
            f_nm = fn[:, None] - fn[None, :]  # f_nm[n,m] = fn - fm
            inv_de2 = np.zeros_like(de)
            inv_de2[mask] = 1.0 / (de[mask] ** 2)
            # Im[vx_nm * vy_mn] = Im[vx[n,m] * vy[m,n]]
            cross = (vx * vy.T).imag  # cross[n,m] = Im(vx[n,m]*vy[m,n])
            local_omega[i_k] = -2.0 * np.sum(f_nm * cross * inv_de2)

    # Gather all eigenvalues
    ek_flat = parallel.gather_array(my_indices, local_ek, total_jobs)

    # Gather Berry curvature if computed
    berry_flat = None
    if do_berry and local_omega is not None:
        berry_flat = parallel.gather_array(my_indices, local_omega, total_jobs)

    if ndim == 1:
        ekset = ek_flat.reshape(nk_list[0], nbands)
        k_grid = {'k1': np.array([k_list[i][0] for i in range(nk_list[0])])}
        # Use the component along the active b-vector
        b_hat = b_active[0] / np.linalg.norm(b_active[0])
        k_grid['k1'] = np.array([np.dot(k_list[i], b_hat) for i in range(nk_list[0])])
    elif ndim == 2:
        ekset = ek_flat.reshape(nk_list[0], nk_list[1], nbands)
        kx = np.array([k_list[i * nk_list[1]][0] for i in range(nk_list[0])])
        kz = np.array([k_list[j][2] for j in range(nk_list[1])])
        k_grid = {'kx': kx, 'kz': kz}
    elif ndim == 3:
        ekset = ek_flat.reshape(nk_list[0], nk_list[1], nk_list[2], nbands)
        k_grid = {}

    # DOS histogram
    dos_bins = int(calc.get('dos_bins', 200))
    all_energies = ek_flat.flatten()
    e_min, e_max = all_energies.min(), all_energies.max()
    margin = 0.05 * (e_max - e_min) if e_max > e_min else 0.5
    counts, bin_edges = np.histogram(all_energies, bins=dos_bins,
                                     range=(e_min - margin, e_max + margin))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    dos_values = counts / (total_jobs * bin_width)  # states per energy per cell

    # Band summary
    band_summary = []
    for ib in range(nbands):
        band_e = ek_flat[:, ib]
        band_summary.append({
            'band': ib,
            'min': float(band_e.min()),
            'max': float(band_e.max()),
            'bandwidth': float(band_e.max() - band_e.min()),
        })

    # Band gap detection
    band_gap = None
    eflist = calc.get('eflist', None)
    if eflist is not None:
        ef = float(np.asarray(eflist).flat[0])
        # Find bands below and above ef
        below = [bs for bs in band_summary if bs['max'] < ef]
        above = [bs for bs in band_summary if bs['min'] > ef]
        if below and above:
            vbm = max(bs['max'] for bs in below)
            cbm = min(bs['min'] for bs in above)
            band_gap = cbm - vbm if cbm > vbm else 0.0

    # Reshape Berry curvature to k-grid
    berry_grid = None
    if berry_flat is not None:
        if ndim == 1:
            berry_grid = berry_flat.reshape(nk_list[0])
        elif ndim == 2:
            berry_grid = berry_flat.reshape(nk_list[0], nk_list[1])
        elif ndim == 3:
            berry_grid = berry_flat.reshape(nk_list[0], nk_list[1], nk_list[2])

    result = {
        'ndim': ndim,
        'ekset': ekset,
        'k_grid': k_grid,
        'dos_energies': bin_centers,
        'dos_values': dos_values,
        'band_summary': band_summary,
        'band_gap': band_gap,
    }
    if berry_grid is not None:
        result['berry_curvature'] = berry_grid
        result['k_list'] = np.array(k_list)

    if parallel.is_root():
        print_summary(result)

    return result


def print_summary(result):
    """Print diagnostic summary to stdout."""
    ndim = result['ndim']
    bs = result['band_summary']
    nbands = len(bs)

    print(f"\n  === Band Structure Summary ({ndim}D system, {nbands} bands) ===")
    for b in bs:
        print(f"    Band {b['band']}: [{b['min']:.4f}, {b['max']:.4f}]  "
              f"(bandwidth = {b['bandwidth']:.4f})")

    if result['band_gap'] is not None:
        print(f"    Band gap: {result['band_gap']:.4f}")
    else:
        print(f"    Band gap: N/A (no ef provided or metallic)")

    # Van Hove singularities: peaks in DOS
    dos = result['dos_values']
    energies = result['dos_energies']
    if len(dos) > 2:
        # Find local maxima
        peaks = []
        for i in range(1, len(dos) - 1):
            if dos[i] > dos[i - 1] and dos[i] > dos[i + 1] and dos[i] > 0.1 * dos.max():
                peaks.append((energies[i], dos[i]))
        if peaks:
            print(f"    DOS peaks (van Hove): ", end="")
            print(", ".join(f"E={e:.4f}" for e, _ in peaks[:8]))
    print()


def plot_all_ek(result, save_path=None):
    """Plot eigenvalue surfaces and/or DOS.

    1D: E(k) + DOS side by side
    2D: 3D surface plot + DOS side by side
    3D: DOS only
    """
    import matplotlib.pyplot as plt

    ndim = result['ndim']
    ekset = result['ekset']
    nbands = ekset.shape[-1]

    if ndim == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                        gridspec_kw={'width_ratios': [2, 1]})
        k1 = result['k_grid']['k1']
        for ib in range(nbands):
            ax1.plot(k1, ekset[:, ib], 'k-', linewidth=0.8)
        ax1.set_xlabel('k')
        ax1.set_ylabel('Energy')
        ax1.set_title('Band structure')
        ax1.grid(True, alpha=0.3)

        # DOS — energy on x, DOS on y
        de = result['dos_energies'][1] - result['dos_energies'][0]
        ax2.bar(result['dos_energies'], result['dos_values'],
                width=de, color='steelblue', edgecolor='none')
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('DOS')
        ax2.set_title('Density of States')
        ax2.set_xlim(ax1.get_ylim())

    elif ndim == 2:
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')

        nk1, nk2 = ekset.shape[0], ekset.shape[1]
        K1 = np.arange(nk1)
        K2 = np.arange(nk2)
        K1g, K2g = np.meshgrid(K1, K2, indexing='ij')

        for ib in range(nbands):
            ax1.plot_surface(K1g, K2g, ekset[:, :, ib],
                             alpha=0.7, cmap='viridis', edgecolor='none')
        ax1.set_xlabel('k1 index')
        ax1.set_ylabel('k2 index')
        ax1.set_zlabel('Energy')
        ax1.set_title('Energy surfaces')
        # Set view: k-axes horizontal, energy vertical
        ax1.view_init(elev=0, azim=-90)

        # DOS — energy on x, DOS on y
        ax2 = fig.add_subplot(122)
        de = result['dos_energies'][1] - result['dos_energies'][0]
        ax2.bar(result['dos_energies'], result['dos_values'],
                width=de, color='steelblue', edgecolor='none')
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('DOS')
        ax2.set_title('Density of States')

    elif ndim == 3:
        fig, ax = plt.subplots(figsize=(6, 5))
        de = result['dos_energies'][1] - result['dos_energies'][0]
        ax.bar(result['dos_energies'], result['dos_values'],
               width=de, color='steelblue', edgecolor='none')
        ax.set_xlabel('Energy')
        ax.set_ylabel('DOS')
        ax.set_title('Density of States (3D)')

    # Annotate band gap if available
    if result['band_gap'] is not None and result['band_gap'] > 0:
        bs = result['band_summary']
        below = [b for b in bs if b['max'] < (bs[0]['min'] + bs[-1]['max']) / 2]
        above = [b for b in bs if b['min'] > (bs[0]['min'] + bs[-1]['max']) / 2]
        if below and above:
            vbm = max(b['max'] for b in below)
            cbm = min(b['min'] for b in above)
            ax_dos = fig.axes[-1]  # last axis is always DOS
            ax_dos.axvline(vbm, color='red', linestyle='--', alpha=0.5, label=f'VBM={vbm:.3f}')
            ax_dos.axvline(cbm, color='blue', linestyle='--', alpha=0.5, label=f'CBM={cbm:.3f}')
            ax_dos.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved to {save_path}")
    else:
        plt.savefig('all_ek.png', dpi=150)
        print(f"  Plot saved to all_ek.png")

    plt.close(fig)
    return fig
