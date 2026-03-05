"""Thin MPI wrapper for k-point parallelization.

If mpi4py is installed, uses MPI communicators for multi-node parallelism.
Otherwise, falls back to serial execution transparently.
"""

import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1


def is_root():
    """True if this is rank 0 (or serial mode)."""
    return rank == 0


def bcast(obj):
    """Broadcast object from rank 0 to all ranks."""
    if comm is None or size == 1:
        return obj
    return comm.bcast(obj, root=0)


def scatter_work(items):
    """Distribute items round-robin across ranks. Returns local slice."""
    if size == 1:
        return list(range(len(items))), items
    local_indices = list(range(rank, len(items), size))
    local_items = [items[i] for i in local_indices]
    return local_indices, local_items


def reduce_sum_array(local_array):
    """Allreduce (SUM) a numpy array across all ranks. Returns summed array."""
    if comm is None or size == 1:
        return local_array
    total = np.zeros_like(local_array)
    comm.Allreduce(local_array, total, op=MPI.SUM)
    return total


def reduce_sum_complex_array(local_array):
    """Allreduce (SUM) a complex numpy array across all ranks."""
    if comm is None or size == 1:
        return local_array
    total = np.zeros_like(local_array)
    comm.Allreduce(local_array, total, op=MPI.SUM)
    return total


def gather_array(local_indices, local_array, total_count):
    """Gather indexed rows into a full array on all ranks.

    Parameters
    ----------
    local_indices : list of int — which rows this rank computed
    local_array : ndarray of shape (n_local, ...) — local results
    total_count : int — total number of rows

    Returns
    -------
    ndarray of shape (total_count, ...) on all ranks
    """
    if size == 1:
        return local_array

    # Place local results into a full-size array (zeros elsewhere)
    shape = (total_count,) + local_array.shape[1:]
    full = np.zeros(shape, dtype=local_array.dtype)
    for i, gi in enumerate(local_indices):
        full[gi] = local_array[i]

    # Sum across ranks — each rank contributed non-overlapping rows
    result = np.zeros_like(full)
    comm.Allreduce(full, result, op=MPI.SUM)
    return result


def print_root(msg):
    """Print only on rank 0."""
    if is_root():
        print(msg)
