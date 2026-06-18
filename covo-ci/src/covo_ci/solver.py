"""
Diagonalization and energy/spin reporting shared by both the aperiodic
and periodic solvers.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from .spin import s2_expectation_correct, s_from_s2


def diagonalize_with_spin(matrix: np.ndarray, fci_strings: np.ndarray, noccp: int, nvirt: int, k: int = 5):
    """
    Diagonalize the CI matrix for the lowest ``k`` roots and report each
    root's energy, <S^2>, S, and spin multiplicity.

    For very small matrices (dim <= k), falls back to dense
    ``np.linalg.eigh`` since ``scipy.sparse.linalg.eigsh`` requires
    k < dim.

    Returns
    -------
    vals : np.ndarray, shape (k,)
    vecs : np.ndarray, shape (dim, k)
    """
    dim = matrix.shape[0]
    k_requested = max(1, min(k, dim))

    if dim < 50:
        # Dense fallback: robust for small matrices, and allows
        # k == dim (np.linalg.eigh has no such restriction, unlike
        # scipy.sparse.linalg.eigsh which requires k < dim).
        all_vals, all_vecs = np.linalg.eigh(matrix)
        order = np.argsort(all_vals)
        k = k_requested
        vals = all_vals[order][:k]
        vecs = all_vecs[:, order][:, :k]
    else:
        k = min(k_requested, dim - 1)
        vals, vecs = eigsh(matrix, k=k, which='SA')
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]

    print("Lowest CI energies and spin multiplicities:")
    for m in range(k):
        e = float(vals[m])
        vec = vecs[:, m]
        s2 = s2_expectation_correct(vec, fci_strings, noccp, nvirt)
        S = s_from_s2(s2)
        mult = int(round(2 * S + 1))
        print(
            f"  state {m + 1:2d}:  "
            f"E = {e: .10f}   "
            f"<S^2> = {s2:.8f}   "
            f"S = {S:.6f}   "
            f"multiplicity = {mult}"
        )

    return vals, vecs
