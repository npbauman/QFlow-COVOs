"""
Hamiltonian matrix construction for the aperiodic (antisymmetrized
spin-orbital integral) format.
"""
from __future__ import annotations

import numpy as np

from .strings import (
    compute_iphase,
    generate_single_double_excitations,
    get_excitation_indices,
    get_idiff_and_diff,
)


def handle_diagonal_element(strings, i, h, v, noccp, nvirt, repulsion, matrix):
    """Compute the diagonal Hamiltonian matrix element for determinant i."""
    string = strings[i]
    ind_set = [l for l in range(len(string)) if string[l] == 1]

    if len(ind_set) != (2 * noccp):
        raise ValueError(f"Occupancy mismatch: expected {2 * noccp}, got {len(ind_set)}")

    diag_val = 0.0

    for p in ind_set:
        diag_val += h[p, p]

    for p in ind_set:
        for q in ind_set:
            diag_val += 0.5 * v[p, q, p, q]

    diag_val += repulsion
    matrix[i, i] += diag_val


def compute_single_excitation_element(strings, j, h, v, nocca, matrix_row_i, diff):
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 1 or len(minus) != 1:
        return None

    p = plus[0]
    q = minus[0]

    string2 = strings[j]
    ind_set = [l for l in range(len(string2)) if string2[l] == 1]

    if len(ind_set) != (2 * nocca):
        raise ValueError("Occupied orbital count mismatch.")

    tmp = string2.copy()
    isum = np.sum(tmp[:q])
    tmp[q] = 0
    isum += np.sum(tmp[:p])
    tmp[p] = 1
    iphase = (-1) ** isum

    one_e = h[p, q]
    two_e = 0.0
    for n in ind_set:
        two_e += v[p, n, q, n]

    return iphase * (one_e + two_e)


def compute_double_excitation_element(string1, v, diff):
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 2 or len(minus) != 2:
        return None

    p, q = sorted(plus)
    r, s = sorted(minus)

    iphase = compute_iphase(p, q, r, s, string1)
    return iphase * v[p, q, r, s]


def build_fci_matrix(fci_strings, h, v, nocca, nvirta, repulsion):
    """Build the (Select-)CI Hamiltonian matrix for the aperiodic integral format."""
    dim_fci = fci_strings.shape[0]
    matrix = np.zeros((dim_fci, dim_fci))

    string_to_index = {tuple(s): idx for idx, s in enumerate(fci_strings)}

    print(f"Dimension of CI matrix: {dim_fci}")

    for i in range(dim_fci):
        handle_diagonal_element(fci_strings, i, h, v, nocca, nvirta, repulsion, matrix)

    for i in range(dim_fci):
        string1 = fci_strings[i]
        excitations = generate_single_double_excitations(string1, nocca + nvirta)
        for ex_str in excitations:
            j = string_to_index.get(ex_str)
            if j is not None and j > i:
                string2 = fci_strings[j]
                idiff, diff = get_idiff_and_diff(string1, string2)
                val = None
                if idiff == 1:
                    val = compute_single_excitation_element(fci_strings, j, h, v, nocca, None, diff)
                elif idiff == 2:
                    val = compute_double_excitation_element(string1, v, diff)

                if val is not None:
                    matrix[i, j] += val
                    matrix[j, i] += val

    return matrix
