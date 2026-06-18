"""
Hamiltonian matrix construction for the periodic (Coulomb/exchange
split) integral format.
"""
from __future__ import annotations

import numpy as np

from .strings import (
    compute_iphase,
    generate_single_double_excitations,
    get_excitation_indices,
    get_idiff_and_diff,
)


def compute_double_excitation_element(string1, string2, vx, spinorbital_map, diff):
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 2 or len(minus) != 2:
        return None

    p, q = sorted(plus)
    r, s = sorted(minus)

    iphase = compute_iphase(p, q, r, s, string1)

    test_string = string1.copy()
    test_string[r] = 0
    test_string[s] = 0
    test_string[p] = 1
    test_string[q] = 1
    if not np.array_equal(test_string, string2):
        return None

    m, ps = spinorbital_map[p]
    n, qs = spinorbital_map[q]
    u, rs = spinorbital_map[r]
    w, ss = spinorbital_map[s]

    val = 0.0
    if ps == rs and qs == ss and ps != qs:
        val = vx[m, u, n, w]
    elif ps == ss and qs == rs and ps != qs:
        val = -vx[m, w, n, u]
    elif ps == rs and qs == ss and ps == qs:
        val = vx[m, u, n, w] - vx[m, w, n, u]

    return iphase * val


def compute_single_excitation_element(strings, j, h, vc, vx, noas, nobs, diff, spinorbital_map):
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 1 or len(minus) != 1:
        return None

    p = plus[0]
    q = minus[0]

    string2 = strings[j]
    ind_set = [(t, *spinorbital_map[t]) for t in range(len(string2)) if string2[t] == 1]

    if len(ind_set) != (noas + nobs):
        raise ValueError("Occupied orbital count mismatch.")

    tmp = string2.copy()
    isum = np.sum(tmp[:q])
    tmp[q] = 0
    isum += np.sum(tmp[:p])
    tmp[p] = 1
    iphase = (-1) ** isum

    p_h, _ = spinorbital_map[p]
    q_h, _ = spinorbital_map[q]
    one_e = h[p_h, q_h]

    w, ps = spinorbital_map[p]
    n, qs = spinorbital_map[q]

    two_e = 0.0
    for t, u, ts in ind_set:
        if ps == ts and p != t and q != t:
            val = 0.5 * vc[u, u, n, w] - vx[n, u, w, u]
        elif ps != ts:
            val = 0.5 * vc[u, u, n, w]
        elif p == t or q == t:
            val = 0.25 * vc[w, n, w, n] - 0.5 * vx[w, n, w, n]
        else:
            val = 0.0
        two_e += val

    return iphase * (one_e + two_e)


def handle_diagonal_element(strings, i, h, vc, vx, noas, nobs, repulsion, matrix, spinorbital_map):
    """
    Compute <D_i|H|D_i>: one-electron contribution + 0.5*sum_pq[J-K or J]
    two-electron contribution + nuclear repulsion.
    """
    string = strings[i]
    ind_set = [l for l in range(len(string)) if string[l] == 1]

    if len(ind_set) != (noas + nobs):
        raise ValueError(f"Occupancy mismatch: expected {noas + nobs}, got {len(ind_set)}")

    diag_val = 0.0

    for p in ind_set:
        p_h, _spin = spinorbital_map[p]
        diag_val += h[p_h, p_h]

    for p in ind_set:
        for q in ind_set:
            m, ps = spinorbital_map[p]
            n, qs = spinorbital_map[q]

            if ps == qs:
                v_pq = 0.5 * vc[m, m, n, n] - vx[n, m, m, n]
            else:
                v_pq = 0.5 * vc[m, m, n, n]

            diag_val += 0.5 * v_pq

    diag_val += repulsion
    matrix[i, i] += diag_val


def build_fci_matrix(fci_strings, h, vc, vx, noccp, nvirt, noas, nobs, repulsion, spinorbital_map):
    """Build the (Select-)CI Hamiltonian matrix for the periodic integral format."""
    dim_fci = fci_strings.shape[0]
    matrix = np.zeros((dim_fci, dim_fci), dtype=np.float64)

    string_to_index = {tuple(s): idx for idx, s in enumerate(fci_strings)}

    print(f"Dimension of CI matrix: {dim_fci}")

    for i in range(dim_fci):
        handle_diagonal_element(fci_strings, i, h, vc, vx, noas, nobs, repulsion, matrix, spinorbital_map)

    for i in range(dim_fci):
        string1 = fci_strings[i]
        excitations = generate_single_double_excitations(string1, noas + nobs)
        for ex_str in excitations:
            j = string_to_index.get(ex_str)
            if j is not None and j > i:
                string2 = fci_strings[j]
                idiff, diff = get_idiff_and_diff(string1, string2)
                val = None
                if idiff == 1:
                    val = compute_single_excitation_element(
                        fci_strings, j, h, vc, vx, noas, nobs, diff, spinorbital_map
                    )
                elif idiff == 2:
                    val = compute_double_excitation_element(
                        string1, string2, vx, spinorbital_map, diff
                    )

                if val is not None:
                    matrix[i, j] += val
                    matrix[j, i] += val

    return matrix
