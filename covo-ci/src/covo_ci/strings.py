"""
Determinant string generation shared by both the aperiodic and periodic
solvers, with optional excitation-level truncation (Select-CI).

Determinant layout convention used throughout this package:

    [ alpha_occ | beta_occ | alpha_virt | beta_virt ]

This is the same convention used in the original fci_string.F-derived
codes. All downstream phase and matrix-element logic assumes this exact
ordering.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def generate_half_strings(norb: int, n_elec: int, max_excitation: int | None = None) -> np.ndarray:
    """
    Generate half-determinant occupation strings (one spin channel) for
    ``n_elec`` electrons in ``norb`` orbitals.

    Parameters
    ----------
    norb : int
        Number of spatial orbitals for this spin channel.
    n_elec : int
        Number of electrons occupying this spin channel.
    max_excitation : int or None
        If ``None``, generate the full set of C(norb, n_elec) strings
        (standard FCI behavior). If an integer, only include strings that
        are *at most* ``max_excitation`` excitations away from the
        reference determinant (first ``n_elec`` orbitals occupied) when
        considering this spin channel alone. This mirrors the half-string
        pruning step in the original Select-CI script: it is intentionally
        a per-spin-channel truncation applied prior to the final
        full-determinant excitation-level filter in
        :func:`build_fci_strings_from_half_strings`, and may in general
        exclude some valid same-total-excitation-level determinants whose
        excitations are unevenly split across spin channels. This behavior
        is preserved for backward compatibility with the original
        Select-CI implementation.

    Returns
    -------
    np.ndarray
        Array of shape (n_strings, norb) of 0/1 occupation strings.
    """
    if max_excitation is None:
        strings = []
        for occ in combinations(range(norb), n_elec):
            bitstring = [0] * norb
            for i in occ:
                bitstring[i] = 1
            strings.append(bitstring)
        return np.array(strings, dtype=int)

    # Select-CI behavior: reference + bounded excitations from reference,
    # restricted to this spin channel.
    ref_string = [1] * n_elec + [0] * (norb - n_elec)
    strings = [ref_string]

    occ_indices = list(range(n_elec))
    virt_indices = list(range(n_elec, norb))

    for exc_level in range(1, max_excitation + 1):
        for holes in combinations(occ_indices, exc_level):
            for particles in combinations(virt_indices, exc_level):
                new_occ = [i for i in occ_indices if i not in holes] + list(particles)
                bitstring = [0] * norb
                for i in new_occ:
                    bitstring[i] = 1
                strings.append(bitstring)

    return np.array(strings, dtype=int)


def count_excitation_level(string, reference) -> int:
    """Number of excitations (holes == particles) between two full strings."""
    diff = np.array(string) - np.array(reference)
    n_holes = np.sum(diff == -1)
    return int(n_holes)


def build_fci_strings_from_half_strings(
    half_str_a: np.ndarray,
    half_str_b: np.ndarray,
    noas: int,
    nvas: int,
    nobs: int,
    nvbs: int,
    max_excitation: int | None = None,
) -> np.ndarray:
    """
    Combine alpha and beta half-strings into full determinant strings,
    laid out as [alpha_occ | beta_occ | alpha_virt | beta_virt].

    If ``max_excitation`` is given, only determinants whose *total*
    excitation level (relative to the combined reference determinant)
    is at most ``max_excitation`` are kept. This is the Select-CI
    determinant filter. Pass ``max_excitation=None`` for full CI (no
    filtering, every alpha/beta half-string combination is kept).

    Returns
    -------
    np.ndarray
        Array of shape (n_determinants, nstot) of 0/1 determinant strings.
    """
    alpha_str = half_str_a.shape[0]
    beta_str = half_str_b.shape[0]

    nos = noas + nobs
    nvs = nvas + nvbs
    nstot = nos + nvs

    ref_string = [1] * nos + [0] * nvs

    strings = []
    for i in range(alpha_str):
        for j in range(beta_str):
            string = [0] * nstot
            string[0:noas] = half_str_a[i, 0:noas]
            string[nos:nos + nvas] = half_str_a[i, noas:noas + nvas]
            string[noas:noas + nobs] = half_str_b[j, 0:nobs]
            string[nos + nvas:nos + nvas + nvbs] = half_str_b[j, nobs:nobs + nvbs]

            if sum(string) != (noas + nobs):
                raise ValueError("Mismatch in total electron count!")

            if max_excitation is not None:
                excitation_level = count_excitation_level(string, ref_string)
                if excitation_level > max_excitation:
                    continue

            strings.append(string)

    return np.array(strings, dtype=int)


def get_idiff_and_diff(string1: np.ndarray, string2: np.ndarray):
    """Return (number of orbital differences / 2, raw difference array)."""
    diff = string1 - string2
    idiff = int(np.sum(np.abs(diff)) // 2)
    return idiff, diff


def get_excitation_indices(diff: np.ndarray):
    """Split a determinant difference into (created, annihilated) indices."""
    plus = [i for i, d in enumerate(diff) if d == -1]
    minus = [i for i, d in enumerate(diff) if d == 1]
    return plus, minus


def compute_iphase(p: int, q: int, r: int, s: int, string: np.ndarray) -> int:
    """Fermionic sign for a double excitation r,s -> p,q acting on ``string``."""
    tmp = string.copy()
    isum = np.sum(tmp[:r])
    tmp[r] = 0
    isum += np.sum(tmp[:s])
    tmp[s] = 0
    isum += np.sum(tmp[:q])
    tmp[q] = 1
    isum += np.sum(tmp[:p])
    tmp[p] = 1
    return (-1) ** isum


def generate_single_double_excitations(string: np.ndarray, n_occ: int):
    """
    Generate all unique single and double excitations of ``string``
    (used to enumerate candidate nonzero off-diagonal Hamiltonian
    elements; ``n_occ`` is unused but kept for interface compatibility).
    """
    occ_indices = [i for i, x in enumerate(string) if x == 1]
    virt_indices = [i for i, x in enumerate(string) if x == 0]
    excitations = []

    for i in occ_indices:
        for a in virt_indices:
            new_str = string.copy()
            new_str[i] = 0
            new_str[a] = 1
            excitations.append(tuple(new_str))

    for i1 in range(len(occ_indices)):
        for i2 in range(i1 + 1, len(occ_indices)):
            for a1 in range(len(virt_indices)):
                for a2 in range(a1 + 1, len(virt_indices)):
                    new_str = string.copy()
                    new_str[occ_indices[i1]] = 0
                    new_str[occ_indices[i2]] = 0
                    new_str[virt_indices[a1]] = 1
                    new_str[virt_indices[a2]] = 1
                    excitations.append(tuple(new_str))

    return excitations
