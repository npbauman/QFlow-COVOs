"""
File readers for the "aperiodic" integral format: one-electron and
two-electron integrals each in a single combined file (the two-electron
file is not pre-split into Coulomb/exchange), plus the spin-orbital
expansion and antisymmetrization needed to build the Hamiltonian.
"""
from __future__ import annotations

import os

import numpy as np


def read_ion_ion_repulsion(filepath: str) -> float:
    """Read the scalar ion-ion (nuclear) repulsion energy from file."""
    with open(filepath, "r") as f:
        return float(f.readline().strip())


def read_one_electron_integrals(filepath: str):
    """
    Read one-electron integrals from file (spatial-orbital basis).

    File format:
        line 1: n_occ n_virt n_entries
        following lines: i j value   (1-based spatial orbital indices)
    """
    with open(filepath, "r") as f:
        first_line = f.readline().split()
        n_occ = int(first_line[0])
        n_virt = int(first_line[1])
        n_entries = int(first_line[2])

        n_orb = n_occ + n_virt
        one_body = np.zeros((n_orb, n_orb))

        for _ in range(n_entries):
            line = f.readline().split()
            i, j, val = int(line[0]) - 1, int(line[1]) - 1, float(line[2])
            one_body[i, j] = val
            one_body[j, i] = val

    return n_occ, n_virt, one_body


def read_two_electron_integrals(filepath: str):
    """
    Read two-electron integrals from a single combined file, applying
    8-fold permutational symmetry.

    File format:
        line 1: n_occ n_virt n_entries
        following lines: i j k l value   (1-based spatial orbital indices)
    """
    with open(filepath, "r") as f:
        first_line = f.readline().split()
        n_occ = int(first_line[0])
        n_virt = int(first_line[1])
        n_entries = int(first_line[2])

        n_orb = n_occ + n_virt
        two_body = np.zeros((n_orb, n_orb, n_orb, n_orb))

        for _ in range(n_entries):
            line = f.readline().split()
            i, j, k, l = int(line[0]) - 1, int(line[1]) - 1, int(line[2]) - 1, int(line[3]) - 1
            val = float(line[4])

            symmetries = [
                (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (k, l, j, i), (l, k, i, j), (l, k, j, i),
            ]
            for idx in symmetries:
                two_body[idx] = val

    return n_occ, n_virt, two_body


def _spinorb_index(i, spin, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta):
    """Map a spatial orbital + spin to a spin-orbital index under the
    [alpha_occ | beta_occ | alpha_virt | beta_virt] ordering."""
    if spin == 0:  # alpha
        if i < n_occ_alpha:
            return i
        else:
            return n_occ_alpha + n_occ_beta + (i - n_occ_alpha)
    else:  # beta
        if i < n_occ_beta:
            return n_occ_alpha + i
        else:
            return n_occ_alpha + n_occ_beta + n_virt_alpha + (i - n_occ_beta)


def build_spin_orbital_integrals(one_body: np.ndarray, two_body: np.ndarray, n_occ: int, n_virt: int):
    """
    Expand spatial-orbital one- and two-electron integrals into
    spin-orbital form (restricted: same spatial integrals for alpha and
    beta spin channels). Cross-spin two-electron blocks are filled
    (spin_p == spin_q and spin_r == spin_s) since this represents the
    spin-summed (not yet antisymmetrized) two-electron tensor.
    """
    n_orb = n_occ + n_virt
    num_spin_orbs = 2 * n_orb

    soei = np.zeros((num_spin_orbs, num_spin_orbs))
    stei = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))

    n_occ_alpha = n_occ
    n_occ_beta = n_occ
    n_virt_alpha = n_virt
    n_virt_beta = n_virt

    for p in range(n_orb):
        for q in range(n_orb):
            for spin in [0, 1]:
                p_so = _spinorb_index(p, spin, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                q_so = _spinorb_index(q, spin, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                soei[p_so, q_so] = one_body[p, q]

    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    for spin_p in [0, 1]:
                        for spin_q in [0, 1]:
                            for spin_r in [0, 1]:
                                for spin_s in [0, 1]:
                                    if (spin_p == spin_q) and (spin_r == spin_s):
                                        p_so = _spinorb_index(
                                            p, spin_p, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta
                                        )
                                        q_so = _spinorb_index(
                                            q, spin_q, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta
                                        )
                                        r_so = _spinorb_index(
                                            r, spin_r, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta
                                        )
                                        s_so = _spinorb_index(
                                            s, spin_s, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta
                                        )
                                        stei[p_so, q_so, r_so, s_so] = two_body[p, q, r, s]

    return soei, stei


def build_antisymmetrized_integrals(stei: np.ndarray, num_spin_orbs: int) -> np.ndarray:
    """Build antisymmetrized two-electron integrals <pq||rs> = <pr|qs> - <ps|qr>."""
    atei = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))
    for p in range(num_spin_orbs):
        for q in range(num_spin_orbs):
            for r in range(num_spin_orbs):
                for s in range(num_spin_orbs):
                    atei[p, q, r, s] = stei[p, r, q, s] - stei[p, s, q, r]
    return atei


def load_aperiodic_hamiltonian(input_dir: str):
    """
    Load all aperiodic-format Hamiltonian data from ``input_dir`` and
    build the spin-orbital antisymmetrized integrals needed by the
    matrix builder.

    Expects the files:
        ion_ion.dat
        one_electron_integrals.dat
        two_electron_integrals.dat

    Returns
    -------
    dict with keys: repulsion, soei, atei, n_occ, n_virt, n_orb
    """
    repulsion = read_ion_ion_repulsion(os.path.join(input_dir, "ion_ion.dat"))

    n_occ_1, n_virt_1, one_body = read_one_electron_integrals(
        os.path.join(input_dir, "one_electron_integrals.dat")
    )
    n_occ_2, n_virt_2, two_body = read_two_electron_integrals(
        os.path.join(input_dir, "two_electron_integrals.dat")
    )

    if n_occ_1 != n_occ_2 or n_virt_1 != n_virt_2:
        raise ValueError("Inconsistent orbital counts between one- and two-electron integral files")

    n_occ, n_virt = n_occ_1, n_virt_1
    n_orb = n_occ + n_virt

    soei, stei = build_spin_orbital_integrals(one_body, two_body, n_occ, n_virt)
    num_spin_orbs = 2 * n_orb
    atei = build_antisymmetrized_integrals(stei, num_spin_orbs)

    return {
        "repulsion": repulsion,
        "soei": soei,
        "atei": atei,
        "n_occ": n_occ,
        "n_virt": n_virt,
        "n_orb": n_orb,
    }
