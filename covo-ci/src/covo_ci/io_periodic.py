"""
File readers for the "periodic" integral format: one-electron integrals
in a single file, and two-electron integrals split into separate
Coulomb and exchange files.
"""
from __future__ import annotations

import os

import numpy as np


def read_ion_ion_repulsion(filepath: str) -> float:
    """Read the scalar ion-ion (nuclear) repulsion energy from file."""
    with open(filepath, "r") as f:
        return float(f.readline().strip())


def read_one_electron_integrals(filename: str):
    """
    Read one-electron integrals from file and expand into spin-orbital
    form (alpha-alpha and beta-beta blocks only; alpha-beta and beta-alpha
    blocks are zero).

    File format:
        line 1: noccp nvirt n_entries
        following lines: p q value   (1-based spatial orbital indices)

    Returns
    -------
    h : np.ndarray, shape (2*norb, 2*norb)
        Spin-orbital one-electron integral matrix.
    noccp, nvirt, norb : int
    """
    with open(filename, "r") as f:
        noccp, nvirt, nh1 = map(int, f.readline().split())
        norb = noccp + nvirt

        ho = np.zeros((norb, norb))
        for _ in range(nh1):
            p, q, val = f.readline().split()
            p, q = int(p) - 1, int(q) - 1
            val = float(val)
            ho[p, q] = val
            ho[q, p] = val  # Hermitian symmetry

    nspin = 2 * norb
    h = np.zeros((nspin, nspin))

    for p in range(norb):
        for q in range(norb):
            pa, qa = p, q
            pb, qb = p + norb, q + norb
            h[pa, qa] = ho[p, q]  # alpha-alpha block
            h[pb, qb] = ho[p, q]  # beta-beta block
            # alpha-beta / beta-alpha blocks remain zero

    return h, noccp, nvirt, norb


def read_two_electron_integrals(filename: str, norb: int) -> np.ndarray:
    """
    Read a two-electron integral tensor (either Coulomb or exchange)
    from file, applying 8-fold permutational symmetry.

    File format:
        line 1: noccp nvirt n_entries
        following lines: p q r s value   (1-based spatial orbital indices)
    """
    tensor = np.zeros((norb, norb, norb, norb))

    with open(filename, "r") as f:
        _noccp, _nvirt, nv2 = map(int, f.readline().split())

        for _ in range(nv2):
            p, q, r, s, val = f.readline().split()
            p, q, r, s = int(p) - 1, int(q) - 1, int(r) - 1, int(s) - 1
            val = float(val)

            perms = [
                (p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p),
            ]
            for a, b, c, d in perms:
                tensor[a, b, c, d] = val

    return tensor


def map_spinorbital_to_spatial_and_spin(k: int, noccp: int, nvirt: int):
    """
    Map a spin-orbital index k (under the [alpha_occ | beta_occ |
    alpha_virt | beta_virt] ordering) to (spatial_index, spin), with
    spin 0 = alpha, spin 1 = beta.
    """
    thres1 = noccp
    thres2 = 2 * noccp
    thres3 = 2 * noccp + nvirt

    if k <= thres1 - 1:
        return k, 0  # alpha occupied
    elif k <= thres2 - 1:
        return k - noccp, 1  # beta occupied
    elif k <= thres3 - 1:
        return k - noccp, 0  # alpha virtual
    else:
        return k - noccp - nvirt, 1  # beta virtual


def build_spinorbital_map(noccp: int, nvirt: int):
    """Precompute the (spatial, spin) mapping for every spin-orbital index."""
    return [map_spinorbital_to_spatial_and_spin(k, noccp, nvirt) for k in range(2 * (noccp + nvirt))]


def load_periodic_hamiltonian(input_dir: str):
    """
    Load all periodic-format Hamiltonian data from ``input_dir``.

    Expects the files:
        ion_ion.dat
        one_electron_integrals.dat
        two_electron_integrals_coulomb.dat
        two_electron_integrals_exchange.dat

    Returns
    -------
    dict with keys: repulsion, h, vc, vx, noccp, nvirt, norb
    """
    repulsion = read_ion_ion_repulsion(os.path.join(input_dir, "ion_ion.dat"))
    h, noccp, nvirt, norb = read_one_electron_integrals(
        os.path.join(input_dir, "one_electron_integrals.dat")
    )
    vc = read_two_electron_integrals(
        os.path.join(input_dir, "two_electron_integrals_coulomb.dat"), norb
    )
    vx = read_two_electron_integrals(
        os.path.join(input_dir, "two_electron_integrals_exchange.dat"), norb
    )

    return {
        "repulsion": repulsion,
        "h": h,
        "vc": vc,
        "vx": vx,
        "noccp": noccp,
        "nvirt": nvirt,
        "norb": norb,
    }
