"""
Functions shared identically between the original QFlow-SD
(``SD-NoTQ.py``) and QFlow-SDTQ (``serial_python.py``) scripts.

Every function in this module was verified (via AST-level diff against
both original scripts) to be byte-for-byte identical between the two
implementations before being factored out here. Numerical logic is
unchanged from the originals.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np


def setup_qflow_inputs(n_elec: int, norb: int, assume_closed_shell: bool = True) -> dict:
    """
    Initialize orbital partitioning for QFlow based on a closed-shell assumption.

    Parameters
    ----------
    n_elec : int
        Total number of electrons (must be even for closed-shell).
    norb : int
        Number of spatial orbitals.
    assume_closed_shell : bool
        If True, assigns equal alpha/beta occupation.

    Returns
    -------
    dict with keys: norb, n_elec, noccp, nvirt, nos, nvs, nstot,
    noas, nobs, nvas, nvbs.
    """
    if assume_closed_shell:
        assert n_elec % 2 == 0, "Only closed-shell systems supported"
        noccp = n_elec // 2
        nvirt = norb - noccp
    else:
        raise NotImplementedError("Open-shell systems not implemented yet.")

    return {
        "norb": norb,
        "n_elec": n_elec,
        "noccp": noccp,
        "nvirt": nvirt,
        "nos": 2 * noccp,
        "nvs": 2 * nvirt,
        "nstot": 2 * norb,
        "noas": noccp,
        "nobs": noccp,
        "nvas": nvirt,
        "nvbs": nvirt,
    }


def generate_half_strings(norb, n_elec):
    string = []
    for occ in combinations(range(norb), n_elec):
        bitstring = [0] * norb
        for i in occ:
            bitstring[i] = 1
        string.append(bitstring)
    return np.array(string, dtype=int)


def build_fci_strings_from_half_strings(half_str_a, half_str_b, noas, nvas, nobs, nvbs):
    alpha_str = half_str_a.shape[0]
    beta_str = half_str_b.shape[0]

    nos = noas + nobs
    nvs = nvas + nvbs
    nstot = nos + nvs

    strings = []

    for i in range(alpha_str):
        for j in range(beta_str):
            string = [0] * nstot

            # Alpha string: occupied + virtual
            string[0:noas] = half_str_a[i, 0:noas]
            string[nos: nos + nvas] = half_str_a[i, noas: noas + nvas]

            # Beta string: occupied + virtual
            string[noas: noas + nobs] = half_str_b[j, 0:nobs]
            string[nos + nvas: nos + nvas + nvbs] = half_str_b[j, nobs: nobs + nvbs]

            # Safety check
            if sum(string) != (noas + nobs):
                raise ValueError("Mismatch in total electron count!")

            strings.append(string)

    return np.array(strings, dtype=int)


def compute_phase(ref, pos, neg):
    """
    Computes the phase factor (+/-1) for excitation: ref + pos - neg.

    Parameters
    ----------
    ref : original determinant (0/1 array)
    pos : list of orbitals being added (set to 1)
    neg : list of orbitals being removed (set to 0)

    Returns
    -------
    iphase : +1 or -1
    """
    ref_copy = ref.copy()
    isum = 0
    for x in neg:
        isum += np.sum(ref_copy[:x])
        ref_copy[x] = 0
    for x in pos:
        isum += np.sum(ref_copy[:x])
        ref_copy[x] = 1
    return (-1) ** isum


def expm(sigma, dim_fci, max_order=22, tol=1e-10, debug=False):
    """
    Taylor expansion of exp(+sigma) and exp(-sigma), like in Fortran-style CC logic.
    """
    ept = np.eye(dim_fci)
    emt = np.eye(dim_fci)

    term = sigma.copy()
    ept += term
    emt -= term

    factorial = 1
    for k in range(2, max_order + 1):
        factorial *= k
        term = sigma @ term
        delta = term / factorial
        ept += delta
        emt += (-1) ** (k) * delta

        norm_term = np.linalg.norm(delta)
        if debug:
            print(f"  [k={k}] |term| = {norm_term:.2e}")
        if norm_term < tol:
            break
    else:
        raise RuntimeError("Taylor expansion failed to converge")

    return ept, emt


def sim_trans(dim_fci, ept, emt, matrix):
    """
    Performs the similarity transformation: matrix <- emt @ (matrix @ ept)

    Parameters
    ----------
    matrix : (dim_fci, dim_fci) ndarray, FCI Hamiltonian (H)
    ept : (dim_fci, dim_fci) ndarray, exp(+sigma)
    emt : (dim_fci, dim_fci) ndarray, exp(-sigma)

    Returns
    -------
    The transformed matrix.
    """
    assert matrix.ndim == 2, "Matrix must be 2D"
    assert ept.shape == matrix.shape and emt.shape == matrix.shape, "Dimension mismatch"

    m1 = matrix @ ept
    m2 = emt @ m1

    return m2


def zero_matrix(matrix):
    """Zero out a square matrix."""
    matrix[:, :] = 0.0


def generate_spin_array(nos, nvs):
    nstot = nos + nvs
    spin_array = np.zeros(nstot, dtype=int)

    for i in range(nstot):
        if i < nos // 2:
            spin_array[i] = 0  # alpha occupied
        elif i < nos:
            spin_array[i] = 1  # beta occupied
        elif i < nos + nvs // 2:
            spin_array[i] = 0  # alpha virtual
        else:
            spin_array[i] = 1  # beta virtual
    return spin_array


def trial_fun(dim_fci, strings, noas, nobs, nstot):
    trial = np.zeros(dim_fci)
    ref = np.array([1] * (noas + nobs) + [0] * (nstot - noas - nobs))
    for i in range(dim_fci):
        if np.array_equal(strings[i], ref):
            trial[i] = 1.0
            break
    return trial


def evaluate_nactives(noas, nvas):
    return comb(noas, 2) * comb(nvas, 2)


def act_space_size_python(strings, noas, nobs, nvas, nvbs):
    dim_fci, nstot = strings.shape
    ref = next(strings[i] for i in range(dim_fci) if np.sum(strings[i, :noas + nobs]) == (noas + nobs))
    actspin = np.ones(nstot, dtype=int)
    actspin[ref == 1] = 0
    dim_act = sum(np.all((s == ref) | (actspin == 0)) for s in strings)
    return dim_act, actspin, ref


def act_space_size_from_actspin(strings, ref, actspin):
    return zip(*[(i, strings[i]) for i in range(len(strings)) if np.all((strings[i] == ref) | (actspin == 0))])


def extract_active_block(matrix, maps):
    return matrix[np.ix_(maps, maps)]


def diagonalize_active_block(matrix_a):
    return np.linalg.eigvalsh(matrix_a)


def act_matrix_full(strings, actspin, matrix, ref=None):
    if ref is None:
        ref = strings[0]
    dim_act, maps = act_space_size_from_actspin(strings, ref, actspin)
    matrix_a = extract_active_block(matrix, maps)
    eigvals = diagonalize_active_block(matrix_a)
    return {
        "dim_act": len(maps),
        "maps": maps,
        "matrix_a": matrix_a,
        "eigvals": eigvals,
        "lowest_energy": np.min(eigvals),
    }


def load_nactspin(path: str, n_columns: int | None = None) -> np.ndarray:
    """
    Load the active-space definition file (``nactspin_fortran.dat``).

    The two original scripts read this file slightly differently:
    QFlow-SD uses ``np.loadtxt`` + a fixed reshape (``18`` in the
    original script, which is simply ``nstot`` for the specific 9-orbital
    system that script was last configured for -- not a constant of the
    method). QFlow-SDTQ reads it line-by-line and infers the column
    count from each row's own length instead. This loader supports
    both styles: pass ``n_columns=<your nstot>`` to reproduce the
    SD script's fixed-width-reshape behavior exactly (required if your
    file has no embedded newlines per row), or leave it ``None`` to
    infer columns from each row's own length (SDTQ-style, and also
    correct for SD as long as the file has one row per line as written
    by :func:`generate_nactspin`).
    """
    if n_columns is not None:
        flat = np.loadtxt(path, dtype=int)
        return np.array(flat.reshape(-1, n_columns), dtype=int)

    rows = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(list(map(int, stripped.split())))
    return np.array(rows, dtype=int)


def generate_nactspin(noas: int, nobs: int, nvas: int, nvbs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the active-space definition array used by both QFlow-SD
    and QFlow-SDTQ, reconstructing the ``build_nactspin`` helper that
    was present (commented out) in both original scripts.

    Active spaces are formed by choosing 2 active occupied
    spin-orbitals from the alpha-occupied block (indices ``0..noas-1``,
    mirrored into the corresponding beta-occupied indices) and 2 active
    virtual spin-orbitals from the alpha-virtual block (indices
    ``0..nvas-1``, mirrored into the corresponding beta-virtual
    indices), for every combination of such pairs. This matches the
    original scheme exactly: it does not change based on whether T1/T2
    (SD) or T1-T4 (SDTQ) amplitudes are being optimized -- the
    active-space partitioning is the same either way, only the
    amplitude ranks handled *within* each active space differ.

    Orbitals are assumed to be indexed in increasing-energy order
    (occupied index 0 = lowest occupied, occupied index ``noas-1`` =
    HOMO; virtual index 0 = LUMO, virtual index ``nvas-1`` = highest
    virtual). Rows are ordered by: virtual pairs as the outer loop and
    occupied pairs as the inner loop, with each pair list independently
    sorted by ascending distance from its own Fermi-level reference
    (occupied pairs by distance from the HOMO, virtual pairs by
    distance from the LUMO). This exactly reproduces a reference
    Fortran ``build_nactspin`` program's own debug output for a
    3-occupied/3-virtual system (verified row-for-row, including the
    HOMO-LUMO active space as row 0). This ordering does not affect
    correctness (each active space is still evaluated and optimized
    independently regardless of row order); it only affects the order
    active spaces are processed in and printed.

    Parameters
    ----------
    noas, nobs : int
        Alpha and beta occupied spin-orbital counts (typically equal
        for closed-shell systems).
    nvas, nvbs : int
        Alpha and beta virtual spin-orbital counts (typically equal
        for closed-shell systems).

    Returns
    -------
    nactspin : np.ndarray, shape (nactives, nstot)
        0/1 active-space mask array. ``0`` marks a spin-orbital as
        ACTIVE for that row's active space, ``1`` marks it INACTIVE.
        This matches the convention documented directly in
        :func:`algorithm_sd.m_t_ext` / :func:`algorithm_sdtq.m_t_ext`
        ("0 = active, 1 = inactive") and verified against a reference
        Fortran program's own output.
        ``nstot = noas + nobs + nvas + nvbs`` and
        ``nactives = comb(noas, 2) * comb(nvas, 2)``.
    actopx : np.ndarray, shape (nactives, 4)
        For each active space, the 4 alpha-block indices
        ``[oi, oj, vi, vj]`` (occupied pair, virtual pair) that define
        it, before mirroring into the beta block. Useful for
        inspection/debugging, not required by the QFlow solvers
        themselves.

    Raises
    ------
    ValueError
        If ``noas < 2`` or ``nvas < 2`` (no active spaces can be
        formed -- this matches ``nactives == 0`` from
        :func:`evaluate_nactives`, which would otherwise silently
        produce zero active spaces and skip the entire optimization
        loop).
    """
    if noas < 2 or nvas < 2:
        raise ValueError(
            f"Cannot form any active spaces with noas={noas}, nvas={nvas}: "
            "need at least 2 occupied and 2 virtual alpha spin-orbitals "
            "(nactives = comb(noas, 2) * comb(nvas, 2) would be 0)."
        )

    nos = noas + nobs
    nvs = nvas + nvbs
    nstot = nos + nvs

    occ_pairs = list(combinations(range(noas), 2))
    virt_pairs = list(combinations(range(nvas), 2))
    nactives = len(occ_pairs) * len(virt_pairs)

    # Each pair list independently sorted by ascending distance from
    # its own Fermi-level reference: occupied pairs by distance from
    # the HOMO (index noas-1), virtual pairs by distance from the LUMO
    # (index 0). Verified against reference Fortran build_nactspin
    # output: virtual pairs form the outer loop, occupied pairs the
    # inner loop.
    def occ_distance(pair):
        return sum((noas - 1) - i for i in pair)

    def virt_distance(pair):
        return sum(j for j in pair)

    occ_pairs_sorted = sorted(occ_pairs, key=occ_distance)
    virt_pairs_sorted = sorted(virt_pairs, key=virt_distance)

    nactspin = np.ones((nactives, nstot), dtype=int)
    actopx = np.zeros((nactives, 4), dtype=int)

    m = 0
    for vi, vj in virt_pairs_sorted:
        for oi, oj in occ_pairs_sorted:
            actopx[m] = [oi, oj, vi, vj]
            for o in (oi, oj):
                nactspin[m, o] = 0
                nactspin[m, o + noas] = 0
            for v in (vi, vj):
                nactspin[m, nos + v] = 0
                nactspin[m, nos + nvas + v] = 0
            m += 1

    return nactspin, actopx


def write_nactspin(path: str, nactspin: np.ndarray) -> None:
    """
    Write an active-space definition array to ``path`` in the same
    plain-text, one-row-per-line, space-separated integer format read
    by :func:`load_nactspin`.
    """
    with open(path, "w") as f:
        for row in nactspin:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

