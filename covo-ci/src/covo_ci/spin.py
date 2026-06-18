"""
Spin operator machinery (S+, S-, <S^2>) shared by both the aperiodic and
periodic solvers. Identical logic in both original scripts; consolidated
here.

Assumes the determinant layout [alpha_occ | beta_occ | alpha_virt | beta_virt]
for the "spatial -> (alpha_idx, beta_idx)" mapping below.
"""
from __future__ import annotations

import numpy as np


def alpha_beta_indices_for_spatial(p: int, noccp: int, nvirt: int):
    """
    Return (alpha_idx, beta_idx) spin-orbital indices for spatial orbital p
    under the determinant ordering [alpha_occ | beta_occ | alpha_virt | beta_virt].
    """
    nos = 2 * noccp  # length of [alpha_occ | beta_occ]

    if p < noccp:  # occupied block
        ia = p
        ib = noccp + p
    else:  # virtual block
        ia = nos + (p - noccp)
        ib = nos + nvirt + (p - noccp)

    return ia, ib


def fermionic_phase_single(p_add: int, q_remove: int, det: np.ndarray) -> float:
    """
    Phase for a_dagger_{p_add} a_{q_remove} acting on |det>, using the
    standard "count occupied before index" convention, consistent with
    the bitstring ordering used throughout this package.
    """
    tmp = det.copy()
    isum = int(np.sum(tmp[:q_remove]))
    tmp[q_remove] = 0
    isum += int(np.sum(tmp[:p_add]))
    return -1.0 if (isum % 2) else 1.0


def splus_terms_of_det(det: np.ndarray, noccp: int, nvirt: int) -> dict:
    """Return dict {tuple(new_det): coeff} for S+|det>."""
    out = {}
    norb = noccp + nvirt
    for p in range(norb):
        ia, ib = alpha_beta_indices_for_spatial(p, noccp, nvirt)
        if det[ib] == 1 and det[ia] == 0:
            coeff = fermionic_phase_single(ia, ib, det)
            new_det = det.copy()
            new_det[ib] = 0
            new_det[ia] = 1
            tup = tuple(new_det)
            out[tup] = out.get(tup, 0.0) + coeff
    return out


def sminus_terms_of_det(det: np.ndarray, noccp: int, nvirt: int) -> dict:
    """Return dict {tuple(new_det): coeff} for S-|det>."""
    out = {}
    norb = noccp + nvirt
    for p in range(norb):
        ia, ib = alpha_beta_indices_for_spatial(p, noccp, nvirt)
        if det[ia] == 1 and det[ib] == 0:
            coeff = fermionic_phase_single(ib, ia, det)
            new_det = det.copy()
            new_det[ia] = 0
            new_det[ib] = 1
            tup = tuple(new_det)
            out[tup] = out.get(tup, 0.0) + coeff
    return out


def apply_op_to_vec_return_dict(vec: np.ndarray, fci_strings: np.ndarray, term_fn) -> dict:
    """
    Apply an operator defined by term_fn(det) -> {det': coeff} to a CI
    vector, returning the result as a dict over determinant tuples
    (not projected onto any particular basis).
    """
    out = {}
    for i, c in enumerate(vec):
        if c == 0.0:
            continue
        det = fci_strings[i]
        terms = term_fn(det)
        for tup, coeff in terms.items():
            out[tup] = out.get(tup, 0.0) + coeff * c
    return out


def project_detdict_to_basis(detdict: dict, str2idx: dict, dim: int) -> np.ndarray:
    """Project a dict over determinant tuples back into a basis vector."""
    y = np.zeros(dim, dtype=float)
    for tup, coeff in detdict.items():
        j = str2idx.get(tup)
        if j is not None:
            y[j] += coeff
    return y


def s2_expectation_correct(vec: np.ndarray, fci_strings: np.ndarray, noccp: int, nvirt: int) -> float:
    """
    <S^2> = <Sz^2> + 1/2( <psi|S+S-|psi> + <psi|S-S+|psi> )

    Computed safely even though intermediate Ms sectors are not
    necessarily present in the working basis (e.g. for a truncated
    Select-CI determinant list).
    """
    dim = len(fci_strings)
    str2idx = {tuple(s): i for i, s in enumerate(fci_strings)}

    # For Nalpha = Nbeta = noccp -> Ms = 0
    Ms = 0.5 * (noccp - noccp)
    sz2 = Ms * Ms

    splus_terms = lambda det: splus_terms_of_det(det, noccp, nvirt)
    sminus_terms = lambda det: sminus_terms_of_det(det, noccp, nvirt)

    # v1 = S- S+ |psi>
    v_sp_dict = apply_op_to_vec_return_dict(vec, fci_strings, splus_terms)
    v_sms_p_dict = {}
    for tup, amp in v_sp_dict.items():
        det = np.array(tup, dtype=int)
        for tup2, coeff2 in sminus_terms(det).items():
            v_sms_p_dict[tup2] = v_sms_p_dict.get(tup2, 0.0) + coeff2 * amp
    v_sms_p = project_detdict_to_basis(v_sms_p_dict, str2idx, dim)

    # v2 = S+ S- |psi>
    v_sm_dict = apply_op_to_vec_return_dict(vec, fci_strings, sminus_terms)
    v_sps_m_dict = {}
    for tup, amp in v_sm_dict.items():
        det = np.array(tup, dtype=int)
        for tup2, coeff2 in splus_terms(det).items():
            v_sps_m_dict[tup2] = v_sps_m_dict.get(tup2, 0.0) + coeff2 * amp
    v_sps_m = project_detdict_to_basis(v_sps_m_dict, str2idx, dim)

    term = 0.5 * (np.dot(vec, v_sms_p) + np.dot(vec, v_sps_m))
    return sz2 + term


def s_from_s2(s2: float) -> float:
    """Solve S(S+1) = s2 for S, clamping s2 to be non-negative first."""
    root = 1.0 + 4.0 * max(float(s2), 0.0)
    return (-1.0 + np.sqrt(root)) / 2.0
