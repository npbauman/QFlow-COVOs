"""
Standalone FCI solver that reads Hamiltonian integrals from files.
python fci_from_files.py /path/to/hamiltonian/directory [output_file]
"""
import numpy as np
import os
import sys
from itertools import combinations
from scipy.sparse.linalg import eigsh

def read_ion_ion_repulsion(filepath):
    """Read ion_ion repulsion energy from file."""
    with open(filepath, 'r') as f:
        repulsion = float(f.readline().strip())
    return repulsion


def read_one_electron_integrals(filepath):
    """Read one-electron integrals from file."""
    with open(filepath, 'r') as f:
        first_line = f.readline().split()
        n_occ = int(first_line[0])
        n_virt = int(first_line[1])
        n_entries = int(first_line[2])
        
        n_orb = n_occ + n_virt
        one_body = np.zeros((n_orb, n_orb))
        
        for _ in range(n_entries):
            line = f.readline().split()
            i, j, val = int(line[0]) - 1, int(line[1]) - 1, float(line[2])  # Convert to 0-based
            one_body[i, j] = val
            one_body[j, i] = val  # Symmetry
    
    return n_occ, n_virt, one_body


def read_two_electron_integrals(filepath):
    """Read two-electron integrals from file."""
    with open(filepath, 'r') as f:
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
            
            # Apply all 8-fold symmetry
            symmetries = [
                (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (k, l, j, i), (l, k, i, j), (l, k, j, i)
            ]
            for idx in symmetries:
                two_body[idx] = val
    
    return n_occ, n_virt, two_body


def build_spin_orbital_integrals(one_body, two_body, n_occ, n_virt):
    """Build spin-orbital one- and two-electron integrals."""
    n_orb = n_occ + n_virt
    num_spin_orbs = 2 * n_orb
    
    soei = np.zeros((num_spin_orbs, num_spin_orbs))
    stei = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))
    
    # Helper function for spin-orbital indexing
    def spinorb_index(i, spin, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta):
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
    
    n_occ_alpha = n_occ
    n_occ_beta = n_occ
    n_virt_alpha = n_virt
    n_virt_beta = n_virt
    
    # Fill one-electron integrals
    for p in range(n_orb):
        for q in range(n_orb):
            for spin in [0, 1]:
                p_so = spinorb_index(p, spin, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                q_so = spinorb_index(q, spin, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                soei[p_so, q_so] = one_body[p, q]
    
    # Fill two-electron integrals
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    for spin_p in [0, 1]:
                        for spin_q in [0, 1]:
                            for spin_r in [0, 1]:
                                for spin_s in [0, 1]:
                                    p_so = spinorb_index(p, spin_p, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                                    q_so = spinorb_index(q, spin_q, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                                    r_so = spinorb_index(r, spin_r, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                                    s_so = spinorb_index(s, spin_s, n_occ_alpha, n_occ_beta, n_virt_alpha, n_virt_beta)
                                    if (spin_p == spin_q) and (spin_r == spin_s):
                                        stei[p_so, q_so, r_so, s_so] = two_body[p, q, r, s] 
    
    return soei, stei


def build_antisymmetrized_integrals(stei, num_spin_orbs):
    """Build antisymmetrized two-electron integrals."""
    atei = np.zeros((num_spin_orbs, num_spin_orbs, num_spin_orbs, num_spin_orbs))
    for p in range(num_spin_orbs):
        for q in range(num_spin_orbs):
            for r in range(num_spin_orbs):
                for s in range(num_spin_orbs):
                    atei[p, q, r, s] = stei[p, r, q, s] - stei[p, s, q, r]
    return atei


def generate_half_strings(norb, n_elec):
    """Generate all possible half strings (alpha or beta)."""
    strings = []
    for occ in combinations(range(norb), n_elec):
        bitstring = [0] * norb
        for i in occ:
            bitstring[i] = 1
        strings.append(bitstring)
    return np.array(strings, dtype=int)


def build_fci_strings_from_half_strings(half_str_a, half_str_b, noas, nvas, nobs, nvbs):
    """Build full FCI determinant strings."""
    alpha_str = half_str_a.shape[0]
    beta_str = half_str_b.shape[0]
    
    nos = noas + nobs
    nvs = nvas + nvbs
    nstot = nos + nvs
    
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
            strings.append(string)
    
    return np.array(strings, dtype=int)


def generate_single_double_excitations(string, n_occ):
    """Generate all unique single and double excitations."""
    occ_indices = [i for i, x in enumerate(string) if x == 1]
    virt_indices = [i for i, x in enumerate(string) if x == 0]
    excitations = []
    
    # Single excitations
    for i in occ_indices:
        for a in virt_indices:
            new_str = string.copy()
            new_str[i] = 0
            new_str[a] = 1
            excitations.append(tuple(new_str))
    
    # Double excitations
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


def get_idiff_and_diff(string1, string2):
    """Get difference between two strings."""
    diff = string1 - string2
    idiff = int(np.sum(np.abs(diff)) // 2)
    return idiff, diff


def get_excitation_indices(diff):
    """Get excitation indices from difference."""
    plus = [i for i, d in enumerate(diff) if d == -1]
    minus = [i for i, d in enumerate(diff) if d == 1]
    return plus, minus


def compute_iphase(p, q, r, s, string):
    """Compute fermionic phase."""
    tmp = string.copy()
    isum = np.sum(tmp[:r]); tmp[r] = 0
    isum += np.sum(tmp[:s]); tmp[s] = 0
    isum += np.sum(tmp[:q]); tmp[q] = 1
    isum += np.sum(tmp[:p]); tmp[p] = 1
    return (-1) ** isum


def handle_diagonal_element(strings, i, h, v, noccp, nvirt, repulsion, matrix):
    """Compute diagonal matrix element."""
    string = strings[i]
    ind_set = [l for l in range(len(string)) if string[l] == 1]
    
    if len(ind_set) != (2 * noccp):
        raise ValueError(f"Occupancy mismatch: expected {2*noccp}, got {len(ind_set)}")
    
    diag_val = 0.0
    
    # One-electron contribution
    for p in ind_set:
        diag_val += h[p, p]
    
    # Two-electron contribution
    for p in ind_set:
        for q in ind_set:
            diag_val += 0.5 * v[p, q, p, q]
    
    # Nuclear repulsion
    diag_val += repulsion
    
    matrix[i, i] += diag_val


def compute_single_excitation_element(strings, i, j, h, v, nocca, nvirta, matrix, diff):
    """Compute single excitation matrix element."""
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 1 or len(minus) != 1:
        return
    
    p = plus[0]
    q = minus[0]
    
    string2 = strings[j]
    ind_set = [l for l in range(len(string2)) if string2[l] == 1]
    
    if len(ind_set) != (2 * nocca):
        raise ValueError("Occupied orbital count mismatch.")
    
    # Fermionic phase
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
    
    total = iphase * (one_e + two_e)
    matrix[i, j] += total
    matrix[j, i] += total


def compute_double_excitation_element(string1, string2, i, j, v, noccp, nvirt, matrix, diff):
    """Compute double excitation matrix element."""
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 2 or len(minus) != 2:
        return
    
    p, q = sorted(plus)
    r, s = sorted(minus)
    
    iphase = compute_iphase(p, q, r, s, string1)
    
    matrix[i, j] += iphase * v[p, q, r, s]
    matrix[j, i] += iphase * v[p, q, r, s]


def build_fci_matrix(fci_strings, h, v, nocca, nvirta, repulsion):
    """Build the full FCI matrix."""
    dim_fci = fci_strings.shape[0]
    matrix = np.zeros((dim_fci, dim_fci))
    
    string_to_index = {tuple(s): idx for idx, s in enumerate(fci_strings)}
    
    print(f"Dimension of FCI matrix: {dim_fci}")
    
    # Diagonal terms
    for i in range(dim_fci):
        handle_diagonal_element(fci_strings, i, h, v, nocca, nvirta, repulsion, matrix)
    
    # Off-diagonal terms
    for i in range(dim_fci):
        string1 = fci_strings[i]
        excitations = generate_single_double_excitations(string1, nocca + nvirta)
        for ex_str in excitations:
            j = string_to_index.get(ex_str)
            if j is not None and j > i:
                string2 = fci_strings[j]
                idiff, diff = get_idiff_and_diff(string1, string2)
                if idiff == 1:
                    compute_single_excitation_element(fci_strings, i, j, h, v, nocca, nvirta, matrix, diff)
                elif idiff == 2:
                    compute_double_excitation_element(string1, string2, i, j, v, nocca, nvirta, matrix, diff)
    
    return matrix

def alpha_beta_indices_for_spatial(p, noccp, nvirt):
    """
    Return (alpha_idx, beta_idx) spin-orbital indices for spatial orbital p
    under your determinant ordering: [α_occ | β_occ | α_virt | β_virt].
    """
    norb = noccp + nvirt
    nos = 2 * noccp  # length of [α_occ | β_occ]

    # alpha index
    if p < noccp:         # occupied block
        ia = p
    else:                 # virtual block
        ia = nos + (p - noccp)

    # beta index
    if p < noccp:         # occupied block
        ib = noccp + p
    else:                 # virtual block
        ib = nos + nvirt + (p - noccp)

    return ia, ib


def fermionic_phase_single(p_add, q_remove, det):
    """
    Phase for a†_{p_add} a_{q_remove} acting on |det>,
    using standard "count occupied before index" rule
    consistent with your bitstring ordering.
    """
    tmp = det.copy()
    isum = int(np.sum(tmp[:q_remove]))
    tmp[q_remove] = 0
    isum += int(np.sum(tmp[:p_add]))
    return -1.0 if (isum % 2) else 1.0


def splus_terms_of_det(det, noccp, nvirt):
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


def sminus_terms_of_det(det, noccp, nvirt):
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


def apply_op_to_vec_return_dict(vec, fci_strings, term_fn):
    """
    Apply an operator defined by term_fn(det)->{det':coeff} to a CI vector.
    Return result in an unrestricted dict over determinant tuples.
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


def project_detdict_to_basis(detdict, str2idx, dim):
    """Project unrestricted dict back into your basis vector."""
    y = np.zeros(dim, dtype=float)
    for tup, coeff in detdict.items():
        j = str2idx.get(tup)
        if j is not None:
            y[j] += coeff
    return y


def s2_expectation_correct(vec, fci_strings, noccp, nvirt):
    """
    <S^2> = <Sz^2> + 1/2( <psi|S+S-|psi> + <psi|S-S+|psi> )
    computed safely even though intermediate Ms sectors aren't in the basis.
    """
    dim = len(fci_strings)
    str2idx = {tuple(s): i for i, s in enumerate(fci_strings)}

    # For your current basis Nα=Nβ=noccp -> Ms = 0
    Ms = 0.5 * (noccp - noccp)
    sz2 = Ms * Ms

    Splus_terms  = lambda det: splus_terms_of_det(det, noccp, nvirt)
    Sminus_terms = lambda det: sminus_terms_of_det(det, noccp, nvirt)

    # v1 = S- S+ |psi>
    v_sp_dict = apply_op_to_vec_return_dict(vec, fci_strings, Splus_terms)
    v_sms_p_dict = {}
    for tup, amp in v_sp_dict.items():
        det = np.array(tup, dtype=int)
        for tup2, coeff2 in Sminus_terms(det).items():
            v_sms_p_dict[tup2] = v_sms_p_dict.get(tup2, 0.0) + coeff2 * amp
    v_sms_p = project_detdict_to_basis(v_sms_p_dict, str2idx, dim)

    # v2 = S+ S- |psi>
    v_sm_dict = apply_op_to_vec_return_dict(vec, fci_strings, Sminus_terms)
    v_sps_m_dict = {}
    for tup, amp in v_sm_dict.items():
        det = np.array(tup, dtype=int)
        for tup2, coeff2 in Splus_terms(det).items():
            v_sps_m_dict[tup2] = v_sps_m_dict.get(tup2, 0.0) + coeff2 * amp
    v_sps_m = project_detdict_to_basis(v_sps_m_dict, str2idx, dim)

    term = 0.5 * (np.dot(vec, v_sms_p) + np.dot(vec, v_sps_m))
    return sz2 + term


def s_from_s2(s2):
    """Solve S(S+1)=s2."""
    root = 1.0 + 4.0 * max(float(s2), 0.0)
    return (-1.0 + np.sqrt(root)) / 2.0


def report_spin_for_roots(vals, vecs, fci_strings, noccp, nvirt, k=None):
    """
    Print E, <S^2>, S, multiplicity for the first k roots.
    vecs expected shape: (dim, nroots)
    """
    if k is None:
        k = vecs.shape[1]

    print("Lowest energies and spin multiplicities:")
    for m in range(k):
        e = float(vals[m])
        v = vecs[:, m]
        s2 = float(s2_expectation_correct(v, fci_strings, noccp, nvirt))
        S = float(s_from_s2(s2))
        mult = int(round(2.0 * S + 1.0))
        print(f"  state {m+1:2d}:  E = {e: .10f}   <S^2> = {s2:.8f}   S = {S:.6f}   multiplicity = {mult}")
def diagonalize_fci_matrix_with_spin(matrix, fci_strings, noccp, nvirt, k=5):
    vals, vecs = eigsh(matrix, k=k, which='SA')
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    print("Lowest FCI energies and spin multiplicities:")
    for m in range(k):
        e = vals[m]
        vec = vecs[:, m]
        s2 = s2_expectation_correct(vec, fci_strings, noccp, nvirt)
        S = s_from_s2(s2)
        mult = int(round(2 * S + 1))
        print(
            f"  state {m+1:2d}:  "
            f"E = {e: .10f}   "
            f"<S^2> = {s2:.8f}   "
            f"S = {S:.6f}   "
            f"multiplicity = {mult}"
        )
    return vals, vecs

def main(input_dir, output_file="FCI_matrix.dat"):
    """Main function to read integrals and compute FCI."""
    print(f"Reading Hamiltonian from directory: {input_dir}")
    
    # Read ion_ion repulsion
    repulsion_file = os.path.join(input_dir, "ion_ion.dat")
    repulsion = read_ion_ion_repulsion(repulsion_file)
    print(f"Ion_ion repulsion: {repulsion:.8f}")
    
    # Read one-electron integrals
    one_e_file = os.path.join(input_dir, "one_electron_integrals.dat")
    n_occ_1, n_virt_1, one_body = read_one_electron_integrals(one_e_file)
    print(f"One-electron integrals: {n_occ_1} occupied, {n_virt_1} virtual orbitals")
    #print(one_body) 
    
    # Read two-electron integrals
    two_e_file = os.path.join(input_dir, "two_electron_integrals.dat")
    n_occ_2, n_virt_2, two_body = read_two_electron_integrals(two_e_file)
    print(f"Two-electron integrals: {n_occ_2} occupied, {n_virt_2} virtual orbitals")
    print(two_body)
    # Verify consistency
    if n_occ_1 != n_occ_2 or n_virt_1 != n_virt_2:
        raise ValueError("Inconsistent orbital counts between one- and two-electron integral files")
    
    n_occ = n_occ_1
    n_virt = n_virt_1
    n_orb = n_occ + n_virt
    n_elec = 2 * n_occ
    
    print(f"\nSystem info:")
    print(f"  Total spatial orbitals: {n_orb}")
    print(f"  Occupied orbitals: {n_occ}")
    print(f"  Virtual orbitals: {n_virt}")
    print(f"  Total electrons: {n_elec}")
    
    # Build spin-orbital integrals
    print("\nBuilding spin-orbital integrals...")
    soei, stei = build_spin_orbital_integrals(one_body, two_body, n_occ, n_virt)
    print("Spin orbital two electron")
    #print(stei)
    # Build antisymmetrized integrals
    print("Building antisymmetrized integrals...")
    num_spin_orbs = 2 * n_orb
    atei = build_antisymmetrized_integrals(stei, num_spin_orbs)
    print("Antisymmetrized two electron integral:")
    print(atei)
    
    # Generate FCI strings
    print("\nGenerating FCI strings...")
    half_str_a = generate_half_strings(n_orb, n_occ)
    half_str_b = generate_half_strings(n_orb, n_occ)
    
    # Sort strings
    half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
    half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]
    
    fci_strings = build_fci_strings_from_half_strings(
        half_str_a, half_str_b, n_occ, n_virt, n_occ, n_virt
    )
    
    # Build FCI matrix
    print("\nBuilding FCI matrix...")
    fci_matrix = build_fci_matrix(fci_strings, soei, atei, n_occ, n_virt, repulsion)
    vals, vecs = diagonalize_fci_matrix_with_spin(fci_matrix, fci_strings, n_occ, n_virt, k=5) 
    # Write FCI matrix to file
    #print(f"\nWriting FCI matrix to {output_file}...")
    #with open(output_file, 'w') as f:
    #    dim = fci_matrix.shape[0]
    #    for i in range(dim):
    #        for j in range(dim):
    #           # if abs(fci_matrix[i, j]) > 1e-12:  # Only write non-zero elements
    #             f.write(f"{i+1} {j+1} {fci_matrix[i,j]:.12e}\n")
    
    # Compute FCI energy
    print("\nDiagonalizing FCI matrix...")
    eigenvalues = np.linalg.eigvalsh(fci_matrix)
    fci_energy = eigenvalues[0]
    
    print(f"\nFCI Ground State Energy: {fci_energy:.10f}")
    print(f"Number of eigenvalues computed: {len(eigenvalues)}")
    
    return fci_energy, fci_matrix


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fci_from_files.py <input_directory> [output_file]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "FCI_matrix.dat"
    
    main(input_dir, output_file)
