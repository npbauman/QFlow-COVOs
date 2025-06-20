#!/usr/bin/env python
# coding: utf-8
# Full FCI Solver (Based on fci_string.F logic)

import cProfile
import numpy as np
from scipy.sparse.linalg import eigsh
from itertools import combinations

# ----------------------------
# Step 1: Read Nuclear Repulsion Energy
# ----------------------------
with open("ion_ion.dat", "r") as repulsion_file:
    repulsion_energy = float(repulsion_file.readline().strip())
# ----------------------------
# Step 2: One electron Integral
# ----------------------------
def read_one_electron_integrals(filename):
     # Read one-electron integrals from a file and return the spinorbital matrix h.
    with open(filename, 'r') as f:
        # Read header: number of occupied, virtual orbitals, and # of integrals
        noccp, nvirt, nh1 = map(int, f.readline().split())
        norb = noccp + nvirt

        # Initialize spatial integral matrix
        ho = np.zeros((norb, norb))

        # Read integrals and symmetrize
        for _ in range(nh1):
            p, q, val = f.readline().split()
            p = int(p) - 1  # Fortran to Python indexing (0-based)
            q = int(q) - 1
            val = float(val)
            ho[p, q] = val
            ho[q, p] = val  # Hermitian symmetry

    # Total spin orbitals (alpha + beta)
    nspin = 2 * norb
    h = np.zeros((nspin, nspin))

    # Expand to spinorbital form
    for p in range(norb):
        for q in range(norb):
            # Alpha indices
            pa = p
            qa = q

            # Beta indices
            pb = p + norb
            qb = q + norb

            # αα block
            h[pa, qa] = ho[p, q]

            # ββ block
            h[pb, qb] = ho[p, q]

            # αβ and βα blocks are zero (implicitly already)

    return h, noccp, nvirt, norb
h_matrix, noccp, nvirt, norb = read_one_electron_integrals("one_electron_integrals.dat")
#print(h_matrix.shape)  # Should be (2*norb, 2*norb)
#print(noccp, nvirt, norb)

# ----------------------------
# Step 3:Two electrons Coulomb and Exchange Integrals
# ----------------------------
def read_two_electron_integrals(filename, norb):
    #Reads two-electron integrals and returns a 4D tensor.
    tensor = np.zeros((norb, norb, norb, norb))

    with open(filename, 'r') as f:
        noccp, nvirt, nv2 = map(int, f.readline().split())

        for _ in range(nv2):
            p, q, r, s, val = f.readline().split()
            p, q, r, s = int(p)-1, int(q)-1, int(r)-1, int(s)-1
            val = float(val)

            perms = [
                (p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)
            ]
            for a, b, c, d in perms:
                tensor[a, b, c, d] = val  

    return tensor

vc = read_two_electron_integrals("two_electron_integrals_coulomb.dat", norb)
vx = read_two_electron_integrals("two_electron_integrals_exchange.dat", norb)
#print(vc.shape, vx.shape)
#print(norb)

# ----------------------------
# Step 4: Generate half alpha and beta determinants (integers)
# ----------------------------

def generate_half_ints(norb, n_elec):
    """Generates sorted integer representations of half-determinants."""
    ints = []
    for occ in combinations(range(norb), n_elec):
        val = 0
        for i in occ:
            # Bits are ordered from left to right (most to least significant)
            val |= (1 << (norb - 1 - i))
        ints.append(val)
    # Sorting is important for deterministic behavior, matching Fortran if needed.
    return np.sort(np.array(ints, dtype=np.int64))

half_ints_a = generate_half_ints(norb, noccp)  # alpha determinants
half_ints_b = generate_half_ints(norb, noccp)  # beta determinants

# ----------------------------
# Step 5: Construct Full FCI Determinants (integers)
# ----------------------------

def build_fci_ints(half_ints_a, half_ints_b, noas, nvas, nobs, nvbs):
    """
    Builds full FCI determinant list (as integers), combining alpha and beta half-determinants.
    The layout is (alpha occ | beta occ | alpha virt | beta virt).
    """
    fci_ints = []
    nstot = noas + nobs + nvas + nvbs
    
    for ha in half_ints_a:
        for hb in half_ints_b:
            # Extract occupied and virtual parts from half-determinants
            alpha_occ_part = (ha >> nvas)
            alpha_virt_part = ha & ((1 << nvas) - 1)
            beta_occ_part = (hb >> nvbs)
            beta_virt_part = hb & ((1 << nvbs) - 1)

            # Assemble the full determinant integer
            fci_int = (alpha_occ_part << (nobs + nvas + nvbs)) | \
                      (beta_occ_part << (nvas + nvbs)) | \
                      (alpha_virt_part << nvbs) | \
                      beta_virt_part
            
            # Safety check for correct electron number
            if bin(fci_int).count('1') != (noas + nobs):
                 raise ValueError("Mismatch in total electron count during FCI determinant construction!")

            fci_ints.append(fci_int)
            
    return np.array(fci_ints, dtype=np.int64)

noas = nobs = noccp
nvas = nvbs = nvirt
fci_ints = build_fci_ints(half_ints_a, half_ints_b, noas, nvas, nobs, nvbs)

# ----------------------------
# Step 6: Shared Uitilities Tools
# ----------------------------

def get_excitation_info_from_ints(int1, int2, nbits):
    """
    Calculates the number of differences and the creation/annihilation indices
    between two integer bitstrings.
    """
    xor = int1 ^ int2
    # Each excitation (particle-hole pair) flips two bits.
    idiff = bin(xor).count('1') // 2

    plus = []
    minus = []
    for i in range(nbits):
        # Check if the i-th bit (from left, 0-indexed) is set in the XOR result
        if (xor >> (nbits - 1 - i)) & 1:
            # If the bit is set, it's a difference. Now check direction.
            if (int1 >> (nbits - 1 - i)) & 1:
                # Bit was in int1, so it's an annihilation (minus)
                minus.append(i)
            else:
                # Bit was not in int1 (so it's in int2), it's a creation (plus)
                plus.append(i)
    return idiff, plus, minus

def compute_double_iphase_from_int(p, q, r, s, int_string, nbits):
    """Computes fermionic phase for a double excitation c_p^+ c_q^+ c_s c_r."""
    def popcount_before(integer, k):
        if k == 0: return 0
        # Mask to count set bits in positions 0 to k-1
        mask = ~((1 << (nbits - k)) - 1)
        return bin(integer & mask).count('1')

    # p, q are creation; r, s are annihilation. Phase is calculated from initial state.
    isum = popcount_before(int_string, r)
    int_string &= ~(1 << (nbits - 1 - r))
    
    isum += popcount_before(int_string, s)
    int_string &= ~(1 << (nbits - 1 - s))

    isum += popcount_before(int_string, q)
    int_string |= (1 << (nbits - 1 - q))

    isum += popcount_before(int_string, p)
    
    return (-1)**isum

def compute_single_iphase_from_int(p, q, int_string, nbits):
    """Computes fermionic phase for a single excitation c_p^+ c_q."""
    def popcount_before(integer, k):
        if k == 0: return 0
        mask = ~((1 << (nbits - k)) - 1)
        return bin(integer & mask).count('1')

    # p is creation, q is annihilation. Phase from initial state.
    isum = popcount_before(int_string, q)
    int_string &= ~(1 << (nbits - 1 - q))
    isum += popcount_before(int_string, p)
    
    return (-1)**isum

def map_spinorbital_to_spatial_and_spin(k, noccp, nvirt):
    thres1 = noccp
    thres2 = 2 * noccp
    thres3 = 2 * noccp + nvirt

    if k < thres1:  # α occupied
        spatial = k
        spin = 0
    elif k < thres2:  # β occupied
        spatial = k - noccp
        spin = 1
    elif k < thres3:  # α virtual
        spatial = k - noccp
        spin = 0
    else:  # β virtual
        spatial = k - noccp - nvirt
        spin = 1

    return spatial, spin


# ----------------------------
# Step 7: Matrix Construction Functions
# ----------------------------

def compute_double_excitation_element(int1, int2, i, j, vx, noccp, nvirt, matrix, plus, minus, spinorbital_map, nbits):
    p, q = sorted(plus)
    r, s = sorted(minus)

    iphase = compute_double_iphase_from_int(p, q, r, s, int1, nbits)

    # Verify that the excitation is correct by re-applying the bit flips
    if int1 ^ (1 << (nbits - 1 - p)) ^ (1 << (nbits - 1 - q)) ^ (1 << (nbits - 1 - r)) ^ (1 << (nbits - 1 - s)) != int2:
        return

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

    matrix[i, j] += iphase * val

def compute_single_excitation_element(int1, int2, i, j, h, vc, vx, noas, nobs, matrix, plus, minus, spinorbital_map, nbits, ind_set):
    p = plus[0]
    q = minus[0]

    # Fermionic phase
    iphase = compute_single_iphase_from_int(p, q, int1, nbits)

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
            # This case seems complex and might need verification against theory
            val = 0.25 * vc[w, n, w, n] - 0.5 * vx[w, n, w, n]
        else:
            val = 0.0
        two_e += val

    total = iphase * (one_e + two_e)
    matrix[i, j] += total

def handle_diagonal_element(int_string, i, h, vc, vx, noccp, nvirt, repulsion, matrix, spinorbital_map, nbits):
    """
    Compute ⟨D_i | H | D_i⟩: the diagonal matrix element for determinant i.
    """
    ind_set = []
    for l in range(nbits):
        if (int_string >> (nbits - 1 - l)) & 1:
            ind_set.append(l)

    if len(ind_set) != (noas + nobs):
        raise ValueError(f"Occupancy mismatch: expected {noas + nobs}, got {len(ind_set)}")

    diag_val = 0.0

    # One-electron contribution
    for p in ind_set:
        p_h, _ = spinorbital_map[p]
        diag_val += h[p_h, p_h]

    # Two-electron contribution
    for p in ind_set:
        for q in ind_set:
            m, ps = spinorbital_map[p]
            n, qs = spinorbital_map[q]

            if ps == qs:
                v_pq = 0.5 * vc[m, m, n, n] - vx[n, m, m, n]
            else:
                v_pq = 0.5 * vc[m, m, n, n]

            diag_val +=  0.5 * v_pq

    # Add ion-ion repulsion
    diag_val += repulsion

    matrix[i, i] += diag_val

# ----------------------------
# Step 8: Build FCI Matrix
# ----------------------------
def generate_single_double_excitations(int_string, n_occ, n):
    """
    Generate all unique single and double excitations from a given integer determinant.
    Returns a list of new integer determinants.
    """
    occ_indices = []
    virt_indices = []
    for i in range(n):
        if (int_string >> (n - 1 - i)) & 1:
            occ_indices.append(i)
        else:
            virt_indices.append(i)
    
    excitations = []

    # Single excitations
    for i in occ_indices:
        for a in virt_indices:
            new_int = int_string ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - a))
            excitations.append(new_int)

    # Double excitations
    for i1_idx in range(len(occ_indices)):
        for i2_idx in range(i1_idx + 1, len(occ_indices)):
            for a1_idx in range(len(virt_indices)):
                for a2_idx in range(a1_idx + 1, len(virt_indices)):
                    i1 = occ_indices[i1_idx]
                    i2 = occ_indices[i2_idx]
                    a1 = virt_indices[a1_idx]
                    a2 = virt_indices[a2_idx]
                    
                    new_int = int_string ^ (1 << (n - 1 - i1)) ^ (1 << (n - 1 - i2)) \
                                       ^ (1 << (n - 1 - a1)) ^ (1 << (n - 1 - a2))
                    excitations.append(new_int)

    return excitations

def build_fci_matrix(fci_ints, h, vc, vx, noccp, nvirt, noas, nobs, repulsion):
    """Iterates over determinant pairs and fills the full Hamiltonian matrix."""
    dim_fci = fci_ints.shape[0]
    matrix = np.zeros((dim_fci, dim_fci))
    nstot = noas + nobs + nvas + nvbs

    spinorbital_map = [
        map_spinorbital_to_spatial_and_spin(k, noccp, nvirt)
        for k in range(2 * (noccp + nvirt))
    ]

    # Build a mapping from integer determinant to its index in the FCI vector
    int_to_index = {s: idx for idx, s in enumerate(fci_ints)}

    # Pre-compute occupied indices for each determinant to avoid re-calculation
    int_to_ind_set = {
        s: [(t, *spinorbital_map[t]) for t in range(nstot) if (s >> (nstot - 1 - t)) & 1]
        for s in fci_ints
    }

    # Diagonal terms
    for i in range(dim_fci):
        handle_diagonal_element(fci_ints[i], i, h, vc, vx, noccp, nvirt, repulsion, matrix, spinorbital_map, nstot)

    # Off-diagonal terms
    for i in range(dim_fci):
        int1 = fci_ints[i]
        excitations_int = generate_single_double_excitations(int1, noas + nobs, nstot)
        
        for int2 in excitations_int:
            j = int_to_index.get(int2)
            # Only compute for upper triangle (j > i)
            if j is not None and j > i:
                idiff, plus, minus = get_excitation_info_from_ints(int1, int2, nstot)
                if idiff == 1:
                    compute_single_excitation_element(int1, int2, i, j, h, vc, vx, noas, nobs, matrix, plus, minus, spinorbital_map, nstot, int_to_ind_set[int2])
                elif idiff == 2:
                    compute_double_excitation_element(int1, int2, i, j, vx, noccp, nvirt, matrix, plus, minus, spinorbital_map, nstot)

    # Symmetrize the matrix since we only filled the upper triangle
    matrix += matrix.T - np.diag(matrix.diagonal())

    return matrix

# ----------------------------
# Step 9: Diagonalize FCI Matrix
# ----------------------------

def diagonalize_fci_matrix(matrix):
    """Diagonalizes the FCI matrix to find the lowest eigenvalue."""
    eigvals, eigvecs = eigsh(matrix, k=1, which='SA')
    lowest_energy = eigvals[0]
    print("Lowest FCI energy:", f"{lowest_energy:.10f}")
    return lowest_energy, eigvals, eigvecs

# ----------------------------
# Step 10: Run Full Solver
# ----------------------------
def main() -> None:
    """Main execution function."""
    fci_matrix = build_fci_matrix(fci_ints, h_matrix, vc, vx, noccp, nvirt, noas, nobs, repulsion_energy)
    diagonalize_fci_matrix(fci_matrix)

if __name__ == '__main__':
    cProfile.run('main()', sort='ncalls')
