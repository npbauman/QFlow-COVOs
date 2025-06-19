#!/usr/bin/env python
# coding: utf-8
# Full FCI Solver (Based on fci_string.F logic, refactored with bitmasks)

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
        noccp, nvirt, nh1 = map(int, f.readline().split())
        norb = noccp + nvirt
        ho = np.zeros((norb, norb))
        for _ in range(nh1):
            p, q, val = f.readline().split()
            p, q, val = int(p) - 1, int(q) - 1, float(val)
            ho[p, q] = val
            ho[q, p] = val
    nspin = 2 * norb
    h = np.zeros((nspin, nspin))
    for p in range(norb):
        for q in range(norb):
            h[p, q] = ho[p, q]
            h[p + norb, q + norb] = ho[p, q]
    return h, noccp, nvirt, norb
h_matrix, noccp, nvirt, norb = read_one_electron_integrals("one_electron_integrals.dat")

# ----------------------------
# Step 3:Two electrons Coulomb and Exchange Integrals
# ----------------------------
def read_two_electron_integrals(filename, norb):
    # Reads two-electron integrals and returns a 4D tensor.
    tensor = np.zeros((norb, norb, norb, norb))
    with open(filename, 'r') as f:
        noccp, nvirt, nv2 = map(int, f.readline().split())
        for _ in range(nv2):
            p, q, r, s, val = f.readline().split()
            p, q, r, s, val = int(p)-1, int(q)-1, int(r)-1, int(s)-1, float(val)
            perms = [
                (p, q, r, s), (q, p, r, s), (p, q, s, r), (q, p, s, r),
                (r, s, p, q), (s, r, p, q), (r, s, q, p), (s, r, q, p)
            ]
            for a, b, c, d in perms:
                tensor[a, b, c, d] = val
    return tensor
vc = read_two_electron_integrals("two_electron_integrals_coulomb.dat", norb)
vx = read_two_electron_integrals("two_electron_integrals_exchange.dat", norb)

# ----------------------------
# Step 4: Generate half alpha and beta bitmasks
# ----------------------------
def generate_half_bitmasks(norb, n_elec):
    """Generates integer bitmasks for n_elec electrons in norb orbitals."""
    bitmasks = []
    for occ in combinations(range(norb), n_elec):
        mask = 0
        for i in occ:
            mask |= (1 << i)
        bitmasks.append(mask)
    return np.array(bitmasks, dtype=np.uint64)

half_mask_a = np.sort(generate_half_bitmasks(norb, noccp))
half_mask_b = np.sort(generate_half_bitmasks(norb, noccp))

# ----------------------------
# Step 5: Construct Full Determinant Bitmasks
# ----------------------------
def build_fci_bitmasks(half_a, half_b, noas, nvas, nobs, nvbs):
    """
    Builds full FCI determinant bitmasks, combining alpha and beta half-masks
    into the specific Fortran memory layout.
    """
    norb = noas + nvas
    nos = noas + nobs
    full_masks = []
    for alpha_mask in half_a:
        alpha_mask = int(alpha_mask)  # Ensure Python int for bitwise ops
        for beta_mask in half_b:
            beta_mask = int(beta_mask)  # Ensure Python int for bitwise ops
            full_mask = 0
            # Map alpha electrons
            for i in range(norb):
                if (alpha_mask >> i) & 1:
                    if i < noas: # Occupied alpha
                        full_mask |= (1 << i)
                    else: # Virtual alpha
                        full_mask |= (1 << (nos + (i - noas)))
            # Map beta electrons
            for i in range(norb):
                if (beta_mask >> i) & 1:
                    if i < nobs: # Occupied beta
                        full_mask |= (1 << (noas + i))
                    else: # Virtual beta
                        full_mask |= (1 << (nos + nvas + (i - nobs)))
            full_masks.append(full_mask)
    return np.array(full_masks, dtype=np.uint64)

noas = nobs = noccp
nvas = nvbs = nvirt
fci_bitmasks = build_fci_bitmasks(half_mask_a, half_mask_b, noas, nvas, nobs, nvbs)

# ----------------------------
# Step 6: Shared Utilities Tools
# ----------------------------
def count_set_bits(n):
    return bin(n).count('1')

def map_spinorbital_to_spatial_and_spin(k, noccp, nvirt):
    # This mapping is specific to the unusual Fortran layout
    thres1 = noccp
    thres2 = 2 * noccp
    thres3 = 2 * noccp + nvirt
    if k < thres1: return k, 0
    if k < thres2: return k - noccp, 1
    if k < thres3: return k - noccp, 0
    return k - noccp - nvirt, 1

def compute_iphase_bitwise(p, q, r, s, mask):
    """Computes phase for a double excitation using bitwise operations."""
    isum = 0
    isum += count_set_bits(mask & ((1 << r) - 1)); mask ^= (1 << r)
    isum += count_set_bits(mask & ((1 << s) - 1)); mask ^= (1 << s)
    isum += count_set_bits(mask & ((1 << q) - 1)); mask |= (1 << q)
    isum += count_set_bits(mask & ((1 << p) - 1))
    return (-1) ** isum

# ----------------------------
# Step 7: Matrix Construction Functions
# ----------------------------
def compute_double_excitation_element(mask1, mask2, i, j, vx, matrix, spinorbital_map):
    diff_mask = mask1 ^ mask2
    p_mask = mask2 & diff_mask
    m_mask = mask1 & diff_mask

    # Optimized bit manipulation to find the two set bits
    p1 = (p_mask & -p_mask).bit_length() - 1
    p2 = (p_mask ^ (1 << p1)).bit_length() - 1

    m1 = (m_mask & -m_mask).bit_length() - 1
    m2 = (m_mask ^ (1 << m1)).bit_length() - 1

    plus = [p1, p2]
    minus = [m1, m2]

    p, q = sorted(plus)
    r, s = sorted(minus)

    iphase = compute_iphase_bitwise(p, q, r, s, mask1)

    m, ps = spinorbital_map[p]
    n, qs = spinorbital_map[q]
    u, rs = spinorbital_map[r]
    w, ss = spinorbital_map[s]

    val = 0.0
    if ps == rs and qs == ss and ps != qs: val = vx[m, u, n, w]
    elif ps == ss and qs == rs and ps != qs: val = -vx[m, w, n, u]
    elif ps == rs and qs == ss and ps == qs: val = vx[m, u, n, w] - vx[m, w, n, u]

    matrix[i, j] += iphase * val
    matrix[j, i] += iphase * val

def compute_single_excitation_element(mask1, mask2, i, j, h, vc, vx, matrix, spinorbital_map):
    diff_mask = mask1 ^ mask2
    p_mask = diff_mask & mask2
    q_mask = diff_mask & mask1
    p = p_mask.bit_length() - 1
    q = q_mask.bit_length() - 1
    
    ind_set = [(t, *spinorbital_map[t]) for t in range(64) if (mask2 >> t) & 1]

    # Fermionic phase calculation on the final state (mask2)
    isum = count_set_bits(mask2 & ((1 << q) - 1))
    isum += count_set_bits((mask2 ^ (1 << q)) & ((1 << p) - 1))
    iphase = (-1) ** isum

    p_h, _ = spinorbital_map[p]
    q_h, _ = spinorbital_map[q]
    one_e = h[p_h, q_h]

    w, ps = spinorbital_map[p]
    n, qs = spinorbital_map[q]

    two_e = 0.0
    for t, u, ts in ind_set:
        if ps == ts and p != t and q != t: val = 0.5 * vc[u, u, n, w] - vx[n, u, w, u]
        elif ps != ts: val = 0.5 * vc[u, u, n, w]
        elif p == t or q == t: val = 0.25 * vc[w, n, w, n] - 0.5 * vx[w, n, w, n]
        else: val = 0.0
        two_e += val

    total = iphase * (one_e + two_e)
    matrix[i, j] += total
    matrix[j, i] += total

def handle_diagonal_element(mask, i, h, vc, vx, repulsion, matrix, spinorbital_map):
    mask = int(mask)  # Ensure Python int for bitwise ops
    ind_set = [l for l in range(64) if (mask >> l) & 1]
    diag_val = 0.0
    # One-electron contribution
    for p in ind_set:
        p_h, _ = spinorbital_map[p]
        diag_val += h[p_h, p_h]
    # Two-electron contribution (logic from FCI-Code.py)
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
    matrix[i, i] = diag_val

# ----------------------------
# Step 8: Build FCI Matrix
# ----------------------------
def generate_single_double_excitation_masks(mask, n_spin_orb, n_elec):
    mask = int(mask)  # Ensure Python int for bitwise ops
    occ_indices = [i for i, x in enumerate(range(n_spin_orb)) if (mask >> i) & 1]
    virt_indices = [i for i, x in enumerate(range(n_spin_orb)) if not ((mask >> i) & 1)]
    excitations = []
    # Singles
    for i in occ_indices:
        for a in virt_indices:
            excitations.append(mask ^ (1 << i) ^ (1 << a))
    # Doubles
    for i1_idx, i1 in enumerate(occ_indices):
        for i2 in occ_indices[i1_idx + 1:]:
            for a1_idx, a1 in enumerate(virt_indices):
                for a2 in virt_indices[a1_idx + 1:]:
                    excitations.append(mask ^ (1 << i1) ^ (1 << i2) ^ (1 << a1) ^ (1 << a2))
    return excitations

def build_fci_matrix(fci_masks, h, vc, vx, noccp, nvirt, repulsion):
    dim_fci = len(fci_masks)
    matrix = np.zeros((dim_fci, dim_fci))
    n_spin_orb = 2 * (noccp + nvirt)
    n_elec = 2 * noccp

    spinorbital_map = [map_spinorbital_to_spatial_and_spin(k, noccp, nvirt) for k in range(n_spin_orb)]
    mask_to_index = {int(mask): idx for idx, mask in enumerate(fci_masks)}  # Ensure keys are int

    for i in range(dim_fci):
        mask1 = int(fci_masks[i])  # Ensure Python int for bitwise ops
        handle_diagonal_element(mask1, i, h, vc, vx, repulsion, matrix, spinorbital_map)
        
        excitations = generate_single_double_excitation_masks(mask1, n_spin_orb, n_elec)
        for ex_mask in excitations:
            j = mask_to_index.get(int(ex_mask))  # Ensure lookup with int
            if j is not None and j > i:
                diff_mask = mask1 ^ int(ex_mask)
                idiff = count_set_bits(diff_mask) // 2
                if idiff == 1:
                    compute_single_excitation_element(mask1, int(ex_mask), i, j, h, vc, vx, matrix, spinorbital_map)
                elif idiff == 2:
                    compute_double_excitation_element(mask1, int(ex_mask), i, j, vx, matrix, spinorbital_map)
    return matrix

# ----------------------------
# Step 9: Diagonalize FCI Matrix
# ----------------------------
def diagonalize_fci_matrix(matrix):
    eigvals, eigvecs = eigsh(matrix, k=1, which='SA')
    lowest_energy = eigvals[0]
    print("Lowest FCI energy:", f"{lowest_energy:.10f}")
    return lowest_energy, eigvals, eigvecs

# ----------------------------
# Step 10: Run Full Solver
# ----------------------------
def main() -> None:
    fci_matrix = build_fci_matrix(fci_bitmasks, h_matrix, vc, vx, noccp, nvirt, repulsion_energy)
    diagonalize_fci_matrix(fci_matrix)

    # filename="FCI_matrix.dat"
    # dim = fci_matrix.shape[0]
    # with open(filename, "w") as f:
    #     for i in range(dim):
    #         for j in range(dim):
    #             if abs(fci_matrix[i, j]) > 1e-10:
    #                 f.write(f"{i:5d} {j:5d} {fci_matrix[i, j]:20.12f}\n")

if __name__ == '__main__':
    cProfile.run('main()', sort='tottime')