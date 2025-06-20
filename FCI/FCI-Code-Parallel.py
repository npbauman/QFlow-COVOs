#!/usr/bin/env python
# coding: utf-8
# Full FCI Solver (Based on fci_string.F logic)

import cProfile
import numpy as np
from scipy.sparse.linalg import eigsh
from itertools import combinations
import multiprocessing as mp
from multiprocessing import shared_memory

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
# Step 4:Generate half alpha and beta strings
# ----------------------------

def generate_half_strings(norb, n_elec):
    strings = []
    for occ in combinations(range(norb), n_elec):
        bitstring = [0] * norb
        for i in occ:
            bitstring[i] = 1
        strings.append(bitstring)
    return np.array(strings, dtype=int)
half_str_a = generate_half_strings(norb, noccp)  # alpha strings
half_str_b = generate_half_strings(norb, noccp)  # beta strings

# change the order just like fortran 
half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]

#print(half_str_a.shape)
#print(half_str_b.shape)

# ----------------------------
# Step 5:Construct Full Determinants (FCI Strings)
# ----------------------------

def build_fci_strings_from_half_strings(half_str_a, half_str_b, noas, nvas, nobs, nvbs):
    """
    Builds full FCI determinant list (strings), combining alpha and beta half strings,
    laid out in the specific way as the Fortran code (alpha occ.| beta occ.| alpha virt. | beta virt.).
    """
    alpha_str = half_str_a.shape[0]
    beta_str  = half_str_b.shape[0]
    
    nos = noas + nobs
    nvs = nvas + nvbs
    nstot = nos + nvs

    strings = []

    for i in range(alpha_str):
        for j in range(beta_str):
            string = [0] * nstot

            # Alpha string: occupied + virtual
            string[0:noas] = half_str_a[i, 0:noas]
            string[nos : nos + nvas] = half_str_a[i, noas : noas + nvas]

            # Beta string: occupied + virtual
            string[noas : noas + nobs] = half_str_b[j, 0:nobs]
            string[nos + nvas : nos + nvas + nvbs] = half_str_b[j, nobs : nobs + nvbs]

            # Safety check
            if sum(string) != (noas + nobs):
                raise ValueError("Mismatch in total electron count!")

            strings.append(string)

    return np.array(strings, dtype=int)
noas = nobs = noccp
nvas = nvbs = nvirt
fci_strings = build_fci_strings_from_half_strings( half_str_a, half_str_b, noas, nvas, nobs, nvbs)
#print(fci_strings.shape)
#print(noas, nvas, nobs, nvbs)
#print(fci_strings[0])
#print(fci_strings[127])

# ----------------------------
# Step 6: Shared Uitilities Tools
# ----------------------------

def get_idiff_and_diff(string1, string2):
    diff = string1 - string2
    idiff = int(np.sum(np.abs(diff)) // 2)
    return idiff, diff

def get_excitation_indices(diff):
    plus = [i for i, d in enumerate(diff) if d == -1]
    minus = [i for i, d in enumerate(diff) if d == 1]
    return plus, minus

def map_spinorbital_to_spatial_and_spin(k, noccp, nvirt):
    thres1 = noccp
    thres2 = 2 * noccp
    thres3 = 2 * noccp + nvirt

    if k <= thres1 - 1:  # k < noccp → α occupied
        spatial = k
        spin = 0
    elif k <= thres2 - 1:  # k in [noccp, 2*noccp-1] → β occupied
        spatial = k - noccp
        spin = 1
    elif k <= thres3 - 1:  # k in [2*noccp, 2*noccp+nvirt-1] → α virtual
        spatial = k - noccp
        spin = 0
    else:  # β virtual
        spatial = k - noccp - nvirt
        spin = 1

    return spatial, spin


def compute_iphase(p, q, r, s, string):
    tmp = string.copy()
    isum = np.sum(tmp[:r]); tmp[r] = 0
    isum += np.sum(tmp[:s]); tmp[s] = 0
    isum += np.sum(tmp[:q]); tmp[q] = 1
    isum += np.sum(tmp[:p]); tmp[p] = 1
    return (-1) ** isum

# ----------------------------
# Step 7: Matrix Construction Functions
# ----------------------------

def compute_double_excitation_element(string1, string2, i, j, vx, noccp, nvirt, matrix, diff, spinorbital_map):
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 2 or len(minus) != 2:
        return

    p, q = sorted(plus)
    r, s = sorted(minus)

    iphase = compute_iphase(p, q, r, s, string1)

    test_string = string1.copy()
    test_string[r] = 0
    test_string[s] = 0
    test_string[p] = 1
    test_string[q] = 1
    if not np.array_equal(test_string, string2):
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

    return iphase * val

def compute_single_excitation_element(strings, i, j, h, vc, vx, noccp, nvirt, matrix, diff, spinorbital_map):
    plus, minus = get_excitation_indices(diff)
    if len(plus) != 1 or len(minus) != 1:
        return None

    p = plus[0]
    q = minus[0]

    string2 = strings[j]
    ind_set = [(t, *spinorbital_map[t])
               for t in range(len(string2)) if string2[t] == 1]

    if len(ind_set) != (noas + nobs):
        raise ValueError("Occupied orbital count mismatch.")

    # Fermionic phase
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

    total = iphase * (one_e + two_e)
    return total


def handle_diagonal_element(strings, i, h, vc, vx, noccp, nvirt, repulsion, matrix, spinorbital_map):
    """
    Compute ⟨D_i | H | D_i⟩: the diagonal matrix element for determinant i.
    Includes:
      - One-electron contribution: sum of h(p, p)
      - Two-electron contribution: 1/2 sum_pq [J - K] or [J] depending on spin
      - Nuclear repulsion energy
    """
    string = strings[i]
    nstot = len(string)
    ind_set = [l for l in range(nstot) if string[l] == 1]

    if len(ind_set) != (noas + nobs):
        raise ValueError(f"Occupancy mismatch: expected {noas + nobs}, got {len(ind_set)}")

    diag_val = 0.0

    # One-electron contribution
    for p in ind_set:
        p_h, spin = spinorbital_map[p]
        diag_val += h[p_h, p_h]
        #print("p =", p, "→ h index =", p_h, "spin =", spin, "h[p,p] =", h[p_h, p_h], "running diag_val =", diag_val)


    # Two-electron contribution
    for p in ind_set:
        for q in ind_set:
            m, ps = spinorbital_map[p]
            n, qs = spinorbital_map[q]


            if ps == qs:
                v_pq = 0.5 * vc[m, m, n, n] - vx[n, m, m, n]
            else:
                v_pq = 0.5 * vc[m, m, n, n]

            diag_val +=  0.5 * v_pq  # Symmetry factor
            
    # Add ion-ion repulsion
    diag_val += repulsion

    matrix[i, i] += diag_val

# ----------------------------
# Step 8: Build FCI Matrix
# ----------------------------
# Iterates over determinant pairs and fills the full Hamiltonian
def generate_single_double_excitations(string, n_occ):
    """
    Generate all unique single and double excitations from a given string.
    Returns a list of new strings (as tuples).
    """
    n = len(string)
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

# These will be set in the parent process before parallel launch
_global_data = {}

def init_worker(fci_strings_, string_to_index_, h_, vc_, vx_,
                noccp_, nvirt_, noas_, nobs_, shm_name_, matrix_shape_, spinorbital_map_):
    global _global_data
    _global_data['fci_strings'] = fci_strings_
    _global_data['string_to_index'] = string_to_index_
    _global_data['h'] = h_
    _global_data['vc'] = vc_
    _global_data['vx'] = vx_
    _global_data['noccp'] = noccp_
    _global_data['nvirt'] = nvirt_
    _global_data['noas'] = noas_
    _global_data['nobs'] = nobs_
    _global_data['shm_name'] = shm_name_
    _global_data['matrix_shape'] = matrix_shape_
    _global_data['spinorbital_map'] = spinorbital_map_


def off_diag_worker(i):
    gd = _global_data  # shortcut

    try:
        existing_shm = shared_memory.SharedMemory(name=gd['shm_name'])
        matrix = np.ndarray(gd['matrix_shape'], dtype=np.float64, buffer=existing_shm.buf)
    except Exception as e:
        print(f"[Worker {i}] Shared memory error: {e}")
        return

    fci_strings = gd['fci_strings']
    string_to_index = gd['string_to_index']
    h = gd['h']
    vc = gd['vc']
    vx = gd['vx']
    noccp = gd['noccp']
    nvirt = gd['nvirt']
    noas = gd['noas']
    nobs = gd['nobs']
    spinorbital_map = gd['spinorbital_map']

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
                    fci_strings, i, j, h, vc, vx, noccp, nvirt, None, diff, spinorbital_map
                )
            elif idiff == 2:
                val = compute_double_excitation_element(
                    string1, string2, i, j, vx, noccp, nvirt, None, diff, spinorbital_map
                )

            if val is not None:
                matrix[i, j] += val
                matrix[j, i] += val

    existing_shm.close()


def build_fci_matrix(fci_strings, h, vc, vx, noccp, nvirt, noas, nobs, repulsion):

    dim_fci = fci_strings.shape[0]
    matrix_shape = (dim_fci, dim_fci)
    shm = shared_memory.SharedMemory(create=True, size=np.prod(matrix_shape) * 8)
    matrix = np.ndarray(matrix_shape, dtype=np.float64, buffer=shm.buf)
    matrix[:] = 0.0

    spinorbital_map = [
        map_spinorbital_to_spatial_and_spin(k, noccp, nvirt)
        for k in range(2 * (noccp + nvirt))
    ]
    string_to_index = {tuple(s): idx for idx, s in enumerate(fci_strings)}

    # Diagonal terms
    for i in range(dim_fci):
        handle_diagonal_element(fci_strings, i, h, vc, vx, noccp, nvirt, repulsion, matrix, spinorbital_map)

    # Off-diagonal terms
    # Start worker pool with shared initialization
    with mp.Pool(
        initializer=init_worker,
        initargs=(fci_strings, string_to_index, h, vc, vx,
                  noccp, nvirt, noas, nobs, shm.name, matrix_shape, spinorbital_map)
    ) as pool:
        pool.map(off_diag_worker, range(dim_fci))

    final_matrix = matrix.copy()
    shm.close()
    shm.unlink()
    return final_matrix


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
    fci_matrix = build_fci_matrix(fci_strings, h_matrix, vc, vx, noccp, nvirt, noas, nobs, repulsion_energy)
    diagonalize_fci_matrix(fci_matrix)

if __name__ == '__main__':
    cProfile.run('main()', sort='ncalls')

# # ----------------------------
# # Step 11: Write FCI Matrix to File (Fortran 1-based indexing)
# # ----------------------------
# def write_fci_matrix_to_file_fortran_index(matrix, filename="FCI_matrix.dat"):
#     dim = matrix.shape[0]
#     with open(filename, "w") as f:
#         for i in range(dim):
#             for j in range(dim):
#                 f.write(f"{i+1:5d} {j+1:5d} {matrix[i, j]:20.12f})")
                
# write_fci_matrix_to_file_fortran_index(fci_matrix, "FCI_matrix.dat")

# # Write only the diagonal elements

# def write_fci_diagonal_to_file_fortran_index(matrix, filename="FCI_matrix_diagonal.dat"):
#     with open(filename, "w") as f:
#         for i in range(matrix.shape[0]):
#             f.write(f"{i+1:5d} {i+1:5d} {matrix[i, i]:20.12f}")

# write_fci_diagonal_to_file_fortran_index(fci_matrix, "FCI_matrix_diagonal.dat")

