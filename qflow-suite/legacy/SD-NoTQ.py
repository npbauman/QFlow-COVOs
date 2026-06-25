# %%
import numpy as np
from itertools import combinations
from numba import njit
from numpy.linalg import eigvals
from functools import lru_cache
from numpy.linalg import norm
import os, json, time, tempfile

def setup_qflow_inputs(n_elec: int, norb: int, assume_closed_shell=True):
    """
    Initialize orbital partitioning for QFlow based on a closed-shell assumption.

    Parameters:
    - n_elec : int : Total number of electrons (must be even for closed-shell)
    - norb   : int : Number of spatial orbitals
    - assume_closed_shell : bool : If True, assigns equal alpha/beta occupation

    Returns:
    - dict with:
        - nos   : total occupied spin orbitals
        - nvs   : total virtual spin orbitals
        - nstot : total spin orbitals
        - noas  : alpha occupied spin orbitals
        - nobs  : beta occupied spin orbitals
        - nvas  : alpha virtual spin orbitals
        - nvbs  : beta virtual spin orbitals
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



# %%
params = setup_qflow_inputs(n_elec=6, norb=9)

norb   = params["norb"]
noas = params["noas"]
nobs = params["nobs"]
nvas = params["nvas"]
nvbs = params["nvbs"]
nos  = params["nos"]
nvs  = params["nvs"]
nstot = params["nstot"]
noccp = params['noccp']
nvirt = params['nvirt']



# %%
def generate_half_strings(norb, n_elec):
    string = []
    for occ in combinations(range(norb), n_elec):
        bitstring = [0] * norb
        for i in occ:
            bitstring[i] = 1
        string.append(bitstring)
    return np.array(string, dtype=int)

half_str_a = generate_half_strings(norb, noccp)  # alpha strings
half_str_b = generate_half_strings(norb, noccp)  # beta strings

# change the order 
half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]

#print(half_str_a.shape)
#print(half_str_b.shape)

def build_fci_strings_from_half_strings(half_str_a, half_str_b, noas, nvas, nobs, nvbs):

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
#noas = nobs = noccp
#nvas = nvbs = nvirt
strings = build_fci_strings_from_half_strings( half_str_a, half_str_b, noas, nvas, nobs, nvbs)
#print(fci_strings.shape)
#print(noas, nvas, nobs, nvbs)
#print("Determinant 1:  ",fci_strings[0])
#print("Determinant 2:  ",fci_strings[1])
#print(fci_strings[4])
#print(fci_strings[127])

# %%
def read_fci_matrix(filename, dim_fci):
    """
    Read symmetric matrix from FCI_Matrix.dat (1-based index).
    """
    matrix = np.zeros((dim_fci, dim_fci))
    with open(filename, 'r') as f:
        for line in f:
            i, j, val = line.strip().split()
            i, j = int(i) - 1, int(j) - 1  # convert to 0-based
            val = float(val)
            matrix[i, j] = val
            if i != j:
                matrix[j, i] = val  # fill symmetric part
    return matrix

# Read the matrix
dim_fci = strings.shape[0]
matrix = read_fci_matrix("FCI_matrix.dat", dim_fci)
#matrix = read_fci_matrix("FCI_matrix_FORTRAN.dat",dim_fci)
matrix_h = matrix.copy()

#print(dim_fci)

# %%
import numpy as np
from itertools import combinations
from math import comb

def trial_fun(dim_fci, strings, noas, nobs, nstot):
    trial = np.zeros(dim_fci)
    ref = np.array([1]*(noas + nobs) + [0]*(nstot - noas - nobs))
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

#def build_nactspin(noas, nobs, nvas, nvbs, nstot):
#    occ_pairs = list(combinations(range(noas), 2))
#    virt_pairs = list(combinations(range(nvas), 2))
#    nactives = len(occ_pairs) * len(virt_pairs)
#    nactspin = np.ones((nactives, nstot), dtype=int)
#    actopx = []
#
#    for idx, (oi, oj) in enumerate(occ_pairs):
#        for vi, vj in virt_pairs:
#            actopx.append([oi, oj, vi, vj])
#            a = idx * len(virt_pairs) + virt_pairs.index((vi, vj))
#            for o in [oi, oj]:
#                nactspin[a, o] = nactspin[a, o + noas] = 0
#            for v in [vi, vj]:
#                nactspin[a, noas + nobs + v] = nactspin[a, noas + nobs + nvas + v] = 0
#
#    return nactspin, np.array(actopx)
nactspin = []

nactspin_flat = np.loadtxt("nactspin_fortran.dat", dtype=int)
nactspin = nactspin_flat.reshape(-1, 18)
nactspin = np.array(nactspin, dtype=int)

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
        "lowest_energy": np.min(eigvals)
    }
trial = trial_fun(dim_fci, strings, noas, nobs, nstot)
#nactspin, actopx = build_nactspin(noas, nobs, nvas, nvbs, nstot)
nactives = evaluate_nactives(noas,nvas)

# %%
def generate_spin_array(nos, nvs):
    nstot = nos + nvs
    spin_array = np.zeros(nstot, dtype=int)

    for i in range(nstot):
        if i < nos // 2:
            spin_array[i] = 0  # α occupied
        elif i < nos:
            spin_array[i] = 1  # β occupied
        elif i < nos + nvs // 2:
            spin_array[i] = 0  # α virtual
        else:
            spin_array[i] = 1  # β virtual
    return spin_array
spin_array = generate_spin_array(nos, nvs)
#print(spin_array[0:])

# %%
def initial_guess_zero(nos, nvs):
    t1 = np.zeros((nos, nvs))
    t2 = np.zeros((nos, nos, nvs, nvs))
    # t3 = np.zeros((nos, nos, nos, nvs, nvs, nvs))
    # t4 = np.zeros((nos, nos, nos, nos, nvs, nvs, nvs, nvs))

    return t1, t2
t1, t2 = initial_guess_zero(nos, nvs)
#print(t1[0])

# %%
def m_t_ext(m, actspin, t1, t2, nos, nvs):
    """
    Extract external T1–T4 amplitudes for a given actspin config (0 = active, 1 = inactive).
    """
    # Precompute boolean masks for inactive orbitals
    occ_mask = actspin[:nos] == 1
    virt_mask = actspin[nos:nos + nvs] == 1

    # T1
    mt1 = np.where((occ_mask[:,None] | virt_mask[None,:]), t1, 0.0)

    # T2
    mt2_mask = (
        occ_mask[:, None] | occ_mask[None, :]  # any inactive occ
    )[:, :, None, None] | (
        virt_mask[:, None] | virt_mask[None, :]  # any inactive virt
    )[None, None, :, :]
    mt2 = np.where(mt2_mask, t2, 0.0)

    # # T3
    # o3 = occ_mask[:, None, None] | occ_mask[None, :, None] | occ_mask[None, None, :]
    # v3 = virt_mask[:, None, None] | virt_mask[None, :, None] | virt_mask[None, None, :]
    # mt3 = np.where(o3[:, :, :, None, None, None] | v3[None, None, None, :, :, :], t3, 0.0)

    # # T4
    # o4 = (
    #     occ_mask[:, None, None, None] | occ_mask[None, :, None, None] |
    #     occ_mask[None, None, :, None] | occ_mask[None, None, None, :]
    # )
    # v4 = (
    #     virt_mask[:, None, None, None] | virt_mask[None, :, None, None] |
    #     virt_mask[None, None, :, None] | virt_mask[None, None, None, :]
    # )
    # mt4 = np.where(o4[:, :, :, :, None, None, None, None] | v4[None, None, None, None, :, :, :, :], t4, 0.0)




    return mt1, mt2



# %%
def expm(sigma, dim_fci, max_order=22, tol=1e-10, debug=False):
    """
    Taylor expansion of exp(+σ) and exp(−σ), like in Fortran-style CC logic.
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
        emt += (-1)**(k) * delta

        norm_term = np.linalg.norm(delta)
        if debug:
            print(f"  [k={k}] ‖term‖ = {norm_term:.2e}")
        if norm_term < tol:
            break
    else:
        raise RuntimeError("Taylor expansion failed to converge")

    return ept, emt


# %%
def compute_phase(ref, pos, neg):
    """
    Computes the phase factor (±1) for excitation: ref + pos - neg.
    
    Parameters:
    - ref : original determinant (0/1 array)
    - pos : list of orbitals being added (set to 1)
    - neg : list of orbitals being removed (set to 0)
    
    Returns:
    - iphase : ±1
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

def get_idiff_and_diff(string1, string2):
    """
    Returns number of excitations (idiff) and the difference bitvector.
    """
    diff = string1 - string2
    idiff = np.count_nonzero(diff) // 2
    return idiff, diff

def get_excitation_indices(diff):
    """
    Returns lists of positions where electrons were added (pos) or removed (neg).
    """
    pos = np.where(diff == 1)[0]   # string1 has 1, string2 has 0
    neg = np.where(diff == -1)[0]  # string1 has 0, string2 has 1
    return pos, neg


# %%

def m_t_ext_exp(mt1, mt2,
                nos, nvs, noas, nobs, nvas, nvbs, nstot,
                dim_fci, strings, actspin,
                debug=False):
    tm = np.zeros((dim_fci, dim_fci))

    virtual_indices = np.arange(nos, nos + nvs)
    occupied_indices = np.arange(0, nos)

    virtual_map = {v: i for i, v in enumerate(virtual_indices)}
    occupied_map = {v: i for i, v in enumerate(occupied_indices)}

    for i in range(dim_fci):
        string1 = strings[i]
        for j in range(dim_fci):
            string2 = strings[j]
            idiff, diff = get_idiff_and_diff(string1, string2)
            if idiff > 2:
                continue

            pos, neg = get_excitation_indices(diff)
            if len(pos) != idiff or len(neg) != idiff:
                continue

            if idiff == 1:
                p, q = pos[0], neg[0]
                if p in virtual_map and q in occupied_map:
                    iphase = compute_phase(string2, [p], [q])
                    tm[i, j] += iphase * mt1[occupied_map[q], virtual_map[p]]

            elif idiff == 2:
                p, q = pos
                r, s = neg
                if all(k in virtual_map for k in (p, q)) and all(k in occupied_map for k in (r, s)):
                    iphase = compute_phase(string2, [p, q], [r, s])
                    tm[i, j] += iphase * mt2[occupied_map[r], occupied_map[s], virtual_map[p], virtual_map[q]]

            # elif idiff == 3:
            #     p, q, r = pos
            #     s, t, u = neg
            #     if all(k in virtual_map for k in (p, q, r)) and all(k in occupied_map for k in (s, t, u)):
            #         iphase = compute_phase(string2, [p, q, r], [s, t, u])
            #         tm[i, j] += iphase * mt3[occupied_map[s], occupied_map[t], occupied_map[u],
            #                                 virtual_map[p], virtual_map[q], virtual_map[r]]

            # elif idiff == 4:
            #     p, q, r, s = pos
            #     t, u, v, w = neg
            #     if all(k in virtual_map for k in (p, q, r, s)) and all(k in occupied_map for k in (t, u, v, w)):
            #         iphase = compute_phase(string2, [p, q, r, s], [t, u, v, w])
            #         tm[i, j] += iphase * mt4[occupied_map[t], occupied_map[u], occupied_map[v], occupied_map[w],
            #                                 virtual_map[p], virtual_map[q], virtual_map[r], virtual_map[s]]

    sigma = tm - tm.T

    if debug:
        print("Max abs(sigma):", np.max(np.abs(sigma)))
        print("Frobenius norm of sigma:", np.linalg.norm(sigma))
        print("Non-zero σ elements:", np.count_nonzero(sigma), "/", sigma.size)
        print("‖σ + σᵀ‖ =", np.linalg.norm(sigma + sigma.T))

    ept, emt = expm(sigma, dim_fci, max_order=22, tol=1e-10, debug=debug)
    deviation = np.linalg.norm(ept @ emt.T - np.eye(dim_fci))

    if debug:
        print(f"‣ exp(+σ)·exp(−σ)^T deviation from I: {deviation:.3e}")
        if deviation > 1e-2:
            print("⚠️ Large deviation from identity!")

    return ept, emt, sigma, deviation


# %%
def sim_trans(dim_fci, ept, emt, matrix):
    """
    Performs the similarity transformation: matrix ← emt @ (matrix @ ept)
    
    Parameters:
    - matrix: (dim_fci, dim_fci) ndarray, FCI Hamiltonian (H)
    - ept:    (dim_fci, dim_fci) ndarray, exp(+sigma)
    - emt:    (dim_fci, dim_fci) ndarray, exp(-sigma)
    - debug:  bool, whether to print diagnostics and eigenvalues

    Returns:
    - None (modifies matrix in-place)
    """
    assert matrix.ndim == 2, "Matrix must be 2D"
    assert ept.shape == matrix.shape and emt.shape == matrix.shape, "Dimension mismatch"

    # Step 1: matrix × ept → m1
    m1 = matrix @ ept #@ is equivalent to matmul() 
    # Step 2: emt × m1 → matrix (overwrite)
    m2 = emt @ m1
   # matrix[:,:] = emt @ m1
    #m1 = np.dot(matrix, ept)
    #m2 = np.dot(emt, m1)
    #matrix[:,:] = m2
    
    return m2

# %%
def optimized_hierarchy_excitations(nos, nvs, nactspin,nactives):
    """
    Optimized version of hierarchy_excitations using NumPy vectorization for t1/t2
    and Numba acceleration for t3/t4.
    """
    nstot = nos + nvs
    #nactives = nactspin.shape[0]
    it1 = np.full((nos, nvs), -1, dtype=int)
    it2 = np.full((nos, nos, nvs, nvs), -1, dtype=int)
    

    occ_active = (nactspin[:, :nos] == 0)
    #print("Occupied active indices:" , occ_active)
    virt_active = (nactspin[:, nos:] == 0)
    #print("Virtual active indices:", virt_active)

    # t1: i -> a
    for m in range(nactives):
        oa = occ_active[m]
        va = virt_active[m]
        mask = np.outer(oa, va)
        indices = np.where(mask & (it1 == -1))
        it1[indices] = m

    # t2: ij -> ab
    for m in range(nactives):
        o = occ_active[m]
        v = virt_active[m]
        occ_mask = np.outer(o, o).astype(bool)
        virt_mask = np.outer(v, v).astype(bool)
        mask4d = occ_mask[:, :, None, None] & virt_mask[None, None, :, :]
        indices = np.where(mask4d & (it2 == -1))
        it2[indices] = m

#     # Use Numba for t3/t4
#     fast_hierarchy_t3_t4(it3, it4, nactspin, nos, nvs, nactives)

    return it1, it2

# @njit
# def fast_hierarchy_t3_t4(it3, it4, nactspin, nos, nvs, nactives):
#     for m in range(nactives):
#         for i in range(nos):
#             for j in range(nos):
#                 for k in range(nos):
#                     for a in range(nvs):
#                         for b in range(nvs):
#                             for c in range(nvs):
#                                 if (nactspin[m, i] + nactspin[m, j] + nactspin[m, k] +
#                                     nactspin[m, nos + a] + nactspin[m, nos + b] + nactspin[m, nos + c] == 0):
#                                     if it3[i, j, k, a, b, c] == -1:
#                                         it3[i, j, k, a, b, c] = m
#         for i in range(nos):
#             for j in range(nos):
#                 for k in range(nos):
#                     for l in range(nos):
#                         for a in range(nvs):
#                             for b in range(nvs):
#                                 for c in range(nvs):
#                                     for d in range(nvs):
#                                         if (nactspin[m, i] + nactspin[m, j] + nactspin[m, k] + nactspin[m, l] +
#                                             nactspin[m, nos + a] + nactspin[m, nos + b] +
#                                             nactspin[m, nos + c] + nactspin[m, nos + d] == 0):
#                                             if it4[i, j, k, l, a, b, c, d] == -1:
#                                                 it4[i, j, k, l, a, b, c, d] = m

it1, it2 = optimized_hierarchy_excitations(nos, nvs, nactspin, nactives)
# for m in range(nactspin.shape[0]):
#     print(f"[m={m}] T1 count:", np.sum(it1 == (m)))
#     print(f"[m={m}] T2 count:", np.sum(it2 == (m)))
#     print(f"[m={m}] T3 count:", np.sum(it3 == (m)))
#     print(f"[m={m}] T4 count:", np.sum(it4 == (m)))


# %%
def mnum12(m, actspin, it1, it2, spin_array, nos, nvs):
    """
    Counts the number of single, double, triple, and quadruple excitations
    for a given excitation level m and active space actspin.

    Parameters:
    - m : int, current excitation level
    - actspin : (nstot,) array of 0 (active) and 1 (inactive) orbital flags
    - it1, it2, it3, it4 : integer arrays indexing excitation classes
    - spin_array : (nstot,) array indicating spin (e.g., 0=α, 1=β)
    - nos : number of occupied orbitals
    - nvs : number of virtual orbitals

    Returns:
    - mnum1, mnum2, mnum3, mnum4 : number of valid 1-, 2-, 3-, 4-body excitations
    """

    mnum1 = mnum2 = mnum3 = mnum4 = 0
    nstot = nos + nvs

    # --- Singles ---
    for i in range(nos):
        for ia in range(nvs):
            ag = nos + ia
            if actspin[i] + actspin[ag] != 0:
                continue  #skip if either orbital is inactive
            if it1[i, ia] != m :  #skip if not in the m-excitation class
                continue
            if spin_array[i] != spin_array[ag]:
                continue  #skip if spin mismatch     
            mnum1 += 1

    # --- Doubles ---
    for i in range(nos):
        for j in range(i + 1, nos):
            for ia in range(nvs):
                for ib in range(ia + 1, nvs):
                    ag, bg = nos + ia, nos + ib
                    if actspin[i] + actspin[j] + actspin[ag] + actspin[bg] != 0:
                        continue
                    if it2[i,j,ia, ib] != m:
                        continue
                    if spin_array[i] + spin_array[j] != spin_array[ag] + spin_array[bg]:
                        continue
                    mnum2 += 1

    # # --- Triples ---
    # for i in range(nos):
    #     for j in range(i + 1, nos):
    #         for k in range(j + 1, nos):
    #             for ia in range(nvs):
    #                 for ib in range(ia + 1, nvs):
    #                     for ic in range(ib + 1, nvs):
    #                         ag, bg, cg = nos + ia, nos + ib, nos + ic
    #                         if actspin[i] + actspin[j] + actspin[k] + actspin[ag] + actspin[bg] + actspin[cg] != 0:
    #                             continue
    #                         if it3[i,j,k,ia,ib,ic] != m:
    #                             continue
    #                         if spin_array[i] + spin_array[j] + spin_array[k] != spin_array[ag] + spin_array[bg] + spin_array[cg]:
    #                             continue
    #                         mnum3 += 1

    # # --- Quadruples ---
    # for i in range(nos):
    #     for j in range(i + 1, nos):
    #         for k in range(j + 1, nos):
    #             for l in range(k + 1, nos):
    #                 for ia in range(nvs):
    #                     for ib in range(ia + 1, nvs):
    #                         for ic in range(ib + 1, nvs):
    #                             for id in range(ic + 1, nvs):
    #                                 ag, bg, cg, dg = nos + ia, nos + ib, nos + ic, nos + id
    #                                 if actspin[i] + actspin[j] + actspin[k] + actspin[l] +actspin[ag] + actspin[bg] + actspin[cg] + actspin[dg] != 0:
    #                                     continue
    #                                 if it4[i,j,k,l,ia,ib,ic,id] != m :
    #                                     continue
    #                                 if spin_array[i]+ spin_array[j] + spin_array[k] + spin_array[l] != spin_array[ag] + spin_array[bg] + spin_array[cg] + spin_array[dg]:
    #                                     continue 
    #                                 mnum4 += 1



    return mnum1, mnum2


# %%
import numpy as np
import itertools

def create_mlists_xm(m, actspin, it1, it2, t1, t2,
                                spin_array, nos, nvs, mnum1max, mnum2max, dim_m_max):

    nstot = nos + nvs
    virt_global = np.arange(nos, nstot)
    occ_active = np.where(actspin[:nos] == 0)[0]
    virt_active = np.where(actspin[nos:] == 0)[0]
    spin_occ = spin_array[:nos]
    spin_virt = spin_array[nos:]

    m_list1 = np.zeros((mnum1max, 2), dtype=int)
    m_list2 = np.zeros((mnum2max, 4), dtype=int)

    xm = np.zeros((dim_m_max,), dtype=float)

    # --- Singles ---
    n = 0
    for i in occ_active:
        for ia in virt_active:
            ag = nos + ia
            if it1[i, ia] == m and spin_occ[i] == spin_virt[ia]:
                m_list1[n, :] = [i, ag]
                xm[n] = t1[i, ia]
                n += 1
    mnum1 = n

    # --- Doubles ---
    n = 0
    for i, j in itertools.combinations(occ_active, 2):
        spin_ij = spin_occ[i] + spin_occ[j]
        for ia, ib in itertools.combinations(virt_active, 2):
            ag, bg = nos + ia, nos + ib
            if it2[i, j, ia, ib] == m and spin_ij == spin_virt[ia] + spin_virt[ib]:
                m_list2[n, :] = [i, j, ag, bg]
                xm[n + mnum1] = t2[i, j, ia, ib]
                n += 1
    mnum2 = n

    # # --- Triples ---
    # n = 0
    # for i, j, k in itertools.combinations(occ_active, 3):
    #     spin_ijk = spin_occ[i] + spin_occ[j] + spin_occ[k]
    #     for ia, ib, ic in itertools.combinations(virt_active, 3):
    #         ag, bg, cg = nos + ia, nos + ib, nos + ic
    #         if (it3[i, j, k, ia, ib, ic] == m and
    #             spin_ijk == spin_virt[ia] + spin_virt[ib] + spin_virt[ic]):
    #             m_list3[n, :] = [i, j, k, ag, bg, cg]
    #             xm[n + mnum1 + mnum2] = t3[i, j, k, ia, ib, ic]
    #             n += 1
    # mnum3 = n

    # # --- Quadruples ---
    # n = 0
    # for i, j, k, l in itertools.combinations(occ_active, 4):
    #     spin_ijkl = spin_occ[i] + spin_occ[j] + spin_occ[k] + spin_occ[l]
    #     for ia, ib, ic, id in itertools.combinations(virt_active, 4):
    #         ag, bg, cg, dg = nos + ia, nos + ib, nos + ic, nos + id
    #         if (it4[i, j, k, l, ia, ib, ic, id] == m and
    #             spin_ijkl == spin_virt[ia] + spin_virt[ib] + spin_virt[ic] + spin_virt[id]):
    #             m_list4[n, :] = [i, j, k, l, ag, bg, cg, dg]
    #             xm[n + mnum1 + mnum2 + mnum3] = t4[i, j, k, l, ia, ib, ic, id]
    #             n += 1
    # mnum4 = n
  
    dim_m = mnum1 + mnum2 

    return (
        m_list1[:mnum1],
        m_list2[:mnum2],
        xm[:dim_m],
        mnum1, mnum2, dim_m
    )


# %%
def m_t_int(m, actspin, t1, t2, nos, nvs):
    """
    Extract internal T amplitudes (mt1–mt4) based on active spin mask (0 = active).
    """
    # Masks: True if ACTIVE
    occ_mask = actspin[:nos] == 0
    virt_mask = actspin[nos:nos + nvs] == 0

    # T1: i → a
    mt1 = np.where(np.outer(occ_mask, virt_mask), t1, 0.0)

    # T2: ij → ab
    o2 = (
        occ_mask[:, None] & occ_mask[None, :]
    )[:, :, None, None]
    v2 = (
        virt_mask[:, None] & virt_mask[None, :]
    )[None, None, :, :]
    mt2 = np.where(o2 & v2, t2, 0.0)

    # # T3: ijk → abc
    # o3 = (
    #     occ_mask[:, None, None] & occ_mask[None, :, None] & occ_mask[None, None, :]
    # )
    # v3 = (
    #     virt_mask[:, None, None] & virt_mask[None, :, None] & virt_mask[None, None, :]
    # )
    # mt3 = np.where(o3[:, :, :, None, None, None] & v3[None, None, None, :, :, :], t3, 0.0)

    # # T4: ijkl → abcd
    # o4 = (
    #     occ_mask[:, None, None, None] & occ_mask[None, :, None, None] &
    #     occ_mask[None, None, :, None] & occ_mask[None, None, None, :]
    # )
    # v4 = (
    #     virt_mask[:, None, None, None] & virt_mask[None, :, None, None] &
    #     virt_mask[None, None, :, None] & virt_mask[None, None, None, :]
    # )
    # mt4 = np.where(o4[:, :, :, :, None, None, None, None] & v4[None, None, None, None, :, :, :, :], t4, 0.0)


    return mt1, mt2

# %%
def commutator(a, b):
    """
    Compute the commutator [a, b] = ab - ba
    """
    # Ensure input matrices are NumPy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    c = a @ b - b @ a  # Equivalent to: ab - ba
    
    return c

# %%
def zero_matrix(matrix):
    """
    Zero out a square matrix.
    """
    matrix[:, :] = 0.0


# %%
def m_excitation(mt1, mt2, nos, nvs, dim_fci, strings):
    tm = np.zeros((dim_fci, dim_fci))

    for i in range(dim_fci):
        for j in range(dim_fci):
            string1 = strings[i]
            string2 = strings[j]
            idiff, diff = get_idiff_and_diff(string1, string2)
            if idiff == 0 or idiff > 4:
                continue

            pos, neg = get_excitation_indices(diff)

            if idiff == 1 and len(pos) == 1 and len(neg) == 1:
                p, q = pos[0], neg[0]
                if p >= nos and q < nos:
                    iphase = compute_phase(string2, [p], [q])
                    tm[i, j] += iphase * mt1[q, p - nos]

            elif idiff == 2 and len(pos) == 2 and len(neg) == 2:
                p, q = sorted(pos)
                r, s = sorted(neg)
                if p >= nos and q >= nos and r < nos and s < nos:
                    iphase = compute_phase(string2, [p, q], [r, s])
                    tm[i, j] += iphase * mt2[r, s, p - nos, q - nos]

            # elif idiff == 3 and len(pos) == 3 and len(neg) == 3:
            #     p, q, r = sorted(pos)
            #     s, t, u = sorted(neg)
            #     if all(x >= nos for x in (p, q, r)) and all(x < nos for x in (s, t, u)):
            #         iphase = compute_phase(string2, [p, q, r], [s, t, u])
            #         tm[i, j] += iphase * mt3[s, t, u, p - nos, q - nos, r - nos]

            # elif idiff == 4 and len(pos) == 4 and len(neg) == 4:
            #     p, q, r, s = sorted(pos)
            #     t, u, v, w = sorted(neg)
            #     if all(x >= nos for x in (p, q, r, s)) and all(x < nos for x in (t, u, v, w)):
            #         iphase = compute_phase(string2, [p, q, r, s], [t, u, v, w])
            #         tm[i, j] += iphase * mt4[t, u, v, w, p - nos, q - nos, r - nos, s - nos]

    sigma = tm - tm.T
    return sigma


# %%
import itertools
from functools import lru_cache

# @lru_cache(maxsize=None)
# def permut_ind4_cached(a, b, c, d):
#     original = (a, b, c, d)
#     perms = list(itertools.permutations(original))
#     result = []
#     position = {val: i for i, val in enumerate(original)}
#     for perm in perms:
#         perm_indices = [position[p] for p in perm]
#         sign = 1
#         for i in range(len(perm_indices)):
#             for j in range(i + 1, len(perm_indices)):
#                 if perm_indices[i] > perm_indices[j]:
#                     sign *= -1
#         result.append((*perm, sign))
#     return result

# def permutation_sign(perm, original):
#     position = {val: i for i, val in enumerate(original)}
#     perm_indices = [position[p] for p in perm]
#     sign = 1
#     for i in range(len(perm_indices)):
#         for j in range(i + 1, len(perm_indices)):
#             if perm_indices[i] > perm_indices[j]:
#                 sign *= -1
#     return sign

def x_fan_out_int(xm, mt1, mt2,
                  mnum1, mnum2,
                  m_list1, m_list2,
                  dim_m, actspin, nos, nvs):
    """
    Distributes internal excitation vector `xm` into mt1–mt4 amplitude tensors.
    """
    offset = 0

    # T1
    for n in range(mnum1):
        i, a_global = m_list1[n]
        a_local = a_global - nos
        assert 0 <= a_local < nvs, f"T1: invalid a_local={a_local}"
        mt1[i, a_local] = xm[offset + n]
    offset += mnum1

    # T2
    for n in range(mnum2):
        i, j, a, b = m_list2[n]
        a_local, b_local = a - nos, b - nos
        assert 0 <= a_local < nvs and 0 <= b_local < nvs, f"T2: invalid a/b"
        val = xm[offset + n]
        mt2[i, j, a_local, b_local] =  val
        mt2[i, j, b_local, a_local] = -val
        mt2[j, i, a_local, b_local] = -val
        mt2[j, i, b_local, a_local] =  val
    offset += mnum2

    # # T3
    # for n in range(mnum3):
    #     i, j, k, a, b, c = m_list3[n]
    #     val = xm[offset + n]
    #     occ = [i, j, k]
    #     virt = [a - nos, b - nos, c - nos]
    #     assert all(0 <= v < nvs for v in virt), f"T3: invalid virtuals {virt}"
    #     for occ_perm in itertools.permutations(occ):
    #         for virt_perm in itertools.permutations(virt):
    #             sign = permutation_sign(occ_perm, occ) * permutation_sign(virt_perm, virt)
    #             mt3[occ_perm[0], occ_perm[1], occ_perm[2],
    #                 virt_perm[0], virt_perm[1], virt_perm[2]] = sign * val
    # offset += mnum3

    # # T4
    # for n in range(mnum4):
    #     i, j, k, l, a, b, c, d = m_list4[n]
    #     val = xm[offset + n]
    #     occ_perms = permut_ind4_cached(i, j, k, l)
    #     virt_perms = permut_ind4_cached(a - nos, b - nos, c - nos, d - nos)
    #     assert all(0 <= v - nos < nvs for v in [a, b, c, d]), "T4: virtual out of range"
    #     for op in occ_perms:
    #         for vp in virt_perms:
    #             sign = op[4] * vp[4]
    #             mt4[op[0], op[1], op[2], op[3],
    #                 vp[0], vp[1], vp[2], vp[3]] = sign * val

    # Final check
    assert xm.shape[0] == mnum1 + mnum2, "xm length mismatch"
    return mt1, mt2


# %%
def fn_m(dim_fci, dim_m,
         mnum1, mnum2,
         mnum1max, mnum2max,dim_m_max,
         nos, nvs, noas, nobs, nvas, nvbs, nstot,
         trial, matrix,
         mt1, mt2,
         m_list1, m_list2,
         xm, strings, actspin,
         debug=False):
    """
    Computes ⟨trial| e^{-σ} H e^{σ} |trial⟩ = trialᵀ · H_eff · trial.

    Parameters:
        xm: (dim_m,) internal amplitudes
        mt1–mt4: scratch arrays reused for amplitudes
        actspin: active space mask for current excitation
        matrix: original FCI Hamiltonian
        trial: reference trial vector

    Returns:
        energy: float, expectation value of similarity-transformed Hamiltonian
    """

    # 1. Fan out xm → mt1–mt4 (internal cluster amplitudes)
    mt1, mt2 = x_fan_out_int(
        xm, mt1, mt2,
        mnum1, mnum2,
        m_list1, m_list2,
        dim_m, actspin, nos, nvs
    )

    # 2. Construct exp(+σ), exp(−σ) from antisymmetric sigma matrix
    ept, emt, sigma, deviation = m_t_ext_exp(
        mt1, mt2,
        nos, nvs, noas, nobs, nvas, nvbs, nstot,
        dim_fci, strings, actspin,
        debug=debug
    )

    if debug and deviation > 1e-2:
        print("⚠️  Warning: exp(σ) · exp(−σ) deviates from identity!")

    # 3. Similarity transformation: matrix_aux = emt @ (matrix @ ept)
    matrix_aux = matrix.copy()
    matrix_aux = sim_trans(dim_fci, ept, emt, matrix)

    # 4. Energy expectation value: trialᵀ · H_eff · trial
    energy = np.dot(trial, matrix_aux @ trial)

    return energy


# %%
import itertools

def update_global_amplitudes(t1, t2,
                             xm, mnum1, mnum2,
                             m_list1, m_list2,
                             dim_m, actspin, nos, nvs):
    """
    Writes external amplitudes `xm` into global cluster amplitudes T1–T4,
    using excitation lists and maintaining antisymmetry where applicable.
    """

    offset = 0

    # T1
    for n in range(mnum1):
        i, a_global = m_list1[n]
        a_local = a_global - nos
        t1[i, a_local] = xm[offset + n]
    offset += mnum1

    # T2: antisymmetric under (i<->j) and (a<->b)
    for n in range(mnum2):
        i, j, a, b = m_list2[n]
        a_local, b_local = a - nos, b - nos
        val = xm[offset + n]
        t2[i, j, a_local, b_local] =  val
        t2[i, j, b_local, a_local] = -val
        t2[j, i, a_local, b_local] = -val
        t2[j, i, b_local, a_local] =  val
    offset += mnum2

    # # T3: antisymmetric via permutations
    # for n in range(mnum3):
    #     i, j, k, a, b, c = m_list3[n]
    #     val = xm[offset + n]
    #     occ = [i, j, k]
    #     virt = [a - nos, b - nos, c - nos]
    #     for occ_perm in itertools.permutations(occ):
    #         for virt_perm in itertools.permutations(virt):
    #             sign = permutation_sign(occ_perm, occ) * permutation_sign(virt_perm, virt)
    #             t3[occ_perm[0], occ_perm[1], occ_perm[2],
    #                virt_perm[0], virt_perm[1], virt_perm[2]] = sign * val
    # offset += mnum3

    # # T4: antisymmetric via cached 4-index permutations
    # for n in range(mnum4):
    #     i, j, k, l, a, b, c, d = m_list4[n]
    #     val = xm[offset + n]
    #     occ_perms = permut_ind4_cached(i, j, k, l)
    #     virt_perms = permut_ind4_cached(a - nos, b - nos, c - nos, d - nos)
    #     for op in occ_perms:
    #         for vp in virt_perms:
    #             sign = op[4] * vp[4]
    #             t4[op[0], op[1], op[2], op[3],
    #                vp[0], vp[1], vp[2], vp[3]] = sign * val


    return t1, t2
def save_global_pool(iter_idx, t1, t2, outdir="amplitudes_ckpt", prefix="amps", save_dense=False):
    """
    Save the global T1/T2 amplitude pool after each global iteration.
    - Writes an NPZ (compressed) with t1, t2, iter
    - Writes a small JSON summary (shapes, nnz, max|val|)
    - Optionally writes a sparse text dump of nonzeros (default off)
    """
    os.makedirs(outdir, exist_ok=True)
    base = f"{prefix}_iter{iter_idx:03d}"
    # ----------NPZ(atomic)-------------
    npz_tmp = tempfile.NamedTemporaryFile(delete=False, dir=outdir, suffix=".npz")
    try:
        np.savez_compressed(npz_tmp, t1=t1, t2=t2, iter=iter_idx)
        npz_tmp.flush(); os.fsync(npz_tmp.fileno()); npz_tmp.close()
        final_npz = os.path.join(outdir, base + ".npz")
        os.replace(npz_tmp.name, final_npz)
    except Exception:
        try: os.unlink(npz_tmp.name)
        except Exception: pass
        raise
     # ---- JSON summary (atomic) ----
    summary = {
        "iter": iter_idx,
        "t1_shape": list(t1.shape),
        "t2_shape": list(t2.shape),
        "t1_nonzero": int(np.count_nonzero(t1)),
        "t2_nonzero": int(np.count_nonzero(t2)),
        "t1_max_abs": float(np.max(np.abs(t1))) if t1.size else 0.0,
        "t2_max_abs": float(np.max(np.abs(t2))) if t2.size else 0.0,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_tmp_path = os.path.join(outdir, base + ".json.tmp")
    json_final_path = os.path.join(outdir, base + ".json")
    with open(json_tmp_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(json_tmp_path, json_final_path)

    # ---- Optional sparse text (one-sided antisym storage) ----
    if not save_dense:
        txt_tmp_path = os.path.join(outdir, base + ".txt.tmp")
        txt_final_path = os.path.join(outdir, base + ".txt")
        with open(txt_tmp_path, "w") as f:
            # T1: i a val (store all nonzeros)
            nz_i, nz_a = np.nonzero(t1)
            for i, a in zip(nz_i, nz_a):
                f.write(f"T1 {i} {a} {t1[i,a]:.16e}\n")
            # T2: store only i<j and a<b to avoid duplicates from antisymmetry
            nos, _, nvs, _ = t2.shape
            for i in range(nos):
                for j in range(i+1, nos):
                    for a in range(nvs):
                        for b in range(a+1, nvs):
                            val = t2[i, j, a, b]
                            if val != 0.0:
                                f.write(f"T2 {i} {j} {a} {b} {val:.16e}\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(txt_tmp_path, txt_final_path)

    return final_npz  # path to the NPZ

# %%
@njit
def zero_t12(mt1, mt2, nos, nvs):
    """
    Zero out all elements of internal amplitude tensors mt1 to mt4.
    """
    # mt1[i, a]
    mt1[:nos, :nvs] = 0.0

    # mt2[i, j, a, b]
    mt2[:nos, :nos, :nvs, :nvs] = 0.0

    

# %%


# %%

def mpraxis(T0, MACHEP, H0, PRIN,
            x, dim_fci,
            mt1, mt2,
            m_list1, m_list2,
            mnum1, mnum2,
            mnum1max, mnum2max, dim_m_max,
            nos, nvs, noas, nobs, nvas, nvbs, nstot,
            trial, matrix,
            strings, actspin,
            zero_t12, m_excitation, commutator, fn_m,
            verbose=True):

    dim_m = len(x)
    gradient = np.zeros(dim_m)
    dstep = 1.0e-1
    max_steps = 10

    # Allocate reusable aux tensors
    aux1 = np.zeros_like(mt1)
    aux2 = np.zeros_like(mt2)
  

    # --- Compute Gradient ---
    offset = 0
    for p, (i, ia_global) in enumerate(m_list1):
        ia = ia_global - nos
        aux1.fill(0.0); aux2.fill(0.0);
        aux1[i, ia] = 1.0
        excitation = m_excitation(aux1, aux2, nos, nvs, dim_fci, strings)
        cm = commutator(matrix, excitation)
        gradient[p] = fn_m(dim_fci, dim_m, mnum1, mnum2, 
                           mnum1max, mnum2max,dim_m_max,
                           nos, nvs, noas, nobs, nvas, nvbs, nstot,
                           trial, cm, mt1, mt2, 
                           m_list1, m_list2, x,
                           strings, actspin)

    offset = mnum1
    for p, (i, j, ia, ib) in enumerate(m_list2):
        aux1.fill(0.0); aux2.fill(0.0); 
        aux2[i, j, ia - nos, ib - nos] = 1.0
        excitation = m_excitation(aux1, aux2,  nos, nvs, dim_fci, strings)
        cm = commutator(matrix, excitation)
        gradient[offset + p] = fn_m(dim_fci, dim_m, mnum1, mnum2, 
                                    mnum1max, mnum2max, dim_m_max,
                                    nos, nvs, noas, nobs, nvas, nvbs, nstot,
                                    trial, cm, mt1, mt2, 
                                    m_list1, m_list2, x,
                                    strings, actspin)

    # offset += mnum2
    # for p, (i, j, k, ia, ib, ic) in enumerate(m_list3):
    #     aux1.fill(0.0); aux2.fill(0.0); aux3.fill(0.0); aux4.fill(0.0)
    #     aux3[i, j, k, ia - nos, ib - nos, ic - nos] = 1.0
    #     excitation = m_excitation(aux1, aux2, aux3, aux4, nos, nvs, dim_fci, strings)
    #     cm = commutator(matrix, excitation)
    #     gradient[offset + p] = fn_m(dim_fci, dim_m, mnum1, mnum2, mnum3, mnum4,
    #                                 mnum1max, mnum2max, mnum3max, mnum4max, dim_m_max,
    #                                 nos, nvs, noas, nobs, nvas, nvbs, nstot,
    #                                 trial, cm, mt1, mt2, mt3, mt4,
    #                                 m_list1, m_list2, m_list3, m_list4, x,
    #                                 strings, actspin)

    # offset += mnum3
    # for p, (i, j, k, l, ia, ib, ic, id) in enumerate(m_list4):
    #     aux1.fill(0.0); aux2.fill(0.0); aux3.fill(0.0); aux4.fill(0.0)
    #     aux4[i, j, k, l, ia - nos, ib - nos, ic - nos, id - nos] = 1.0
    #     excitation = m_excitation(aux1, aux2, aux3, aux4, nos, nvs, dim_fci, strings)
    #     cm = commutator(matrix, excitation)
    #     gradient[offset + p] = fn_m(dim_fci, dim_m, mnum1, mnum2, mnum3, mnum4,
    #                                 mnum1max, mnum2max, mnum3max, mnum4max, dim_m_max,
    #                                 nos, nvs, noas, nobs, nvas, nvbs, nstot,
    #                                 trial, cm, mt1, mt2, mt3, mt4,
    #                                 m_list1, m_list2, m_list3, m_list4, x,
    #                                 strings, actspin)




    # --- Simple Gradient Descent ---
    xene_old = fn_m(dim_fci, dim_m, mnum1, mnum2,
                    mnum1max, mnum2max, dim_m_max,
                    nos, nvs, noas, nobs, nvas, nvbs, nstot,
                    trial, matrix, mt1, mt2, 
                    m_list1, m_list2, x,
                    strings, actspin)

    for r in range(max_steps):
        x_new = x - dstep * gradient
        xene = fn_m(dim_fci, dim_m, mnum1, mnum2,
                    mnum1max, mnum2max, dim_m_max,
                    nos, nvs, noas, nobs, nvas, nvbs, nstot,
                    trial, matrix, mt1, mt2, 
                    m_list1, m_list2, x_new,
                    strings, actspin)
        if xene < xene_old:
            x = x_new
            xene_old = xene
        else:
            break

    fmin = xene_old
    return fmin, x, gradient


# %%

def run_qflow_loop(
    nactives, nactspin, strings, matrix_h, dim_fci, nstot,
    nos, nvs, noas, nobs, nvas, nvbs,
    t1, t2, 
    it1, it2,
    #fn_m, m_excitation, commutator, zero_t1234,
    m_t_ext, 
    m_t_ext_exp, 
    sim_trans, 
    mnum12, 
    create_mlists_xm,
    m_t_int,
    mpraxis,
    update_global_amplitudes,
    mnum1max, mnum2max,dim_m_max,
    spin_array,
    trial,
    verbose=True
):
    maxiter = 20
    xene_old = 0.0
    #trial = np.zeros(dim_fci)
    #trial[0] = 1.0
    energy_list = []

    for i in range(maxiter):
        if verbose:
            print(f"\n>>> Global QFlow Iteration {i+1}")
        for m in range(nactives):
            actspin = nactspin[m]
            matrix = matrix_h.copy()
            #print(f">>> Active space {m}: actspin = {actspin}")
            
            # Step 1: External T amplitudes for m-th active space
            mt1, mt2 = m_t_ext(m, actspin, t1, t2,
                                         nos, nvs)
            # print(">>> [m_t_ext] max|mt1|:", np.max(np.abs(mt1)))
            # print("Shape is mt1: ", mt1.shape, "mt2 : ", mt2.shape, "mt3 : ", mt3.shape, "mt4 :", mt4.shape)
            # # Print a few representative entries from mt1
            # print(">>> [mt2] first few entries:")
            # for i in range(min(4, mt2.shape[0])):
            #     print(f"  row {i}: {mt2[i, :5]}")  # first 5 virtuals of row i

    #         # Step 2: Matrix exponentials
            ept, emt, sigma, deviation = m_t_ext_exp(mt1, mt2,
                                    nos, nvs, noas, nobs, nvas, nvbs, nstot,
                                    dim_fci, strings, actspin)
    #         print(f">>> Checking mt1–mt4 before sigma construction:")
    #         print("‣ max|mt1|:", np.max(np.abs(mt1)))
    #         print("‣ max|mt2|:", np.max(np.abs(mt2)))
    #         print("‣ max|mt3|:", np.max(np.abs(mt3)))
    #         print("‣ max|mt4|:", np.max(np.abs(mt4)))
            #print(">>> [mt2] first few entries:")
            #for i in range(min(4, mt2.shape[0])):
            #     print(f"  row {i}: {mt2[i, :5]}")  # first 5 virtuals of row i
    #         if deviation > 1e-2:
    #             print(f"[m={m}] ⚠️ Large deviation: {deviation:.2e}")

    #         # Step 3: Similarity transform
            matrix = sim_trans(dim_fci, ept, emt, matrix)
            #print(matrix.shape)
            #print("Post-transform matrix(1,1):", matrix[0, 0])
            #print("Post-transform matrix(1,1):", matrix[1, 1])
            #print("Post-transform matrix(1,2):", matrix[4, 4])

    #         # # Step 4: Count internal excitation terms
            mnum1, mnum2 = mnum12(m, actspin, it1, it2, spin_array, nos, nvs)
            # Optional safety check for T1+T2-only mode
            
            #dim_m = mnum1 + mnum2 + mnum3 + mnum4
            #print(f"[m={m}] mnum1={mnum1} mnum2={mnum2} mnum3={mnum3} mnum4={mnum4}")

            #print("print_dim_m", dim_m)
            #print("mnum1 + mnum2 + mnum3 + mnum4 =", mnum1 + mnum2 + mnum3 + mnum4)

    #         assert mnum1 <= mnum1max, "Exceeded mnum1max"
    #         assert mnum2 <= mnum2max, "Exceeded mnum2max"
    #         assert mnum3 <= mnum3max, "Exceeded mnum3max"
    #         assert mnum4 <= mnum4max, "Exceeded mnum4max"
    #         assert dim_m <= dim_m_max, "Exceeded dim_m_max"

    #         # Step 4–5: Create m_lists, excitation counts, and xm
            m_list1, m_list2, xm, mnum1, mnum2, dim_m = create_mlists_xm(m, actspin, it1, it2, t1, t2,
                      spin_array, nos, nvs, mnum1max, mnum2max, dim_m_max)
            # print(f"--- [m={m}] Excitation Summary ---")
            # print(f"Singles (T1): {mnum1}")
            # print(f"m_list1 has :", len(m_list1))
            # if mnum1 > 0:
            #     print("  T1 excitation:", m_list1)
            #     print("  T1 amplitude:", xm)

            # print(f"Doubles (T2): {mnum2}")
            # print(f"m_list2 has :", len(m_list2))
            # if mnum2 > 0:
            #      print("  T2 excitation:", m_list2)
            #      print("  T2 amplitude:", xm[mnum1])  # offset by mnum1

            # print(f"Triples (T3): {mnum3}")
            # print(f"m_list3 has :", len(m_list3))
            # if mnum3 > 0:
            #      print("  T3 excitation:", m_list3)
            #      print("  T3 amplitude:", xm[mnum1 + mnum2])  # offset by T1+T2

            #print(f"Quadruples (T4): {mnum4}")
            #if mnum4 > 0:
            #     print("  First T4 excitation:", m_list4[0])
            #     print("  T4 amplitude:", xm[mnum1 + mnum2 + mnum3])  # offset by T1+T2+T3


    #         # Step 6: Internal T amplitudes
            mt1, mt2 = m_t_int(m, actspin, t1, t2, nos, nvs)
            # print("mt1.shape =", mt1.shape)
            # print("mt2.shape =", mt2.shape)
            # print("mt3.shape =", mt3.shape)
            # print("mt4.shape =", mt4.shape)

# Optional: print total number of elements for checking consistency with xm
            #print("Total elements in mt1:", np.count_nonzero(mt1))
            #print("Total elements in mt2:", np.count_nonzero(mt2))
            #print("Total elements in mt3:", np.count_nonzero(mt3))
            #print("Total elements in mt4:", np.count_nonzero(mt4))

                                   
    #         # Step 7: Run minimization
            energy, xm_opt,_ = mpraxis(
                 T0=0.00, MACHEP=1e-18, H0=0.3, PRIN=0, x=xm, dim_fci=dim_fci,
                 mt1=mt1, mt2=mt2,
                 m_list1=m_list1, m_list2=m_list2, 
                 mnum1=mnum1, mnum2=mnum2,
                 mnum1max=mnum1max, mnum2max=mnum2max,
                dim_m_max=dim_m_max,
                 nos=nos, nvs=nvs, noas=noas, nobs=nobs, nvas=nvas, nvbs=nvbs, nstot=nstot,
                 trial=trial, matrix=matrix, 
                 strings=strings, actspin=actspin,zero_t12=zero_t12, m_excitation=m_excitation,
                 commutator=commutator, fn_m=fn_m,
                 verbose=verbose)
            energy_list.append(energy)

            if verbose:
                 print(f"  Active space {m+1}/{nactives} optimized energy: {energy:.10f}")

#     #         # Step 8: Update global amplitudes
            update_global_amplitudes(t1, t2,
                         xm_opt, mnum1, mnum2,
                         m_list1, m_list2,
                         dim_m, actspin, nos, nvs)
            save_global_pool(i+1, t1, t2, outdir="amplitudes_ckpt", prefix="amps", save_dense=False)
        #    t1, t2, t3, t4 = x_fan_out_int(xm, mt1, mt2, mt3, mt4, mnum1, mnum2, mnum3,mnum4, 
         #          m_list1, m_list2, m_list3, m_list4, dim_m, actspin)

    #     # Convergence check
        # if m == 0:
        #         if np.abs(energy - xene_old) < 1e-6:
        #             print(f"\n✅ Converged at global iteration {i+1} with energy: {energy:.10f}")
        #             return energy, energy_list
        #         xene_old = energy
    #energy_list.append(energy)        

    return energy, energy_list


# %%
mnum1max = 8
mnum2max = 18

dim_m_max = 26

final_energy, energy_list = run_qflow_loop(
    nactives=nactives,
    nactspin=nactspin,
    strings=strings,
    matrix_h=matrix_h,
    dim_fci=dim_fci,
    nstot=nstot,
    nos=nos, nvs=nvs, noas=noas, nobs=nobs, nvas=nvas, nvbs=nvbs,
    t1=t1, t2=t2, 
    it1=it1, it2=it2,
    # fn_m=fn_m,
    # m_excitation=m_excitation,
    # commutator=commutator,
    # zero_t1234=zero_t1234,
    m_t_ext=m_t_ext,
    m_t_ext_exp=m_t_ext_exp,
    sim_trans=sim_trans,
    mnum12=mnum12,  
    create_mlists_xm=create_mlists_xm,
    m_t_int=m_t_int,
    mpraxis=mpraxis,
    update_global_amplitudes=update_global_amplitudes,
    mnum1max=mnum1max, mnum2max=mnum2max,
    dim_m_max=dim_m_max,
    spin_array=spin_array,
    trial=trial,
    verbose=True
)

# %%
#pip install git-filter-repo


# %%



