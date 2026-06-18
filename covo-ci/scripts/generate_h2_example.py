"""
Generate small H2/STO-3G example Hamiltonian files (both aperiodic and
periodic formats) for the covo-ci test/tutorial examples, using PySCF
to compute the integrals and as a ground-truth FCI energy check.
"""
import os

import numpy as np
from pyscf import gto, scf, fci, ao2mo

OUT_APERIODIC = "examples/aperiodic_h2"
OUT_PERIODIC = "examples/periodic_h2"

os.makedirs(OUT_APERIODIC, exist_ok=True)
os.makedirs(OUT_PERIODIC, exist_ok=True)

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

n_orb = mf.mo_coeff.shape[1]
n_occ = mol.nelectron // 2
n_virt = n_orb - n_occ

h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
h1_mo = mf.mo_coeff.T @ h1_ao @ mf.mo_coeff

eri_ao = mol.intor("int2e")
eri_mo = ao2mo.full(eri_ao, mf.mo_coeff, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
# eri_mo is in chemist notation (pq|rs) = integral over p(1)q(1) 1/r12 r(2)s(2)

repulsion = mol.energy_nuc()

print(f"n_orb={n_orb} n_occ={n_occ} n_virt={n_virt}")
print(f"Nuclear repulsion: {repulsion:.10f}")

# ---- Reference: PySCF native FCI ----
cisolver = fci.FCI(mf)
cisolver.nroots = 4
e_fci, civecs = cisolver.kernel()
print("PySCF FCI energies:", e_fci)

with open(os.path.join(OUT_APERIODIC, "pyscf_reference.txt"), "w") as f:
    f.write("PySCF native FCI energies (Hartree), H2/STO-3G, R=0.74 Angstrom:\n")
    for i, e in enumerate(np.atleast_1d(e_fci)):
        f.write(f"  state {i + 1}: {e:.10f}\n")
with open(os.path.join(OUT_PERIODIC, "pyscf_reference.txt"), "w") as f:
    f.write("PySCF native FCI energies (Hartree), H2/STO-3G, R=0.74 Angstrom:\n")
    for i, e in enumerate(np.atleast_1d(e_fci)):
        f.write(f"  state {i + 1}: {e:.10f}\n")

# ============================================================
# Write ion_ion.dat (shared format)
# ============================================================
for out_dir in (OUT_APERIODIC, OUT_PERIODIC):
    with open(os.path.join(out_dir, "ion_ion.dat"), "w") as f:
        f.write(f"{repulsion:.12f}\n")

# ============================================================
# Write one_electron_integrals.dat (shared format)
# Format: noccp nvirt n_entries
#         i j value   (1-based, i<=j, upper triangle only)
# ============================================================
entries_h1 = []
for i in range(n_orb):
    for j in range(i, n_orb):
        val = h1_mo[i, j]
        if abs(val) > 1e-14:
            entries_h1.append((i + 1, j + 1, val))

for out_dir in (OUT_APERIODIC, OUT_PERIODIC):
    with open(os.path.join(out_dir, "one_electron_integrals.dat"), "w") as f:
        f.write(f"{n_occ} {n_virt} {len(entries_h1)}\n")
        for i, j, val in entries_h1:
            f.write(f"{i} {j} {val:.12f}\n")

# ============================================================
# Aperiodic format: single combined two-electron integral file,
# physicist-notation-compatible storage consistent with
# io_aperiodic.read_two_electron_integrals's 8-fold symmetry fill
# (i j | k l) chemist notation, matching the original aperiodic_fci.py
# convention where two_body[i,j,k,l] is later used as <ij|kl>-style
# Coulomb integrals in build_spin_orbital_integrals.
# ============================================================
seen = set()
entries_h2_combined = []
for i in range(n_orb):
    for j in range(n_orb):
        for k in range(n_orb):
            for l in range(n_orb):
                key = tuple(sorted([
                    (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                    (k, l, i, j), (k, l, j, i), (l, k, i, j), (l, k, j, i),
                ]))
                if key in seen:
                    continue
                seen.add(key)
                val = eri_mo[i, j, k, l]
                if abs(val) > 1e-14:
                    entries_h2_combined.append((i + 1, j + 1, k + 1, l + 1, val))

with open(os.path.join(OUT_APERIODIC, "two_electron_integrals.dat"), "w") as f:
    f.write(f"{n_occ} {n_virt} {len(entries_h2_combined)}\n")
    for i, j, k, l, val in entries_h2_combined:
        f.write(f"{i} {j} {k} {l} {val:.12f}\n")

# ============================================================
# Periodic format: separate Coulomb (i j | k l) = eri_mo[i,j,k,l] and
# exchange tensors. Given the periodic matrix-element formulas treat
# vc and vx as numerically equal chemist-notation ERI tensors (vc and
# vx differ only in which contraction pattern they're used for), we
# write the same physical ERI tensor into both files. This matches
# the original periodic_fci_serial.py usage where the same MO-basis
# eri tensor format is consumed by both vc and vx, with the J/K
# distinction handled entirely by the index contraction pattern in
# the matrix-element formulas, not by the values stored on disk.
# ============================================================
with open(os.path.join(OUT_PERIODIC, "two_electron_integrals_coulomb.dat"), "w") as f:
    f.write(f"{n_occ} {n_virt} {len(entries_h2_combined)}\n")
    for i, j, k, l, val in entries_h2_combined:
        f.write(f"{i} {j} {k} {l} {val:.12f}\n")

with open(os.path.join(OUT_PERIODIC, "two_electron_integrals_exchange.dat"), "w") as f:
    f.write(f"{n_occ} {n_virt} {len(entries_h2_combined)}\n")
    for i, j, k, l, val in entries_h2_combined:
        f.write(f"{i} {j} {k} {l} {val:.12f}\n")

print("Done writing example files.")
