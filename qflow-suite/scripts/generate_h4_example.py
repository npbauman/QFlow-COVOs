"""
Generate the bundled H4/STO-3G example data for qflow-suite tests and
the tutorial. Requires covo-ci (for FCI matrix generation) and pyscf
(for the integral files) to be installed.

Run from the repository root:
    python3 scripts/generate_h4_example.py
"""
import os

from pyscf import gto, scf, ao2mo

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "h4_sd")
os.makedirs(OUT, exist_ok=True)

mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", unit="Angstrom", verbose=0)
mf = scf.RHF(mol)
mf.kernel()

n_orb = mf.mo_coeff.shape[1]
n_occ = mol.nelectron // 2
n_virt = n_orb - n_occ
print(f"n_orb={n_orb} n_occ={n_occ} n_virt={n_virt}")

h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
h1_mo = mf.mo_coeff.T @ h1_ao @ mf.mo_coeff

eri_ao = mol.intor("int2e")
eri_mo = ao2mo.full(eri_ao, mf.mo_coeff, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
repulsion = mol.energy_nuc()

with open(os.path.join(OUT, "ion_ion.dat"), "w") as f:
    f.write(f"{repulsion:.12f}\n")

entries_h1 = []
for i in range(n_orb):
    for j in range(i, n_orb):
        val = h1_mo[i, j]
        if abs(val) > 1e-14:
            entries_h1.append((i + 1, j + 1, val))
with open(os.path.join(OUT, "one_electron_integrals.dat"), "w") as f:
    f.write(f"{n_occ} {n_virt} {len(entries_h1)}\n")
    for i, j, val in entries_h1:
        f.write(f"{i} {j} {val:.12f}\n")

seen = set()
entries_h2 = []
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
                    entries_h2.append((i + 1, j + 1, k + 1, l + 1, val))
with open(os.path.join(OUT, "two_electron_integrals.dat"), "w") as f:
    f.write(f"{n_occ} {n_virt} {len(entries_h2)}\n")
    for i, j, k, l, val in entries_h2:
        f.write(f"{i} {j} {k} {l} {val:.12f}\n")

print("Done writing H4 example integral files.")
print()
print("Next: run covo-ci to generate FCI_matrix.dat, e.g.:")
print(f"  cd {OUT}")
print("  covo-ci run --mode aperiodic --input-dir . --excitation-level full --n-roots 1 --save-matrix FCI_matrix.dat")
print()
print("The active-space definition is generated automatically by 'qflow-suite run' "
      "every time it's called -- there is no separate file to generate here.")
