# Tutorial: running covo-ci from scratch

This tutorial walks through installing covo-ci, understanding the
two supported input formats, running Full CI and Select CI
calculations, interpreting the output, and extracting specific spin
states across a batch of calculations. It assumes basic familiarity
with Python and the command line, but no prior familiarity with this
codebase.

## 1. Background: what this code does

Given a set of one- and two-electron integrals (in some one-particle
orbital basis) and a nuclear repulsion energy, this code:

1. Generates all Slater determinants for a given number of electrons
   and orbitals (optionally truncated to a maximum excitation level
   relative to the Hartree-Fock reference determinant — this is the
   Select CI feature).
2. Builds the full CI Hamiltonian matrix in that determinant basis,
   using explicit Slater-Condon rules for the diagonal, single-, and
   double-excitation matrix elements.
3. Diagonalizes the matrix (using a dense solver for small systems
   and `scipy.sparse.linalg.eigsh` for larger ones) to get the lowest
   `--n-roots` eigenstates.
4. For each eigenstate, computes `<S^2>` and reports the total spin
   `S` and spin multiplicity `2S+1`, which is how you'd identify e.g.
   "the first singlet state" or "the lowest triplet."

The orbitals are assumed to come from a restricted (same spatial
orbitals for alpha and beta spin) reference, and the determinant
ordering convention used internally throughout the codebase is:

```
[ alpha occupied | beta occupied | alpha virtual | beta virtual ]
```

You don't need to construct this ordering yourself — it's handled
internally — but it's useful to know if you ever need to debug a
matrix element by hand.

## 2. Installing

```bash
git clone https://github.com/<your-org>/covo-ci.git
cd covo-ci
python3 -m venv .venv          # optional, but recommended
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the install:

```bash
covo-ci --help
```

You should see usage information for the `run` and `extract`
subcommands.

## 3. Two input formats: which one do I need?

You have two options depending on how your one- and two-electron
integrals were generated:

- **aperiodic**: if your two-electron integrals come as a single file
  with `(ij|kl)` chemist-notation values and you haven't separately
  computed Coulomb vs. exchange tensors, use `--mode aperiodic`.
- **periodic**: if your integral-generation pipeline produces
  *separate* Coulomb and exchange two-electron integral files (common
  in periodic/solid-state electronic structure workflows), use
  `--mode periodic`.

Both modes need the same `ion_ion.dat` and
`one_electron_integrals.dat` file formats (see the
[README](../README.md#input-file-formats) for exact specifications);
they differ only in the two-electron integral file(s).

> **Important**: the periodic mode's Slater-Condon formulas are
> written for the in-house integral-generation convention this code
> was originally developed against. If you're plugging in integrals
> from a different source for periodic mode, validate against a
> system with a known answer before trusting production results —
> see `examples/periodic_h2/EXPECTED_OUTPUT.txt` for more on this
> caveat.

## 4. Your first calculation: H2 (aperiodic)

The repository ships a tiny worked example for both modes:
`examples/aperiodic_h2/` and `examples/periodic_h2/`, both for H2 at
0.74 Angstrom in an STO-3G basis. Let's run Full CI on the aperiodic
one:

```bash
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 \
              --excitation-level full --n-roots 4
```

Expected output (abbreviated):

```
Mode: aperiodic
Full CI (no excitation limit)
Reading Hamiltonian from directory: examples/aperiodic_h2
One-/two-electron integrals: 1 occupied, 1 virtual spatial orbitals
Ion-ion repulsion: 0.71510434
Number of determinants: 4
Dimension of CI matrix: 4
Lowest CI energies and spin multiplicities:
  state  1:  E = -1.1372838345   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
  state  2:  E = -0.5307733570   <S^2> = 2.00000000   S = 1.000000   multiplicity = 3
  state  3:  E = -0.1683524330   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
  state  4:  E =  0.4831426731   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
```

These four numbers are the exact ground and excited state energies of
H2/STO-3G — they match PySCF's own native FCI solver to all 10
reported digits. If you reproduce these numbers, your installation is
working correctly.

Reading the output: state 1 is the singlet ground state, state 2 is
the lowest triplet, states 3 and 4 are higher singlets. The
`multiplicity` column is exactly what you'd scan for if you wanted
"the lowest triplet energy", for example.

## 5. Restricting the determinant space: Select CI

Full CI scales combinatorially with system size, so for anything
beyond a handful of orbitals you'll want to truncate the determinant
space by excitation level relative to the Hartree-Fock reference. Use
`--excitation-level`:

```bash
# Hartree-Fock only (single reference determinant, no correlation)
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 --excitation-level 0 --n-roots 1

# CIS: singles only
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 --excitation-level 1 --n-roots 1

# CISD: singles + doubles
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 --excitation-level 2 --n-roots 1

# Full CI (equivalent to a sufficiently high integer, or 'full')
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 --excitation-level full --n-roots 1
```

For this minimal H2/STO-3G case (only 1 occupied + 1 virtual spatial
orbital per spin), level 2 already equals Full CI, so you won't see a
difference past CISD here — but for larger active spaces, each
increasing level adds more determinants and the ground-state energy
will monotonically decrease (variational principle) as you approach
Full CI.

A quick sanity check worth doing on your own systems: confirm that
`--excitation-level 0` gives the same energy as a separate
Hartree-Fock calculation, and that increasing the level never
*raises* the ground-state energy.

## 6. Running the periodic-format example

Same idea, different integral files:

```bash
covo-ci run --mode periodic --input-dir examples/periodic_h2 \
              --excitation-level full --n-roots 4
```

See `examples/periodic_h2/EXPECTED_OUTPUT.txt` for the numbers you
should reproduce, and read the caveat there about what this example
does and doesn't validate.

## 7. Saving the full CI matrix to disk

By default, `covo-ci run` only prints the summary energy/spin table
— it does not print or save the full Hamiltonian matrix, since for
anything beyond a toy system that matrix is far too large to be
useful on screen. If you need the matrix itself (e.g. to inspect
specific elements, or feed it into another tool), pass
`--save-matrix PATH`:

```bash
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 \
              --excitation-level full --n-roots 4 \
              --save-matrix /tmp/h2_fci_matrix.dat
```

This writes one line per *nonzero* matrix element, as
`i j value` using 1-based indexing:

```
     1      1  -1.116759307397e+00
     1      4   1.812104620150e-01
     2      2  -3.495628949860e-01
     ...
```

The matrix is symmetric, so both `(i, j)` and `(j, i)` are written.
Only nonzero elements are included to keep the file size manageable —
for a Full CI calculation on anything but the smallest systems, the
matrix is mostly sparse anyway.

## 8. Bringing your own data

Create a directory containing:

- `ion_ion.dat`
- `one_electron_integrals.dat`
- *(aperiodic)* `two_electron_integrals.dat`, **or**
- *(periodic)* `two_electron_integrals_coulomb.dat` and
  `two_electron_integrals_exchange.dat`

following the formats described in the
[README](../README.md#input-file-formats), then point `--input-dir`
at it. There's nothing else to configure — orbital counts, electron
counts, etc. are all read directly from the header lines of your
integral files.

A common workflow is to loop over many geometries/permutations, each
in its own subdirectory, and redirect each run's output to a file:

```bash
for geom_dir in perm-*/; do
    covo-ci run --mode periodic --input-dir "$geom_dir" \
                  --excitation-level 2 --n-roots 5 > "${geom_dir}/fci.out"
done
```

## 9. Extracting specific spin states across many runs

Once you have a batch of output files like the above, use
`covo-ci extract` to pull out (for example) the lowest singlet from
every geometry in one shot:

```bash
covo-ci extract --pattern "perm-*/fci.out" --multiplicity 1
```

```
perm-1.0/fci.out: first singlet -> state 1, E = -1.1372838345
perm-1.5/fci.out: first singlet -> state 1, E = -1.1372838345
perm-2.0/fci.out: first singlet -> state 1, E = -1.1372838345
```

Output files are sorted numerically by any `perm-N` value found in
the path, so the results come back in geometry order even if your
shell's glob expansion wouldn't sort them that way (e.g. `perm-10`
sorting after rather than before `perm-2`).

To get triplets instead:

```bash
covo-ci extract --pattern "perm-*/fci.out" --multiplicity 3
```

If your output files are flat log files instead of subdirectories
(e.g. `perm1.0.log`, `perm1.5.log`, ...), just change the pattern:

```bash
covo-ci extract --pattern "perm*.log" --multiplicity 1
```

## 10. Troubleshooting

**"Inconsistent orbital counts between one- and two-electron integral
files" (aperiodic mode)** — the `noccp`/`nvirt` header values in
`one_electron_integrals.dat` and `two_electron_integrals.dat` don't
match. Check that both files were generated from the same calculation.

**"Occupancy mismatch" / "Occupied orbital count mismatch"** — usually
indicates a malformed or truncated determinant string, often from a
bug in custom code calling the lower-level functions directly rather
than through `covo-ci run`. If you're using the CLI as documented,
this shouldn't occur; please open an issue with your input files if
it does.

**Energies don't match a reference calculation** — for aperiodic mode,
first reproduce the bundled H2 example exactly (Section 4); if that
matches but your real system doesn't, the most likely culprit is the
integral file convention (chemist vs. physicist notation, or an
indexing/symmetry mismatch) rather than the solver itself. For
periodic mode, see the caveat in Section 3 — this mode is not meant
to be checked against generic molecular FCI codes without first
confirming the Coulomb/exchange convention used by your own integral
generator matches what `hamiltonian_periodic.py` expects.

**Calculation is too slow / runs out of memory** — Full CI is
combinatorial in cost. Use `--excitation-level` to truncate the
determinant space (Section 5), and note that the matrix-build step is
pure Python/NumPy (not optimized for very large determinant counts);
this codebase is intended for small-to-moderate active spaces and
method development/validation rather than production-scale FCI.

## 11. Next steps

- See the [README](../README.md) for the full command-line and file
  format reference.
- See `tests/test_covo_ci.py` for additional worked examples of
  calling the underlying Python API directly (`run_aperiodic`,
  `run_periodic`) rather than through the CLI, if you want to
  integrate this into a larger Python workflow.
