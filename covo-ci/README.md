# covo-ci

A unified Full CI (FCI) / Select CI solver for quantum chemistry, built
from determinant strings and explicit Slater-Condon matrix elements
(rather than relying on an external FCI library). It supports two
Hamiltonian integral file formats:

- **aperiodic** — one-electron integrals and a single combined
  two-electron integral file (8-fold permutational symmetry).
- **periodic** — one-electron integrals plus *separate* Coulomb and
  exchange two-electron integral files.

Both modes share the same determinant-generation, spin (`<S^2>`), and
diagonalization machinery, and both support optional **excitation-level
truncation** (Select CI: HF, CIS, CISD, CISDT, ... up to Full CI).

> **Naming note**: "covo" refers to Correlation Optimized Virtual
> Orbitals (COVOs), the orbital framework this code is built around.
> "ci" reflects that this package handles the (Select) CI matrix
> construction and diagonalization stage.
> 
A separate, companion package, [`qflow-suite`](GitHub address), implements the QFlow correlation-energy optimization method, which consumes the `FCI_matrix.dat` file this package can produce via `--save-matrix` (see [Quick start](#quick-start) below).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input file formats](#input-file-formats)
- [Command-line reference](#command-line-reference)
- [Extracting specific spin states from output](#extracting-specific-spin-states-from-output)
- [Example data](#example-data)
- [Repository layout](#repository-layout)
- [Tutorial](docs/TUTORIAL.md)
- [Contributing / running tests](#contributing--running-tests)

## Installation

Requires Python >= 3.9.

```bash
git clone https://github.com/<your-org>/covo-ci.git
cd covo-ci
pip install -e .
```

This installs the `covo-ci` command-line tool along with its two
dependencies, NumPy and SciPy.

To also install development/test dependencies:

```bash
pip install -e ".[dev]"
```

## Quick start

Run a Full CI calculation on the bundled H2/STO-3G example
(aperiodic format):

```bash
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 --excitation-level full --n-roots 4
```

Run a Select CI (CISD, i.e. excitation level 2) calculation on the
periodic-format example:

```bash
covo-ci run --mode periodic --input-dir examples/periodic_h2 --excitation-level 2 --n-roots 5
```

To also save the full CI Hamiltonian matrix to disk (e.g. for
debugging or for use in an external tool), add `--save-matrix`:

```bash
covo-ci run --mode aperiodic --input-dir examples/aperiodic_h2 --excitation-level full --n-roots 4 --save-matrix FCI_matrix.dat
```

Each run prints the energy, `<S^2>`, total spin `S`, and spin
multiplicity of every requested root, e.g.:

```
Lowest CI energies and spin multiplicities:
  state  1:  E = -1.1372838345   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
  state  2:  E = -0.5307733570   <S^2> = 2.00000000   S = 1.000000   multiplicity = 3
  state  3:  E = -0.1683524330   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
  state  4:  E =  0.4831426731   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
```

For a more detailed walkthrough including how to generate your own
input files, see the [full tutorial](docs/TUTORIAL.md).

## Input file formats

Both modes require an `ion_ion.dat` file and a
`one_electron_integrals.dat` file, in a directory you point `--input-dir`
at.

**`ion_ion.dat`** — a single line containing the nuclear (ion-ion)
repulsion energy:

```
0.715104339100
```

**`one_electron_integrals.dat`** — header line `noccp nvirt n_entries`,
followed by `n_entries` lines of `i j value` (1-based spatial orbital
indices, only the upper triangle is needed since the matrix is
symmetrized on read):

```
1 1 1
1 1 -1.253309786646
```

### Aperiodic mode additionally requires

**`two_electron_integrals.dat`** — header line `noccp nvirt n_entries`,
followed by `n_entries` lines of `i j k l value` (1-based spatial
orbital indices, chemist notation `(ij|kl)`; 8-fold permutational
symmetry is applied automatically, so only one representative per
symmetry-equivalence class needs to be listed).

### Periodic mode additionally requires

**`two_electron_integrals_coulomb.dat`** and
**`two_electron_integrals_exchange.dat`** — same header/line format as
above, one file for the Coulomb integral tensor and one for the
exchange integral tensor. These are read and combined according to the
periodic solver's own Coulomb/exchange Slater-Condon formulas; see
[`docs/TUTORIAL.md`](docs/TUTORIAL.md) for details and
`examples/periodic_h2/EXPECTED_OUTPUT.txt` for an important caveat
about what this format does and doesn't validate against.

## Command-line reference

```
covo-ci run --mode {aperiodic,periodic} --input-dir DIR
              [--excitation-level LEVEL] [--n-roots N]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--mode` | yes | — | `aperiodic` or `periodic`, selects the integral file format. |
| `--input-dir` | yes | — | Directory containing the integral files described above. |
| `--excitation-level` | no | `full` | `full` (Full CI), or a non-negative integer: `0`=HF reference only, `1`=CIS, `2`=CISD, `3`=CISDT, etc. |
| `--n-roots` | no | `5` | Number of lowest eigenstates to compute and report. Automatically capped at the determinant-space dimension if smaller. |
| `--save-matrix` | no | not written | If given a file path, writes the full CI Hamiltonian matrix's nonzero elements to that file (1-based `i j value` lines), in addition to running diagonalization. |

```
covo-ci extract --pattern PATTERN [--multiplicity M]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--pattern` | no | `perm-*/fci.out` | Glob pattern for `covo-ci run` output files/logs to scan, e.g. `perm-*/fci.out` or `perm*.log`. |
| `--multiplicity` | no | `1` | Spin multiplicity to extract (1=singlet, 2=doublet, 3=triplet, ...). |

## Extracting specific spin states from output

If you've run `covo-ci run` across many geometries/permutations and
redirected each run's output to a file (e.g.
`perm-1.0/fci.out`, `perm-1.5/fci.out`, ...), you can pull out the
lowest state of a given spin multiplicity from every file at once:

```bash
covo-ci extract --pattern "perm-*/fci.out" --multiplicity 1   # singlets
covo-ci extract --pattern "perm-*/fci.out" --multiplicity 3   # triplets
```

This also works on a flat directory of log files instead of
subdirectories:

```bash
covo-ci extract --pattern "perm*.log" --multiplicity 1
```

## Example data

`examples/aperiodic_h2/` and `examples/periodic_h2/` contain a small
H2/STO-3G (R = 0.74 Angstrom) test case for each mode, along with an
`EXPECTED_OUTPUT.txt` describing the energies you should reproduce by
running the commands in [Quick start](#quick-start). The aperiodic
example's energies are cross-checked against PySCF's own native FCI
solver and match to all 10 reported digits; see
`examples/periodic_h2/EXPECTED_OUTPUT.txt` for why the periodic
example is a *reproducibility* check rather than a physics validation.

## Repository layout

```
covo-ci/
├── src/covo_ci/
│   ├── cli.py                    # `covo-ci` command-line entry point
│   ├── strings.py                # determinant string generation + excitation-level filter
│   ├── spin.py                   # S+, S-, <S^2> machinery (shared)
│   ├── solver.py                 # diagonalization + energy/spin reporting (shared)
│   ├── io_aperiodic.py           # aperiodic integral file readers + spin-orbital expansion
│   ├── io_periodic.py            # periodic integral file readers
│   ├── hamiltonian_aperiodic.py  # Hamiltonian matrix builder, aperiodic format
│   ├── hamiltonian_periodic.py   # Hamiltonian matrix builder, periodic format
│   └── extract.py                # singlet/triplet/... state extraction utility
├── examples/
│   ├── aperiodic_h2/
│   └── periodic_h2/
├── tests/
│   └── test_covo_ci.py
├── docs/
│   └── TUTORIAL.md
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Contributing / running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
```

Pull requests are welcome. Please run the test suite and linter before
submitting.

## License

MIT — see [LICENSE](LICENSE).
