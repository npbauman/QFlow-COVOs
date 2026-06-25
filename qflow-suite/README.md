# qflow-suite

**qflow-suite** implements QFlow, an iterative similarity-transformed
active-space method for recovering electron correlation energy from
an exact (or Select-CI truncated) Hamiltonian matrix. It refines
cluster amplitudes one active space at a time, using a Taylor-expanded
similarity transformation and a Praxis-style local minimizer at each
step, and provides two truncation levels:

Two variants:
QFlow-SD - T1 and T2 (singles + doubles)
QFlow-SDTQ - T1–T4 (singles, doubles, triples, quadruples)

The active-space definition is generated automatically in memory for
every run.

> **Companion package**: qflow-suite consumes the `FCI_matrix.dat`-style
> output of [`covo-ci`](https://github.com/npbauman/QFlow-COVOs/tree/main/covo-ci) (run
> `covo-ci run ... --save-matrix FCI_matrix.dat`). The two packages are
> independent and separately installable — qflow-suite depends only on
> the file format covo-ci writes, not on its code.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Input format](#input-format)
- [Command-line reference](#command-line-reference)
- [Choosing SD vs. SDTQ](#choosing-sd-vs-sdtq)
- [Example](#example)
- [Repository layout](#repository-layout)
- [Provenance and design notes](#provenance-and-design-notes)
- [Development](#development)
- [License](#license)

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/npbauman/QFlow-COVOs/tree/main/qflow-suite.git
cd qflow-suite
pip install -e .
```

This installs the `qflow-suite` command and its dependencies: NumPy,
SciPy, [Numba](https://numba.pydata.org/) (JIT-accelerated inner
loops), and [joblib](https://joblib.readthedocs.io/) (parallel
excitation-pair precomputation in SDTQ mode).

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Quick start

Produce an FCI matrix with [covo-ci](https://github.com/npbauman/QFlow-COVOs/tree/main/covo-ci):

```bash
covo-ci run --mode aperiodic --input-dir path/to/data \
    --excitation-level full --n-roots 1 --save-matrix FCI_matrix.dat
```

Then run QFlow:

```bash
# Singles + doubles
qflow-suite run --level sd --n-elec 4 --norb 4 --matrix-file FCI_matrix.dat

# Singles, doubles, triples, quadruples
qflow-suite run --level sdtq --n-elec 4 --norb 4 --matrix-file FCI_matrix.dat
```

Each run prints the optimized energy for every active space at 
every global iteration, then the final converged QFlow energy. The number of global
iterations defaults to 2; pass
`--max-iter N` to run more or fewer:

```bash
qflow-suite run --level sd --n-elec 4 --norb 4 --matrix-file FCI_matrix.dat --max-iter 5
```

By default a copy of the active-space definition actually used is
logged to `nactspin_fortran.dat` for reference (it's regenerated and
overwritten every run, and never read back — pass `--no-actspin-log`
to skip writing it). 
To keep that log file and QFlow-SD's amplitude
checkpoints out of your working directory, use `--output-dir`:

```bash
qflow-suite run --level sd --n-elec 4 --norb 4 \
    --matrix-file FCI_matrix.dat --output-dir results/
```

See [Example](#example) for a complete, runnable walkthrough.

## Input format

The only required input is **`FCI_matrix.dat`**: the Hamiltonian
matrix as nonzero `i j value` lines, 1-based indexing — exactly
covo-ci's `--save-matrix` output.

```
     1      1  -1.253309786646e+00
     1      4   1.812104620150e-01
     2      2  -3.495628949860e-01
     ...
```

The active-space definition is derived purely from `--n-elec` and
`--norb`: 2 active occupied and 2 active virtual spin-orbitals are
chosen at a time, for every combination (`nactives = C(noas, 2) ×
C(nvas, 2)`), mirrored identically across alpha and beta spin
channels. Orbitals are assumed indexed in increasing-energy order, and
active spaces are generated starting from the HOMO–LUMO pair through
to the lowest-occupied/highest-virtual pair. This generation logic
reproduces a reference Fortran implementation exactly (see
[Provenance and design notes](#provenance-and-design-notes)), and
active-space *order* has no effect on correctness — every active space
is optimized independently regardless of where it falls in the
sequence.

## Command-line reference

```
qflow-suite run --level {sd,sdtq} --n-elec N --norb M
                [--matrix-file PATH] [--max-iter N] [--ckpt-dir DIR]
                [--output-dir DIR] [--no-actspin-log] [--quiet]
```

| Flag | Required | Default | Description |
|---|:---:|---|---|
| `--level` | ✓ | — | `sd` or `sdtq`. |
| `--n-elec` | ✓ | — | Total electron count (closed-shell, must be even). |
| `--norb` | ✓ | — | Number of spatial orbitals. |
| `--matrix-file` | | `FCI_matrix.dat` | Path to the FCI Hamiltonian matrix. Unaffected by `--output-dir`. |
| `--max-iter` | | `2` | Maximum number of global QFlow iterations (each iteration optimizes every active space once). Must be ≥ 1. |
| `--ckpt-dir` | | `amplitudes_ckpt` | *(`sd` only)* Directory for per-iteration T1/T2 checkpoints. Placed inside `--output-dir` if that's given and this isn't set explicitly. |
| `--output-dir` | | current directory | Place the active-space log and `--ckpt-dir`'s default inside this directory. Created if missing. Never overrides an explicitly-given `--ckpt-dir`. |
| `--no-actspin-log` | | off | Skip writing the active-space definition log file entirely. |
| `--quiet` | | off | Suppress per-iteration progress output. |

## Choosing SD vs. SDTQ

QFlow-SD optimizes only T1/T2 amplitudes — cheap, but its converged
energy sits above the exact result by however much triples/quadruples
correlation it omits. QFlow-SDTQ includes T3/T4 and recovers the exact
energy for systems small enough that the active-space hierarchy spans
the full correlation space (demonstrated in
[`examples/h4_sd/EXPECTED_OUTPUT.txt`](examples/h4_sd/EXPECTED_OUTPUT.txt)),
at substantially higher cost. A practical workflow: use SD for
exploration on full-size systems, and SDTQ on smaller representative
subsystems to quantify how much truncation error SD is leaving out.

## Example

[`examples/h4_sd/`](examples/h4_sd/) contains a complete H4/STO-3G
system (generated by [`scripts/generate_h4_example.py`](scripts/generate_h4_example.py))
with a pre-computed `FCI_matrix.dat`. It's a minimal active-space case
(exactly one active space spans the whole system), so it's primarily a
correctness/connectivity check rather than a demonstration of
multi-active-space behavior: QFlow-SDTQconverges to within `~5e-8` Hartree 
of the exact FCI energy on this system, confirming the full covo-ci → qflow-suite pipeline end to end.

```bash
cd examples/h4_sd
qflow-suite run --level sd   --n-elec 4 --norb 4 --matrix-file FCI_matrix.dat
qflow-suite run --level sdtq --n-elec 4 --norb 4 --matrix-file FCI_matrix.dat
```

Expected energies are documented in
[`EXPECTED_OUTPUT.txt`](examples/h4_sd/EXPECTED_OUTPUT.txt).

## Repository layout

```
qflow-suite/
├── src/qflow_suite/
│   ├── cli.py              # `qflow-suite` command-line entry point
│   ├── common.py           # logic shared between SD and SDTQ (string/active-space generation)
│   ├── algorithm_sd.py     # QFlow-SD numerical core (vendored, see note below)
│   └── algorithm_sdtq.py   # QFlow-SDTQ numerical core (vendored, see note below)
├── examples/h4_sd/          # runnable H4/STO-3G example + expected output
├── scripts/generate_h4_example.py
├── tests/test_qflow_suite.py
├── legacy/                  # original, unrefactored reference scripts
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Development

```bash
pip install -e ".[dev]"
pytest                       # full suite, including slow end-to-end runs
pytest -m "not slow"         # fast checks only (sub-second)
ruff check src/
```

The test suite (16 tests) covers active-space generation correctness
(including an exact match against reference Fortran output), the FCI
matrix reader, and full end-to-end QFlow-SD/SDTQ runs against the
bundled H4 example, validated against its known FCI energy.

## License

[MIT](LICENSE)
