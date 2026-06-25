# Legacy scripts

Original, unrefactored standalone scripts this package was built from,
kept for provenance and reference only. Not maintained, not tested,
not part of the installable `qflow_suite` package.

| Original script | Superseded by |
|---|---|
| `SD-NoTQ.py` (QFlow with singles + doubles only) | `qflow-suite run --level sd` |
| `serial_python.py` (QFlow with singles, doubles, triples, quadruples) | `qflow-suite run --level sdtq` |

Both originals are full top-to-bottom procedural scripts: hardcoded
`n_elec`/`norb` at the top, module-level code that reads
`FCI_matrix.dat` and `nactspin_fortran.dat` on import, and a
`run_qflow_loop` driver at the bottom. The refactor turns this
top-level script code into parameterized functions and a CLI, without
changing any of the internal numerical algorithm functions (active
space construction, T-amplitude bookkeeping, similarity
transformation, Powell/PRAXIS-style minimization, etc.) — those are
preserved exactly as originally written.
