# Legacy scripts

These are the original, unrefactored standalone scripts this package
was built from, kept here for provenance and reference only. They are
not maintained, not tested, and not part of the installable
`fci_suite` package — use the `fci-suite` CLI documented in the
top-level README instead.

| Original script | Superseded by |
|---|---|
| `aperiodic_fci.py` | `fci-suite run --mode aperiodic` |
| `periodic_fci_serial.py` | `fci-suite run --mode periodic --excitation-level full` |
| `Select-CI.py` | `fci-suite run --mode periodic --excitation-level N` |
| `extract_log_files.py` (orig. `extract.py`) | `fci-suite extract --pattern "perm*.log"` |
| `extract_singlet_perm_dirs.py` (orig. `extract_singlet.py`) | `fci-suite extract --pattern "perm-*/fci.out"` |
