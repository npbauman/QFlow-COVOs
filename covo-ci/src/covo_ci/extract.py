"""
Extract the lowest-energy state of a given spin multiplicity from one or
more FCI/Select-CI output files (as printed by ``covo-ci run``).

Supports two common output layouts seen in practice:
  - perm-*/fci.out   (one subdirectory per geometry/permutation)
  - perm*.log        (one log file per geometry/permutation, flat directory)

Usage:
    covo-ci extract --pattern "perm-*/fci.out" --multiplicity 1
    covo-ci extract --pattern "perm*.log" --multiplicity 3
"""
from __future__ import annotations

import argparse
import glob
import os
import re

# Example line this pattern matches:
#   state  4:  E = -7.3756009471   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
STATE_LINE_PATTERN = re.compile(
    r"state\s+(\d+):\s+E\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)"
    r".*?multiplicity\s*=\s*(\d+)"
)

SECTION_HEADER = "Lowest"  # matches both "Lowest FCI energies..." and "Lowest CI energies..."


def extract_first_state_of_multiplicity(out_path: str, multiplicity: int = 1):
    """
    Parse an FCI/Select-CI output file and return the first state matching
    ``multiplicity`` found in the "Lowest ... energies and spin
    multiplicities:" section.

    Returns
    -------
    dict with keys {state, energy, line}, or None if not found / file missing.
    """
    if not os.path.exists(out_path):
        return None

    with open(out_path, "r") as f:
        lines = f.readlines()

    in_section = False
    for line in lines:
        if SECTION_HEADER in line and "energies and spin multiplicities" in line:
            in_section = True
            continue

        if in_section:
            match = STATE_LINE_PATTERN.search(line)
            if match:
                state_idx = int(match.group(1))
                energy = float(match.group(2))
                mult = int(match.group(3))
                if mult == multiplicity:
                    return {"state": state_idx, "energy": energy, "line": line.strip()}

    return None


def _sort_key(path: str):
    """Sort paths numerically by any embedded perm-N value, falling back
    to alphabetical order."""
    base = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    match = re.search(r"perm-?([0-9]+(?:\.[0-9]+)?)", base)
    return (float(match.group(1)), path) if match else (float("inf"), path)


def extract_all(pattern: str, multiplicity: int = 1):
    """
    Find all files matching ``pattern`` (a glob pattern, e.g.
    "perm-*/fci.out" or "perm*.log") and extract the first state of the
    requested multiplicity from each.

    Returns a list of (path, result_dict_or_None) tuples, sorted by any
    numeric perm-N index found in the path.
    """
    paths = sorted(glob.glob(pattern), key=_sort_key)
    results = []
    for path in paths:
        result = extract_first_state_of_multiplicity(path, multiplicity)
        results.append((path, result))
    return results


MULTIPLICITY_NAMES = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="covo-ci-extract",
        description="Extract the lowest state of a given spin multiplicity from FCI/Select-CI output files.",
    )
    parser.add_argument(
        "--pattern",
        default="perm-*/fci.out",
        help="Glob pattern for output files, e.g. 'perm-*/fci.out' or 'perm*.log'. Default: 'perm-*/fci.out'.",
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity to extract (1=singlet, 2=doublet, 3=triplet, ...). Default: 1.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    label = MULTIPLICITY_NAMES.get(args.multiplicity, f"multiplicity-{args.multiplicity}")
    results = extract_all(args.pattern, args.multiplicity)

    if not results:
        print(f"No files matched pattern: {args.pattern}")
        return 1

    for path, result in results:
        if result is None:
            if not os.path.exists(path):
                print(f"{path}: file not found")
            else:
                print(f"{path}: no {label} state found")
        else:
            print(f"{path}: first {label} -> state {result['state']}, E = {result['energy']:.10f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
