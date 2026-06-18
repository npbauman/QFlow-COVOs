"""
Unified command-line entry point for the FCI / Select-CI solver.

Usage:
    covo-ci run --mode aperiodic --input-dir path/to/data
    covo-ci run --mode periodic  --input-dir path/to/data --excitation-level 2
    covo-ci run --mode periodic  --input-dir path/to/data --excitation-level full
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import io_aperiodic, io_periodic
from .extract import extract_all, MULTIPLICITY_NAMES
from .hamiltonian_aperiodic import build_fci_matrix as build_fci_matrix_aperiodic
from .hamiltonian_periodic import build_fci_matrix as build_fci_matrix_periodic
from .solver import diagonalize_with_spin
from .strings import build_fci_strings_from_half_strings, generate_half_strings

LEVEL_NAMES = {0: "Reference (HF)", 1: "CIS", 2: "CISD", 3: "CISDT", 4: "CISDTQ"}


def save_fci_matrix(matrix: np.ndarray, path: str) -> None:
    """
    Write the full CI matrix to ``path`` as sparse-style "i j value" lines,
    using 1-based (Fortran-style) indexing.

    By default, only nonzero matrix elements are written.
    To write the full matrix, comment out the nonzero-only block
    and uncomment the full-matrix block.
    """
    dim = matrix.shape[0]
    with open(path, "w") as f:
#Write only nonzero matrix elements
#        for i in range(dim):
#            for j in range(dim):
#                val = matrix[i, j]
#                if val != 0.0:
#                    f.write(f"{i + 1:6d} {j + 1:6d} {val:20.12e}\n")
#Write full matrix
        for i in range(dim):
            for j in range(dim):
                val = matrix[i,j]
                f.write(f"{i + 1:6d} {j + 1:6d} {val:20.12e}\n")
 
    print(f"Wrote {dim}x{dim} CI matrix ({np.count_nonzero(matrix)} elements) to {path}")


def _parse_excitation_level(value: str):
    """Parse the --excitation-level argument: an int, or 'full'/'fci'."""
    if value is None:
        return None
    if value.lower() in ("full", "fci"):
        return None
    try:
        level = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid excitation level: {value!r}. Use a non-negative integer, or 'full'."
        ) from exc
    if level < 0:
        raise argparse.ArgumentTypeError("Excitation level must be >= 0.")
    return level


def _describe_level(level):
    if level is None:
        return "Full CI (no excitation limit)"
    name = LEVEL_NAMES.get(level, f"CI with max {level} excitations")
    return f"Select CI with max excitation level {level} ({name})"


def run_aperiodic(input_dir: str, excitation_level, n_roots: int, save_matrix_path: str | None = None):
    data = io_aperiodic.load_aperiodic_hamiltonian(input_dir)
    n_occ, n_virt = data["n_occ"], data["n_virt"]

    print(f"One-/two-electron integrals: {n_occ} occupied, {n_virt} virtual spatial orbitals")
    print(f"Ion-ion repulsion: {data['repulsion']:.8f}")

    half_str_a = generate_half_strings(n_occ + n_virt, n_occ, excitation_level)
    half_str_b = generate_half_strings(n_occ + n_virt, n_occ, excitation_level)
    half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
    half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]

    fci_strings = build_fci_strings_from_half_strings(
        half_str_a, half_str_b, n_occ, n_virt, n_occ, n_virt, excitation_level
    )
    print(f"Number of determinants: {fci_strings.shape[0]}")

    matrix = build_fci_matrix_aperiodic(fci_strings, data["soei"], data["atei"], n_occ, n_virt, data["repulsion"])

    if save_matrix_path is not None:
        save_fci_matrix(matrix, save_matrix_path)

    return diagonalize_with_spin(matrix, fci_strings, n_occ, n_virt, k=n_roots)


def run_periodic(input_dir: str, excitation_level, n_roots: int, save_matrix_path: str | None = None):
    data = io_periodic.load_periodic_hamiltonian(input_dir)
    noccp, nvirt, norb = data["noccp"], data["nvirt"], data["norb"]

    print(f"One-/two-electron integrals: {noccp} occupied, {nvirt} virtual spatial orbitals")
    print(f"Ion-ion repulsion: {data['repulsion']:.8f}")

    half_str_a = generate_half_strings(norb, noccp, excitation_level)
    half_str_b = generate_half_strings(norb, noccp, excitation_level)
    half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
    half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]

    noas = nobs = noccp
    nvas = nvbs = nvirt
    fci_strings = build_fci_strings_from_half_strings(
        half_str_a, half_str_b, noas, nvas, nobs, nvbs, excitation_level
    )
    print(f"Number of determinants: {fci_strings.shape[0]}")

    spinorbital_map = io_periodic.build_spinorbital_map(noccp, nvirt)
    matrix = build_fci_matrix_periodic(
        fci_strings, data["h"], data["vc"], data["vx"], noccp, nvirt, noas, nobs, data["repulsion"], spinorbital_map
    )

    if save_matrix_path is not None:
        save_fci_matrix(matrix, save_matrix_path)

    return diagonalize_with_spin(matrix, fci_strings, noccp, nvirt, k=n_roots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="covo-ci",
        description="Full CI / Select CI solver for aperiodic and periodic integral file formats.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an FCI or Select-CI calculation.")
    run_parser.add_argument(
        "--mode",
        choices=["aperiodic", "periodic"],
        required=True,
        help="Integral file format: 'aperiodic' (single combined two-electron integral file) "
             "or 'periodic' (separate Coulomb/exchange two-electron integral files).",
    )
    run_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the integral files (ion_ion.dat, one_electron_integrals.dat, "
             "and the mode-appropriate two-electron integral file(s)).",
    )
    run_parser.add_argument(
        "--excitation-level",
        default="full",
        type=str,
        help="Maximum excitation level relative to the Hartree-Fock reference determinant: "
             "an integer (0=HF, 1=CIS, 2=CISD, 3=CISDT, ...), or 'full' for Full CI. Default: full.",
    )
    run_parser.add_argument(
        "--n-roots",
        default=5,
        type=int,
        help="Number of lowest eigenstates to compute and report. Default: 5.",
    )
    run_parser.add_argument(
        "--save-matrix",
        default=None,
        metavar="PATH",
        help="If given, write the full CI Hamiltonian matrix's nonzero elements to PATH "
             "(1-based 'i j value' lines), in addition to running diagonalization. "
             "Not written by default.",
    )

    extract_parser = subparsers.add_parser(
        "extract", help="Extract the lowest state of a given spin multiplicity from output files."
    )
    extract_parser.add_argument(
        "--pattern",
        default="perm-*/fci.out",
        help="Glob pattern for output files, e.g. 'perm-*/fci.out' or 'perm*.log'. Default: 'perm-*/fci.out'.",
    )
    extract_parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity to extract (1=singlet, 2=doublet, 3=triplet, ...). Default: 1.",
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            excitation_level = _parse_excitation_level(args.excitation_level)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
            return 2

        print(f"Mode: {args.mode}")
        print(_describe_level(excitation_level))
        print(f"Reading Hamiltonian from directory: {args.input_dir}")

        if args.mode == "aperiodic":
            run_aperiodic(args.input_dir, excitation_level, args.n_roots, save_matrix_path=args.save_matrix)
        else:
            run_periodic(args.input_dir, excitation_level, args.n_roots, save_matrix_path=args.save_matrix)

        return 0

    if args.command == "extract":
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
