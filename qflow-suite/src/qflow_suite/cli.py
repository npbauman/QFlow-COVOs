"""
Unified command-line entry point for qflow-suite.

Wraps the original QFlow-SD (``SD-NoTQ.py``) and QFlow-SDTQ
(``serial_python.py``) driver scripts as parameterized functions, with
no changes to the underlying numerical algorithms. Both read the
``FCI_matrix.dat``-style file produced by covo-ci's
``--save-matrix`` flag. The active-space definition is always
generated automatically in memory for the given --n-elec/--norb (see
:func:`qflow_suite.common.generate_nactspin`) -- there is no manual
active-space file to supply or keep in sync, so a stale or
mismatched active-space file can never be accidentally reused.

Usage:
    qflow-suite run --level sd   --n-elec 6 --norb 9 --matrix-file FCI_matrix.dat
    qflow-suite run --level sdtq --n-elec 6 --norb 6 --matrix-file FCI_matrix.dat
"""
from __future__ import annotations

import argparse
import os
import sys
from math import comb

import numpy as np

from . import algorithm_sd, algorithm_sdtq, common


def _check_dimension(dim_fci: int, strings: np.ndarray, n_elec: int, norb: int) -> None:
    """Same sanity check present in the original QFlow-SDTQ script."""
    n_alpha = n_elec // 2
    n_beta = n_elec // 2
    expected_dim = comb(norb, n_alpha) * comb(norb, n_beta)
    if dim_fci != expected_dim:
        raise ValueError(
            f"Mismatch: strings.shape[0] = {dim_fci}, but expected FCI dimension = {expected_dim} "
            f"for {n_alpha} alpha, {n_beta} beta electrons in {norb} orbitals."
        )


def _generate_actspin(noas: int, nobs: int, nvas: int, nvbs: int, log_path: str | None) -> np.ndarray:
    """
    Generate the active-space definition in memory for the current
    system size (always -- there is no file to load or reuse). If
    ``log_path`` is given, also write the generated array there purely
    as a record of what was used for this run; that file is never read
    back, so it can never go stale or be accidentally mismatched with
    a different system.
    """
    nactspin, _actopx = common.generate_nactspin(noas, nobs, nvas, nvbs)
    print(f"Generated {nactspin.shape[0]} active space(s) for this run (shape {nactspin.shape}).")
    if log_path is not None:
        common.write_nactspin(log_path, nactspin)
        print(f"Wrote a copy of the generated active-space definition to '{log_path}' (for reference only).")
    return nactspin


def run_sd(n_elec: int, norb: int, matrix_file: str,
           max_iter: int = 20, verbose: bool = True,
           ckpt_dir: str = "amplitudes_ckpt", actspin_log_path: str | None = "nactspin_fortran.dat"):
    """
    Run QFlow-SD (singles + doubles) given an FCI matrix file. Mirrors
    the original SD-NoTQ.py driver exactly; only the hardcoded
    parameters (n_elec, norb, file paths, and the global iteration
    count) have been made into arguments, and the active-space
    definition (originally read from a file) is now always generated
    automatically for the given system size.
    """
    params = common.setup_qflow_inputs(n_elec=n_elec, norb=norb)
    noas, nobs = params["noas"], params["nobs"]
    nvas, nvbs = params["nvas"], params["nvbs"]
    nos, nvs, nstot = params["nos"], params["nvs"], params["nstot"]
    noccp = params["noccp"]

    half_str_a = common.generate_half_strings(norb, noccp)
    half_str_b = common.generate_half_strings(norb, noccp)
    half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
    half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]

    strings = common.build_fci_strings_from_half_strings(half_str_a, half_str_b, noas, nvas, nobs, nvbs)
    dim_fci = strings.shape[0]
    print(f"Determinant space dimension: {dim_fci}")

    matrix_h = algorithm_sd.read_fci_matrix(matrix_file, dim_fci)

    nactspin = _generate_actspin(noas, nobs, nvas, nvbs, actspin_log_path)
    nactives = nactspin.shape[0]

    trial = common.trial_fun(dim_fci, strings, noas, nobs, nstot)
    spin_array = common.generate_spin_array(nos, nvs)
    t1, t2 = algorithm_sd.initial_guess_zero(nos, nvs)
    it1, it2 = algorithm_sd.optimized_hierarchy_excitations(nos, nvs, nactspin, nactives)

    mnum1max = 8
    mnum2max = 18
    dim_m_max = 26

    print(f"Number of active spaces: {nactives}")

    final_energy, energy_list = algorithm_sd.run_qflow_loop(
        nactives=nactives,
        nactspin=nactspin,
        strings=strings,
        matrix_h=matrix_h,
        dim_fci=dim_fci,
        nstot=nstot,
        nos=nos, nvs=nvs, noas=noas, nobs=nobs, nvas=nvas, nvbs=nvbs,
        t1=t1, t2=t2,
        it1=it1, it2=it2,
        m_t_ext=algorithm_sd.m_t_ext,
        m_t_ext_exp=algorithm_sd.m_t_ext_exp,
        sim_trans=common.sim_trans,
        mnum12=algorithm_sd.mnum12,
        create_mlists_xm=algorithm_sd.create_mlists_xm,
        m_t_int=algorithm_sd.m_t_int,
        mpraxis=algorithm_sd.mpraxis,
        update_global_amplitudes=algorithm_sd.update_global_amplitudes,
        mnum1max=mnum1max, mnum2max=mnum2max,
        dim_m_max=dim_m_max,
        spin_array=spin_array,
        trial=trial,
        verbose=verbose,
        outdir=ckpt_dir,
        maxiter=max_iter,
    )

    print(f"\nFinal QFlow-SD energy: {final_energy:.10f}")
    return final_energy, energy_list


def run_sdtq(n_elec: int, norb: int, matrix_file: str,
             max_iter: int = 20, verbose: bool = True, actspin_log_path: str | None = "nactspin_fortran.dat"):
    """
    Run QFlow-SDTQ (singles, doubles, triples, quadruples) given an FCI
    matrix file. Mirrors the original serial_python.py driver exactly;
    only the hardcoded parameters (n_elec, norb, file paths, and the
    global iteration count) have been made into arguments, and the
    active-space definition (originally read from a file) is now
    always generated automatically for the given system size.
    """
    params = common.setup_qflow_inputs(n_elec=n_elec, norb=norb)
    noas, nobs = params["noas"], params["nobs"]
    nvas, nvbs = params["nvas"], params["nvbs"]
    nos, nvs, nstot = params["nos"], params["nvs"], params["nstot"]
    noccp = params["noccp"]

    half_str_a = common.generate_half_strings(norb, noccp)
    half_str_b = common.generate_half_strings(norb, noccp)
    half_str_a = half_str_a[np.lexsort(half_str_a.T[::1])]
    half_str_b = half_str_b[np.lexsort(half_str_b.T[::1])]

    strings = common.build_fci_strings_from_half_strings(half_str_a, half_str_b, noas, nvas, nobs, nvbs)
    dim_fci = strings.shape[0]
    _check_dimension(dim_fci, strings, n_elec, norb)
    print(f"Determinant space dimension: {dim_fci}")

    matrix_h = algorithm_sdtq.read_fci_matrix(matrix_file, dim_fci, n_elec)

    nactspin = _generate_actspin(noas, nobs, nvas, nvbs, actspin_log_path)
    nactives = nactspin.shape[0]
    print(f"Active-space definition shape: {nactspin.shape}")

    trial = common.trial_fun(dim_fci, strings, noas, nobs, nstot)
    spin_array = common.generate_spin_array(nos, nvs)
    t1, t2, t3, t4 = algorithm_sdtq.initial_guess_zero(nos, nvs)
    it1, it2, it3, it4 = algorithm_sdtq.optimized_hierarchy_excitations(nos, nvs, nactspin, nactives)

    mnum1max = 8
    mnum2max = 18
    mnum3max = 8
    mnum4max = 1
    dim_m_max = 35

    print(f"Number of active spaces: {nactives}")

    final_energy, energy_list = algorithm_sdtq.run_qflow_loop(
        nactives=nactives,
        nactspin=nactspin,
        strings=strings,
        matrix_h=matrix_h,
        dim_fci=dim_fci,
        nstot=nstot,
        nos=nos, nvs=nvs, noas=noas, nobs=nobs, nvas=nvas, nvbs=nvbs,
        t1=t1, t2=t2, t3=t3, t4=t4,
        it1=it1, it2=it2, it3=it3, it4=it4,
        m_t_ext=algorithm_sdtq.m_t_ext,
        m_t_ext_exp=algorithm_sdtq.m_t_ext_exp,
        sim_trans=common.sim_trans,
        mnum1234=algorithm_sdtq.mnum1234,
        create_mlists_xm=algorithm_sdtq.create_mlists_xm,
        m_t_int=algorithm_sdtq.m_t_int,
        mpraxis=algorithm_sdtq.mpraxis,
        update_global_amplitudes=algorithm_sdtq.update_global_amplitudes,
        mnum1max=mnum1max, mnum2max=mnum2max,
        mnum3max=mnum3max, mnum4max=mnum4max,
        dim_m_max=dim_m_max,
        spin_array=spin_array,
        trial=trial,
        verbose=verbose,
        maxiter=max_iter,
    )

    print(f"\nFinal QFlow-SDTQ energy: {final_energy:.10f}")
    return final_energy, energy_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qflow-suite",
        description="QFlow correlation energy optimizer (SD and SDTQ variants), "
                     "consuming an FCI matrix file produced by covo-ci. The active-space "
                     "definition is always generated automatically for the given system size.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a QFlow calculation.")
    run_parser.add_argument(
        "--level",
        choices=["sd", "sdtq"],
        required=True,
        help="QFlow variant: 'sd' (singles + doubles) or 'sdtq' (singles, doubles, triples, quadruples).",
    )
    run_parser.add_argument(
        "--n-elec", type=int, required=True,
        help="Total number of electrons (closed-shell, must be even).",
    )
    run_parser.add_argument("--norb", type=int, required=True, help="Number of spatial orbitals.")
    run_parser.add_argument(
        "--matrix-file",
        default="FCI_matrix.dat",
        help="Path to the FCI Hamiltonian matrix file (1-based 'i j value' lines), "
             "e.g. produced by 'covo-ci run ... --save-matrix FCI_matrix.dat'. Default: FCI_matrix.dat.",
    )
    run_parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum number of global QFlow iterations (each iteration optimizes every active "
             "space once). Default: 20, matching the original scripts' hardcoded value.",
    )
    run_parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="(--level sd only) Directory to write per-iteration T1/T2 amplitude checkpoints to. "
             "Default: 'amplitudes_ckpt', placed inside --output-dir if given.",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="If given, place the amplitude checkpoint directory (--level sd only) and the "
             "logged copy of the generated active-space definition inside this directory "
             "instead of the current one. Created if it doesn't exist. Does not affect "
             "--matrix-file, or --ckpt-dir if it was given explicitly.",
    )
    run_parser.add_argument(
        "--no-actspin-log",
        action="store_true",
        help="Don't write a copy of the generated active-space definition to disk. By default "
             "a copy is written to 'nactspin_fortran.dat' (or inside --output-dir if given) "
             "purely as a record of what was used for this run; it is never read back.",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration progress output.")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        verbose = not args.quiet

        # Resolve --ckpt-dir's default, placing it inside --output-dir when
        # given and --ckpt-dir wasn't explicitly overridden. An
        # explicitly-given --ckpt-dir is always used exactly as given,
        # regardless of --output-dir.
        output_dir = args.output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        ckpt_dir = args.ckpt_dir
        if ckpt_dir is None:
            ckpt_dir = os.path.join(output_dir, "amplitudes_ckpt") if output_dir else "amplitudes_ckpt"

        actspin_log_path = None
        if not args.no_actspin_log:
            actspin_log_path = (
                os.path.join(output_dir, "nactspin_fortran.dat") if output_dir else "nactspin_fortran.dat"
            )

        print(f"QFlow level: {args.level}")
        print(f"n_elec={args.n_elec}, norb={args.norb}")
        print(f"Matrix file: {args.matrix_file}")
        print(f"Max global iterations: {args.max_iter}")
        if args.level == "sd":
            print(f"Checkpoint directory: {ckpt_dir}")

        if args.max_iter < 1:
            parser.error(f"--max-iter must be >= 1 (got {args.max_iter}).")

        try:
            if args.level == "sd":
                run_sd(
                    n_elec=args.n_elec,
                    norb=args.norb,
                    matrix_file=args.matrix_file,
                    max_iter=args.max_iter,
                    verbose=verbose,
                    ckpt_dir=ckpt_dir,
                    actspin_log_path=actspin_log_path,
                )
            else:
                run_sdtq(
                    n_elec=args.n_elec,
                    norb=args.norb,
                    matrix_file=args.matrix_file,
                    max_iter=args.max_iter,
                    verbose=verbose,
                    actspin_log_path=actspin_log_path,
                )
        except ValueError as exc:
            parser.error(str(exc))
            return 2

        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
