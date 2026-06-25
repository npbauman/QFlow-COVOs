"""
Smoke and regression tests for qflow_suite.

These run the bundled H4/STO-3G example (a minimal but nontrivial
active-space case: noas=nobs=2, nvas=nvbs=2, giving exactly one active
space) end-to-end for both QFlow-SD and QFlow-SDTQ, and check
convergence behavior against the known FCI ground-state energy (from
the companion covo-ci package's own bundled example data).

The active-space definition is always generated automatically in
memory by run_sd/run_sdtq (see qflow_suite.common.generate_nactspin);
there is no manual active-space file to supply.
"""
import os

import numpy as np
import pytest

from qflow_suite import algorithm_sd, algorithm_sdtq, common
from qflow_suite.cli import main, run_sd, run_sdtq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
H4_EXAMPLE = os.path.join(REPO_ROOT, "examples", "h4_sd")

# Exact FCI ground-state energy for this H4/STO-3G example, as computed
# by covo-ci (see examples/h4_sd/EXPECTED_OUTPUT.txt).
FCI_REFERENCE_ENERGY = -2.1663874486


def test_common_setup_qflow_inputs():
    params = common.setup_qflow_inputs(n_elec=4, norb=4)
    assert params["noccp"] == 2
    assert params["nvirt"] == 2
    assert params["nstot"] == 8


def test_common_string_generation_matches_expected_dimension():
    params = common.setup_qflow_inputs(n_elec=4, norb=4)
    half_a = common.generate_half_strings(params["norb"], params["noccp"])
    half_b = common.generate_half_strings(params["norb"], params["noccp"])
    strings = common.build_fci_strings_from_half_strings(
        half_a, half_b, params["noas"], params["nvas"], params["nobs"], params["nvbs"]
    )
    assert strings.shape[0] == 36  # comb(4,2) * comb(4,2)


def test_generate_nactspin_matches_bundled_h4_example():
    nactspin, actopx = common.generate_nactspin(noas=2, nobs=2, nvas=2, nvbs=2)
    assert nactspin.shape == (1, 8)
    np.testing.assert_array_equal(nactspin, np.array([[0, 0, 0, 0, 0, 0, 0, 0]]))


def test_generate_nactspin_write_and_reload_roundtrip(tmp_path):
    nactspin, _ = common.generate_nactspin(noas=2, nobs=2, nvas=2, nvbs=2)
    out_path = tmp_path / "nactspin_fortran.dat"
    common.write_nactspin(str(out_path), nactspin)
    reloaded = common.load_nactspin(str(out_path), n_columns=None)
    np.testing.assert_array_equal(reloaded, nactspin)


def test_generate_nactspin_active_space_count_and_structure():
    # noas=3, nvas=6 -> comb(3,2)*comb(6,2) = 3*15 = 45 active spaces,
    # matching the original SD-NoTQ.py script's own system size (n_elec=6, norb=9).
    nactspin, actopx = common.generate_nactspin(noas=3, nobs=3, nvas=6, nvbs=6)
    assert nactspin.shape == (45, 18)
    assert actopx.shape == (45, 4)
    # Every row should mark exactly 4 spin-orbitals active (2 occ pair + 2 virt
    # pair), mirrored into both alpha and beta blocks -> 8 zeros per row total.
    assert np.all(np.sum(nactspin == 0, axis=1) == 8)


def test_generate_nactspin_orders_homo_lumo_first():
    """Row 0 should always be the HOMO-LUMO active space (highest
    occupied pair, lowest virtual pair), and the last row should be
    the lowest-occupied/highest-virtual active space -- this matches
    a user-provided reference example for an 8-electron, 8-orbital
    system (noas=nobs=nvas=nvbs=4)."""
    nactspin, actopx = common.generate_nactspin(noas=4, nobs=4, nvas=4, nvbs=4)
    assert nactspin.shape == (36, 16)
    np.testing.assert_array_equal(actopx[0], [2, 3, 0, 1])  # HOMO pair, LUMO pair
    np.testing.assert_array_equal(actopx[-1], [0, 1, 2, 3])  # lowest occ pair, highest virt pair

    # 0 = active under this convention: row 0's active spin-orbitals
    # should be exactly the HOMO pair (alpha+beta) and LUMO pair (alpha+beta).
    active_indices_row0 = set(np.where(nactspin[0] == 0)[0].tolist())
    assert active_indices_row0 == {2, 3, 6, 7, 8, 9, 12, 13}


def test_generate_nactspin_matches_fortran_reference_h6():
    """Exact row-for-row match against real Fortran build_nactspin
    program output (3 occupied, 3 virtual spatial orbitals per spin
    channel -- a 6-electron, 6-orbital closed-shell system)."""
    nactspin, actopx = common.generate_nactspin(noas=3, nobs=3, nvas=3, nvbs=3)
    expected = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
    ])
    np.testing.assert_array_equal(nactspin, expected)


def test_generate_nactspin_raises_for_too_small_system():
    with pytest.raises(ValueError):
        common.generate_nactspin(noas=1, nobs=1, nvas=1, nvbs=1)


def test_read_fci_matrix_sd_and_sdtq_agree():
    """Both QFlow variants' read_fci_matrix should parse the same file
    identically (they differ only in an unused extra parameter)."""
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    dim_fci = 36
    m_sd = algorithm_sd.read_fci_matrix(matrix_file, dim_fci)
    m_sdtq = algorithm_sdtq.read_fci_matrix(matrix_file, dim_fci, n_elec=4)
    np.testing.assert_allclose(m_sd, m_sdtq)
    # Matrix should be symmetric
    np.testing.assert_allclose(m_sd, m_sd.T)


@pytest.mark.slow
def test_qflow_sd_runs_and_improves_on_hartree_fock(tmp_path, monkeypatch):
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)  # keep auto-written nactspin/checkpoint files out of the repo

    final_energy, energy_list = run_sd(n_elec=4, norb=4, matrix_file=matrix_file, verbose=False)

    # QFlow-SD should land between the FCI reference and a clearly
    # uncorrelated (much higher) energy; and should not overshoot below
    # the variational FCI ground state.
    assert final_energy > FCI_REFERENCE_ENERGY - 1e-6
    assert final_energy < FCI_REFERENCE_ENERGY + 0.05


@pytest.mark.slow
def test_qflow_sd_runs_are_reproducible_across_repeated_calls(tmp_path, monkeypatch):
    """Since the active-space definition is regenerated fresh every
    call (never loaded from a possibly-stale file), repeated calls for
    the same system size must still produce identical results."""
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)

    energy_1, _ = run_sd(n_elec=4, norb=4, matrix_file=matrix_file, verbose=False)
    energy_2, _ = run_sd(n_elec=4, norb=4, matrix_file=matrix_file, verbose=False)
    assert energy_1 == pytest.approx(energy_2)


def test_run_sd_respects_max_iter(tmp_path, monkeypatch):
    """A smaller --max-iter should produce fewer (and less converged)
    energy_list entries than the default, while still running fine."""
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)

    final_energy, energy_list = run_sd(n_elec=4, norb=4, matrix_file=matrix_file, max_iter=3, verbose=False)
    assert len(energy_list) == 3
    # 3 iterations on this system should still be variationally above
    # the exact FCI ground state (no overshoot), but not yet as
    # converged as the full 20-iteration default.
    assert final_energy > FCI_REFERENCE_ENERGY - 1e-6


def test_cli_max_iter_invalid_value_errors_cleanly(tmp_path, monkeypatch, capsys):
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main([
            "run", "--level", "sd", "--n-elec", "4", "--norb", "4",
            "--matrix-file", matrix_file, "--max-iter", "0", "--quiet",
        ])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--max-iter must be >= 1" in captured.err


@pytest.mark.slow
def test_qflow_sdtq_converges_close_to_fci(tmp_path, monkeypatch):
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)

    final_energy, energy_list = run_sdtq(n_elec=4, norb=4, matrix_file=matrix_file, verbose=False)

    # With triples and quadruples included for this small system, QFlow-SDTQ
    # should recover the FCI energy to within tight tolerance.
    assert abs(final_energy - FCI_REFERENCE_ENERGY) < 1e-4
    # Variational bound: should not go below the true FCI ground state
    # (allowing tiny numerical slack).
    assert final_energy > FCI_REFERENCE_ENERGY - 1e-6


@pytest.mark.slow
def test_cli_output_dir_places_actspin_log_and_ckpt_dir_inside_it(tmp_path, monkeypatch):
    """--output-dir should redirect the logged active-space file copy
    and the checkpoint dir into it, without affecting --matrix-file."""
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "results"

    rc = main([
        "run", "--level", "sd", "--n-elec", "4", "--norb", "4",
        "--matrix-file", matrix_file, "--output-dir", str(output_dir), "--quiet",
    ])
    assert rc == 0
    assert (output_dir / "nactspin_fortran.dat").exists()
    assert (output_dir / "amplitudes_ckpt").is_dir()
    # Should not have leaked a stray nactspin file into the cwd itself.
    assert not (tmp_path / "nactspin_fortran.dat").exists()


@pytest.mark.slow
def test_cli_no_actspin_log_suppresses_the_logged_file(tmp_path, monkeypatch):
    matrix_file = os.path.join(H4_EXAMPLE, "FCI_matrix.dat")
    monkeypatch.chdir(tmp_path)

    rc = main([
        "run", "--level", "sd", "--n-elec", "4", "--norb", "4",
        "--matrix-file", matrix_file, "--no-actspin-log", "--quiet",
    ])
    assert rc == 0
    assert not (tmp_path / "nactspin_fortran.dat").exists()
