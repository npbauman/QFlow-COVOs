"""
qflow_suite: QFlow correlation energy optimization, in two variants:

- QFlow-SD: singles + doubles (T1, T2) cluster amplitudes.
- QFlow-SDTQ: singles, doubles, triples, and quadruples (T1-T4)
  cluster amplitudes.

Both consume an FCI Hamiltonian matrix file (such as the
``--save-matrix`` output of the companion ``covo-ci`` package) and an
active-space definition file, and iteratively optimize cluster
amplitudes via similarity-transformed active-space energy
minimization.
"""

__version__ = "0.1.0"
