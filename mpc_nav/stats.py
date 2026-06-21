"""Tiny numeric helpers shared across the package, scripts, and tools.

Kept dependency-free (numpy only) and side-effect-free so scripts can import it
without triggering the log-directory creation in :mod:`mpc_nav.io_utils`.
"""
from __future__ import annotations
import numpy as np


def rms(a: np.ndarray) -> float:
    """Root-mean-square of ``a``; ``nan`` for an empty input."""
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")
