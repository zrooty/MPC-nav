"""Geometry helpers: angle wrapping, circle path, orbit error metrics."""
from __future__ import annotations
import math
from typing import NamedTuple, Tuple
import numpy as np

from . import config


def wrap(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


class RadialKinematics(NamedTuple):
    """Groundspeed and radial-error quantities shared by the loiter laws."""
    vg: np.ndarray   # ground-velocity vector [N, E]
    Vg: float        # groundspeed magnitude (floored at 1.0)
    A: np.ndarray    # vector from circle centre to aircraft [N, E]
    r: float         # distance to centre (+1e-9 guard)
    Ahat: np.ndarray # radial unit vector
    e_r: float       # signed radial error rho - R (>0 outside)
    rdot: float      # radial velocity Ahat·vg (>0 moving outward)


def radial_kinematics(x: np.ndarray, wind: Tuple[float, float],
                      path: "CirclePath") -> RadialKinematics:
    """Common groundspeed + radial-error block used by L1/PI/PID loiter laws.

    Centralises the ``+1e-9`` distance guard and the ``max(1.0, …)``
    groundspeed floor so the three controllers stay consistent.
    """
    n, e, chi, mu, p, V, thr = x
    wn, we = wind
    vg = np.array([V * math.cos(chi) + wn, V * math.sin(chi) + we])
    Vg = max(1.0, float(np.linalg.norm(vg)))
    A = np.array([n, e]) - np.asarray(path.C, dtype=float)
    r = float(np.linalg.norm(A)) + 1e-9
    Ahat = A / r
    return RadialKinematics(vg, Vg, A, r, Ahat, r - path.R, float(Ahat @ vg))


class CirclePath:
    def __init__(self, C: Tuple[float, float], R: float, cw: bool = True):
        self.C = tuple(C)
        self.R = float(R)
        self.cw = bool(cw)

    def closest_angle(self, p: Tuple[float, float]) -> float:
        r = np.array(p, dtype=float) - np.asarray(self.C, dtype=float)
        return math.atan2(r[1], r[0])


def make_start_state(pos, heading_type, bank0, Va_init, thr_init) -> np.ndarray:
    """Build the 7-state initial vector [n, e, chi, mu, p, V, thr]."""
    n0, e0 = pos
    if isinstance(heading_type, (int, float)):
        chi0 = float(heading_type)
    elif heading_type == "to_center":
        chi0 = math.atan2(config.circle_C[1] - e0, config.circle_C[0] - n0)
    elif heading_type == "east":
        chi0 = 0.0
    elif heading_type == "north":
        chi0 = math.pi / 2.0
    else:
        raise ValueError("Unknown start_heading_type")
    return np.array([n0, e0, chi0, bank0, 0.0, float(Va_init), float(thr_init)], dtype=float)


def crosstrack_series(path: CirclePath, n_series, e_series) -> np.ndarray:
    C = np.asarray(path.C)
    return np.array([np.linalg.norm(np.array([ni, ei]) - C) - path.R
                     for ni, ei in zip(n_series, e_series)])


def orbit_metrics(n, e, chi, C, R, ccw: bool = True):
    """Return (rho, |rho-R|, chi_tangent, echi) for a point relative to a circle."""
    r = np.array([n, e]) - np.asarray(C, float)
    rho = float(np.linalg.norm(r)) + 1e-9
    theta = math.atan2(r[1], r[0])
    s = +1.0 if ccw else -1.0
    chi_tan = theta + s * (math.pi / 2.0)
    echi = wrap(chi_tan - chi)
    return rho, abs(rho - R), chi_tan, echi
