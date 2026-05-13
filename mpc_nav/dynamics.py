"""7-state plant dynamics with RK4 integrator.

State vector x = [n, e, chi, mu, p, V, thr].
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from . import config

_G = 9.81


def rk4_step_long(x: np.ndarray, aileron_cmd: float, thr_cmd: float, Ts: float,
                  wind: Tuple[float, float],
                  m: float, k_d: float, k_t: float, tau_thr: float) -> np.ndarray:
    """One RK4 step of the extended 7-state plant."""
    wn, we = wind

    def f(z: np.ndarray) -> np.ndarray:
        n, e, chi, mu, p, V, thr = z
        V_eff = max(1.0, float(V))  # prevent division by zero in chi_dot
        return np.array([
            V * math.cos(chi) + wn,
            V * math.sin(chi) + we,
            (_G / V_eff) * math.tan(mu),
            p,
            config.a_p * p + config.b_p * aileron_cmd,
            (k_t * thr - k_d * V * V) / m,
            (thr_cmd - thr) / tau_thr,
        ], dtype=float)

    k1 = f(x)
    k2 = f(x + 0.5 * Ts * k1)
    k3 = f(x + 0.5 * Ts * k2)
    k4 = f(x + Ts * k3)
    return x + (Ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
