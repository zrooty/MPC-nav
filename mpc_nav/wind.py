"""Wind models. Selected via ``config.wind_mode``."""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from . import config


def wind_at(t: float, state: Tuple[float, float, float, float] | None = None) -> Tuple[float, float]:
    """Return (wn, we) at time ``t`` [s] according to global wind_* settings."""
    mode = config.wind_mode

    if mode == "constant":
        return config.wind_mean

    if mode == "rotating":
        vmag = math.hypot(*config.wind_mean)
        theta0 = math.atan2(config.wind_mean[1], config.wind_mean[0])
        theta = theta0 + math.radians(config.wind_rot_deg_s) * t
        return (vmag * math.cos(theta), vmag * math.sin(theta))

    if mode == "gust":
        base = np.array(config.wind_mean, float)
        vmag = np.linalg.norm(base) + config.gust_amp * math.sin(2 * math.pi * t / config.gust_T)
        theta = math.atan2(base[1], base[0])
        return (vmag * math.cos(theta), vmag * math.sin(theta))

    if mode == "randomwalk":
        theta0 = math.atan2(config.wind_mean[1], config.wind_mean[0])
        theta = theta0 + math.radians(config.rw_sigma_deg_s) * math.sin(0.7 * t)
        vmag = min(config.wind_max,
                   max(0.0, np.linalg.norm(config.wind_mean) + config.rw_sigma_mps_s * math.sin(1.3 * t + 0.4)))
        return (vmag * math.cos(theta), vmag * math.sin(theta))

    if mode == "custom":
        if state is None:
            n, e = 0.0, 0.0
        else:
            n, e = state[0], state[1]
        toC = np.array(config.circle_C) - np.array([n, e])
        ang = math.atan2(toC[1], toC[0])
        vmag = min(config.wind_max, 6.0 + 3.0 * math.sin(0.3 * t))
        return (vmag * math.cos(ang), vmag * math.sin(ang))

    return config.wind_mean


def wind_mode_str() -> str:
    """Human-readable label for the current wind mode."""
    return {
        "constant":   "konstan",
        "rotating":   f"rotating (ω={config.wind_rot_deg_s:.1f}°/s)",
        "gust":       f"gust (A={config.gust_amp:.1f} m/s, T={config.gust_T:.0f} s)",
        "randomwalk": "random-walk",
        "custom":     "custom",
    }.get(config.wind_mode, str(config.wind_mode))
