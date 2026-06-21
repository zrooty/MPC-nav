"""L1 loiter controller.

``dir = -1`` → CW (right / negative bank);
``dir = +1`` → CCW (left  / positive bank).
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from .geometry import CirclePath, radial_kinematics

_G = 9.81


class L1Loiter:
    def __init__(self, period: float, damping: float, bank_lim_deg: float, loiter_direction: int):
        self.period = float(period)
        self.damping = float(damping)
        self.bank_lim = float(np.radians(bank_lim_deg))
        self.dir = -1 if (loiter_direction < 0) else +1

    def command(self, x: np.ndarray, path: CirclePath,
                Va_for_ctrl: float, wind: Tuple[float, float]) -> float:
        # Va_for_ctrl is unused (we use groundspeed) — kept for API symmetry.
        rk = radial_kinematics(x, wind, path)
        vg, Vg, Ahat, r = rk.vg, rk.Vg, rk.Ahat, rk.r

        omega = 2.0 * math.pi / self.period
        Kx = omega ** 2
        Kv = 2.0 * self.damping * omega
        K_L1 = 4.0 * (self.damping ** 2)
        L1_dist = (1.0 / math.pi) * self.damping * self.period * Vg  # groundspeed-based lookahead

        # Capture geometry
        xtrackVelCap = Ahat[0] * vg[1] - Ahat[1] * vg[0]
        ltrackVelCap = -(Ahat @ vg)
        Nu_cap = math.atan2(xtrackVelCap, ltrackVelCap)
        Nu_cap = np.clip(Nu_cap, -math.pi / 2, math.pi / 2)
        latAccCap = K_L1 * Vg * Vg / max(L1_dist, 1e-6) * math.sin(Nu_cap)

        # Circle-following PD
        xtrackVelCirc = -ltrackVelCap
        xtrackErrCirc = r - path.R
        latAccCircPD = xtrackErrCirc * Kx + xtrackVelCirc * Kv

        # Tangential velocity sign per direction
        sign_dir = +1.0 if self.dir > 0 else -1.0
        velTangent = xtrackVelCap * sign_dir

        # Prevent wrong-way capture
        if ltrackVelCap < 0.0 and velTangent < 0.0:
            latAccCircPD = max(latAccCircPD, 0.0)

        # Centripetal feed-forward
        latAccCircCtr = (velTangent ** 2) / max(0.5 * path.R, (path.R + xtrackErrCirc))
        latAccCirc = sign_dir * (latAccCircPD + latAccCircCtr)

        # Capture-vs-circle switch (capture only when outside the circle)
        if xtrackErrCirc > 0.0 and sign_dir * latAccCap < sign_dir * latAccCirc:
            latAcc = latAccCap
        else:
            latAcc = latAccCirc

        phi_cmd = math.atan2(latAcc, _G)
        return float(np.clip(phi_cmd, -self.bank_lim, self.bank_lim))
