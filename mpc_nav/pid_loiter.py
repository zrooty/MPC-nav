"""PID loiter controller — a functional classical baseline vs L1 and MPC.

This is the stable counterpart of the (deliberately naive) ``PILoiter``. It
adds a derivative term on the **radial velocity** ``rdot`` — the missing
damping that makes a pure radial PI diverge (see ``.docs/pi_baseline_notes.md``
and ``pi_loiter.py``). The ``Kd * rdot`` term is the direct analogue of L1's
``Kv`` cross-track-rate feedback.

Control law (bank-angle output, ``phi_cmd``):

    e_r     = rho - R                         # signed radial error, >0 outside
    rdot    = Ahat . v_g                      # radial velocity, >0 moving outward
    phi_pid = Kp*e_r + Kd*rdot + Ki*integral(e_r)
    phi_ff  = atan(Vg^2 / (g * R))            # centripetal steady-state bank
    phi_cmd = sign_dir * (phi_ff + phi_pid)

``rdot`` is the exact analytic radial speed (``Ahat·v_g``), so no numerical
differentiation / derivative filtering is needed.

``dir`` / ``sign_dir`` follow the package convention used by L1Loiter:
``loiter_direction < 0`` -> CW -> negative bank.

Anti-windup: the integrator is frozen whenever the (unsaturated) bank command
is past the limit *and* the radial error would push it further into
saturation — mirroring the conditional anti-windup of the throttle PI.
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from .geometry import CirclePath, radial_kinematics

_G = 9.81


class PIDLoiter:
    def __init__(self, Kp: float, Ki: float, Kd: float, bank_lim_deg: float,
                 loiter_direction: int, Ts: float,
                 use_centripetal_ff: bool = True):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.bank_lim = float(np.radians(bank_lim_deg))
        self.dir = -1 if (loiter_direction < 0) else +1
        self.sign_dir = +1.0 if self.dir > 0 else -1.0
        self.Ts = float(Ts)
        self.use_ff = bool(use_centripetal_ff)
        self.integ = 0.0   # integral of radial error [m·s]

    def reset(self) -> None:
        self.integ = 0.0

    def command(self, x: np.ndarray, path: CirclePath,
                Va_for_ctrl: float, wind: Tuple[float, float]) -> float:
        # Va_for_ctrl is unused (we use groundspeed) — kept for API symmetry.
        rk = radial_kinematics(x, wind, path)
        e_r = rk.e_r                          # >0 outside the circle
        rdot = rk.rdot                        # >0 moving outward

        # PID on radial error. Tentatively advance the integrator, then decide
        # whether to commit it (conditional anti-windup, below). The Kd term
        # is what makes this stable where the pure-PI baseline diverges; with
        # Kd=0 this reduces exactly to PILoiter.
        integ_new = self.integ + e_r * self.Ts
        phi_pid = self.Kp * e_r + self.Kd * rdot + self.Ki * integ_new

        phi_ff = math.atan((rk.Vg * rk.Vg) / (_G * path.R)) if self.use_ff else 0.0

        phi_unsat = self.sign_dir * (phi_ff + phi_pid)
        phi_cmd = float(np.clip(phi_unsat, -self.bank_lim, self.bank_lim))

        # Conditional anti-windup: freeze the integrator only when saturated
        # AND the error would drive the command deeper into saturation.
        saturated = abs(phi_unsat) > self.bank_lim + 1e-12
        push = self.sign_dir * self.Ki * e_r          # incremental effect on phi_unsat
        if not (saturated and np.sign(push) == np.sign(phi_unsat)):
            self.integ = integ_new

        return phi_cmd
