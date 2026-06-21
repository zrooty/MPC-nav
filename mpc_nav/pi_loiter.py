"""PI loiter controller — a deliberately simple (naive) baseline.

Control law (bank-angle output, ``phi_cmd``):

    e_r     = rho - R                         # signed radial error, >0 outside
    phi_pi  = Kp * e_r + Ki * integral(e_r)
    phi_ff  = atan(Vg^2 / (g * R))            # centripetal steady-state bank
    phi_cmd = sign_dir * (phi_ff + phi_pi)

The centripetal feed-forward ``phi_ff`` is the bank required to hold the
nominal circle at the current groundspeed; the PI term corrects the radial
error around it.

``dir`` / ``sign_dir`` follow the package convention used by L1Loiter:
``loiter_direction < 0`` → CW → negative bank.

Anti-windup: the integrator is frozen whenever the (unsaturated) bank command
is past the limit *and* the radial error would push it further into
saturation — mirroring the conditional anti-windup of the throttle PI.

WARNING — this baseline is UNSTABLE for the loiter task and does not hold the
circle (cross-track RMS grows to hundreds of metres vs a 90 m radius). It is
kept as a baseline precisely to document that a naive position-only PI is
insufficient (see ``.docs/pi_baseline_notes.md``). Two structural causes:

  1. The map (bank → radial position) is ~double-integrator, so a pure PI
     (no derivative action) is structurally underdamped → it limit-cycles or
     diverges. Cross-track *rate* damping (an L1-style ``Kv`` / a D term) is
     required for stability.
  2. A position-only bank law ignores heading, so it cannot perform capture
     from an off-circle initial condition.

Both remain even after the simulation's inner-loop sampling bug was fixed
(plant roll now sub-stepped at Ts/5, see config.inner_loop_substeps), so the
divergence is intrinsic to the law — not an artifact. To turn this into a
*functional* baseline, add a ``Kd * rdot`` term (→ PID) or use a
heading-cascade guidance law.
"""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from .geometry import CirclePath

_G = 9.81


class PILoiter:
    def __init__(self, Kp: float, Ki: float, bank_lim_deg: float,
                 loiter_direction: int, Ts: float,
                 use_centripetal_ff: bool = True):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
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
        n, e, chi, mu, p, V, thr = x
        wn, we = wind

        vg = np.array([V * math.cos(chi) + wn, V * math.sin(chi) + we])
        Vg = max(1.0, float(np.linalg.norm(vg)))

        A = np.array([n, e]) - np.asarray(path.C, dtype=float)
        r = float(np.linalg.norm(A)) + 1e-9
        e_r = r - path.R                      # >0 outside the circle

        # PI on radial error. Tentatively advance the integrator, then decide
        # whether to commit it (conditional anti-windup, below).
        integ_new = self.integ + e_r * self.Ts
        phi_pi = self.Kp * e_r + self.Ki * integ_new

        phi_ff = math.atan((Vg * Vg) / (_G * path.R)) if self.use_ff else 0.0

        phi_unsat = self.sign_dir * (phi_ff + phi_pi)
        phi_cmd = float(np.clip(phi_unsat, -self.bank_lim, self.bank_lim))

        # Conditional anti-windup: freeze the integrator only when saturated
        # AND the error would drive the command deeper into saturation.
        saturated = abs(phi_unsat) > self.bank_lim + 1e-12
        push = self.sign_dir * self.Ki * e_r          # incremental effect on phi_unsat
        if not (saturated and np.sign(push) == np.sign(phi_unsat)):
            self.integ = integ_new

        return phi_cmd
