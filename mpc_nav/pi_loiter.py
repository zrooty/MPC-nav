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

from .pid_loiter import PIDLoiter


class PILoiter(PIDLoiter):
    """Naive radial PI = :class:`PIDLoiter` with the derivative term removed.

    The full control law (groundspeed feed-forward + conditional anti-windup)
    lives in ``PIDLoiter``; this subclass simply pins ``Kd = 0`` so the missing
    cross-track-rate damping — the whole point of this negative baseline — is
    encoded directly in the type. Behaviour is identical to the previous
    standalone implementation.
    """

    def __init__(self, Kp: float, Ki: float, bank_lim_deg: float,
                 loiter_direction: int, Ts: float,
                 use_centripetal_ff: bool = True):
        super().__init__(Kp, Ki, 0.0, bank_lim_deg, loiter_direction, Ts,
                         use_centripetal_ff=use_centripetal_ff)
