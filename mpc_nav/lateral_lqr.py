"""6-state lateral LQR inner loop from Nugroho et al. (2022).

State  x = [phi, p, theta, q, psi, r]   (idx 0..5)
Input  u = [tau1, tau2, tau3]           (aileron, elevator, rudder torques)

For 2D lateral simulation only phi (=mu, bank angle) and p (roll rate) are
physically meaningful; theta, q, psi, r are dummy zero-trim states because
there is no longitudinal/yaw dynamics in this 2D model.

State-space (eq. 12 of the article), with small-UAV inertia parameterised
through ``a_p`` and ``b_p`` so the LQR closed-loop matches the plant:

    phi_dot   = p
    p_dot     = a_p * p + b_p * aileron
    theta_dot = q                   (dummy, q=0)
    q_dot     = 0                   (dummy, no elevator)
    psi_dot   = r                   (dummy, r=0)
    r_dot     = 0                   (dummy)

Only tau1 (roll torque → aileron) is active. The output is ``aileron_cmd``
(rad) derived from the K[0] feedback on phi and p.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_continuous_are

from . import config


# Inertia estimate for ~1.2 kg UAV, consistent with the article.
_IXX = 0.05   # [kg.m²] roll inertia
_IYY = 0.10   # [kg.m²] pitch inertia (dummy)
_IZZ = 0.12   # [kg.m²] yaw inertia (dummy)


def _build_AB() -> tuple[np.ndarray, np.ndarray]:
    """Linearised 6-state, 3-input lateral-dynamics matrices at trim (q=r=0)."""
    # At trim q=r=0 all cross-coupling terms vanish.
    r_trim = 0.0
    q_trim = 0.0

    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, (_IYY - _IZZ) * r_trim / _IXX, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, (_IZZ - _IXX) * r_trim / _IYY, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, (_IXX - _IYY) * q_trim / _IZZ, 0, 0, 0, 0],
    ], dtype=float)

    B = np.zeros((6, 3))
    B[1, 0] = 1.0 / _IXX   # p_dot from tau1
    B[3, 1] = 1.0 / _IYY   # q_dot from tau2 (dummy)
    B[5, 2] = 1.0 / _IZZ   # r_dot from tau3 (dummy)
    return A, B


# Tuning: roll response ~0.5s, no overshoot.
# Q_phi large → tight phi tracking; Q_p damps roll rate.
_Q_LQR = np.diag([
    80.0,   # phi   – tight tracking
    5.0,    # p     – damping
    1.0,    # theta (dummy)
    0.5,    # q     (dummy)
    1.0,    # psi   (dummy)
    0.5,    # r     (dummy)
])
_R_LQR = np.diag([
    1.0,    # tau1 (aileron)
    10.0,   # tau2 (dummy – heavy so it stays inactive)
    10.0,   # tau3 (dummy)
])


class LateralLQR:
    def __init__(self):
        A, B = _build_AB()
        P = solve_continuous_are(A, B, _Q_LQR, _R_LQR)
        self.K = np.linalg.inv(_R_LQR) @ B.T @ P   # shape (3, 6)

        # Convert tau1 → aileron_cmd (rad) using the plant identity
        # p_dot = a_p*p + b_p*aileron and B[1,0]=1/Ixx, so:
        #   aileron = tau1 / (b_p * Ixx)
        self._aileron_scale = 1.0 / (config.b_p * _IXX)
        self._ail_lim = np.radians(config.aileron_limit_deg)

        print(f"[LateralLQR] K (roll channel, state=[phi,p]): {self.K[0, :2]}")

    def step(self, mu_ref: float, mu: float, p: float) -> float:
        """Aileron command from 6-state feedback. Dummy states held at zero.

        Parameters
        ----------
        mu_ref : target bank angle [rad] from MPC/L1
        mu     : actual bank angle [rad]
        p      : actual roll rate  [rad/s]

        Returns
        -------
        aileron_cmd [rad], clipped to ±aileron_limit_deg.
        """
        x_err = np.array([
            mu - mu_ref,   # phi error
            p,             # p (ref=0 at trim)
            0.0,           # theta (dummy)
            0.0,           # q     (dummy)
            0.0,           # psi   (dummy)
            0.0,           # r     (dummy)
        ], dtype=float)
        tau1 = float(-self.K[0, :] @ x_err)
        aileron_cmd = tau1 * self._aileron_scale
        return float(np.clip(aileron_cmd, -self._ail_lim, self._ail_lim))
