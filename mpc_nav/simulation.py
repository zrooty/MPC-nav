"""Closed-loop simulation: builds controllers, runs the time loop, returns logs."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

from . import config
from .geometry import CirclePath, make_start_state, orbit_metrics
from .wind import wind_at
from .dynamics import rk4_step_long
from .lateral_lqr import LateralLQR
from .l1_loiter import L1Loiter
from .ltv_mpc import LTVMPC_OSQP, MPCWeights
from .pi_loiter import PILoiter
from .pid_loiter import PIDLoiter


# ----------------------------------------------------------------------
# Controller construction
# ----------------------------------------------------------------------
def build_controllers() -> Tuple[CirclePath, L1Loiter, L1Loiter, LTVMPC_OSQP,
                                 PILoiter, PIDLoiter, LateralLQR]:
    path = CirclePath(config.circle_C, config.circle_R, cw=config.cw)
    l1_dir = -1 if config.cw else +1
    l1_green = L1Loiter(config.L1_period, config.L1_damping, config.bank_limit_deg, l1_dir)
    l1_orange = L1Loiter(config.L1_period, config.L1_damping, config.bank_limit_deg, l1_dir)
    weights = MPCWeights(config.w_et, config.w_echi, config.w_mu,
                         config.w_u, config.w_et_T, config.w_echi_T)
    lateral_lqr = LateralLQR()
    mpc = LTVMPC_OSQP(
        Ts=config.Ts, N=config.N_horizon, Va_init=config.Va_ref, tau_mu=config.tau_mu,
        bank_limit_deg=config.bank_limit_deg, slew_limit_deg_s=config.slew_limit_deg_s,
        weights=weights, path=path,
        w_du=config.w_du, w_mu_Term=config.w_mu_Term,
        use_groundspeed_mu_ss=True, Va_nominal=config.Va_nominal,
        use_w_du_scaling=config.use_w_du_scaling,
        lateral_lqr=lateral_lqr, a_p_roll=config.a_p, b_p_roll=config.b_p,
    )
    pi = PILoiter(config.PI_Kp, config.PI_Ki, config.bank_limit_deg, l1_dir,
                  config.Ts, use_centripetal_ff=config.PI_centripetal_ff)
    pid = PIDLoiter(config.PID_Kp, config.PID_Ki, config.PID_Kd,
                    config.bank_limit_deg, l1_dir, config.Ts,
                    use_centripetal_ff=config.PID_centripetal_ff)
    return path, l1_green, l1_orange, mpc, pi, pid, lateral_lqr


# ----------------------------------------------------------------------
# Plant integration with sub-stepped LQR inner loop
# ----------------------------------------------------------------------
def _step_plant(x: np.ndarray, mu_ref: float, thr_cmd: float,
                lateral_lqr: LateralLQR, wk: Tuple[float, float]) -> np.ndarray:
    """Advance the 7-state plant one outer step (config.Ts).

    The LQR roll inner loop has a fast closed-loop pole (~-49 rad/s) that is
    under-sampled at Ts=0.1 s (|s*dt|~5 >> RK4 limit), which leaves a steady
    bank-tracking bias (e.g. 25.0 deg realised for a 26.5 deg command) and
    inner-loop ripple. We therefore integrate the plant in
    ``config.inner_loop_substeps`` sub-steps, recomputing the LQR aileron each
    sub-step, so the inner loop is sampled fast enough to track exactly. The
    outer bank reference ``mu_ref`` and ``thr_cmd`` are held over the step.
    """
    nsub = max(1, int(config.inner_loop_substeps))
    dt = config.Ts / nsub
    for _ in range(nsub):
        aileron_cmd = lateral_lqr.step(mu_ref, x[3], x[4])
        x = rk4_step_long(x, aileron_cmd, thr_cmd, dt, wk,
                          config.m_aircraft, config.k_drag,
                          config.k_thrust, config.tau_thr)
    return x


# ----------------------------------------------------------------------
# Throttle PI (anti-windup)
# ----------------------------------------------------------------------
def _throttle_step(V_actual: float, integ: float) -> Tuple[float, float]:
    err = config.Va_ref - V_actual
    integ_new = integ + err * config.Ts
    thr_cmd = float(np.clip(
        config.thr0 + config.Kp_thr * err + config.Ki_thr * integ_new,
        0.0, 1.0,
    ))
    # Conditional anti-windup: roll the integrator back when saturated
    if (thr_cmd == 0.0 and err < 0) or (thr_cmd == 1.0 and err > 0):
        integ_new -= err * config.Ts
    return thr_cmd, integ_new


# ----------------------------------------------------------------------
# Hybrid L1 ↔ MPC mode logic
# ----------------------------------------------------------------------
def _hybrid_update(x: np.ndarray, mode: str, ok_cnt: int, blend: float,
                   blend_rate: float) -> Tuple[str, int, float, float, float]:
    """Advance the hybrid state machine. Returns (mode, ok_cnt, blend, rerr, echi)."""
    _, rerr, _, echi = orbit_metrics(x[0], x[1], x[2],
                                     config.circle_C, config.circle_R,
                                     ccw=(not config.cw))
    if mode == "CAPTURE_L1":
        cond_r = rerr < config.r_err_enter_frac * config.circle_R
        cond_ang = abs(np.degrees(echi)) < config.head_align_deg
        ok_cnt = ok_cnt + 1 if (cond_r and cond_ang) else max(0, ok_cnt - 1)
        if ok_cnt >= config.hold_enter_steps:
            mode = "MPC_TRACK"
    else:  # MPC_TRACK
        if (rerr > config.r_err_exit_frac * config.circle_R
                or abs(np.degrees(echi)) > config.head_exit_deg):
            mode = "CAPTURE_L1"
            ok_cnt = 0
    target = 1.0 if mode == "MPC_TRACK" else 0.0
    blend += blend_rate if target > blend else -blend_rate
    blend = float(np.clip(blend, 0.0, 1.0))
    return mode, ok_cnt, blend, rerr, float(echi)


# ----------------------------------------------------------------------
# Main simulation loop
# ----------------------------------------------------------------------
def simulate(path: CirclePath, l1_green: L1Loiter, l1_orange: L1Loiter,
             mpc: LTVMPC_OSQP, pi: PILoiter, pid: PIDLoiter,
             lateral_lqr: LateralLQR
             ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray],
                        Dict[str, np.ndarray], Dict[str, np.ndarray],
                        Dict[str, list], Dict[str, list]]:
    steps = int(config.T_end / config.Ts)

    L1log = {k: [] for k in ("n", "e", "chi", "mu", "p", "V", "thr", "u_cmd")}
    MPClog = {k: [] for k in ("n", "e", "chi", "mu", "p", "V", "thr", "u_cmd")}
    PIlog = {k: [] for k in ("n", "e", "chi", "mu", "p", "V", "thr", "u_cmd")}
    PIDlog = {k: [] for k in ("n", "e", "chi", "mu", "p", "V", "thr", "u_cmd")}
    HYBlog = {"t": [], "mode": [], "blend": [], "rerr": [], "echi_deg": []}
    QPlog = {"iter": [], "obj": [], "status": []}

    x1 = make_start_state(config.start_pos, config.start_heading_type,
                          config.start_bank, config.Va_ref, config.thr0)
    x2 = x1.copy()
    x3 = x1.copy()
    x4 = x1.copy()
    u_prev = 0.0
    integ1 = 0.0
    integ2 = 0.0
    integ3 = 0.0
    integ4 = 0.0

    mode = "CAPTURE_L1" if config.use_hybrid_l1_mpc else "MPC_TRACK"
    ok_cnt = 0
    blend = 0.0
    blend_rate = config.Ts / max(config.Ts, float(config.blend_seconds))

    for k in range(steps):
        t = k * config.Ts
        wk = wind_at(t, state=x2)

        # ---- Throttle PI for all plants
        thr_cmd1, integ1 = _throttle_step(x1[5], integ1)
        thr_cmd2, integ2 = _throttle_step(x2[5], integ2)
        thr_cmd3, integ3 = _throttle_step(x3[5], integ3)
        thr_cmd4, integ4 = _throttle_step(x4[5], integ4)

        # ---- Bank commands
        u1 = l1_green.command(x1, path, config.Va_ref, wk)
        uL1_cap = l1_orange.command(x2, path, config.Va_ref, wk)
        uMPC, qpinfo = mpc.step(x2[:mpc.nx], wk, u_prev, V_meas=x2[5])
        u3 = pi.command(x3, path, config.Va_ref, wk)
        u4 = pid.command(x4, path, config.Va_ref, wk)

        QPlog["iter"].append(qpinfo.get("iter", np.nan))
        QPlog["obj"].append(qpinfo.get("obj", np.nan))
        QPlog["status"].append(qpinfo.get("status", ""))

        # ---- Hybrid L1 → MPC
        if config.use_hybrid_l1_mpc:
            mode, ok_cnt, blend, rerr, echi = _hybrid_update(
                x2, mode, ok_cnt, blend, blend_rate)
            u2 = (1.0 - blend) * uL1_cap + blend * uMPC
        else:
            rerr = 0.0
            echi = 0.0
            u2 = uMPC

        if config.use_cmd_filter:
            u2_applied = (1.0 - config.alpha_cmd) * u_prev + config.alpha_cmd * u2
        else:
            u2_applied = u2

        # ---- 7-state plant integration with sub-stepped LQR inner loop
        #      (LQR mu_ref → aileron recomputed each sub-step inside _step_plant)
        x1 = _step_plant(x1, u1, thr_cmd1, lateral_lqr, wk)
        x2 = _step_plant(x2, u2_applied, thr_cmd2, lateral_lqr, wk)
        x3 = _step_plant(x3, u3, thr_cmd3, lateral_lqr, wk)
        x4 = _step_plant(x4, u4, thr_cmd4, lateral_lqr, wk)

        # ---- Logging
        for log, xk, uk in ((L1log, x1, u1), (MPClog, x2, u2_applied),
                            (PIlog, x3, u3), (PIDlog, x4, u4)):
            log["n"].append(xk[0]); log["e"].append(xk[1]); log["chi"].append(xk[2])
            log["mu"].append(xk[3]); log["p"].append(xk[4])
            log["V"].append(xk[5]); log["thr"].append(xk[6])
            log["u_cmd"].append(uk)
        HYBlog["t"].append(t)
        HYBlog["mode"].append(mode)
        HYBlog["blend"].append(blend)
        HYBlog["rerr"].append(rerr)
        HYBlog["echi_deg"].append(float(np.degrees(echi)))

        u_prev = u2_applied

    # list → np.array
    for log in (L1log, MPClog, PIlog, PIDlog):
        for key in log:
            log[key] = np.array(log[key])
    return L1log, MPClog, PIlog, PIDlog, HYBlog, QPlog
