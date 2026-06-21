"""Auto-tune the loiter controllers (L1, PID, LTV-MPC) on the sim plant.

PI is excluded on purpose — it is the documented divergent negative baseline
(see mpc_nav/pi_loiter.py), so there is nothing to tune.

Each candidate is scored by running ONE controller alone on the same plant the
full simulation uses: throttle PI + sub-stepped LQR inner loop, integrated by
``mpc_nav.simulation._step_plant`` / ``_throttle_step`` so the tuner and the
sim never drift apart. Tuning happens at the operating point currently set in
``mpc_nav/config.py`` (wind, R, Va, horizon, …).

Search method (per the agreed design):
  * L1  — grid over (period, damping)            [2D, interpretable]
  * PID — grid over (Kp, Kd, Ki)                 [3D]
  * MPC — random search + Nelder-Mead refine     [8D weights, N_horizon fixed]

Objective (lower is better), identical to scripts/tune_l1.py:
    score = RMS + 0.3 * settle/10 + 0.05 * bank_rate_RMS

Output is a REPORT ONLY: a ranked table, the best parameters, the improvement
vs the current config, and ready-to-paste config.py lines. It never edits
config.py.

Run from the repo root (SITL not required — this is sim-only):
    PYTHONPATH=. python scripts/auto_tune.py            # all three
    PYTHONPATH=. python scripts/auto_tune.py l1 pid     # subset
"""
from __future__ import annotations
import sys
import io
import itertools
import contextlib
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

import mpc_nav.config as config
# Suppress the log-directory side-effect that importing io_utils (via plotting)
# would trigger; we never plot here.
config.make_animation = False

from mpc_nav.geometry import CirclePath, make_start_state, crosstrack_series
from mpc_nav.wind import wind_at
from mpc_nav.lateral_lqr import LateralLQR
from mpc_nav.l1_loiter import L1Loiter
from mpc_nav.pid_loiter import PIDLoiter
from mpc_nav.ltv_mpc import LTVMPC_OSQP, MPCWeights
from mpc_nav.simulation import _step_plant, _throttle_step
from mpc_nav.stats import rms as _rms

# ---- search budget ----------------------------------------------------------
# NOTE: one MPC evaluation is a full 150 s sim with a QP solved every step
# (~80 s wall-clock on this machine). L1/PID evals are ~0.3 s. So MPC dominates:
# (random + refine) * ~80 s. Keep these small unless you can wait — e.g.
# 20 + 30 ~= 50 evals ~= 1 hour. Raise them for a more thorough (overnight) search.
MPC_RANDOM_SAMPLES = 20       # random candidates before the local refine
MPC_REFINE_ITERS   = 30       # Nelder-Mead max iterations
RNG_SEED           = 0        # deterministic random search


@contextlib.contextmanager
def _silence():
    """Mute the per-construction prints from LateralLQR / LTVMPC_OSQP."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ======================================================================
# One closed-loop evaluation
# ======================================================================
def _evaluate(make_ctrl: Callable[[], object], lqr: LateralLQR,
              path: CirclePath, is_mpc: bool) -> Dict[str, float]:
    """Run a single controller alone on the plant; return a metric dict."""
    with _silence():
        ctrl = make_ctrl()

    x = make_start_state(config.start_pos, config.start_heading_type,
                         config.start_bank, config.Va_ref, config.thr0)
    steps = int(config.T_end / config.Ts)
    n_log = np.empty(steps)
    e_log = np.empty(steps)
    u_log = np.empty(steps)

    integ = 0.0
    u_prev = 0.0
    with _silence():
        for k in range(steps):
            t = k * config.Ts
            wk = wind_at(t, state=x)
            thr_cmd, integ = _throttle_step(x[5], integ)
            if is_mpc:
                mu_ref, _ = ctrl.step(x[:ctrl.nx], wk, u_prev, V_meas=x[5])
            else:
                mu_ref = ctrl.command(x, path, config.Va_ref, wk)
            x = _step_plant(x, mu_ref, thr_cmd, lqr, wk)
            u_prev = mu_ref
            n_log[k] = x[0]
            e_log[k] = x[1]
            u_log[k] = mu_ref

    et = crosstrack_series(path, n_log, e_log)
    ae = np.abs(et)
    n_trans = int(round(40.0 / config.Ts))
    n_ss = int(round(70.0 / config.Ts))
    brate = np.diff(np.degrees(u_log)) / config.Ts

    settle = config.T_end
    for i, v in enumerate(ae):
        if v <= 5.0:
            settle = i * config.Ts
            break

    return {
        "rms": _rms(et),
        "max": float(ae.max()),
        "iae": float(np.sum(ae) * config.Ts),
        "settle": settle,
        "trans_rms": _rms(et[:n_trans]),
        "ss_mean": float(np.mean(ae[n_ss:])),
        "brate_rms": _rms(brate),
    }


def _score(m: Dict[str, float]) -> float:
    """Composite objective (same weighting as scripts/tune_l1.py)."""
    if not np.isfinite(m["rms"]):
        return float("inf")
    return m["rms"] + 0.3 * m["settle"] / 10.0 + 0.05 * m["brate_rms"]


# ======================================================================
# Controller factories
# ======================================================================
def _l1_dir() -> int:
    return -1 if config.cw else +1


def _make_l1(period: float, damping: float) -> Callable[[], object]:
    return lambda: L1Loiter(period, damping, config.bank_limit_deg, _l1_dir())


def _make_pid(Kp: float, Kd: float, Ki: float) -> Callable[[], object]:
    return lambda: PIDLoiter(Kp, Ki, Kd, config.bank_limit_deg, _l1_dir(),
                             config.Ts, use_centripetal_ff=config.PID_centripetal_ff)


# MPC weight vector order: [w_et, w_echi, w_mu, w_u, w_et_T, w_echi_T, w_du, w_mu_Term]
MPC_KEYS = ["w_et", "w_echi", "w_mu", "w_u", "w_et_T", "w_echi_T", "w_du", "w_mu_Term"]
MPC_BOUNDS = np.array([
    (0.5, 20.0),    # w_et
    (0.5, 20.0),    # w_echi
    (0.01, 5.0),    # w_mu
    (0.1, 10.0),    # w_u
    (0.01, 20.0),   # w_et_T
    (0.01, 20.0),   # w_echi_T
    (10.0, 1000.0), # w_du
    (0.01, 20.0),   # w_mu_Term
])


def _make_mpc(path: CirclePath, lqr: LateralLQR, wv) -> Callable[[], object]:
    w_et, w_echi, w_mu, w_u, w_et_T, w_echi_T, w_du, w_mu_Term = wv
    weights = MPCWeights(w_et, w_echi, w_mu, w_u, w_et_T, w_echi_T)
    return lambda: LTVMPC_OSQP(
        Ts=config.Ts, N=config.N_horizon, Va_init=config.Va_ref,
        tau_mu=config.tau_mu, bank_limit_deg=config.bank_limit_deg,
        slew_limit_deg_s=config.slew_limit_deg_s, weights=weights, path=path,
        w_du=float(w_du), w_mu_Term=float(w_mu_Term), use_groundspeed_mu_ss=True,
        Va_nominal=config.Va_nominal, use_w_du_scaling=config.use_w_du_scaling,
        lateral_lqr=lqr, a_p_roll=config.a_p, b_p_roll=config.b_p,
    )


# ======================================================================
# Per-controller tuners
# ======================================================================
def tune_l1(lqr, path) -> Tuple[dict, dict]:
    periods = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0]
    dampings = [0.65, 0.70, 0.75, 0.80, 0.85]
    best = None
    print(f"[L1 ] grid {len(periods)}x{len(dampings)} = {len(periods)*len(dampings)} runs")
    for period, damping in itertools.product(periods, dampings):
        m = _evaluate(_make_l1(period, damping), lqr, path, is_mpc=False)
        cand = {"params": {"L1_period": period, "L1_damping": damping},
                "score": _score(m), "metrics": m}
        if best is None or cand["score"] < best["score"]:
            best = cand
    return best, {"L1_period": config.L1_period, "L1_damping": config.L1_damping}


def tune_pid(lqr, path) -> Tuple[dict, dict]:
    Kps = [0.005, 0.010, 0.015, 0.020, 0.025, 0.035, 0.050, 0.060]
    Kds = [0.02, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25]
    Kis = [0.0, 0.0005, 0.001, 0.002]
    n = len(Kps) * len(Kds) * len(Kis)
    best = None
    print(f"[PID] grid {len(Kps)}x{len(Kds)}x{len(Kis)} = {n} runs")
    for i, (Kp, Kd, Ki) in enumerate(itertools.product(Kps, Kds, Kis)):
        m = _evaluate(_make_pid(Kp, Kd, Ki), lqr, path, is_mpc=False)
        cand = {"params": {"PID_Kp": Kp, "PID_Kd": Kd, "PID_Ki": Ki},
                "score": _score(m), "metrics": m}
        if best is None or cand["score"] < best["score"]:
            best = cand
        if (i + 1) % 64 == 0:
            print(f"      {i+1}/{n}  best score so far {best['score']:.3f}")
    return best, {"PID_Kp": config.PID_Kp, "PID_Kd": config.PID_Kd, "PID_Ki": config.PID_Ki}


def tune_mpc(lqr, path) -> Tuple[dict, dict]:
    rng = np.random.default_rng(RNG_SEED)
    lo, hi = MPC_BOUNDS[:, 0], MPC_BOUNDS[:, 1]
    log_lo, log_hi = np.log(lo), np.log(hi)

    def obj(wv) -> float:
        wv = np.clip(wv, lo, hi)                      # keep candidates feasible
        m = _evaluate(_make_mpc(path, lqr, wv), lqr, path, is_mpc=True)
        return _score(m)

    # ---- random search (log-uniform within bounds) ----
    print(f"[MPC] random search {MPC_RANDOM_SAMPLES} samples (8D weights)")
    best_w = np.array([config.w_et, config.w_echi, config.w_mu, config.w_u,
                       config.w_et_T, config.w_echi_T, config.w_du, config.w_mu_Term])
    best_s = obj(best_w)                              # seed with current config
    for j in range(MPC_RANDOM_SAMPLES):
        cand = np.exp(rng.uniform(log_lo, log_hi))
        s = obj(cand)
        if s < best_s:
            best_s, best_w = s, cand
        if (j + 1) % 15 == 0:
            print(f"      {j+1}/{MPC_RANDOM_SAMPLES}  best score so far {best_s:.3f}")

    # ---- Nelder-Mead local refine from the best random point ----
    print(f"[MPC] Nelder-Mead refine (<= {MPC_REFINE_ITERS} iters)")
    res = minimize(obj, best_w, method="Nelder-Mead",
                   options={"maxiter": MPC_REFINE_ITERS, "xatol": 1e-2, "fatol": 1e-3})
    if res.fun < best_s:
        best_s, best_w = float(res.fun), np.clip(res.x, lo, hi)

    m = _evaluate(_make_mpc(path, lqr, best_w), lqr, path, is_mpc=True)
    params = {k: float(v) for k, v in zip(MPC_KEYS, best_w)}
    cur = {k: getattr(config, k) for k in MPC_KEYS}
    return {"params": params, "score": best_s, "metrics": m}, cur


# ======================================================================
# Reporting
# ======================================================================
def _fmt_metrics(m: dict) -> str:
    return (f"RMS={m['rms']:.3f}m  MAX={m['max']:.2f}m  IAE={m['iae']:.1f}  "
            f"settle={m['settle']:.1f}s  SS={m['ss_mean']:.3f}m  "
            f"bankrate={m['brate_rms']:.2f}deg/s")


def _report(name: str, best: dict, current_params: dict,
            lqr, path, is_mpc: bool) -> None:
    cur_m = _evaluate(
        (_make_mpc(path, lqr, [current_params[k] for k in MPC_KEYS]) if is_mpc
         else (_make_l1(current_params["L1_period"], current_params["L1_damping"]) if name == "L1"
               else _make_pid(current_params["PID_Kp"], current_params["PID_Kd"], current_params["PID_Ki"]))),
        lqr, path, is_mpc)
    cur_s = _score(cur_m)
    drms = cur_m["rms"] - best["metrics"]["rms"]
    pct = drms / cur_m["rms"] * 100 if cur_m["rms"] else 0.0

    print(f"\n{'='*72}\n{name} - best score {best['score']:.3f} (current {cur_s:.3f})")
    print(f"  current : {_fmt_metrics(cur_m)}")
    print(f"  best    : {_fmt_metrics(best['metrics'])}")
    print(f"  delta RMS {drms:+.3f} m ({pct:+.1f}%)")
    print(f"  config.py:")
    for k, v in best["params"].items():
        cur_v = current_params[k]
        mark = "" if abs(v - cur_v) < 1e-9 else "   # was %g" % cur_v
        print(f"    {k:14s} = {v:.6g}{mark}")


# ======================================================================
def main(which: List[str]) -> None:
    # Line-buffer stdout so progress shows live when piped to a file/pager
    # (otherwise Python full-buffers and the log stays empty until the end).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    path = CirclePath(config.circle_C, config.circle_R, cw=config.cw)
    with _silence():
        lqr = LateralLQR()

    wn, we = config.wind_mean
    print(f"Auto-tune @ Va={config.Va_ref:.0f} m/s  R={config.circle_R:.0f} m  "
          f"wind={np.hypot(wn, we):.0f} m/s {config.wind_mode}  "
          f"score = RMS + 0.3*settle/10 + 0.05*bankrate")

    if "l1" in which:
        best, cur = tune_l1(lqr, path)
        _report("L1", best, cur, lqr, path, is_mpc=False)
    if "pid" in which:
        best, cur = tune_pid(lqr, path)
        _report("PID", best, cur, lqr, path, is_mpc=False)
    if "mpc" in which:
        best, cur = tune_mpc(lqr, path)
        _report("MPC", best, cur, lqr, path, is_mpc=True)

    print(f"\n{'='*72}\nReport only - config.py was NOT modified. "
          f"Paste the lines above to apply.")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    valid = {"l1", "pid", "mpc"}
    if "pi" in args:
        print("PI is the divergent negative baseline and is not tunable — skipping.")
    sel = [a for a in args if a in valid] or ["l1", "pid", "mpc"]
    main(sel)
