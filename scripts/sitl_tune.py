"""SITL-based auto-tuner for the loiter controllers (L1, PID, MPC) — NOT PI.

This is the SITL counterpart of scripts/auto_tune.py. Where auto_tune.py scores
candidates against the fast sim plant (mpc_nav/config.py operating point),
THIS script scores them by actually FLYING them on the live SITL vehicle, at the
driver's operating point (hardware/mavlink_driver.py: R=180, FBWB roll loop, …).
That matters because the SITL plant (autopilot FBWB) differs from the sim's LQR,
so the optimum is in a different place — see CLAUDE.md / .docs/sitl_test_protocol.

It reuses scripts/sitl_auto_eval.py wholesale: the same MAVLink connection,
reader/control/sender threads, capture gating, circle recording and metrics.
Controller PARAMETERS are changed LIVE between runs by mutating the controller
objects held in the driver's Shared state (S._l1 / S._pid / S._mpc). ctl_worker
reads those objects' attributes every loop, so a mutation takes effect on the
next control tick — no driver restart, no reconnection.

Because every evaluation is a real flight (capture + a few orbits + RTL settle ~
3-4 min each), the search is a BOUNDED coordinate descent: from the driver's
current values, try a couple of neighbours per parameter, keep improvements,
and stop at FLIGHT_BUDGET. The flight count is predictable and printed up front.

Output is a REPORT ONLY: best parameters as ready-to-paste mavlink_driver.py
constants. It never edits the driver.

PRECONDITION (same as sitl_auto_eval): the vehicle must already be AIRBORNE near
the loiter centre. This script takes roll in FBWB via RC1 override; it does NOT
take off for you.

Run from the repo root, with SITL up:
    PYTHONPATH=. python scripts/sitl_tune.py pid        # tune one controller
    PYTHONPATH=. python scripts/sitl_tune.py l1 mpc
    PYTHONPATH=. python scripts/sitl_tune.py             # all three (long!)
"""
from __future__ import annotations
import os
import sys
import math
import time
import threading
from typing import Dict, List, Tuple

# Make the repo root importable when run as `python scripts/sitl_tune.py`
# (that puts scripts/ on sys.path, not the root where the packages live).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import scripts.sitl_auto_eval as se
from scripts.sitl_auto_eval import (
    wait_airborne, rtl_settle, engage_fbwb, reset_controllers,
    fly_and_record, last_circle_mask, metrics,
)
import hardware.mavlink_driver as d

# ===================== TUNING PARAMETERS (EDIT HERE) ==================
TUNE_CIRCLES   = 2        # orbits flown per candidate (fewer than eval's 3 = faster)
SCORE_LAST     = 1.0      # score over the last N circle(s)
FLIGHT_BUDGET  = 12       # hard cap on number of candidate flights per controller
BANKRATE_W     = 0.05     # smoothness weight in the score (score = rms + w*bankrate)
DIVERGE_PENALTY = 1.0e6   # added if a candidate fails to complete its circles

# Neighbour generators for coordinate descent (relative to the current best).
# Each returns the candidate values to try for that one parameter.
L1_STEPS = {
    "period":  lambda v: [v * 0.75, v * 1.25],
    "damping": lambda v: [max(0.50, v - 0.10), min(0.95, v + 0.10)],
}
PID_STEPS = {
    "Kp": lambda v: [v * 0.5, v * 2.0],
    "Kd": lambda v: [v * 0.5, v * 2.0],
    "Ki": lambda v: ([0.0005, 0.002] if v == 0.0 else [v * 0.5, v * 2.0]),
}
# MPC: only the highest-leverage few — tau (FBWB fidelity), w_du (smoothness),
# w_et/w_echi (tracking ratio). Extend if you have flight budget to spare.
MPC_STEPS = {
    "tau":    lambda v: [v * 0.6, v * 1.5],
    "w_du":   lambda v: [v * 0.5, v * 2.0],
    "w_et":   lambda v: [v * 0.5, v * 2.0],
    "w_echi": lambda v: [v * 0.5, v * 2.0],
}

# Map internal keys -> the mavlink_driver.py constant names (for the report).
DRIVER_NAMES = {
    "l1":  {"period": "L1_PERIOD", "damping": "L1_DAMPING"},
    "pid": {"Kp": "PID_KP", "Kd": "PID_KD", "Ki": "PID_KI"},
    "mpc": {"tau": "MPC_TAU_MU", "w_du": "MPC_W_DU", "w_et": "MPC_W_ET", "w_echi": "MPC_W_ECHI"},
}


# ===================== live parameter access =========================
def current_params(mode: str) -> Dict[str, float]:
    if mode == "l1":
        return {"period": float(d.L1_PERIOD), "damping": float(d.L1_DAMPING)}
    if mode == "pid":
        return {"Kp": float(d.PID_KP), "Kd": float(d.PID_KD), "Ki": float(d.PID_KI)}
    if mode == "mpc":
        return {"tau": float(d.MPC_TAU_MU), "w_du": float(d.MPC_W_DU),
                "w_et": float(d.MPC_W_ET), "w_echi": float(d.MPC_W_ECHI)}
    raise ValueError(mode)


def apply_params(S: "d.Shared", mode: str, p: Dict[str, float]) -> None:
    """Mutate the LIVE controller object so ctl_worker picks it up next tick."""
    if mode == "l1":
        S._l1.period = p["period"]
        S._l1.damping = p["damping"]
    elif mode == "pid":
        S._pid.Kp, S._pid.Kd, S._pid.Ki = p["Kp"], p["Kd"], p["Ki"]
    elif mode == "mpc":
        S._mpc.tau = p["tau"]
        S._mpc.w_du = p["w_du"]
        S._mpc.W.w_et = p["w_et"]
        S._mpc.W.w_echi = p["w_echi"]
        # Force a clean OSQP re-setup so a previous weighting's warm-start
        # doesn't bias this candidate.
        S._mpc._prob = None
        S._mpc._z_prev = None
        S._mpc._y_prev = None
        S._mpc._u_opt_seq = None


def steps_for(mode: str) -> dict:
    return {"l1": L1_STEPS, "pid": PID_STEPS, "mpc": MPC_STEPS}[mode]


# ===================== scoring =======================================
def score(met: dict) -> float:
    if not met.get("complete", False) or not math.isfinite(met["rms"]):
        return DIVERGE_PENALTY + (met["rms"] if math.isfinite(met["rms"]) else 1e3)
    return met["rms"] + BANKRATE_W * met["bank_rate_rms"]


def _fmt(met: dict) -> str:
    return (f"RMS={met['rms']:.3f}m  peak={met['max']:.3f}m  IAE={met['iae']:.1f}  "
            f"bankrate={met['bank_rate_rms']:.2f}deg/s  "
            f"{'OK' if met['complete'] else 'INCOMPLETE'}")


# ===================== one candidate flight ==========================
def fly_candidate(m, S: "d.Shared", mode: str, path, p: Dict[str, float]) -> Tuple[float, dict]:
    rtl_settle(m, S)                     # common base state
    apply_params(S, mode, p)
    reset_controllers(S)                 # clear PI/PID integ + blend state
    engage_fbwb(m, S)                    # take roll in FBWB
    rec = fly_and_record(S, mode, path)
    mask = last_circle_mask(rec)
    complete = bool(len(rec["phi"]) and rec["phi"][-1] >= se.N_CIRCLES * 2 * math.pi - 1e-6)
    met = metrics(rec, mask, complete)
    return score(met), met


# ===================== bounded coordinate descent ====================
def tune_controller(m, S: "d.Shared", mode: str, path) -> Tuple[Dict[str, float], dict]:
    steps = steps_for(mode)
    best_p = current_params(mode)

    print(f"\n[TUNE] === {mode.upper()} === start {best_p}")
    flights = 1
    best_s, best_m = fly_candidate(m, S, mode, path, best_p)
    print(f"[TUNE] baseline  score {best_s:.3f}  {_fmt(best_m)}")

    for key, gen in steps.items():
        for val in gen(best_p[key]):
            if flights >= FLIGHT_BUDGET:
                print(f"[TUNE] flight budget {FLIGHT_BUDGET} reached - stopping early")
                return best_p, best_m
            cand = dict(best_p)
            cand[key] = val
            flights += 1
            print(f"[TUNE] flight {flights}/{FLIGHT_BUDGET}: {key}={val:.4g}")
            s, met = fly_candidate(m, S, mode, path, cand)
            tag = "accept" if s < best_s else "reject"
            print(f"[TUNE]   score {s:.3f}  {_fmt(met)}  -> {tag}")
            if s < best_s:
                best_s, best_m, best_p = s, met, cand

    return best_p, best_m


# ===================== report ========================================
def report(mode: str, best_p: Dict[str, float], best_m: dict,
           start_p: Dict[str, float]) -> None:
    names = DRIVER_NAMES[mode]
    print(f"\n{'='*70}\n{mode.upper()} best - {_fmt(best_m)}")
    print("  mavlink_driver.py:")
    for k, v in best_p.items():
        was = start_p[k]
        mark = "" if abs(v - was) < 1e-12 else f"   # was {was:g}"
        print(f"    {names[k]:14s} = {v:.6g}{mark}")


# ===================== main ==========================================
def main(which: List[str]) -> None:
    n_per = {m: 1 + sum(len(steps_for(m)[k](current_params(m)[k])) for k in steps_for(m))
             for m in which}
    total = sum(min(FLIGHT_BUDGET, n) for n in n_per.values())
    print(f"[TUNE] controllers: {which}")
    print(f"[TUNE] up to {total} candidate flights "
          f"({TUNE_CIRCLES} circles each, capped at {FLIGHT_BUDGET}/controller)")
    print(f"[TUNE] estimated rough time: ~{total * 3.5:.0f}-{total * 5:.0f} min "
          f"(flight + RTL settle per candidate)")

    se.N_CIRCLES = TUNE_CIRCLES
    se.PLOT_LAST_CIRCLES = SCORE_LAST

    m = d.connect_mavlink(d.MAVLINK_URL)
    S = d.Shared()
    S.tx_enable = False
    th_rx = threading.Thread(target=d.state_reader, args=(m, S), daemon=True)
    th_ct = threading.Thread(target=d.ctl_worker,   args=(m, S), daemon=True)
    th_tx = threading.Thread(target=d.att_sender,   args=(m, S, d.TX_HZ), daemon=True)
    th_rx.start(); th_ct.start(); th_tx.start()

    wait_airborne(S)
    path = d.CirclePath(d.LOITER_C, d.LOITER_R, cw=d.LOITER_CW)

    results = {}
    try:
        for mode in which:
            start_p = current_params(mode)
            best_p, best_m = tune_controller(m, S, mode, path)
            results[mode] = (best_p, best_m, start_p)
    finally:
        # Always hand the vehicle back to the autopilot.
        S.tx_enable = False
        try:
            d.set_mode(m, "RTL")
            print("[TUNE] RTL requested.")
        except Exception:
            pass
        S.stop.set()
        time.sleep(0.3)

    for mode, (best_p, best_m, start_p) in results.items():
        report(mode, best_p, best_m, start_p)
    print(f"\n{'='*70}\nReport only - mavlink_driver.py was NOT modified. "
          f"Paste the lines above to apply.")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:]]
    valid = {"l1", "pid", "mpc"}
    if "pi" in args:
        print("PI is the divergent negative baseline and is not tunable - skipping.")
    sel = [a for a in args if a in valid] or ["l1", "pid", "mpc"]
    main(sel)
