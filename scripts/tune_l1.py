"""L1 loiter parameter sweep — find best (L1_period, L1_damping).

Run from repo root:
    PYTHONPATH=. python scripts/tune_l1.py

Only the L1 plant is simulated (no MPC/PI/PID overhead), so the full sweep
of 45 combinations finishes in a few seconds.
"""
import sys
import numpy as np

import mpc_nav.config as config
from mpc_nav.geometry import CirclePath, make_start_state, crosstrack_series
from mpc_nav.wind import wind_at
from mpc_nav.dynamics import rk4_step_long
from mpc_nav.lateral_lqr import LateralLQR
from mpc_nav.l1_loiter import L1Loiter

# Suppress log-directory creation side-effect from io_utils
config.make_animation = False


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2)))


def _settle(e: np.ndarray, tol: float = 5.0) -> float:
    for i, v in enumerate(e):
        if abs(v) <= tol:
            return i * config.Ts
    return config.T_end


def run_l1(period: float, damping: float) -> dict:
    """Simulate L1-only and return a metric dict."""
    path = CirclePath(config.circle_C, config.circle_R, cw=config.cw)
    l1_dir = -1 if config.cw else +1
    l1 = L1Loiter(period, damping, config.bank_limit_deg, l1_dir)
    lqr = LateralLQR()

    x = make_start_state(config.start_pos, config.start_heading_type,
                         config.start_bank, config.Va_ref, config.thr0)

    steps = int(config.T_end / config.Ts)
    n_log = np.empty(steps)
    e_log = np.empty(steps)
    u_log = np.empty(steps)

    integ = 0.0
    nsub = max(1, int(config.inner_loop_substeps))
    dt_sub = config.Ts / nsub

    for k in range(steps):
        t = k * config.Ts
        wk = wind_at(t, state=x)

        # Throttle PI with anti-windup
        err_v = config.Va_ref - x[5]
        integ += err_v * config.Ts
        thr_cmd = float(np.clip(
            config.thr0 + config.Kp_thr * err_v + config.Ki_thr * integ,
            0.0, 1.0))
        if (thr_cmd == 0.0 and err_v < 0) or (thr_cmd == 1.0 and err_v > 0):
            integ -= err_v * config.Ts

        mu_ref = l1.command(x, path, config.Va_ref, wk)

        for _ in range(nsub):
            ail = lqr.step(mu_ref, x[3], x[4])
            x = rk4_step_long(x, ail, thr_cmd, dt_sub, wk,
                              config.m_aircraft, config.k_drag,
                              config.k_thrust, config.tau_thr)

        n_log[k] = x[0]
        e_log[k] = x[1]
        u_log[k] = mu_ref

    et = crosstrack_series(path, n_log, e_log)
    ae = np.abs(et)
    n_trans = int(round(40.0 / config.Ts))
    n_ss    = int(round(70.0 / config.Ts))
    brate   = np.diff(np.degrees(u_log)) / config.Ts

    return {
        "rms":       _rms(et),
        "max":       float(ae.max()),
        "iae":       float(np.sum(ae) * config.Ts),
        "settle":    _settle(et),
        "trans_rms": _rms(et[:n_trans]),
        "ss_mean":   float(np.mean(ae[n_ss:])),
        "brate_rms": _rms(brate),
    }


def main():
    periods  = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0]
    dampings = [0.65, 0.70, 0.75, 0.80, 0.85]

    hdr = (f"{'T':>5} {'zeta':>5} {'RMS':>7} {'MAX':>7} {'IAE':>8} "
           f"{'Settle':>7} {'SS':>7} {'BkRt':>7}")
    print("L1 parameter sweep  (Va=%.0f m/s, R=%.0f m, wind=%.0f m/s %s)" % (
        config.Va_ref, config.circle_R,
        float(np.hypot(*config.wind_mean)), config.wind_mode))
    print(hdr)
    print("-" * len(hdr))

    results = []
    for period in periods:
        for damping in dampings:
            m = run_l1(period, damping)
            # composite score: weight RMS most, then settle + smoothness
            score = m["rms"] + 0.3 * m["settle"] / 10.0 + 0.05 * m["brate_rms"]
            results.append((score, period, damping, m))
            print(f"{period:5.1f} {damping:5.2f} {m['rms']:7.2f} {m['max']:7.2f} "
                  f"{m['iae']:8.1f} {m['settle']:7.1f} {m['ss_mean']:7.3f} "
                  f"{m['brate_rms']:7.2f}")

    results.sort()
    best_score, bp, bz, bm = results[0]

    print("\n" + "=" * len(hdr))
    print(f"BEST:  L1_period={bp:.1f}  L1_damping={bz:.2f}")
    print(f"  RMS={bm['rms']:.3f} m  MAX={bm['max']:.3f} m  settle={bm['settle']:.1f} s"
          f"  SS={bm['ss_mean']:.3f} m  BkRate={bm['brate_rms']:.2f} deg/s")

    # Compare against current config values
    curr = run_l1(config.L1_period, config.L1_damping)
    print(f"\nCurrent (T={config.L1_period}, z={config.L1_damping}):")
    print(f"  RMS={curr['rms']:.3f} m  MAX={curr['max']:.3f} m  settle={curr['settle']:.1f} s"
          f"  SS={curr['ss_mean']:.3f} m  BkRate={curr['brate_rms']:.2f} deg/s")

    drms = curr["rms"] - bm["rms"]
    print(f"\nImprovement:  delta_RMS={drms:+.3f} m  ({drms/curr['rms']*100:+.1f}%)")

    print(f"\nUpdate config.py:")
    print(f"  L1_period  = {bp}")
    print(f"  L1_damping = {bz}")

    return bp, bz, bm


if __name__ == "__main__":
    main()
