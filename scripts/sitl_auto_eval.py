#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SITL automated controller evaluation.

Flies each lateral controller (L1 / PID / PI / MPC / blend) on the SAME airborne
vehicle for a fixed number of full circles, records the trajectory, and emits:

  * one cross-track-error plot per controller (last circle only),
  * one Ref-vs-trajectory plot per controller (last circle only),
  * a combined cross-track-error comparison plot,
  * a markdown report with a per-controller metrics table + embedded plots.

It reuses mavlink_driver.py wholesale: the same MAVLink connection, state reader,
control worker and RC-override sender. Controllers are switched LIVE by writing
`Shared.lat_mode` (ctl_worker reads it every loop), so no restart is needed
between runs. Because only the LAST circle is plotted/scored, each run's capture
transient (and its different start point) is excluded — the comparison is a fair
steady-state one.

PRECONDITION: the vehicle must already be AIRBORNE (e.g. take off in AUTO/TAKEOFF/
FBWA and climb above MIN_AIRBORNE_ALT near the loiter centre). This script then
takes roll control in FBWB via RC1 override. It does NOT take off for you.

Run from the repo root, with SITL up:
    python scripts/sitl_auto_eval.py
"""
import os, sys, math, time, csv, threading, datetime as _dt

# Make the repo root importable when run as `python scripts/sitl_auto_eval.py`
# (that puts scripts/ on sys.path, not the root where the hardware package lives).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import psutil

import matplotlib
matplotlib.use("Agg")              # headless: render straight to PNG
import matplotlib.pyplot as plt

from hardware import mavlink_driver as d   # connection, threads, Shared, constants

# ====================== EVALUATION PARAMETERS (EDIT HERE) ==============
MODES             = ["l1", "pid", "mpc", "blend"]  # controllers to test, in order
N_CIRCLES         = 3          # full orbits to fly per controller
PLOT_LAST_CIRCLES = 1.5        # how many trailing orbits to plot/score
WIND_LABEL        = "windless" # shown in titles/report (set SIM_WIND_SPD=0 in SITL)

SAMPLE_HZ         = 20.0       # trajectory recording rate [Hz]

# Each run is preceded by RTL so every controller starts from the SAME base state.
RTL_SETTLE_S      = 30.0       # seconds in RTL between runs (roll override released)

# Capture gating: start counting circles only once the vehicle is ON the orbit.
CAPTURE_TOL_FRAC  = 0.05       # |rho-R| < 5% R counts as "captured"
CAPTURE_HOLD_S    = 3.0        # ...held this long
CAPTURE_TIMEOUT_S = 120.0      # give up waiting for capture, count anyway

COUNT_TIMEOUT_S   = 300.0      # safety cap on the counting phase per controller
MIN_AIRBORNE_ALT  = 20.0       # [m] consider airborne above this altitude
SETTLE_AFTER_TX_S = 3.0        # let FBWB + RC override settle after re-engaging

RUN_DIR = _dt.datetime.now().strftime("logs/auto_eval_%Y%m%d-%H%M%S")

# CSV schema — IDENTICAL to mavlink_driver.py's logger_thread, so each controller's
# per-run log matches the files under logs/logs_run_*/.
CSV_FIELDS = [
    "t", "mode_fc", "lat_mode", "hyb_phase", "hyb_blend",
    "hz_rx", "hz_ctl", "hz_tx",
    "N", "E", "vN", "vE", "Vg", "Va",
    "roll_deg", "pitch_deg", "yaw_deg", "psi_deg",
    "cog_deg", "alt", "vz", "thr",
    "rho", "rerr", "echi_deg",
    "cmd_roll_deg", "uL1_deg", "uMPC_deg", "uPI_deg", "uPID_deg",
    "qp_status", "qp_iter", "qp_fallback",
    "exec_time_ms", "jitter_ms", "cpu_percent", "mav_rx_latency_ms",
]


# ====================== Helpers =======================================
def wait_airborne(S: "d.Shared") -> None:
    print(f"[EVAL] Waiting for airborne (alt > {MIN_AIRBORNE_ALT:.0f} m) + fresh state ...")
    while True:
        with S.state_lock:
            st = d.NavState(**vars(S.state))
        fresh = (time.monotonic() - st.ts) < 1.0
        if fresh and st.alt > MIN_AIRBORNE_ALT and st.Vg > 3.0:
            print(f"[EVAL] Airborne: alt={st.alt:.0f} m  Vg={st.Vg:.1f} m/s  Va={st.Va:.1f} m/s")
            return
        time.sleep(0.5)


def rtl_settle(m, S: "d.Shared", seconds: float = RTL_SETTLE_S) -> None:
    """Release roll control and sit in RTL for `seconds` so every run starts
    from the same base state (RTL loiter near home / circle centre)."""
    S.tx_enable = False              # att_sender now releases RC1 override
    try:
        d.set_mode(m, "RTL")
    except Exception:
        pass
    print(f"[EVAL]     RTL settle {seconds:.0f}s (common base) ...")
    time.sleep(seconds)


def engage_fbwb(m, S: "d.Shared") -> None:
    """Switch to FBWB and re-enable the RC1 roll override, then let it settle."""
    try:
        d.set_mode(m, "FBWB")
    except Exception:
        pass
    S.tx_enable = True
    time.sleep(SETTLE_AFTER_TX_S)


def reset_controllers(S: "d.Shared") -> None:
    """Reset per-controller state so each run starts clean."""
    for c in (S._pi, S._pid):
        try:
            if c is not None:
                c.reset()
        except Exception:
            pass
    # blend state machine
    S.hybrid_phase = "CAPTURE_L1"
    S._blend_raw = 0.0
    S.blend = 0.0
    S._ok_cnt = 0


def fly_and_record(S: "d.Shared", mode: str, path) -> dict:
    """Switch to `mode`, wait for capture, then record N_CIRCLES orbits."""
    Cn, Ce = path.C            # circle centre (North, East)
    R = float(path.R)
    cap_tol = CAPTURE_TOL_FRAC * R

    reset_controllers(S)
    S.lat_mode = mode
    print(f"[EVAL] >>> mode={mode}  (capture tol {cap_tol:.1f} m)")

    # --- capture phase ---
    t0 = time.monotonic()
    hold_start = None
    while True:
        time.sleep(1.0 / SAMPLE_HZ)
        rerr = S.rerr
        captured = (rerr == rerr) and (rerr < cap_tol)   # rerr==rerr filters NaN
        if captured:
            hold_start = hold_start or time.monotonic()
            if time.monotonic() - hold_start >= CAPTURE_HOLD_S:
                print(f"[EVAL]     captured (rerr={rerr:.2f} m), counting {N_CIRCLES} circles")
                break
        else:
            hold_start = None
        if time.monotonic() - t0 > CAPTURE_TIMEOUT_S:
            print(f"[EVAL]     capture TIMEOUT (rerr={rerr:.2f} m) — counting anyway")
            break

    # --- counting / recording phase ---
    rec = {k: [] for k in ("t", "N", "E", "abs_err", "bank_deg", "phi")}
    csv_rows = []
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)            # prime the CPU meter (first call = 0)
    theta_prev = None
    phi_acc = 0.0
    t_start = time.monotonic()
    target = N_CIRCLES * 2.0 * math.pi
    while True:
        time.sleep(1.0 / SAMPLE_HZ)
        with S.state_lock:
            st = d.NavState(**vars(S.state))
        with S.cmd_lock:
            cmd = float(S.cmd_roll)

        tnow_rel = time.monotonic() - t_start
        dn, de = st.N - Cn, st.E - Ce
        theta = math.atan2(de, dn)
        if theta_prev is not None:
            phi_acc += abs(d._wrap(theta - theta_prev))   # unsigned cumulative angle
        theta_prev = theta
        rho = math.hypot(dn, de)

        rec["t"].append(tnow_rel)
        rec["N"].append(st.N); rec["E"].append(st.E)
        rec["abs_err"].append(abs(rho - R))
        rec["bank_deg"].append(math.degrees(cmd))
        rec["phi"].append(phi_acc)

        # full driver-schema CSV row
        cpu = proc.cpu_percent(interval=None)
        csv_rows.append([
            f"{tnow_rel:.3f}", st.mode, mode, S.hybrid_phase, f"{S.blend:.3f}",
            f"{S.hz_rx:.1f}", f"{S.hz_ctl:.1f}", f"{S.hz_tx:.1f}",
            f"{st.N:.2f}", f"{st.E:.2f}", f"{st.vN:.2f}", f"{st.vE:.2f}", f"{st.Vg:.2f}", f"{st.Va:.2f}",
            f"{math.degrees(st.roll):.2f}", f"{math.degrees(st.pitch):.2f}",
            f"{math.degrees(st.yaw):.2f}", f"{math.degrees(st.psi):.2f}",
            (f"{st.cog_deg:.1f}" if st.cog_deg is not None else ""),
            f"{st.alt:.1f}", f"{st.vz:.2f}", f"{st.throttle:.2f}",
            f"{rho:.2f}", f"{S.rerr:.2f}", f"{S.echi_deg:.1f}",
            f"{math.degrees(cmd):.2f}", f"{math.degrees(S.uL1_last):.2f}",
            f"{math.degrees(S.uMPC_last):.2f}", f"{math.degrees(S.uPI_last):.2f}",
            f"{math.degrees(S.uPID_last):.2f}",
            S.qp_status, S.qp_iter, S.qp_fallback,
            f"{S.exec_time_ctl*1000:.2f}", f"{S.jitter_ctl*1000:.2f}",
            f"{cpu:.1f}", f"{S.mav_rx_latency*1000:.2f}",
        ])

        if phi_acc >= target:
            print(f"[EVAL]     done: {N_CIRCLES} circles in {rec['t'][-1]:.1f} s, "
                  f"{len(rec['t'])} samples")
            break
        if time.monotonic() - t_start > COUNT_TIMEOUT_S:
            print(f"[EVAL]     count TIMEOUT after {rec['t'][-1]:.1f} s "
                  f"(phi={phi_acc:.1f}/{target:.1f} rad) — likely divergent")
            break

    # per-controller CSV log (same columns as logs/logs_run_*/)
    csv_path = os.path.join(RUN_DIR, f"{mode}.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(CSV_FIELDS)
        wcsv.writerows(csv_rows)
    print(f"[EVAL]     CSV: {csv_path} ({len(csv_rows)} rows)")

    for k in rec:
        rec[k] = np.asarray(rec[k], dtype=float)
    return rec


def last_circle_mask(rec: dict) -> np.ndarray:
    phi = rec["phi"]
    if len(phi) == 0:
        return np.zeros(0, dtype=bool)
    phi_end = phi[-1]
    span = 2.0 * math.pi * PLOT_LAST_CIRCLES
    return phi >= max(0.0, phi_end - span)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def metrics(rec: dict, mask: np.ndarray, complete: bool) -> dict:
    ae = rec["abs_err"][mask]
    t = rec["t"][mask]
    bank = rec["bank_deg"][mask]
    if len(t) > 1:
        dt = np.diff(t)
        dt[dt <= 0] = np.nan
        brate = np.diff(bank) / dt
        brate_rms = _rms(brate[~np.isnan(brate)])
        period = float(t[-1] - t[0])
        iae = float(np.trapz(ae, t))   # integral of abs error over time [m·s]
    else:
        brate_rms = float("nan"); period = float("nan"); iae = float("nan")
    return {
        "rms": _rms(ae),
        "max": float(ae.max()) if len(ae) else float("nan"),
        "mean": float(ae.mean()) if len(ae) else float("nan"),
        "iae": iae,
        "bank_rate_rms": brate_rms,
        "circle_period_s": period,
        "complete": complete,
    }


# ====================== Plotting ======================================
def plot_crosstrack(mode: str, rec: dict, mask: np.ndarray, out: str) -> None:
    ae = rec["abs_err"][mask]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(ae) + 1), ae, color="#3a6ea5", lw=1.6, label="Cross-track error")
    ax.set_title(f"{mode.upper()} crosstrack error {WIND_LABEL}")
    ax.set_xlabel("t"); ax.set_ylabel("m")
    ax.grid(True, axis="y", alpha=0.4)
    ax.margins(x=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(mode: str, rec: dict, mask: np.ndarray, path, out: str) -> None:
    Cn, Ce = path.C
    R = float(path.R)
    ang = np.linspace(0, 2 * math.pi, 400)
    ref_x = Ce + R * np.cos(ang)        # East on x
    ref_y = Cn + R * np.sin(ang)        # North on y
    traj_x = rec["E"][mask]
    traj_y = rec["N"][mask]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(ref_x, ref_y, color="#c0392b", lw=1.6, label="Ref")
    ax.plot(traj_x, traj_y, color="#3a6ea5", lw=1.8, label="Trajectory")
    ax.set_title(f"{mode.upper()} - Trajectory")
    ax.set_xlabel("m"); ax.set_ylabel("m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: dict, masks: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for mode in results:
        ae = results[mode]["abs_err"][masks[mode]]
        ax.plot(np.arange(1, len(ae) + 1), ae, lw=1.4, label=mode.upper())
    ax.set_title(f"Cross-track error comparison — last {PLOT_LAST_CIRCLES} circle(s) ({WIND_LABEL})")
    ax.set_xlabel(f"sample (last {PLOT_LAST_CIRCLES} circle(s))"); ax.set_ylabel("abs cross-track error [m]")
    ax.grid(True, alpha=0.4); ax.margins(x=0)
    ax.legend(loc="upper right", ncol=len(results), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ====================== Report ========================================
def write_report(results: dict, masks: dict, mets: dict, path) -> str:
    R = float(path.R)
    md = []
    md.append("# SITL Lateral Controller Evaluation\n")
    md.append("**Operating point**\n")
    md.append("| Parameter | Value |")
    md.append("|---|---|")
    md.append(f"| Circle radius R | {R:.0f} m |")
    md.append(f"| Direction | {'CW' if d.LOITER_CW else 'CCW'} |")
    md.append(f"| Wind | {WIND_LABEL} |")
    md.append(f"| Circles flown / controller | {N_CIRCLES} |")
    md.append(f"| Plotted / scored | last {PLOT_LAST_CIRCLES} circle(s) |")
    md.append(f"| RTL settle between runs | {RTL_SETTLE_S:.0f} s |")
    md.append(f"| Autopilot | {d.AUTOPILOT_TYPE} (FBWB + RC1 override) |\n")

    # metrics table (best stable value per row in bold; divergent runs excluded)
    md.append(f"**Metrics over the last {PLOT_LAST_CIRCLES} circle(s)** (lower is better)\n")
    names = list(results.keys())
    keys = [("rms", "RMS abs e_t [m]"), ("max", "Peak abs e_t [m]"),
            ("mean", "Mean abs e_t [m]"), ("iae", "IAE [m·s]"),
            ("bank_rate_rms", "Bank-rate RMS [deg/s]"),
            ("circle_period_s", "Circle period [s]")]
    stable = [n for n in names if mets[n]["complete"] and mets[n]["rms"] < 0.5 * R]
    md.append("| Metric | " + " | ".join(x.upper() for x in names) + " |")
    md.append("|" + "---|" * (len(names) + 1))
    for key, label in keys:
        best = min((mets[n][key] for n in stable if mets[n][key] == mets[n][key]),
                   default=None) if key != "circle_period_s" else None
        cells = []
        for n in names:
            v = mets[n][key]
            s = "n/a" if v != v else f"{v:.3f}"
            if best is not None and v == v and n in stable and abs(v - best) < 1e-9:
                s = f"**{s}**"
            cells.append(s)
        md.append(f"| {label} | " + " | ".join(cells) + " |")
    md.append("")

    # incomplete / divergent flags
    incomplete = [n for n in names if not mets[n]["complete"]]
    if incomplete:
        md.append("> ⚠️ Did not complete " + f"{N_CIRCLES} circles within the time cap: "
                  + ", ".join(x.upper() for x in incomplete)
                  + " — diverged or failed to capture within the cap.\n")

    md.append("**Legend** — abs cross-track error = `|rho - R|`. IAE is the "
              "integral of abs error over time (overall tracking effort). Bank-rate RMS "
              "measures command smoothness. Each run starts from a common RTL base.\n")

    # per-controller plots
    for mode in results:
        md.append(f"## {mode.upper()}\n")
        m = mets[mode]
        md.append(f"RMS `{m['rms']:.3f}` m · peak `{m['max']:.3f}` m · "
                  f"IAE `{m['iae']:.3f}` m·s · "
                  f"bank-rate RMS `{m['bank_rate_rms']:.2f}` deg/s"
                  + ("" if m["complete"] else " · **did not complete 2 circles**") + "\n")
        md.append(f"Full log (driver CSV schema): [{mode}.csv]({mode}.csv)\n")
        md.append(f"![{mode} crosstrack]({mode}_crosstrack.png)\n")
        md.append(f"![{mode} trajectory]({mode}_trajectory.png)\n")

    md.append("## Comparison\n")
    md.append("![comparison](comparison_crosstrack.png)\n")

    report_path = os.path.join(RUN_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return report_path


# ====================== Main ==========================================
def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[EVAL] Output dir: {RUN_DIR}")
    print(f"[EVAL] Modes: {MODES}  | {N_CIRCLES} circles each, plotting last {PLOT_LAST_CIRCLES}")

    m = d.connect_mavlink(d.MAVLINK_URL)
    S = d.Shared()
    S.tx_enable = False            # don't grab roll until we explicitly engage

    # Start reader + controller first; the RC-override sender starts now too but
    # stays RELEASED (tx_enable=False) until each run engages FBWB.
    th_rx = threading.Thread(target=d.state_reader, args=(m, S), daemon=True)
    th_ct = threading.Thread(target=d.ctl_worker,   args=(m, S), daemon=True)
    th_tx = threading.Thread(target=d.att_sender,   args=(m, S, d.TX_HZ), daemon=True)
    th_rx.start(); th_ct.start(); th_tx.start()

    wait_airborne(S)

    path = d.CirclePath(d.LOITER_C, d.LOITER_R, cw=d.LOITER_CW)

    results = {}
    try:
        for mode in MODES:
            print(f"[EVAL] === {mode.upper()} ===")
            rtl_settle(m, S)        # common base state before every run
            engage_fbwb(m, S)       # take roll back in FBWB
            results[mode] = fly_and_record(S, mode, path)
    finally:
        # Hand the vehicle back to the autopilot no matter what.
        S.tx_enable = False
        try:
            d.set_mode(m, "RTL")
            print("[EVAL] RTL requested.")
        except Exception:
            pass
        S.stop.set()
        time.sleep(0.3)

    # ---- post-process: plots + report ----
    print("[EVAL] Rendering plots + report ...")
    masks, mets = {}, {}
    for mode, rec in results.items():
        mask = last_circle_mask(rec)
        masks[mode] = mask
        complete = bool(rec["phi"][-1] >= N_CIRCLES * 2 * math.pi - 1e-6) if len(rec["phi"]) else False
        mets[mode] = metrics(rec, mask, complete)
        plot_crosstrack(mode, rec, mask, os.path.join(RUN_DIR, f"{mode}_crosstrack.png"))
        plot_trajectory(mode, rec, mask, path, os.path.join(RUN_DIR, f"{mode}_trajectory.png"))

    plot_compare(results, masks, os.path.join(RUN_DIR, "comparison_crosstrack.png"))
    report_path = write_report(results, masks, mets, path)

    print(f"[EVAL] Done. Report: {report_path}")
    for mode in results:
        mm = mets[mode]
        print(f"   {mode:>5}: RMS={mm['rms']:.3f} m  peak={mm['max']:.3f} m  "
              f"IAE={mm['iae']:.3f} m.s  "
              f"bankrate={mm['bank_rate_rms']:.2f} deg/s  "
              f"{'OK' if mm['complete'] else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
