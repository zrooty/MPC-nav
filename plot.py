#!/usr/bin/env python3
"""
plot_mavlog_loiter.py

Membaca CSV log MAVLink dan membuat:
  1) plot attitude roll vs command roll (nav_roll/cmd_roll)
  2) plot cross-track error (rerr) + analisis (RMS, IAE, MAX, settling time)
  3) plot lintasan loiter (N vs E) dan fit lingkaran referensi untuk menghitung rerr jika perlu

Usage:
  python plot_mavlog_loiter.py path/to/log.csv --outdir out_plots --tol 5.0

Dependencies:
  pandas, numpy, matplotlib

Author: hasil kerja paksa dari asisten yang enggan tapi kompeten
"""
import argparse
import os
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_numeric_series(s: pd.Series) -> pd.Series:
    """Cleans a pandas series and converts to numeric where possible."""
    if s is None:
        return s
    # remove common noise characters, handle locale comma decimal
    cleaned = s.astype(str).str.strip().str.replace(r'[^\d\-\+eE\.,]', '', regex=True)
    # try replace comma decimal if no dot present often
    # if majority contain comma and not dot, replace comma->dot
    has_comma = cleaned.str.contains(',', regex=False).sum()
    has_dot = cleaned.str.contains('\.', regex=True).sum()
    if has_comma > has_dot:
        cleaned = cleaned.str.replace(',', '.')
    return pd.to_numeric(cleaned, errors='coerce')


def fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit circle using linear (Kasa) least squares method.
    Returns (xc, yc, R).
    """
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, c = sol
    R = math.sqrt(max(0.0, xc * xc + yc * yc + c))
    return float(xc), float(yc), float(R)


def compute_crosstrack_from_circle(N: np.ndarray, E: np.ndarray, xc: float, yc: float, R: float) -> np.ndarray:
    rho = np.sqrt((N - xc) ** 2 + (E - yc) ** 2)
    return rho - R


def compute_metrics(t: np.ndarray, err: np.ndarray, tol: float = 5.0, settling_window_s: float = 2.0):
    abs_err = np.abs(err)
    rms = math.sqrt(np.nanmean(err ** 2)) if np.isfinite(err).any() else float('nan')
    dt = (t[1] - t[0]) if len(t) >= 2 else 0.1
    iae = float(np.nansum(abs_err) * dt)
    maxe = float(np.nanmax(abs_err)) if np.isfinite(abs_err).any() else float('nan')
    # settling time: first time |e| <= tol for continuous settling_window_s seconds
    win_n = max(1, int(round(settling_window_s / dt)))
    settling_time = float('nan')
    if len(abs_err) >= win_n:
        for i in range(0, len(abs_err) - win_n + 1):
            block = abs_err[i:i + win_n]
            if np.all(np.isfinite(block)) and np.all(block <= tol):
                settling_time = float(t[i])
                break
    return dict(RMS=rms, IAE=iae, MAX=maxe, settling_time=settling_time)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_rolls(t: np.ndarray, roll: Optional[np.ndarray], cmd_roll: Optional[np.ndarray], extras: dict, outpath: str):
    plt.figure(figsize=(10, 4))
    plotted = False
    if roll is not None:
        plt.plot(t, roll, label='attitude roll (deg)')
        plotted = True
    if cmd_roll is not None:
        plt.plot(t, cmd_roll, label='cmd_roll (deg)')
        plotted = True
    if 'uL1' in extras and extras['uL1'] is not None:
        plt.plot(t, extras['uL1'], '--', alpha=0.7, label='uL1 (deg)')
    if 'uMPC' in extras and extras['uMPC'] is not None:
        plt.plot(t, extras['uMPC'], ':', alpha=0.7, label='uMPC (deg)')
    if not plotted:
        print("No roll/cmd_roll data to plot.")
        return
    plt.xlabel('Time [s]')
    plt.ylabel('Roll [deg]')
    plt.title('Attitude roll vs command roll')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"Saved roll plot to {outpath}")


def plot_crosstrack(t: np.ndarray, rerr: np.ndarray, outpath: str, tol: float = 5.0):
    plt.figure(figsize=(10, 4))
    plt.plot(t, rerr, label='cross-track error (m)', color='b')
    # plt.plot(t, np.abs(rerr), label='|e| (m)', alpha=0.6)
    # plt.axhline(tol, color='r', ls='--', label=f'tol {tol} m')
    plt.xlabel('Time [s]')
    plt.ylabel('Cross-track error [m]')
    plt.title('Cross-track error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"Saved crosstrack plot to {outpath}")


def plot_trajectory(N: np.ndarray, E: np.ndarray, t: np.ndarray, xc: float, yc: float, R: float, outpath: str):
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(N, E, c=t, s=6, cmap='viridis', label='position (N,E)')
    al = np.linspace(0, 2 * np.pi, 400)
    plt.plot(xc + R * np.cos(al), yc + R * np.sin(al), 'r--', lw=1.5, label=f'fit circle R={R:.2f} m')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('North [m]')
    plt.ylabel('East [m]')
    plt.title('Loiter trajectory (N vs E)')
    plt.colorbar(sc, label='time [s]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"Saved trajectory plot to {outpath}")


def main():
    p = argparse.ArgumentParser(description="Plot MAVLink loiter logs: roll, crosstrack, trajectory")
    p.add_argument('csv', help='Path to MAVLink CSV log')
    p.add_argument('--outdir', '-o', default='plots_mavlog', help='Output directory for plots and csv')
    p.add_argument('--timecol', default='t', help='Time column name in CSV (default: t)')
    p.add_argument('--roll', default='roll_deg', help='Attitude roll column name (deg)')
    p.add_argument('--cmdroll', default='cmd_roll_deg', help='Command roll column name (deg)')
    p.add_argument('--ncol', default='N', help='North position column name')
    p.add_argument('--ecol', default='E', help='East position column name')
    p.add_argument('--rerr', default='rerr', help='Cross-track error column name (optional)')
    p.add_argument('--rho', default='rho', help='Range/radius column name (optional)')
    p.add_argument('--tol', type=float, default=5.0, help='Tolerance for settling (m)')
    p.add_argument('--settle-window', type=float, default=2.0, help='Settling window (s)')
    args = p.parse_args()

    csv_path = args.csv
    outdir = args.outdir
    ensure_dir(outdir)

    print("Loading CSV:", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    print("CSV loaded. Columns:", list(df.columns))

    # Convert columns
    col_time = args.timecol if args.timecol in df.columns else None
    t_series = safe_numeric_series(df[col_time]) if col_time is not None else None

    if t_series is None or t_series.notnull().sum() == 0:
        print("Time column not found or invalid, using index-based time with dt=0.1s")
        t = np.arange(len(df)) * 0.1
    else:
        t = t_series.to_numpy(dtype=float)
        # If time looks like unix timestamp or monotonic but large, try shift to start at zero
        if np.nanmin(t) > 1e5:
            t = t - t[0]

    # Roll columns
    roll = safe_numeric_series(df[args.roll]) if args.roll in df.columns else None
    cmd_roll = safe_numeric_series(df[args.cmdroll]) if args.cmdroll in df.columns else None

    # Extras
    extras = {}
    for k in ['uL1', 'uMPC']:
        if k in df.columns:
            extras[k] = safe_numeric_series(df[k]).to_numpy(dtype=float)
        else:
            extras[k] = None

    # N,E
    N_ser = safe_numeric_series(df[args.ncol]) if args.ncol in df.columns else None
    E_ser = safe_numeric_series(df[args.ecol]) if args.ecol in df.columns else None

    N = N_ser.to_numpy(dtype=float) if (N_ser is not None and N_ser.notnull().sum() > 0) else None
    E = E_ser.to_numpy(dtype=float) if (E_ser is not None and E_ser.notnull().sum() > 0) else None

    # rerr if present
    rerr = None
    if args.rerr in df.columns:
        rerr_ser = safe_numeric_series(df[args.rerr])
        if rerr_ser is not None and rerr_ser.notnull().sum() > 0:
            rerr = rerr_ser.to_numpy(dtype=float)
            print(f"Using provided cross-track error column '{args.rerr}'")

    # If no rerr but rho present, try use rho (distance to something) - but prefer circle fit anyway
    if rerr is None and N is not None and E is not None and len(N) > 10:
        print("No rerr provided, fitting circle to (N,E) to estimate cross-track error.")
        xc, yc, R = fit_circle_kasa(N, E)
        rerr = compute_crosstrack_from_circle(N, E, xc, yc, R)
        used_circle = True
        print(f"Fitted circle center (N,E)=({xc:.3f}, {yc:.3f}), R={R:.3f} m")
    else:
        used_circle = False
        xc = yc = R = float('nan')

    # Plot roll vs cmd_roll
    roll_arr = roll.to_numpy(dtype=float) if (roll is not None and roll.notnull().sum() > 0) else None
    cmd_roll_arr = cmd_roll.to_numpy(dtype=float) if (cmd_roll is not None and cmd_roll.notnull().sum() > 0) else None
    plot_rolls(t, roll_arr, cmd_roll_arr, extras, os.path.join(outdir, 'roll_vs_cmd.png'))

    # Plot crosstrack and analyze
    if rerr is not None:
        # ensure length matches time vector if possible
        if len(rerr) != len(t):
            # if N/E used to compute rerr, they likely align with df indices; try to match length
            minlen = min(len(rerr), len(t))
            rerr = rerr[:minlen]
            t_used = t[:minlen]
        else:
            t_used = t
        metrics = compute_metrics(t_used, rerr, tol=args.tol, settling_window_s=args.settle_window)
        plot_crosstrack(t_used, rerr, os.path.join(outdir, 'crosstrack_error.png'), tol=args.tol)
        print("Crosstrack metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("No cross-track error data and could not compute from positions.")

    # Plot trajectory and save CSV with fitted values if available
    if N is not None and E is not None and np.isfinite(xc) and not math.isnan(R):
        plot_trajectory(N, E, t, xc, yc, R, os.path.join(outdir, 'trajectory_loiter_fit.png'))
        # save csv with computed rerr if applicable
        out_csv = os.path.join(outdir, 'loiter_positions_with_rerr.csv')
        rho_fit = np.sqrt((N - xc) ** 2 + (E - yc) ** 2)
        rerr_fit = rho_fit - R
        df_out = pd.DataFrame(dict(t=t[:len(N)], N=N, E=E, rho_fit=rho_fit, rerr_fit=rerr_fit))
        df_out.to_csv(out_csv, index=False)
        print("Wrote loiter positions + rerr to:", out_csv)
    else:
        print("Insufficient N/E data to plot trajectory or compute fit.")

    print("All done. Output directory:", outdir)


if __name__ == '__main__':
    main()
