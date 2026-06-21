"""
Roll stabiliser diagnostic: step 0 -> 25 deg.
Plot bank angle and roll rate only.

Usage:
    python scripts/lqr_roll_step.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpc_nav.lateral_lqr import LateralLQR
from mpc_nav import config


def simulate(lqr, mu_ref_deg, mu0_deg=0.0, duration=3.0, dt=0.005):
    mu_ref = np.radians(mu_ref_deg)
    mu = np.radians(mu0_deg)
    p = 0.0
    a_p, b_p = config.a_p, config.b_p

    n = int(duration / dt)
    t   = np.zeros(n + 1)
    mus = np.zeros(n + 1)
    ps  = np.zeros(n + 1)
    mus[0] = mu

    for i in range(n):
        ail = lqr.step(mu_ref, mu, p)
        def f(phi, pp):
            return pp, a_p * pp + b_p * ail
        k1 = f(mu, p)
        k2 = f(mu + 0.5*dt*k1[0], p + 0.5*dt*k1[1])
        k3 = f(mu + 0.5*dt*k2[0], p + 0.5*dt*k2[1])
        k4 = f(mu +     dt*k3[0], p +     dt*k3[1])
        mu += (dt/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        p  += (dt/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        t[i+1]   = (i+1)*dt
        mus[i+1] = mu
        ps[i+1]  = p

    return t, np.degrees(mus), np.degrees(ps)


def metrics(t, phi, target):
    delta = target - phi[0]
    lo, hi = phi[0] + 0.1*delta, phi[0] + 0.9*delta
    band = abs(delta) * 0.02

    idx_lo = next((i for i, v in enumerate(phi) if v >= lo), None)
    idx_hi = next((i for i, v in enumerate(phi) if v >= hi), None)
    rise = (t[idx_hi] - t[idx_lo]) if (idx_lo is not None and idx_hi is not None) else float("nan")

    settle = float("nan")
    for i in range(len(phi)-1, -1, -1):
        if abs(phi[i] - target) > band:
            settle = t[min(i+1, len(t)-1)]
            break

    overshoot = max(0.0, (max(phi) - target) / abs(delta) * 100)
    return rise, settle, overshoot


if __name__ == "__main__":
    lqr = LateralLQR()
    print(f"  K full (3x6):\n{lqr.K}")
    print(f"  K roll channel [phi, p, theta, q, psi, r]: {lqr.K[0]}")

    t, phi, p = simulate(lqr, mu_ref_deg=25.0)

    rise, settle, overshoot = metrics(t, phi, target=25.0)
    print(f"  Rise time   (10–90%) : {rise:.3f} s")
    print(f"  Settle time (±2%)    : {settle:.3f} s")
    print(f"  Peak overshoot       : {overshoot:.1f} %")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle("LQR Roll Stabiliser — Step 0° → 25°")

    ax1.axhline(25, color="k", lw=0.8, ls="--", label="ref 25°")
    ax1.plot(t, phi, label="bank angle φ")
    ax1.set_ylabel("Bank angle [°]")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, p, color="tab:orange", label="roll rate p")
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_ylabel("Roll rate p [°/s]")
    ax2.set_xlabel("Time [s]")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out = pathlib.Path(__file__).parent / "lqr_roll_step.png"
    plt.savefig(out, dpi=120)
    print(f"Plot saved -> {out}")
    plt.show()
