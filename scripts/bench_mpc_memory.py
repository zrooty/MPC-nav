#!/usr/bin/env python3
"""MPC memory + timing benchmark.

Replicates the exact ctl_worker parameters from hardware/mavlink_driver.py and
drives 1500 mpc.step() calls on a steady-state orbit, matching SITL conditions
(PLOT_LAST_SAMPLES = 1500 in sitl_auto_eval.py).

Run from repo root:
    python scripts/bench_mpc_memory.py
"""
import os, sys, math, tracemalloc, time
import numpy as np
import psutil

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mpc_nav.geometry import CirclePath
from mpc_nav.ltv_mpc import LTVMPC_OSQP, MPCWeights

# ── same as mavlink_driver.py ──────────────────────────────────────────────
N_STEPS   = 1500
Ts        = 0.10
LOITER_C  = (0.0, 0.0)
LOITER_R  = 180.0
LOITER_CW = False
Va        = 22.0

W = MPCWeights(w_et=2.0, w_echi=1.0, w_mu=10.0, w_u=3.0, w_et_T=2.0, w_echi_T=1.0)
path = CirclePath(LOITER_C, LOITER_R, cw=LOITER_CW)
mpc = LTVMPC_OSQP(
    Ts=Ts, N=40, Va_init=Va, tau_mu=0.45,
    bank_limit_deg=50.0, slew_limit_deg_s=9.0,
    weights=W, path=path, w_du=200.0, w_mu_Term=1.0,
    use_groundspeed_mu_ss=False,
    Va_min=6.0, Va_max=30.0, Va_lp_tau=1.0, Va_nominal=Va,
    use_w_du_scaling=True,
)
# ──────────────────────────────────────────────────────────────────────────

omega    = Va / LOITER_R                                  # orbital rate [rad/s]
sign_dir = -1 if LOITER_CW else +1
mu_ss    = sign_dir * math.atan(Va**2 / (9.81 * LOITER_R))  # steady bank

proc = psutil.Process(os.getpid())

# — baseline RSS before any calls (after init/JIT) —
rss_before = proc.memory_info().rss
tracemalloc.start()
snap_pre = tracemalloc.take_snapshot()

u_prev = mu_ss
wind   = (0.0, 0.0)       # calm; set non-zero to test wind feed
theta  = 0.0
call_ms = np.empty(N_STEPS)

for i in range(N_STEPS):
    N_pos = LOITER_R * math.cos(theta)
    E_pos = LOITER_R * math.sin(theta)
    chi   = theta + sign_dir * math.pi / 2   # tangent course CCW/CW
    x     = np.array([N_pos, E_pos, chi, u_prev], float)

    t0      = time.perf_counter()
    u_cmd, _ = mpc.step(x, wind, u_prev, V_meas=Va)
    call_ms[i] = (time.perf_counter() - t0) * 1e3

    u_prev  = u_cmd
    theta  += sign_dir * omega * Ts

snap_post = tracemalloc.take_snapshot()
current_b, peak_b = tracemalloc.get_traced_memory()
tracemalloc.stop()

rss_after = proc.memory_info().rss
top = snap_post.compare_to(snap_pre, "lineno")

print()
print("=" * 56)
print(f"  MPC Memory Benchmark  — {N_STEPS} calls, N=40, 4-state, R={LOITER_R:.0f}m")
print("=" * 56)
print(f"  tracemalloc peak     : {peak_b/1024:.1f} KB")
print(f"  tracemalloc net      : {current_b/1024:.1f} KB")
print(f"  RSS delta (calls)    : {(rss_after-rss_before)/1024:.1f} KB")
print(f"  RSS total after init : {rss_after/1024/1024:.1f} MB")
print()
print(f"  Solve time per call  (Ts budget = {Ts*1000:.0f} ms):")
print(f"    mean   : {call_ms.mean():.2f} ms   headroom {Ts*1000-call_ms.mean():.1f} ms")
print(f"    median : {np.median(call_ms):.2f} ms")
print(f"    p95    : {np.percentile(call_ms, 95):.2f} ms")
print(f"    max    : {call_ms.max():.2f} ms")
print()
print("  Top tracemalloc delta (by file):")
for s in top[:6]:
    if s.size_diff > 0:
        print(f"    +{s.size_diff/1024:6.1f} KB  {s.traceback[0]}")
print("=" * 56)
