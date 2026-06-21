# PI Loiter Baseline — Stability Analysis

This note documents the **PI loiter controller** added as a third baseline
(`mpc_nav/pi_loiter.py`) alongside L1 and LTV-MPC, and explains **why a pure
proportional–integral law on the radial error is unstable** for this loiter
task. It is retained as a *documented negative baseline*: evidence that a naive
PI is insufficient and that the L1/MPC structure (or at least cross-track-rate
damping) is required.

Operating point: circle radius `R = 90 m`, cruise airspeed `Va_ref ≈ 21 m/s`,
outer control step `Ts = 0.10 s` with the LQR roll inner loop sub-stepped at
`Ts/5 = 0.02 s` (see §3 and `config.inner_loop_substeps`).

## 1. Control law

The controller outputs a bank-angle command directly:

```
e_r     = ρ − R                          (signed radial / cross-track error, >0 outside)
φ_pi    = Kp·e_r + Ki·∫ e_r dt
φ_ff    = atan( Vg² / (g·R) )            (centripetal steady-state bank, groundspeed-based)
φ_cmd   = sign_dir · ( φ_ff + φ_pi )     (sign_dir = −1 for CW, +1 for CCW)
```

with conditional anti-windup (integrator frozen while the bank command is
saturated and the error pushes it further into saturation). Default gains:
`Kp = 0.020 rad/m`, `Ki = 0.003 rad/(m·s)`, centripetal FF on.

The feed-forward is algebraically exact: since `tan(φ_ff) = Vg²/(g·R)`, a
perfectly tracked `φ_ff` commands a turn radius of exactly `R` at **any**
airspeed. So in principle FF-only should hold the circle. In practice the
feedback law destabilises it — see below.

## 2. Observed behaviour (150 s, default initial condition)

| Controller | Cross-track RMS | Last-10 s RMS | Peak |
|---|---|---|---|
| L1         | ~7 m   | ~0.5 m | 64 m (capture) |
| LTV-MPC    | ~23 m  | ~7 m   | 96 m (capture) |
| **PI (pure)** | **~310 m** | **~610 m** | **~640 m → diverges** |

The PI trajectory spirals inward and never holds the circle, for every gain in
a `Kp∈[0.002,0.02] × Ki∈[0,0.002]` sweep. The other two controllers hold the
90 m circle.

## 3. Why the pure PI fails — two fundamental causes

**(a) Pure PI on a double-integrator plant is underdamped.**
The map *bank → radial position* is effectively a double integrator
(`χ̇ ∝ tanφ`, then position integrates velocity). A PI controller adds further
phase lag and provides **no damping** of the radial *rate* `ṙ`. The closed
loop is therefore oscillatory/unstable for any non-trivial gain. L1 supplies
this damping explicitly through its `Kv·ẋ_track` term; the MPC supplies it
through the predicted-trajectory cost. The pure PI has no equivalent.

**(b) A position-only bank law is heading-blind.**
`φ_cmd` depends only on position (`e_r`), not on heading `χ`. The bank that
"turns toward the circle" only reduces `e_r` if the aircraft is already moving
roughly tangentially. From an off-circle start heading at the centre, the
commanded bank can curve the velocity *away* from the path → positive feedback
→ the trajectory flies off. Capture requires a guidance law that accounts for
heading (as L1 does).

These two causes are intrinsic to the φ=PI(e_r) structure and **remain even
after the inner-loop fix below**, which is why the PI still diverges (~310 m)
in §2.

## 4. A separate latent bug found & fixed: under-sampled LQR inner loop

While investigating the PI, a pre-existing simulation bug surfaced that
affected **all** controllers. The lateral-LQR closed-loop roll dynamics have a
fast pole at ≈ −49.5 rad/s. The plant was integrated at `Ts = 0.10 s` with the
aileron recomputed only once per step, so `|s·Δt| ≈ 4.95`, well beyond the RK4
stability limit (~2.8). The roll loop therefore tracked the commanded bank
with a **bias**:

| Commanded bank | Realised bank (1 step, Ts=0.1) | Realised (5 sub-steps, dt=0.02) |
|---|---|---|
| 7.25° (cruise at 10.6 m/s)  | ~6.9° (small bias) | 7.25° |
| 26.5° (cruise at 21 m/s)    | **25.0° (−1.5° bias)** + ripple | 26.5° (exact) |

A shallower-than-commanded bank ⇒ wider turn. At the low airspeed the bias was
negligible, which is why the controllers *appeared* fine; at the corrected
21 m/s the bias is large enough that even the **MPC diverged** (cross-track RMS
273 m). 

**Fix:** integrate the plant in `config.inner_loop_substeps = 5` sub-steps per
`Ts`, recomputing the LQR aileron each sub-step (`_step_plant` in
`simulation.py`). The inner loop is then sampled at 0.02 s and tracks the
commanded bank exactly. After the fix the MPC recovers (RMS 273 m → ~23 m) and
L1 holds to ~0.5 m. The pure PI still diverges — confirming its failure is
structural (causes a/b), not an artifact of the inner-loop bias.

Related corrected setting: `config.k_thrust` was raised 9 → 60 N so the
aircraft can actually reach `Va_ref = 21 m/s` (the old value capped airspeed at
`√(9/0.08) ≈ 10.6 m/s`). Cruise throttle is now ≈ 0.59.

## 5. What makes the PI work (for future reference)

Adding **cross-track-rate damping** restores stability — a derivative term on
the radial velocity `ṙ = Â·v_g`, turning the law into a PID:

```
φ_cmd = sign_dir · ( φ_ff + Kp·e_r + Kd·ṙ + Ki·∫ e_r dt )
```

Verified at 21 m/s (with inner-loop sub-stepping), e.g. `Kp=0.02, Kd=0.10,
Ki=0.001`: cross-track RMS **~9 m**, last-10 s RMS **~6 m**, peak 63.6 m
(capture overshoot) — i.e. it now holds the circle, comparable to L1/MPC. The
`Kd·ṙ` term is the direct analogue of L1's `Kv` damping. A heading-cascade
guidance law (PI on `e_r` → commanded course offset → bank from heading error)
is an alternative that is naturally damped.

## 6. Reproducing

The three controllers run in parallel in `mpc_nav/simulation.py` (plants
`x1`=L1, `x2`=MPC, `x3`=PI). Run the package:

```
python -m mpc_nav.main
```

Artifacts (in the timestamped output folder) include the PI series (drawn in
red): `traj_compare_with_wind.png`, `et_compare.png`, `bank_compare.png`,
`et_series_vs_time.png`, `metrics_compare.csv`, `metrics_crosstrack.csv`,
`et_series.csv`.
